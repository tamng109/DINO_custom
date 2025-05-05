import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    """
    Multi-Scale Deformable Attention Module with hard-pruning support.
    After accumulating head importance, call `hard_prune_heads(ratio)` to remove
    the least important heads and rebuild internal projections.
    """
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model must be divisible by n_heads, but got {d_model} and {n_heads}')
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "Setting head dimension to a power of 2 improves CUDA performance.")

        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        # projection layers
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

        # buffers for pruning
        self.register_buffer('head_mask', torch.ones(n_heads, dtype=torch.bool))
        self.register_buffer('head_importance', torch.zeros(n_heads))

        # flag to track importance accumulation
        self.track_importance = True

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= (i + 1)
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight, 0.)
        constant_(self.attention_weights.bias, 0.)
        xavier_uniform_(self.value_proj.weight)
        constant_(self.value_proj.bias, 0.)
        xavier_uniform_(self.output_proj.weight)
        constant_(self.output_proj.bias, 0.)

    def prune_least_important_heads(self, ratio=0.25):
        """
        Soft mask: sets head_mask for least important heads to False.
        """
        num_to_prune = int(self.n_heads * ratio)
        if num_to_prune <= 0:
            return
        _, idx = torch.topk(self.head_importance, num_to_prune, largest=False)
        self.head_mask[idx] = False
        # zero out importance to avoid reselection
        self.head_importance[idx] = 0
        print(f"[MSDeformAttn] Soft-pruned heads: {idx.tolist()}")

    def hard_prune_heads(self, ratio=0.25):
        """
        Hard prune: remove parameters of least important heads and rebuild layers.
        """
        num_to_prune = int(self.n_heads * ratio)
        if num_to_prune <= 0:
            return
        # determine heads to keep
        importance = self.head_importance.clone()
        _, prune_idx = torch.topk(importance, num_to_prune, largest=False)
        keep_idx = [h for h in range(self.n_heads) if h not in prune_idx.tolist()]
        new_h = len(keep_idx)
        assert self.d_model % new_h == 0, f"d_model {self.d_model} must be divisible by new_n_heads {new_h}"
        new_hd = self.d_model // new_h
        # --- rebuild sampling_offsets ---
        per_off = self.n_levels * self.n_points * 2
        W = self.sampling_offsets.weight.view(self.n_heads, per_off, self.d_model)
        b = self.sampling_offsets.bias.view(self.n_heads, per_off)
        Wk = W[keep_idx].reshape(new_h * per_off, self.d_model)
        bk = b[keep_idx].reshape(new_h * per_off)
        self.sampling_offsets = nn.Linear(self.d_model, new_h * per_off, bias=True)
        self.sampling_offsets.weight.data.copy_(Wk)
        self.sampling_offsets.bias.data.copy_(bk)
        # --- rebuild attention_weights ---
        per_att = self.n_levels * self.n_points
        W = self.attention_weights.weight.view(self.n_heads, per_att, self.d_model)
        b = self.attention_weights.bias.view(self.n_heads, per_att)
        Wk = W[keep_idx].reshape(new_h * per_att, self.d_model)
        bk = b[keep_idx].reshape(new_h * per_att)
        self.attention_weights = nn.Linear(self.d_model, new_h * per_att, bias=True)
        self.attention_weights.weight.data.copy_(Wk)
        self.attention_weights.bias.data.copy_(bk)
        # --- update value & output projections ---
        self.value_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.output_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        xavier_uniform_(self.value_proj.weight)
        constant_(self.value_proj.bias, 0.)
        xavier_uniform_(self.output_proj.weight)
        constant_(self.output_proj.bias, 0.)
        # update attributes and buffers
        self.n_heads = new_h
        self.head_dim = new_hd
        self.head_mask = self.head_mask[keep_idx].clone()
        self.head_importance = self.head_importance[keep_idx].clone()
        # Reset importance tracking after hard prune
        self.track_importance = False
        print(f"[MSDeformAttn] Hard-pruned to {new_h} heads.")

    def get_active_heads(self):
        """
        Returns a list of active heads (those not pruned).
        """
        return torch.nonzero(self.head_mask).flatten().tolist()

    def forward(self, query, reference_points, input_flatten,
                input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:,0]*input_spatial_shapes[:,1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[...,None], 0)
        # reshape value
        value = value.view(N, Len_in, self.n_heads, self.head_dim)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # apply mask (soft)
        attention_weights = F.softmax(attention_weights, -1)
        # accumulate importance
        if self.track_importance:
            with torch.no_grad():
                imp = attention_weights.abs().sum(dim=[1,3,4])  # (N, n_heads)
                self.head_importance += imp.sum(dim=0)
        # compute sampling locations as before
        # 1) Nếu ref_pts là [N, Len_q, 2], expand lên [N, Len_q, n_levels, 2]
        if reference_points.dim() == 3 and reference_points.size(-1) == 2:
            reference_points = reference_points.unsqueeze(2).expand(-1, -1, self.n_levels, -1)
        # 2) Nếu ref_pts là [N, Len_q, n_levels, 2], giữ nguyên
        elif reference_points.dim() == 4 and reference_points.size(-1) == 2:
            pass
        # 3) Nếu ref_pts là [N, Len_q, n_levels, 4], giữ nguyên
        elif reference_points.dim() == 4 and reference_points.size(-1) == 4:
            pass
        else:
            raise ValueError(f"Unsupported reference_points shape {tuple(reference_points.shape)}")
        if reference_points.size(-1) == 2:
            # offset normalization (n_levels,2) → (1,1,1,n_levels,1,2)
            offset_norm = torch.stack([input_spatial_shapes[...,1], input_spatial_shapes[...,0]], -1)
            offset_norm = offset_norm[None,None,None,:,None,:]
            sampling_locs = reference_points[:, :, None, :, None, :] \
                            + sampling_offsets / offset_norm
        elif reference_points.size(-1) == 4:
            # split x,y and w,h
            xy = reference_points[..., :2]      # (N,L,n_levels,2)
            wh = reference_points[..., 2:]      # (N,L,n_levels,2)
            # (1,1,1,n_levels,2)
            sampling_locs = xy[:, :, None, :, None, :] \
                            + sampling_offsets / self.n_points * wh[:, :, None, :, None, :] * 0.5
        else:
            # không thể xảy ra do bước chuẩn hóa
            raise RuntimeError

        # deformable attention kernel
        if value.dtype == torch.float16:
            output = MSDeformAttnFunction.apply(value.to(torch.float32), input_spatial_shapes,
                input_level_start_index, sampling_locs.to(torch.float32), attention_weights, self.im2col_step)
            output = output.to(torch.float16)
        else:
            output = MSDeformAttnFunction.apply(value, input_spatial_shapes,
                input_level_start_index, sampling_locs, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output
