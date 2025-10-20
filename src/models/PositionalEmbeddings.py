import math

import torch
import torch.nn.functional as F
from torch import nn

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def get_alibi_slopes(n_heads: int):
    def get_slopes_power_of_two(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]
    if math.log2(n_heads).is_integer():
        slopes = get_slopes_power_of_two(n_heads)
    else:
        m = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_two(m)
        extra = [slopes[-1] * (0.5 ** (i + 1)) for i in range(n_heads - m)]
        slopes.extend(extra)
    return torch.tensor(slopes, dtype=torch.float32)


class Alibi2DBias(nn.Module):
    def __init__(self, num_heads, r_left=1.0, r_right=1.0):
        super().__init__()
        slopes = -get_alibi_slopes(num_heads).to(device='cuda')
        self.register_buffer("slopes", slopes)
        self.r_left = r_left
        self.r_right = r_right

    def forward(self, coords):
        B, N, _ = coords.shape
        x, y = coords[..., 0].float(), coords[..., 1].float()

        dx = (x[:, :, None] - x[:, None, :]).abs()
        dy = (y[:, :, None] - y[:, None, :]).abs()
        dist = dx + dy

        # raster order
        flat = y * x.max().add(1) + x
        le_mask = (flat[:, None, :] <= flat[:, :, None]).float()

        slopes = self.slopes.to(coords.device)
        left = slopes.view(1, -1, 1, 1) * self.r_left
        right = slopes.view(1, -1, 1, 1) * self.r_right

        dist = dist.unsqueeze(1).expand(-1, slopes.numel(), -1, -1)
        le_mask = le_mask.unsqueeze(1)

        bias = dist * (le_mask * left + (1 - le_mask) * right)

        return bias

class Cable2DPatchoutCLS(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # Linear layers to generate f (bias contributions) and g (weights)
        self.f_proj = nn.Linear(d_model, num_heads)
        self.g_proj = nn.Linear(d_model, num_heads)

    def forward(self, x, coords, has_cls=True):
        B, T, D = x.shape
        H = self.num_heads
        W = coords[..., 0].max().item() + 1  # max x + 1 for raster flattening
        flat_idx = coords[..., 1] * W + coords[..., 0]

        le_mask = (flat_idx[:, :, None] >= flat_idx[:, None, :]).float()

        f = -F.relu(self.f_proj(x))  # (B, T, H)
        g = F.softplus(self.g_proj(x))  # (B, T, H)

        # Compute pairwise relative biases
        bias_delta = torch.einsum('b i j, b j h -> b i j h', le_mask, f) - torch.einsum('b j i, bj  h -> b i j h', le_mask, f)
        bias = bias_delta * g[:, None, :, :]  # (B, T, T, H)
        bias = bias.permute(0, 3, 1, 2).contiguous()  # (B, H, T, T)

        if has_cls:
            B, H, _, _ = bias.shape
            bias_with_cls = torch.zeros(B, H, T+1, T+1, device=bias.device, dtype=bias.dtype)
            bias_with_cls[:, :, 1:, 1:] = bias
            # CLS token bias stays zero
            bias = bias_with_cls

        return bias