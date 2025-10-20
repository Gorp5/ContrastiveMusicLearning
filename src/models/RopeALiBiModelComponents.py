import torch.nn as nn
import torch
import math

from flash_attn import flash_attn_func
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached

def rotate_half(x: torch.Tensor):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)

def get_alibi_slopes(n_heads: int):
    # Press et al. method for decreasing slopes per head (common ALiBi initialization)
    # returns tensor shape (n_heads,)
    def get_slopes_power_of_two(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]
    if math.log2(n_heads).is_integer():
        slopes = torch.tensor(get_slopes_power_of_two(n_heads), dtype=torch.float32)
    else:
        # fallback for non-power-of-two heads
        m = 2 ** math.floor(math.log2(n_heads))
        slopes = torch.tensor(get_slopes_power_of_two(m), dtype=torch.float32)
        extra = torch.tensor([slopes[-1] * (0.5 ** (i + 1)) for i in range(n_heads - m)], dtype=torch.float32)
        slopes = torch.cat([slopes, extra], dim=0)
    return slopes  # shape (n_heads,)

# ---------------------- Utility ----------------------
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


# ---------------------- 2D ALiBi Bias ----------------------
class Alibi2DBias(nn.Module):
    def __init__(self, num_heads, r_left=1.0, r_right=1.0):
        super().__init__()
        slopes = get_alibi_slopes(num_heads).to(device='cuda')
        self.register_buffer("slopes", slopes)
        self.r_left = r_left
        self.r_right = r_right

    def forward(self, coords):
        """
        coords: (B, N, 2) tensor with (x, y) integer positions of kept patches
        """
        B, N, _ = coords.shape
        x, y = coords[..., 0].float(), coords[..., 1].float()

        dx = (x[:, :, None] - x[:, None, :]).abs()
        dy = (y[:, :, None] - y[:, None, :]).abs()
        dist = dx + dy  # (B, N, N)

        # raster order based on (y * width + x)
        flat = y * x.max().add(1) + x
        le_mask = (flat[:, None, :] <= flat[:, :, None]).float()

        slopes = self.slopes.to(coords.device)
        left = slopes.view(1, -1, 1, 1) * self.r_left
        right = slopes.view(1, -1, 1, 1) * self.r_right

        dist = dist.unsqueeze(1).expand(-1, slopes.numel(), -1, -1)
        le_mask = le_mask.unsqueeze(1)

        bias = dist * (le_mask * left + (1 - le_mask) * right)
        return bias  # (B, H, N, N)


# ---------------------- Multihead Attention ----------------------
class CustomMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, alibi_bias=None):
        B, N, D = x.shape
        H, Hd = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, N, H, Hd).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, Hd).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, Hd).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if alibi_bias is not None:
            attn = attn + alibi_bias

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


# ---------------------- Transformer Encoder Layer ----------------------
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=16, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.self_attn = CustomMultiheadAttention(d_model, num_heads, dropout)
        self.alibi_2d = Alibi2DBias(num_heads)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, mask=None, coords=None):
        alibi_bias = self.alibi_2d(coords) if coords is not None else None

        src2 = self.self_attn(src, mask=mask, alibi_bias=alibi_bias)
        src = self.norm1(src + self.dropout1(src2))

        src2 = self.linear2(self.activation(self.linear1(src)))
        src = self.norm2(src + self.dropout2(src2))
        return src


# ---------------------- Full Encoder ----------------------
class CustomTransformerEncoder(nn.Module):
    def __init__(self, num_layers=8, d_model=256, num_heads=16, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None, coords=None):
        for layer in self.layers:
            src = layer(src, mask=mask, coords=coords)
        return self.norm(src)