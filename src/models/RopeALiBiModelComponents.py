import torch.nn as nn
import torch
import math

from flash_attn import flash_attn_func

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
    """Rotates half the hidden dims of the input."""
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

class RoPEALiBiMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.embed_dim = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, alibi_slopes=None, pos_emb=None, mask=None):

        QB, QL, _ = query.shape

        # Linear projections
        q = self.q_proj(query).view(QB, QL, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(QB, QL, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(QB, QL, self.num_heads, self.head_dim)

        if pos_emb is not None:
            cos, sin = pos_emb()
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.half()
        k = k.half()
        v = v.half()

        output = flash_attn_func(
            q, k, v,
            dropout_p=0.0,
            causal=True,
            alibi_slopes=alibi_slopes
        )

        output = output.view(QB, QL, self.embed_dim)

        output = output.float()

        return output


class RoPEALiBiTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=16, dim_feedforward=256, dropout=0.1):
        super().__init__()

        self.self_attn = RoPEALiBiMultiheadAttention(d_model, num_heads, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, src, alibi_slopes, pos_emb, mask):
        # src: [Batch, Time, Dim]
        t = src.size(1)

        # Self-attention Block
        src2 = self.self_attn(src, src, src, alibi_slopes, pos_emb, mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed-forward Block
        src2 = self.linear1(src)
        src2 = self.activation(src2)
        src2 = self.linear2(src2)

        # Regularization
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class RoPEALiBiTransformerEncoder(nn.Module):
    def __init__(self, num_layers=8, d_model=256, num_heads=16, dim_feedforward=256, seq_len=256, dropout=0.1,
                 checkpointing=False, use_rope=False, use_alibi=False, device='cpu', custom_slopes=-1):
        super().__init__()

        self.layers = nn.ModuleList([
            RoPEALiBiTransformerEncoderLayer(d_model=d_model, num_heads=num_heads, dim_feedforward=dim_feedforward,
                                             dropout=dropout).to(device, dtype=torch.float32)
            for _ in range(num_layers)
        ])

        self.checkpointing = checkpointing
        self.custom_slopes = custom_slopes

        if use_alibi:
            if custom_slopes > 0:
                self.alibi_slopes = get_ranged_slopes(num_heads, num_layers, device, max=custom_slopes, dtype=torch.float32)
            else:
                self.alibi_slopes = get_alibi_slopes(num_heads).to(device, dtype=torch.float32)
        else:
            self.alibi_slopes = None

        if use_rope:
            self.rope_matrix = RotaryEmbedding(dim=d_model // num_heads,
                                              max_position_embeddings=seq_len,
                                              base=10000)
        else:
            self.rope_matrix = None

        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        alibi_slopes = self.alibi_slopes
        for index, layer in enumerate(self.layers):
            if self.custom_slopes > 0.0:
                alibi_slopes = self.alibi_slopes[index]
            src = layer(src, alibi_slopes, self.rope_matrix, mask)

        return self.norm(src)


class RoPEALiBiTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=16, dim_feedforward=256, dropout=0.1):
        super().__init__()

        self.self_attn = RoPEALiBiMultiheadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = RoPEALiBiMultiheadAttention(d_model, num_heads, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, tgt, memory, alibi_slopes, pos_emb, mask):
        t = tgt.size(1)
        l = memory.size(1)


        # Self Attention Block
        tgt2 = self.self_attn(tgt, tgt, tgt, alibi_slopes=alibi_slopes, pos_emb=pos_emb, mask=mask)
        tgt2 = self.dropout1(tgt2)
        tgt = tgt + self.norm1(tgt2)

        # Cross-attention Block
        tgt2 = self.cross_attn(tgt, memory, memory, alibi_slopes=alibi_slopes, pos_emb=pos_emb, mask=mask)
        tgt2 = self.dropout2(tgt2)
        tgt = tgt + self.norm2(tgt2)

        # Feed-forward Block
        tgt2 = self.linear1(tgt)
        tgt2 = self.activation(tgt2)
        tgt2 = self.linear2(tgt2)

        # Regularization
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class RoPEALiBiTransformerDecoder(nn.Module):
    def __init__(self, num_layers=8, d_model=256, num_heads=16, dim_feedforward=256, seq_len=256, dropout=0.1, checkpointing=False, use_rope=True, use_alibi=True, device='cpu', custom_slopes=-1):
        super().__init__()

        self.layers = nn.ModuleList([
            RoPEALiBiTransformerDecoderLayer(d_model=d_model, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.checkpointing = checkpointing
        self.custom_slopes = custom_slopes

        if use_alibi:
            if custom_slopes > 0:
                self.alibi_slopes = get_ranged_slopes(num_heads, num_layers, device, max=custom_slopes, dtype=torch.float32)
            else:
                self.alibi_slopes = get_alibi_slopes(num_heads).to(device, dtype=torch.float32)
        else:
            self.alibi_slopes = None

        if use_rope:
            self.rope_matrix = RotaryEmbedding(dim=d_model // num_heads,
                                              max_position_embeddings=seq_len,
                                              base=10000).to(device, dtype=torch.float32)
        else:
            self.rope_matrix = None

        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, mask=None):
        alibi_slopes = self.alibi_slopes
        for index, layer in enumerate(self.layers):
            if self.custom_slopes > 0:
                alibi_slopes = self.alibi_slopes[index]
            tgt = layer(tgt, memory, alibi_slopes, self.rope_matrix,  mask)

        return self.norm(tgt)

    def set_checkpointing(self, checkpointing):
        self.checkpointing = checkpointing

def get_RoPE_matrix(seq_len, head_dim, device='cpu'):
    # Create position indices (seq_len,)
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)

    # Create dimension indices (head_dim/2,)
    dim_indices = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)

    # Compute scaling (Standard formula uses 10000^(2i/head_dim))
    inv_freq = 1.0 / (10000 ** (dim_indices / head_dim))

    # Compute sinusoid frequencies for each position.
    sinusoid_inp = torch.einsum("i,j->ij", pos, inv_freq)
    sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()

    # Expand sin and cos to [1,1,seq_len, head_dim/2] for broadcasting.
    sin = sin.unsqueeze(0).unsqueeze(0)
    cos = cos.unsqueeze(0).unsqueeze(0)

    pos_emb = torch.cat([cos, sin], dim=-1)
    return pos_emb

def apply_RoPE(x, pos_emb):
    """
    x: [B, H, L, D]
    pos_emb: [1, 1, L, D]
    """

    # Split even and odd channels
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    cos = pos_emb[..., ::2]
    sin = pos_emb[..., 1::2]

    return torch.cat([
        x_even * cos - x_odd * sin,
        x_even * sin + x_odd * cos
    ], dim=-1)


def get_alibi_slopes(nheads):
    def get_slopes_power_of_2(nheads):
        start = 2 ** (-(2 ** -(math.log2(nheads) - 3)))
        ratio = start
        return torch.tensor([start * ratio**i for i in range(nheads)])

    if math.log2(nheads).is_integer():
        return get_slopes_power_of_2(nheads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(nheads))
        return torch.cat(
            (get_slopes_power_of_2(closest_power_of_2),
            get_alibi_slopes(2 * closest_power_of_2)[0::2][: nheads - closest_power_of_2])
        )

def get_ranged_slopes(nheads, nlevels, device, max=4, dtype=torch.float32):
    base_slopes = get_alibi_slopes(nheads)  # [nheads]

    # Construct hierarchical slopes: each level can be a scaled copy of base slopes
    slopes = []
    for lvl in range(nlevels):
        scale = max * (2 ** -lvl)
        slopes.append((base_slopes * scale).to(device, dtype=dtype))

    return slopes  # [nlevels, nheads]

def build_bidirectional_symmetrical_alibi_bias(n_heads, sample_len, device):

    context_position = torch.arange(sample_len)[:, None].to(device)
    memory_position = torch.arange(sample_len)[None, :].to(device)
    relative_position = memory_position - context_position
    relative_position = torch.abs(relative_position).unsqueeze(0).expand(n_heads, -1, -1)

    slopes = torch.Tensor(get_alibi_slopes(n_heads)).to(device) * -1
    alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position
    alibi = alibi.view(n_heads, sample_len, sample_len)

    return alibi

def build_causal_alibi_bias(n_heads, tgt_len, src_len, device):
    # Get slopes
    slopes = torch.Tensor(get_alibi_slopes(n_heads)).to(device)

    # Create a distance matrix for the sequence.
    # We assume causal attention. For bidirectional, you might adjust.
    tgt_positions = torch.arange(tgt_len, device=device).unsqueeze(1)  # [tgt_len, 1]
    src_positions = torch.arange(src_len, device=device).unsqueeze(0)  # [1, src_len]

    # For each query position i and key position j,
    # distance = i - j (clamped to non-negative for causal attention)
    distance = tgt_positions - src_positions  # [seq_len, seq_len]

    # For causal attention, ensure positions in the future get a very low score.
    distance = distance.clamp(min=0).unsqueeze(0)  # [1, seq_len, seq_len]
    alibi = slopes.view(n_heads, 1, 1) * distance  # broadcast to [n_heads, seq_len, seq_len]

    return alibi  # No extra unsqueeze for batch dimension.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        encodings = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the div_term for the sine and cosine functions
        coefficient = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Sin for even and cos for odd
        encodings[:, 0::2] = torch.sin(position * coefficient)
        encodings[:, 1::2] = torch.cos(position * coefficient)

        # Change dimensions
        encodings = encodings.unsqueeze(0).transpose(0, 1)
        self.register_buffer('encodings', encodings)

    def forward(self, x):
        # Add positional encoding to the input tensor
        return x + self.encodings[:x.size(0), :]