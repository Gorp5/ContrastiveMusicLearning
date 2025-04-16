import torch.nn as nn
import torch
import torch.nn.functional as F

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

    def forward(self, query, key, value, alibi_bias, pos_emb=None):
        # query, key, value: [B, L, D]
        B, L, _ = query.shape
        # Linear projections, then reshape to [B, L, num_heads, head_dim] and transpose: [B, num_heads, L, head_dim]
        q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to q and k. (Assumes head_dim is even.)
        if pos_emb is not None:
            q = apply_RoPE(q, pos_emb)
            k = apply_RoPE(k, pos_emb)

        # Compute raw attention scores (dot product) [B, num_heads, L, L]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  ###

        # Add ALiBi bias (broadcasting over batch dimension)
        attn_scores = attn_scores + alibi_bias.unsqueeze(0)  # [B, num_heads, L, L]

        attn_weights = F.softmax(attn_scores, dim=-1) ###
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  ###
        # Reassemble: transpose and reshape to [B, L, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        output = self.out_proj(attn_output)
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

    def forward(self, src, alibi_bias, pos_emb):
        # src: [B, L, embed_dim]

        # Self-attention with residual connection and layer norm
        src2 = self.self_attn(src, src, src, alibi_bias, pos_emb)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed-forward network
        src2 = self.linear1(src)
        src2 = self.activation(src2)
        src2 = self.linear2(src2)

        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class RoPEALiBiTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=16, dim_feedforward=256, dropout=0.1):
        super().__init__()
        # Self-attention for the decoder (causal)
        self.self_attn = RoPEALiBiMultiheadAttention(d_model, num_heads, dropout=dropout)
        # Cross-attention: attend over encoder output (keys, values)
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

    def forward(self, tgt, memory,
                self_alibi_bias, self_pos_emb,
                cross_alibi_bias, cross_pos_emb):

        # Decode self-attention (with causal ALiBi, if desired)
        tgt2 = self.self_attn(tgt, tgt, tgt, self_alibi_bias, self_pos_emb)
        tgt2 = self.dropout1(tgt2)
        tgt = tgt + self.norm1(tgt2)

        # Cross-attention: Query = decoder, Keys/Values = encoder output
        tgt2 = self.cross_attn(tgt, memory, memory, cross_alibi_bias, cross_pos_emb)
        tgt2 = self.dropout2(tgt2)
        tgt = tgt + self.norm2(tgt2)

        # Feed-forward network
        tgt2 = self.linear1(tgt)
        tgt2 = self.activation(tgt2)
        tgt2 = self.linear2(tgt2)

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class RoPEALiBiTransformerEncoder(nn.Module):
    def __init__(self, num_layers=8, d_model=256, num_heads=16, dim_feedforward=256, seq_len=256, dropout=0.1, device='cpu'):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPEALiBiTransformerEncoderLayer(d_model=d_model, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.register_buffer("alibi_bias", build_alibi_bias(num_heads, seq_len, seq_len, device))
        self.register_buffer("pos_emb", get_RoPE_matrix(seq_len, d_model // num_heads, device))

        # self.alibi_bias = build_alibi_bias(num_heads, seq_len, seq_len, device).to(device)
        # self.pos_emb = get_RoPE_matrix(seq_len, d_model // num_heads, device).to(device)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        for layer in self.layers:
            src = layer(src, self.alibi_bias, self.pos_emb)
        return self.norm(src)


class RoPEALiBiTransformerDecoder(nn.Module):
    def __init__(self, num_layers=8, d_model=256, num_heads=16, dim_feedforward=256, seq_len=256, dropout=0.1, device='cpu'):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPEALiBiTransformerDecoderLayer(d_model=d_model, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.register_buffer("self_alibi_bias", build_alibi_bias(num_heads, seq_len, seq_len, device))
        self.register_buffer("self_pos_emb", get_RoPE_matrix(seq_len, d_model // num_heads, device))

        self.register_buffer("cross_alibi_bias", build_alibi_bias(num_heads, seq_len, seq_len, device))
        self.register_buffer("cross_pos_emb", get_RoPE_matrix(seq_len, d_model // num_heads, device))

        # self.self_alibi_bias = build_alibi_bias(num_heads, seq_len, seq_len, device).to(device)
        # self.cross_alibi_bias = build_alibi_bias(num_heads, seq_len, seq_len, device).to(device)
        # self.self_pos_emb = get_RoPE_matrix(seq_len, d_model // num_heads, device).to(device)
        # self.cross_pos_emb = get_RoPE_matrix(seq_len, d_model // num_heads, device).to(device)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory):
        for layer in self.layers:
            tgt = layer(tgt, memory, self.self_alibi_bias, self.self_pos_emb, self.cross_alibi_bias, self.cross_pos_emb)
        return self.norm(tgt)


# --- RoPE Application ---
def get_RoPE_matrix(seq_len, head_dim, device='cpu'):
    """
    Applies Rotary Positional Embedding (RoPE) to tensor x.
    Assumes x is of shape (batch, heads, seq_len, head_dim) and that head_dim is even.
    This function creates the rotation angles on the fly.
    """

    # Create position indices (seq_len,) on device of x
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)  # [seq_len]
    # Create dimension indices (head_dim/2,)
    dim_indices = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    # Compute scaling factors (the standard formula uses 10000^(2i/head_dim))
    inv_freq = 1.0 / (10000 ** (dim_indices / head_dim))  # [head_dim/2]

    # Compute sinusoid frequencies for each position.
    sinusoid_inp = torch.einsum("i,j->ij", pos, inv_freq)  # [seq_len, head_dim/2]
    sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()  # each: [seq_len, head_dim/2]

    # Expand sin and cos to [1,1,seq_len, head_dim/2] for broadcasting.
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim/2]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim/2]

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

# def apply_RoPE(x, pos_emb):
#     head_dim = x.size(-1)
#     cos_pos = pos_emb[:, :head_dim // 2]  # [L, head_dim//2]
#     sin_pos = pos_emb[:, head_dim // 2:]  # [L, head_dim//2]
#
#     # Split the input tensor along the head dimension into two halves
#     x1 = x[..., :head_dim // 2]  # [B, num_heads, L, head_dim//2]
#     x2 = x[..., head_dim // 2:]  # [B, num_heads, L, head_dim//2]
#
#     # Apply the rotation:
#     print(x1.shape)
#     print(cos_pos.shape)
#     print(sin_pos.shape)
#
#     x_rot_first = x1 * cos_pos - x2 * sin_pos
#     x_rot_second = x2 * cos_pos + x1 * sin_pos
#
#     # Concatenate back along the head dimension
#     x_rot = torch.cat([x_rot_first, x_rot_second], dim=-1)
#     return x_rot

# --- ALiBi Bias ---
def get_alibi_slopes(n_heads):
    # Implementation borrowed from ALiBi paper/repositories.
    def get_slopes(n):
        # If n is a power of 2, use the formula directly
        import math
        def power_of_two(i):
            return 2 ** (-(2 ** -(math.log2(n) - 3)) * i)

        return [power_of_two(i) for i in range(n)]

    # If not a power-of-two, use a simple strategy (or use a more refined approach)
    return torch.tensor(get_slopes(n_heads))

def build_alibi_bias(n_heads, tgt_len, src_len, device):
    """
    Build an ALiBi bias tensor of shape (n_heads, seq_len, seq_len)
    that can be added to the attention logits.
    """
    slopes = get_alibi_slopes(n_heads).to(device)  # [n_heads]
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