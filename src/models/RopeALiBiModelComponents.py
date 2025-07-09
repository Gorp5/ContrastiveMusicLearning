import torch.nn as nn
import torch
import torch.nn.functional as F
import math
#from torch.utils.checkpoint import checkpoint

from flash_attn import flash_attn_func

class RoPEALiBiMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, alibi_slopes, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.alibi_slopes = alibi_slopes.to('cuda', dtype=torch.float32)

        self.embed_dim = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, alibi_bias=None, pos_emb=None, mask=None):
        query = query.half()
        key = key.half()
        value = value.half()

        QB, QL, _ = query.shape
        KB, KL, _ = key.shape
        VB, VL, _ = value.shape

        query = query.view(QB, QL, self.num_heads, self.head_dim)
        key = key.view(KB, KL, self.num_heads, self.head_dim)
        value = value.view(VB, VL, self.num_heads, self.head_dim)

        output = flash_attn_func(
            query, key, value,
            dropout_p=0.0,
            causal=True,
            alibi_slopes=self.alibi_slopes
        )

        output = output.view(QB, QL, self.embed_dim)

        output = output.float()

        # #query, key, value: [Batch, Time, Dim]
        # B, L, _ = query.shape
        #
        # # Linear projections
        # q = self.q_proj(query)
        # q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        #
        # k = self.k_proj(key)
        # k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        #
        # v = self.v_proj(value)
        # v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        #
        # # Apply RoPE to q and k
        # if pos_emb is not None:
        #     q = apply_RoPE(q, pos_emb)
        #     k = apply_RoPE(k, pos_emb)
        #
        # # Compute raw attention scores
        # attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        # if mask is not None:
        #     # Assuming mask is [batch_size, seq_len] with True at positions to mask
        #     mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
        #     #mask = mask.expand(-1, attn_scores.size(1), attn_scores.size(2),-1)  # [batch_size, num_heads, seq_len, seq_len]
        #     attn_scores.masked_fill_(mask, float('-inf'))
        #
        # # Add ALiBi bias
        # if alibi_bias is not None:
        #     attn_scores = attn_scores + alibi_bias.unsqueeze(0)
        #
        # # Regularization
        # attn_weights = F.softmax(attn_scores, dim=-1)
        # attn_weights = self.dropout(attn_weights)
        #
        # # Apply Attention
        # attn_output = torch.matmul(attn_weights, v)
        #
        # # Reassemble
        # attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        # output = self.out_proj(attn_output)
        return output


class RoPEALiBiTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=16, dim_feedforward=256, dropout=0.1):
        super().__init__()

        self.self_attn = RoPEALiBiMultiheadAttention(d_model, num_heads, get_alibi_slopes(num_heads), dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, src, alibi_bias, pos_emb, mask):
        # src: [Batch, Time, Dim]

        # Self-attention Block
        src2 = self.self_attn(src, src, src, alibi_bias, pos_emb, mask)
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
                 checkpointing=False, use_rope=True, use_alibi=True, device='cpu'):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPEALiBiTransformerEncoderLayer(d_model=d_model, num_heads=num_heads, dim_feedforward=dim_feedforward,
                                             dropout=dropout)
            for _ in range(num_layers)
        ])

        self.checkpointing = checkpointing

        if use_alibi:
                self.register_buffer("alibi_bias", build_causal_alibi_bias(num_heads, seq_len, seq_len, device).to('cuda', dtype=torch.float32))
        else:
            self.alibi_bias = None

        if use_rope:
            self.register_buffer("pos_emb", get_RoPE_matrix(seq_len, d_model // num_heads, device).to('cuda', dtype=torch.float32))
        else:
            self.pos_emb = None

        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        for layer in self.layers:
            # if self.checkpointing:
            #     src = checkpoint(layer, src, self.alibi_bias, self.pos_emb, mask)
            # else:
            src = layer(src, self.alibi_bias, self.pos_emb, mask)

        return self.norm(src)

    def set_checkpointing(self, checkpointing):
        self.checkpointing = checkpointing


class RoPEALiBiTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=16, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.self_attn = RoPEALiBiMultiheadAttention(d_model, num_heads, get_alibi_slopes(num_heads), dropout=dropout)

        self.cross_attn = RoPEALiBiMultiheadAttention(d_model, num_heads, get_alibi_slopes(num_heads), dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, tgt, memory, self_alibi_bias, self_pos_emb, cross_alibi_bias, cross_pos_emb, mask):
        t = tgt.size(1)
        L = memory.size(1)

        # slice biases and position embeddings
        bias_self = self_alibi_bias[:, :t, :t] if self_alibi_bias is not None else None
        pos_self = self_pos_emb[:t] if self_pos_emb is not None else None
        bias_cross = cross_alibi_bias[:, :t, :L] if cross_alibi_bias is not None else None
        pos_cross = cross_pos_emb[:L] if cross_pos_emb is not None else None

        # Self Attention Block
        tgt2 = self.self_attn(tgt, tgt, tgt, bias_self, pos_self, mask)
        tgt2 = self.dropout1(tgt2)
        tgt = tgt + self.norm1(tgt2)

        # Cross-attention Block
        tgt2 = self.cross_attn(tgt, memory, memory, bias_cross, pos_cross, mask)
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
    def __init__(self, num_layers=8, d_model=256, num_heads=16, dim_feedforward=256, seq_len=256, dropout=0.1, checkpointing=False, use_rope=True, use_alibi=True, device='cpu'):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPEALiBiTransformerDecoderLayer(d_model=d_model, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.checkpointing = checkpointing

        if use_alibi:
            self.register_buffer("self_alibi_bias", build_causal_alibi_bias(num_heads, seq_len, seq_len, device))
            self.register_buffer("cross_alibi_bias", build_causal_alibi_bias(num_heads, seq_len, seq_len, device))
        else:
            self.self_alibi_bias = None
            self.cross_alibi_bias = None

        if use_rope:
            self.register_buffer("self_pos_emb", get_RoPE_matrix(seq_len, d_model // num_heads, device))
            self.register_buffer("cross_pos_emb", get_RoPE_matrix(seq_len, d_model // num_heads, device))
        else:
            self.self_pos_emb = None
            self.cross_pos_emb = None


        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, mask=None):
        for layer in self.layers:
            # if self.checkpointing:
            #     tgt = checkpoint(layer, tgt, memory, self.self_alibi_bias, self.self_pos_emb, self.cross_alibi_bias, self.cross_pos_emb, mask)
            # else:
            tgt = layer(tgt, memory, self.self_alibi_bias, self.self_pos_emb, self.cross_alibi_bias, self.cross_pos_emb, mask)

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
        return [start * ratio**i for i in range(nheads)]

    if math.log2(nheads).is_integer():
        return torch.tensor(get_slopes_power_of_2(nheads))
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(nheads))
        return torch.tensor(
            get_slopes_power_of_2(closest_power_of_2)
            + get_alibi_slopes(2 * closest_power_of_2)[0::2][: nheads - closest_power_of_2]
        )

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