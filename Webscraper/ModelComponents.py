import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

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
        # query, key, value: [Batch, Time, Dim]
        B, L, _ = query.shape

        # Linear projections
        q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to q and k
        if pos_emb is not None:
            q = apply_RoPE(q, pos_emb)
            k = apply_RoPE(k, pos_emb)

        # Compute raw attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Add ALiBi bias
        if alibi_bias is not None:
            attn_scores = attn_scores + alibi_bias.unsqueeze(0)

        # Regularization
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply Attention
        attn_output = torch.matmul(attn_weights, v)

        # Reassemble
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
        # src: [Batch, Time, Dim]

        # Self-attention Block
        src2 = self.self_attn(src, src, src, alibi_bias, pos_emb)
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

        self.register_buffer("alibi_bias", build_alibi_bias(num_heads, seq_len, seq_len, device)) if use_alibi else None

        self.register_buffer("pos_emb", get_RoPE_matrix(seq_len, d_model // num_heads, device)) if use_rope else None

        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        for layer in self.layers:
            if self.checkpointing:
                src = checkpoint(layer, src, self.alibi_bias, self.pos_emb)
            else:
                src = layer(src, self.alibi_bias, self.pos_emb)

        return self.norm(src)

    def set_checkpointing(self, checkpointing):
        self.checkpointing = checkpointing


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

    def forward(self, tgt, memory, self_alibi_bias, self_pos_emb, cross_alibi_bias, cross_pos_emb):

        # Self Attention Block
        tgt2 = self.self_attn(tgt, tgt, tgt, self_alibi_bias, self_pos_emb)
        tgt2 = self.dropout1(tgt2)
        tgt = tgt + self.norm1(tgt2)

        # Cross-attention Block
        tgt2 = self.cross_attn(tgt, memory, memory, cross_alibi_bias, cross_pos_emb)
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

        self.register_buffer("self_alibi_bias", build_alibi_bias(num_heads, seq_len, seq_len, device)) if use_alibi else None
        self.register_buffer("cross_alibi_bias", build_alibi_bias(num_heads, seq_len, seq_len, device)) if use_alibi else None

        self.register_buffer("self_pos_emb", get_RoPE_matrix(seq_len, d_model // num_heads, device)) if use_rope else None
        self.register_buffer("cross_pos_emb", get_RoPE_matrix(seq_len, d_model // num_heads, device)) if use_rope else None

        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory):
        for layer in self.layers:
            if self.checkpointing:
                tgt = checkpoint(layer, tgt, memory, self.self_alibi_bias, self.self_pos_emb, self.cross_alibi_bias, self.cross_pos_emb)
            else:
                tgt = layer(tgt, memory, self.self_alibi_bias, self.self_pos_emb, self.cross_alibi_bias, self.cross_pos_emb)

        return self.norm(tgt)

    def set_checkpointing(self, checkpointing):
        self.checkpointing = checkpointing


# TODO: Needs to implement optional RoPE and ALiBi
class RoPEALiBiConformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=16, dim_feedforward=256, dropout=0.1, kernel_size=5, stride=1, seq_len=256, device='cpu'):
        super().__init__()

        self.self_attn = RoPEALiBiMultiheadAttention(d_model, num_heads, dropout=dropout)

        self.linear1 = nn.Linear(d_model * 2, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model * 2)

        self.convolution = DepthwiseConvolutionBlock(d_model, kernel_size, stride=stride)

        self.projection = nn.Linear(seq_len, seq_len // stride)
        self.projection2 = nn.Linear(d_model, d_model * stride)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model * 2)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

        self.register_buffer("alibi_bias", build_alibi_bias(num_heads, seq_len, seq_len, device))
        self.register_buffer("pos_emb", get_RoPE_matrix(seq_len, d_model // num_heads, device))

    def forward(self, src):
        # src: [Batch, Time, Dim]

        # Self-attention, residual connection, layer norm
        src2 = self.self_attn(src, src, src, self.alibi_bias, self.pos_emb)
        src2 = self.dropout1(src2)
        src = src + self.norm1(src2)

        # Convolution
        conv = self.convolution(src)

        # Project along 2 dimensions for skip connection
        proj = src.permute(0, 2, 1)
        proj = self.projection(proj)
        proj = proj.permute(0, 2, 1)
        proj = self.projection2(proj)

        # Skip
        src = proj + conv

        # Feed-forward network
        src2 = self.linear1(src)
        src2 = self.activation(src2)
        src2 = self.linear2(src2)


        # Regularization
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


# TODO: Needs to implement optional RoPE and ALiBi
class RoPEALiBiConformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=16, dim_feedforward=256, kernel_size=5, seq_len=256, dropout=0.1, stride=1, device='cpu'):
        super().__init__()
        self.self_attn = RoPEALiBiMultiheadAttention(d_model, num_heads, dropout=dropout)

        self.cross_attn = RoPEALiBiMultiheadAttention(d_model, num_heads, dropout=dropout)

        self.convolution = DepthwiseConvolutionTransposeBlock(d_model, kernel_size, stride)

        self.projection = nn.Linear(seq_len, seq_len * stride)
        self.projection2 = nn.Linear(d_model, d_model // stride)

        self.linear1 = nn.Linear(d_model // 2, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model // 2)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model // 2)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self.register_buffer("self_alibi_bias", build_alibi_bias(num_heads, seq_len, seq_len, device))
        self.register_buffer("self_pos_emb", get_RoPE_matrix(seq_len, d_model // num_heads, device))


    def forward(self, memory):

        # Self-Attention
        tgt2 = self.self_attn(memory, memory, memory, self.self_alibi_bias, self.self_pos_emb)
        tgt2 = self.dropout1(tgt2)
        memory = memory + self.norm1(tgt2)

        # Cross-attention
        tgt2 = self.cross_attn(tgt, memory, memory, self.cross_alibi_bias, self.cross_pos_emb)
        tgt2 = self.dropout2(tgt2)
        tgt = tgt + self.norm2(tgt2)

        # Convolution
        conv = self.convolution(memory)

        # Projection across 2 dimensions
        proj = memory.permute(0, 2, 1)
        proj = self.projection(proj)
        proj = proj.permute(0, 2, 1)
        proj = self.projection2(proj)

        memory = proj + conv

        # Feed-forward network
        tgt2 = self.linear1(memory)
        tgt2 = self.activation(tgt2)
        tgt2 = self.linear2(tgt2)

        # Regularization
        memory = memory + self.dropout3(tgt2)
        memory = self.norm3(memory)

        return memory

# TODO: Needs to implement optional RoPE and ALiBi
class RoPEALiBiConformerEncoder(nn.Module):
    def __init__(self, num_layers=5, d_model=256, num_heads=16, dim_feedforward=256, seq_len=256, dropout=0.1, stride=1, kernel_size=5, device='cpu'):
        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(RoPEALiBiConformerEncoderLayer(d_model=d_model, num_heads=num_heads, dim_feedforward=dim_feedforward,
                                              dropout=dropout, kernel_size=kernel_size, stride=stride, seq_len=seq_len, device=device))
        self.norm = nn.LayerNorm(b)

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return self.norm(src)

# TODO: Needs to implement optional RoPE and ALiBi
class RoPEALiBiConformerDecoder(nn.Module):
    def __init__(self, num_layers=8, d_model=256, num_heads=16, dim_feedforward=256, seq_len=256, dropout=0.1,
                 kernel_size=5, device='cpu', stride = 1):
        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                RoPEALiBiConformerDecoderLayer(d_model=d_model, num_heads=num_heads, dim_feedforward=dim_feedforward,
                                               dropout=dropout, kernel_size=kernel_size, stride=stride, seq_len=seq_len,
                                               device=device))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, memory):
        for layer in self.layers:
            memory = layer(memory)

        return self.norm(memory)

class DepthwiseConvolutionBlock(nn.Module):
    def __init__(self, d_model=256, kernel_size=3, stride=1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        # Pointwise convolution + GLU
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)

        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=stride,
            groups=d_model
        )

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(d_model)

        # Activation (Swish)
        self.activation = F.hardswish

        # Pointwise convolution
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

        # Dropout
        self.dropout = nn.Dropout(0.1)

        # Gelu
        self.glu = nn.GELU()

        # Downsample for residual (If stride > 1)
        if stride > 1:
            self.residual_downsample = nn.AvgPool1d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        residual = x

        # Depthwise Seperable Convolution
        x = self.layer_norm(x)
        x = x.transpose(1, 2)

        x = self.pointwise_conv1(x)
        x = self.glu(x)

        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)

        # Residual
        residual = residual.transpose(1, 2)
        residual = self.residual_downsample(residual)
        residual = residual.transpose(1, 2)

        x = x + residual

        return x

class DepthwiseConvolutionTransposeBlock(nn.Module):
    def __init__(self, d_model=256, kernel_size=3, stride=1):
        super().__init__()

        self.residual_downsample = nn.Upsample(1)

        # Dropout
        self.dropout = nn.Dropout(0.1)

        # Pointwise convolution
        self.pointwise_conv2 = nn.ConvTranspose1d(d_model, d_model, kernel_size=1)

        # Activation (Swish)
        self.activation = F.hardswish

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(d_model)

        # Depthwise convolution
        self.depthwise_conv = nn.ConvTranspose1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=stride,
            groups=d_model
        )

        # Pointwise convolution + GLU
        self.pointwise_conv1 = nn.ConvTranspose1d(d_model, 2 * d_model, kernel_size=1)


        self.glu = nn.GELU()

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)

        x = self.pointwise_conv1(x)
        x = self.glu(x)

        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)

        residual = residual.transpose(1, 2)
        residual = self.residual_downsample(residual)
        residual = residual.transpose(1, 2)
        x = x + residual

        return x


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
    # Get slopes
    slopes = get_alibi_slopes(n_heads).to(device)

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