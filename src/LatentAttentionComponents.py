import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

from ModelComponents import build_alibi_bias, get_RoPE_matrix, apply_RoPE


class RoPEALiBiLatentSelfAttention(nn.Module):
    def __init__(self, d_model, proj_dimension, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.embed_dim = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim ** -0.5

        self.latent_proj = nn.Linear(d_model, proj_dimension)

        self.q_proj = nn.Linear(proj_dimension, d_model)
        self.k_proj = nn.Linear(proj_dimension, d_model)
        self.v_proj = nn.Linear(proj_dimension, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, alibi_bias, pos_emb=None, mask=None):
        # query, key, value: [Batch, Time, Dim]
        B, L, _ = src.shape

        # Latent Projection
        projection = self.latent_proj(src)

        # Linear projections
        q = self.q_proj(projection).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(projection).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(projection).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to q and k
        if pos_emb is not None:
            q = apply_RoPE(q, pos_emb)
            k = apply_RoPE(k, pos_emb)

        # Compute raw attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if mask is not None:
            # Assuming mask is [batch_size, seq_len] with True at positions to mask
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            #mask = mask.expand(-1, attn_scores.size(1), attn_scores.size(2),-1)  # [batch_size, num_heads, seq_len, seq_len]
            attn_scores.masked_fill_(mask, float('-inf'))

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

class RoPEALiBiLatentCrossAttention(nn.Module):
    def __init__(self, d_model, proj_dimension, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.embed_dim = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim ** -0.5

        self.latent_proj = nn.Linear(d_model, proj_dimension)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(proj_dimension, d_model)
        self.v_proj = nn.Linear(proj_dimension, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, alibi_bias, pos_emb=None, mask=None):
        # query, key, value: [Batch, Time, Dim]
        B, L, _ = src.shape

        # Latent Projection
        projection = self.latent_proj(src)

        # Linear projections
        q = self.q_proj(tgt).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(projection).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(projection).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to q and k
        if pos_emb is not None:
            q = apply_RoPE(q, pos_emb)
            k = apply_RoPE(k, pos_emb)

        # Compute raw attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if mask is not None:
            # Assuming mask is [batch_size, seq_len] with True at positions to mask
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            # mask = mask.expand(-1, attn_scores.size(1), attn_scores.size(2),-1)  # [batch_size, num_heads, seq_len, seq_len]
            attn_scores.masked_fill_(mask, float('-inf'))

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


class RoPEALiBiLatentTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=16, dim_feedforward=256, proj_dimension=64, dropout=0.1):
        super().__init__()

        self.self_attn = RoPEALiBiLatentSelfAttention(d_model, proj_dimension, num_heads, dropout=dropout)

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
        src2 = self.self_attn(src, alibi_bias, pos_emb, mask)
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


class RoPEALiBiLatentTransformerEncoder(nn.Module):
    def __init__(self, num_layers=8, d_model=256, proj_dimension=64, num_heads=16, dim_feedforward=256, seq_len=256, dropout=0.1,
                 checkpointing=False, use_rope=True, use_alibi=True, device='cpu'):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPEALiBiLatentTransformerEncoderLayer(d_model=d_model, proj_dimension=proj_dimension, num_heads=num_heads, dim_feedforward=dim_feedforward,
                                             dropout=dropout)
            for _ in range(num_layers)
        ])

        self.checkpointing = checkpointing

        if use_alibi:
            self.register_buffer("alibi_bias", build_alibi_bias(num_heads, seq_len, seq_len, device))
        else:
            self.alibi_bias = None

        if use_rope:
            self.register_buffer("pos_emb", get_RoPE_matrix(seq_len, d_model // num_heads, device))
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


class RoPEALiBiLatentTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=16, dim_feedforward=256, proj_dimension=128, dropout=0.1):
        super().__init__()

        self.self_attn = RoPEALiBiLatentSelfAttention(d_model, proj_dimension, num_heads, dropout=dropout)

        self.cross_attn = RoPEALiBiLatentCrossAttention(d_model, proj_dimension, num_heads, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, memory, tgt, self_alibi_bias, self_pos_emb, cross_alibi_bias, cross_pos_emb, mask):

        # Self Attention Block
        tgt2 = self.self_attn(tgt, self_alibi_bias, self_pos_emb, mask)
        tgt2 = self.dropout1(tgt2)
        tgt = tgt + self.norm1(tgt2)

        # Cross-attention Block
        tgt2 = self.cross_attn(memory, tgt, cross_alibi_bias, cross_pos_emb, mask)
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


class RoPEALiBiLatentTransformerDecoder(nn.Module):
    def __init__(self, num_layers=8, d_model=256, num_heads=16, dim_feedforward=256, proj_dimension=128, seq_len=256, dropout=0.1, checkpointing=False, use_rope=True, use_alibi=True, device='cpu'):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPEALiBiLatentTransformerDecoderLayer(d_model=d_model, proj_dimension=proj_dimension, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.checkpointing = checkpointing

        if use_alibi:
            self.register_buffer("self_alibi_bias", build_alibi_bias(num_heads, seq_len, seq_len, device))
            self.register_buffer("cross_alibi_bias", build_alibi_bias(num_heads, seq_len, seq_len, device))
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
            #     tgt = checkpoint(layer, memory, tgt, self.self_alibi_bias, self.self_pos_emb, self.cross_alibi_bias, self.cross_pos_emb, mask)
            # else:
            tgt = layer(memory, tgt, self.self_alibi_bias, self.self_pos_emb, self.cross_alibi_bias, self.cross_pos_emb, mask)

        return self.norm(tgt)

    def set_checkpointing(self, checkpointing):
        self.checkpointing = checkpointing