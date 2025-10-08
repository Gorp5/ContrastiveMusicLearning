import math

import torch.nn as nn
import torch

from models import RopeALiBiModelComponents
from einops.layers.torch import Rearrange

from models.RopeALiBiModelComponents import get_alibi_slopes


class AudioViTEncoder(nn.Module):
    def __init__(self, patch_size=8, input_dim=128, num_heads=16, encoder_layers=8, length=256, d_model=256, dim_feedforward=512, checkpointing=False, dropout=0.1,
                 latent_space=256, use_alibi=True, use_pooling=False, CLS=False, use_rope=True, masking_percent=0.0, variational=False, custom_slopes=-1):
        super(AudioViTEncoder, self).__init__()

        patch_dim = patch_size * patch_size

        self.model_dim = d_model
        self.latent_space = latent_space

        patches_vertical = input_dim // patch_size
        patches_horizontal = length // patch_size

        self.total_patches = int(patches_vertical * patches_horizontal * (1.0 - masking_percent))
        self.variational = variational

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size, h=patches_vertical, w=patches_horizontal),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.positional_embedding = RopeALiBiModelComponents.PositionalEncoding(d_model=d_model, max_len=length + 1)
        #self.positional_embedding = nn.Parameter(torch.randn(patches_vertical * patches_horizontal + 1, d_model))
        #self.add_positional_embedding = PositionalEncodings2D(max_length=patches_horizontal, height=patches_vertical, d_model=d_model)
        #self.positional_embedding = SinusoidalPositionalEmbedding2D(dim=d_model, max_h=patches_vertical, max_w=patches_horizontal)

        self.masking = RandMasker(masking_percent=masking_percent)

        self.CLS = CLS
        self.projection_gelu = nn.GELU()

        # Project latent back to initial query space (one vector per patch)
        #self.latent_to_queries = nn.Linear(latent_space, d_model * self.total_patches)

        # Learnable positional embeddings for queries
        self.query_pos = nn.Parameter(torch.randn(self.total_patches, self.model_dim))
        self.latent_to_q = nn.Linear(self.latent_space, self.model_dim * self.total_patches)

        self.length = length

        self.encoder = RopeALiBiModelComponents.RoPEALiBiTransformerEncoder(num_layers=encoder_layers,
                                                                            d_model=d_model,
                                                                            num_heads=num_heads,
                                                                            dim_feedforward=dim_feedforward,
                                                                            seq_len=self.total_patches,
                                                                            dropout=dropout,
                                                                            checkpointing=checkpointing,
                                                                            use_alibi=use_alibi,
                                                                            use_rope=use_rope,
                                                                            custom_slopes=custom_slopes,
                                                                            device='cuda')

        self.use_pooling = use_pooling
        if self.use_pooling:
            self.attention_pooling = AttentionPooling(d_model=d_model, latent_dim=latent_space, num_heads=8)
        elif self.CLS:
            self.cls = torch.nn.Parameter(
                torch.randn(1, 1, d_model)
            )

            self.encode_to_latent = nn.Linear(d_model, latent_space)
            self.encode_to_latent_gelu = nn.GELU()
        else:
            self.encode_to_latent = nn.Linear(d_model * self.total_patches, latent_space)
            self.encode_to_latent_gelu = nn.GELU()

        self.mean_layer = nn.Linear(latent_space, latent_space)
        self.logvar_layer = nn.Linear(latent_space, latent_space)

        self.final_out = nn.Linear(latent_space, latent_space)

    def forward(self, x, mask=None, masked=False):
        x = x.unsqueeze(1)
        x = self.to_patch_embedding(x)
        if self.CLS:
            B, T, F = x.shape
            x = torch.cat([self.cls.expand(B, 1, F), x], dim=1)

        x = self.positional_embedding(x)
        #x = x + self.positional_embedding.expand(B, -1, -1)[:x.size(0), :]

        if masked:
            x = self.masking(x)

        x = self.encoder(x, mask)

        if self.use_pooling:
            x = self.attention_pooling(x, mask)
        else:
            if self.CLS:
                x = x[:, 0, :]
            else:
                x = x.reshape(x.size(0), -1)

            x = self.encode_to_latent(x)
            #x = self.encode_to_latent_gelu(x)

        x = self.final_out(x)

        if self.variational:
            mean = self.mean_layer(x)
            logvar = self.logvar_layer(x)

            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            x = mean + std * eps, mean, logvar

        return x


class RandMasker(nn.Module):
    def __init__(self, masking_percent=0.5):
        super().__init__()
        self.masking_percent = masking_percent

    def forward(self, input):
        if self.masking_percent == 0.0:
            return input

        B, T, F = input.shape
        x = int(T * (1 - self.masking_percent))  # total tokens to keep
        x_rest = x - 1  # leave room for the first token

        # Handle edge case: only the first token survives
        if x_rest <= 0:
            return input[:, :1, :]  # [B, 1, F]

        # Sample indices from [1..T-1]
        idx0 = torch.randint(0, (T - 1) - x_rest + 1, (B, x_rest), device=input.device)
        idx0 = torch.sort(idx0, dim=1).values

        # Add offsets to ensure strictly increasing indices
        offset = torch.arange(x_rest, device=input.device).view(1, x_rest)
        sampled_idx = idx0 + offset + 1  # shift by +1 to skip the first token

        # Prepend the saved first element (index 0)
        cls_idx = torch.zeros((B, 1), dtype=torch.long, device=input.device)
        idx = torch.cat([cls_idx, sampled_idx], dim=1)  # [B, x]

        # Gather along the T dimension
        batch_idx = torch.arange(B, device=input.device).view(B, 1).expand(B, x)
        output = input[batch_idx, idx, :]  # [B, x, F]
        return output


class PositionalEncodings2D(nn.Module):
    def __init__(self, max_length=16, height=8, d_model=256):
        super().__init__()
        self.height = height
        self.max_length = max_length
        self.frequency_tokens = nn.Parameter(torch.randn(1, height, 1, d_model))
        self.time_tokens = nn.Parameter(torch.randn(1, 1, max_length, d_model))

    def forward(self, x, mask=None):
        B, combined, F = x.shape
        x = x.reshape(B, self.height, -1, F)
        B, H, T, F = x.shape

        if T > self.max_length:
            raise IndexError # Throws an Error

        x = x + self.frequency_tokens.expand(B, -1, T, -1)
        x = x + self.time_tokens[:,:,:T,:].expand(B, self.height, -1, -1)

        return x.reshape(B, H * T, F)


class SinusoidalPositionalEmbedding2D(nn.Module):
    """
    2D sinusoidal positional embeddings.
    Input shape: [B, H, W, F]
    Output shape: [B, H, W, F]
    """
    def __init__(self, dim: int, max_h: int, max_w: int):
        super().__init__()
        self.dim = dim
        self.max_h = max_h
        self.max_w = max_w

        # Precompute embeddings for both dimensions
        self.register_buffer("pe_h", self._build_sinusoid_table(max_h, dim))
        self.register_buffer("pe_w", self._build_sinusoid_table(max_w, dim))

    def _build_sinusoid_table(self, length: int, dim: int):
        """Builds sinusoidal table of shape [length, dim]."""
        position = torch.arange(length, dtype=torch.float).unsqueeze(1)  # [length, 1]
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        return pe  # [length, dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, F]
        Returns:
            [B, H, W, F] with positional encodings added
        """

        B, combined, F = x.shape
        x = x.reshape(B, self.height, -1, F)
        B, H, W, F = x.shape
        assert F == self.dim, "Feature dim mismatch"

        # Slice embeddings
        pe_h = self.pe_h[:H].unsqueeze(1)  # [H, 1, F]
        pe_w = self.pe_w[:W].unsqueeze(0)  # [1, W, F]

        # Broadcast to grid
        pos_emb = pe_h + pe_w              # [H, W, F]
        pos_emb = pos_emb.unsqueeze(0).expand(B, H, W, F)

        x = x + pos_emb
        return x.reshape(B, combined, F)

class AttentionPooling(nn.Module):
    def __init__(self, d_model, latent_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.pool_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.attn = RopeALiBiModelComponents.RoPEALiBiMultiheadAttention(d_model, num_heads, get_alibi_slopes(num_heads), dropout=dropout)

        self.to_latent = nn.Linear(d_model, latent_dim)

    def forward(self, x, mask=None):
        B = x.size(0)

        memory = self.pool_token.expand(B, 1, -1)
        pooled = self.attn(memory, x, x, mask=mask)

        return self.to_latent(pooled.squeeze(1))
