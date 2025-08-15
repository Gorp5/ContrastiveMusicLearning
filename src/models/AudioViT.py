
import torch.nn as nn
import torch

from models import RopeALiBiModelComponents
from einops.layers.torch import Rearrange

from models.RopeALiBiModelComponents import get_alibi_slopes


class AudioViT(nn.Module):
    def __init__(self, patch_size=8, input_dim=128, num_heads=16, encoder_layers=8, decoder_layers=8, length=256, d_model=256, dim_feedforward=512, checkpointing=False, dropout=0.1,
                 latent_space=256, use_alibi=True, use_pooling=False):
        super(AudioViT, self).__init__()

        patch_dim = patch_size * patch_size

        self.model_dim = d_model
        self.latent_space = latent_space

        patches_vertical = input_dim // patch_size
        patches_horizontal = length // patch_size

        self.total_patches = patches_vertical * patches_horizontal

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size, h=patches_vertical, w=patches_horizontal),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )

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
                                                                            use_rope=False,
                                                                            device='cuda'
                                                                            )
        self.use_pooling = use_pooling
        if self.use_pooling:
            self.attention_pooling = AttentionPooling(d_model=d_model, latent_dim=latent_space, num_heads=8)
        else:
            self.encode_to_latent = nn.Linear(d_model * self.total_patches, latent_space)
            self.encode_to_latent_gelu = nn.GELU()

        # Latent Space Normalization
        self.norm = nn.LayerNorm(latent_space)

        self.encode_from_latent = nn.Linear(latent_space, d_model * self.total_patches)
        self.encode_from_latent_gelu = nn.GELU()

        self.decoder = RopeALiBiModelComponents.RoPEALiBiTransformerDecoder(num_layers=decoder_layers,
                                                                            d_model=d_model,
                                                                            num_heads=num_heads,
                                                                            dim_feedforward=dim_feedforward,
                                                                            seq_len=self.total_patches,
                                                                            dropout=dropout,
                                                                            checkpointing=checkpointing,
                                                                            use_alibi=use_alibi,
                                                                            use_rope=False,
                                                                            device='cuda')

        self.from_patch_embedding = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=patches_vertical, w=patches_horizontal),
        )

    def forward(self, x, mask=None):

        latent = self.to_latent(x, mask)
        memory = self.from_latent(latent, mask)

        return memory, latent

    def to_latent(self, x, mask):
        x = x.unsqueeze(1)
        # Project to Model Dimension
        memory = self.to_patch_embedding(x)

        # Encoder Block
        memory = self.encoder(memory, mask)  # Pass through Transformer encoder

        # Reshape for decoder input (Concatenation)
        memory = memory.reshape(memory.size(0), -1)

        # Project to Latent Space
        if self.use_pooling:
            memory = self.attention_pooling(memory, mask)
        else:
            memory = self.encode_to_latent(memory)
            memory = self.encode_to_latent_gelu(memory)


        memory = self.norm(memory)

        return memory


    def from_latent(self, latent, mask):
        B = latent.size(0)

        # Project latent vector into decoder memory (cross-attention keys/values)
        memory = self.encode_from_latent(latent)
        memory = self.encode_from_latent_gelu(memory)
        memory = memory.view(B, self.total_patches, self.model_dim)

        # Project latent vector to total_patches * d_model
        queries = self.latent_to_q(latent)  # shape: [B, total_patches * d_model]
        queries = queries.view(B, self.total_patches, self.model_dim)

        queries = queries + self.query_pos.unsqueeze(0)  # if you want explicit positional info

        # Run decoder: queries cross-attend to memory
        memory = self.decoder(queries, memory, mask)  # assumes decoder(query, memory, mask)

        # Project to Output Dimension
        memory = self.from_patch_embedding(memory)
        memory = memory.squeeze(1)
        return memory



    def set_checkpointing(self, checkpointing):
        self.encoder.set_checkpointing(checkpointing)
        self.decoder.set_checkpointing(checkpointing)

class AttentionPooling(nn.Module):
    def __init__(self, d_model, latent_dim, num_heads=8, dropout=0.1):
        super().__init__()
        # Learnable pooling token (like [CLS])
        self.pool_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Multihead attention: pool_token queries the sequence
        self.attn = RopeALiBiModelComponents.RoPEALiBiMultiheadAttention(d_model, num_heads, get_alibi_slopes(num_heads), dropout=dropout)

        # Project pooled output to latent space
        self.to_latent = nn.Linear(d_model, latent_dim)

    def forward(self, x, mask=None):
        """
        x: [B, seq_len, d_model]  (transformer encoder output)
        mask: optional attention mask
        """
        B = x.size(0)

        # Perform attention: query=pool_token, key/value=x
        memory = self.pool_token.expand(B, 1, -1)
        pooled = self.attn(memory, x, x, mask = mask)  # [B, 1, d_model]

        # Project to latent space
        return self.to_latent(pooled.squeeze(1))  # [B, latent_dim]
