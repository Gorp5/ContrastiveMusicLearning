import torch.nn as nn
import torch
import torch.nn.functional as F
from models import RopeALiBiModelComponents


class AudioVariationalTransformer(nn.Module):
    def __init__(self, input_dim=64, num_heads=16, transformer_layers=8, length=256, d_model=256, dim_feedforward=512, checkpointing=False, dropout=0.1,
                 latent_space=64, name_extension="", use_alibi=True, use_rope=True, device='cuda'):
        super(AudioVariationalTransformer, self).__init__()
        self.name = f"AudioVariationalTransformer-LatentSpace{latent_space}-Heads{num_heads}-TrasformerLayers{transformer_layers}-DModel{length}-Dropout{dropout}{name_extension}"


        self.query_tokens = nn.Parameter(torch.randn(length // 2, d_model))

        self.device = device

        # Try bigger Conv kernel
        self.in_channels = 64
        self.start_conv_channels = 64
        self.out_channels = 512

        self.stride = 2
        self.padding = 2
        self.kernel_size = 5
        self.output_padding = 1


        # Test with no projection layer
        self.projection = nn.Linear(self.out_channels, d_model)
        self.projection_gelu = nn.GELU()

        # Last Linear Layer
        self.fc_out = nn.Linear(d_model, self.out_channels)
        self.fc_gelu = nn.GELU()

        self.encoder = RopeALiBiModelComponents.RoPEALiBiTransformerEncoder(num_layers=transformer_layers,
                                                                            d_model=d_model,
                                                                            num_heads=num_heads,
                                                                            dim_feedforward=dim_feedforward,
                                                                            seq_len=length,
                                                                            dropout=dropout,
                                                                            checkpointing=checkpointing,
                                                                            use_alibi=use_alibi,
                                                                            use_rope=use_rope,
                                                                            device=device)
        self.inputConvOuter = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

        self.inputConvMiddle = nn.Conv1d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding
        )

        self.inputConvInner = nn.Conv1d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding
        )

        self.outputConvInner = nn.ConvTranspose1d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding
        )


        self.outputConvMiddle = nn.ConvTranspose1d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding
        )

        self.outputConvOuter = nn.ConvTranspose1d(
            in_channels=self.out_channels,
            out_channels=self.in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=1
        )

        self.encode_to_latent = nn.Sequential(
            nn.Linear(d_model * length // 2, latent_space // 2),
            nn.GELU(),
            nn.Linear(latent_space // 2, latent_space),
            nn.GELU(),
            nn.LayerNorm(latent_space)
        )

        self.contrastive_head = ContrastiveHead(latent_space, projection_dim=latent_space)

        # latent mean and variance
        self.mean_layer = nn.Linear(latent_space, latent_space)
        self.logvar_layer = nn.Linear(latent_space, latent_space)

        # Latent Space

        self.encode_from_latent = nn.Sequential(
            nn.Linear(latent_space, d_model * length // 2),
            nn.GELU(),
            nn.LayerNorm(d_model * length // 2)
        )

        self.decoder = RopeALiBiModelComponents.RoPEALiBiTransformerDecoder(num_layers=max(1, transformer_layers // 3),
                                                                            d_model=d_model,
                                                                            num_heads=num_heads,
                                                                            dim_feedforward=dim_feedforward,
                                                                            seq_len=length,
                                                                            dropout=dropout,
                                                                            checkpointing=checkpointing,
                                                                            use_alibi=use_alibi,
                                                                            use_rope=use_rope,
                                                                            device=device)



    def forward(self, x, mask=None, return_contrastive=False):
        mean, logvar = self.to_latent(x, mask)
        z = self.reparameterization(mean, logvar)
        reconstruction = self.from_latent(z, mask)

        if return_contrastive:
            z_proj = self.contrastive_head(z)
            return reconstruction, mean, logvar, z_proj

        return reconstruction, mean, logvar

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def to_latent(self, x, mask):
        # x: [B, T, L]
        # B, T, L = x.shape

        x = x.permute(0, 2, 1)

        # Reshape to apply same conv to each of the L channels: [B * L, 1, T]
        #x = x.view(B * L, self.in_channels, T)

        memory = self.inputConvOuter(x)
        memory = self.inputConvMiddle(memory)
        memory = self.inputConvInner(memory)

        # [B, L', T']
        #memory = memory.view(B, T // 2, self.out_channels * self.out_channels2 * L)

        memory = memory.permute(0, 2, 1)

        #[B, T, L]
        # Project to Model Dimension
        memory = self.projection(memory)
        memory = self.projection_gelu(memory)
        # [B, 128, 256]
        # Encoder Block
        memory = self.encoder(memory, mask)  # Pass through Transformer encoder

        # [B, 128, 256]
        # Reshape for decoder input (Concatenation)
        memory = memory.reshape(memory.size(0), -1)
        # [B, 32768]
        # Project to Latent Space
        memory = self.encode_to_latent(memory)

        # [B, 128]

        mean = self.mean_layer(memory)
        logvar = self.logvar_layer(memory)

        return mean, logvar

    def from_latent(self, x, mask):
        # Project from Latent Space
        memory = self.encode_from_latent(x)
        # [B, 32768]
        # Reshape from concatenated representation and get Query Tokens
        queries = self.query_tokens.unsqueeze(0).repeat(x.size(0), 1, 1)
        memory = memory.reshape(x.size(0), queries.shape[1], -1)
        # [B, 128 256]
        # Decoder Block
        memory = self.decoder(queries, memory, mask)

        # Project to Output Dimension
        memory = self.fc_out(memory)
        memory = self.fc_gelu(memory)

        #B, T, L = memory.shape

        # [B, T // 2, L * 16 * 64]
        memory = memory.permute(0, 2, 1)
        # [B, L * 16 * 64, T // 2]

        memory = self.outputConvInner(memory)
        memory = self.outputConvMiddle(memory)
        memory = self.outputConvOuter(memory)

        memory = memory.permute(0, 2, 1)

        return memory

    def set_checkpointing(self, checkpointing):
        self.encoder.set_checkpointing(checkpointing)
        self.decoder.set_checkpointing(checkpointing)

class ContrastiveHead(nn.Module):
    def __init__(self, latent_dim, projection_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, projection_dim)
        )

    def forward(self, z):
        z_proj = self.projection(z)
        z_proj = F.normalize(z_proj, dim=-1)  # L2 normalization
        return z_proj
