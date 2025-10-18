import torch.nn as nn
import torch
from models import RopeALiBiModelComponents


class AudioTransformerWeavedEmbedding(nn.Module):
    def __init__(self, input_dim=64, num_heads=16, encoder_layers=16, decoder_layers=8, length=256, d_model=256, dim_feedforward=512, checkpointing=False, dropout=0.1,
                 latent_space=64, name_extension="", use_alibi=True, use_rope=False, autoregressive=False, genre_count=0, mood_count=0):
        super(AudioTransformerWeavedEmbedding, self).__init__()
        self.name = f"AudioTransformerWeavedEmbedding-LatentSpace{latent_space}-Heads{num_heads}-EncoderLayers{encoder_layers}-DecoderLayers{decoder_layers}-DModel{length}-Dropout{dropout}-AutoRegressive{autoregressive}{name_extension}"

        self.model_dim = d_model
        self.projection = nn.Linear(input_dim, d_model)
        self.projection_gelu = nn.GELU()

        self.query_tokens = nn.Parameter(torch.randn(length, d_model))

        self.length = length

        self.encoder_featurewise = RopeALiBiModelComponents.CustomTransformerEncoder(num_layers=encoder_layers // 4,
                                                                                     d_model=length,
                                                                                     num_heads=num_heads,
                                                                                     dim_feedforward=dim_feedforward,
                                                                                     seq_len=d_model,
                                                                                     dropout=dropout,
                                                                                     use_alibi=False,
                                                                                     use_rope=True,
                                                                                     device='cuda')

        self.encoder_timewise = RopeALiBiModelComponents.CustomTransformerEncoder(num_layers=encoder_layers,
                                                                                  d_model=d_model,
                                                                                  num_heads=num_heads,
                                                                                  dim_feedforward=dim_feedforward,
                                                                                  seq_len=length,
                                                                                  dropout=dropout,
                                                                                  use_alibi=use_alibi,
                                                                                  use_rope=use_rope,
                                                                                  device='cuda')

        self.encode_to_latent = nn.Linear(d_model * length, latent_space)
        self.encode_to_latent_gelu = nn.GELU()

        # Last Linear Layer
        self.fc_out = nn.Linear(d_model, input_dim)
        self.fc_gelu = nn.GELU()

    def forward(self, x, mask=None):

        memory = x.permute(0, 2, 1)

        # Project to Model Dimension
        memory = self.projection(memory)
        memory = self.projection_gelu(memory)

        memory_2 = memory.permute(0, 2, 1)
        memory_2 = self.encoder_featurewise(memory_2, mask)  # Pass through Transformer encoder
        memory_2 = memory_2.permute(0, 2, 1)

        memory = memory + memory_2

        # Encoder Block
        memory = self.encoder_timewise(memory, mask)  # Pass through Transformer encoder

        # Reshape for decoder input (Concatenation)
        memory = memory.reshape(memory.size(0), -1)

        # Project to Latent Space
        memory = self.encode_to_latent(memory)
        memory = self.encode_to_latent_gelu(memory)

        # Project to Output Dimension
        memory = self.fc_out(memory)
        memory = self.fc_gelu(memory)

        memory = memory.permute(0, 2, 1)

        return memory
