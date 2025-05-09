import torch.nn as nn
import math


from typing import Any, Callable, Optional, Union

import torch
from torch import Tensor


device = "cuda"


# ============ Rotational Positional Embeddings ============ #
def create_rotary_embedding(seq_len, d_model):
    theta = 10000 ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    angles = position / theta
    sin, cos = torch.sin(angles), torch.cos(angles)

    # Concatenate sin and cos into a full transformation matrix
    pos_emb_matrix = torch.cat([sin, cos], dim=-1)  # Shape: (seq_len, d_model)
    return pos_emb_matrix.to(device)





class AudioTransformerEncoderCNNReconstruction(nn.Module):
    def __init__(self, input_dim=64, num_heads=16, transformer_layers=8, length=256, d_model=256, dropout=0.1,
                 latent_space=64):
        super(AudioTransformerEncoderCNNReconstruction, self).__init__()
        self.name = f"AudioTransformerEncoderCNNReconstruction-LatentSpace{latent_space}-Heads{num_heads}-TrasformerLayers{transformer_layers}-DModel{length}-Dropout{dropout}"

        self.projection = nn.Linear(input_dim, d_model)

        # Positional Embeddings
        self.positional_embeddings = create_rotary_embedding(length,
                                                             d_model)  #PositionalEncoding(d_model=d_model, max_len=length + 1)
        self.query_tokens = nn.Parameter(torch.randn(length, d_model))

        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=num_heads,
                                                   dim_feedforward=2 * d_model,
                                                   dropout=dropout,
                                                   batch_first=True)

        self.to_latent1 = nn.Linear(d_model, latent_space)
        self.to_latent_gelu1 = nn.GELU()

        self.from_latent1 = nn.Linear(latent_space, d_model)
        self.from_latent1_gelu1 = nn.GELU()

        self.upscale1 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=11, padding=5, stride=2, output_padding=1)
        self.upscale2 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=5, padding=2, stride=2, output_padding=1)
        self.upscale3 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.upscale_gelu1 = nn.GELU()
        self.upscale_gelu2 = nn.GELU()
        self.upscale_gelu3 = nn.GELU()

        self.latent_to_memory1 = nn.Linear(d_model * 8, d_model * length)
        self.latent_to_memory_gelu1 = nn.GELU()

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                   nhead=num_heads,
                                                   dim_feedforward=2 * d_model,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=transformer_layers)

        # Last Linear Layer
        self.fc_out = nn.Linear(d_model, input_dim)

    def forward(self, input):
        memory = self.to_latent(input)
        memory = self.from_latent(memory)
        return memory

    def to_latent(self, x):
        memory = self.projection(x)  # Project the whole sequence

        memory = memory * self.positional_embeddings

        cls_tokens = self.cls_token.expand(memory.size(0), -1, -1)  # Ensure batch size compatibility
        memory = torch.cat([cls_tokens, memory], dim=1)  # Concatenate CLS token to the input

        memory = self.encoder(memory)  # Pass through Transformer encoder

        memory = memory[:, 0, :]  # Extract the CLS token as the global representation
        memory = memory.unsqueeze(1)  # Reshape for decoder input

        memory  = self.to_latent1(memory)
        memory  = self.to_latent_gelu1(memory)

        return memory

    def from_latent(self, x):
        queries = self.query_tokens.unsqueeze(0).repeat(x.size(0), 1, 1)

        # Learn to transform latent representation into a sequence

        memory = self.from_latent1(x)
        memory = self.from_latent1_gelu1(memory)


        memory = self.upscale1(memory)
        #memory = self.upscale_gelu1(memory)

        memory = self.upscale2(memory)
        #memory = self.upscale_gelu2(memory)

        memory = self.upscale3(memory)
        memory = self.upscale_gelu3(memory)


        memory = self.latent_to_memory1(memory).reshape(x.size(0), queries.shape[1], -1)
        memory = self.latent_to_memory_gelu1(memory)

        memory = memory * self.positional_embeddings
        memory = self.decoder(queries, memory)
        memory = self.fc_out(memory)
        #memory = self.fc_(memory)

        return memory


class AudioTransformerEncoder(nn.Module):
    def __init__(self, input_dim=64, num_heads=16, transformer_layers=8, length=256, d_model=256, dropout=0.1,
                 latent_space=64):
        super(AudioTransformerEncoder, self).__init__()
        self.name = f"AudioTransformerEncoder-LatentSpace{latent_space}-Heads{num_heads}-TrasformerLayers{transformer_layers}-DModel{length}-Dropout{dropout}"

        self.projection = nn.Linear(input_dim, d_model)
        self.projection_gelu = nn.GELU()

        # Positional Embeddings
        self.positional_embeddings = create_rotary_embedding(length,
                                                             d_model)  #PositionalEncoding(d_model=d_model, max_len=length + 1)
        self.query_tokens = nn.Parameter(torch.randn(length, d_model))

        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.W_E = nn.Parameter(torch.Tensor(d_model, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=num_heads,
                                                   dim_feedforward=2 * d_model,
                                                   dropout=dropout,
                                                   batch_first=True)

        self.to_latent1 = nn.Linear(d_model, latent_space)
        self.to_latent1_gelu = nn.GELU()

        self.from_latent1 = nn.Linear(latent_space, d_model * length)
        self.from_latent1_gelu = nn.GELU()

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                   nhead=num_heads,
                                                   dim_feedforward=2 * d_model,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=transformer_layers)

        # Last Linear Layer
        self.fc_out = nn.Linear(d_model, input_dim)
        self.fc_gelu = nn.GELU()
    def forward(self, input):
        memory = self.to_latent(input)
        memory = self.from_latent(memory)
        return memory

    def to_latent(self, x):
        memory = self.projection(x)  # Project the whole sequence
        memory = self.projection_gelu(memory)

        memory = memory * self.positional_embeddings

        cls_tokens = self.cls_token.expand(memory.size(0), -1, -1)  # Ensure batch size compatibility
        memory = torch.cat([cls_tokens, memory], dim=1)  # Concatenate CLS token to the input

        memory = self.encoder(memory)  # Pass through Transformer encoder
        memory = memory[:, 0, :]  # Extract the CLS token as the global representation

        memory = self.to_latent1(memory)
        memory = self.to_latent1_gelu(memory)

        memory = memory.unsqueeze(1)  # Reshape for decoder input

        return memory

    def from_latent(self, x):
        queries = self.query_tokens.unsqueeze(0).repeat(x.size(0), 1, 1)

        # Learn to transform latent representation into a sequence
        memory = self.from_latent1(x)
        memory = self.from_latent1_gelu(memory).reshape(x.size(0), queries.shape[1], -1)

        memory = memory * torch.matmul(self.positional_embeddings, self.W_E)
        memory = self.decoder(queries, memory)

        memory = self.fc_out(memory)
        memory = self.fc_gelu(memory)

        return memory

class GroupedTransformerBlock(nn.TransformerEncoderLayer):
    def __init__(self, d_model=256, nhead=16, dim_feedforward=512, dropout=0.1, batch_first=True, group_number=4):
        super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=batch_first)
        self.group_number = group_number

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, is_causal: bool = False,):
        x = src.reshape(src.shape[0], src.shape[1] // self.group_number, self.group_number, src.shape[2])
        x = x.reshape(x.shape[0], x.shape[1] // self.group_number, x.shape[2] * self.group_number)
        x = super().forward(x)
        x = x.reshape(x.shape[0], x.shape[1], 3, x.shape[2] // 3)
        x = x.reshape(x.shape[0], x.shape[1] * 3, x.shape[2] // 3)
        return x


class GroupedAudioTransformerEncoder(nn.Module):
    def __init__(self, input_dim=64, num_heads=16, transformer_layers=8, length=256, d_model=256, dropout=0.1,
                 latent_space=64, group=3):
        super(GroupedAudioTransformerEncoder, self).__init__()
        self.name = f"GroupedAudioTransformerEncoder-LatentSpace{latent_space}-Heads{num_heads}-TrasformerLayers{transformer_layers}-DModel{length}-Dropout{dropout}"

        self.projection = nn.Linear(input_dim, d_model)

        # Positional Embeddings
        self.positional_embeddings = create_rotary_embedding(length,
                                                             d_model)  # PositionalEncoding(d_model=d_model, max_len=length + 1)
        self.query_tokens = nn.Parameter(torch.randn(length, d_model))

        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = GroupedTransformerBlock(d_model=d_model,
                                                   nhead=num_heads,
                                                   dim_feedforward=2 * d_model,
                                                   dropout=dropout,
                                                   batch_first=True,
                                                   group_number=group)

        self.latent_to_memory = nn.Linear(d_model, d_model * length)

        self.W_E = nn.Parameter(torch.Tensor(d_model, d_model))

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Transformer Decoder
        decoder_layer = GroupedTransformerBlock(d_model=d_model,
                                                   nhead=num_heads,
                                                   dim_feedforward=2 * d_model,
                                                   dropout=dropout,
                                                   batch_first=True,
                                                   group_number=group)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=transformer_layers)

        # Last Linear Layer
        self.fc_out = nn.Linear(d_model, input_dim)

    def forward(self, input):
        memory = self.to_latent(input)
        memory = self.from_latent(memory)
        return memory

    def to_latent(self, x):
        memory = self.projection(x)  # Project the whole sequence

        memory = memory * torch.matmul(self.positional_embeddings, self.W_E)

        cls_tokens = self.cls_token.expand(memory.size(0), -1, -1)  # Ensure batch size compatibility
        memory = torch.cat([cls_tokens, memory], dim=1)  # Concatenate CLS token to the input

        memory = self.encoder(memory)  # Pass through Transformer encoder

        global_feature = memory[:, 0, :]  # Extract the CLS token as the global representation
        memory = global_feature.unsqueeze(1)  # Reshape for decoder input

        return memory

    def from_latent(self, x):
        queries = self.query_tokens.unsqueeze(0).repeat(x.size(0), 1, 1)

        # Learn to transform latent representation into a sequence
        memory = self.latent_to_memory(x).reshape(x.size(0), queries.shape[1], -1)
        memory = memory * torch.matmul(self.positional_embeddings, self.W_E)
        memory = self.decoder(queries, memory)
        memory = self.fc_out(memory)

        return memory



# self.first_convolution                = nn.Conv1d(in_channels=input_dim, out_channels=conv_features, kernel_size=11, padding=5, stride=2)
# self.fourth_convolution_out  = nn.ConvTranspose1d(in_channels=conv_features, out_channels=input_dim, kernel_size=11, padding=5, stride=2)