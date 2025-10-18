import torch.nn as nn
import torch
import ModelComponents

class AudioTransformerSingleLinearEncoderOnly(nn.Module):
    def __init__(self, input_dim=64, num_heads=16, transformer_layers=8, length=256, d_model=256, dim_feedforward=512, dropout=0.1,
                 latent_space=64, name_extension=""):
        super(AudioTransformerSingleLinearEncoderOnly, self).__init__()
        self.name = f"AudioTransformerSingleLinearEncoderOnly-LatentSpace{latent_space}-Heads{num_heads}-TrasformerLayers{transformer_layers}-DModel{length}-Dropout{dropout}{name_extension}"

        self.projection = nn.Linear(input_dim, d_model)
        self.projection_gelu = nn.GELU()

        self.query_tokens = nn.Parameter(torch.randn(length, d_model))

        self.d_model = d_model
        self.encoder = ModelComponents.CustomTransformerEncoder(num_layers=transformer_layers,
                                                                d_model=d_model,
                                                                num_heads=num_heads,
                                                                dim_feedforward=dim_feedforward,
                                                                seq_len=256,
                                                                dropout=dropout,
                                                                device='cuda')

        self.encode_to_latent = nn.Linear(d_model * length, latent_space)
        self.encode_to_latent_gelu = nn.GELU()

        # Latent Space

        self.encode_from_latent = nn.Linear(latent_space, d_model * length)
        self.encode_from_latent_gelu = nn.GELU()

        self.decoder = ModelComponents.CustomTransformerEncoder(num_layers=transformer_layers,
                                                                d_model=d_model,
                                                                num_heads=num_heads,
                                                                dim_feedforward=dim_feedforward,
                                                                seq_len=256,
                                                                dropout=dropout,
                                                                device='cuda')

        # Last Linear Layer
        self.fc_out = nn.Linear(d_model, input_dim)
        self.fc_gelu = nn.GELU()

    def forward(self, x):
        memory = self.to_latent(x)
        memory = self.from_latent(memory)
        return memory

    def to_latent(self, x):
        # Project to Model Dimension
        memory = self.projection(x)
        memory = self.projection_gelu(memory)

        # Encoder Block
        memory = self.encoder(memory)  # Pass through Transformer encoder

        # Reshape for decoder input (Concatenation)
        memory = memory.reshape(memory.size(0), -1)

        # Project to Latent Space
        memory = self.encode_to_latent(memory)
        memory = self.encode_to_latent_gelu(memory)

        return memory

    def from_latent(self, x):
        # Project from Latent Space
        memory = self.encode_from_latent(x)
        memory = self.encode_from_latent_gelu(memory)

        # Reshape from concatenated representation and get Query Tokens
        memory = memory.reshape(x.size(0), self.d_model, -1)

        # Decoder Block
        memory = self.decoder(memory)

        # Project to Output Dimension
        memory = self.fc_out(memory)
        memory = self.fc_gelu(memory)

        return memory