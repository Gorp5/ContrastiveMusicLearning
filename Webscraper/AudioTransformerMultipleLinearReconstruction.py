import torch.nn as nn
import torch
import RoPEALiBiComponents
import ModelComponents


class AudioTransformerMultipleLinearReconstruction(nn.Module):
    def __init__(self, input_dim=64, num_heads=16, transformer_layers=8, length=256, d_model=256, dim_feedforward=512, dropout=0.1,
                 latent_space=64):
        super(AudioTransformerMultipleLinearReconstruction, self).__init__()
        self.name = f"AudioTransformerMultipleLinearReconstruction-LatentSpace{latent_space}-Heads{num_heads}-TrasformerLayers{transformer_layers}-DModel{length}-Dropout{dropout}"

        self.projection = nn.Linear(input_dim, d_model)
        self.projection_gelu = nn.GELU()

        self.query_tokens = nn.Parameter(torch.randn(length, d_model))

        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.encoder = ModelComponents.RoPEALiBiTransformerEncoder(num_layers=transformer_layers,
                                                                    d_model=d_model,
                                                                    num_heads=num_heads,
                                                                    dim_feedforward=dim_feedforward,
                                                                    seq_len=256,
                                                                    dropout=dropout,
                                                                    device='cuda')


        self.to_latent1 = nn.Linear(d_model * length, d_model)
        self.to_latent1_gelu = nn.GELU()

        self.to_latent2 = nn.Linear(d_model, d_model)
        self.to_latent2_gelu = nn.GELU()

        self.to_latent3 = nn.Linear(d_model, d_model // 2)
        self.to_latent3_gelu = nn.GELU()

        self.to_latent4 = nn.Linear(d_model // 2, d_model // 2)
        self.to_latent4_gelu = nn.GELU()

        self.to_latent5 = nn.Linear(d_model // 2, latent_space)
        self.to_latent5_gelu = nn.GELU()

        # Latent Space

        self.from_latent1 = nn.Linear(latent_space, d_model // 2)
        self.from_latent1_gelu = nn.GELU()

        self.from_latent2 = nn.Linear(d_model // 2, d_model // 2)
        self.from_latent2_gelu = nn.GELU()

        self.from_latent3 = nn.Linear(d_model // 2, d_model)
        self.from_latent3_gelu = nn.GELU()

        self.from_latent4 = nn.Linear(d_model, d_model)
        self.from_latent4_gelu = nn.GELU()

        self.from_latent5 = nn.Linear(d_model, d_model * length)
        self.from_latent5_gelu = nn.GELU()

        # Transformer Decoder
        self.decoder = ModelComponents.RoPEALiBiTransformerDecoder(num_layers=transformer_layers,
                                                                    d_model=d_model,
                                                                    num_heads=num_heads,
                                                                    dim_feedforward=dim_feedforward,
                                                                    seq_len=256,
                                                                    dropout=dropout,
                                                                    device='cuda')

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

        memory = self.encoder(memory)  # Pass through Transformer encoder

        # Reshape for decoder input (Concatenation)
        memory = memory.reshape(memory.size(0), -1)

        memory = self.to_latent1(memory)
        memory = self.to_latent1_gelu(memory)

        memory = self.to_latent2(memory)
        memory = self.to_latent2_gelu(memory)

        memory = self.to_latent3(memory)
        memory = self.to_latent3_gelu(memory)

        memory = self.to_latent4(memory)
        memory = self.to_latent4_gelu(memory)

        memory = self.to_latent5(memory)
        memory = self.to_latent5_gelu(memory)

        memory = memory.unsqueeze(1)  # Reshape for decoder input
        return memory

    def from_latent(self, x):
        # Learn to transform latent representation into a sequence
        memory = self.from_latent1(x)
        memory = self.from_latent1_gelu(memory)

        memory = self.from_latent2(memory)
        memory = self.from_latent2_gelu(memory)

        memory = self.from_latent3(memory)
        memory = self.from_latent3_gelu(memory)

        memory = self.from_latent4(memory)
        memory = self.from_latent4_gelu(memory)

        memory = self.from_latent5(memory)
        memory = self.from_latent5_gelu(memory)

        # Reshape from concatenated representation
        queries = self.query_tokens.unsqueeze(0).repeat(x.size(0), 1, 1)
        memory = memory.reshape(x.size(0), queries.shape[1], -1)

        # Decoder Block
        memory = self.decoder(queries, memory)

        memory = self.fc_out(memory)
        memory = self.fc_gelu(memory)

        return memory