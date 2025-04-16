import torch.nn as nn
import torch
import ModelComponents
from torch import flatten



class AudioTransformerCNNReconstruction(nn.Module):
    def __init__(self, input_dim=64, num_heads=16, transformer_layers=8, length=256, d_model=256, dim_feedforward=512, dropout=0.1,
                 latent_space=64):
        super(AudioTransformerCNNReconstruction, self).__init__()
        self.name = f"AudioTransformerCNNReconstruction-LatentSpace{latent_space}-Heads{num_heads}-TrasformerLayers{transformer_layers}-DModel{length}-Dropout{dropout}"

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

        output_dim = length // (4 * 2 * 2 * 2)

        self.conv_shrink_1 = nn.Sequential(nn.Conv1d(in_channels=length, out_channels=32, kernel_size=11, stride=4, padding=5),
                                           nn.BatchNorm1d(num_features=32),
                                           nn.GELU(),
                                           nn.AvgPool1d(kernel_size=11, stride=2, padding=5))

        self.conv_shrink_2 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=96, kernel_size=5, stride=2, padding=2),
                                           nn.BatchNorm1d(num_features=96),
                                           nn.GELU(),
                                           nn.AvgPool1d(kernel_size=5, stride=2, padding=2))

        self.conv_shrink_3 = nn.Sequential(nn.Conv1d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                                           nn.BatchNorm1d(num_features=96),
                                           nn.GELU())

        self.conv_shrink_4 = nn.Sequential(nn.Conv1d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
                                           nn.BatchNorm1d(num_features=64),
                                           nn.GELU())

        self.conv_shrink_5 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                           nn.BatchNorm1d(num_features=64),
                                           nn.GELU(),
                                           nn.AvgPool1d(kernel_size=3, stride=2, padding=1))

        # Skip Connection
        self.latent_skip_in = nn.Linear(d_model * length, 64 * output_dim)

        self.to_latent1 = nn.Linear(64 * output_dim, latent_space)
        self.to_latent1_gelu = nn.GELU()

        # Latent Space

        self.from_latent1 = nn.Linear(latent_space, 64 * output_dim)
        self.from_latent1_gelu = nn.GELU()

        # Skip Connection
        self.latent_skip_out = nn.Linear(64 * output_dim, d_model * length)

        self.conv_grow_1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.GELU(),
            nn.BatchNorm1d(num_features=64),
            nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        )

        self.conv_grow_2 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm1d(num_features=64),
            nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        )

        self.conv_grow_3 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm1d(num_features=64),
            nn.ConvTranspose1d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1),
        )

        self.conv_grow_4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.GELU(),
            nn.BatchNorm1d(num_features=96),
            nn.ConvTranspose1d(in_channels=96, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

        self.conv_grow_5 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.GELU(),
            nn.BatchNorm1d(num_features=32),
            nn.ConvTranspose1d(in_channels=32, out_channels=length, kernel_size=11, stride=4, padding=4, output_padding=1),
        )


        # self.conv_grow_1 = nn.Sequential(
        #     nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(num_features=64),
        #     nn.GELU(),
        #     nn.Upsample(scale_factor=2))

        # self.conv_grow_2 = nn.Sequential(
        #     nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(num_features=64),
        #     nn.GELU())

        # self.conv_grow_3 = nn.Sequential(
        #     nn.ConvTranspose1d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(num_features=96),
        #     nn.GELU())

        # self.conv_grow_4 = nn.Sequential(
        #     nn.ConvTranspose1d(in_channels=96, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1),
        #     nn.BatchNorm1d(num_features=32),
        #     nn.GELU(),
        #     nn.Upsample(scale_factor=2))

        # self.conv_grow_5 = nn.Sequential(
        #     nn.ConvTranspose1d(in_channels=32, out_channels=length, kernel_size=11, stride=4, padding=4, output_padding=1),
        #     nn.BatchNorm1d(num_features=length),
        #     nn.GELU(),
        #     nn.Upsample(scale_factor=2))

        #self.conv_grow = Multi1DConvFeature(layers[::-1], upsample=True)

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

        encoded = self.encoder(memory)  # Pass through Transformer encoder

        # Convolutions
        memory = self.conv_shrink_1(encoded)
        memory = self.conv_shrink_2(memory)
        memory = self.conv_shrink_3(memory)
        memory = self.conv_shrink_4(memory)
        memory = self.conv_shrink_5(memory)

        # Reshape for decoder input (Concatenation)
        memory = memory.reshape(memory.size(0), -1)

        # Skip
        # Reshape for decoder input (Concatenation)
        encoded = encoded.reshape(encoded.size(0), -1)
        skip = self.latent_skip_in(encoded)

        memory += skip

        # To latent space
        memory = self.to_latent1(memory)
        memory = self.to_latent1_gelu(memory)
        memory = memory.unsqueeze(1)  # Reshape for decoder input
        return memory

    def from_latent(self, x):
        # Learn to transform latent representation into a sequence
        memory = self.from_latent1(x)
        projected = self.from_latent1_gelu(memory)

        memory = memory.reshape(x.size(0), 64, -1)

        # Convolutions
        memory = self.conv_grow_1(memory)
        memory = self.conv_grow_2(memory)
        memory = self.conv_grow_3(memory)
        memory = self.conv_grow_4(memory)
        memory = self.conv_grow_5(memory)

        # Reshape from concatenated representation
        queries = self.query_tokens.unsqueeze(0).repeat(x.size(0), 1, 1)

        # Skip
        skip = self.latent_skip_out(projected)
        skip = skip.reshape(x.size(0), queries.shape[1], -1)
        memory += skip

        # Decoder Block
        memory = self.decoder(queries, memory)

        memory = self.fc_out(memory)
        memory = self.fc_gelu(memory)

        return memory