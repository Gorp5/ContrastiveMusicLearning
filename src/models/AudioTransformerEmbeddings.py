import torch.nn as nn
import torch
from models import RopeALiBiModelComponents


class AudioTransformerEmbeddings(nn.Module):
    def __init__(self, input_dim=64, num_heads=16, encoder_layers=16, decoder_layers=8, length=256, d_model=256, dim_feedforward=512, checkpointing=False, dropout=0.1,
                 latent_space=64, name_extension="", use_alibi=True, use_rope=False, autoregressive=False, num_classes=50, genre_count=0, mood_count=0, patching = None, custom_slope=-1):
        super(AudioTransformerEmbeddings, self).__init__()
        self.name = f"AudioTransformer-LatentSpace{latent_space}-Heads{num_heads}-EncoderLayers{encoder_layers}-DecoderLayers{decoder_layers}-DModel{length}-Dropout{dropout}-AutoRegressive{autoregressive}{name_extension}"

        self.autoregressive = autoregressive
        self.model_dim = d_model
        self.projection = nn.Linear(input_dim, d_model)
        self.projection_gelu = nn.GELU()

        self.query_tokens = nn.Parameter(torch.randn(length, d_model))
        self.bos_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.teacher_forcing_projection = nn.Linear(input_dim, d_model)

        self.length = length

        self.encoder = RopeALiBiModelComponents.CustomTransformerEncoder(num_layers=encoder_layers,
                                                                         d_model=d_model,
                                                                         num_heads=num_heads,
                                                                         dim_feedforward=dim_feedforward,
                                                                         seq_len=length,
                                                                         dropout=dropout,
                                                                         checkpointing=checkpointing,
                                                                         use_alibi=use_alibi,
                                                                         use_rope=use_rope,
                                                                         custom_slopes=custom_slope,
                                                                         device='cuda')


        if patching is None:
            self.encode_to_latent = nn.Linear(d_model * length, latent_space)
            self.encode_to_latent_gelu = nn.GELU()
            self.encode_from_latent = nn.Linear(latent_space, d_model * length)
            self.encode_from_latent_gelu = nn.GELU()
        elif patching == "AvgPoolFeature":
            self.projection = TransformPoolFeaturesLinear(input_dim, d_model, pool_size=2)
        elif patching == "ConvStack":
            self.projection = TransformConvStack(input_dim, d_model, num_layers=3, kernel_size=(3, 3), hidden=[32, 64])

        # Latent Space
        self.classification_head = nn.Linear(d_model * length, num_classes)


    def forward(self, x, mask=None):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        latent = self.to_latent(x, mask)

        return latent

    def to_latent(self, x, mask):
        # Project to Model Dimension
        memory = self.projection(x)
        #memory = self.projection_gelu(memory)

        # Encoder Block
        memory = self.encoder(memory, mask)  # Pass through Transformer encoder

        # Reshape for decoder input (Concatenation)
        memory = memory.reshape(memory.size(0), -1)

        # Project to Latent Space
        #memory = self.encode_to_latent(memory)
        #memory = self.encode_to_latent_gelu(memory)

        memory = self.classification_head(memory)

        return memory


    def from_latent(self, x, mask):

        # Project from Latent Space
        memory = self.encode_from_latent(x)
        memory = self.encode_from_latent_gelu(memory)

        # Reshape from concatenated representation and get Query Tokens
        queries = self.query_tokens.unsqueeze(0).repeat(x.size(0), 1, 1)
        memory = memory.reshape(x.size(0), queries.shape[1], -1)

        # Decoder Block
        memory = self.decoder(queries, memory, mask)

        # Project to Output Dimension
        memory = self.fc_out(memory)
        memory = self.fc_gelu(memory)

        return memory


    def from_latent_Autoregressive(self, x, tgt=None, mask=None):
        # Project from Latent Space
        memory = self.encode_from_latent(x)
        memory = self.encode_from_latent_gelu(memory)

        # Reshape from concatenated representation and get Query Tokens
        if tgt is None:
            tgt = self.query_tokens.unsqueeze(0).repeat(x.size(0), 1, 1)
        else:
            bos = self.bos_token.expand(x.size(0), 1, self.model_dim)  # [B, 1, D]

            tgt = self.teacher_forcing_projection(tgt)
            # Projected Ground Truth
            tgt = torch.cat([bos, tgt[:, :-1]], dim=1)

        memory = memory.reshape(x.size(0), self.model_dim, -1)

        # Decoder Block
        memory = self.decoder(tgt, memory, mask)

        # Project to Output Dimension
        memory = self.fc_out(memory)
        memory = self.fc_gelu(memory)

        return memory


    def set_checkpointing(self, checkpointing):
        self.encoder.set_checkpointing(checkpointing)
        self.decoder.set_checkpointing(checkpointing)


# -----------------------------
# 1. Conv2D vertical patch → FC
# -----------------------------
class TransformConvLinear(nn.Module):
    def __init__(self, F, D, W=3, S=2, X=8, kernel_width=7):
        super().__init__()
        # Input [B, T, F] → [B, 1, T, F]
        self.conv = nn.Conv2d(1, X, kernel_size=(W, kernel_width), stride=(S, kernel_width // 2), padding=(S // 2, kernel_width // 2))
        # Output: [B, X, T', 1] → [B, T', X]
        self.fc = nn.Linear(X, D)

    def forward(self, x):
        B, T, F = x.shape
        x = x.unsqueeze(1)               # [B, 1, T, F]
        x = self.conv(x)                 # [B, X, T', 1]
        x = x.squeeze(-1).transpose(1, 2) # [B, T', X]
        x = self.fc(x)                   # [B, T', D]
        return x


class InverseTransformConvLinear(nn.Module):
    def __init__(self, F, D, W=3, S=2, X=8):
        super().__init__()
        self.fc = nn.Linear(D, X)
        self.deconv = nn.ConvTranspose2d(X, 1, kernel_size=(W, F), stride=(S, 1))

    def forward(self, x):
        x = self.fc(x)                   # [B, T', X]
        x = x.transpose(1, 2).unsqueeze(-1) # [B, X, T', 1]
        x = self.deconv(x)               # [B, 1, T, F]
        return x.squeeze(1)              # [B, T, F]


# -----------------------------
# 2. Conv stack across features
# -----------------------------
class TransformConvStack(nn.Module):
    def __init__(self, F, D, num_layers=3, kernel_size=(3, 3), hidden=None):
        super().__init__()
        if hidden is None:
            hidden = [64, 96]
        layers = []
        in_ch = 1
        for i in range(num_layers - 1):
            layers.append(nn.Conv2d(in_ch, hidden[i], kernel_size, padding=(1, 1)))
            layers.append(nn.ReLU())
            in_ch = hidden[i]
        layers.append(nn.Conv2d(in_ch, hidden[-1], kernel_size=(3, F), padding=(1, 0)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, T, F]
        x = self.net(x)     # [B, D, T’, 1]
        x = x.squeeze(3).permute(0, 2, 1)  # [B, T’, D]

        return


class InverseTransformConvStack(nn.Module):
    def __init__(self, F, D, num_layers=3, kernel_size=(3, 3), hidden=None):
        super().__init__()
        layers = []
        in_ch = D
        layers.append(nn.ConvTranspose2d(in_ch, D, kernel_size=(3, F), padding=(1, 0)))
        layers.append(nn.ReLU())
        for i in range(num_layers - 1):
            layers.append(nn.ConvTranspose2d(hidden, hidden[i], kernel_size, padding=(1, 1)))
            layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose2d(hidden, 1, kernel_size=(3, 3), padding=(1, 1)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2).unsqueeze(-1)  # [B, D, T’, 1]
        x = self.net(x)                      # [B, 1, T, F]
        return x.squeeze(1)                  # [B, T, F]


# -----------------------------
# 3. AvgPool across features → FC
# -----------------------------
class TransformPoolFeaturesLinear(nn.Module):
    def __init__(self, F, D, pool_size=2):
        super().__init__()
        self.pool = nn.AvgPool1d(pool_size)
        self.fc = nn.Linear(F // pool_size, D)

    def forward(self, x):
        x = self.pool(x)                 # [B, F’, T]
        x = self.fc(x)                   # [B, T, D]
        return x


class InverseTransformPoolFeaturesLinear(nn.Module):
    def __init__(self, F, D, pool_size=2):
        super().__init__()
        self.fc = nn.Linear(D, F // pool_size)
        self.upsample = nn.Upsample(scale_factor=pool_size, mode="nearest")

    def forward(self, x):
        x = self.fc(x)                   # [B, T, F’]
        x = x.transpose(1, 2)            # [B, F’, T]
        x = self.upsample(x)             # [B, F, T]
        return x.transpose(1, 2)         # [B, T, F]


# -----------------------------
# 4. AvgPool across time → FC
# -----------------------------
class TransformPoolTimeLinear(nn.Module):
    def __init__(self, F, D, pool_size=2):
        super().__init__()
        self.pool = nn.AvgPool1d(pool_size)
        self.fc = nn.Linear(F, D)

    def forward(self, x):
        x = self.pool(x.transpose(1, 2)).transpose(1, 2)  # [B, T’, F]
        return self.fc(x)                                 # [B, T’, D]


class InverseTransformPoolTimeLinear(nn.Module):
    def __init__(self, F, D, pool_size=2):
        super().__init__()
        self.fc = nn.Linear(D, F)
        self.upsample = nn.Upsample(scale_factor=pool_size, mode="nearest")

    def forward(self, x):
        x = self.fc(x)                     # [B, T’, F]
        x = self.upsample(x.transpose(1, 2)).transpose(1, 2)  # [B, T, F]
        return x


# -----------------------------
# 5. Small Conv1D across time
# -----------------------------
class Transform1DConvTimeLinear(nn.Module):
    def __init__(self, F, D, out_channels=64, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(F, out_channels, kernel_size, padding=kernel_size // 2)
        self.fc = nn.Linear(out_channels, D)

    def forward(self, x):
        x = x.transpose(1, 2)             # [B, F, T]
        x = self.conv(x).transpose(1, 2)  # [B, T, F’]
        return self.fc(x)                 # [B, T, D]


class InverseTransform1DConvTimeLinear(nn.Module):
    def __init__(self, F, D, out_channels=64, kernel_size=3):
        super().__init__()
        self.fc = nn.Linear(D, out_channels)
        self.deconv = nn.ConvTranspose1d(out_channels, F, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        x = self.fc(x).transpose(1, 2)    # [B, F’, T]
        x = self.deconv(x).transpose(1, 2) # [B, T, F]
        return x


# -----------------------------
# 6. Small Conv1D across time (per feature)
# -----------------------------
class Transform1DConvTimeLinear(nn.Module):
    def __init__(self, F, D, hidden=64, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(F, F, kernel_size, padding=kernel_size // 2, groups=F)  # depthwise
        self.fc = nn.Linear(F, D)

    def forward(self, x):
        x = x.transpose(1, 2)             # [B, F, T]
        x = self.conv(x).transpose(1, 2)  # [B, T, F]
        return self.fc(x)                 # [B, T, D]


class InverseTransform6(nn.Module):
    def __init__(self, F, D, hidden=64, kernel_size=3):
        super().__init__()
        self.fc = nn.Linear(D, F)
        self.deconv = nn.ConvTranspose1d(F, F, kernel_size, padding=kernel_size // 2, groups=F)

    def forward(self, x):
        x = self.fc(x).transpose(1, 2)    # [B, F, T]
        x = self.deconv(x).transpose(1, 2) # [B, T, F]
        return x