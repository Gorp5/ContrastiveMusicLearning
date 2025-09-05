import torch
import torch.nn.functional as F
from torch import nn

from models import RopeALiBiModelComponents
from models.RopeALiBiModelComponents import RoPEALiBiMultiheadAttention


class AudioConformer(nn.Module):
    def __init__(self, input_dim=64, num_heads=16, encoder_layers=16, decoder_layers=8, length=256, d_model=256, dim_feedforward=512, kernel_size=5, dropout=0.1,
                 latent_space=64, name_extension="", autoregressive=False, use_alibi=False):
        super(AudioConformer, self).__init__()
        self.name = f"AudioConformer-LatentSpace{latent_space}-Heads{num_heads}-TrasformerLayers{encoder_layers}-DModel{length}-Dropout{dropout}{name_extension}"

        self.projection = nn.Linear(input_dim, d_model)
        self.projection_gelu = nn.GELU()

        self.query_tokens = nn.Parameter(torch.randn(length, d_model))

        # CLS Token
        #self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.encoder = ALiBiConformerEncoder(num_layers=encoder_layers,
                                                                    d_model=d_model,
                                                                    num_heads=num_heads,
                                                                    dim_feedforward=dim_feedforward,
                                                                    seq_len=length,
                                                                    dropout=dropout,
                                                                    use_alibi=use_alibi,
                                                                    device='cuda')

        self.encode_to_latent = nn.Linear(d_model * length, latent_space)
        self.encode_to_latent_gelu = nn.GELU()

        # Latent Space

        self.encode_from_latent = nn.Linear(latent_space, d_model * length)
        self.encode_from_latent_gelu = nn.GELU()


        # Transformer Decoder
        self.decoder = ALiBiConformerDecoder(num_layers=decoder_layers,
                                                                    d_model=d_model,
                                                                    num_heads=num_heads,
                                                                    dim_feedforward=dim_feedforward,
                                                                    seq_len=length,
                                                                    dropout=dropout,
                                                                    use_alibi=use_alibi,
                                                                    device='cuda')

        # Last Linear Layer
        self.fc_out = nn.Linear(d_model, input_dim)
        self.fc_gelu = nn.GELU()

        self.autoregressive = autoregressive

    def forward(self, x, mask=None):
        x = x.permute(0, 2, 1)
        latent = self.to_latent(x, mask)

        if self.autoregressive:
            memory = self.from_latent_Autoregressive(latent, mask=mask, tgt=x[:, :-1])
        else:
            memory = self.from_latent(latent, mask)

        memory = memory.permute(0, 2, 1)
        return memory, latent

    def to_latent(self, x, mask):

        # Project to Model Dimension
        memory = self.projection(x)
        memory = self.projection_gelu(memory)

        # Encoder Block
        memory = self.encoder(memory, mask)  # Pass through Transformer encoder

        # Reshape for decoder input (Concatenation)
        memory = memory.reshape(memory.size(0), -1)

        # Project to Latent Space
        memory = self.encode_to_latent(memory)
        memory = self.encode_to_latent_gelu(memory)

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

class ALiBiConformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=16, dim_feedforward=256, dropout=0.1):
        super().__init__()

        self.self_attn = RoPEALiBiMultiheadAttention(d_model, num_heads, RopeALiBiModelComponents.get_alibi_slopes(num_heads), dropout=dropout)

        in_channels = 1
        expansion_factor = 1
        kernel_size = 5

        # [B, F, T]
        self.convolution = DepthwiseConvolutionBlock(d_model=d_model, kernel_size=3, stride=1)

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
        src2 = self.self_attn(src, src, src, alibi_bias, pos_emb, mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src = self.convolution(src)

        # Feed-forward Block
        src2 = self.linear1(src)
        src2 = self.activation(src2)
        src2 = self.linear2(src2)

        # Regularization
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class ALiBiConformerEncoder(nn.Module):
    def __init__(self, num_layers=8, d_model=256, num_heads=16, dim_feedforward=256, seq_len=256, dropout=0.1,
                 checkpointing=False, use_rope=True, use_alibi=True, device='cpu'):
        super().__init__()
        self.layers = nn.ModuleList([
            ALiBiConformerEncoderLayer(d_model=d_model, num_heads=num_heads, dim_feedforward=dim_feedforward,
                                             dropout=dropout)
            for _ in range(num_layers)
        ])

        self.checkpointing = checkpointing

        if use_alibi:
                self.register_buffer("alibi_bias", RopeALiBiModelComponents.build_causal_alibi_bias(num_heads, seq_len, seq_len, device).to('cuda', dtype=torch.float32))
        else:
            self.alibi_bias = None

        if use_rope:
            self.register_buffer("pos_emb", RopeALiBiModelComponents.get_RoPE_matrix(seq_len, d_model // num_heads, device).to('cuda', dtype=torch.float32))
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


class ALiBiConformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=16, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.self_attn = RoPEALiBiMultiheadAttention(d_model, num_heads, RopeALiBiModelComponents.get_alibi_slopes(num_heads), dropout=dropout)
        self.cross_attn = RoPEALiBiMultiheadAttention(d_model, num_heads, RopeALiBiModelComponents.get_alibi_slopes(num_heads), dropout=dropout)

        # [B, F, T]
        self.convolution = DepthwiseConvolutionTransposeBlock(d_model=d_model, kernel_size=3, stride=1)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, tgt, memory, self_alibi_bias, self_pos_emb, cross_alibi_bias, cross_pos_emb, mask):
        t = tgt.size(1)
        L = memory.size(1)

        # slice biases and position embeddings
        bias_self = self_alibi_bias[:, :t, :t] if self_alibi_bias is not None else None
        pos_self = self_pos_emb[:t] if self_pos_emb is not None else None
        bias_cross = cross_alibi_bias[:, :t, :L] if cross_alibi_bias is not None else None
        pos_cross = cross_pos_emb[:L] if cross_pos_emb is not None else None

        # Self Attention Block
        tgt2 = self.self_attn(tgt, tgt, tgt, bias_self, pos_self, mask)
        tgt2 = self.dropout1(tgt2)
        tgt = tgt + self.norm1(tgt2)

        # Cross-attention Block
        tgt2 = self.cross_attn(tgt, memory, memory, bias_cross, pos_cross, mask)
        tgt2 = self.dropout2(tgt2)
        tgt = tgt + self.norm2(tgt2)

        # Convolution Block
        # tgt = tgt + self.convolution(tgt)

        # Feed-forward Block
        tgt2 = self.linear1(tgt)
        tgt2 = self.activation(tgt2)
        tgt2 = self.linear2(tgt2)

        # Regularization
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class ALiBiConformerDecoder(nn.Module):
    def __init__(self, num_layers=8, d_model=256, num_heads=16, dim_feedforward=256, seq_len=256, dropout=0.1, checkpointing=False, use_rope=True, use_alibi=True, device='cpu'):
        super().__init__()
        self.layers = nn.ModuleList([
            ALiBiConformerDecoderLayer(d_model=d_model, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.checkpointing = checkpointing

        if use_alibi:
            self.register_buffer("self_alibi_bias", RopeALiBiModelComponents.build_causal_alibi_bias(num_heads, seq_len, seq_len, device))
            self.register_buffer("cross_alibi_bias", RopeALiBiModelComponents.build_causal_alibi_bias(num_heads, seq_len, seq_len, device))
        else:
            self.self_alibi_bias = None
            self.cross_alibi_bias = None

        if use_rope:
            self.register_buffer("self_pos_emb", RopeALiBiModelComponents.get_RoPE_matrix(seq_len, d_model // num_heads, device))
            self.register_buffer("cross_pos_emb", RopeALiBiModelComponents.get_RoPE_matrix(seq_len, d_model // num_heads, device))
        else:
            self.self_pos_emb = None
            self.cross_pos_emb = None


        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, mask=None):
        for layer in self.layers:
            # if self.checkpointing:
            #     tgt = checkpoint(layer, tgt, memory, self.self_alibi_bias, self.self_pos_emb, self.cross_alibi_bias, self.cross_pos_emb, mask)
            # else:
            tgt = layer(tgt, memory, self.self_alibi_bias, self.self_pos_emb, self.cross_alibi_bias, self.cross_pos_emb, mask)

        return self.norm(tgt)


class DepthwiseConvolutionBlock(nn.Module):
    def __init__(self, d_model=256, kernel_size=3, stride=1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        # Pointwise convolution + GLU
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)

        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=stride,
            groups=d_model
        )

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(d_model)

        # Activation (Swish)
        self.activation = F.hardswish

        # Pointwise convolution
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

        # Dropout
        self.dropout = nn.Dropout(0.1)

        # Gelu
        self.glu = nn.GELU()

        # Downsample for residual (If stride > 1)
        if stride > 1:
            self.residual_downsample = nn.AvgPool1d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        residual = x

        # Depthwise Seperable Convolution
        x = self.layer_norm(x)
        x = x.transpose(1, 2)

        x = self.pointwise_conv1(x)
        x = self.glu(x)

        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)

        # Residual
        # residual = residual.transpose(1, 2)
        # residual = self.residual_downsample(residual)
        # residual = residual.transpose(1, 2)
        #
        # x = x + residual

        return x

class DepthwiseConvolutionTransposeBlock(nn.Module):
    def __init__(self, d_model=256, kernel_size=3, stride=1):
        super().__init__()

        self.residual_downsample = nn.Upsample(1)

        # Dropout
        self.dropout = nn.Dropout(0.1)

        # Pointwise convolution
        self.pointwise_conv2 = nn.ConvTranspose1d(d_model, d_model, kernel_size=1)

        # Activation (Swish)
        self.activation = F.hardswish

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(d_model)

        # Depthwise convolution
        self.depthwise_conv = nn.ConvTranspose1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=stride,
            groups=d_model
        )

        # Pointwise convolution + GLU
        self.pointwise_conv1 = nn.ConvTranspose1d(d_model, d_model, kernel_size=1)

        self.glu = nn.GELU()

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)

        x = self.pointwise_conv1(x)
        x = self.glu(x)

        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)

        # residual = residual.transpose(1, 2)
        # residual = self.residual_downsample(residual)
        # residual = residual.transpose(1, 2)

        # x = x + residual

        return x