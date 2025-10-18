import torch.nn as nn
import torch
from models import RopeALiBiModelComponents


class AudioTransformerClassifier(nn.Module):
    def __init__(self,input_dim=128, num_heads=16, encoder_layers=16, length=256, d_model=256, dim_feedforward=512, checkpointing=False, dropout=0.1,
                 latent_space=64, name_extension="", use_alibi=True, use_rope=False, autoregressive=False, num_classes=50, custom_slopes=True):
        super(AudioTransformerClassifier, self).__init__()
        self.name = f"AudioTransformerClassifier-LatentSpace{latent_space}-Heads{num_heads}-EncoderLayers{encoder_layers}-DModel{length}-Dropout{dropout}-AutoRegressive{autoregressive}{name_extension}"

        self.autoregressive = autoregressive
        self.model_dim = d_model
        self.projection = nn.Linear(input_dim, d_model)
        self.projection_gelu = nn.GELU()

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
                                                                         custom_slopes=custom_slopes,
                                                                         device='cuda')

        self.classification_head = nn.Linear(d_model * length, num_classes)

    def forward(self, x, mask=None):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)

        # Project to Model Dimension
        memory = self.projection(x)
        memory = self.projection_gelu(memory)

        # Encoder Block
        memory = self.encoder(memory, mask)  # Pass through Transformer encoder

        # Reshape for decoder input (Concatenation)
        memory = memory.reshape(memory.size(0), -1)

        # Project to Latent Space
        memory = self.classification_head(memory)

        return memory
