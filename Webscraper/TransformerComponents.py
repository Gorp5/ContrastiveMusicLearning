from torch import nn


class Multi2DConvFeature(nn.Module):
    def __init__(self, features, conv_layers=2, kernel_size=(1, 3), stride=(1, 2),
                 padding=(0, 1), dropout_rate=0.1, upsample=False):
        super().__init__()
        layers = []

        # Build a sequential block of conv_layers convolutional blocks.
        # Each block uses a 2D conv layer with kernel size (1,3), stride (1,2), and padding (0,1).
        # This configuration reduces the sequence (width) dimension roughly by half per block.
        for _ in range(conv_layers):
            if upsample:
                layers.append(
                    nn.ConvTranspose2d(
                        in_channels=layer_features,
                        out_channels=int(layer_features * 2),
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding
                    )
                )
                layer_features = layer_features * 2
            else:
                layers.append(
                    nn.Conv2d(
                        in_channels=layer_features,
                        out_channels=int(layer_features / 2),
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding
                    )
                )
                layer_features = layer_features // 2
            layers.append(nn.ReLU())
            layers.append(nn.Dropout2d(dropout_rate))
        self.conv_net = nn.Sequential(*layers)
        # After several conv layers, the width dimension may not be exactly 1.
        # We use AdaptiveAvgPool2d to collapse the width dimension to 1.
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, features, sequence_length).
        Returns:
            Tensor of shape (batch, features, 1) after reducing the sequence dimension.
        """
        # Unsqueeze to add a "height" dimension: shape -> (batch, features, 1, sequence_length)
        x = x.unsqueeze(2)
        # Pass through the stacked convolutional blocks.
        x = self.conv_net(x)  # Now shape is (batch, features, 1, W_out) where W_out < sequence_length.
        # Use adaptive average pooling to collapse width dimension to 1.
        # x = self.pool(x)      # Shape becomes (batch, features, 1, 1)
        x = x.squeeze(3)  # Final shape: (batch, features, 1)
        return x


class TemporalConv1dBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=8, kernel_size=3, stride=4, padding=1, upscale=False):
        super().__init__()
        if not upscale:
            self.conv = nn.Conv1d(
                in_channels=in_channels,  # per feature
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        else:
            self.conv = nn.ConvTranspose1d(
                in_channels=in_channels,  # per feature
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        self.num_filters = out_channels
        self.stride = stride

    def forward(self, x):
        B, T, F = x.shape

        memory = x.permute(0, 2, 1)  # [B, F, T]
        memory = memory.reshape(-1, 1, memory.shape[-1])  # [B*F, 1, T]
        memory = self.conv(memory)  # [B*F, num_filters, T // stride]
        memory = memory.view(B, F, self.num_filters, T // self.stride).permute(0, 3, 1, 2).reshape(B, T // self.stride, F * self.num_filters)

        return memory


#
# class GroupedTransformerEncoderBlock(nn.TransformerEncoderLayer):
#     def __init__(self, d_model=256, nhead=16, dim_feedforward=512, dropout=0.1, batch_first=True, group_number=4):
#         super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
#                          batch_first=batch_first)
#         self.group_number = group_number
#
#     def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
#                 is_causal: bool = False, ):
#         x = src.reshape(src.shape[0], src.shape[1] // self.group_number, self.group_number, src.shape[2])
#         x = x.reshape(src.shape[0], src.shape[1] // self.group_number, src.shape[2] * self.group_number)
#         mem = super().forward(x)
#         x = x.reshape(mem.shape[0], mem.shape[1], self.group_number, mem.shape[2] // self.group_number)
#         x = x.reshape(mem.shape[0], mem.shape[1] * self.group_number, mem.shape[2] // self.group_number)
#         return x
#
# class GroupedTransformerDecoderBlock(nn.TransformerDecoderLayer):
#     def __init__(self, d_model=256, nhead=16, dim_feedforward=512, dropout=0.1, batch_first=True, group_number=4):
#         super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
#                          batch_first=batch_first)
#         self.group_number = group_number
#
#     def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
#                 memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
#                 memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: bool = False,
#                 memory_is_causal: bool = False, ):
#         tgt_x = memory.reshape(tgt.shape[0], tgt.shape[1] // self.group_number, self.group_number, tgt.shape[2])
#         tgt_x = tgt_x.reshape(tgt.shape[0], tgt.shape[1] // self.group_number, tgt.shape[2] * self.group_number)
#
#         x = memory.reshape(memory.shape[0], memory.shape[1] // self.group_number, self.group_number,
#                            memory.shape[2])
#         x = x.reshape(memory.shape[0], memory.shape[1] // self.group_number, memory.shape[2] * self.group_number)
#
#         mem = super().forward(tgt_x, x)
#
#         x = mem.reshape(mem.shape[0], mem.shape[1], self.group_number, mem.shape[2] // self.group_number)
#         x = x.reshape(mem.shape[0], mem.shape[1] * self.group_number, mem.shape[2] // self.group_number)
#         print(x.shape)
#         return x