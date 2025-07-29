import torch.nn as nn

class ShortChunkCNN(nn.Module):
    '''
    Short-chunk CNN architecture.
    So-called vgg-ish model with a small receptive field.
    Deeper layers, smaller pooling (2x2).
    '''
    def __init__(self,
                n_channels=64,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=128,
                n_class=50):
        super(ShortChunkCNN, self).__init__()

        # self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Conv_2d(1,  n_channels,       shape=(3, 3),   stride=(1, 1),      pool=(2, 2))
        self.layer2 = Conv_2d(n_channels,     n_channels,       shape=(3, 3),   stride=(1, 1),      pool=(2, 2))
        self.layer3 = Conv_2d(n_channels,     n_channels * 2,   shape=(3, 3),   stride=(1, 1),      pool=(2, 2))
        self.layer4 = Conv_2d(n_channels * 2, n_channels * 2,   shape=(3, 3),   stride=(1, 1),      pool=(2, 2))
        self.layer5 = Conv_2d(n_channels * 2, n_channels * 2,   shape=(3, 3),   stride=(1, 1),      pool=(2, 2))
        self.layer6 = Conv_2d(n_channels * 2, n_channels * 2,   shape=(3, 3),   stride=(1, 1),      pool=(2, 2))
        self.layer7 = Conv_2d(n_channels * 2, n_channels * 4,   shape=(3, 3),   stride=(1, 1),      pool=(2, 2))


        # Dense
        self.dense1 = nn.Linear(n_channels * 4, n_channels * 4)
        self.bn = nn.BatchNorm1d(n_channels * 4)
        self.dense2 = nn.Linear(n_channels * 4, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Spectrogram
        #x = self.spec(x)
        #x = self.to_db(x)
        #x = x.unsqueeze(1)
        #x = self.spec_bn(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(-2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(-1)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        #x = nn.Sigmoid()(x)

        return x


class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=(3, 3), stride=(1, 1), pool=(2, 2)):
        super(Conv_2d, self).__init__()
        padding = (shape[0] // 2, shape[1] // 2)
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=shape, stride=stride, padding=padding),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool)
        )

    def forward(self, x):
        return self.block(x)