from torch import nn


class WarpedChunkCNN(nn.Module):
    '''
    Tall-chunk CNN architecture.
    So-called vgg-ish model with a small receptive field.
    Deeper layers, smaller pooling (2x2).
    '''

    def __init__(self, n_channels=64, n_class=50):
        super().__init__()

        # CNN
        self.layer1 = Conv_2d(1,    1 * n_channels, shape=(3, 3),   stride=(2, 1),      pool=(2, 2))  # F↓
        self.layer2 = Conv_2d(1 * n_channels,   1 * n_channels, shape=(3, 3),   stride=(2, 1),      pool=(2, 2))  # F↓
        self.layer3 = Conv_2d(1 * n_channels,   2 * n_channels, shape=(3, 3),   stride=(1, 1),      pool=(2, 2))  # T↓
        self.layer4 = Conv_2d(2 * n_channels,   4 * n_channels, shape=(3, 3),   stride=(1, 1),      pool=(2, 2))  # T↓
        self.layer5 = Conv_2d(4 * n_channels,   4 * n_channels, shape=(3, 3),   stride=(1, 1),      pool=(1, 1))  # T↓
        self.layer6 = Conv_2d(4 * n_channels,   4 * n_channels, shape=(3, 3),   stride=(1, 1),      pool=(1, 2))  # T↓
        self.layer7 = Conv_2d(6 * n_channels,   6 * n_channels, shape=(3, 3),   stride=(1, 1),      pool=(1, 1))  # T↓
        self.layer8 = Conv_2d(6 * n_channels,   6 * n_channels, shape=(3, 3),   stride=(1, 1),      pool=(1, 2))  # T↓


        #self.global_pool = nn.AdaptiveMaxPool2d((1, 1))  # output shape: [B, C, 1, 1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6 * n_channels, 6 * n_channels),
            nn.BatchNorm1d(6 * n_channels),
            nn.Linear(6 * n_channels, n_class),
            nn.Dropout(0.5),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer2_5(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        # = self.global_pool(x)
        x = self.classifier(x)
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