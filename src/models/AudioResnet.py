import torch
from torch import nn


class AudioResnet(nn.Module):
    def __init__(self, num_classes=249):
        super(AudioResnet, self).__init__()

        self.convIn = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=True)
        self.bnIn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.maxpoolIn = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stack1
        self.stack1 = nn.ModuleList([ResidualBlock(64) for _ in range(3)])
        self.stack2 = nn.ModuleList([ResidualBlock(128) for _ in range(4)])
        self.stack3 = nn.ModuleList([ResidualBlock(256) for _ in range(6)])
        self.stack4 = nn.ModuleList([ResidualBlock(512) for _ in range(3)])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fcOut = nn.Linear(512, num_classes, bias=True)
        #self.softmax = nn.LogSoftmax(dim=-1)

        # Initilise weights in fully connected layer
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                m.bias.data.zero_()
    def forward(self, x):
        x = self.convIn(x)
        x = self.bnIn(x)
        x = self.relu(x)
        x = self.maxpoolIn(x)

        for l in self.stack1:
            x = l(x)

        for l in self.stack2:
            x = l(x)

        for l in self.stack3:
            x = l(x)

        for l in self.stack4:
            x = l(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fcOut(x)


class ResidualBlock(nn.Module):
    def __init__(self, filters, downsample = False):
        super(ResidualBlock, self).__init__()
        # Determine subsampling
        s = 0.5 if downsample else 1.0

        self.conv1 = nn.Sequential(
                        nn.Conv2d(int(filters * s), filters, kernel_size = 3, stride = int(1/s), padding = 1),
                        nn.BatchNorm2d(filters),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(filters, filters, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(filters))
        self.downsample = nn.AvgPool2d(kernel_size=1, stride=2)
        self.relu = nn.ReLU()

        # Initialise weights according to the method described in
        # “Delving deep into rectifiers: Surpassing human-level performance on ImageNet
        # classification” - He, K. et al. (2015)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def shortcut(self, z, x):
        """
        Implements parameter free shortcut connection by identity mapping.
        If dimensions of input x are greater than activations then this
        is rectified by downsampling and then zero padding dimension 1
        as described by option A in paper.

        Parameters:
        - x: tensor
             the input to the block
        - z: tensor
             activations of block prior to final non-linearity
        """
        if x.shape != z.shape:
            d = self.downsample(x)
            p = torch.mul(d, 0)
            return z + torch.cat((d, p), dim=1)
        else:
            return z + x

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)

        out = self.shortcut(out, x)

        out = self.relu(out)
        return out