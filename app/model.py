import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(4, outputs), nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(4, outputs), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = ConvBlock(4, 16)
        self.encoder2 = ConvBlock(16, 32)
        self.encoder3 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(64, 128)
        self.decoder3 = ConvBlock(192, 64)
        self.decoder2 = ConvBlock(96, 32)
        self.decoder1 = ConvBlock(48, 16)
        self.output = nn.Conv2d(16, 4, 1)

    def upsample(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        x = self.bottleneck(self.pool(e3))
        x = self.decoder3(self.upsample(x, e3))
        x = self.decoder2(self.upsample(x, e2))
        x = self.decoder1(self.upsample(x, e1))
        return self.output(x)
