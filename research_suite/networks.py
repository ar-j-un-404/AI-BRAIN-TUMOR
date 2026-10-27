import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self, cin, cout, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(4, cout), nn.LeakyReLU(.01, inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(4, cout),
            nn.LeakyReLU(.01, inplace=True), nn.Dropout3d(dropout))
    def forward(self, x): return self.layers(x)

class UNet3D(nn.Module):
    def __init__(self, channels=4, base=8, dropout=.1):
        super().__init__()
        self.e1 = Block(channels, base, dropout); self.e2 = Block(base, base*2, dropout)
        self.e3 = Block(base*2, base*4, dropout); self.mid = Block(base*4, base*8, dropout)
        self.d3 = Block(base*12, base*4, dropout); self.d2 = Block(base*6, base*2, dropout)
        self.d1 = Block(base*3, base, dropout); self.out = nn.Conv3d(base, 4, 1)
    def forward(self, x):
        a = self.e1(x); b = self.e2(F.max_pool3d(a, 2)); c = self.e3(F.max_pool3d(b, 2))
        d = self.mid(F.max_pool3d(c, 2))
        def up(z, skip): return torch.cat([F.interpolate(z, size=skip.shape[2:], mode='trilinear', align_corners=False), skip], 1)
        d = self.d3(up(d, c)); d = self.d2(up(d, b)); return self.out(self.d1(up(d, a)))

class Classifier3D(nn.Module):
    def __init__(self, channels=2):
        super().__init__()
        self.features = nn.Sequential(Block(channels, 8), nn.MaxPool3d(2),
            Block(8, 16), nn.MaxPool3d(2), Block(16, 32), nn.MaxPool3d(2), Block(32, 64))
        self.head = nn.Linear(64, 1)
    def forward(self, x): return self.head(self.features(x).mean((2, 3, 4))).squeeze(1)

def segmentation_loss(logits, target):
    loss = F.cross_entropy(logits.float(), target)
    probs = logits.float().softmax(1); terms = []
    for labels in ((1, 2, 3), (2, 3), (3,)):
        p = probs[:, labels].sum(1); t = torch.zeros_like(target, dtype=torch.bool)
        for label in labels: t |= target == label
        t = t.float(); axes = (1, 2, 3)
        terms.append(1 - ((2*(p*t).sum(axes)+1e-5)/(p.sum(axes)+t.sum(axes)+1e-5)).mean())
    return loss + torch.stack(terms).mean()

def enable_mc_dropout(model):
    model.eval()
    for layer in model.modules():
        if isinstance(layer, nn.Dropout3d): layer.train()
