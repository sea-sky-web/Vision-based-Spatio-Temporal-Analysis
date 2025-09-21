"""Simple BEV detection head producing a single-channel output heatmap."""
import torch.nn as nn


class SimpleBEVHead(nn.Module):
    def __init__(self, cfg, out_channels=1):
        super().__init__()
        Cb = 32
        self.net = nn.Sequential(
            nn.Conv2d(Cb, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)
