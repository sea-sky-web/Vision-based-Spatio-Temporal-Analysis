import torch
import torch.nn as nn
import torch.nn.functional as F

class AppearanceHead(nn.Module):
    def __init__(self, in_channels: int, emb_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_channels, emb_dim)

    def forward(self, roi_feats: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, C, H, W) - ROI features
        Output: (B, D) - L2-normalized embedding
        """
        x = self.conv1(roi_feats)
        x = self.bn1(x)
        x = self.relu(x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
