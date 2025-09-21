"""Minimal PyTorch Dataset for Wildtrack used in dry-run.

If real images/annotations are missing, the dataset will generate random tensors
with shapes based on the config so the forward pass can be exercised.
"""
from pathlib import Path
import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np


class WildtrackDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_root = Path(cfg.DATA_ROOT)

        # Attempt to find image files; if not present, will use synthetic data
        self.has_real = self.data_root.exists()
        self.num_cameras = getattr(cfg, 'NUM_CAMERAS', 7)

        # create a small index
        self.length = 10

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        C, H, W = self.cfg.IMAGE_SIZE
        # images: (NumCams, C, H, W)
        if self.has_real:
            # For brevity, we don't implement full real loader in the scaffold.
            imgs = torch.zeros((self.num_cameras, C, H, W), dtype=torch.float32)
        else:
            imgs = torch.randn((self.num_cameras, C, H, W), dtype=torch.float32)

        sample = {
            'images': imgs,
            'meta': {'idx': idx}
        }
        return sample
