import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class ViewEncoder(nn.Module, ABC):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: Tensor[B*V, 3, H, W] or Tensor[B, V, 3, H, W]
        Returns: Tensor[B, V, C, Hf, Wf]
        """
        raise NotImplementedError

    def load_pretrained(self, weights_path: str):
        try:
            state = torch.load(weights_path, map_location='cpu')
            self.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"[ViewEncoder] load_pretrained failed: {e}")

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False