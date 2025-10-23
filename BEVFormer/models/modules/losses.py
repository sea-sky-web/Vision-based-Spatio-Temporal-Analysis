import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """Binary focal loss on logits in [B,1,H,W].
        Applies sigmoid inside.
        """
        # sigmoid
        probs = torch.sigmoid(logits)
        eps = 1e-6
        probs = torch.clamp(probs, eps, 1.0 - eps)
        pt = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss