from __future__ import annotations

import torch
from torch import nn

from .datasets import IGNORE_INDEX


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != IGNORE_INDEX
        if not bool(valid.any()):
            return logits.sum() * 0.0
        probs = torch.sigmoid(logits.squeeze(1))
        target_f = target.float().clamp(0, 1)
        probs = probs[valid]
        target_f = target_f[valid]
        intersection = (probs * target_f).sum()
        denom = probs.sum() + target_f.sum()
        return 1.0 - ((2.0 * intersection + self.smooth) / (denom + self.smooth))


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != IGNORE_INDEX
        if not bool(valid.any()):
            return logits.sum() * 0.0
        target_f = target.float().clamp(0, 1)
        bce_map = self.bce(logits.squeeze(1), target_f)
        bce_loss = bce_map[valid].mean()
        return self.bce_weight * bce_loss + self.dice_weight * self.dice(logits, target)
