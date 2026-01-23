# file: minecraftscc/sparse/losses.py
import torch
from torch import nn


def sparse_ce_loss(logits: torch.Tensor,
        target_labels: torch.Tensor,
        direct_mask: torch.Tensor | None = None,
        direct_coef: float = 2.0,
        class_weights: torch.Tensor | None = None) -> torch.Tensor:
    """
    Args:
        logits: [Q, C]
        target_labels: [Q] with ignore_index = -1
    """
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="none", weight=class_weights)
    loss = criterion(logits, target_labels.long())  # [Q]
    if direct_mask is not None:
        loss = loss * (1.0 + direct_mask.to(loss.dtype) * (direct_coef - 1.0))
    return loss.mean()
