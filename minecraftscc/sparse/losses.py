# file: minecraftscc/sparse/losses.py
import torch
from torch import nn


def sparse_ce_loss(logits: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
    """
    Args:
        logits: [Q, C]
        target_labels: [Q] with ignore_index = -1
    """
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
    return criterion(logits, target_labels.long())
