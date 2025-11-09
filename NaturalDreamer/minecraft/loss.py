from torch import nn


def CE_ssc_loss(pred, target, class_weights):
    """
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    """
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=-1, reduction="mean"
    )
    loss = criterion(pred, target.long())

    return loss

