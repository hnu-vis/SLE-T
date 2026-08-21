




import torch
import torch.nn as nn
from mmseg.models.builder import LOSSES
from mmseg.models.losses.utils import weight_reduce_loss


def dice_loss(pred, target, weight=None, eps=1e-3, reduction="mean", avg_factor=None):


















    input = pred.flatten(1)
    target = target.flatten(1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + eps
    c = torch.sum(target * target, 1) + eps
    d = (2 * a) / (b + c)
    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def naive_dice_loss(pred, target, weight=None, eps=1e-3, reduction="mean", avg_factor=None):
















    input = pred.flatten(1)
    target = target.flatten(1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input, 1)
    c = torch.sum(target, 1)
    d = (2 * a + eps) / (b + c + eps)
    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module(force=True)
class DiceLoss(nn.Module):
    def __init__(self, use_sigmoid=True, activate=True, reduction="mean", naive_dice=False, loss_weight=1.0, eps=1e-3):



























        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate

    def forward(self, pred, target, weight=None, reduction_override=None, avg_factor=None):


















        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                raise NotImplementedError

        if self.naive_dice:
            loss = self.loss_weight * naive_dice_loss(
                pred, target, weight, eps=self.eps, reduction=reduction, avg_factor=avg_factor
            )
        else:
            loss = self.loss_weight * dice_loss(
                pred, target, weight, eps=self.eps, reduction=reduction, avg_factor=avg_factor
            )

        return loss
