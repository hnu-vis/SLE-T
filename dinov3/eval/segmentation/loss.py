




import functools

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def reduce_loss(loss, reduction) -> torch.Tensor:









    reduction_enum = nn._reduction.get_enum(reduction)
    
    assert reduction_enum in [0, 1, 2]
    if reduction_enum == 0:
        return loss
    if reduction_enum == 1:
        return loss.mean()
    return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None) -> torch.Tensor:











    
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        
        if reduction == "mean":
            
            
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):










    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction="mean", avg_factor=None, **kwargs):
        
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


def get_class_weight(class_weight):






    if isinstance(class_weight, str):
        class_weight = np.load(class_weight)

    return class_weight


@weighted_loss
def dice_loss(pred, target, valid_mask, smooth=1, exponent=2, class_weight=None, ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            dice_loss = binary_dice_loss(
                pred[:, i], target[..., i], valid_mask=valid_mask, smooth=smooth, exponent=exponent
            )
            if class_weight is not None:
                dice_loss *= class_weight[i]
            total_loss += dice_loss
    return total_loss / num_classes


@weighted_loss
def binary_dice_loss(pred, target, valid_mask, smooth=1, exponent=2, **kwargs):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth

    return 1 - num / den


class DiceLoss(nn.Module):






















    def __init__(
        self,
        smooth=1,
        exponent=2,
        reduction="mean",
        class_weight=None,
        loss_weight=1.0,
        ignore_index=255,
        loss_name="loss_dice",
        **kwargs,
    ):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self, pred, target, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()

        loss = self.loss_weight * dice_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index,
        )
        return loss


@weighted_loss
def multilabel_dice_loss(pred, target, valid_mask, smooth=1, exponent=2, class_weight=None, ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            dice_loss = binary_dice_loss(
                pred[:, i], target[:, i], valid_mask=valid_mask, smooth=smooth, exponent=exponent
            )
            if class_weight is not None:
                dice_loss *= class_weight[i]
            total_loss += dice_loss
    return total_loss / num_classes


class MultilabelDiceLoss(DiceLoss):
    def forward(self, pred, target, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.sigmoid(pred)
        if False:
            valid_mask = (target[..., self.ignore_index] == 0).long()
        else:
            valid_mask = torch.ones_like(target[:, 0]).long()

        loss = self.loss_weight * multilabel_dice_loss(
            pred,
            target,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index,
        )
        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        class_weight=None,
        loss_weight=1.0,
        reduction="mean",
        avg_factor=None,
        ignore_index=255,
        avg_non_ignore=False,
    ):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.avg_factor = avg_factor
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore

    def forward(self, pred, label):
        loss = F.cross_entropy(pred, label, weight=self.class_weight, reduction="none", ignore_index=self.ignore_index)

        if (self.avg_factor is None) and self.avg_non_ignore and self.reduction == "mean":
            avg_factor = label.numel() - (label == self.ignore_index).sum().item()
        else:
            avg_factor = None

        loss = weight_reduce_loss(loss, weight=self.weight, reduction=self.reduction, avg_factor=avg_factor)
        return self.loss_weight * loss


class MultiSegmentationLoss(nn.Module):




    def __init__(self, diceloss_weight=0.0, celoss_weight=0.0):
        super(MultiSegmentationLoss, self).__init__()

        if diceloss_weight > 0:
            self.loss = MultilabelDiceLoss(loss_weight=diceloss_weight)
        elif celoss_weight > 0:
            self.loss = CrossEntropyLoss(reduction="mean", loss_weight=celoss_weight)
        else:
            self.loss = lambda _: 0

    def forward(self, pred, gt):

        return self.loss(pred, gt)
