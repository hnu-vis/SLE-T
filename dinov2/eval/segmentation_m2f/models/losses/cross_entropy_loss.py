




import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import LOSSES
from mmseg.models.losses.utils import get_class_weight, weight_reduce_loss


def cross_entropy(
    pred,
    label,
    weight=None,
    class_weight=None,
    reduction="mean",
    avg_factor=None,
    ignore_index=-100,
    avg_non_ignore=False,
):























    
    
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction="none", ignore_index=ignore_index)

    
    
    
    
    if (avg_factor is None) and avg_non_ignore and reduction == "mean":
        avg_factor = label.numel() - (label == ignore_index).sum().item()
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, target_shape, ignore_index):

    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()

    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        bin_label_weights = bin_label_weights * valid_mask

    return bin_labels, bin_label_weights, valid_mask


def binary_cross_entropy(
    pred,
    label,
    weight=None,
    reduction="mean",
    avg_factor=None,
    class_weight=None,
    ignore_index=-100,
    avg_non_ignore=False,
    **kwargs,
):




















    if pred.size(1) == 1:
        
        
        assert label.max() <= 1, "For pred with shape [N, 1, H, W], its label must have at " "most 2 classes"
        pred = pred.squeeze()
    if pred.dim() != label.dim():
        assert (pred.dim() == 2 and label.dim() == 1) or (pred.dim() == 4 and label.dim() == 3), (
            "Only pred shape [N, C], label shape [N] or pred shape [N, C, " "H, W], label shape [N, H, W] are supported"
        )
        
        
        label, weight, valid_mask = _expand_onehot_labels(label, weight, pred.shape, ignore_index)
    else:
        
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        if weight is not None:
            weight = weight * valid_mask
        else:
            weight = valid_mask
    
    if reduction == "mean" and avg_factor is None and avg_non_ignore:
        avg_factor = valid_mask.sum().item()

    loss = F.binary_cross_entropy_with_logits(pred, label.float(), pos_weight=class_weight, reduction="none")
    
    loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(
    pred, target, label, reduction="mean", avg_factor=None, class_weight=None, ignore_index=None, **kwargs
):





















    assert ignore_index is None, "BCE loss does not support ignore_index"
    assert reduction == "mean" and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(pred_slice, target, weight=class_weight, reduction="mean")[None]


@LOSSES.register_module(force=True)
class CrossEntropyLoss(nn.Module):




















    def __init__(
        self,
        use_sigmoid=False,
        use_mask=False,
        reduction="mean",
        class_weight=None,
        loss_weight=1.0,
        loss_name="loss_ce",
        avg_non_ignore=False,
    ):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.avg_non_ignore = avg_non_ignore
        if not self.avg_non_ignore and self.reduction == "mean":
            warnings.warn(
                "Default ``avg_non_ignore`` is False, if you would like to "
                "ignore the certain label and average loss over non-ignore "
                "labels, which is the same with PyTorch official "
                "cross_entropy, set ``avg_non_ignore=True``."
            )

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy
        self._loss_name = loss_name

    def extra_repr(self):

        s = f"avg_non_ignore={self.avg_non_ignore}"
        return s

    def forward(
        self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, ignore_index=-100, **kwargs
    ):

        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            avg_non_ignore=self.avg_non_ignore,
            ignore_index=ignore_index,
            **kwargs,
        )
        return loss_cls

    @property
    def loss_name(self):











        return self._loss_name
