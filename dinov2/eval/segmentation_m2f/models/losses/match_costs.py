




import torch
import torch.nn.functional as F

from ..builder import MATCH_COST


@MATCH_COST.register_module()
class ClassificationCost:



















    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, cls_pred, gt_labels):









        
        
        
        
        cls_score = cls_pred.softmax(-1)
        cls_cost = -cls_score[:, gt_labels]
        return cls_cost * self.weight


@MATCH_COST.register_module()
class DiceCost:









    def __init__(self, weight=1.0, pred_act=False, eps=1e-3):
        self.weight = weight
        self.pred_act = pred_act
        self.eps = eps

    def binary_mask_dice_loss(self, mask_preds, gt_masks):










        mask_preds = mask_preds.reshape((mask_preds.shape[0], -1))
        gt_masks = gt_masks.reshape((gt_masks.shape[0], -1)).float()
        numerator = 2 * torch.einsum("nc,mc->nm", mask_preds, gt_masks)
        denominator = mask_preds.sum(-1)[:, None] + gt_masks.sum(-1)[None, :]
        loss = 1 - (numerator + self.eps) / (denominator + self.eps)
        return loss

    def __call__(self, mask_preds, gt_masks):








        if self.pred_act:
            mask_preds = mask_preds.sigmoid()
        dice_cost = self.binary_mask_dice_loss(mask_preds, gt_masks)
        return dice_cost * self.weight


@MATCH_COST.register_module()
class CrossEntropyLossCost:








    def __init__(self, weight=1.0, use_sigmoid=True):
        assert use_sigmoid, "use_sigmoid = False is not supported yet."
        self.weight = weight
        self.use_sigmoid = use_sigmoid

    def _binary_cross_entropy(self, cls_pred, gt_labels):









        cls_pred = cls_pred.flatten(1).float()
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        pos = F.binary_cross_entropy_with_logits(cls_pred, torch.ones_like(cls_pred), reduction="none")
        neg = F.binary_cross_entropy_with_logits(cls_pred, torch.zeros_like(cls_pred), reduction="none")
        cls_cost = torch.einsum("nc,mc->nm", pos, gt_labels) + torch.einsum("nc,mc->nm", neg, 1 - gt_labels)
        cls_cost = cls_cost / n

        return cls_cost

    def __call__(self, cls_pred, gt_labels):








        if self.use_sigmoid:
            cls_cost = self._binary_cross_entropy(cls_pred, gt_labels)
        else:
            raise NotImplementedError

        return cls_cost * self.weight
