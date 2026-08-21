




from abc import ABCMeta, abstractmethod

import torch

from ..builder import MASK_ASSIGNERS, build_match_cost

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


class AssignResult(metaclass=ABCMeta):


    def __init__(self, num_gts, gt_inds, labels):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.labels = labels

    @property
    def info(self):
        info = {
            "num_gts": self.num_gts,
            "gt_inds": self.gt_inds,
            "labels": self.labels,
        }
        return info


class BaseAssigner(metaclass=ABCMeta):


    @abstractmethod
    def assign(self, masks, gt_masks, gt_masks_ignore=None, gt_labels=None):

        pass


@MASK_ASSIGNERS.register_module()
class MaskHungarianAssigner(BaseAssigner):




















    def __init__(
        self,
        cls_cost=dict(type="ClassificationCost", weight=1.0),
        dice_cost=dict(type="DiceCost", weight=1.0),
        mask_cost=dict(type="MaskFocalCost", weight=1.0),
    ):
        self.cls_cost = build_match_cost(cls_cost)
        self.dice_cost = build_match_cost(dice_cost)
        self.mask_cost = build_match_cost(mask_cost)

    def assign(self, cls_pred, mask_pred, gt_labels, gt_masks, img_meta, gt_masks_ignore=None, eps=1e-7):






























        assert gt_masks_ignore is None, "Only case when gt_masks_ignore is None is supported."
        num_gts, num_queries = gt_labels.shape[0], cls_pred.shape[0]

        
        assigned_gt_inds = cls_pred.new_full((num_queries,), -1, dtype=torch.long)
        assigned_labels = cls_pred.new_full((num_queries,), -1, dtype=torch.long)
        if num_gts == 0 or num_queries == 0:
            
            if num_gts == 0:
                
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, labels=assigned_labels)

        
        
        if self.cls_cost.weight != 0 and cls_pred is not None:
            cls_cost = self.cls_cost(cls_pred, gt_labels)
        else:
            cls_cost = 0

        if self.mask_cost.weight != 0:
            
            
            
            mask_cost = self.mask_cost(mask_pred, gt_masks)
        else:
            mask_cost = 0

        if self.dice_cost.weight != 0:
            dice_cost = self.dice_cost(mask_pred, gt_masks)
        else:
            dice_cost = 0
        cost = cls_cost + mask_cost + dice_cost

        
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" ' "to install scipy first.")

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(cls_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(cls_pred.device)

        
        
        assigned_gt_inds[:] = 0
        
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(num_gts, assigned_gt_inds, labels=assigned_labels)
