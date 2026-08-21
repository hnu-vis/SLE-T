
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers,
    
)

from IPython import embed

"""
This is removed from the newer versions of detectron2, we add it back for compatibility.
We don't use focal loss in our code, but it is in the code for Adaptive Teacher.
"""
from fvcore.nn import giou_loss, smooth_l1_loss
from detectron2.layers import cat, cross_entropy, nonzero_tuple
from detectron2.structures import Boxes
from detectron2.utils.events import get_event_storage

def _log_classification_stats(pred_logits, gt_classes, prefix="fast_rcnn"):







    num_instances = gt_classes.numel()

    if num_instances == 0:
        return

    pred_classes = pred_logits.argmax(dim=1)
    bg_class_ind = pred_logits.shape[1] - 1

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

    storage = get_event_storage()
    storage.put_scalar(f"{prefix}/cls_accuracy", num_accurate / num_instances)

    if num_fg > 0:
        storage.put_scalar(f"{prefix}/fg_cls_accuracy", fg_num_accurate / num_fg)
        storage.put_scalar(f"{prefix}/false_negative", num_false_negative / num_fg)

class FastRCNNOutputs:





    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
    ):























        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type

        self.image_shapes = [x.image_size for x in proposals]

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"

            
            
            if proposals[0].has("gt_classes"):
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)

                
                
                
                
                gt_boxes = [
                    p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes for p in proposals
                ]
                self.gt_boxes = box_type.cat(gt_boxes)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(self.proposals) == 0  

    def softmax_cross_entropy_loss(self):



        _log_classification_stats(self.pred_class_logits, self.gt_classes)
        return cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def box_reg_loss(self):



        if self._no_instances:
            return 0.0 * self.pred_proposal_deltas.sum()

        box_dim = self.proposals.tensor.size(1)  
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1
        
        
        
        
        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]

        if cls_agnostic_bbox_reg:
            
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            
            
            
            
            gt_class_cols = box_dim * self.gt_classes[fg_inds, None] + torch.arange(
                box_dim, device=device
            )

        if self.box_reg_loss_type == "smooth_l1":
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor, self.gt_boxes.tensor
            )
            loss_box_reg = smooth_l1_loss(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                self.proposals.tensor[fg_inds],
            )
            loss_box_reg = giou_loss(
                fg_pred_boxes,
                self.gt_boxes.tensor[fg_inds],
                reduction="sum",
            )
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def losses(self):



        return {"loss_cls": self.softmax_cross_entropy_loss(), "loss_box_reg": self.box_reg_loss()}

    def predict_boxes(self):



        pred = self.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)
        return pred.split(self.num_preds_per_image, dim=0)

    def predict_probs(self):



        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

'''
use smooth label for target pseudo label 
'''
class FastRCNNOutputLayers_smooth_label(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape):
        super(FastRCNNOutputLayers_smooth_label, self).__init__(cfg, input_shape)
    
    def losses(self, predictions, proposals, branch='supervised'):










        scores, proposal_deltas = predictions

        
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            
            
            
            
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        if self.use_sigmoid_ce:
            loss_cls = self.sigmoid_cross_entropy_loss(scores, gt_classes)
        else:
            
            
            if(branch == 'supervised'):
                loss_cls = cross_entropy(scores, gt_classes, reduction="mean")
            elif(branch == 'supervised_target'):
                
                loss_cls = cross_entropy(scores, gt_classes, reduction="mean", label_smoothing=0.1)
                
                
            else:
                raise NotImplementedError()
            
            

        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
    


class FastRCNNFocaltLossOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape):
        super(FastRCNNFocaltLossOutputLayers, self).__init__(cfg, input_shape)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

    def losses(self, predictions, proposals):






        scores, proposal_deltas = predictions
        losses = FastRCNNFocalLoss(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            num_classes=self.num_classes,
        ).losses()

        return losses


class FastRCNNFocalLoss(FastRCNNOutputs):





    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        num_classes=80,
    ):
        super(FastRCNNFocalLoss, self).__init__(
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta,
            box_reg_loss_type,
        )
        self.num_classes = num_classes

    def losses(self):
        return {
            "loss_cls": self.comput_focal_loss(),
            "loss_box_reg": self.box_reg_loss(),
        }

    def comput_focal_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            FC_loss = FocalLoss(
                gamma=1.5,
                num_classes=self.num_classes,
            )
            total_loss = FC_loss(input=self.pred_class_logits, target=self.gt_classes)
            total_loss = total_loss / self.gt_classes.shape[0]

            return total_loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        gamma=1.0,
        num_classes=80,
    ):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

        self.num_classes = num_classes

    def forward(self, input, target):
        
        CE = F.cross_entropy(input, target, reduction="none")
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE
        return loss.sum()
