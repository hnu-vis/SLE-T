







import torch

from ..builder import BBOX_SAMPLERS
from .base_sampler import BaseSampler
from .mask_sampling_result import MaskSamplingResult


@BBOX_SAMPLERS.register_module()
class MaskPseudoSampler(BaseSampler):


    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):

        raise NotImplementedError

    def _sample_neg(self, **kwargs):

        raise NotImplementedError

    def sample(self, assign_result, masks, gt_masks, **kwargs):









        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = masks.new_zeros(masks.shape[0], dtype=torch.uint8)
        sampling_result = MaskSamplingResult(pos_inds, neg_inds, masks, gt_masks, assign_result, gt_flags)
        return sampling_result
