




import torch


class SamplingResult:


















    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)

            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds.long(), :]

        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):

        return torch.cat([self.pos_bboxes, self.neg_bboxes])

    def to(self, device):








        _dict = self.__dict__
        for key, value in _dict.items():
            if isinstance(value, torch.Tensor):
                _dict[key] = value.to(device)
        return self

    def __nice__(self):
        data = self.info.copy()
        data["pos_bboxes"] = data.pop("pos_bboxes").shape
        data["neg_bboxes"] = data.pop("neg_bboxes").shape
        parts = [f"'{k}': {v!r}" for k, v in sorted(data.items())]
        body = "    " + ",\n    ".join(parts)
        return "{\n" + body + "\n}"

    @property
    def info(self):

        return {
            "pos_inds": self.pos_inds,
            "neg_inds": self.neg_inds,
            "pos_bboxes": self.pos_bboxes,
            "neg_bboxes": self.neg_bboxes,
            "pos_is_gt": self.pos_is_gt,
            "num_gts": self.num_gts,
            "pos_assigned_gt_inds": self.pos_assigned_gt_inds,
        }

    @classmethod
    def random(cls, rng=None, **kwargs):




















        from mmdet.core.bbox import demodata
        from mmdet.core.bbox.assigners.assign_result import AssignResult
        from mmdet.core.bbox.samplers.random_sampler import RandomSampler

        rng = demodata.ensure_rng(rng)

        
        num = 32
        pos_fraction = 0.5
        neg_pos_ub = -1

        assign_result = AssignResult.random(rng=rng, **kwargs)

        
        bboxes = demodata.random_boxes(assign_result.num_preds, rng=rng)
        gt_bboxes = demodata.random_boxes(assign_result.num_gts, rng=rng)

        if rng.rand() > 0.2:
            
            gt_bboxes = gt_bboxes.squeeze()
            bboxes = bboxes.squeeze()

        if assign_result.labels is None:
            gt_labels = None
        else:
            gt_labels = None

        if gt_labels is None:
            add_gt_as_proposals = False
        else:
            add_gt_as_proposals = True  

        sampler = RandomSampler(
            num, pos_fraction, neg_pos_ub=neg_pos_ub, add_gt_as_proposals=add_gt_as_proposals, rng=rng
        )
        self = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        return self
