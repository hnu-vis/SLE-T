import math
from typing import List, Optional
import torch
from torch import nn
from torchvision.ops import RoIPool

from detectron2.layers import ROIAlign, ROIAlignRotated, cat, nonzero_tuple, shapes_to_tensor
from detectron2.structures import Boxes
from detectron2.utils.tracing import assert_fx_safe, is_fx_tracing
from detectron2.modeling.poolers import convert_boxes_to_pooler_format, _create_zeros

class SingleScaleROIPooler(nn.Module):





    def __init__(
        self,
        output_size,
        scales,
        sampling_ratio,
        pooler_type
    ):














        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size

        assert len(scales) == 1, "SingleScaleROIPooler only supports single scale"

        if pooler_type == "ROIAlign":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False
                )
                for scale in scales
            )
        elif pooler_type == "ROIAlignV2":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
                )
                for scale in scales
            )
        elif pooler_type == "ROIPool":
            self.level_poolers = nn.ModuleList(
                RoIPool(output_size, spatial_scale=scale) for scale in scales
            )
        elif pooler_type == "ROIAlignRotated":
            self.level_poolers = nn.ModuleList(
                ROIAlignRotated(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio)
                for scale in scales
            )
        else:
            raise ValueError("Unknown pooler type: {}".format(pooler_type))

    def forward(self, x: List[torch.Tensor], box_lists: List[Boxes]):














        num_level_assignments = len(self.level_poolers)

        if not is_fx_tracing():
            torch._assert(
                isinstance(x, list) and isinstance(box_lists, list),
                "Arguments to pooler must be lists",
            )
        assert_fx_safe(
            len(x) == num_level_assignments,
            "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
                num_level_assignments, len(x)
            ),
        )
        assert_fx_safe(
            len(box_lists) == x[0].size(0),
            "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
                x[0].size(0), len(box_lists)
            ),
        )
        if len(box_lists) == 0:
            return _create_zeros(None, x[0].shape[1], *self.output_size, x[0])

        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)

        return self.level_poolers[0](x[0], pooler_fmt_boxes)
