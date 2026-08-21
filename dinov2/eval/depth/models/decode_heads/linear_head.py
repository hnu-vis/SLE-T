




import torch
import torch.nn as nn

from ...ops import resize
from ..builder import HEADS
from .decode_head import DepthBaseDecodeHead


@HEADS.register_module()
class BNHead(DepthBaseDecodeHead):


    def __init__(self, input_transform="resize_concat", in_index=(0, 1, 2, 3), upsample=1, **kwargs):
        super().__init__(**kwargs)
        self.input_transform = input_transform
        self.in_index = in_index
        self.upsample = upsample
        
        if self.classify:
            self.conv_depth = nn.Conv2d(self.channels, self.n_bins, kernel_size=1, padding=0, stride=1)
        else:
            self.conv_depth = nn.Conv2d(self.channels, 1, kernel_size=1, padding=0, stride=1)

    def _transform_inputs(self, inputs):







        if "concat" in self.input_transform:
            inputs = [inputs[i] for i in self.in_index]
            if "resize" in self.input_transform:
                inputs = [
                    resize(
                        input=x,
                        size=[s * self.upsample for s in inputs[0].shape[2:]],
                        mode="bilinear",
                        align_corners=self.align_corners,
                    )
                    for x in inputs
                ]
            inputs = torch.cat(inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _forward_feature(self, inputs, img_metas=None, **kwargs):








        
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            if len(x) == 2:
                x, cls_token = x[0], x[1]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                cls_token = cls_token[:, :, None, None].expand_as(x)
                inputs[i] = torch.cat((x, cls_token), 1)
            else:
                x = x[0]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                inputs[i] = x
        x = self._transform_inputs(inputs)
        
        return x

    def forward(self, inputs, img_metas=None, **kwargs):

        output = self._forward_feature(inputs, img_metas=img_metas, **kwargs)
        output = self.depth_pred(output)

        return output
