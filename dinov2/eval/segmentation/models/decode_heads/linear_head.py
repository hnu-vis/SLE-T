




import torch
import torch.nn as nn

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize


@HEADS.register_module()
class BNHead(BaseDecodeHead):


    def __init__(self, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        assert self.in_channels == self.channels
        self.bn = nn.SyncBatchNorm(self.in_channels)
        self.resize_factors = resize_factors

    def _forward_feature(self, inputs):










        
        x = self._transform_inputs(inputs)
        
        feats = self.bn(x)
        
        return feats

    def _transform_inputs(self, inputs):







        if self.input_transform == "resize_concat":
            
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            
            for i, x in enumerate(inputs):
                if len(x.shape) == 2:
                    inputs[i] = x[:, :, None, None]
            
            inputs = [inputs[i] for i in self.in_index]
            
            
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), (len(self.resize_factors), len(inputs))
                inputs = [
                    resize(input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area")
                    for x, f in zip(inputs, self.resize_factors)
                ]
                
            upsampled_inputs = [
                resize(input=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):

        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
