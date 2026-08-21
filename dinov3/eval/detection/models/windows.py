




import math

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from ..util.misc import NestedTensor


class WindowsWrapper(torch.nn.Module):















    def __init__(self, backbone, n_windows_w, n_windows_h, patch_size):
        
        super().__init__()
        self._backbone = backbone
        self._n_windows_w = n_windows_w
        self._n_windows_h = n_windows_h
        self._patch_size = patch_size
        self.strides = backbone.strides
        self.num_channels = [el * 2 for el in backbone.num_channels]  

    def forward(self, tensor_list: NestedTensor):
        tensors = tensor_list.tensors
        original_h, original_w = tensors.shape[2], tensors.shape[3]
        
        window_h = math.ceil((original_h // self._n_windows_h) / self._patch_size) * self._patch_size
        window_w = math.ceil((original_w // self._n_windows_w) / self._patch_size) * self._patch_size
        all_h = [window_h] * (self._n_windows_h - 1) + [original_h - window_h * (self._n_windows_h - 1)]
        all_w = [window_w] * (self._n_windows_w - 1) + [original_w - window_w * (self._n_windows_w - 1)]
        all_h_cumsum = [0] + list(np.cumsum(all_h))
        all_w_cumsum = [0] + list(np.cumsum(all_w))
        window_patch_features = [[0 for _ in range(self._n_windows_w)] for _ in range(self._n_windows_h)]

        for ih in range(self._n_windows_h):
            for iw in range(self._n_windows_w):
                window_tensor = v2.functional.crop(
                    tensors, top=all_h_cumsum[ih], left=all_w_cumsum[iw], height=all_h[ih], width=all_w[iw]
                )
                window_mask = v2.functional.crop(
                    tensor_list.mask, top=all_h_cumsum[ih], left=all_w_cumsum[iw], height=all_h[ih], width=all_w[iw]
                )
                window_patch_features[ih][iw] = self._backbone(NestedTensor(tensors=window_tensor, mask=window_mask))[0]

        window_tensors = torch.cat(
            [
                torch.cat([el.tensors for el in window_patch_features[ih]], dim=-1)  
                for ih in range(len(window_patch_features))
            ],
            dim=-2,
        )
        
        resized_global_tensor = v2.functional.resize(tensors, size=(window_h, window_w))
        global_features = self._backbone(
            NestedTensor(tensors=resized_global_tensor, mask=tensor_list.mask)
        )  

        concat_tensors = torch.cat(
            [v2.functional.resize(global_features[0].tensors, size=window_tensors.shape[-2:]), window_tensors], dim=1
        )
        global_mask = F.interpolate(tensor_list.mask[None].float(), size=concat_tensors.shape[-2:]).to(torch.bool)[0]
        out = [NestedTensor(tensors=concat_tensors, mask=global_mask)]
        return out
