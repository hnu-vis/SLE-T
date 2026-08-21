
from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts
from detectron2.checkpoint import DetectionCheckpointer


from typing import Any
from fvcore.common.checkpoint import _strip_prefix_if_present, _IncompatibleKeys

import logging
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
import os
from detectron2.utils import comm
from urllib.parse import urlparse

class DetectionTSCheckpointer(DetectionCheckpointer):
    def _load_model(self, checkpoint):
        if checkpoint.get("__author__", None) == "Caffe2":
            
            if checkpoint.get("matching_heuristics", False):
                self._convert_ndarray_to_tensor(checkpoint["model"])
                
                model_state_dict = self.model.modelStudent.state_dict()
                renamed_ckpt = align_and_update_state_dicts(
                    model_state_dict,
                    checkpoint["model"],
                    c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
                )
                checkpoint["model"] = renamed_ckpt

            
            incompatible = self._load_student_model(checkpoint)

            model_buffers = dict(self.model.modelStudent.named_buffers(recurse=False))
            for k in ["pixel_mean", "pixel_std"]:
                
                
                
                if k in model_buffers:
                    try:
                        incompatible.missing_keys.remove(k)
                    except ValueError:
                        pass
            return incompatible

        elif all("vgg" in x for x in checkpoint["model"].keys()):
            
            model_state_dict = self.model.modelStudent.state_dict()
            renamed_ckpt = align_and_update_state_dicts(
                model_state_dict,
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
            
            checkpoint["model"] = renamed_ckpt

            
            incompatible = self._load_student_model(checkpoint)

            model_buffers = dict(self.model.modelStudent.named_buffers(recurse=False))
            for k in ["pixel_mean", "pixel_std"]:
                
                
                
                if k in model_buffers:
                    try:
                        incompatible.missing_keys.remove(k)
                    except ValueError:
                        pass
            return incompatible

        elif 'modelStudent.model.transformer.encoder.layers.0.norms.0.weight' in checkpoint['model'].keys(): 
            if 'modelStudent.transformer.encoder.layers.0.norms.0.weight' in self.model.state_dict().keys(): 
                new_key = []
                new_vals = []
                for key, value in checkpoint['model'].items():
                    new_key.append(key.replace('.model.','.'))
                    new_vals.append(value)
                checkpoint['model'] = OrderedDict(zip(new_key,new_vals))
            incompatible = super()._load_model(checkpoint)
            return incompatible

        elif 'lm_head.bias' in checkpoint['model'].keys(): 
            new_key = []
            new_vals = []
            for key, value in checkpoint['module'].items():
                new_key.append('backbone.net.' + key)
                new_vals.append(value)
            checkpoint['model'] = OrderedDict(zip(new_key,new_vals))
            incompatible = self._load_student_model(checkpoint, wrapped_model=True)

            model_buffers = dict(self.model.modelStudent.named_buffers(recurse=False))
            for k in ["pixel_mean", "pixel_std"]:
                
                
                
                if k in model_buffers:
                    try:
                        incompatible.missing_keys.remove(k)
                    except ValueError:
                        pass
            return incompatible

        elif "cls_token" in checkpoint["model"].keys():
            
            model_state_dict = self.model.modelStudent.backbone.state_dict()
            renamed_ckpt = align_and_update_state_dicts(
                model_state_dict,
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
            
            checkpoint["model"] = renamed_ckpt

            
            incompatible = self._load_student_model(checkpoint, backbone_only=True)

            model_buffers = dict(self.model.modelStudent.named_buffers(recurse=False))
            for k in ["pixel_mean", "pixel_std"]:
                
                
                
                if k in model_buffers:
                    try:
                        incompatible.missing_keys.remove(k)
                    except ValueError:
                        pass
            return incompatible

        else:  
            if checkpoint.get("matching_heuristics", False):
                self._convert_ndarray_to_tensor(checkpoint["model"])
                
                model_state_dict = self.model.state_dict()
                align_and_update_state_dicts(
                    model_state_dict,
                    checkpoint["model"],
                    c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
                )
                checkpoint["model"] = model_state_dict 
            
            incompatible = super()._load_model(checkpoint)

            model_buffers = dict(self.model.named_buffers(recurse=False))
            for k in ["pixel_mean", "pixel_std"]:
                
                
                
                if k in model_buffers:
                    try:
                        incompatible.missing_keys.remove(k)
                    except ValueError:
                        pass
            return incompatible

    def _load_student_model(self, checkpoint: Any, backbone_only=False, wrapped_model=False) -> _IncompatibleKeys:  
        checkpoint_state_dict = checkpoint.pop("model")
        self._convert_ndarray_to_tensor(checkpoint_state_dict)

        
        
        
        _strip_prefix_if_present(checkpoint_state_dict, "module.")

        
        if backbone_only:
            model_state_dict = self.model.modelStudent.backbone.state_dict()        
        if wrapped_model:
            model_state_dict = self.model.modelStudent.model.state_dict()
        else:
            model_state_dict = self.model.modelStudent.state_dict()
        incorrect_shapes = []
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    incorrect_shapes.append((k, shape_checkpoint, shape_model))
                    checkpoint_state_dict.pop(k)
        
        if backbone_only:
            incompatible = self.model.modelStudent.backbone.load_state_dict(checkpoint_state_dict, strict=False)
        if wrapped_model:
            incompatible = self.model.modelStudent.model.load_state_dict(checkpoint_state_dict, strict=False)
        else:
            incompatible = self.model.modelStudent.load_state_dict(checkpoint_state_dict, strict=False)
        
        
        
        return _IncompatibleKeys(
            missing_keys=incompatible.missing_keys,
            unexpected_keys=incompatible.unexpected_keys,
            incorrect_shapes=incorrect_shapes,
        )


















































