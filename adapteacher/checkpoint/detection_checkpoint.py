
from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts
from detectron2.checkpoint import DetectionCheckpointer


from typing import Any
from fvcore.common.checkpoint import _strip_prefix_if_present, _IncompatibleKeys


class DetectionTSCheckpointer(DetectionCheckpointer):
    def _load_model(self, checkpoint):
        if checkpoint.get("__author__", None) == "Caffe2":
            
            if checkpoint.get("matching_heuristics", False):
                self._convert_ndarray_to_tensor(checkpoint["model"])
                
                model_state_dict = self.model.modelStudent.state_dict()
                align_and_update_state_dicts(
                    model_state_dict,
                    checkpoint["model"],
                    c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
                )
                checkpoint["model"] = model_state_dict

            
            incompatible = self._load_student_model(checkpoint)

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

    def _load_student_model(self, checkpoint: Any) -> _IncompatibleKeys:  
        checkpoint_state_dict = checkpoint.pop("model")
        self._convert_ndarray_to_tensor(checkpoint_state_dict)

        
        
        
        _strip_prefix_if_present(checkpoint_state_dict, "module.")

        
        model_state_dict = self.model.modelStudent.state_dict()
        incorrect_shapes = []
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    incorrect_shapes.append((k, shape_checkpoint, shape_model))
                    checkpoint_state_dict.pop(k)
        
        incompatible = self.model.modelStudent.load_state_dict(
            checkpoint_state_dict, strict=False
        )
        return _IncompatibleKeys(
            missing_keys=incompatible.missing_keys,
            unexpected_keys=incompatible.unexpected_keys,
            incorrect_shapes=incorrect_shapes,
        )


















































