from typing import Tuple, Optional
import torch.nn as nn
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone, Backbone
from detectron2.modeling.roi_heads import build_roi_heads

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from adapteacher.modeling.meta_arch.rcnn import DAobjTwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN_GC

@META_ARCH_REGISTRY.register()
class DAobjTwoStagePseudoLabGeneralizedRCNN_shortcut(DAobjTwoStagePseudoLabGeneralizedRCNN):
    def __init__(
        self,
        cfg
    ):










        super(DAobjTwoStagePseudoLabGeneralizedRCNN_shortcut, self).__init__(cfg)
        

    
    
    
    
    
    
    
    
    
    
    
    
    

    def forward_backbone(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        return features
    
    def forward_spm(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        spm_f = self.backbone.bb_spm_forward(images.tensor)
        return spm_f


@META_ARCH_REGISTRY.register()
class DAobjTwoStagePseudoLabGeneralizedRCNN_GC_shortcut(DAobjTwoStagePseudoLabGeneralizedRCNN_GC):
    def __init__(
        self,
        cfg
    ):










        super(DAobjTwoStagePseudoLabGeneralizedRCNN_GC_shortcut, self).__init__(cfg)
        

    
    
    
    
    
    
    
    
    
    
    
    
    

    def forward_backbone(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        return features