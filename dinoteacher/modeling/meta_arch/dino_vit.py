import torch
from torch.nn import functional as F
from typing import Dict, Tuple, List, Optional, Union, Callable
from detectron2.structures import ImageList, Instances
from dinoteacher.engine.build_dino import DinoVitFeatureExtractor, DinoVitAdapterFeatureExtractor, DinoVitFeatureExtractor_14_16
from detectron2.modeling.backbone import BACKBONE_REGISTRY

from .heterogeneous_KD import ContextBlock, ContextBlock_stride, ContextBlock_lt
from IPython import embed

from .adv_grl import GradReverse, FCDiscriminator_img

@BACKBONE_REGISTRY.register()
def build_dino_vit_backbone(cfg, _):
    return DinoVitFeatureExtractor_wrapper(cfg)


class DinoVitFeatureExtractor_wrapper(DinoVitFeatureExtractor):
    def __init__(self, cfg, output_layer='dino_out'):
        if 'dino' in cfg.MODEL.BACKBONE.NAME and cfg.SEMISUPNET.DINO_BBONE_LR_SCALE:
            freeze = False
        else:
            freeze = True
        super(DinoVitFeatureExtractor_wrapper, self).__init__(cfg, model_name=cfg.SEMISUPNET.DINO_BBONE_MODEL, 
                                            normalize_feature=False, freeze=freeze, image_format=cfg.INPUT.FORMAT)
        self.output_layer = output_layer
        
        if(cfg.INPUT.FORMAT != 'RGB'):
            raise NotImplementedError()
        
        
        self.cfg = cfg

    def forward(self, x):
        
        
        if(self.cfg.INPUT.FORMAT == 'BGR'):
            x = x[:,[2,1,0],:,:]
        elif(self.cfg.INPUT.FORMAT == 'RGB'):
            pass
            
            
        else:
            raise NotImplementedError()
        batch_size, _, height, width = x.size()
        
        assert (height % self.patch_size) == 0
        assert (width % self.patch_size) == 0
        f_height = height // self.patch_size
        f_width = width // self.patch_size

        x = self.encoder.get_intermediate_layers(x)[0] 
        if "v2" not in self.model_name:
            x = x[:,1:,:] 

        if self.normalize_feature:
            x = F.normalize(x, p=2, dim=2)      

        x_grid_features = x.contiguous().transpose(1, 2).contiguous().view(batch_size, self.embed_dim, f_height, f_width)

        return {self.output_layer: x_grid_features}

@BACKBONE_REGISTRY.register()
def build_dino_vit_backbone_14_16(cfg, _):
    return DinoVitFeatureExtractor_wrapper_14_16(cfg)

class DinoVitFeatureExtractor_wrapper_14_16(DinoVitFeatureExtractor_14_16):
    def __init__(self, cfg, output_layer='dino_out'):
        freeze = cfg.SEMISUPNET.FREEZE
        super(DinoVitFeatureExtractor_wrapper_14_16, self).__init__(cfg, model_name=cfg.SEMISUPNET.DINO_BBONE_MODEL, 
                                                    normalize_feature=False)
        self.output_layer = output_layer
        self.cfg = cfg
        
        if(cfg.SEMISUPNET.GC_BLK == True):
            self.gcBlock = ContextBlock_lt(inplanes=cfg.SEMISUPNET.DINO_OUT_DIM, \
            out_channels=cfg.SEMISUPNET.DOWNSTREAM_OUT_DIM)
            
            
        
        if(cfg.INPUT.FORMAT != 'RGB'):
            raise NotImplementedError()
        
        if(cfg.SEMISUPNET.SPM_GRL == True):
            self.D_img = FCDiscriminator_img(self._out_feature_channels['dino_out']) 
    
    def bb_spm_forward(self, x):
        
        
        
        if(self.cfg.INPUT.FORMAT == 'BGR'):
            x = x[:,[2,1,0],:,:]
        elif(self.cfg.INPUT.FORMAT == 'RGB'):
            pass
        else:
            raise NotImplementedError()
        
        c = self.encoder.spm_forward(x)
        
        return c

    def forward(self, x):
        
        
        if(self.cfg.INPUT.FORMAT == 'BGR'):
            x = x[:,[2,1,0],:,:]
        elif(self.cfg.INPUT.FORMAT == 'RGB'):
            pass
        else:
            raise NotImplementedError()
        batch_size, _, height, width = x.size()
        
        assert (height % self.patch_size) == 0
        assert (width % self.patch_size) == 0
        f_height = height // self.patch_size
        f_width = width // self.patch_size
        
        out = None
        if(self.cfg.SEMISUPNET.VA_TYPE == 'None'):

            x = self.encoder.get_intermediate_layers(x)[0] 
            if "v2" not in self.model_name:
                x = x[:,1:,:] 

            if self.normalize_feature:
                x = F.normalize(x, p=2, dim=2)      

            x_grid_features = x.contiguous().transpose(1, 2).contiguous().view(batch_size, self.embed_dim, f_height, f_width)

            out = x_grid_features

        elif (self.cfg.SEMISUPNET.VA_TYPE == 'light'):
            
            
            
            f1, f2, f3, f4 = self.encoder(x)
        
            if self.normalize_feature:
                f1 = F.normalize(f1, p=2, dim=1)
                f2 = F.normalize(f2, p=2, dim=1)
                f3 = F.normalize(f3, p=2, dim=1)
                f4 = F.normalize(f4, p=2, dim=1)
        
            out = f3

        else:
            raise NotImplementedError()
        
        if(self.cfg.SEMISUPNET.GC_BLK == True):
            
            
            return {self.output_layer: self.gcBlock(out)}
        
        else:
            return {self.output_layer: out}


@BACKBONE_REGISTRY.register()
def build_dino_vit_backbone_gc(cfg, _):
    return DinoVitFeatureExtractor_gc_wrapper(cfg)


class DinoVitFeatureExtractor_gc_wrapper(DinoVitFeatureExtractor):
    def __init__(self, cfg, output_layer='gc_out'):
        if 'dino' in cfg.MODEL.BACKBONE.NAME and cfg.SEMISUPNET.DINO_BBONE_LR_SCALE:
            freeze = False
        else:
            freeze = True
        super(DinoVitFeatureExtractor_gc_wrapper, self).__init__(cfg, model_name=cfg.SEMISUPNET.DINO_BBONE_MODEL, normalize_feature=False, freeze=freeze, image_format=cfg.INPUT.FORMAT)
        self.output_layer = output_layer
        self.gcBlock = ContextBlock(inplanes=cfg.SEMISUPNET.DINO_OUT_DIM, ratio=1/16.0, out_channels=cfg.SEMISUPNET.DOWNSTREAM_OUT_DIM)

        self._out_features = ['gc_out']
        
        self._out_feature_channels = {'gc_out':cfg.SEMISUPNET.DOWNSTREAM_OUT_DIM}
        
        self._out_feature_strides = {'gc_out':self.patch_size}
        
        self.cfg = cfg
        
    def forward(self, x):
        
        
        if(self.cfg.INPUT.FORMAT == 'BGR'):
            x = x[:,[2,1,0],:,:]
        elif(self.cfg.INPUT.FORMAT == 'RGB'):
            pass
        else:
            raise NotImplementedError()
        batch_size, _, height, width = x.size()
        
        assert (height % self.patch_size) == 0
        assert (width % self.patch_size) == 0
        f_height = height // self.patch_size
        f_width = width // self.patch_size

        x = self.encoder.get_intermediate_layers(x)[0] 
        if "v2" not in self.model_name:
            x = x[:,1:,:] 

        
        

        
        
        x_grid_features = x.contiguous().transpose(1, 2).contiguous().view(batch_size, self.embed_dim, f_height, f_width)   

        x_grid_features = self.gcBlock(x_grid_features)

        
        if self.normalize_feature:
            x_grid_features = F.normalize(x_grid_features, p=2, dim=1)
        
        return {self.output_layer: x_grid_features}

@BACKBONE_REGISTRY.register()
def build_dino_vit_adapter_backbone(cfg, _):
    return DinoVitAdapterFeatureExtractor_wrapper(cfg)


'''
Now the wrapper is designed for Dinov2, 

TODO: 
    1. MUlti scale (FPN)
    2. Dinov3
'''

class DinoVitAdapterFeatureExtractor_wrapper(DinoVitAdapterFeatureExtractor):
    def __init__(self, cfg):
        
        
        
        
        freeze = cfg.SEMISUPNET.FREEZE
        super(DinoVitAdapterFeatureExtractor_wrapper, self).__init__(cfg, 
            model_name=cfg.SEMISUPNET.DINO_BBONE_MODEL, normalize_feature=False, freeze=freeze, is_BGR=cfg.INPUT.FORMAT)
        
        
        self.output_layer = 'va_out_s16'
        
        self._out_features = ['va_out_s16']
        self._out_feature_channels = {'va_out_s16':cfg.SEMISUPNET.DOWNSTREAM_OUT_DIM}
        self._out_feature_strides = {'va_out_s16':self.patch_size}
        
    def forward(self, x):
        x = x[:,[2,1,0],:,:]
        batch_size, _, height, width = x.size()
        
        assert (height % self.patch_size) == 0
        assert (width % self.patch_size) == 0
        f_height = height // self.patch_size
        f_width = width // self.patch_size
        
        
        f1, f2, f3, f4 = self.encoder(x)
        
        if self.normalize_feature:
            f1 = F.normalize(f1, p=2, dim=1)
            f2 = F.normalize(f2, p=2, dim=1)
            f3 = F.normalize(f3, p=2, dim=1)
            f4 = F.normalize(f4, p=2, dim=1)
        
        
        
        
        
        
        return {self.output_layer: f3}
        
        
        
        
        