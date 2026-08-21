
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch
from typing import Union, List, Dict, Any, cast
from detectron2.modeling.backbone import (
    ResNet,
    Backbone,
    build_resnet_backbone,
    BACKBONE_REGISTRY
)
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool, LastLevelP6P7

from IPython import embed

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs: Dict[str, List[Union[str, int]]] = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class vgg_backbone(Backbone):


















    def __init__(self, cfg):
        super().__init__()

        self.vgg = make_layers(cfgs['vgg16'],batch_norm=True)

        self._initialize_weights()
        
        _out_feature_channels = [64, 128, 256, 512, 512]
        _out_feature_strides = [2, 4, 8, 16, 32]
        
        
        
        
        

        

        self.stages = [nn.Sequential(*list(self.vgg._modules.values())[0:7]),\
                    nn.Sequential(*list(self.vgg._modules.values())[7:14]),\
                    nn.Sequential(*list(self.vgg._modules.values())[14:24]),\
                    nn.Sequential(*list(self.vgg._modules.values())[24:34]),\
                    nn.Sequential(*list(self.vgg._modules.values())[34:]),]
        self._out_feature_channels = {}
        self._out_feature_strides = {}
        self._stage_names = []

        for i, stage in enumerate(self.stages):
            name = "vgg{}".format(i)
            self.add_module(name, stage)
            self._stage_names.append(name)
            self._out_feature_channels[name] = _out_feature_channels[i]
            self._out_feature_strides[name] = _out_feature_strides[i]

        self._out_features = self._stage_names

        del self.vgg
        
        if(cfg is not None):
            if(("vgg16" in cfg.MODEL.WEIGHTS) and ("prefix" not in cfg.MODEL.WEIGHTS)):
                checkpoint = torch.load(cfg.MODEL.WEIGHTS, map_location='cuda')
                self.load_state_dict(checkpoint)
        

    def forward(self, x):
        features = {}
        for name, stage in zip(self._stage_names, self.stages):
            x = stage(x)
            
            
            features[name] = x
        
        

        return features

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
                

class vgg_backbone_lc(Backbone):


















    def __init__(self, cfg):
        super().__init__()

        self.vgg = make_layers(cfgs['vgg16'],batch_norm=True)

        
        _out_feature_channels = [64, 128, 256, 512, 512]
        _out_feature_strides = [2, 4, 8, 16, 32]
        
        
        
        
        

        

        self.stages = [nn.Sequential(*list(self.vgg._modules.values())[0:7]),\
                    nn.Sequential(*list(self.vgg._modules.values())[7:14]),\
                    nn.Sequential(*list(self.vgg._modules.values())[14:24]),\
                    nn.Sequential(*list(self.vgg._modules.values())[24:34]),\
                    nn.Sequential(*list(self.vgg._modules.values())[34:]),]
        self._out_feature_channels = {}
        self._out_feature_strides = {}
        self._stage_names = []

        for i, stage in enumerate(self.stages):
            name = "vgg{}".format(i)
            self.add_module(name, stage)
            self._stage_names.append(name)
            self._out_feature_channels[name] = _out_feature_channels[i]
            self._out_feature_strides[name] = _out_feature_strides[i]

         
        self.vgg3_proj = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.GELU()
        )
        
        
        self.vgg4_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.vgg4_proj = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.GELU()
        )
        
        
        self.fusion = nn.Sequential(
            nn.Conv2d(512 + 512, 768, 3, padding=1),
            nn.BatchNorm2d(768),
            nn.GELU(),
            nn.Conv2d(768, 768, 1)
        )
        
        
        self.skip = nn.Conv2d(512 + 512, 768, 1) if (512 + 512) != 768 else nn.Identity()

        self._out_feature_channels["vgg_s16_fusion"] = 768
        self._out_feature_strides["vgg_s16_fusion"] = 16
        self._out_features = self._stage_names + ["vgg_s16_fusion"]

        self._initialize_weights()

        del self.vgg
        
        if(cfg is not None):
            if(("vgg16" in cfg.MODEL.WEIGHTS) and ("prefix" not in cfg.MODEL.WEIGHTS)):
                checkpoint = torch.load(cfg.MODEL.WEIGHTS, map_location='cuda')
                self.load_state_dict(checkpoint, strict=False)
        

    def forward(self, x):
        features = {}
        for name, stage in zip(self._stage_names, self.stages):
            x = stage(x)
            
            
            features[name] = x
        
        
        vgg3_feat = features["vgg3"]
        vgg4_feat = features["vgg4"]

        
        
        
        vgg3_proj = self.vgg3_proj(vgg3_feat)
        vgg4_up = self.vgg4_upsample(vgg4_feat)
        if vgg4_up.shape[-2:] != vgg3_feat.shape[-2:]:
            vgg4_up = F.interpolate(
                vgg4_up,
                size=vgg3_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        vgg4_proj = self.vgg4_proj(vgg4_up)

        concat = torch.cat([vgg3_proj, vgg4_proj], dim=1)
        features["vgg_s16_fusion"] = self.fusion(concat) + self.skip(concat)

        return features

    def _initialize_weights(self) -> None:
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                
                if 'fusion.3' in name:  
                    nn.init.constant_(m.weight, 0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


@BACKBONE_REGISTRY.register() 
def build_vgg_backbone(cfg, _):
    return vgg_backbone(cfg)

@BACKBONE_REGISTRY.register()
def build_vgg_lc_backbone(cfg, _):
    return vgg_backbone_lc(cfg)


@BACKBONE_REGISTRY.register() 
def build_vgg_fpn_backbone(cfg, _):
    
    
    
    
    
    
    

    bottom_up = vgg_backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        
    )
    

    return backbone
