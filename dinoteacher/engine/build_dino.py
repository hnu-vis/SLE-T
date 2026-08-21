



import os
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(torch, "amp") and hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
    if not hasattr(torch.amp, "custom_fwd"):
        def _custom_fwd(*args, **kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k != "device_type"}
            return torch.cuda.amp.custom_fwd(*args, **kwargs)

        torch.amp.custom_fwd = _custom_fwd
    if not hasattr(torch.amp, "custom_bwd"):
        def _custom_bwd(*args, **kwargs):
            if args and callable(args[0]):
                return torch.cuda.amp.custom_bwd(args[0])
            return lambda fn: torch.cuda.amp.custom_bwd(fn)

        torch.amp.custom_bwd = _custom_bwd

import torchvision.transforms as T
import numpy as np
from dinov2.hub.backbones import dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14, dinov2_vitb14_reg, dinov2_vitl14_reg
from detectron2.modeling.backbone import Backbone
from detectron2.layers import ShapeSpec
from typing import Dict
from detectron2.structures import ImageList

import os
import time
from IPython import embed
from dinoteacher.modeling.meta_arch.heterogeneous_KD import ContextBlock, ContextBlock_stride, ContextBlock_lt
from dinov2.hub.backbones import dinov2_va_b16, dinov2_va_s16, \
    dinov2_va_lt_b14, dinov2_va_lt_b14_vgg, dinov2_va_lt_g14_vgg, dinov2_va_lt_l14_vgg
from dinov3.hub.backbones import dinov3_va_lt_b16_vgg

import torch._utils
try:
    torch._utils._rebuild_parameter_v2
except AttributeError:
    def _set_obj_state(obj, state):
        if isinstance(state, tuple):
            if not len(state) == 2:
                raise RuntimeError(f"Invalid serialized state: {state}")
            dict_state = state[0]
            slots_state = state[1]
        else:
            dict_state = state
            slots_state = None

        for k, v in dict_state.items():
            setattr(obj, k, v)

        if slots_state:
            for k, v in slots_state.items():
                setattr(obj, k, v)
        return obj
    def _rebuild_parameter_v2(data, requires_grad, backward_hooks, state):
        param = torch.nn.Parameter(data, requires_grad)
        param._backward_hooks = backward_hooks
        param = _set_obj_state(param, state)
        return param
    torch._utils._rebuild_parameter_v2 = _rebuild_parameter_v2


class dino_preprocessing():




    def __init__(self, pixel_mean, pixel_std):
        normalize = T.Normalize(mean=pixel_mean, std=pixel_std)        
        self.preprocessing_img = normalize

    def __call__(self, image):
        return self.preprocessing_img(image)

class DinoVitFeatureExtractor(Backbone):



    def __init__(self, cfg, model_name='dinov2_vits14', normalize_feature=True, freeze=True, image_format='BGR'):
        super(DinoVitFeatureExtractor, self).__init__()
        
        
        
        
        self.image_format = image_format
        pixel_mean = [123.675, 116.280, 103.530]        
        pixel_std = [58.395, 57.120, 57.375]            
        if(self.image_format == 'BGR'):
            pixel_mean = pixel_mean[::-1]
            pixel_std = pixel_std[::-1]
        self.preprocessing = dino_preprocessing(pixel_mean, pixel_std)

        self.normalize_feature = normalize_feature
        if ("v2" not in model_name) and ("v3" not in model_name):
            
            self.model_name = model_name
            local_dir = "dinoteacher/engine/dinov1/hub/facebookresearch_dino_main"
            self.encoder = torch.hub.load(local_dir, source='local', model=model_name, path=model_name)
            if freeze:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()                    
            self.embed_dim = self.encoder.embed_dim
            self.patch_size = int(model_name.rsplit('vit',1)[-1][1:])
            assert (cfg.INPUT.DINO_PATCH_SIZE == self.patch_size), f'Config patch size is {cfg.INPUT.DINO_PATCH_SIZE} while loaded model has a patch size of {self.patch_size}'
        else:
            dino_v2_models = {
                "dinov2_vits14": (14, 384, dinov2_vits14), 
                "dinov2_vitb14": (14, 768, dinov2_vitb14),
                "dinov2_vitl14": (14, 1024, dinov2_vitl14),
                "dinov2_vitg14": (14, 1536, dinov2_vitg14),
                "dinov2_vitb14_reg4": (14, 768, dinov2_vitb14_reg),
                "dinov2_vitl14_reg4": (14, 1024, dinov2_vitl14_reg),
                
            }
            
            name_to_weights = {"dinov2_vits14": "dinov2_vits14_pretrain.pth",
                            "dinov2_vitb14": "dinov2_vitb14_pretrain.pth",
                            "dinov2_vitl14": "dinov2_vitl14_pretrain.pth",
                            "dinov2_vitg14": "dinov2_vitg14_pretrain.pth",
                            "dinov2_vitb14_reg4": "dinov2_vitb14_reg4_pretrain.pth",
                            "dinov2_vitl14_reg4": "dinov2_vitl14_reg4_pretrain.pth",
                            
            }
            
            self.model_name = model_name
            assert (
                self.model_name in dino_v2_models.keys()
            ), f"class DinoV2VitFeatureExtractor(nn.Module): is only available for {dino_v2_models.keys()}"
            
            path_to_pretrained_weights = "weights/" + model_name + "_pretrain.pth"
            
            
            assert (
                os.path.exists(path_to_pretrained_weights)
            ), f"DINO pretrained model path {path_to_pretrained_weights} does not exist!"
            print(f"Model Path: {path_to_pretrained_weights}")
            
            patch_size, embed_dim, model_func_name = dino_v2_models[self.model_name]
            
            if patch_size == 16 and "v2" in model_name:
                img_size = 592
            else:
                img_size = 518
            self.encoder = model_func_name(pretrained=False, patch_size=patch_size, img_size=img_size)
            self.encoder.load_state_dict(torch.load(path_to_pretrained_weights),strict=False)
            if freeze:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()
            
            assert self.encoder.embed_dim == embed_dim
            self.embed_dim = self.encoder.embed_dim
            self.patch_size = patch_size
            self._out_features = ['dino_out']
            self._out_feature_channels = {'dino_out':self.encoder.blocks[-1].norm2.bias.shape[0]}
            self._out_feature_strides = {'dino_out':self.patch_size}


    def forward(self, x):
        
        
        
        if self.image_format == 'BGR':
            x = [torch.tensor(img['image'])[[2,1,0],:,:].float().to(device=next(self.encoder.parameters()).device) for img in x]
        elif self.image_format == 'RGB':
            x = [torch.tensor(img['image']).float().to(device=next(self.encoder.parameters()).device) for img in x]
        else:
            raise NotImplementedError()
        x = ImageList.from_tensors(x, self.patch_size).tensor
        x = self.preprocessing(x)
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

        return x_grid_features
    
    @property
    def size_divisibility(self) -> int:
        return self.patch_size

    @property
    def padding_constraints(self) -> Dict[str, int]:
        return {'size_divisibility': self.size_divisibility, 'square_size': 0}

    def output_shape(self):




        
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

class DinoVitFeatureExtractor_14_16(Backbone):



    def __init__(self, cfg, model_name='dinov2_vits14', normalize_feature=True):
        super(DinoVitFeatureExtractor_14_16, self).__init__()
        
        self.image_format = cfg.INPUT.FORMAT
        self.cfg = cfg
        pixel_mean = [123.675, 116.280, 103.530]        
        pixel_std = [58.395, 57.120, 57.375]            
        if(self.image_format == 'BGR'):
            pixel_mean = pixel_mean[::-1]
            pixel_std = pixel_std[::-1]
        self.preprocessing = dino_preprocessing(pixel_mean, pixel_std)

        self.normalize_feature = normalize_feature
        
        if(self.cfg.SEMISUPNET.GC_BLK == True):
            self.gcBlock = ContextBlock_lt(inplanes=cfg.SEMISUPNET.DINO_OUT_DIM, \
                out_channels=cfg.SEMISUPNET.DOWNSTREAM_OUT_DIM)
        
        if "v1" in model_name:
            
            self.model_name = model_name
            local_dir = "dinoteacher/engine/dinov1/hub/facebookresearch_dino_main"
            self.encoder = torch.hub.load(local_dir, source='local', model=model_name, path=model_name)
            if freeze:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()                    
            self.embed_dim = self.encoder.embed_dim
            self.patch_size = int(model_name.rsplit('vit',1)[-1][1:])
            assert (cfg.INPUT.DINO_PATCH_SIZE == self.patch_size), f'Config patch size is {cfg.INPUT.DINO_PATCH_SIZE} while loaded model has a patch size of {self.patch_size}'
        else:
            
            self.model_name = model_name
            
            dino_v2_models = {
                "dinov2_vits14": [14, 384, "dinov2_vits14"], 
                "dinov2_vits16": [16, 384, "dinov2_vits16"], 
                "dinov2_vitb14": [14, 768, "dinov2_vitb14"],
                "dinov2_vitl14": [14, 1024, "dinov2_vitl14"],
                "dinov2_vitg14": [14, 1536, "dinov2_vitg14"],
                "dinov3_vitb16": [16, 768, "dinov3_vitb16"],
                
                
                
                "dinov2_vitb14_14to16": [16, 768, "dinov2_vitb14_14to16"],
                "dinov2_vits14_14to16": [16, 384, "dinov2_vits14_14to16"],
            }
            dinov2_va_proj = {
                "dinov2_vits14" : "dinov2_vits14",
                "dinov2_vitb14_lt" : "dinov2_vitb14_lt",
                "dinov2_vitb14_lt_vgg" : "dinov2_vitb14_lt_vgg",
                "dinov2_vitl14_lt_vgg" : "dinov2_vitl14_lt_vgg",
                "dinov2_vitg14_lt_vgg" : "dinov2_vitg14_lt_vgg",
                "dinov3_vitb16_lt_vgg" : "dinov3_va_lt_b16_vgg",
                "dinov3_va_lt_b16_vgg" : "dinov3_va_lt_b16_vgg",
                "dinov2_vits14_lt" : "dinov2_vits14",
                "dinov2_vits14_14to16" : "dinov2_vits14_16",
                "dinov2_vits14_14to16_va" : "dinov2_va_s16",
                "dinov2_vitb14_14to16_va" : "dinov2_va_b16",
            }
            if self.model_name in dino_v2_models:
                
                if(cfg.SEMISUPNET.VA_TYPE == "raw"):
                    self.model_name = dino_v2_models[self.model_name][2] + '_va'
                
                
                elif cfg.SEMISUPNET.VA_TYPE == 'light':
                    self.model_name = dino_v2_models[self.model_name][2] + '_lt'
                
                else:
                    raise NotImplementedError()
                
                if(cfg.SEMISUPNET.SPM_TYPE == "vgg"):
                    self.model_name = self.model_name + '_vgg'
            
            
            vit_adapter_models = {
                "dinov2_vits14": (14, 384, dinov2_vits14),
                "dinov2_vitb14_lt": (16, 14, 768, dinov2_va_lt_b14),
                "dinov2_vitb14_lt_vgg": (32, 14, 768, dinov2_va_lt_b14_vgg),
                "dinov2_vitl14_lt_vgg": (32, 14, 1024, dinov2_va_lt_l14_vgg),
                "dinov2_vitg14_lt_vgg": (32, 14, 1536, dinov2_va_lt_g14_vgg),
                "dinov3_va_lt_b16_vgg": (32, 16, 768, dinov3_va_lt_b16_vgg),
                "dinov2_vits14_16": (16, 384, dinov2_vits14),
                "dinov2_va_b16": (16, 768, dinov2_va_b16),
                "dinov2_va_s16": (16, 384, dinov2_va_s16),
            }
            
            
            
            
            assert (
                self.model_name in dinov2_va_proj.keys()
            ), f"class DinoV2VitFeatureExtractor(nn.Module): is only available for {list(dino_v2_models.keys()) + list(dinov2_va_proj.keys())}"
            pretrained_weight_names = {
                "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
                "dinov3_vitb16_lt_vgg": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
                "dinov3_va_lt_b16_vgg": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            }
            path_to_pretrained_weights = os.path.join(
                "weights",
                pretrained_weight_names.get(model_name, pretrained_weight_names.get(self.model_name, model_name + "_pretrain.pth")),
            )
            
            
            assert (
                os.path.exists(path_to_pretrained_weights)
            ), f"DINO v2 pretrained model path {path_to_pretrained_weights} does not exist!"
            print(f"Model Path: {path_to_pretrained_weights}")
            
            feature_stride, patch_size, embed_dim, model_func_name = vit_adapter_models[dinov2_va_proj[self.model_name]]
            if(cfg.SEMISUPNET.FEATURE_STRIDE != None):
                feature_stride = cfg.SEMISUPNET.FEATURE_STRIDE
                
            
            if patch_size == 16 and (("v2" in model_name) or ("v3" in model_name)):
                img_size = 592
            else:
                img_size = 518
                
            self.encoder = model_func_name(patch_size=patch_size, img_size=img_size,
                                            interaction_indexes=cfg.SEMISUPNET.INTERACTION_INDEXES, 
                                            blk_groups=cfg.SEMISUPNET.BLK_GROUPS,
                                            feature_stride=feature_stride,
                                            add_vit_feature=cfg.SEMISUPNET.ADD_VIT_FEA,
                                            )
            
            if ((cfg.SEMISUPNET.SPM_TYPE == "vgg") and (cfg.SEMISUPNET.CKPT_PATH == None)):
                vgg_pretrained_weight_path = "weights/" + "vgg16_bn-6c64b313_converted.pth"
                self.encoder.spm.load_state_dict(torch.load(vgg_pretrained_weight_path), strict=False)
                print("="*50)
                print("Successfully loaded vgg pretrained weight...")
                print("="*50)
                
            
            if(cfg.SEMISUPNET.CKPT_PATH != None):
                ckpt_path = cfg.SEMISUPNET.CKPT_PATH
                full_state_dict = torch.load(ckpt_path, map_location='cuda')
                
                prefix = "modelTeacher.backbone."
                encoder_state_dict = {
                    k[len(prefix):]: v
                    for k, v in full_state_dict['model'].items()
                    if k.startswith(prefix)
                }
                
                
                
                self.load_state_dict(encoder_state_dict, strict=False)
                print("="*50)
                print("feature_extractor ckpt loaded successfully...")
                print("="*50)
            else:
                self.encoder.load_state_dict(torch.load(path_to_pretrained_weights, map_location='cpu'),strict=False)
            
            
            
            
            
            freeze = cfg.SEMISUPNET.FREEZE
                
            if freeze == 'freeze_DINO':
                if cfg.SEMISUPNET.DINO_TUNE_LAYER:
                    dino_tune_layers = [f"blocks.{i}." for i in cfg.SEMISUPNET.DINO_TUNE_LAYER]
                else:
                    dino_tune_layers = None
                dino_freeze_roots = [
                    'blocks', 'cls_token', 'mask_token', 'norm', 'patch_embed',
                    'pos_embed', 'rope_embed', 'storage_tokens', 'storage_token'
                ]
                for key, value in self.encoder.named_parameters():
                    if (key.split('.')[0] in dino_freeze_roots):
                        value.requires_grad = False
                        
                        
                        
                    
                    if ((dino_tune_layers is not None) and (any(layer in key for layer in dino_tune_layers))):
                        value.requires_grad = True
                        
                        
                    
                    
                
                for key, value in self.encoder.named_modules():
                    if (key.split('.')[0] in dino_freeze_roots):
                        value.eval()
                    
                    
                    
                    
                    
                    if ((dino_tune_layers is not None) and (any(layer in key for layer in dino_tune_layers))):
                        value.eval()
    
                        
            elif freeze == 'freeze_all':
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()
                
            elif freeze == 'No':
                pass
            
            else:
                raise NotImplementedError()
            
            
            assert self.encoder.embed_dim == embed_dim
            self.embed_dim = self.encoder.embed_dim
            self.patch_size = patch_size
            self._out_features = ['dino_out']
            self._out_feature_channels = {'dino_out':self.encoder.blocks[-1].norm2.bias.shape[0]}
            
            if (cfg.SEMISUPNET.GC_BLK == False):
                self._out_feature_channels = {'dino_out':self.encoder.blocks[-1].norm2.bias.shape[0]}
                self._out_feature_strides = {'dino_out':feature_stride}
            else:
                
                
                self._out_feature_channels = {'dino_out':self.cfg.SEMISUPNET.DOWNSTREAM_OUT_DIM}
                self._out_feature_strides = {'dino_out':feature_stride * 2}


    def forward(self, x):
        
        
        
        if self.image_format == 'BGR':
            x = [torch.tensor(img['image'])[[2,1,0],:,:].float().to(device=next(self.encoder.parameters()).device) for img in x]
        elif self.image_format == 'RGB':
            x = [torch.tensor(img['image']).float().to(device=next(self.encoder.parameters()).device) for img in x]
        else:
            raise NotImplementedError()
        
        x = ImageList.from_tensors(x, self.patch_size).tensor
        x = self.preprocessing(x)
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

            x_grid_features = x.contiguous().transpose(1, 2).contiguous().view(batch_size, self.embed_dim, f_height, f_width)
            
            out = x_grid_features

        elif (self.cfg.SEMISUPNET.VA_TYPE == 'light'):
            
            f1, f2, f3, f4 = self.encoder(x)

            out = f3

        else:
            raise NotImplementedError()
        
        if(self.cfg.SEMISUPNET.GC_BLK == True):
            out = self.gcBlock(out)
        
        if self.normalize_feature:
            out = F.normalize(out, p=2, dim=1)
        
        return out
    
    @property
    def size_divisibility(self) -> int:
        return self.patch_size

    @property
    def padding_constraints(self) -> Dict[str, int]:
        return {'size_divisibility': self.size_divisibility, 'square_size': 0}

    def output_shape(self):




        
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


class DinoVitFeatureExtractor_GC(Backbone):



    def __init__(self, cfg=None, model_name='dinov2_vits14', normalize_feature=True, freeze=True, is_BGR=True):
        super(DinoVitFeatureExtractor_GC, self).__init__()
        
        pixel_mean = [123.675, 116.280, 103.530]
        pixel_std = [58.395, 57.120, 57.375]
        self.preprocessing = dino_preprocessing(pixel_mean, pixel_std)

        self.is_BGR = is_BGR
        self.normalize_feature = normalize_feature
        
        self.gcBlock = ContextBlock(inplanes=cfg.SEMISUPNET.DINO_OUT_DIM, \
            ratio=1/16.0, out_channels=cfg.SEMISUPNET.DOWNSTREAM_OUT_DIM)
        
        if "v2" not in model_name:
            
            self.model_name = model_name
            local_dir = "dinoteacher/engine/dinov1/hub/facebookresearch_dino_main"
            self.encoder = torch.hub.load(local_dir, source='local', model=model_name, path=model_name)
            if freeze:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()

            self.embed_dim = self.encoder.embed_dim
            self.patch_size = int(model_name.rsplit('vit',1)[-1][1:])
            
        else:
            dino_v2_models = {
                "dinov2_vits14": (14, 384, dinov2_vits14), 
                "dinov2_vitb14": (14, 768, dinov2_vitb14),
                "dinov2_vitl14": (14, 1024, dinov2_vitl14),
                "dinov2_vitg14": (14, 1536, dinov2_vitg14),
                "dinov2_vitb14_reg4": (14, 768, dinov2_vitb14_reg),
                "dinov2_vitl14_reg4": (14, 1024, dinov2_vitl14_reg),
                
            }
            
            name_to_weights = {"dinov2_vits14": "dinov2_vits14_pretrain.pth",
                            "dinov2_vitb14": "dinov2_vitb14_pretrain.pth",
                            "dinov2_vitl14": "dinov2_vitl14_pretrain.pth",
                            "dinov2_vitg14": "dinov2_vitg14_pretrain.pth",
                            "dinov2_vitb14_reg4": "dinov2_vitb14_reg4_pretrain.pth",
                            "dinov2_vitl14_reg4": "dinov2_vitl14_reg4_pretrain.pth",
                            
            }
            
            self.model_name = model_name
            assert (
                self.model_name in dino_v2_models.keys()
            ), f"class DinoV2VitFeatureExtractor(nn.Module): is only available for {dino_v2_models.keys()}"
            path_to_pretrained_weights = "weights/" + model_name + "_pretrain.pth"
            assert (
                os.path.exists(path_to_pretrained_weights)
            ), f"DINO v2 pretrained model path {path_to_pretrained_weights} does not exist!"
            print(f"Model Path: {path_to_pretrained_weights}")
            
            patch_size, embed_dim, model_func_name = dino_v2_models[self.model_name]
            
            if patch_size == 16 and "v2" in model_name:
                img_size = 592
            else:
                img_size = 518  
            self.encoder = model_func_name(pretrained=False, patch_size=patch_size, img_size=img_size)
            self.encoder.load_state_dict(torch.load(path_to_pretrained_weights),strict=False)
            if freeze:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()

            gc_ckpt_path = cfg.SEMISUPNET.GC_CKPT_PATH
            full_state_dict = torch.load(gc_ckpt_path, map_location='cuda')
            
            prefix = "modelStudent.backbone.gcBlock."
            gc_state_dict = {
                k[len(prefix):]: v
                for k, v in full_state_dict['model'].items()
                if k.startswith(prefix)
            }
            
            
            
            print(self.gcBlock.channel_add_conv[0].weight[0][0])
            print(self.gcBlock.load_state_dict(gc_state_dict))
            print(self.gcBlock.channel_add_conv[0].weight[0][0])
            
            if freeze:
                for param in self.gcBlock.parameters():
                    param.requires_grad = False
                self.gcBlock.eval()
            
            
            assert self.encoder.embed_dim == embed_dim
            self.embed_dim = self.encoder.embed_dim
            self.patch_size = patch_size
            self._out_features = ['gc_out']
            self._out_feature_channels = {'gc_out':self.encoder.blocks[-1].norm2.bias.shape[0]}
            self._out_feature_strides = {'gc_out':self.patch_size}


    def forward(self, x):
        
        if self.is_BGR:
            x = [torch.tensor(img['image'])[[2,1,0],:,:].float().to(device=next(self.encoder.parameters()).device) for img in x]
        else:
            x = [torch.tensor(img['image']).float().to(device=next(self.encoder.parameters()).device) for img in x]
        x = ImageList.from_tensors(x, self.patch_size).tensor
        x = self.preprocessing(x)
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
            
        
        return x_grid_features
    
    @property
    def size_divisibility(self) -> int:
        return self.patch_size

    @property
    def padding_constraints(self) -> Dict[str, int]:
        return {'size_divisibility': self.size_divisibility, 'square_size': 0}

    def output_shape(self):




        
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }



class DinoVitAdapterFeatureExtractor(Backbone):



    def __init__(self, cfg, model_name='dinov2_vits14', normalize_feature=True, freeze='freeze_DINO', is_BGR=True):
        super(DinoVitAdapterFeatureExtractor, self).__init__()
        
        self.image_format = cfg.INPUT.FORMAT
        self.cfg = cfg
        pixel_mean = [123.675, 116.280, 103.530]
        pixel_std = [58.395, 57.120, 57.375]
        if(self.image_format == 'BGR'):
            raise NotImplementedError()
        self.preprocessing = dino_preprocessing(pixel_mean, pixel_std)

        self.is_BGR = is_BGR
        self.normalize_feature = normalize_feature
        if "v2" not in model_name:
            pass
            
            
            
            
            
            
            
            
            
            
            
        else:
            dino_v2_models = {
                
                
                
                
                
                
                "dinov2_vitb14_14to16": (16, 768, "dinov2_vitb14_14to16"),
                "dinov2_vits14_14to16": (16, 384, "dinov2_vits14_14to16"),
                
            }
            if cfg.SEMISUPNET.VA_TYPE == 'light':
                dino_v2_models[self.model_name][2] = dino_v2_models[self.model_name][2] + 'lt'
            
                
            dinov2_va_proj = {
                "dinov2_vitb14_14to16" : "dinov2_va_b16",
                "dinov2_vitb14_14to16_lt" : "dinov2_va_b16_lt",
                "dinov2_vits14_14to16" : "dinov2_va_s16",
            }
            
            vit_adapter_models = {
                "dinov2_va_b16": (16, 768, dinov2_va_b16),
                "dinov2_va_b16_lt": (16, 768, dinov2_va_lt_b16),
                "dinov2_va_s16": (16, 384, dinov2_va_s16),
            }
            
            name_to_weights = {"dinov2_vits14": "dinov2_vits14_pretrain.pth",
                            "dinov2_vitb14": "dinov2_vitb14_pretrain.pth",
                            "dinov2_vitl14": "dinov2_vitl14_pretrain.pth",
                            "dinov2_vitg14": "dinov2_vitg14_pretrain.pth",
                            "dinov2_vitb14_reg4": "dinov2_vitb14_reg4_pretrain.pth",
                            "dinov2_vitl14_reg4": "dinov2_vitl14_reg4_pretrain.pth",
                            
            }
            
            self.model_name = model_name
            assert (
                self.model_name in dino_v2_models.keys()
            ), f"class DinoV2VitFeatureExtractor(nn.Module): is only available for {dino_v2_models.keys()}"
            path_to_pretrained_weights = "weights/" + model_name + "_pretrain.pth"
            
            
            assert (
                os.path.exists(path_to_pretrained_weights)
            ), f"DINO v2 pretrained model path {path_to_pretrained_weights} does not exist!"
            print(f"Model Path: {path_to_pretrained_weights}")
            
            
            
            patch_size, embed_dim, model_func_name = vit_adapter_models[dinov2_va_proj[self.model_name]]
            
            if patch_size == 16 and "v2" in model_name:
                img_size = 592
            else:
                img_size = 518
            
            
            
            
            
            
            self.encoder = model_func_name(patch_size=patch_size, img_size=img_size)
            
            
            self.encoder.load_state_dict(torch.load(path_to_pretrained_weights),strict=False)
            
            
            if freeze == 'freeze_DINO':
                for key, value in self.encoder.named_parameters():
                    if (key.split('.')[0] in ['blocks', 'cls_token', 'mask_token', 'norm', 'patch_embed', 'pos_embed']):
                        value.requires_grad = False
                        
                        
                        
                
                for key, value in self.encoder.named_modules():
                    if (key.split('.')[0] in ['blocks', 'cls_token', 'mask_token', 'norm', 'patch_embed', 'pos_embed']):
                        value.eval()
                        
            elif freeze == 'freeze_all':
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()
                
            elif freeze == 'No':
                pass
            
            else:
                raise NotImplementedError()
                
            
            assert self.encoder.embed_dim == embed_dim
            self.embed_dim = self.encoder.embed_dim
            self.patch_size = patch_size
            self._out_features = ['va_out_s16']
            self._out_feature_channels = {'va_out_s16':self.encoder.norm3.bias.shape[0]}
            self._out_feature_strides = {'va_out_s16':self.patch_size}


    def forward(self, x):
        
        if self.image_format == 'BGR':
            x = [torch.tensor(img['image'])[[2,1,0],:,:].float().to(device=next(self.encoder.parameters()).device) for img in x]
        elif self.image_format == 'RGB':
            x = [torch.tensor(img['image']).float().to(device=next(self.encoder.parameters()).device) for img in x]
        else:
            raise NotImplementedError()
        x = ImageList.from_tensors(x, self.patch_size).tensor
        x = self.preprocessing(x)
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

        assert (f_height == f3.shape[2])
        assert (f_width == f3.shape[3])

        
        return f3
    
    @property
    def size_divisibility(self) -> int:
        return self.patch_size

    @property
    def padding_constraints(self) -> Dict[str, int]:
        return {'size_divisibility': self.size_divisibility, 'square_size': 0}

    def output_shape(self):




        
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
