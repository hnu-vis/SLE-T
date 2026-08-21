




from enum import Enum
from typing import Union

import torch

from .utils import _DINOV2_BASE_URL, _make_dinov2_model_name

from .cls_vit_adapter import ViTAdapter_dino, ViTAdapter_dino_lt, ViTAdapter_dino_lt_norm, ViTAdapter_dino_lt_norm_vgg

from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from functools import partial


class Weights(Enum):
    LVD142M = "LVD142M"


def _make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    pretrained: bool = False,
    weights: Union[Weights, str] = Weights.LVD142M,
    **kwargs,
):
    from ..models import vision_transformer as vits

    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    model_base_name = _make_dinov2_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained:
        model_full_name = _make_dinov2_model_name(arch_name, patch_size, num_register_tokens)
        url = _DINOV2_BASE_URL + f"/{model_base_name}/{model_full_name}_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    return model



dinov2_info ={
    "dinov2_va_s16" : (384, 12, 6, 4, 0.2, 6, 1.0),
    "dinov2_va_b16" : (768, 12, 12, 4, 0.3, 12, 0.5),
    "dinov2_va_l16" : (1024, 24, 16, 4, 0.4, 16, 0.5),
    "dinov2_va_g14" : (1536, 40, 24, 4, 0.4, 16, 0.5),                  
}



'''
interaction_indexes are different for different type of dinov2 backbone.
'''



def dinov2_va_b16(patch_size=16, img_size=592, interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]], **kwargs):
    embed_dim, depth, num_heads, mlp_ratio, drop_path_rate, deform_num_heads, deform_ratio = dinov2_info['dinov2_va_b16']
    
    va_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=1.0,   
        ffn_layer= "mlp",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias= False,
        interpolate_offset= 0.1,
        
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        deform_num_heads=deform_num_heads,
        deform_ratio=deform_ratio,
        block_fn=partial(Block, attn_class=MemEffAttention),
        interaction_indexes=interaction_indexes,
    )
    
    va_kwargs.update(**kwargs)
    vit_adapter_dino = ViTAdapter_dino(**va_kwargs)
    
    return vit_adapter_dino




def dinov2_va_lt_b14(patch_size=16, img_size=592, 
        interaction_indexes=[[11, 11]], blk_groups=[0, 10], feature_stride=None, add_vit_feature=True, **kwargs):
    embed_dim, depth, num_heads, mlp_ratio, drop_path_rate, deform_num_heads, deform_ratio = dinov2_info['dinov2_va_b16']
    
    va_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=1.0,   
        ffn_layer= "mlp",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias= False,
        interpolate_offset= 0.1,
        
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        deform_num_heads=deform_num_heads,
        deform_ratio=deform_ratio,
        block_fn=partial(Block, attn_class=MemEffAttention),
        interaction_indexes=interaction_indexes,
        blk_groups=blk_groups,
        feature_stride=feature_stride,
        add_vit_feature=add_vit_feature,
    )
    
    va_kwargs.update(**kwargs)
    vit_adapter_dino = ViTAdapter_dino_lt_norm(**va_kwargs)
    
    return vit_adapter_dino


def dinov2_va_lt_b14_vgg(patch_size=16, img_size=592, 
        interaction_indexes=[[11, 11]], blk_groups=[0, 10], feature_stride=None, add_vit_feature=True, **kwargs):
    embed_dim, depth, num_heads, mlp_ratio, drop_path_rate, deform_num_heads, deform_ratio = dinov2_info['dinov2_va_b16']
    
    va_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=1.0,   
        ffn_layer= "mlp",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias= False,
        interpolate_offset= 0.1,
        
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        deform_num_heads=deform_num_heads,
        deform_ratio=deform_ratio,
        block_fn=partial(Block, attn_class=MemEffAttention),
        interaction_indexes=interaction_indexes,
        blk_groups=blk_groups,
        feature_stride=feature_stride,
        add_vit_feature=add_vit_feature,
    )
    
    va_kwargs.update(**kwargs)
    vit_adapter_dino = ViTAdapter_dino_lt_norm_vgg(**va_kwargs)
    
    return vit_adapter_dino

def dinov2_va_lt_l14_vgg(patch_size=16, img_size=592, 
        interaction_indexes=[[23, 23]], blk_groups=[0, 22], feature_stride=None, add_vit_feature=True, **kwargs):
    embed_dim, depth, num_heads, mlp_ratio, drop_path_rate, deform_num_heads, deform_ratio = dinov2_info['dinov2_va_l16']
    
    va_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=1.0,   
        ffn_layer= "mlp",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias= False,
        interpolate_offset= 0.1,
        
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        deform_num_heads=deform_num_heads,
        deform_ratio=deform_ratio,
        block_fn=partial(Block, attn_class=MemEffAttention),
        interaction_indexes=interaction_indexes,
        blk_groups=blk_groups,
        feature_stride=feature_stride,
        add_vit_feature=add_vit_feature,
    )
    
    va_kwargs.update(**kwargs)
    vit_adapter_dino = ViTAdapter_dino_lt_norm_vgg(**va_kwargs)
    
    return vit_adapter_dino


def dinov2_va_lt_g14_vgg(patch_size=16, img_size=592, 
        interaction_indexes=[[39, 39]], blk_groups=[0, 38], feature_stride=None, add_vit_feature=True, **kwargs):
    embed_dim, depth, num_heads, mlp_ratio, drop_path_rate, deform_num_heads, deform_ratio = dinov2_info['dinov2_va_g14']
    
    va_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=1.0,   
        ffn_layer= "mlp",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias= False,
        interpolate_offset= 0.1,
        
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        deform_num_heads=deform_num_heads,
        deform_ratio=deform_ratio,
        block_fn=partial(Block, attn_class=MemEffAttention),
        interaction_indexes=interaction_indexes,
        blk_groups=blk_groups,
        feature_stride=feature_stride,
        add_vit_feature=add_vit_feature,
    )
    
    va_kwargs.update(**kwargs)
    vit_adapter_dino = ViTAdapter_dino_lt_norm_vgg(**va_kwargs)
    
    return vit_adapter_dino

def dinov2_va_s16(patch_size=16, img_size=592, interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]], add_vit_feature=True, **kwargs):
    embed_dim, depth, num_heads, mlp_ratio, drop_path_rate, deform_num_heads, deform_ratio = dinov2_info['dinov2_va_s16']
    
    va_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=1.0,   
        ffn_layer= "mlp",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias= False,
        interpolate_offset= 0.1,
        
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        deform_num_heads=deform_num_heads,
        deform_ratio=deform_ratio,
        block_fn=partial(Block, attn_class=MemEffAttention),
        interaction_indexes= interaction_indexes,
        add_vit_feature=add_vit_feature,
    )
    va_kwargs.update(**kwargs)
    vit_adapter_dino = ViTAdapter_dino_lt_norm(**va_kwargs)
    
    return vit_adapter_dino

def dinov2_vits14(*, pretrained: bool = False, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):



    return _make_dinov2_model(arch_name="vit_small", pretrained=pretrained, weights=weights, **kwargs)


def dinov2_vitb14(*, pretrained: bool = False, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):



    return _make_dinov2_model(arch_name="vit_base", pretrained=pretrained, weights=weights, **kwargs)


def dinov2_vitl14(*, pretrained: bool = False, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):



    return _make_dinov2_model(arch_name="vit_large", pretrained=pretrained, weights=weights, **kwargs)


def dinov2_vitg14(*, pretrained: bool = False, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):



    return _make_dinov2_model(
        arch_name="vit_giant2",
        ffn_layer="swiglufused",
        weights=weights,
        pretrained=pretrained,
        **kwargs,
    )


def dinov2_vits14_reg(*, pretrained: bool = False, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):



    return _make_dinov2_model(
        arch_name="vit_small",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitb14_reg(*, pretrained: bool = False, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):



    return _make_dinov2_model(
        arch_name="vit_base",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitl14_reg(*, pretrained: bool = False, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):



    return _make_dinov2_model(
        arch_name="vit_large",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitg14_reg(*, pretrained: bool = False, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):



    return _make_dinov2_model(
        arch_name="vit_giant2",
        ffn_layer="swiglufused",
        weights=weights,
        pretrained=pretrained,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )
