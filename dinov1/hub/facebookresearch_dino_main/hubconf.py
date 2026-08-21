












import torch
from torchvision.models.resnet import resnet50

import vision_transformer as vits

dependencies = ["torch", "torchvision"]


def dino_vits16(pretrained=True, **kwargs):




    model = vits.__dict__["vit_small"](patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_vits8(pretrained=True, **kwargs):




    model = vits.__dict__["vit_small"](patch_size=8, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_vitb16(pretrained=True, **kwargs):




    model = vits.__dict__["vit_base"](patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_vitb8(pretrained=True, **kwargs):




    model = vits.__dict__["vit_base"](patch_size=8, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_resnet50(pretrained=True, **kwargs):




    model = resnet50(pretrained=False, **kwargs)
    model.fc = torch.nn.Identity()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=False)
    return model


def dino_xcit_small_12_p16(pretrained=True, **kwargs):



    model = torch.hub.load('facebookresearch/xcit:main', "xcit_small_12_p16", num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_xcit_small_12_p8(pretrained=True, **kwargs):



    model = torch.hub.load('facebookresearch/xcit:main', "xcit_small_12_p8", num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_xcit_medium_24_p16(pretrained=True, **kwargs):



    model = torch.hub.load('facebookresearch/xcit:main', "xcit_medium_24_p16", num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_xcit_medium_24_p8(pretrained=True, **kwargs):



    model = torch.hub.load('facebookresearch/xcit:main', "xcit_medium_24_p8", num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model
