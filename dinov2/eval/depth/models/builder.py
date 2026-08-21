




import warnings

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
from mmcv.utils import Registry

MODELS = Registry("models", parent=MMCV_MODELS)
ATTENTION = Registry("attention", parent=MMCV_ATTENTION)


BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
DEPTHER = MODELS


def build_backbone(cfg):

    return BACKBONES.build(cfg)


def build_neck(cfg):

    return NECKS.build(cfg)


def build_head(cfg):

    return HEADS.build(cfg)


def build_loss(cfg):

    return LOSSES.build(cfg)


def build_depther(cfg, train_cfg=None, test_cfg=None):

    if train_cfg is not None or test_cfg is not None:
        warnings.warn("train_cfg and test_cfg is deprecated, " "please specify them in model", UserWarning)
    assert cfg.get("train_cfg") is None or train_cfg is None, "train_cfg specified in both outer field and model field "
    assert cfg.get("test_cfg") is None or test_cfg is None, "test_cfg specified in both outer field and model field "
    return DEPTHER.build(cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
