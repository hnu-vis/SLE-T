
import torch
import torch.nn as nn
from detectron2.modeling.backbone import Backbone, BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from IPython import embed

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(
            channels,
            channels * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class resnet50_backbone(Backbone):











    def __init__(self, cfg):
        super().__init__()

        self.in_channels = 64
        self.res0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.res1 = self._make_layer(64, 3)
        self.res2 = self._make_layer(128, 4, stride=2)
        self.res3 = self._make_layer(256, 6, stride=2)
        self.res4 = self._make_layer(512, 3, stride=2)

        self.stages = [
            self.res0,
            self.res1,
            self.res2,
            self.res3,
            self.res4,
        ]
        _out_feature_channels = [64, 256, 512, 1024, 2048]
        _out_feature_strides = [4, 4, 8, 16, 32]

        self._out_feature_channels = {}
        self._out_feature_strides = {}
        self._stage_names = []

        for i, stage in enumerate(self.stages):
            name = "res{}".format(i)
            self._stage_names.append(name)
            self._out_feature_channels[name] = _out_feature_channels[i]
            self._out_feature_strides[name] = _out_feature_strides[i]

        self._out_features = self._stage_names
        self._initialize_weights()

        if cfg is not None:
            weights = cfg.MODEL.WEIGHTS
            if weights and ("resnet" in weights.lower()) and ("prefix" not in weights):
                checkpoint = torch.load(weights, map_location="cpu")
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    checkpoint = checkpoint["model"]
                self.load_state_dict(checkpoint, strict=False)

    def _make_layer(self, channels, blocks, stride=1):
        downsample = None
        out_channels = channels * Bottleneck.expansion
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = [Bottleneck(self.in_channels, channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        features = {}
        for name, stage in zip(self._stage_names, self.stages):
            x = stage(x)
            features[name] = x
        
        
        return features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


@BACKBONE_REGISTRY.register()
def build_resnet50_backbone(cfg, _):
    return resnet50_backbone(cfg)


@BACKBONE_REGISTRY.register()
def build_resnet50_fpn_backbone(cfg, _):
    bottom_up = resnet50_backbone(cfg)
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
