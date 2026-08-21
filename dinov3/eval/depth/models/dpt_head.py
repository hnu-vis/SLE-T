




import torch
from torch import nn


def kaiming_init(
    module: nn.Module,
    a: float = 0,
    mode: str = "fan_out",
    nonlinearity: str = "relu",
    bias: float = 0,
    distribution: str = "normal",
) -> None:
    assert distribution in ["uniform", "normal"]
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class ConvModule(nn.Module):
















































    _abbr_ = "conv_block"

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        inplace=True,
        with_spectral_norm=False,
        padding_mode="zeros",
        order=("conv", "norm", "act"),
    ):
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ["zeros", "circular"]
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {"conv", "norm", "act"}

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        
        
        
        

        
        conv_padding = 0 if self.with_explicit_padding else padding
        
        self.conv = nn.Conv2d(  
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        
        if self.with_norm:
            
            if order.index("norm") > order.index("conv"):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            
            
            self.add_module("bn", torch.nn.SyncBatchNorm(norm_channels))
            
            
            
            
            self.norm_name = "bn"
        else:
            self.norm_name = None  

        
        if self.with_activation:
            act_cfg_ = act_cfg.copy()  
            
            if act_cfg_["type"] not in ["Tanh", "PReLU", "Sigmoid", "HSigmoid", "Swish", "GELU"]:
                act_cfg_.setdefault("inplace", inplace)
            self.activate = nn.ReLU()  

        
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        
        
        
        
        
        
        
        
        
        if not hasattr(self.conv, "init_weights"):
            if self.with_activation and self.act_cfg["type"] == "LeakyReLU":
                nonlinearity = "leaky_relu"
                a = self.act_cfg.get("negative_slope", 0.01)
            else:
                nonlinearity = "relu"
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm_name, 1, bias=0)

    def forward(self, x: torch.Tensor, activate: bool = True, norm: bool = True, debug: bool = False) -> torch.Tensor:
        for layer in self.order:
            if debug:
                breakpoint()
            if layer == "conv":
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x


class UpConvHead(nn.Module):


















    def __init__(self, features, n_output_channels, n_hidden_channels=32):
        super(UpConvHead, self).__init__()
        self.n_output_channels = n_output_channels
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, n_hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_hidden_channels, n_output_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.head(x)
        return x


class ReassembleBlocks(nn.Module):











    def __init__(
        self,
        in_channels=[1024, 1024, 1024, 1024],
        out_channels=[128, 256, 512, 1024],
        readout_type="project",
        use_batchnorm=False,
    ):
        super(ReassembleBlocks, self).__init__()

        assert readout_type in ["ignore", "add", "project"]
        self.readout_type = readout_type

        self.projects = nn.ModuleList(
            [
                ConvModule(
                    in_channels=in_channels[channel_index],
                    out_channels=out_channel,
                    kernel_size=1,
                    act_cfg=None,
                )
                for channel_index, out_channel in enumerate(out_channels)
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )
        if self.readout_type == "project":
            self.readout_projects = nn.ModuleList()
            for i in range(len(self.projects)):
                self.readout_projects.append(nn.Sequential(nn.Linear(2 * in_channels[i], in_channels[i]), nn.GELU()))

        self.batchnorm_layers = nn.ModuleList(
            [nn.SyncBatchNorm(channel) if use_batchnorm else nn.Identity(channel) for channel in in_channels]
        )

    def forward(self, inputs):
        assert isinstance(inputs, list)
        out = []
        for i, x in enumerate(inputs):
            assert len(x) == 2
            x, cls_token = x[0], x[1]
            feature_shape = x.shape
            if self.readout_type == "project":
                x = x.flatten(2).permute((0, 2, 1))
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
                x = x.permute(0, 2, 1).reshape(feature_shape)
            elif self.readout_type == "add":
                x = x.flatten(2) + cls_token.unsqueeze(-1)
                x = x.reshape(feature_shape)
            else:
                pass
            x = self.batchnorm_layers[i](x)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        return out


class PreActResidualConvUnit(nn.Module):










    def __init__(self, in_channels, act_cfg, norm_cfg, stride=1, dilation=1, init_cfg=None):
        super(PreActResidualConvUnit, self).__init__()  
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False,
            order=("act", "conv", "norm"),
        )
        self.conv2 = ConvModule(
            in_channels,
            in_channels,
            3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False,
            order=("act", "conv", "norm"),
        )

    def forward(self, inputs):
        inputs_ = inputs.clone()
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x + inputs_


class FeatureFusionBlock(nn.Module):












    def __init__(self, in_channels, act_cfg, norm_cfg, expand=False, align_corners=True, init_cfg=None):
        super(FeatureFusionBlock, self).__init__()  
        self.in_channels = in_channels
        self.expand = expand
        self.align_corners = align_corners
        self.out_channels = in_channels
        if self.expand:
            self.out_channels = in_channels // 2
        self.project = ConvModule(self.in_channels, self.out_channels, kernel_size=1, act_cfg=None, bias=True)
        self.res_conv_unit1 = PreActResidualConvUnit(in_channels=self.in_channels, act_cfg=act_cfg, norm_cfg=norm_cfg)
        self.res_conv_unit2 = PreActResidualConvUnit(in_channels=self.in_channels, act_cfg=act_cfg, norm_cfg=norm_cfg)

    def forward(self, *inputs):
        x = inputs[0]

        if len(inputs) == 2:
            if x.shape != inputs[1].shape:
                res = torch.nn.functional.interpolate(
                    inputs[1],
                    size=(x.shape[2], x.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                res = inputs[1]
            x = x + self.res_conv_unit1(res)
        x = self.res_conv_unit2(x)  

        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        

        x = self.project(x)  
        return x


class DPTHead(nn.Module):














    def __init__(
        self,
        in_channels=(1024, 1024, 1024, 1024),
        channels=256,
        post_process_channels=[128, 256, 512, 1024],
        readout_type="project",
        expand_channels=False,
        n_output_channels=256,
        use_batchnorm=False,  
        **kwargs,
    ):
        super(DPTHead, self).__init__(**kwargs)
        self.channels = channels
        self.n_output_channels = n_output_channels
        self.in_channels = in_channels
        self.expand_channels = expand_channels
        self.norm_cfg = None  
        self.reassemble_blocks = ReassembleBlocks(
            in_channels=in_channels,
            out_channels=post_process_channels,
            readout_type=readout_type,
            use_batchnorm=use_batchnorm,
        )

        self.post_process_channels = [
            channel * (2**i) if expand_channels else channel for i, channel in enumerate(post_process_channels)
        ]
        self.convs = nn.ModuleList()
        for channel in self.post_process_channels:
            self.convs.append(ConvModule(channel, self.channels, kernel_size=3, padding=1, act_cfg=None, bias=False))
        self.fusion_blocks = nn.ModuleList()
        self.act_cfg = {"type": "ReLU"}
        for _ in range(len(self.convs)):
            self.fusion_blocks.append(FeatureFusionBlock(self.channels, self.act_cfg, self.norm_cfg))
        self.fusion_blocks[0].res_conv_unit1 = None
        self.project = ConvModule(self.channels, self.channels, kernel_size=3, padding=1, norm_cfg=self.norm_cfg)
        self.num_fusion_blocks = len(self.fusion_blocks)
        self.num_reassemble_blocks = len(self.reassemble_blocks.resize_layers)
        self.num_post_process_channels = len(self.post_process_channels)
        assert self.num_fusion_blocks == self.num_reassemble_blocks
        assert self.num_reassemble_blocks == self.num_post_process_channels
        self.conv_depth = UpConvHead(self.channels, self.n_output_channels)

    def forward_features(self, inputs):
        assert len(inputs) == self.num_reassemble_blocks, (
            f"Expected {self.num_reassemble_blocks} inputs, got {len(inputs)}."
        )
        x = [inp for inp in inputs]

        x = self.reassemble_blocks(x)
        x = [self.convs[i](feature) for i, feature in enumerate(x)]
        out = self.fusion_blocks[0](x[-1])

        for i in range(1, len(self.fusion_blocks)):
            out = self.fusion_blocks[i](out, x[-(i + 1)])

        out = self.project(out)
        return out

    def forward(self, inputs):
        out = self.forward_features(inputs)
        return self.conv_depth(out)

    def predict(self, inputs, rescale_to=(512, 512)):
        out = self.forward_features(inputs)
        return self.conv_depth.predict(out, rescale_to=rescale_to)
