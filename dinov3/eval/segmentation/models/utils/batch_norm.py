





import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn import BatchNorm2d

import dinov3.distributed as distributed


class FrozenBatchNorm2d(nn.Module):



















    _version = 3

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)
        self.register_buffer("num_batches_tracked", None)

    def forward(self, x):
        if x.requires_grad:
            
            
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype  
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            
            
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            
            
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):













        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
            res.num_batches_tracked = module.num_batches_tracked
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res

    @classmethod
    def convert_frozenbatchnorm2d_to_batchnorm2d(cls, module: nn.Module) -> nn.Module:













        res = module
        if isinstance(module, FrozenBatchNorm2d):
            res = torch.nn.BatchNorm2d(module.num_features, module.eps)

            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data.clone().detach()
            res.running_var.data = module.running_var.data.clone().detach()
            res.eps = module.eps
            res.num_batches_tracked = module.num_batches_tracked
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozenbatchnorm2d_to_batchnorm2d(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


def get_norm(norm, out_channels):









    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm2d,
            
            "SyncBN": nn.SyncBatchNorm,
            "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            
            "nnSyncBN": nn.SyncBatchNorm,
            "naiveSyncBN": NaiveSyncBatchNorm,
            
            "naiveSyncBN_N": lambda channels: NaiveSyncBatchNorm(channels, stats_mode="N"),
            "LN": lambda channels: LayerNorm(channels),
        }[norm]
    return norm(out_channels)


class NaiveSyncBatchNorm(BatchNorm2d):



























    def __init__(self, *args, stats_mode="", **kwargs):
        super().__init__(*args, **kwargs)
        assert stats_mode in ["", "N"]
        self._stats_mode = stats_mode

    def forward(self, input):
        if distributed.get_world_size() == 1 or not self.training:
            return super().forward(input)

        B, C = input.shape[0], input.shape[1]

        half_input = input.dtype == torch.float16
        if half_input:
            
            input = input.float()
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        if self._stats_mode == "":
            assert B > 0, 'SyncBatchNorm(stats_mode="") does not support zero batch size.'
            vec = torch.cat([mean, meansqr], dim=0)
            vec = torch.distributed.nn.all_reduce(vec) * (1.0 / dist.get_world_size())
            mean, meansqr = torch.split(vec, C)
            momentum = self.momentum
        else:
            if B == 0:
                vec = torch.zeros([2 * C + 1], device=mean.device, dtype=mean.dtype)
                vec = vec + input.sum()  
            else:
                vec = torch.cat(
                    [
                        mean,
                        meansqr,
                        torch.ones([1], device=mean.device, dtype=mean.dtype),
                    ],
                    dim=0,
                )
            vec = torch.distributed.nn.all_reduce(vec * B)

            total_batch = vec[-1].detach()
            momentum = total_batch.clamp(max=1) * self.momentum  
            mean, meansqr, _ = torch.split(vec / total_batch.clamp(min=1), C)  

        var = meansqr - mean * mean
        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)

        self.running_mean += momentum * (mean.detach() - self.running_mean)
        self.running_var += momentum * (var.detach() - self.running_var)
        ret = input * scale + bias
        if half_input:
            ret = ret.half()
        return ret


class CycleBatchNormList(nn.ModuleList):















    def __init__(self, length: int, bn_class=nn.BatchNorm2d, **kwargs):






        self._affine = kwargs.pop("affine", True)
        super().__init__([bn_class(**kwargs, affine=False) for k in range(length)])
        if self._affine:
            
            channels = self[0].num_features
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))
        self._pos = 0

    def forward(self, x):
        ret = self[self._pos](x)
        self._pos = (self._pos + 1) % len(self)

        if self._affine:
            w = self.weight.reshape(1, -1, 1, 1)
            b = self.bias.reshape(1, -1, 1, 1)
            return ret * w + b
        else:
            return ret

    def extra_repr(self):
        return f"affine={self._affine}"


class LayerNorm(nn.Module):







    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
