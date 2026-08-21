






















import copy
from typing import List, Optional

import dinov3.distributed as distributed
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


import torchvision
from torch import Tensor


def reduce_dict(input_dict, average=True):








    world_size = distributed.get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    
    if tensor_list[0].ndim == 3:
        
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    def __len__(self):
        return len(self.tensors)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):

    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    





    return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
        norm_type,
    )
    return total_norm


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def get_param_dict(model, args, return_name=False, use_layerwise_decay=False):
    
    for n, p in model.named_parameters():
        if match_name_keywords(n, args.lr_backbone_names) and match_name_keywords(n, args.lr_linear_proj_names):
            raise ValueError

    param_dicts = [
        {
            "params": [
                p if not return_name else n
                for n, p in model.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and not match_name_keywords(n, args.wd_norm_names)
                and p.requires_grad
            ],
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p if not return_name else n
                for n, p in model.named_parameters()
                if match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and not match_name_keywords(n, args.wd_norm_names)
                and p.requires_grad
            ],
            "lr": args.lr_backbone,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p if not return_name else n
                for n, p in model.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and match_name_keywords(n, args.lr_linear_proj_names)
                and not match_name_keywords(n, args.wd_norm_names)
                and p.requires_grad
            ],
            "lr": args.lr * args.lr_linear_proj_mult,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p if not return_name else n
                for n, p in model.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and match_name_keywords(n, args.wd_norm_names)
                and p.requires_grad
            ],
            "lr": args.lr,
            "weight_decay": args.weight_decay * args.wd_norm_mult,
        },
        {
            "params": [
                p if not return_name else n
                for n, p in model.named_parameters()
                if match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and match_name_keywords(n, args.wd_norm_names)
                and p.requires_grad
            ],
            "lr": args.lr_backbone,
            "weight_decay": args.weight_decay * args.wd_norm_mult,
        },
        {
            "params": [
                p if not return_name else n
                for n, p in model.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and match_name_keywords(n, args.lr_linear_proj_names)
                and match_name_keywords(n, args.wd_norm_names)
                and p.requires_grad
            ],
            "lr": args.lr * args.lr_linear_proj_mult,
            "weight_decay": args.weight_decay * args.wd_norm_mult,
        },
    ]
    return param_dicts


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):

    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
