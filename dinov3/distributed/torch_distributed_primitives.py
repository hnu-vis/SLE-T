




from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.nn import functional as F

from .torch_distributed_wrapper import get_default_process_group, get_world_size


def reduce_dict(input_dict: Dict[Any, torch.Tensor], average: bool = True) -> Dict[Any, torch.Tensor]:









    world_size = get_world_size()
    if world_size <= 1:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        stacked_values = torch.stack(values, dim=0)
        dist.all_reduce(stacked_values)
        if average:
            stacked_values /= world_size
        reduced_dict = {k: v for k, v in zip(names, stacked_values)}
    return reduced_dict


def _simple_gather_all_tensors(result: torch.Tensor, group: Any, world_size: int) -> List[torch.Tensor]:
    gathered_result = [torch.zeros_like(result) for _ in range(world_size)]
    dist.all_gather(gathered_result, result, group)
    return gathered_result


def gather_all_tensors(result: torch.Tensor, group: Optional[Any] = None) -> List[torch.Tensor]:














    if group is None:
        group = get_default_process_group()

    
    result = result.contiguous()

    world_size = get_world_size()
    dist.barrier(group=group)

    
    if result.ndim == 0:
        return _simple_gather_all_tensors(result, group, world_size)

    
    local_size = torch.tensor(result.shape, device=result.device)
    local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(local_sizes, local_size, group=group)
    max_size = torch.stack(local_sizes).max(dim=0).values
    all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

    
    if all_sizes_equal:
        return _simple_gather_all_tensors(result, group, world_size)

    
    pad_dims = []
    pad_by = (max_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    result_padded = F.pad(result, pad_dims)
    gathered_result = [torch.zeros_like(result_padded) for _ in range(world_size)]
    dist.all_gather(gathered_result, result_padded, group)
    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    return gathered_result
