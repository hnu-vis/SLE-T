




import logging
import os
import random
import socket
import subprocess
from datetime import timedelta
from enum import Enum
from typing import List, Sequence

import torch
import torch.distributed as dist

logger = logging.getLogger("dinov3")

_DEFAULT_PROCESS_GROUP = None
_PROCESS_SUBGROUP = None
_BUILTIN_PRINT = None


def is_distributed_enabled() -> bool:




    return dist.is_available() and dist.is_initialized()


def get_rank(group=None) -> int:




    if not is_distributed_enabled():
        return 0
    return dist.get_rank(group=group)


def get_world_size(group=None) -> int:




    if not is_distributed_enabled():
        return 1
    return dist.get_world_size(group=group)


def is_main_process(group=None) -> bool:




    return get_rank(group) == 0


def save_in_main_process(*args, **kwargs) -> None:

    group = kwargs.pop("group", None)
    if not is_main_process(group):
        return
    torch.save(*args, **kwargs)


def _restrict_print_to_main_process() -> None:

    import builtins as __builtin__

    global _BUILTIN_PRINT
    _BUILTIN_PRINT = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_main_process() or force:
            _BUILTIN_PRINT(*args, **kwargs)

    __builtin__.print = print


def _get_master_port(seed: int = 0) -> int:
    MIN_MASTER_PORT, MAX_MASTER_PORT = (20_000, 60_000)

    master_port_str = os.environ.get("MASTER_PORT")
    if master_port_str is None:
        rng = random.Random(seed)
        return rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)

    return int(master_port_str)


def _get_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        
        
        s.bind(("", 0))
        port = s.getsockname()[1]
        return port


def _parse_slurm_node_list(s: str) -> List[str]:
    return subprocess.check_output(["scontrol", "show", "hostnames", s], text=True).splitlines()


class JobType(Enum):
    TORCHELASTIC = "TorchElastic"
    SLURM = "Slurm"
    MANUAL = "manual"


class TorchDistributedEnvironment:








    def __init__(self):
        if "TORCHELASTIC_RUN_ID" in os.environ:
            
            self.job_id = os.environ["TORCHELASTIC_RUN_ID"]
            self.job_type = JobType.TORCHELASTIC

            self.master_addr = os.environ["MASTER_ADDR"]
            self.master_port = int(os.environ["MASTER_PORT"])
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        elif "SLURM_JOB_ID" in os.environ:
            
            self.job_id = int(os.environ["SLURM_JOB_ID"])
            self.job_type = JobType.SLURM

            node_count = int(os.environ["SLURM_JOB_NUM_NODES"])
            nodes = _parse_slurm_node_list(os.environ["SLURM_JOB_NODELIST"])
            assert len(nodes) == node_count

            self.master_addr = nodes[0]
            self.master_port = _get_master_port(seed=self.job_id)
            self.rank = int(os.environ["SLURM_PROCID"])
            self.world_size = int(os.environ["SLURM_NTASKS"])
            self.local_rank = int(os.environ["SLURM_LOCALID"])
            self.local_world_size = self.world_size // node_count
        else:
            
            self.job_id = None
            self.job_type = JobType.MANUAL

            self.master_addr = "127.0.0.1"
            self.master_port = _get_available_port()
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.local_world_size = 1

        assert self.rank < self.world_size
        assert self.local_rank < self.local_world_size

    def export(
        self,
        *,
        overwrite: bool,
        nccl_async_error_handling: bool = False,
    ) -> "TorchDistributedEnvironment":
        
        
        
        env_vars = {
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(self.master_port),
            "RANK": str(self.rank),
            "WORLD_SIZE": str(self.world_size),
            "LOCAL_RANK": str(self.local_rank),
            "LOCAL_WORLD_SIZE": str(self.local_world_size),
        }
        if nccl_async_error_handling:
            env_vars.update(
                {
                    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",  
                }
            )

        if not overwrite:
            for k, v in env_vars.items():
                
                if k not in os.environ:
                    continue
                if os.environ[k] == v:
                    continue
                raise RuntimeError(f"Cannot export environment variables as {k} is already set")

        os.environ.update(env_vars)
        return self

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def __str__(self):
        return (
            f"{self.job_type.value} job "
            + (f"({self.job_id}) " if self.job_id else "")
            + f"using {self.master_addr}:{self.master_port} "  
            f"(rank={self.rank}, world size={self.world_size})"
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"master_addr={self.master_addr},"  
            f"master_port={self.master_port},"  
            f"rank={self.rank},"  
            f"world_size={self.world_size},"  
            f"local_rank={self.local_rank},"  
            f"local_world_size={self.local_world_size}"
            ")"
        )


def enable_distributed(
    *,
    set_cuda_current_device: bool = True,
    overwrite: bool = False,
    nccl_async_error_handling: bool = False,
    restrict_print_to_main_process: bool = True,
    timeout: timedelta | None = None,
):















    global _DEFAULT_PROCESS_GROUP

    if _DEFAULT_PROCESS_GROUP is not None:
        raise RuntimeError("Distributed mode has already been enabled")

    torch_env = TorchDistributedEnvironment()
    logger.info(f"PyTorch distributed environment: {torch_env}")
    torch_env.export(
        overwrite=overwrite,
        nccl_async_error_handling=nccl_async_error_handling,
    )

    if set_cuda_current_device:
        torch.cuda.set_device(torch_env.local_rank)

    dist.init_process_group(backend="nccl", timeout=timeout)
    dist.barrier()

    if restrict_print_to_main_process:
        _restrict_print_to_main_process()

    
    _DEFAULT_PROCESS_GROUP = torch.distributed.group.WORLD


def get_default_process_group():
    return _DEFAULT_PROCESS_GROUP


def disable_distributed() -> None:
    global _BUILTIN_PRINT
    if _BUILTIN_PRINT is not None:
        import builtins as __builtin__

        __builtin__.print = _BUILTIN_PRINT

    global _PROCESS_SUBGROUP
    
    if _PROCESS_SUBGROUP is not None:
        torch.distributed.destroy_process_group(_PROCESS_SUBGROUP)
        _PROCESS_SUBGROUP = None

    global _DEFAULT_PROCESS_GROUP
    if _DEFAULT_PROCESS_GROUP is not None:  
        torch.distributed.destroy_process_group(_DEFAULT_PROCESS_GROUP)
        _DEFAULT_PROCESS_GROUP = None


def new_subgroups(all_subgroup_ranks: Sequence[Sequence[int]]):










    all_ranks = tuple(rank for subgroup_ranks in all_subgroup_ranks for rank in subgroup_ranks)
    rank = get_rank()
    assert len(all_ranks) == len(set(all_ranks))
    assert rank in all_ranks

    global _PROCESS_SUBGROUP
    assert _PROCESS_SUBGROUP is None

    for subgroup_ranks in all_subgroup_ranks:
        subgroup = torch.distributed.new_group(subgroup_ranks)
        if rank in subgroup_ranks:
            _PROCESS_SUBGROUP = subgroup


def get_process_subgroup():




    return _PROCESS_SUBGROUP or _DEFAULT_PROCESS_GROUP


def get_subgroup_rank() -> int:




    return get_rank(group=get_process_subgroup())


def get_subgroup_size() -> int:




    return get_world_size(group=get_process_subgroup())


def is_subgroup_main_process() -> bool:




    return get_rank(group=get_process_subgroup()) == 0
