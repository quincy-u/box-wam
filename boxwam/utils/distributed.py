"""Minimal single-node distributed launch: one process per listed CUDA device."""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank: int, world_size: int, device: str, port: int, fn: Callable, args: tuple) -> None:
    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
    try:
        fn(*args, rank=rank, world_size=world_size, device=torch_device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def launch(fn: Callable, devices: list[str], *args, port: int = 29500) -> None:
    """Run ``fn(*args, rank=, world_size=, device=)`` on every device, in parallel processes."""
    world_size = len(devices)
    if world_size == 1:
        _worker(0, 1, devices[0], port, fn, args)
        return
    mp.spawn(_spawn_entry, args=(world_size, devices, port, fn, args), nprocs=world_size, join=True)


def _spawn_entry(rank: int, world_size: int, devices: list[str], port: int, fn: Callable, args: tuple) -> None:
    _worker(rank, world_size, devices[rank], port, fn, args)


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor
