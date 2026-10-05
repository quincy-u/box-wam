"""AdamW with decoupled parameter groups and warmup + cosine schedules."""

from __future__ import annotations

import math

import torch

from boxwam.config import OptimConfig


def build_optimizer(model: torch.nn.Module, cfg: OptimConfig) -> torch.optim.AdamW:
    """Weight decay applies to matrices only; biases and LayerNorm gains are excluded."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if name.endswith("bias") or parameter.ndim == 1 else decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay, "decay": True},
        {"params": no_decay, "weight_decay": 0.0, "decay": False},
    ]
    return torch.optim.AdamW(groups, lr=cfg.start_lr, betas=cfg.betas, eps=cfg.eps)


class Schedules:
    """Per-step learning rate (linear warmup, then cosine) and cosine weight decay.

    ``step`` counts optimizer updates starting at 1. The cosine period is
    ``schedule_scale * epochs * steps_per_epoch``, so with the default 1.25 the schedule never
    reaches its end within the run; with ``final_lr == lr`` the rate is constant after warmup.
    """

    def __init__(self, cfg: OptimConfig):
        self.cfg = cfg
        self.warmup_steps = int(cfg.warmup_epochs * cfg.steps_per_epoch)
        self.total_steps = int(cfg.schedule_scale * cfg.epochs * cfg.steps_per_epoch)

    def lr(self, step: int) -> float:
        cfg = self.cfg
        if step < self.warmup_steps:
            return cfg.start_lr + (cfg.lr - cfg.start_lr) * step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return max(cfg.final_lr, cfg.final_lr + (cfg.lr - cfg.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)))

    def weight_decay(self, step: int) -> float:
        cfg = self.cfg
        value = cfg.final_weight_decay + (cfg.weight_decay - cfg.final_weight_decay) * 0.5 * (
            1.0 + math.cos(math.pi * step / self.total_steps)
        )
        low, high = sorted((cfg.weight_decay, cfg.final_weight_decay))
        return min(max(value, low), high)

    def apply(self, optimizer: torch.optim.Optimizer, step: int) -> tuple[float, float]:
        lr, weight_decay = self.lr(step), self.weight_decay(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
            if group.get("decay"):
                group["weight_decay"] = weight_decay
        return lr, weight_decay
