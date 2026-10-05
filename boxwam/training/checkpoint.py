"""Checkpoint files.

A Box-WAM checkpoint is a ``torch.save`` dict with::

    model      predictor state dict
    config     the TrainConfig it was trained with (as a dict)
    stats      action/state statistics used for normalisation (dict of arrays)
    epoch      number of completed epochs
    optimizer  optional optimizer state (training checkpoints only)
    metrics    optional validation metrics of this epoch

``config`` and ``stats`` make a checkpoint self-contained for inference: ``BoxWAMPolicy``
rebuilds the model from them without any other file.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from boxwam.data.normalization import ActionStats
from boxwam.model.boxwam import BoxWAM, load_predictor_state


def unwrap(model: nn.Module) -> BoxWAM:
    return model.module if isinstance(model, DistributedDataParallel) else model


def stats_to_dict(stats: ActionStats) -> dict:
    """Plain Python lists, so checkpoints load under any NumPy version."""
    return {key: getattr(stats, key).tolist() for key in ActionStats.REQUIRED_KEYS} | {
        "gripper_dims": list(stats.gripper_dims)
    }


def stats_from_dict(data: dict) -> ActionStats:
    arrays = {key: np.asarray(data[key], dtype=np.float32) for key in ActionStats.REQUIRED_KEYS}
    return ActionStats(gripper_dims=tuple(int(index) for index in data["gripper_dims"]), **arrays)


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    config,
    stats: ActionStats,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    metrics: dict | None = None,
) -> None:
    payload = {
        "model": unwrap(model).state_dict(),
        "config": dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(config),
        "stats": stats_to_dict(stats),
        "epoch": epoch,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if metrics:
        payload["metrics"] = metrics
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model_weights(model: nn.Module, checkpoint: dict) -> None:
    load_predictor_state(unwrap(model), checkpoint["model"])
