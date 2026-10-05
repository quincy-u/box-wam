"""Typed configuration for training. Loaded from YAML; unknown keys are rejected.

See ``configs/`` for complete examples. Every section maps to one dataclass below.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from boxwam.data.augmentation import AugmentationConfig
from boxwam.model.encoder import DEFAULT_ENCODER


@dataclass
class DataConfig:
    episodes: str  # episode list file (one directory per line) or a root directory to scan
    stats: str  # .npz with action/state percentiles, see scripts/compute_action_stats.py
    cameras: list[str]  # camera names: video_<camera>.npy files, in the order fed to the model
    gripper_dims: list[int]  # action dimensions kept absolute (not relative to the state)
    task: str | None = None  # keep only episodes whose path contains this substring
    use_valid_starts: bool = False  # sample window starts from valid_idx.npy
    val_fraction: float = 0.02
    preload: bool = False  # hold the whole dataset in memory
    num_workers: int = 8
    batch_size: int = 128  # per GPU


@dataclass
class ModelConfig:
    state_dim: int
    action_dim: int
    horizon: int = 15  # number of action (and box) steps predicted per chunk
    box_dim: int = 16  # 16 per tracked object (8 corners x 2 pixel coordinates)
    action_attend_box: bool = True  # action tokens read the predicted box trajectory
    box_attend_action: bool = False  # box tokens read the action chunk
    encoder: str = DEFAULT_ENCODER
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 16
    drop_path: float = 0.1
    activation_checkpointing: bool = False
    compile: bool = True


@dataclass
class LossConfig:
    action_weight: float = 2.0
    box_weight: float = 1.0
    action_exponent: float = 2.0  # action velocity error is |v - v*|^p / p
    timestep_scale: float = 0.999  # t = scale * (1 - u)
    timestep_beta: tuple[float, float] = (1.5, 1.0)  # u ~ Beta(a, b); (1, 1) is uniform
    val_steps: int = 10  # Euler steps for validation rollouts


@dataclass
class OptimConfig:
    epochs: int
    steps_per_epoch: int = 300
    lr: float = 1e-4
    start_lr: float = 5.25e-5  # linear warmup starts here
    final_lr: float = 1e-4  # cosine decays to here (equal to lr -> constant after warmup)
    warmup_epochs: float = 5
    schedule_scale: float = 1.25  # cosine period as a multiple of the run length
    weight_decay: float = 0.04
    final_weight_decay: float = 0.04
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip: float = 1.0


@dataclass
class TrainConfig:
    run_name: str
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    output_dir: str = "checkpoints"  # checkpoints and logs go to <output_dir>/<run_name>/
    seed: int = 239
    dtype: str = "bfloat16"  # autocast dtype for the predictor: bfloat16 | float16 | float32
    save_every: int = 5  # keep a copy of the weights every N epochs (besides latest/best)
    init_from: str | None = None  # checkpoint whose weights initialise the model (training starts at epoch 0)
    resume: str | None = None  # checkpoint to continue from (weights, optimizer state and epoch counter)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name


# ------------------------------------------------------------------ loading
def _convert(value: Any, annotation: Any, path: str) -> Any:
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Union or (origin is not None and str(origin) == "<class 'types.UnionType'>"):
        if value is None:
            return None
        annotation = next(arg for arg in args if arg is not type(None))
        return _convert(value, annotation, path)
    if dataclasses.is_dataclass(annotation):
        return from_dict(annotation, value, path)
    if origin is tuple:
        return tuple(_convert(item, args[0], f"{path}[{index}]") for index, item in enumerate(value))
    if origin is list:
        return [_convert(item, args[0], f"{path}[{index}]") for index, item in enumerate(value)]
    if annotation is float:
        return float(value)
    if annotation is int:
        if isinstance(value, float) and not value.is_integer():
            raise TypeError(f"{path}: expected an integer, got {value}")
        return int(value)
    if annotation is bool and not isinstance(value, bool):
        raise TypeError(f"{path}: expected a boolean, got {value!r}")
    return value


def from_dict(cls, data: dict[str, Any], path: str = "") -> Any:
    """Build dataclass ``cls`` from a nested dict, rejecting unknown or missing keys."""
    if not isinstance(data, dict):
        raise TypeError(f"{path or cls.__name__}: expected a mapping, got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise KeyError(f"{path or cls.__name__}: unknown keys {unknown}; valid keys are {sorted(fields)}")
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for name, f in fields.items():
        key = f"{path}.{name}" if path else name
        if name in data:
            kwargs[name] = _convert(data[name], hints[name], key)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise KeyError(f"{key} is required")
    return cls(**kwargs)


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides (values parsed as YAML) to a config dict."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like section.key=value")
        dotted, raw = item.split("=", 1)
        node = data
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return data


def load_train_config(path: str | Path, overrides: list[str] | None = None) -> TrainConfig:
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    data = apply_overrides(data, overrides or [])
    cfg = from_dict(TrainConfig, data)
    if len(cfg.data.gripper_dims) and max(cfg.data.gripper_dims) >= cfg.model.action_dim:
        raise ValueError("data.gripper_dims must index dimensions of model.action_dim")
    return cfg


def to_dict(cfg: Any) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
