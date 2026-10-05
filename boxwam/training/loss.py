"""Batch preparation, the rectified-flow training loss and validation rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from boxwam.config import LossConfig
from boxwam.data.augmentation import AugmentationConfig, augment
from boxwam.data.normalization import ActionStats, normalize_boxes
from boxwam.model.boxwam import BoxWAM
from boxwam.model.encoder import DinoPatchEncoder
from boxwam.model.flow import integrate, interpolate, sample_timesteps


@dataclass
class Batch:
    patches: torch.Tensor  # (B, num_patches, encoder_dim) frozen DINOv3 features
    state_raw: torch.Tensor  # (B, state_dim)
    state: torch.Tensor  # normalised state
    actions_raw: torch.Tensor  # (B, H, action_dim) absolute targets
    actions: torch.Tensor  # normalised relative targets
    boxes: torch.Tensor | None  # (B, H, box_dim) normalised future boxes

    @property
    def size(self) -> int:
        return self.state.shape[0]


@torch.no_grad()
def prepare_batch(
    raw: dict,
    *,
    encoder: DinoPatchEncoder,
    stats: ActionStats,
    augmentation: AugmentationConfig,
    box_dim: int,
    device: torch.device,
) -> Batch:
    """Augment, encode and normalise one data-loader batch."""
    images = [image.to(device, non_blocking=True) for image in raw["images"]]
    boxes_px = raw["boxes"].to(device, non_blocking=True) if box_dim else None
    images, boxes_px = augment(images, boxes_px, augmentation)
    patches = torch.cat([encoder(image) for image in images], dim=1)
    state_raw = raw["state"].to(device, non_blocking=True).float()
    actions_raw = raw["actions"].to(device, non_blocking=True).float()
    if actions_raw.shape[-1] != stats.action_dim:
        raise ValueError(f"action.npy has {actions_raw.shape[-1]} dims but the statistics have {stats.action_dim}")
    boxes = None
    if box_dim:
        if boxes_px.shape[-1] < box_dim:
            raise ValueError(
                f"proj_3d_obb.npy provides {boxes_px.shape[-1]} box values per step, model needs {box_dim}"
            )
        boxes = normalize_boxes(boxes_px[..., :box_dim], augmentation.crop_size)
    return Batch(
        patches=patches,
        state_raw=state_raw,
        state=stats.normalize_state(state_raw),
        actions_raw=actions_raw,
        actions=stats.normalize_actions(actions_raw, state_raw),
        boxes=boxes,
    )


def flow_matching_loss(
    model: BoxWAM, batch: Batch, cfg: LossConfig, *, amp_dtype: torch.dtype, amp: bool
) -> tuple[torch.Tensor, dict[str, float]]:
    """Joint rectified-flow loss over actions and (optionally) boxes.

    The action term is ``mean |v - v*|^p / p`` and the box term is a mean squared error;
    both regress the velocity ``target - noise`` at a shared random timestep. The two terms are
    combined with ``action_weight`` and ``box_weight`` and normalised by their sum.
    """
    t = sample_timesteps(batch.size, device=batch.state.device, scale=cfg.timestep_scale, beta=cfg.timestep_beta)
    noisy_actions, action_velocity = interpolate(batch.actions, torch.randn_like(batch.actions), t)
    noisy_boxes = box_velocity = None
    if batch.boxes is not None:
        noisy_boxes, box_velocity = interpolate(batch.boxes, torch.randn_like(batch.boxes), t)

    with torch.autocast(device_type=batch.state.device.type, dtype=amp_dtype, enabled=amp):
        predicted = model(batch.patches, batch.state, noisy_actions, noisy_boxes, t)
        p = cfg.action_exponent
        action_loss = (predicted.action - action_velocity).abs().pow(p).mean() / p
        total = cfg.action_weight * action_loss
        weight = cfg.action_weight
        box_loss = torch.zeros((), device=total.device)
        if box_velocity is not None:
            box_loss = F.mse_loss(predicted.box.float(), box_velocity)
            total = total + cfg.box_weight * box_loss
            weight += cfg.box_weight
        total = total / weight
    return total, {"action_loss": action_loss.item(), "box_loss": box_loss.item()}


@torch.no_grad()
def rollout_metrics(
    model: BoxWAM, batch: Batch, stats: ActionStats, *, num_steps: int, amp_dtype: torch.dtype, amp: bool
) -> dict[str, float]:
    """Integrate the flow from noise and measure errors of the denoised chunk.

    Returns summed (not averaged) squared errors so results can be reduced across ranks:
    ``action_se`` in absolute action units, ``box_se`` in normalised box units, and ``count``.
    """
    device = batch.state.device
    actions = torch.randn(batch.size, model.horizon, model.action_dim, device=device)
    boxes = torch.randn(batch.size, model.horizon, model.box_dim, device=device) if batch.boxes is not None else None
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
        cache = model.encode_observation(batch.patches, batch.state)
        actions, boxes = integrate(model, cache, batch.state, actions, boxes, num_steps)
    predicted = stats.denormalize_actions(actions, batch.state_raw)
    metrics = {
        "action_se": F.mse_loss(predicted, batch.actions_raw, reduction="sum").item() / predicted[0].numel(),
        "count": float(batch.size),
    }
    if boxes is not None:
        metrics["box_se"] = F.mse_loss(boxes, batch.boxes, reduction="sum").item() / boxes[0].numel()
    return metrics
