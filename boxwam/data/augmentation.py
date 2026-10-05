"""Batched GPU augmentation that keeps images and projected boxes aligned.

One random geometric transform (random resized crop, optional horizontal flip, small
rotation) and one colour jitter are sampled per sample and applied identically to every
camera view of that sample. Box corner coordinates receive the same geometric transform so
the supervision stays pixel-aligned with the augmented image.

Inputs are uint8 images ``(B, H, W, 3)`` per camera; outputs are ImageNet-normalised float
images ``(B, 3, crop, crop)`` and boxes in crop-pixel coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from boxwam.model.encoder import IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class AugmentationConfig:
    """Ranges are ``(low, high)`` for uniform sampling. The defaults disable every augmentation."""

    crop_size: int = 224
    crop_scale: tuple[float, float] = (1.0, 1.0)  # area fraction of the random resized crop
    crop_ratio: tuple[float, float] = (1.0, 1.0)  # aspect ratio of the crop
    rotation_degrees: tuple[float, float] = (0.0, 0.0)
    horizontal_flip: bool = False
    brightness: tuple[float, float] = (1.0, 1.0)
    contrast: tuple[float, float] = (1.0, 1.0)
    saturation: tuple[float, float] = (1.0, 1.0)
    hue: tuple[float, float] = (0.0, 0.0)

    def geometry_only(self) -> AugmentationConfig:
        """Same crop statistics without rotation, flip or colour changes (used for validation)."""
        return AugmentationConfig(crop_size=self.crop_size, crop_scale=self.crop_scale, crop_ratio=self.crop_ratio)


def _sample_parameters(batch: int, height: int, width: int, cfg: AugmentationConfig, generator=None) -> dict:
    def uniform(low: float, high: float) -> torch.Tensor:
        return torch.empty(batch).uniform_(low, high, generator=generator)

    scale = uniform(*cfg.crop_scale)
    ratio = torch.exp(uniform(math.log(cfg.crop_ratio[0]), math.log(cfg.crop_ratio[1])))
    area = scale * (height * width)
    crop_w = torch.sqrt(area * ratio).round().clamp(1.0, float(width))
    crop_h = torch.sqrt(area / ratio).round().clamp(1.0, float(height))
    max_top = (height - crop_h).clamp(min=0)
    max_left = (width - crop_w).clamp(min=0)
    top = (torch.rand(batch, generator=generator) * (max_top + 1.0)).floor().clamp(max=max_top)
    left = (torch.rand(batch, generator=generator) * (max_left + 1.0)).floor().clamp(max=max_left)
    flip = torch.rand(batch, generator=generator) < 0.5 if cfg.horizontal_flip else torch.zeros(batch, dtype=torch.bool)
    return {
        "top": top,
        "left": left,
        "crop_h": crop_h,
        "crop_w": crop_w,
        "angle": uniform(*cfg.rotation_degrees) * (math.pi / 180.0),
        "flip": flip,
        "brightness": uniform(*cfg.brightness),
        "contrast": uniform(*cfg.contrast),
        "saturation": uniform(*cfg.saturation),
        "hue": uniform(*cfg.hue),
    }


def _transform_images(images: torch.Tensor, p: dict, out_size: int) -> torch.Tensor:
    """Crop + flip + rotate ``(B, C, H, W)`` float images to ``(B, C, out, out)`` with one bilinear resample."""
    batch, channels, height, width = images.shape
    cos, sin = torch.cos(p["angle"]), torch.sin(p["angle"])
    flip_sign = torch.where(p["flip"], -torch.ones_like(cos), torch.ones_like(cos))
    cx = p["left"] + p["crop_w"] / 2.0
    cy = p["top"] + p["crop_h"] / 2.0
    hx, hy = p["crop_w"] / 2.0, p["crop_h"] / 2.0
    sx, sy = 2.0 / width, 2.0 / height
    theta = torch.stack(
        [
            torch.stack([flip_sign * cos * hx * sx, -flip_sign * sin * hx * sx, cx * sx - 1.0], dim=-1),
            torch.stack([sin * hy * sy, cos * hy * sy, cy * sy - 1.0], dim=-1),
        ],
        dim=-2,
    )
    grid = F.affine_grid(theta, size=[batch, channels, out_size, out_size], align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _transform_boxes(boxes: torch.Tensor, p: dict, out_size: int) -> torch.Tensor:
    """Apply the same crop + flip + rotation to ``(B, T, D)`` flattened ``(x, y)`` corner coordinates."""
    batch, steps, dim = boxes.shape
    points = boxes.view(batch, steps, -1, 2)
    view = lambda v: v.view(batch, 1, 1)  # noqa: E731
    x = (points[..., 0] - view(p["left"])) * (float(out_size) / view(p["crop_w"]))
    y = (points[..., 1] - view(p["top"])) * (float(out_size) / view(p["crop_h"]))
    if bool(p["flip"].any()):
        x = torch.where(view(p["flip"]), (float(out_size) - 1.0) - x, x)
    center = (float(out_size) - 1.0) * 0.5
    cos, sin = torch.cos(view(p["angle"])), torch.sin(view(p["angle"]))
    rx, ry = x - center, y - center
    rotated_x = cos * rx + sin * ry + center
    rotated_y = -sin * rx + cos * ry + center
    return torch.stack([rotated_x, rotated_y], dim=-1).reshape(batch, steps, dim)


def _rgb_to_hsv_shift(x: torch.Tensor, hue_shift: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    maxc, _ = x.max(dim=1, keepdim=True)
    minc, _ = x.min(dim=1, keepdim=True)
    delta = maxc - minc
    saturation = torch.where(maxc > 0, delta / maxc.clamp(min=1e-12), torch.zeros_like(maxc))
    safe = delta.clamp(min=1e-12)
    rc, gc, bc = (maxc - r) / safe, (maxc - g) / safe, (maxc - b) / safe
    hue = torch.where(r == maxc, bc - gc, torch.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
    hue = torch.where(delta == 0, torch.zeros_like(hue), (hue / 6.0) % 1.0)
    hue = (hue + hue_shift) % 1.0
    sector = (hue * 6.0).floor()
    frac = hue * 6.0 - sector
    value = maxc
    p_ = value * (1.0 - saturation)
    q_ = value * (1.0 - frac * saturation)
    t_ = value * (1.0 - (1.0 - frac) * saturation)
    sector = sector.long() % 6
    cases = [(value, t_, p_), (q_, value, p_), (p_, value, t_), (p_, q_, value), (t_, p_, value), (value, p_, q_)]
    out = torch.zeros_like(x)
    for index, (rr, gg, bb) in enumerate(cases):
        mask = sector == index
        out[:, 0:1] = torch.where(mask, rr, out[:, 0:1])
        out[:, 1:2] = torch.where(mask, gg, out[:, 1:2])
        out[:, 2:3] = torch.where(mask, bb, out[:, 2:3])
    return out


def _color_jitter(x: torch.Tensor, p: dict, apply_hue: bool) -> torch.Tensor:
    """Brightness, contrast, saturation and hue jitter on ``(B, 3, H, W)`` images in ``[0, 1]``."""
    view = lambda v: v.view(-1, 1, 1, 1)  # noqa: E731
    x = x * view(p["brightness"])
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    mean = gray.mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * view(p["contrast"]) + mean
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    x = (gray + view(p["saturation"]) * (x - gray)).clamp(0.0, 1.0)
    if apply_hue:
        x = _rgb_to_hsv_shift(x, view(p["hue"])).clamp(0.0, 1.0)
    return x


@torch.no_grad()
def augment(
    images: list[torch.Tensor],
    boxes: torch.Tensor | None,
    cfg: AugmentationConfig,
    generator: torch.Generator | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor | None]:
    """Augment every camera view (and the boxes) of a batch with one shared transform per sample.

    ``images``: one uint8 tensor ``(B, H, W, 3)`` per camera, all on the same device.
    ``boxes``: optional ``(B, T, D)`` pixel coordinates in the source frame (any camera-agnostic
    frame of the same resolution as the images).
    Returns ImageNet-normalised ``(B, 3, crop, crop)`` images and boxes in crop pixels.
    """
    batch, height, width, _ = images[0].shape
    device = images[0].device
    params = _sample_parameters(batch, height, width, cfg, generator)
    params = {key: value.to(device) for key, value in params.items()}
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    apply_hue = cfg.hue != (0.0, 0.0)

    out_images = []
    for view in images:
        x = view.permute(0, 3, 1, 2).float() / 255.0
        x = _transform_images(x, params, cfg.crop_size)
        x = _color_jitter(x, params, apply_hue)
        out_images.append((x - mean) / std)
    out_boxes = _transform_boxes(boxes.to(device), params, cfg.crop_size) if boxes is not None else None
    return out_images, out_boxes
