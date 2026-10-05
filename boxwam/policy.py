"""Inference: load a checkpoint and turn one observation into an action chunk (and boxes).

Example::

    policy = BoxWAMPolicy.from_checkpoint("best.pt", device="cuda:0")
    actions, boxes = policy.act([head_rgb_uint8], joint_positions)
    # actions: (H, action_dim) absolute targets, boxes: (H, K, 8, 2) image fractions or None
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from boxwam.config import ModelConfig, from_dict
from boxwam.data.normalization import ActionStats, denormalize_boxes
from boxwam.model.boxwam import BoxWAM
from boxwam.model.encoder import DinoPatchEncoder, normalize_images
from boxwam.model.flow import euler_schedule
from boxwam.training.checkpoint import load_checkpoint, load_model_weights, stats_from_dict

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class PolicyOutput:
    actions: np.ndarray  # (H, action_dim) absolute target positions
    boxes: np.ndarray | None  # (H, K, 8, 2) projected corners as image fractions, or None
    latency_ms: float


class BoxWAMPolicy:
    def __init__(
        self,
        model: BoxWAM,
        encoder: DinoPatchEncoder,
        stats: ActionStats,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        num_steps: int = 50,
        image_size: int = 224,
        compile: bool = False,
        config: dict | None = None,
    ):
        self.model = model.to(device).eval()
        self.config = config  # training config the checkpoint was produced with, if known
        self.encoder = encoder.to(device)
        self.stats = stats
        self.device = device
        self.dtype = dtype
        self.num_steps = num_steps
        self.image_size = image_size
        self._encode = self.model.encode_observation
        self._velocity = self.model.predict_velocity
        if compile and device.type == "cuda":
            self._encode = torch.compile(self._encode)
            self._velocity = torch.compile(self._velocity)

    # ------------------------------------------------------------ loading
    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cuda:0",
        num_steps: int = 50,
        dtype: str = "bfloat16",
        compile: bool = False,
    ) -> BoxWAMPolicy:
        """Build a policy from a checkpoint, which carries its training config and statistics."""
        checkpoint = load_checkpoint(path)
        if "config" not in checkpoint or "stats" not in checkpoint:
            raise ValueError(f"{path} is not a Box-WAM checkpoint: it lacks the embedded config or statistics")
        config = checkpoint["config"]
        stats = stats_from_dict(checkpoint["stats"])
        model_cfg = from_dict(ModelConfig, config["model"], "model")
        image_size = int(config.get("augmentation", {}).get("crop_size", 224))
        encoder = DinoPatchEncoder(model_cfg.encoder)
        from boxwam.training.train import build_model

        model = build_model(
            model_cfg,
            encoder_dim=encoder.embed_dim,
            num_cameras=len(config["data"]["cameras"]),
            image_size=image_size,
            patch_size=encoder.patch_size,
        )
        load_model_weights(model, checkpoint)
        return cls(
            model,
            encoder,
            stats,
            device=torch.device(device),
            dtype=DTYPES[dtype],
            num_steps=num_steps,
            image_size=image_size,
            compile=compile,
            config=config,
        )

    # ------------------------------------------------------------- acting
    def preprocess(self, images: Sequence[np.ndarray]) -> list[torch.Tensor]:
        """uint8 RGB ``(H, W, 3)`` frames -> ImageNet-normalised ``(1, 3, S, S)`` tensors."""
        tensors = []
        for image in images:
            frame = Image.fromarray(np.asarray(image, dtype=np.uint8)).resize(
                (self.image_size, self.image_size), Image.LANCZOS
            )
            tensor = torch.from_numpy(np.asarray(frame).copy()).permute(2, 0, 1).float().div_(255.0)
            tensors.append(normalize_images(tensor)[None].to(self.device))
        return tensors

    @torch.no_grad()
    def act(self, images: Sequence[np.ndarray], state: np.ndarray) -> PolicyOutput:
        """Predict one action chunk from the current camera frames and robot state."""
        if len(images) != self.model.num_cameras:
            raise ValueError(f"policy expects {self.model.num_cameras} camera images, got {len(images)}")
        start = time.perf_counter()
        patches = torch.cat([self.encoder(image) for image in self.preprocess(images)], dim=1)
        state_raw = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).reshape(1, -1)
        normalized_state = self.stats.normalize_state(state_raw)
        actions = torch.randn(1, self.model.horizon, self.model.action_dim, device=self.device)
        boxes = (
            torch.randn(1, self.model.horizon, self.model.box_dim, device=self.device) if self.model.has_boxes else None
        )
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.dtype != torch.float32 and self.device.type == "cuda",
        ):
            cache = self._encode(patches, normalized_state)
            for time_value, dt in euler_schedule(self.num_steps):
                t = torch.full((1,), time_value, device=self.device)
                velocity = self._velocity(cache, normalized_state, actions, boxes, t)
                actions += dt * velocity.action.float()
                if boxes is not None:
                    boxes += dt * velocity.box.float()
        actions = self.stats.denormalize_actions(actions, state_raw)[0].cpu().numpy()
        box_output = None
        if boxes is not None:
            box_output = denormalize_boxes(boxes)[0].reshape(self.model.horizon, -1, 8, 2).cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return PolicyOutput(actions=actions, boxes=box_output, latency_ms=(time.perf_counter() - start) * 1000.0)


__all__ = ["BoxWAMPolicy", "PolicyOutput"]
