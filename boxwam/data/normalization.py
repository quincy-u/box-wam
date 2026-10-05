"""Normalisation of robot states, actions and projected boxes.

States and actions are mapped to roughly ``[-1, 1]`` with per-dimension 1st/99th percentiles
computed over the training set (see ``scripts/compute_action_stats.py``). Arm actions are
expressed *relative to the current state* before normalisation; gripper dimensions are kept
absolute and are not normalised (they already live in ``[0, 1]``).

Boxes are 2-D pixel coordinates of projected 3-D box corners. They are mapped to ``[-1, 1]``
by dividing by the image side length, with a small margin for corners just outside the frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

EPS = 1e-6
BOX_MARGIN = 0.05


def _to_tensor(values: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=np.float32), device=like.device)


def quantile_normalize(values: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return (values - low) / (high - low + EPS) * 2.0 - 1.0


def quantile_denormalize(values: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return (values + 1.0) * 0.5 * (high - low + EPS) + low


@dataclass(frozen=True)
class ActionStats:
    """Percentile statistics of relative arm actions and absolute states."""

    action_q01: np.ndarray
    action_q99: np.ndarray
    state_q01: np.ndarray
    state_q99: np.ndarray
    gripper_dims: tuple[int, ...]

    REQUIRED_KEYS = ("action_q01", "action_q99", "state_q01", "state_q99")

    @classmethod
    def load(cls, path: str | Path, gripper_dims: Sequence[int]) -> ActionStats:
        with np.load(path) as data:
            missing = [key for key in cls.REQUIRED_KEYS if key not in data]
            if missing:
                raise KeyError(f"{path} is missing statistics {missing}")
            arrays = {key: np.asarray(data[key], dtype=np.float32) for key in cls.REQUIRED_KEYS}
        gripper = tuple(sorted(int(index) for index in gripper_dims))
        action_dim = arrays["action_q01"].shape[0]
        if any(index < 0 or index >= action_dim for index in gripper):
            raise ValueError(f"gripper_dims {gripper} out of range for action_dim={action_dim}")
        return cls(gripper_dims=gripper, **arrays)

    @property
    def action_dim(self) -> int:
        return int(self.action_q01.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.state_q01.shape[0])

    @property
    def arm_dims(self) -> list[int]:
        return [index for index in range(self.action_dim) if index not in self.gripper_dims]

    # ---------------------------------------------------------------- state
    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """``(B, state_dim)`` raw state -> ``[-1, 1]``."""
        return quantile_normalize(state.float(), _to_tensor(self.state_q01, state), _to_tensor(self.state_q99, state))

    # -------------------------------------------------------------- actions
    def normalize_actions(self, actions: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        """``(B, H, action_dim)`` absolute targets -> normalised relative targets.

        Arm dimensions become ``normalise(action - current_state)``; gripper dimensions pass
        through unchanged.
        """
        actions = actions.float().clone()
        arm = self.arm_dims
        if arm:
            deltas = actions[..., arm] - current_state.float()[:, None, arm]
            actions[..., arm] = quantile_normalize(
                deltas, _to_tensor(self.action_q01[arm], actions), _to_tensor(self.action_q99[arm], actions)
            )
        return actions

    def denormalize_actions(self, actions: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`normalize_actions`: returns absolute target positions."""
        actions = actions.float().clone()
        arm = self.arm_dims
        if arm:
            deltas = quantile_denormalize(
                actions[..., arm], _to_tensor(self.action_q01[arm], actions), _to_tensor(self.action_q99[arm], actions)
            )
            actions[..., arm] = deltas + current_state.float()[:, None, arm]
        return actions


# ------------------------------------------------------------------- boxes
def normalize_boxes(boxes_px: torch.Tensor, image_size: int) -> torch.Tensor:
    """Pixel corner coordinates (in a square ``image_size`` frame) -> ``[-1 - m, 1 + m]``."""
    fractions = (boxes_px.float() / float(image_size)).clamp(-BOX_MARGIN, 1.0 + BOX_MARGIN)
    return fractions * 2.0 - 1.0


def denormalize_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """Normalised corners -> image fractions in ``[-m, 1 + m]`` (multiply by width/height to get pixels)."""
    return ((boxes.float() + 1.0) * 0.5).clamp(-BOX_MARGIN, 1.0 + BOX_MARGIN)
