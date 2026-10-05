"""Transforms for the joint OBB-trajectory channel used by pi0.5 with ``predict_obb=True``.

Reads a per-frame OBB trajectory in *native camera pixel coordinates*, normalizes it to
roughly ``[-1, 1]`` (the same box normalization as the standalone Box-WAM policy), and
writes a single flat ``future_obb_gt`` tensor.

OBB slot layout (per frame):
    flat[0:16]   = LEFT-hand-targeted object's 8 corners as [u0, v0, u1, v1, ..., u7, v7]
    flat[16:32]  = RIGHT-hand-targeted object's 8 corners (zeros when no target / off-screen)

There is no per-slot validity mask: empty hands carry an all-zero target so the model
explicitly learns "no target -> denoise to zero" rather than leaving that slot
unconstrained at inference. End-of-episode pad frames (when LeRobot clamps OOB
``delta_timestamps`` to the last valid frame) are also zeroed out for the same reason.

Normalization is per-axis because the RoboTwin head-camera frames are non-square: ``u`` is
divided by ``image_width`` and ``v`` by ``image_height``.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class RobotwinObbInputs(transforms.DataTransformFn):
    """Convert raw-pixel OBB trajectories to normalized per-hand OBB inputs.

    Reads ``data[src_key]`` of shape ``[H, 2 * 16]`` and writes:
      ``future_obb_gt``: float32 ``[H, 2 * 16]`` in ~``[-1, 1]``. Empty slots / pad frames are 0.
    """

    # Native camera pixel dimensions used during OBB projection (RoboTwin head_camera default).
    image_width: float = 320.0
    image_height: float = 240.0
    # Corners may lie up to this fraction of the image outside it before they are clamped.
    clamp_margin: float = 0.05

    # Layout of the flat OBB tensor (always 2 slots: left, right).
    num_slots: int = 2
    per_slot_dim: int = 16  # 8 corners x (u, v)

    # Where to read the raw-pixel OBB trajectory from.
    src_key: str = "future_obb_pixels"
    # Where to read the per-frame "is padding" boolean (LeRobot clamps OOB delta_timestamps to
    # the last valid frame; without this, end-of-episode windows train on stale targets).
    pad_key: str = "future_obb_is_pad"
    # Where to write the normalized OBB.
    dst_obb_key: str = "future_obb_gt"

    def __call__(self, data: dict) -> dict:
        if self.src_key not in data:
            return data
        obb_px = np.asarray(data.pop(self.src_key), dtype=np.float32)
        if obb_px.ndim != 2:
            raise ValueError(f"{self.src_key} must be 2-D [H, D]; got shape {obb_px.shape}")
        h, total = obb_px.shape
        per_slot = self.per_slot_dim
        if total != self.num_slots * per_slot:
            raise ValueError(
                f"{self.src_key} last dim {total} != num_slots ({self.num_slots}) * "
                f"per_slot_dim ({per_slot})"
            )

        # Per-axis pixel->[-1, 1] normalization (head camera is non-square 320x240).
        obb_xy = obb_px.reshape(h, self.num_slots, per_slot // 2, 2)  # [H, 2, 8, 2]
        x = obb_xy[..., 0] / float(self.image_width)
        y = obb_xy[..., 1] / float(self.image_height)
        x = np.clip(x, -self.clamp_margin, 1.0 + self.clamp_margin) * 2.0 - 1.0
        y = np.clip(y, -self.clamp_margin, 1.0 + self.clamp_margin) * 2.0 - 1.0
        obb_norm = np.stack([x, y], axis=-1).reshape(h, total).astype(np.float32)

        # Detect raw zero-padded slots (empty hand) and re-zero them after normalization (otherwise
        # the (-1, -1) corner would look like a real off-screen prediction). Same for end-of-episode
        # pad frames.
        per_slot_max_abs = np.abs(obb_px.reshape(h, self.num_slots, per_slot)).max(axis=-1)
        empty_slot = (per_slot_max_abs <= 1e-6).astype(np.float32)  # [H, num_slots]
        slot_zero_mask = np.repeat(empty_slot, per_slot, axis=-1)  # [H, total]
        obb_norm = obb_norm * (1.0 - slot_zero_mask)

        if self.pad_key in data:
            is_pad = np.asarray(data.pop(self.pad_key), dtype=bool)  # [H]
            obb_norm = obb_norm * (~is_pad)[:, None].astype(np.float32)

        return {**data, self.dst_obb_key: obb_norm}


@dataclasses.dataclass(frozen=True)
class RobotwinObbOutputs(transforms.DataTransformFn):
    """Inverse of :class:`RobotwinObbInputs` for inference.

    Reads the model's predicted OBB trajectory at ``data[src_key]`` (shape
    ``[H, num_slots, per_slot_dim]`` in roughly ``[-1, 1]``, as returned by
    ``pi0.sample_actions`` when ``predict_obb=True``) and writes image-fractional
    corners at ``data[dst_key]`` (shape ``[H, num_slots, 8, 2]``) so drawing code
    can simply multiply by the runtime frame's ``(W, H)``.

    Slot 0 = left-hand-targeted OBB, slot 1 = right-hand-targeted OBB. Slots that
    correspond to an empty hand will (after training) denoise toward zero.

    Train/native pixel-space mismatch
    ---------------------------------
    ``RobotwinObbInputs`` divides x by ``image_width`` and y by ``image_height``, so the
    model's output (after undoing the ``2x - 1`` shift) is exactly
    ``pixel / (image_width, image_height)`` in TRAIN-norm space. If the runtime frame is a
    different size, scale by ``(image_width / native_image_width, image_height /
    native_image_height)`` to recover image-fractional coords on the runtime frame.
    """

    num_slots: int = 2
    per_slot_dim: int = 16

    # Per-axis post-output scale; compensates for any train-norm vs native-pixel mismatch.
    # Use ``image_width_train / native_image_width`` (similarly for y).
    scale_x: float = 1.0
    scale_y: float = 1.0

    src_key: str = "future_obb"
    dst_key: str = "future_obb_pixels"

    def __call__(self, data: dict) -> dict:
        if self.src_key not in data:
            return data
        obb = np.asarray(data[self.src_key], dtype=np.float32)
        # Accept both flat ``[H, num_slots * per_slot_dim]`` and split
        # ``[H, num_slots, per_slot_dim]`` shapes.
        if obb.ndim == 2:
            h, total = obb.shape
            if total != self.num_slots * self.per_slot_dim:
                raise ValueError(
                    f"{self.src_key} last dim {total} != num_slots ({self.num_slots}) * "
                    f"per_slot_dim ({self.per_slot_dim})"
                )
            obb = obb.reshape(h, self.num_slots, self.per_slot_dim)
        elif obb.ndim != 3:
            raise ValueError(f"{self.src_key} must be 2-D or 3-D; got shape {obb.shape}")

        h, ns, per_slot = obb.shape
        # Undo ``2x - 1`` to get fractional per axis, then rescale to compensate for any
        # train-norm vs native-pixel-space mismatch.
        corners = obb.reshape(h, ns, per_slot // 2, 2)
        corners = (corners + 1.0) * 0.5
        if self.scale_x != 1.0 or self.scale_y != 1.0:
            corners = corners * np.array([self.scale_x, self.scale_y], dtype=np.float32)
        return {**data, self.dst_key: corners.astype(np.float32)}
