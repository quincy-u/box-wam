#!/usr/bin/env python
"""Mark window starts whose action chunk actually moves the robot (writes ``valid_idx.npy``).

RoboTwin demonstrations contain long stretches where the arms rest while the planner
waits. A window starting at ``t`` is kept when, over the chunk ``action[t : t + horizon]``,
either the arm joints move by at least ``--arm-eps`` (L2 norm relative to ``state[t]``) or a
gripper moves by at least ``--gripper-eps``. Enable ``data.use_valid_starts`` in the training
config to sample only from the kept starts.

    python scripts/filter_static_chunks.py --episodes data/robotwin/episodes.txt --horizon 15 --gripper-dims 6 13
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxwam.data.episodes import find_episodes  # noqa: E402


def valid_starts(
    state: np.ndarray, action: np.ndarray, *, horizon: int, gripper_dims: list[int], arm_eps: float, gripper_eps: float
) -> np.ndarray:
    length = min(len(state), len(action))
    num_windows = length - horizon
    if num_windows <= 0:
        return np.zeros(0, dtype=np.int64)
    arm = [d for d in range(action.shape[1]) if d not in gripper_dims]
    starts = np.arange(num_windows)
    chunks = action[starts[:, None] + np.arange(horizon)[None, :]]  # (W, H, A)
    anchor = state[starts][:, None, :]  # action dimension d is compared with state dimension d
    arm_motion = (
        np.linalg.norm(chunks[:, :, arm] - anchor[:, :, arm], axis=-1).max(axis=1) if arm else np.zeros(num_windows)
    )
    gripper_motion = (
        np.abs(chunks[:, :, gripper_dims] - anchor[:, :, gripper_dims]).max(axis=(1, 2))
        if gripper_dims
        else np.zeros(num_windows)
    )
    return starts[(arm_motion >= arm_eps) | (gripper_motion >= gripper_eps)].astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--task")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--gripper-dims", type=int, nargs="*", default=[])
    parser.add_argument("--arm-eps", type=float, default=0.05)
    parser.add_argument("--gripper-eps", type=float, default=0.05)
    args = parser.parse_args()

    total = kept = 0
    for episode in tqdm(find_episodes(args.episodes, args.task), desc="episodes"):
        state = np.load(episode / "state.npy").astype(np.float64)
        action = np.load(episode / "action.npy").astype(np.float64)
        starts = valid_starts(
            state,
            action,
            horizon=args.horizon,
            gripper_dims=args.gripper_dims,
            arm_eps=args.arm_eps,
            gripper_eps=args.gripper_eps,
        )
        np.save(episode / "valid_idx.npy", starts)
        total += max(min(len(state), len(action)) - args.horizon, 0)
        kept += len(starts)
    print(f"kept {kept:,} / {total:,} window starts ({100 * kept / max(total, 1):.1f}%)")


if __name__ == "__main__":
    main()
