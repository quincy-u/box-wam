#!/usr/bin/env python
"""Compute the action/state percentiles used to normalise training targets.

For every episode and every window start ``t`` (``t + horizon <= T``), the anchor state is
``state[t]`` and the targets are ``action[t : t + horizon]``. Arm dimensions of the targets are
made relative to the anchor state; gripper dimensions stay absolute. Percentiles (1st / 99th)
are taken over all windows and written to an ``.npz`` consumed by ``data.stats`` in the
training config.

    python scripts/compute_action_stats.py --episodes data/robotwin/episodes.txt \
        --horizon 15 --gripper-dims 6 13 --output data/robotwin/action_stats.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxwam.data.episodes import find_episodes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", required=True, help="episode list file or root directory")
    parser.add_argument("--task", help="only episodes whose path contains this substring")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--gripper-dims", type=int, nargs="*", default=[], help="absolute (non-relative) action dims")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    episodes = find_episodes(args.episodes, args.task)
    states, actions = [], []
    for episode in tqdm(episodes, desc="episodes"):
        state = np.load(episode / "state.npy").astype(np.float64)
        action = np.load(episode / "action.npy").astype(np.float64)
        length = min(len(state), len(action))
        num_windows = length - args.horizon + 1
        if num_windows <= 0:
            continue
        arm = [d for d in range(action.shape[1]) if d not in args.gripper_dims]
        starts = np.arange(num_windows)
        anchor = state[starts]  # (W, S)
        windows = action[starts[:, None] + np.arange(args.horizon)[None, :]]  # (W, H, A)
        windows[:, :, arm] -= anchor[:, None, arm]
        states.append(anchor)
        actions.append(windows.reshape(-1, action.shape[1]))
    states = np.concatenate(states)
    actions = np.concatenate(actions)
    stats = {
        "action_q01": np.percentile(actions, 1, axis=0),
        "action_q99": np.percentile(actions, 99, axis=0),
        "action_mean": actions.mean(0),
        "action_std": actions.std(0),
        "state_q01": np.percentile(states, 1, axis=0),
        "state_q99": np.percentile(states, 99, axis=0),
        "state_mean": states.mean(0),
        "state_std": states.std(0),
        "horizon": np.array(args.horizon),
        "gripper_dims": np.array(args.gripper_dims, dtype=np.int64),
        "num_episodes": np.array(len(episodes)),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        **{key: value.astype(np.float32) if value.dtype.kind == "f" else value for key, value in stats.items()},
    )
    np.set_printoptions(precision=3, suppress=True)
    print(f"{len(episodes)} episodes, {len(actions):,} action steps -> {args.output}")
    print("action q01:", stats["action_q01"])
    print("action q99:", stats["action_q99"])
    print("state  q01:", stats["state_q01"])
    print("state  q99:", stats["state_q99"])


if __name__ == "__main__":
    main()
