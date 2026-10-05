#!/usr/bin/env python3
"""Convert RoboTwin endpose+OBB HDF5 episodes -> a LeRobot v2.1 dataset (16-dim EEF + 2-slot OBB).

Features of the output dataset:
  observation.state    : float32[16] = [left x,y,z,q1..q4,gripper, right x,y,z,q1..q4,gripper]  (endpose)
  action               : float32[16] = next-frame endpose  (action[t]=state[t+1], last frame repeated)
  observation.images.cam_high / cam_left_wrist / cam_right_wrist : video, native 480x640, TRUE RGB
  observation.obb_head : float32[32] = [left_16 | right_16] 2-slot per-hand OBB (head-cam pixels;
                          slot 0 = left-hand target, slot 1 = right-hand target, empty hand -> zeros)
  task (per frame)     : one native RoboTwin `seen` instruction for the episode

One invocation converts ONE (task, config) HDF5 dir into ONE LeRobot dataset (a "part"); the parts
are later concatenated with merge_lerobot_datasets.py (which carries the obb_head column through
verbatim). Example, for data recorded with RoboTwin-boxwam's ``collect_data.sh``:

    uv run scripts/robotwin/convert_robotwin_to_lerobot.py \
        --task-config-dir ../../eval/RoboTwin-boxwam/data/adjust_bottle/demo_clean \
        --repo-id parts/demo_clean/adjust_bottle --root ~/robotwin_parts/demo_clean/adjust_bottle

RGB note: RoboTwin's pkl2hdf5 JPEG-encodes SAPIEN RGB via cv2.imencode (BGR-in-RGB); `_decode_native`
flips the channels back to true RGB.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from robotwin_hdf5 import (  # sibling module in scripts/robotwin/
    NUM_OBB_SLOTS,
    OBB_PER_OBJECT_DIM,
    _decode_native,
    _patch_lerobot_image_writer,
    build_per_hand_obb,
)

# RoboTwin camera group -> LeRobot key (three cameras; RoboTwin's front_camera is not used).
CAM_MAP = (
    ("head_camera", "cam_high"),
    ("left_camera", "cam_left_wrist"),
    ("right_camera", "cam_right_wrist"),
)
OBB_DIM = NUM_OBB_SLOTS * OBB_PER_OBJECT_DIM  # 32


def assemble_endpose(f: h5py.File, n: int) -> np.ndarray:
    """[n,16] = [left xyz+quat(7), left_gripper(1), right xyz+quat(7), right_gripper(1)]."""
    le = np.asarray(f["endpose/left_endpose"][:n], dtype=np.float32)      # [n,7]
    lg = np.asarray(f["endpose/left_gripper"][:n], dtype=np.float32).reshape(n, 1)
    re = np.asarray(f["endpose/right_endpose"][:n], dtype=np.float32)     # [n,7]
    rg = np.asarray(f["endpose/right_gripper"][:n], dtype=np.float32).reshape(n, 1)
    return np.concatenate([le, lg, re, rg], axis=1)


def pick_instruction(task_config_dir: Path, ep_idx: int) -> str | None:
    """One deterministic `seen` instruction per episode."""
    p = task_config_dir / "instructions" / f"episode{ep_idx}.json"
    if p.is_file():
        d = json.loads(p.read_text())
        seen = d.get("seen") or []
        if seen:
            return seen[ep_idx % len(seen)]
    return None


def _probe_hw(hdf5_paths: list[Path]) -> tuple[int, int]:
    import io
    from PIL import Image
    with h5py.File(hdf5_paths[0], "r") as f:
        img = Image.open(io.BytesIO(bytes(f["observation/head_camera/rgb"][0]))).convert("RGB")
    return img.size[1], img.size[0]  # (H, W)


def create_dataset(repo_id: str, root: Path, hw: tuple[int, int], fps: int):
    import shutil

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    _patch_lerobot_image_writer()
    h, w = hw
    # 16-dim endpose dimension names of observation.state / action.
    EEF_NAMES = [
        "left_x", "left_y", "left_z", "left_q1", "left_q2", "left_q3", "left_q4", "left_gripper",
        "right_x", "right_y", "right_z", "right_q1", "right_q2", "right_q3", "right_q4", "right_gripper",
    ]
    # 32-dim OBB names: [left slot 8 corners | right slot 8 corners], each corner (x, y) in head-cam pixels.
    OBB_NAMES = [f"{side}_c{c}_{ax}" for side in ("left", "right") for c in range(8) for ax in ("x", "y")]
    # The per-dim name lists are nested one level: names == [[...16 names...]].
    features = {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": [EEF_NAMES]},
        "action": {"dtype": "float32", "shape": (16,), "names": [EEF_NAMES]},
        "observation.obb_head": {"dtype": "float32", "shape": (OBB_DIM,), "names": [OBB_NAMES]},
    }
    for _, dst in CAM_MAP:
        features[f"observation.images.{dst}"] = {
            "dtype": "video", "shape": (3, h, w), "names": ["channels", "height", "width"]
        }
    if root.exists():
        shutil.rmtree(root)
    return LeRobotDataset.create(
        repo_id=repo_id, fps=fps, root=root, robot_type="aloha",
        features=features, use_videos=True, image_writer_processes=0, image_writer_threads=12,
    )


def add_episode(dataset, ep_path: Path, task_config_dir: Path, ep_idx: int) -> tuple[bool, str]:
    import torch

    try:
        with h5py.File(ep_path, "r") as f:
            n = f["endpose/left_endpose"].shape[0]
            endpose = assemble_endpose(f, n)                              # [n,16]
            state = endpose
            action = np.concatenate([endpose[1:], endpose[-1:]], axis=0)  # next-endpose; last repeated
            cams = {}
            for src, dst in CAM_MAP:
                key = f"observation/{src}/rgb"
                if key not in f:
                    return False, f"missing {key}"
                cams[dst], _, _ = _decode_native(f[key])                  # TRUE RGB (channel flip inside)
            obb_px, _ = build_per_hand_obb(f, cam_group_name="head_camera", n_frames=n)  # [n,2,16]
        obb_flat = obb_px.reshape(n, OBB_DIM).astype(np.float32)          # [n,32] = [left16|right16]
        instr = pick_instruction(task_config_dir, ep_idx) or "manipulate the object"
        for i in range(n):
            frame = {
                "observation.state": torch.from_numpy(state[i]),
                "action": torch.from_numpy(action[i]),
                "observation.obb_head": torch.from_numpy(obb_flat[i]),
                "task": instr,
            }
            for _, dst in CAM_MAP:
                frame[f"observation.images.{dst}"] = cams[dst][i]
            dataset.add_frame(frame)
        dataset.save_episode()
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        return False, f"ERR {type(e).__name__}: {e}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task-config-dir", required=True,
                   help="RoboTwin data/<task>/<config> dir (contains data/*.hdf5 + instructions/).")
    p.add_argument("--repo-id", required=True, help="LeRobot repo_id for this part.")
    p.add_argument("--root", required=True, help="On-disk output dir for this part's LeRobot dataset.")
    p.add_argument("--fps", type=int, default=50, help="LeRobot fps label (the released dataset uses 50).")
    args = p.parse_args(argv)

    tcdir = Path(args.task_config_dir).expanduser().resolve()
    eps = sorted((tcdir / "data").glob("episode*.hdf5"), key=lambda q: int(q.stem.replace("episode", "")))
    if not eps:
        print(f"no episode*.hdf5 under {tcdir}/data", file=sys.stderr)
        return 2

    dataset = create_dataset(args.repo_id, Path(args.root).expanduser().resolve(), _probe_hw(eps), args.fps)
    n_ok = n_bad = 0
    skipped: list[tuple[str, str]] = []
    for ep in eps:
        ep_idx = int(ep.stem.replace("episode", ""))
        ok, msg = add_episode(dataset, ep, tcdir, ep_idx)
        if not ok:
            # A failed add_frame/save_episode leaves LeRobot's episode_buffer partially
            # mutated (save_episode pops "size" before it can crash on video encoding), so
            # EVERY later episode would cascade-fail with KeyError: 'size'. Reset the buffer,
            # then retry this episode once — covers transient async image-writer errors like
            # a missing frame PNG. LeRobot reuses the (un-incremented) episode index, so the
            # failed episode's orphan parquet is overwritten by the retry/next episode and the
            # dataset stays internally consistent.
            dataset.episode_buffer = dataset.create_episode_buffer()
            ok, retry_msg = add_episode(dataset, ep, tcdir, ep_idx)
            if not ok:
                dataset.episode_buffer = dataset.create_episode_buffer()
                msg = f"{msg} | retry: {retry_msg}"
        if ok:
            n_ok += 1
        else:
            n_bad += 1
            skipped.append((ep.name, msg))
            print(f"[skip] {ep.name}: {msg}", file=sys.stderr)
    print(f"Wrote {n_ok} episodes to {args.root} ({n_bad} skipped)")
    if skipped:
        print(f"[skipped-summary] {len(skipped)} episode(s) permanently skipped after retry:", file=sys.stderr)
        for name, msg in skipped:
            print(f"   {name}: {msg}", file=sys.stderr)
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
