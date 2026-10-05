#!/usr/bin/env python
"""Convert RoboTwin 2.0 demonstrations (HDF5) into Box-WAM episode directories.

Requires demonstrations collected with our RoboTwin fork, which stores projected 3-D box
corners under ``observation/<camera>/proj_3d_obb/<object>``. Each episode becomes::

    <output>/<task>/<task_config>/episode_000000/
        video_<camera>.npy   (T, size, size, 3) uint8 RGB, one per camera
        state.npy            (T, 14)  joint positions   = joint_action/vector[t]
        action.npy           (T, 14)  next joint targets = joint_action/vector[t + 1]
        proj_3d_obb.npy      (T, K, 16) box corners of the K tracked objects, in video pixels

By default every object with a box is a slot (sorted by name). Use ``--objects`` to pick and
order objects explicitly, or ``--arm-slots`` for a fixed ``[left-arm target, right-arm target]``
layout driven by the per-frame ``object_target`` labels (zeros when an arm has no target).

    python scripts/prepare_robotwin_episodes.py --robotwin-data ~/RoboTwin/data --task-config demo_clean \
        --tasks open_laptop lift_pot --output data/robotwin
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

RESAMPLE = Image.LANCZOS


def decode_frame(blob) -> np.ndarray:
    """RoboTwin JPEG-encodes RGB arrays with OpenCV (which assumes BGR); decoding with OpenCV
    swaps the channels back, so the returned array is already RGB."""
    return cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)


def resize(frame: np.ndarray, size: int) -> np.ndarray:
    if frame.shape[0] == size and frame.shape[1] == size:
        return frame
    return np.asarray(Image.fromarray(frame).resize((size, size), RESAMPLE), dtype=np.uint8)


def read_boxes(
    f: h5py.File, camera: str, n: int, objects: list[str] | None, arm_slots: bool, scale: tuple[float, float]
) -> np.ndarray | None:
    group = f["observation"][camera].get("proj_3d_obb")
    if group is None:
        return None
    rows = {}
    for name in group:
        row = np.asarray(group[name][:n], dtype=np.float32).reshape(n, -1)[:, :16].copy()  # articulated: first box only
        row[:, 0::2] *= scale[0]
        row[:, 1::2] *= scale[1]
        rows[name] = row
    if arm_slots:
        boxes = np.zeros((n, 2, 16), dtype=np.float32)
        for slot, arm in enumerate(("left", "right")):
            targets = [t.decode() if isinstance(t, bytes) else str(t or "") for t in f["object_target"][arm][:n]]
            for t, name in enumerate(targets):
                if name in rows:
                    boxes[t, slot] = rows[name][t]
        return boxes
    names = objects or sorted(rows)
    missing = [name for name in names if name not in rows]
    if missing:
        raise KeyError(f"objects {missing} have no boxes; available: {sorted(rows)}")
    return np.stack([rows[name] for name in names], axis=1)


def convert(job: tuple) -> str:
    h5_path, out_dir, cameras, size, objects, arm_slots = job
    out_dir = Path(out_dir)
    if (out_dir / "state.npy").is_file():
        return "skipped"
    with h5py.File(h5_path) as f:
        vector = np.asarray(f["joint_action"]["vector"], dtype=np.float32)
        n = len(vector) - 1
        if n < 2:
            return "too short"
        scale = None
        for camera in cameras:
            blobs = f["observation"][camera]["rgb"]
            frames = [decode_frame(blobs[t]) for t in range(n)]
            if scale is None:  # boxes are projected in the first camera's native resolution
                scale = (size / frames[0].shape[1], size / frames[0].shape[0])
            video = np.stack([resize(frame, size) for frame in frames])
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / f"video_{camera}.npy", video)
        boxes = read_boxes(f, cameras[0], n, objects, arm_slots, scale)
    np.save(out_dir / "state.npy", vector[:-1])
    np.save(out_dir / "action.npy", vector[1:])
    if boxes is not None:
        np.save(out_dir / "proj_3d_obb.npy", boxes)
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--robotwin-data", required=True, help="RoboTwin data root: <root>/<task>/<task_config>/data/episode*.hdf5"
    )
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cameras", nargs="+", default=["head_camera"], help="first camera defines the box frame")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--objects", nargs="*", help="box slots in this order (default: all objects, sorted)")
    parser.add_argument("--arm-slots", action="store_true", help="two slots: left-arm target, right-arm target")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    jobs = []
    for task in args.tasks:
        files = sorted(
            (Path(args.robotwin_data).expanduser() / task / args.task_config / "data").glob("episode*.hdf5"),
            key=lambda path: int(path.stem[len("episode") :]),
        )
        if not files:
            raise FileNotFoundError(f"no episodes for {task}/{args.task_config} under {args.robotwin_data}")
        for path in files:
            index = int(path.stem[len("episode") :])
            out_dir = Path(args.output) / task / args.task_config / f"episode_{index:06d}"
            jobs.append((str(path), str(out_dir), args.cameras, args.size, args.objects, args.arm_slots))

    with Pool(args.workers) as pool:
        outcomes = list(tqdm(pool.imap_unordered(convert, jobs), total=len(jobs), desc="episodes"))
    for outcome in sorted(set(outcomes)):
        print(f"{outcomes.count(outcome):6d} {outcome}")
    listing = Path(args.output) / "episodes.txt"
    episodes = sorted(path.parent.resolve() for path in Path(args.output).rglob("state.npy"))
    listing.write_text("\n".join(str(path) for path in episodes) + "\n")
    print(f"{len(episodes)} episodes listed in {listing}")


if __name__ == "__main__":
    main()
