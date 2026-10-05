"""Helpers for reading RoboTwin-boxwam HDF5 episodes (OBB annotations, JPEG frames) into LeRobot.

Used by ``convert_robotwin_to_lerobot.py``.
"""

from __future__ import annotations

import io

import h5py
import numpy as np
from PIL import Image

# Per-object dim: 8 corners * (u, v).
OBB_PER_OBJECT_DIM = 16
# Always two OBB slots: (left, right).
NUM_OBB_SLOTS = 2


def build_per_hand_obb(
    f: h5py.File, cam_group_name: str, n_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compose ``[T, 2, 16]`` raw-pixel OBB tensor and ``[T, 2]`` validity mask, where slot
    0 = left-hand-targeted OBB, slot 1 = right-hand-targeted OBB.

    A hand "has a target" at frame t iff ``object_target/<side>[t]`` is a non-empty string
    AND the named object has a non-zero OBB in ``observation/<cam>/proj_3d_obb/<obj>``.
    """
    targets = {
        "left": np.asarray(f["object_target/left"][:n_frames]),
        "right": np.asarray(f["object_target/right"][:n_frames]),
    }
    obb_root = f[f"observation/{cam_group_name}/proj_3d_obb"]
    # Cache the per-object OBB stacks (avoid repeated HDF5 reads inside the per-frame loop).
    per_obj_obb: dict[str, np.ndarray] = {}
    for name in obb_root:
        per_obj_obb[name] = np.asarray(obb_root[name][:n_frames], dtype=np.float32)
        if per_obj_obb[name].shape != (n_frames, OBB_PER_OBJECT_DIM):
            raise ValueError(
                f"obb[{name}] shape {per_obj_obb[name].shape} != ({n_frames}, {OBB_PER_OBJECT_DIM})"
            )

    obb_px = np.zeros((n_frames, NUM_OBB_SLOTS, OBB_PER_OBJECT_DIM), dtype=np.float32)
    valid = np.zeros((n_frames, NUM_OBB_SLOTS), dtype=np.float32)
    for slot, side in enumerate(("left", "right")):
        for t in range(n_frames):
            raw = targets[side][t]
            name = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
            if not name:
                continue
            if name not in per_obj_obb:
                # Target named, but no OBB stream for it — treat as missing.
                continue
            row = per_obj_obb[name][t]
            if not np.any(np.abs(row) > 1e-6):
                continue
            obb_px[t, slot] = row
            valid[t, slot] = 1.0
    return obb_px, valid




def _decode_native(rgb_bytes_arr: h5py.Dataset) -> tuple[np.ndarray, int, int]:
    """Decode JPEG-encoded HDF5 rgb to uint8 ``[T, H, W, 3]`` at native resolution.

    RoboTwin's pkl2hdf5 encodes JPEGs via ``cv2.imencode`` on RGB-ordered SAPIEN frames.
    ``cv2.imencode`` treats input as BGR, so every JPEG in the HDF5 stores R and B swapped
    relative to the actual rendered scene. Decoding with PIL ``.convert("RGB")`` yields the
    swapped data; flipping the last axis here restores the true scene colors so the
    LeRobot mp4 (and the OBB visualizer downstream) renders correctly.
    """
    first = np.asarray(Image.open(io.BytesIO(rgb_bytes_arr[0])).convert("RGB"), dtype=np.uint8)
    first = first[..., ::-1]
    h, w, _ = first.shape
    t = rgb_bytes_arr.shape[0]
    out = np.empty((t, h, w, 3), dtype=np.uint8)
    out[0] = first
    for i in range(1, t):
        img = Image.open(io.BytesIO(rgb_bytes_arr[i])).convert("RGB")
        if img.size != (w, h):
            img = img.resize((w, h), Image.BILINEAR)
        out[i] = np.asarray(img, dtype=np.uint8)[..., ::-1]
    return out, h, w


def _patch_lerobot_image_writer() -> None:
    """Make LeRobot's PNG writer fail loudly + retry on transient I/O errors.

    Upstream ``write_image`` (``lerobot.common.datasets.image_writer.write_image``)
    catches every exception and only prints a warning. Combined with the worker
    always calling ``queue.task_done()`` regardless of write outcome, this means
    ``wait_until_done()`` returns success even when PNGs failed to land on disk —
    and ``save_episode`` then blows up trying to read the missing files, corrupting
    the dataset state and silently dropping every subsequent episode.

    This patch wraps ``write_image`` with bounded retries (helpful on shared NFS
    where transient ``OSError``s happen under load) and re-raises after the final
    attempt so the worker thread terminates noisily instead of silently producing
    a half-written episode.
    """
    import time
    import threading
    from lerobot.common.datasets import image_writer as _iw

    _orig_write = _iw.write_image
    # Note: don't lock around the write itself — the whole point of the writer pool
    # is concurrency. We just want loud failures, not serial writes.

    def _retry_write(image, fpath):
        last_exc = None
        for attempt in range(4):
            try:
                _orig_write(image, fpath)
                return
            except Exception as e:  # noqa: BLE001
                last_exc = e
                time.sleep(0.1 * (attempt + 1))
        # After 4 attempts, surface the failure so wait_until_done sees it.
        raise OSError(f"write_image failed for {fpath} after 4 attempts: {last_exc}")

    _iw.write_image = _retry_write

    # The worker loop calls write_image then task_done(); to also propagate failures
    # so wait_until_done raises, wrap worker_thread_loop to set a "had_error" flag.
    _had_error = {"flag": False, "first": None, "lock": threading.Lock()}

    def _new_worker_loop(q):
        import queue as _q  # noqa: F401
        while True:
            item = q.get()
            try:
                if item is None:
                    return
                image_array, fpath = item
                try:
                    _retry_write(image_array, fpath)
                except Exception as e:  # noqa: BLE001
                    with _had_error["lock"]:
                        if not _had_error["flag"]:
                            _had_error["flag"] = True
                            _had_error["first"] = (str(fpath), str(e))
            finally:
                q.task_done()

    _iw.worker_thread_loop = _new_worker_loop

    # Hook wait_until_done so a single failed PNG aborts the whole conversion run.
    from lerobot.common.datasets.image_writer import AsyncImageWriter as _AIW
    _orig_wait = _AIW.wait_until_done

    def _wait_and_check(self):
        _orig_wait(self)
        if _had_error["flag"]:
            path, exc = _had_error["first"]
            _had_error["flag"] = False  # reset for next episode
            _had_error["first"] = None
            raise OSError(f"image_writer write failed during episode (first error: {path}: {exc})")

    _AIW.wait_until_done = _wait_and_check
