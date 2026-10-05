"""Visualize the 2-slot per-hand OBB the way the OBB head consumes it.

Per frame t, draw AT MOST two OBBs on the head camera:
  * slot 0 (LEFT hand)  = OBB of object_target/left[t]   -> CYAN
  * slot 1 (RIGHT hand) = OBB of object_target/right[t]  -> GREEN
A hand with no target (empty string) or an object with an all-zero OBB draws nothing.
This mirrors build_per_hand_obb in openpi-boxwam's scripts/robotwin/robotwin_hdf5.py exactly.

Usage: python visualize_obb_2slot.py <episode.hdf5> --out <mp4> [--cam head_camera] [--fps 30]
"""
import argparse
import io

import cv2
import h5py
import numpy as np
from PIL import Image

# 8-corner box edges (same ordering as Base_Task.save_obb2d).
BOX_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
LEFT_COLOR = (0, 255, 255)   # cyan  (RGB) -> left hand  (slot 0)
RIGHT_COLOR = (0, 255, 0)    # green (RGB) -> right hand (slot 1)


def _decode(raw):
    # RoboTwin's pkl2hdf5 JPEG-encodes SAPIEN RGB frames via cv2.imencode, which treats input
    # as BGR -> the stored JPEG has R and B swapped vs the true scene. Flip the last axis to
    # restore true RGB (matches _decode_native in openpi-boxwam's scripts/robotwin/robotwin_hdf5.py).
    return np.ascontiguousarray(np.array(Image.open(io.BytesIO(bytes(raw))))[..., ::-1])


def _s(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam", default="head_camera")
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    with h5py.File(a.hdf5, "r") as f:
        rgb = f[f"observation/{a.cam}/rgb"][:]
        n = rgb.shape[0]
        tgtL = [_s(x) for x in f["object_target/left"][:n]]
        tgtR = [_s(x) for x in f["object_target/right"][:n]]
        root = f[f"observation/{a.cam}/proj_3d_obb"]
        per_obj = {k: root[k][:] for k in root}

    sample = _decode(rgb[0])
    h, w = sample.shape[:2]
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))

    def draw(vis, obj, i, color):
        # Match build_per_hand_obb: skip empty target, unknown object, or all-zero OBB.
        if not obj or obj not in per_obj:
            return None
        row = per_obj[obj][i]
        if not np.any(np.abs(row) > 1e-6):
            return None
        pts = row.reshape(8, 2).astype(int)
        for x, y in BOX_EDGES:
            cv2.line(vis, tuple(pts[x]), tuple(pts[y]), color, 2)
        return obj

    for i in range(n):
        vis = _decode(rgb[i]).copy()
        lo = draw(vis, tgtL[i], i, LEFT_COLOR)
        ro = draw(vis, tgtR[i], i, RIGHT_COLOR)
        cv2.putText(vis, f"L(cyan):{lo or '-'}   R(green):{ro or '-'}",
                    (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"saved {a.out} ({n} frames)")


if __name__ == "__main__":
    main()
