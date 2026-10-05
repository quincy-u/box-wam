"""Draw predicted box trajectories onto RoboTwin rollout videos.

The RoboTwin adapters save one ``episode_XXXX.npz`` per episode (``step_indices`` and ``boxes`` of shape
``(N, H, K, 8, 2)`` in image fractions) and RoboTwin records ``episode<N>.mp4`` next to ``_result.txt``.
For every frame the box predicted for that step by the most recent chunk is drawn; the colour encodes how far
into the chunk the frame is (green = just predicted, red = end of the horizon). Requires OpenCV.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

# Corner i has sign pattern ((i >> 2) & 1, (i >> 1) & 1, i & 1) along the box axes, so the twelve edges
# connect corners that differ in exactly one bit.
EDGES = [(i, i ^ (1 << bit)) for bit in range(3) for i in range(8) if not (i >> bit) & 1]


def step_color(k: int, horizon: int) -> tuple[int, int, int]:
    import cv2

    hue = int(round(60.0 * (1.0 - k / max(1, horizon - 1))))  # OpenCV hue: 60 green -> 0 red
    bgr = cv2.cvtColor(np.array([[[hue, 255, 230]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_boxes(frame: np.ndarray, boxes: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
    import cv2

    height, width = frame.shape[:2]
    for corners in boxes:  # (8, 2) image fractions
        if not np.any(corners):
            continue
        points = np.round(corners * [width, height]).astype(np.int32)
        for a, b in EDGES:
            cv2.line(frame, tuple(points[a]), tuple(points[b]), color, thickness)
        for point in points:
            cv2.circle(frame, tuple(point), thickness, color, -1)


class VideoWriter:
    """Write H.264 through ffmpeg when available (browser friendly), else fall back to OpenCV."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        import cv2

        path.parent.mkdir(parents=True, exist_ok=True)
        self.process = self.writer = None
        if shutil.which("ffmpeg"):
            command = [
                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{size[0]}x{size[1]}", "-r", f"{fps:.3f}", "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path),
            ]  # fmt: skip
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
        else:
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame: np.ndarray) -> None:
        if self.process is not None:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.writer.write(frame)

    def close(self) -> None:
        if self.process is not None:
            self.process.stdin.close()
            self.process.wait()
        elif self.writer is not None:
            self.writer.release()


def render_episode(video_path: Path, chunks: dict, output_path: Path) -> int:
    """Overlay the predicted boxes of one episode; returns the number of frames written."""
    import cv2

    steps, boxes = np.asarray(chunks["step_indices"]), np.asarray(chunks["boxes"])  # boxes: (N, H, K, 8, 2)
    horizon = boxes.shape[1]
    reader = cv2.VideoCapture(str(video_path))
    fps = reader.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    frame_index = 0
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        # The chunk predicted at step s covers steps s + 1 ... s + H.
        candidates = np.nonzero((steps < frame_index) & (frame_index <= steps + horizon))[0]
        if candidates.size:
            chunk = candidates[-1]
            offset = frame_index - int(steps[chunk]) - 1
            draw_boxes(frame, boxes[chunk, offset], step_color(offset, horizon))
        if writer is None:
            writer = VideoWriter(output_path, fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        frame_index += 1
    reader.release()
    if writer is not None:
        writer.close()
    return frame_index


def render_directory(boxes_dir: Path, videos_dir: Path, output_dir: Path) -> list[Path]:
    """Render every ``episode_XXXX.npz`` in ``boxes_dir`` onto ``episode<N>.mp4`` in ``videos_dir``."""
    rendered = []
    for npz_path in sorted(Path(boxes_dir).glob("episode_*.npz")):
        episode = int(re.search(r"episode_(\d+)", npz_path.stem).group(1))
        video_path = Path(videos_dir) / f"episode{episode}.mp4"
        if not video_path.is_file():
            continue
        output_path = Path(output_dir) / f"episode{episode}_boxes.mp4"
        if render_episode(video_path, np.load(npz_path), output_path):
            rendered.append(output_path)
    return rendered
