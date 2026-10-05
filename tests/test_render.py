import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from boxwam.robotwin.render import EDGES, render_directory  # noqa: E402


def test_edges_connect_neighbouring_corners():
    assert len(EDGES) == 12
    assert all(bin(a ^ b).count("1") == 1 for a, b in EDGES)


def test_render_directory_overlays_boxes(tmp_path):
    videos = tmp_path / "eval"
    videos.mkdir()
    writer = cv2.VideoWriter(str(videos / "episode0.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for _ in range(12):
        writer.write(np.full((48, 64, 3), 255, dtype=np.uint8))
    writer.release()

    boxes_dir = tmp_path / "boxes"
    boxes_dir.mkdir()
    corners = np.array([[x, y] for x in (0.3, 0.7) for y in (0.3, 0.7)] * 2, dtype=np.float32).reshape(8, 2)
    horizon = 4
    boxes = np.tile(corners, (2, horizon, 1, 1, 1))  # (chunks=2, H, K=1, 8, 2)
    np.savez(boxes_dir / "episode_0000.npz", step_indices=np.array([0, 4], dtype=np.int32), boxes=boxes)

    rendered = render_directory(boxes_dir, videos, tmp_path / "out")
    assert [path.name for path in rendered] == ["episode0_boxes.mp4"]
    reader = cv2.VideoCapture(str(rendered[0]))
    assert int(reader.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    reader.set(cv2.CAP_PROP_POS_FRAMES, 2)
    ok, frame = reader.read()
    assert ok and (frame < 200).any(), "a box was drawn on a frame covered by a chunk"
    reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = reader.read()
    assert ok and not (frame < 200).any(), "no prediction covers the very first frame"
