import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxwam.model.boxwam import BoxWAM  # noqa: E402

TINY = dict(
    encoder_dim=32,
    embed_dim=48,
    depth=2,
    num_heads=2,
    image_size=32,
    patch_size=16,
    num_cameras=1,
    state_dim=3,
    action_dim=4,
    horizon=3,
)


@pytest.fixture
def tiny_kwargs():
    return dict(TINY)


@pytest.fixture
def tiny_model():
    return BoxWAM(**TINY, box_dim=16, action_attend_box=True, box_attend_action=False).eval()


def write_episode(
    root: Path,
    name: str,
    *,
    length: int,
    cameras=("cam",),
    state_dim=3,
    action_dim=4,
    boxes: int | None = 1,
    valid_starts=None,
    seed=0,
) -> Path:
    """Create a synthetic episode directory in the release data format."""
    rng = np.random.default_rng(seed)
    path = root / name
    path.mkdir(parents=True)
    for camera in cameras:
        np.save(path / f"video_{camera}.npy", rng.integers(0, 256, (length, 32, 32, 3), dtype=np.uint8))
    np.save(path / "state.npy", rng.normal(size=(length, state_dim)).astype(np.float32))
    np.save(path / "action.npy", rng.normal(size=(length, action_dim)).astype(np.float32))
    if boxes:
        np.save(path / "proj_3d_obb.npy", (rng.random((length, boxes, 16)) * 32).astype(np.float32))
    if valid_starts is not None:
        np.save(path / "valid_idx.npy", np.asarray(valid_starts, dtype=np.int64))
    return path
