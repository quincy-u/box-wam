"""Episode dataset for Box-WAM training.

Data format
-----------
Every episode is a directory::

    episode_dir/
        video_<camera>.npy   uint8 (T, H, W, 3)    one file per camera, RGB
        state.npy            float (T, state_dim)  proprioceptive state at every step
        action.npy           float (T, action_dim) target position commanded at every step
        proj_3d_obb.npy      float (T, K, 16)      optional: K box slots x 8 corners x (u, v) in
                                                   video pixel coordinates
        valid_idx.npy        int   (M,)            optional: window starts to sample from

Episodes are listed in a text file (one directory per line) or discovered by scanning a root
directory for folders that contain ``state.npy``.

Samples
-------
Each sample is a random window of ``horizon + 1`` steps starting at ``t``::

    images  current frame ``t`` for every camera, uint8 (H, W, 3)
    state   state[t]
    actions action[t : t + H]          the chunk to predict
    boxes   boxes[t + 1 : t + H + 1]   the future box trajectory (flattened to K * 16)

Windows that would run past the end of the episode are padded by repeating the last step.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

BOX_CORNER_DIM = 16


def find_episodes(source: str | Path, task: str | None = None) -> list[Path]:
    """Resolve an episode list file or a root directory into sorted episode directories."""
    source = Path(source).expanduser()
    if source.is_file():
        episodes = [Path(line.strip()).expanduser() for line in source.read_text().splitlines() if line.strip()]
    elif source.is_dir():
        episodes = sorted(path.parent for path in source.rglob("state.npy"))
    else:
        raise FileNotFoundError(f"episode source not found: {source}")
    if task:
        episodes = [path for path in episodes if task in str(path)]
        if not episodes:
            raise ValueError(f"no episode path under {source} contains the task name {task!r}")
    return episodes


def split_episodes(episodes: Sequence[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Deterministically hold out ``ceil(val_fraction * N)`` episodes for validation."""
    if val_fraction <= 0:
        return list(episodes), []
    num_val = min(len(episodes), max(1, int(np.ceil(len(episodes) * val_fraction))))
    order = np.random.RandomState(seed).permutation(len(episodes))
    val = [episodes[i] for i in order[:num_val]]
    train = [episodes[i] for i in order[num_val:]] or list(episodes)
    return train, val


class EpisodeDataset(Dataset):
    def __init__(
        self,
        episodes: Sequence[Path],
        *,
        cameras: Sequence[str],
        horizon: int,
        load_boxes: bool = False,
        use_valid_starts: bool = False,
        preload: bool = False,
    ):
        if not cameras:
            raise ValueError("at least one camera is required")
        self.episodes = [Path(path) for path in episodes]
        self.cameras = list(cameras)
        self.horizon = int(horizon)
        self.load_boxes = load_boxes
        self.use_valid_starts = use_valid_starts
        if use_valid_starts:
            self.episodes = self._episodes_with_valid_starts(self.episodes)
        if not self.episodes:
            raise ValueError("the dataset has no episodes")
        self._cache: dict[Path, dict] | None = None
        if preload:
            self._cache = {path: self._load_episode(path, mmap=False) for path in self.episodes}
            total = sum(array.nbytes for entry in self._cache.values() for array in entry.values())
            logger.info("Preloaded %d episodes (%.1f GB)", len(self.episodes), total / 1024**3)

    # -------------------------------------------------------------- loading
    @staticmethod
    def _episodes_with_valid_starts(episodes: list[Path]) -> list[Path]:
        kept = []
        for path in episodes:
            valid = path / "valid_idx.npy"
            if valid.is_file() and np.load(valid).size > 0:
                kept.append(path)
        logger.info("Kept %d / %d episodes with a non-empty valid_idx.npy", len(kept), len(episodes))
        if not kept:
            raise ValueError("use_valid_starts is set but no episode has a non-empty valid_idx.npy")
        return kept

    def _load_episode(self, path: Path, *, mmap: bool) -> dict:
        mode = "r" if mmap else None
        entry = {
            "state": np.load(path / "state.npy", mmap_mode=mode),
            "action": np.load(path / "action.npy", mmap_mode=mode),
        }
        for camera in self.cameras:
            video = path / f"video_{camera}.npy"
            if not video.is_file():
                raise FileNotFoundError(
                    f"{video} not found (available: {sorted(p.name for p in path.glob('video_*.npy'))})"
                )
            entry[camera] = np.load(video, mmap_mode=mode)
        if self.load_boxes:
            boxes = np.load(path / "proj_3d_obb.npy").astype(np.float32)
            if boxes.ndim == 2:
                boxes = boxes[:, None, :]
            if boxes.ndim != 3 or boxes.shape[-1] != BOX_CORNER_DIM:
                raise ValueError(f"{path}/proj_3d_obb.npy must be (T, K, 16), got {boxes.shape}")
            entry["boxes"] = boxes.reshape(boxes.shape[0], -1)
        if self.use_valid_starts:
            entry["valid_starts"] = np.load(path / "valid_idx.npy").astype(np.int64)
        return entry

    def _episode(self, index: int) -> dict:
        path = self.episodes[index]
        if self._cache is not None:
            return self._cache[path]
        return self._load_episode(path, mmap=True)

    # ------------------------------------------------------------- sampling
    def _window(self, entry: dict, length: int) -> np.ndarray:
        """Indices of one random window of ``horizon + 1`` steps."""
        window = self.horizon + 1
        latest_start = length - window
        if latest_start < 0:  # short episode: use everything and repeat the last step
            return np.minimum(np.arange(window), length - 1)
        if self.use_valid_starts:
            candidates = entry["valid_starts"]
            candidates = candidates[(candidates >= 0) & (candidates <= latest_start)]
            if candidates.size:
                start = int(np.random.choice(candidates))
                return np.arange(start, start + window)
        start = np.random.randint(0, latest_start + 1)
        return np.arange(start, start + window)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict:
        entry = self._episode(index)
        length = min(len(entry["state"]), len(entry["action"]), *(len(entry[camera]) for camera in self.cameras))
        indices = self._window(entry, length)
        current = indices[0]
        sample = {
            "images": [torch.from_numpy(np.array(entry[camera][current], copy=True)) for camera in self.cameras],
            "state": torch.from_numpy(np.array(entry["state"][current], dtype=np.float32)),
            "actions": torch.from_numpy(np.array(entry["action"][indices[:-1]], dtype=np.float32)),
        }
        if self.load_boxes:
            box_indices = np.minimum(indices[1:], len(entry["boxes"]) - 1)
            sample["boxes"] = torch.from_numpy(np.array(entry["boxes"][box_indices], copy=True))
        return sample
