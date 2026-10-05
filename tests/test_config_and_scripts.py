import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import write_episode

from boxwam.config import TrainConfig, apply_overrides, from_dict, load_train_config
from boxwam.training.optim import Schedules

REPO = Path(__file__).resolve().parents[1]
CONFIGS = sorted((REPO / "configs" / "train").rglob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=[str(p.relative_to(REPO)) for p in CONFIGS])
def test_shipped_configs_load(path):
    cfg = load_train_config(path)
    assert cfg.model.box_dim in (0, 16, 32)
    assert cfg.model.box_dim % 16 == 0


def test_unknown_and_missing_keys():
    data = yaml.safe_load((REPO / "configs/train/robotwin.yaml").read_text())
    data["model"]["typo"] = 1
    with pytest.raises(KeyError):
        from_dict(TrainConfig, data)
    del data["model"]["typo"], data["model"]["state_dim"]
    with pytest.raises(KeyError):
        from_dict(TrainConfig, data)


def test_overrides():
    cfg = load_train_config(
        REPO / "configs/train/robotwin.yaml",
        ["model.box_dim=32", "data.task=pick_dual_bottles", "optim.lr=2e-4"],
    )
    assert cfg.model.box_dim == 32 and cfg.data.task == "pick_dual_bottles" and cfg.optim.lr == 2e-4
    assert apply_overrides({}, ["a.b.c=[1, 2]"]) == {"a": {"b": {"c": [1, 2]}}}


def test_schedules():
    cfg = load_train_config(REPO / "configs/train/robotwin.yaml").optim
    schedules = Schedules(cfg)
    assert schedules.lr(0) == pytest.approx(cfg.start_lr)
    assert schedules.lr(schedules.warmup_steps) == pytest.approx(cfg.lr)
    assert schedules.lr(schedules.warmup_steps + 1000) == pytest.approx(cfg.lr), "final_lr == lr keeps it constant"
    assert schedules.weight_decay(10) == pytest.approx(cfg.weight_decay)


def run(script, *args):
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), *map(str, args)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_stats_and_filter_scripts(tmp_path):
    root = tmp_path / "episodes"
    for i in range(3):
        write_episode(root, f"ep{i}", length=20, state_dim=4, boxes=None, seed=i)
    still = write_episode(root, "still", length=20, state_dim=4, boxes=None, seed=9)
    np.save(still / "action.npy", np.tile(np.load(still / "state.npy")[:1], (20, 1)))  # never moves
    np.save(still / "state.npy", np.tile(np.load(still / "state.npy")[:1], (20, 1)))

    out = tmp_path / "stats.npz"
    run("compute_action_stats.py", "--episodes", root, "--horizon", 5, "--gripper-dims", 3, "--output", out)
    stats = np.load(out)
    assert stats["action_q01"].shape == (4,) and stats["state_q99"].shape == (4,)
    assert (stats["action_q01"] <= stats["action_q99"]).all()

    run("filter_static_chunks.py", "--episodes", root, "--horizon", 5, "--gripper-dims", 3)
    assert np.load(still / "valid_idx.npy").size == 0, "a motionless episode has no valid starts"
    moving = np.load(root / "ep0" / "valid_idx.npy")
    assert moving.size == 15 and moving.max() == 14


def test_job_result_dir_picks_own_seed():
    from boxwam.robotwin.evaluate import _job_result_dir

    # sorted by modification time: seed 1's folder was written last
    new = [Path("2026-10-04 10:00:00_seed11"), Path("2026-10-04 10:00:00_seed2"), Path("2026-10-04 10:00:01_seed1")]
    assert _job_result_dir(new, 2).name.endswith("_seed2")
    assert _job_result_dir(new, 11).name.endswith("_seed11")
    assert _job_result_dir(new, 1).name.endswith("_seed1")
    # untagged folders (upstream RoboTwin): newest one, as before
    assert _job_result_dir([Path("a"), Path("b")], 0) == Path("b")
    assert _job_result_dir([], 0) is None
