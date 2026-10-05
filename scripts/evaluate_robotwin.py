#!/usr/bin/env python
"""Evaluate a policy in RoboTwin 2.0.

    python scripts/evaluate_robotwin.py --config configs/eval/robotwin_boxwam.yaml \
        --checkpoint checkpoints/robotwin-boxwam-open_laptop/best.pt --tasks open_laptop --gpus 0 1

Every command-line option overrides the corresponding key of the evaluation YAML; the resolved
config is saved next to the results.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxwam.robotwin.evaluate import run_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="evaluation YAML (see configs/eval/)")
    parser.add_argument("--checkpoint")
    parser.add_argument("--label", help="name of the results folder (default: checkpoint directory name)")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--task-config", choices=["demo_clean", "demo_randomized"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--gpus", nargs="+", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    for key in ("checkpoint", "label", "tasks", "task_config", "seeds", "episodes", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.gpus is not None:
        cfg["robotwin"]["gpus"] = args.gpus
    if "checkpoint" in cfg:
        cfg["checkpoint"] = str(Path(cfg["checkpoint"]).expanduser().resolve())

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="boxwam_eval_", delete=False) as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
        resolved = handle.name
    run_evaluation(resolved)


if __name__ == "__main__":
    main()
