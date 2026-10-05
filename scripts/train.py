#!/usr/bin/env python
"""Train a Box-WAM policy.

Examples::

    # single GPU, one RoboTwin task
    python scripts/train.py --config configs/train/robotwin/boxwam.yaml --task open_laptop --devices cuda:0

    # four GPUs, real-world task, override any config value with section.key=value
    python scripts/train.py --config configs/train/real_world/make_coffee_boxwam.yaml \
        --devices cuda:0 cuda:1 cuda:2 cuda:3 --set data.batch_size=96

    # continue an interrupted run
    python scripts/train.py --config ... --task open_laptop --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxwam.config import load_train_config  # noqa: E402
from boxwam.training.train import run_training  # noqa: E402
from boxwam.utils.distributed import launch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="training YAML (see configs/train/)")
    parser.add_argument("--devices", nargs="+", default=["cuda:0"], help="one process is started per device")
    parser.add_argument("--task", help="RoboTwin task name: filters episodes and is appended to run_name")
    parser.add_argument("--run-name", help="override run_name")
    parser.add_argument("--resume", action="store_true", help="continue from <output_dir>/<run_name>/latest.pt")
    parser.add_argument(
        "--set",
        nargs="*",
        action="extend",
        default=[],
        metavar="KEY=VALUE",
        help="config overrides, e.g. optim.epochs=10 (repeatable)",
    )
    parser.add_argument("--port", type=int, default=29500, help="rendezvous port for multi-GPU runs")
    args = parser.parse_args()

    overrides = list(args.set)
    cfg = load_train_config(args.config, overrides)
    if args.task:
        overrides.append(f"data.task={args.task}")
        if not args.run_name and not any(item.startswith("run_name=") for item in args.set):
            overrides.append(f"run_name={cfg.run_name}-{args.task}")
    if args.run_name:
        overrides.append(f"run_name={args.run_name}")
    cfg = load_train_config(args.config, overrides)
    if args.resume:
        latest = cfg.run_dir / "latest.pt"
        if not latest.is_file():
            raise FileNotFoundError(f"--resume requested but {latest} does not exist")
        cfg.resume = str(latest)
    launch(run_training, args.devices, cfg, port=args.port)


if __name__ == "__main__":
    main()
