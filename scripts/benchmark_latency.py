#!/usr/bin/env python
"""Measure end-to-end inference latency of a checkpoint (image preprocessing + encoder + flow).

python scripts/benchmark_latency.py --checkpoint release/open_laptop.pt --steps 50 --compile
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxwam.policy import BoxWAMPolicy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=50, help="Euler steps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--image-size", type=int, nargs=2, default=[240, 320], metavar=("H", "W"), help="camera resolution"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    policy = BoxWAMPolicy.from_checkpoint(
        args.checkpoint, device=args.device, num_steps=args.steps, dtype=args.dtype, compile=args.compile
    )
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, (*args.image_size, 3), dtype=np.uint8) for _ in range(policy.model.num_cameras)]
    state = rng.normal(size=policy.model.state_dim).astype(np.float32)
    for _ in range(args.warmup):
        policy.act(images, state)
    latencies = [policy.act(images, state).latency_ms for _ in range(args.iters)]
    model = policy.model
    print(f"{model.num_cameras} camera(s), horizon {model.horizon}, box_dim {model.box_dim}")
    print(f"{args.steps} Euler steps, {args.dtype}, compile={args.compile}")
    print(
        f"mean {statistics.mean(latencies):.1f} ms | median {statistics.median(latencies):.1f} ms | "
        f"p95 {np.percentile(latencies, 95):.1f} ms | {1000 / statistics.mean(latencies):.1f} Hz"
    )


if __name__ == "__main__":
    main()
