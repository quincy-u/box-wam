#!/usr/bin/env python
"""Re-render predicted box trajectories onto RoboTwin rollout videos.

The evaluation runner does this automatically (``render_boxes: true``); use this script to render again,
e.g. with a different output folder.

    python scripts/render_boxes.py --boxes results/<label>/<task_config>/boxes/<task>_seed0 \
        --videos <RoboTwin result folder with episode<N>.mp4> --output renders/<task>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxwam.robotwin.render import render_directory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boxes", required=True, help="directory with episode_XXXX.npz files")
    parser.add_argument("--videos", required=True, help="RoboTwin result directory with episodeN.mp4 files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for path in render_directory(Path(args.boxes), Path(args.videos).expanduser(), Path(args.output)):
        print(path)


if __name__ == "__main__":
    main()
