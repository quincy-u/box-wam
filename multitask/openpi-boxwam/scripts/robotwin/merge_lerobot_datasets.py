#!/usr/bin/env python3
"""Merge per-(task, config) LeRobot v2.1 datasets into a SINGLE LeRobot v2.1 dataset that openpi can
load via ``LeRobotDataset(repo_id)``.

Input: the parts written by ``convert_robotwin_to_lerobot.py``, laid out as ``<src-root>/<split>/<task>/``
(each with its own ``meta/`` + ``data/chunk-000/*.parquet`` + ``videos/chunk-000/<cam>/*.mp4``), e.g.
``<src-root>/demo_clean/adjust_bottle`` and ``<src-root>/demo_randomized/adjust_bottle``. openpi's data loader
(``create_torch_dataset``) instantiates exactly ONE ``LeRobotDataset`` from a single ``repo_id``, so we
physically concatenate all episodes into one merged dataset:

    uv run scripts/robotwin/merge_lerobot_datasets.py --src-root ~/robotwin_parts \
        --out-root ~/robotwin_merged --repo-id <user>/robotwin_eef_obb

This is a metadata + file-copy merge (NO video re-encode): the source mp4s are copied verbatim and
only the per-frame index bookkeeping in the parquet is rewritten. Specifically, for every episode we
renumber ``episode_index`` -> global id, ``index`` -> global running frame id, and ``task_index`` ->
a global, string-deduplicated task id. Videos are copied to the merged tree under the correct
1000-episode chunk. The merged ``meta/{info,episodes,episodes_stats,tasks}.jsonl/json`` are rewritten
with global totals. Extra per-frame columns (``observation.obb_head``) are carried through verbatim.

State/action are 16-dim end-effector (``left_{x,y,z,q1..q4,gripper}`` + right); we DON'T touch their
values -- openpi pads them to the model action dim and computes its own norm stats downstream.

The merged dataset is symlinked into ``$HF_LEROBOT_HOME`` (default ``~/.cache/huggingface/lerobot``)
so openpi resolves it by repo_id.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa

CHUNK_SIZE = 1000  # LeRobot default; merged dataset re-chunks episodes at this size.
VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def find_task_dirs(root: Path, splits: list[str]) -> list[Path]:
    """Return sorted per-task dataset dirs (each contains meta/, data/, videos/)."""
    task_dirs: list[Path] = []
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            print(f"[warn] split dir missing: {split_dir}", file=sys.stderr)
            continue
        for d in sorted(split_dir.iterdir()):
            if (d / "meta" / "info.json").is_file():
                task_dirs.append(d)
    return task_dirs


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_of(ep_idx: int) -> str:
    return f"chunk-{ep_idx // CHUNK_SIZE:03d}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-root", required=True, help="directory holding <split>/<task>/ LeRobot parts.")
    p.add_argument("--out-root", required=True, help="output directory of the merged dataset.")
    p.add_argument("--repo-id", required=True, help="repo_id key used to symlink into $HF_LEROBOT_HOME.")
    p.add_argument("--splits", default="demo_clean,demo_randomized",
                   help="comma-separated split subdirs under src-root to merge.")
    p.add_argument("--copy-workers", type=int, default=16, help="parallel mp4 copy threads.")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="include a task even if #parquet/#mp4 don't match its info.json (default: error).")
    p.add_argument("--exclude-file", default="",
                   help="path to a file of '<split>/<task>\\t<local_ep>' lines; those episodes are "
                        "skipped (e.g. episodes with a corrupt source video).")
    p.add_argument("--link-hf-home", action="store_true", default=True,
                   help="symlink $HF_LEROBOT_HOME/<repo_id> -> out-root after merge.")
    p.add_argument("--dry-run", action="store_true", help="scan + validate only; write nothing.")
    args = p.parse_args(argv)

    src_root = Path(args.src_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    # Optional per-episode exclusion set (keyed by source-relative task dir + local episode index).
    exclude: set[tuple[str, int]] = set()
    if args.exclude_file:
        with open(args.exclude_file) as f:
            for line in f:
                if not line.strip():
                    continue
                tr, le = line.rstrip("\n").split("\t")
                exclude.add((tr, int(le)))
        print(f"Loaded {len(exclude)} episodes to exclude from {args.exclude_file}")

    task_dirs = find_task_dirs(src_root, splits)
    if not task_dirs:
        print(f"No task datasets found under {src_root} for splits {splits}", file=sys.stderr)
        return 2
    print(f"Found {len(task_dirs)} per-task datasets across splits {splits}")

    # Reference feature/fps from the first task; assert the rest match.
    ref_info = json.load(open(task_dirs[0] / "meta" / "info.json"))
    ref_fps = ref_info["fps"]
    ref_features = ref_info["features"]

    if not args.dry_run:
        if out_root.exists():
            print(f"Removing existing out-root: {out_root}")
            shutil.rmtree(out_root)
        (out_root / "meta").mkdir(parents=True, exist_ok=True)

    # Global running state.
    global_ep = 0
    global_frame = 0
    task_str_to_idx: dict[str, int] = {}
    merged_tasks: list[dict] = []          # {"task_index", "task"}
    merged_episodes: list[dict] = []       # {"episode_index", "tasks", "length"}
    merged_ep_stats: list[dict] = []       # {"episode_index", "stats"}
    copy_jobs: list[tuple[Path, Path]] = []  # (src_mp4, dst_mp4)

    def get_task_idx(task_str: str) -> int:
        if task_str not in task_str_to_idx:
            idx = len(task_str_to_idx)
            task_str_to_idx[task_str] = idx
            merged_tasks.append({"task_index": idx, "task": task_str})
        return task_str_to_idx[task_str]

    n_bad = 0
    n_excluded = 0
    for ti, tdir in enumerate(task_dirs):
        task_rel = os.path.relpath(str(tdir), str(src_root))
        info = json.load(open(tdir / "meta" / "info.json"))
        assert info["fps"] == ref_fps, f"fps mismatch in {tdir}: {info['fps']} != {ref_fps}"
        local_tasks = {t["task_index"]: t["task"] for t in load_jsonl(tdir / "meta" / "tasks.jsonl")}
        episodes = load_jsonl(tdir / "meta" / "episodes.jsonl")
        ep_stats = {e["episode_index"]: e["stats"] for e in load_jsonl(tdir / "meta" / "episodes_stats.jsonl")}
        n_eps = info["total_episodes"]

        # Completeness check: every episode needs 1 parquet + len(VIDEO_KEYS) mp4.
        pq_dir = tdir / "data" / "chunk-000"
        n_pq = len(list(pq_dir.glob("episode_*.parquet"))) if pq_dir.is_dir() else 0
        n_mp4 = sum(
            len(list((tdir / "videos" / "chunk-000" / vk).glob("episode_*.mp4")))
            for vk in VIDEO_KEYS
            if (tdir / "videos" / "chunk-000" / vk).is_dir()
        )
        if n_pq != n_eps or n_mp4 != n_eps * len(VIDEO_KEYS):
            msg = (f"[{'skip' if args.allow_incomplete else 'BAD '}] {tdir.name}: "
                   f"parquet={n_pq}/{n_eps} mp4={n_mp4}/{n_eps * len(VIDEO_KEYS)}")
            print(msg, file=sys.stderr)
            n_bad += 1
            if args.allow_incomplete:
                continue
            print("Refusing to merge an incomplete dataset (pass --allow-incomplete to skip bad tasks).",
                  file=sys.stderr)
            return 3

        for local_ep in range(n_eps):
            if (task_rel, local_ep) in exclude:
                n_excluded += 1
                continue
            src_pq = pq_dir / f"episode_{local_ep:06d}.parquet"
            ep_meta = next((e for e in episodes if e["episode_index"] == local_ep), None)
            if ep_meta is None:
                raise RuntimeError(f"episode {local_ep} missing from {tdir}/meta/episodes.jsonl")
            new_ep = global_ep
            length = ep_meta["length"]

            # --- rewrite parquet index columns ---
            table = pq.read_table(src_pq)
            n_rows = table.num_rows
            assert n_rows == length, f"{src_pq}: rows {n_rows} != episodes.jsonl length {length}"
            local_task_idx = table.column("task_index").to_pylist()
            new_task_idx = [get_task_idx(local_tasks[t]) for t in local_task_idx]
            new_index = list(range(global_frame, global_frame + n_rows))
            new_epidx = [new_ep] * n_rows
            table = table.set_column(table.schema.get_field_index("task_index"),
                                     "task_index", pa.array(new_task_idx, type=table.schema.field("task_index").type))
            table = table.set_column(table.schema.get_field_index("index"),
                                     "index", pa.array(new_index, type=table.schema.field("index").type))
            table = table.set_column(table.schema.get_field_index("episode_index"),
                                     "episode_index", pa.array(new_epidx, type=table.schema.field("episode_index").type))

            if not args.dry_run:
                dst_pq = out_root / "data" / chunk_of(new_ep) / f"episode_{new_ep:06d}.parquet"
                dst_pq.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, dst_pq)
                for vk in VIDEO_KEYS:
                    src_mp4 = tdir / "videos" / "chunk-000" / vk / f"episode_{local_ep:06d}.mp4"
                    dst_mp4 = out_root / "videos" / chunk_of(new_ep) / vk / f"episode_{new_ep:06d}.mp4"
                    copy_jobs.append((src_mp4, dst_mp4))

            # --- merged meta accumulation ---
            ep_tasks = [local_tasks[t] for t in sorted(set(local_task_idx))]
            for t in ep_tasks:
                get_task_idx(t)
            merged_episodes.append({"episode_index": new_ep, "tasks": ep_tasks, "length": length})
            if local_ep in ep_stats:
                merged_ep_stats.append({"episode_index": new_ep, "stats": ep_stats[local_ep]})

            global_ep += 1
            global_frame += n_rows

        print(f"[{ti + 1:3d}/{len(task_dirs)}] {tdir.name}: +{n_eps} eps "
              f"(total eps={global_ep}, frames={global_frame}, tasks={len(merged_tasks)})")

    print(f"\nMerged totals: episodes={global_ep} frames={global_frame} tasks={len(merged_tasks)} "
          f"(bad/skipped tasks={n_bad}, excluded episodes={n_excluded})")

    if args.dry_run:
        print("Dry run: no files written.")
        return 0

    # --- copy videos in parallel ---
    print(f"Copying {len(copy_jobs)} mp4 files with {args.copy_workers} workers...")

    def _copy(job):
        s, d = job
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s, d)

    done = 0
    with ThreadPoolExecutor(max_workers=args.copy_workers) as ex:
        for _ in ex.map(_copy, copy_jobs):
            done += 1
            if done % 5000 == 0:
                print(f"  copied {done}/{len(copy_jobs)} mp4")
    print(f"  copied {done}/{len(copy_jobs)} mp4")

    # --- write merged meta ---
    meta_dir = out_root / "meta"
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for t in merged_tasks:
            f.write(json.dumps(t) + "\n")
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for e in merged_episodes:
            f.write(json.dumps(e) + "\n")
    with open(meta_dir / "episodes_stats.jsonl", "w") as f:
        for e in merged_ep_stats:
            f.write(json.dumps(e) + "\n")

    total_chunks = math.ceil(global_ep / CHUNK_SIZE)
    merged_info = dict(ref_info)
    merged_info.update({
        "total_episodes": global_ep,
        "total_frames": global_frame,
        "total_tasks": len(merged_tasks),
        "total_videos": global_ep * len(VIDEO_KEYS),
        "total_chunks": total_chunks,
        "chunks_size": CHUNK_SIZE,
        "fps": ref_fps,
        "splits": {"train": f"0:{global_ep}"},
        "features": ref_features,
    })
    with open(meta_dir / "info.json", "w") as f:
        json.dump(merged_info, f, indent=4)

    print(f"Wrote merged dataset to {out_root}")

    # --- symlink into HF_LEROBOT_HOME so openpi resolves by repo_id ---
    if args.link_hf_home:
        hf_home = Path(os.environ.get("HF_LEROBOT_HOME",
                                      str(Path("~/.cache/huggingface/lerobot").expanduser())))
        link = hf_home / args.repo_id
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            if link.is_symlink():
                link.unlink()
            else:
                print(f"[warn] {link} exists and is not a symlink; leaving it. Point it manually.",
                      file=sys.stderr)
                return 0
        link.symlink_to(out_root)
        print(f"Symlinked {link} -> {out_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
