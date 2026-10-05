"""Run RoboTwin 2.0 evaluations of Box-WAM or pi0.5 policies and summarise success rates.

RoboTwin lives in its own conda environment (SAPIEN, CuRobo, ...), so every (task, seed) job
is a subprocess running RoboTwin's ``script/eval_policy.py`` with that interpreter. Jobs are
spread over the GPUs listed in the config. This repository is put on ``PYTHONPATH`` so RoboTwin
can import the policy adapters in ``boxwam.robotwin``.

Results are read back from RoboTwin's ``eval_result/`` tree and written as
``summary.json`` / ``summary.md`` under ``output_dir``; the predicted box trajectories are saved under
``boxes/`` and drawn onto the rollout videos under ``videos/``.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

import yaml

from boxwam.utils.logging import get_logger

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_CONFIG = Path(__file__).resolve().parent / "deploy_policy.yml"
POLICY_MODULES = {"boxwam": "boxwam.robotwin.boxwam_policy", "pi05": "boxwam.robotwin.pi05_policy"}

logger = get_logger("boxwam.robotwin")


@dataclass
class Job:
    task: str
    seed: int


@dataclass
class JobResult:
    task: str
    seed: int
    success_rate: float | None
    return_code: int
    minutes: float
    result_dir: str | None
    videos: list[str]


def _expand(path: str, base: Path | None = None) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    return (base / path if base else path).resolve()


def _result_dirs(root: Path, task: str, policy_module: str, task_config: str, label: str) -> list[Path]:
    base = root / "eval_result" / task / policy_module / task_config / label
    if not base.is_dir():
        return []
    return sorted((path for path in base.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime)


def _job_result_dir(new_dirs: list[Path], seed: int) -> Path | None:
    """The newest result folder of this job among the folders created while it ran.

    Concurrent runs of the same task (other seeds) create folders in the same place at the same time; RoboTwin-boxwam
    tags each folder with its seed (``<timestamp>_seed<N>``), so prefer this seed's folders when they exist.
    """
    tagged = [path for path in new_dirs if path.name.endswith(f"_seed{seed}")]
    candidates = tagged or new_dirs
    return candidates[-1] if candidates else None


def _parse_success_rate(result_dir: Path | None) -> float | None:
    """The last numeric line of ``_result.txt`` is the success rate in ``[0, 1]``."""
    if result_dir is None or not (result_dir / "_result.txt").is_file():
        return None
    values = []
    for line in (result_dir / "_result.txt").read_text().splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return values[-1] if values else None


STALLED = -2


def _run_with_watchdog(command: list[str], cwd: Path, env: dict, log_path: Path, options: dict) -> int:
    """Run one RoboTwin evaluation, killing it when its log stops growing or it exceeds the time limit.

    The simulator occasionally hangs inside a render call; ``stall_minutes`` (default 10) bounds how long
    the log may stay silent before the process group is killed and ``STALLED`` is returned so the caller
    can retry. ``timeout_minutes`` bounds the total wall time of one attempt.
    """
    stall = float(options.get("stall_minutes", 10)) * 60
    timeout = options.get("timeout_minutes")
    with open(log_path, "a") as log:
        log.write(" ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        started = last_growth = time.time()
        last_size = log_path.stat().st_size
        while process.poll() is None:
            time.sleep(15)
            size = log_path.stat().st_size
            if size != last_size:
                last_size, last_growth = size, time.time()
            reason = None
            if time.time() - last_growth > stall:
                reason, code = f"no log output for {stall / 60:.1f} minutes", STALLED
            elif timeout and time.time() - started > timeout * 60:
                reason, code = f"exceeded {timeout} minutes", -1
            if reason:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                log.write(f"\nKILLED: {reason}\n")
                return code
        return process.returncode


def run_job(cfg: dict, config_path: Path, job: Job, gpu: int, output_dir: Path) -> JobResult:
    root = _expand(cfg["robotwin"]["root"], REPO_ROOT)
    python = cfg["robotwin"].get("python") or sys.executable
    python = str(_expand(python)) if ("/" in python or python.startswith("~")) else (shutil.which(python) or python)
    policy_module = POLICY_MODULES[cfg["policy"]]
    task_config = str(cfg.get("task_config", "demo_randomized"))
    label = str(cfg["label"])

    overrides = {
        "policy_name": policy_module,
        "task_name": job.task,
        "task_config": task_config,
        "ckpt_setting": label,
        "seed": job.seed,
        "test_num": int(cfg.get("episodes", 100)),
        "instruction_type": str(cfg.get("instruction_type", "unseen")),
        "policy_config": str(config_path),
    }
    command = [python, "script/eval_policy.py", "--config", str(DEPLOY_CONFIG), "--overrides"]
    for key, value in overrides.items():
        command += [key, str(value)]

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(REPO_ROOT)
    # RoboTwin needs tools from its own environment at run time (nvcc and ninja for curobo's kernel builds, ffmpeg
    # for the rollout videos); put that environment's bin/ first even when this runner runs in another one.
    if Path(python).is_absolute():
        env["PATH"] = f"{Path(python).parent}{os.pathsep}{env.get('PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["EGL_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    box_dir = output_dir / "boxes" / f"{job.task}_seed{job.seed}"
    if cfg.get("save_boxes", True):
        env["BOXWAM_BOX_DIR"] = str(box_dir)
    for key, value in (cfg["robotwin"].get("env") or {}).items():
        env[key] = str(value)

    before = {path.resolve() for path in _result_dirs(root, job.task, policy_module, task_config, label)}
    log_path = output_dir / "logs" / f"{job.task}_seed{job.seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")
    logger.info("[gpu %d] %s seed %d -> %s", gpu, job.task, job.seed, log_path)
    start = time.time()
    retries = int(cfg["robotwin"].get("retries", 2))
    for attempt in range(retries + 1):
        code = _run_with_watchdog(command, root, env, log_path, cfg["robotwin"])
        if code != STALLED:
            break
        logger.warning("[gpu %d] %s seed %d stalled; retrying (%d/%d)", gpu, job.task, job.seed, attempt + 1, retries)
    new_dirs = [
        path for path in _result_dirs(root, job.task, policy_module, task_config, label) if path.resolve() not in before
    ]
    result_dir = _job_result_dir(new_dirs, job.seed)
    rate = _parse_success_rate(result_dir)
    logger.info("[gpu %d] %s seed %d: success rate %s (exit %d)", gpu, job.task, job.seed, rate, code)
    videos = []
    if cfg.get("save_boxes", True) and cfg.get("render_boxes", True) and result_dir is not None:
        videos = render_box_videos(box_dir, result_dir, output_dir / "videos" / f"{job.task}_seed{job.seed}")
    return JobResult(
        job.task, job.seed, rate, code, (time.time() - start) / 60, str(result_dir) if result_dir else None, videos
    )


def render_box_videos(box_dir: Path, result_dir: Path, output_dir: Path) -> list[str]:
    """Draw the saved box predictions onto RoboTwin's rollout videos (needs OpenCV)."""
    try:
        import cv2  # noqa: F401

        from boxwam.robotwin.render import render_directory
    except ImportError:
        logger.warning("OpenCV is not installed; skipping box rendering (pip install -e '.[data]')")
        return []
    rendered = render_directory(box_dir, result_dir, output_dir)
    if rendered:
        logger.info("rendered %d video(s) with predicted boxes -> %s", len(rendered), output_dir)
    return [str(path) for path in rendered]


def summarise(results: list[JobResult], cfg: dict, output_dir: Path) -> dict:
    per_task = {}
    for task in sorted({result.task for result in results}):
        rates = [result.success_rate for result in results if result.task == task and result.success_rate is not None]
        per_task[task] = {
            "mean": statistics.mean(rates) if rates else None,
            "std": statistics.stdev(rates) if len(rates) > 1 else 0.0,
            "rates": rates,
        }
    means = [entry["mean"] for entry in per_task.values() if entry["mean"] is not None]
    summary = {
        "policy": cfg["policy"],
        "label": cfg["label"],
        "checkpoint": cfg.get("checkpoint"),
        "task_config": cfg.get("task_config", "demo_randomized"),
        "episodes_per_seed": int(cfg.get("episodes", 100)),
        "average_success_rate": statistics.mean(means) if means else None,
        "tasks": per_task,
        "jobs": [result.__dict__ for result in results],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [f"# {cfg['label']} ({summary['task_config']})", "", "| task | success rate | seeds |", "|---|---|---|"]
    for task, entry in per_task.items():
        mean = "n/a" if entry["mean"] is None else f"{100 * entry['mean']:.1f} ± {100 * entry['std']:.1f}"
        lines.append(f"| {task} | {mean} | {len(entry['rates'])} |")
    if summary["average_success_rate"] is not None:
        lines += ["", f"Average over tasks: **{100 * summary['average_success_rate']:.1f}%**"]
    if any(result.videos for result in results):
        lines += ["", f"Rollout videos with predicted boxes: `{output_dir / 'videos'}`"]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def default_label(checkpoint: str | None) -> str | None:
    """Results folder name for a checkpoint: its file name, or the run folder for generic names like best.pt."""
    if not checkpoint:
        return None
    path = Path(checkpoint)
    generic = path.stem in ("best", "latest") or path.stem.startswith("epoch_")
    return path.parent.name if generic else path.stem


def run_evaluation(config_path: str | Path) -> dict:
    config_path = _expand(str(config_path))
    with open(config_path) as handle:
        cfg = yaml.safe_load(handle)
    if cfg["policy"] not in POLICY_MODULES:
        raise ValueError(f"policy must be one of {sorted(POLICY_MODULES)}")
    root = _expand(cfg["robotwin"]["root"], REPO_ROOT)
    if not (root / "script" / "eval_policy.py").is_file():
        raise FileNotFoundError(f"RoboTwin checkout not found at {root} (robotwin.root)")
    if cfg["policy"] == "boxwam" and not _expand(cfg["checkpoint"]).is_file():
        raise FileNotFoundError(f"checkpoint not found: {cfg['checkpoint']}")
    cfg.setdefault("label", default_label(cfg.get("checkpoint")) or cfg["policy"])

    output_dir = _expand(cfg.get("output_dir") or f"results/{cfg['label']}/{cfg.get('task_config', 'demo_randomized')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, output_dir / "eval_config.yaml")

    jobs = [Job(task, int(seed)) for task in cfg["tasks"] for seed in cfg.get("seeds", [0])]
    gpus: Queue = Queue()
    for gpu in cfg["robotwin"].get("gpus", [0]):
        gpus.put(int(gpu))

    def worker(job: Job) -> JobResult:
        gpu = gpus.get()
        try:
            return run_job(cfg, config_path, job, gpu, output_dir)
        finally:
            gpus.put(gpu)

    with ThreadPoolExecutor(max_workers=gpus.qsize()) as pool:
        results = list(pool.map(worker, jobs))
    summary = summarise(results, cfg, output_dir)
    logger.info("Summary written to %s", output_dir / "summary.md")
    print((output_dir / "summary.md").read_text(), file=sys.stdout)
    return summary
