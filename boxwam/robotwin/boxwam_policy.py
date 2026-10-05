"""RoboTwin policy adapter for Box-WAM checkpoints.

RoboTwin's ``script/eval_policy.py`` imports the module named by ``policy_name`` and calls
``get_model(usr_args)``, ``eval(TASK_ENV, model, observation)`` and ``reset_model(model)``.
``usr_args`` is its deploy config merged with command-line overrides; the runner in
``boxwam.robotwin.evaluate`` adds ``policy_config``, the path to our evaluation YAML.

The policy predicts a chunk of ``horizon`` joint targets, executes the first
``num_open_loop_steps`` of them open-loop, then re-plans from a fresh observation. Predicted box
trajectories are collected per episode and written to ``$BOXWAM_BOX_DIR`` when that variable
is set (the evaluation runner draws them onto the rollout videos).
"""

from __future__ import annotations

import atexit
import os
from collections import deque
from pathlib import Path

import numpy as np
import yaml

from boxwam.policy import BoxWAMPolicy


class RoboTwinBoxWAM:
    def __init__(self, policy: BoxWAMPolicy, *, cameras: list[str], num_open_loop_steps: int, box_dir: str | None):
        self.policy = policy
        self.cameras = cameras
        self.num_open_loop_steps = num_open_loop_steps
        self.box_dir = box_dir
        self.queue: deque = deque()
        self.step = 0
        self.episode = 0
        self.latencies_ms: list[float] = []
        self.boxes: list[tuple[int, np.ndarray]] = []

    def plan(self, observation: dict) -> None:
        images = [np.asarray(observation["observation"][camera]["rgb"]) for camera in self.cameras]
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        output = self.policy.act(images, state)
        self.latencies_ms.append(output.latency_ms)
        if output.boxes is not None:
            self.boxes.append((self.step, output.boxes))
        self.queue.extend(output.actions[: self.num_open_loop_steps])

    def save_boxes(self) -> None:
        if not self.box_dir or not self.boxes:
            return
        Path(self.box_dir).mkdir(parents=True, exist_ok=True)
        np.savez(
            Path(self.box_dir) / f"episode_{self.episode:04d}.npz",
            step_indices=np.asarray([step for step, _ in self.boxes], dtype=np.int32),
            boxes=np.stack([boxes for _, boxes in self.boxes]),
        )

    def reset(self) -> None:
        if len(self.latencies_ms) > 1:
            print(
                f"[Box-WAM] mean inference latency after warm-up: {np.mean(self.latencies_ms[1:]):.1f} ms", flush=True
            )
        self.queue.clear()
        self.boxes.clear()
        self.latencies_ms.clear()
        self.step = 0


def get_model(usr_args: dict) -> RoboTwinBoxWAM:
    with open(usr_args["policy_config"]) as handle:
        cfg = yaml.safe_load(handle)
    options = cfg.get("boxwam") or {}
    policy = BoxWAMPolicy.from_checkpoint(
        cfg["checkpoint"],
        device="cuda:0",
        num_steps=int(options.get("flow_steps", 50)),
        dtype=str(options.get("dtype", "bfloat16")),
        compile=bool(options.get("compile", True)),
    )
    cameras = options.get("cameras") or policy.config["data"]["cameras"]
    model = RoboTwinBoxWAM(
        policy,
        cameras=list(cameras),
        num_open_loop_steps=int(cfg.get("num_open_loop_steps", policy.model.horizon)),
        box_dir=os.environ.get("BOXWAM_BOX_DIR"),
    )
    atexit.register(model.save_boxes)  # the last episode may end without a reset_model call
    return model


def eval(TASK_ENV, model: RoboTwinBoxWAM, observation: dict) -> None:  # noqa: A001 - name required by RoboTwin
    if not model.queue:
        model.plan(observation)
    while model.queue:
        TASK_ENV.take_action(np.asarray(model.queue.popleft(), dtype=np.float32), action_type="qpos")
        TASK_ENV.get_obs()  # refresh the frame RoboTwin records to the rollout video
        model.step += 1


def reset_model(model: RoboTwinBoxWAM) -> None:
    """Called by RoboTwin before every episode; only completed episodes are saved and counted."""
    if model.step > 0:
        model.save_boxes()
        model.episode += 1
    model.reset()
