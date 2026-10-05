"""RoboTwin policy adapter for pi0.5 + Box-WAM served by openpi.

The model runs in a separate process: ``scripts/serve_policy.py`` from our openpi fork
(``multitask/README.md``). This adapter only needs ``openpi_client`` (a small pure-Python
websocket client) inside the RoboTwin environment and talks to the server given by
``pi05.server`` in the evaluation YAML or by ``$PI05_POLICY_SERVER``.

Two robot interfaces are supported, selected by ``pi05.state``:

* ``joint``: 14-dim joint positions in, executed with ``action_type="qpos"``;
* ``endpose``: 16-dim end-effector poses ``[left xyz, quat, gripper, right ...]`` in,
  executed with ``action_type="ee"`` (RoboTwin runs inverse kinematics).

Besides the actions, the server returns the predicted boxes ``future_obb_pixels`` with shape
``(H, K, 8, 2)`` in image fractions; they are collected exactly like the Box-WAM adapter does.
"""

from __future__ import annotations

import atexit
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

CAMERAS = {"cam_high": "head_camera", "cam_left_wrist": "left_camera", "cam_right_wrist": "right_camera"}


def _server_address(cfg: dict) -> tuple[str, int]:
    raw = str(cfg.get("server") or os.environ.get("PI05_POLICY_SERVER") or "127.0.0.1:8000")
    host, _, port = raw.rpartition(":")
    return host or "127.0.0.1", int(port)


def _endpose_state(observation: dict) -> np.ndarray:
    pose = observation["endpose"]
    return np.concatenate(
        [
            np.asarray(pose["left_endpose"], dtype=np.float32),
            np.asarray([pose["left_gripper"]], dtype=np.float32),
            np.asarray(pose["right_endpose"], dtype=np.float32),
            np.asarray([pose["right_gripper"]], dtype=np.float32),
        ]
    )


class RoboTwinPi05:
    def __init__(self, cfg: dict, *, num_open_loop_steps: int, box_dir: str | None):
        self.host, self.port = _server_address(cfg)
        self.state_source = str(cfg.get("state", "joint"))
        if self.state_source not in ("joint", "endpose"):
            raise ValueError("pi05.state must be 'joint' or 'endpose'")
        self.action_type = "ee" if self.state_source == "endpose" else "qpos"
        self.swap_rgb_bgr = bool(cfg.get("swap_rgb_bgr", False))
        self.num_open_loop_steps = num_open_loop_steps
        self.box_dir = box_dir
        self.client = None
        self.queue: deque = deque()
        self.step = 0
        self.episode = 0
        self.latencies_ms: list[float] = []
        self.boxes: list[tuple[int, np.ndarray]] = []

    def connect(self) -> None:
        if self.client is None:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy

            self.client = WebsocketClientPolicy(host=self.host, port=self.port)
            print(f"[pi0.5] connected to {self.host}:{self.port}: {self.client.get_server_metadata()}", flush=True)

    def encode(self, observation: dict, prompt: str) -> dict:
        images = {}
        for key, camera in CAMERAS.items():
            rgb = np.asarray(observation["observation"][camera]["rgb"])
            if self.swap_rgb_bgr:
                rgb = rgb[..., ::-1]
            images[key] = np.ascontiguousarray(np.moveaxis(rgb, -1, 0))
        if self.state_source == "endpose":
            state = _endpose_state(observation)
        else:
            state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        return {"images": images, "state": state, "prompt": prompt}

    def plan(self, observation: dict, prompt: str) -> None:
        self.connect()
        start = time.perf_counter()
        output = self.client.infer(self.encode(observation, prompt))
        self.latencies_ms.append((time.perf_counter() - start) * 1000.0)
        actions = np.asarray(output["actions"])
        boxes = output.get("future_obb_pixels")
        if boxes is not None:
            self.boxes.append((self.step, np.asarray(boxes)))
        self.queue.extend(actions[: self.num_open_loop_steps])

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
            print(f"[pi0.5] mean inference latency after warm-up: {np.mean(self.latencies_ms[1:]):.1f} ms", flush=True)
        self.queue.clear()
        self.boxes.clear()
        self.latencies_ms.clear()
        self.step = 0


def get_model(usr_args: dict) -> RoboTwinPi05:
    with open(usr_args["policy_config"]) as handle:
        cfg = yaml.safe_load(handle)
    model = RoboTwinPi05(
        cfg.get("pi05") or {},
        num_open_loop_steps=int(cfg.get("num_open_loop_steps", 50)),
        box_dir=os.environ.get("BOXWAM_BOX_DIR"),
    )
    atexit.register(model.save_boxes)
    return model


def eval(TASK_ENV, model: RoboTwinPi05, observation: dict) -> None:  # noqa: A001 - name required by RoboTwin
    if not model.queue:
        model.plan(observation, TASK_ENV.get_instruction())
    while model.queue:
        TASK_ENV.take_action(np.asarray(model.queue.popleft(), dtype=np.float32), action_type=model.action_type)
        TASK_ENV.get_obs()
        model.step += 1


def reset_model(model: RoboTwinPi05) -> None:
    """Called by RoboTwin before every episode; only completed episodes are saved and counted."""
    if model.step > 0:
        model.save_boxes()
        model.episode += 1
    model.reset()
