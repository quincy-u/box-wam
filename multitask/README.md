# pi0.5 + Box-WAM

The default form of Box-WAM adds box-trajectory prediction to the pi0.5 vision-language-action model and
fine-tunes it on 50 RoboTwin 2.0 tasks, giving one language-conditioned policy for all tasks (paper Table 2).
Because this touches openpi's model code, data pipeline and serving, the implementation is a fork of openpi:

**[`multitask/openpi-boxwam`](openpi-boxwam)** (based on openpi commit `650c5b0`)

Evaluation runs through this repository: `scripts/evaluate_robotwin.py --config configs/eval/robotwin_pi05.yaml`.

## What the fork adds to pi0.5

* **Config** (`src/openpi/models/pi0_config.py`): `Pi0Config(predict_obb=True, max_objects=2, obb_dim=32,
  lambda_obb=1.0)`. Two box slots, one for the object targeted by each arm, eight projected corners each.
* **Tokens** (`src/openpi/models/pi0.py`): `action_horizon` box tokens are inserted into the action-expert
  suffix in front of the action tokens, embedded by `obb_in_proj` and decoded by `obb_out_proj`. These two linear
  layers are the only new parameters; the pi0.5 base weights load unchanged
  (`CheckpointWeightLoader(extra_missing_regex=r".*obb_(?:in|out)_proj.*")`).
* **Attention**: the box block opens its own attention group, so the prefix (images, language, state) cannot see
  the boxes, boxes see the prefix, action tokens see the prefix and the boxes, and boxes cannot see the actions.
* **Objective**: boxes are denoised with the same rectified-flow timestep as the actions and the loss is
  `action_loss + lambda_obb · box_loss` (both mean squared velocity errors); `compute_loss` returns
  `(loss, {"action_loss", "obb_loss"})` so both are logged. Empty slots have an all-zero target.
* **Transforms** (`src/openpi/policies/robotwin_obb_policy.py`): `RobotwinObbInputs` turns pixel corner
  coordinates into normalised targets (divide by the head-camera resolution, clamp to `[−0.05, 1.05]`, map to
  `[−1, 1]`, zero padded frames); `RobotwinObbOutputs` inverts it at inference and returns
  `future_obb_pixels` with shape `(50, 2, 8, 2)` in image fractions.
* **Data config hooks** on `LeRobotAlohaDataConfig`: `extra_passthrough_keys`, `extra_data_transform_inputs`,
  `extra_output_keys`, `extra_data_transform_outputs` and `output_action_dim` (16 for end-effector control).
* **Sampling**: `sample_actions` returns `(actions, future_obb)`; `Policy.infer` exposes the boxes to the
  websocket server, which is how the RoboTwin adapter in this repository receives them.

Training config: `pi05_robotwin_boxwam` (`src/openpi/training/config.py`): pi0.5 base weights, batch 256,
AdamW with gradient clipping 1.0, cosine schedule with 100 warm-up steps and peak learning rate 5e-5, 16-dim
end-effector state and action, three cameras, `prompt_from_task`. Training is JAX only.

## Dataset

The Box-WAM RoboTwin dataset `quincyu/robotwin_eef_obb_27500` is a LeRobot v2.1 dataset of 27,500 RoboTwin 2.0 episodes:
for each of the 50 tasks, 50 clean and 500 randomized episodes rendered at 640 × 480 with three cameras, 16-dim
end-effector state and action (`[left xyz, quaternion, gripper, right …]`, `action[t] = state[t + 1]`), the
task instruction as prompt, and `observation.obb_head` with the two box slots in head-camera pixels.
`scripts/robotwin/convert_robotwin_to_lerobot.py` and `scripts/robotwin/merge_lerobot_datasets.py` in the
fork turn RoboTwin-boxwam recordings (which include the boxes; task configs `demo_clean_eef` and
`demo_randomized_eef`, 640 × 480 cameras) into this format.

## Commands (in multitask/openpi-boxwam)

```bash
uv sync                                                        # environment, see the fork's README
uv run scripts/compute_norm_stats.py --config-name pi05_robotwin_boxwam
uv run scripts/train.py pi05_robotwin_boxwam --exp-name boxwam
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config pi05_robotwin_boxwam --policy.dir checkpoints/pi05_robotwin_boxwam/boxwam/34999
```

Then, from this repository (with `openpi-client` installed in the RoboTwin environment):

```bash
python scripts/evaluate_robotwin.py --config configs/eval/robotwin_pi05.yaml \
    --tasks adjust_bottle beat_block_hammer ... --task-config demo_randomized --episodes 100 --gpus 0
```

`configs/eval/robotwin_pi05.yaml` selects the end-effector interface (`pi05.state: endpose`; actions are
executed with RoboTwin's inverse kinematics) and the default protocol: 50-step chunks with 36 steps executed
before re-planning, unseen instructions, 100 trials per task, one seed. Predicted box trajectories are saved
under `boxes/` and drawn onto the rollout videos under `videos/` of the results folder.

## Released checkpoint

The 35k-step end-effector model from the paper is `quincyu/pi05_robotwin_obb_eef` on the Hugging Face Hub
(parameters and normalisation statistics, about 12 GB):

```bash
uv run huggingface-cli download quincyu/pi05_robotwin_obb_eef --local-dir checkpoints/pi05_robotwin_boxwam/release
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config pi05_robotwin_boxwam --policy.dir checkpoints/pi05_robotwin_boxwam/release/34999
```

The dataset it was trained on is `quincyu/robotwin_eef_obb_27500` (LeRobot v2.1).
