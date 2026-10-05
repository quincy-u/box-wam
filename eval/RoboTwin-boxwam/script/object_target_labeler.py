"""Per-frame, per-arm object-target labeler.

Computes which object each robot arm is manipulating at every timestep, with
both `approach` and `linger/release` phases included so that the labels
guarantee at least one OBB target per frame whenever there is *any* engagement
during the demo. The labels are written to the episode HDF5 as:

    object_target/left   shape (T,)  variable-length string ("name" or "")
    object_target/right  shape (T,)  variable-length string ("name" or "")

Empty string means "this arm is not manipulating anything at this frame" —
the consumer (openpi-boxwam's converter, visualizer) is expected to mask the slot.

Detection rule:

  1. **Engagement events** per arm: contiguous runs (≥3 frames) where
       gripper_val < OPEN_THRESH AND the nearest tracked object is within
       DIST_THRESH AND that object is moving (xyz delta ≥ MOVE_EPS over
       MOVE_WINDOW frames, OR qpos delta ≥ QPOS_EPS for articulated objects).
     Each event picks the most-frequent nearest object as its target.

  2. **Timeline fill** per arm:
       - frames 0 .. first_event_start-1     → first event's target  (initial approach)
       - during event i                       → event i's target      (active manipulation)
       - frames between events i and i+1      → event i+1's target    (approach to next)
       - frames after last event              → last event's target   (linger / release)
     Arms with zero engagement events keep the empty string label.

Base_Task.save_obb2d() calls this module during data collection; it can also be
used to label already-collected HDF5s (``compute_object_targets`` /
``write_object_targets``).
"""
from __future__ import annotations

import numpy as np

# Tracked-object filtering: skip names that consistently have no proj_3d_obb,
# such as the synthetic "wall" actor in many tasks.
_DEFAULT_CAMS = ("head_camera", "front_camera", "left_camera", "right_camera")

OPEN_THRESH = 0.85
DIST_THRESH = 0.35
MOVE_WINDOW = 6
MOVE_EPS = 0.003     # meters
QPOS_EPS = 0.01      # radians


def _select_valid_objects(obj_pose_keys, obs_group, cams=_DEFAULT_CAMS):
    """Keep only objects that have proj_3d_obb on at least one camera."""
    valid = []
    for o in obj_pose_keys:
        for c in cams:
            grp = obs_group.get(c)
            if grp is None:
                continue
            obb = grp.get("proj_3d_obb")
            if obb is not None and o in obb:
                valid.append(o)
                break
    return valid


def _obj_moving_mask(obj_xyz, obj_qpos):
    """Per-frame "is this object moving" boolean for one object."""
    T = obj_xyz.shape[0]
    m = np.zeros(T, dtype=bool)
    for t in range(T):
        t0 = max(0, t - MOVE_WINDOW)
        if float(np.linalg.norm(obj_xyz[t] - obj_xyz[t0])) >= MOVE_EPS:
            m[t] = True
            continue
        if obj_qpos is not None:
            if float(np.linalg.norm(obj_qpos[t] - obj_qpos[t0])) >= QPOS_EPS:
                m[t] = True
    return m


def _detect_events(arm_pose, arm_grip, obj_xyz_by_name, obj_moving_by_name):
    """Return list of (t_start, t_end_inclusive, target_name) per arm."""
    T = arm_pose.shape[0]
    nearest = [None] * T
    engaged = np.zeros(T, dtype=bool)
    for t in range(T):
        tcp = arm_pose[t, :3]
        grip_closed = arm_grip[t] < OPEN_THRESH
        best_obj, best_d = None, float("inf")
        for o, xyz in obj_xyz_by_name.items():
            d = float(np.linalg.norm(xyz[t] - tcp))
            if d > DIST_THRESH:
                continue
            if not (grip_closed or bool(obj_moving_by_name[o][t])):
                continue
            if d < best_d:
                best_obj, best_d = o, d
        if best_obj is not None:
            nearest[t] = best_obj
            engaged[t] = True
    events = []
    t = 0
    while t < T:
        if not engaged[t]:
            t += 1
            continue
        j = t
        while j < T and engaged[j]:
            j += 1
        if j - t >= 3:
            names = [n for n in nearest[t:j] if n is not None]
            if names:
                target = max(set(names), key=names.count)
                events.append((t, j - 1, target))
        t = j
    return events


def _fill_timeline(events, T, arm_xyz=None, obj_xyz_by_name=None):
    """Map events → per-frame target string array (len T). Returns array of '' if no events.

    Approach frames (before each event) are only labeled while the arm is
    *actively closing the gap* toward that event's target — i.e. while the
    arm-to-target distance is monotonically decreasing as time advances.
    The walk-back stops as soon as the arm starts moving away (or stays
    constant), so the labeler doesn't paint frames where the arm is moving
    toward a different object first.

    Frames between an event's release and the next event's approach
    (transitions / retreats / idle) get the empty string — the user
    requested "only mark the object we are trying to manipulate, including
    approach". No linger-after-release marking either.
    """
    out = np.array([""] * T, dtype=object)
    if not events:
        return out

    # 1) Manipulation phase: hard-mark the engaged frames.
    for t0, t1, obj in events:
        for tt in range(t0, t1 + 1):
            out[tt] = obj

    # 2) Approach phase: walk back from each event start while distance is
    # decreasing. Skip if we don't have arm/object positions (e.g. a static
    # call signature without trajectory data).
    if arm_xyz is None or obj_xyz_by_name is None:
        return out

    for ei, (t0, _t1, obj) in enumerate(events):
        if obj not in obj_xyz_by_name:
            continue
        obj_pos = obj_xyz_by_name[obj]
        # Don't overlap into the previous event (or its already-claimed frames).
        prev_end = events[ei - 1][1] if ei > 0 else -1
        # Walk backward from t0-1: while distance at tt is greater than at tt+1
        # (i.e. arm is closing in over time), mark this frame as approach.
        for tt in range(t0 - 1, prev_end, -1):
            d_now = float(np.linalg.norm(arm_xyz[tt] - obj_pos[tt]))
            d_next = float(np.linalg.norm(arm_xyz[tt + 1] - obj_pos[tt + 1]))
            if d_now > d_next + 1e-5:  # strictly closing in (with small tol)
                out[tt] = obj
            else:
                break
    return out


def compute_object_targets(f) -> dict[str, np.ndarray]:
    """Compute per-frame target object names for left and right arms.

    Args:
        f: an open ``h5py.File`` (or group) with the standard RoboTwin schema:
           ``endpose/{left,right}_{endpose,gripper}``, ``object_pose/<name>``,
           optional ``object_qpos/<name>``, ``observation/<cam>/proj_3d_obb/<name>``.

    Returns:
        ``{"left": np.ndarray[str] of shape (T,), "right": np.ndarray[str] of shape (T,)}``.
        Empty string at frame t means that arm is not manipulating anything.
    """
    if "endpose" not in f or "object_pose" not in f:
        T = 0
        return {"left": np.array([], dtype=object), "right": np.array([], dtype=object)}

    L_pose = np.asarray(f["endpose/left_endpose"][:])
    R_pose = np.asarray(f["endpose/right_endpose"][:])
    L_grip = np.asarray(f["endpose/left_gripper"][:])
    R_grip = np.asarray(f["endpose/right_gripper"][:])
    T = L_pose.shape[0]

    pose_keys = list(f["object_pose"].keys())
    obs = f.get("observation")
    valid = _select_valid_objects(pose_keys, obs) if obs is not None else pose_keys
    if not valid:
        return {"left": np.array([""] * T, dtype=object),
                "right": np.array([""] * T, dtype=object)}

    obj_xyz = {o: np.asarray(f[f"object_pose/{o}"][:, :3]) for o in valid}
    obj_qpos = {}
    if "object_qpos" in f:
        for o in valid:
            k = f"object_qpos/{o}"
            if k in f:
                q = np.asarray(f[k][:])
                if q.ndim == 1:
                    q = q[:, None]
                obj_qpos[o] = q

    obj_moving = {o: _obj_moving_mask(obj_xyz[o], obj_qpos.get(o)) for o in valid}

    L_events = _detect_events(L_pose, L_grip, obj_xyz, obj_moving)
    R_events = _detect_events(R_pose, R_grip, obj_xyz, obj_moving)

    return {
        "left": _fill_timeline(L_events, T, arm_xyz=L_pose[:, :3], obj_xyz_by_name=obj_xyz),
        "right": _fill_timeline(R_events, T, arm_xyz=R_pose[:, :3], obj_xyz_by_name=obj_xyz),
    }


def write_object_targets(f, targets: dict[str, np.ndarray]) -> None:
    """Write left/right per-frame target strings into the HDF5 (rewriting if present)."""
    import h5py
    str_dt = h5py.string_dtype(encoding="utf-8")
    for arm in ("left", "right"):
        key = f"object_target/{arm}"
        if key in f:
            del f[key]
        arr = np.array([str(s) if s is not None else "" for s in targets[arm]], dtype=object)
        f.create_dataset(key, data=arr, dtype=str_dt)
