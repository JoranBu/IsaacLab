# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Portable cloth reset snapshots and optional GPU-cached reset sampling.

Unlike a rigid-object pose, cloth requires every simulated vertex position and
velocity. Positions in archives are relative to the environment origin; replay
adds the destination origin so a snapshot works in any vectorized environment.
Archives contain numeric/string arrays only and are loaded without pickle.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

PHASE_NAMES = ("reaching", "near_corner", "grasped", "near_goal")
_STATE_KEYS = ("joint_pos", "joint_vel", "joint_target", "cloth_nodal_state", "gripper_command", "phase")


def capture_state(env: ManagerBasedEnv, phase: int) -> dict[str, torch.Tensor]:
    """Clone the live, unassisted state; do not zero its velocities or keep solver buffers."""
    robot, cloth = env.scene["robot"], env.scene["cloth"]
    nodes = cloth.data.nodal_state_w.torch.clone()
    nodes[..., :3] -= env.scene.env_origins[:, None]
    return {
        "joint_pos": robot.data.joint_pos.torch.clone(),
        "joint_vel": robot.data.joint_vel.torch.clone(),
        "joint_target": robot.actuators.target_command.position.torch.clone(),
        "cloth_nodal_state": nodes,
        "gripper_command": torch.where(env.action_manager.action[:, -1:] < 0, -1.0, 1.0),
        "phase": torch.full((env.num_envs,), phase, dtype=torch.long, device=env.device),
    }


def _validate_arrays(data: dict[str, np.ndarray]) -> None:
    """Reject malformed/invalid archives before writing or replaying any state."""
    rows = len(data["phase"])
    joints = len(data["joint_names"])
    vertices = len(data["mesh_default_local"])
    shapes = {
        "joint_pos": (rows, joints),
        "joint_vel": (rows, joints),
        "joint_target": (rows, joints),
        "cloth_nodal_state": (rows, vertices, 6),
        "gripper_command": (rows, 1),
        "phase": (rows,),
    }
    if rows == 0 or joints == 0 or vertices == 0:
        raise ValueError("A reset dataset must contain states, joint names, and cloth vertices.")
    if len(set(data["joint_names"].tolist())) != joints or data["mesh_default_local"].shape != (vertices, 3):
        raise ValueError("Invalid joint names or default mesh geometry.")
    for key, shape in shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid {key}: expected finite values with shape {shape}.")
    if not np.isfinite(data["mesh_default_local"]).all():
        raise ValueError("Default mesh must be finite.")
    phase = data["phase"]
    if not np.issubdtype(phase.dtype, np.integer) or ((phase < 0) | (phase >= len(PHASE_NAMES))).any():
        raise ValueError("Invalid reset phase indices.")
    if not np.isin(data["gripper_command"], (-1.0, 1.0)).all():
        raise ValueError("Gripper commands must be -1 (closed) or +1 (open).")


def save_dataset(path: Path, env: ManagerBasedEnv, states: dict[str, torch.Tensor], metadata: dict) -> None:
    """Validate and atomically replace an archive; a failed write preserves existing data."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Reset archives must use the .npz extension.")
    cloth = env.scene["cloth"]
    arrays = {key: states[key].detach().cpu().numpy() for key in _STATE_KEYS}
    arrays["joint_names"] = np.asarray(env.scene["robot"].joint_names, dtype=np.str_)
    arrays["mesh_default_local"] = (
        (cloth.data.default_nodal_state_w.torch[0, :, :3] - env.scene.env_origins[0]).cpu().numpy()
    )
    _validate_arrays(arrays)
    arrays["metadata"] = np.asarray(json.dumps({"schema_version": 1, "phases": PHASE_NAMES, **metadata}, default=str))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_dataset(path: str | Path, env: ManagerBasedEnv) -> dict[str, torch.Tensor]:
    """Load onto the environment device, verifying mesh ordering and named robot joints."""
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in (*_STATE_KEYS, "joint_names", "mesh_default_local")}
        metadata = json.loads(str(archive["metadata"].item()))
    if metadata["schema_version"] != 1 or tuple(metadata["phases"]) != PHASE_NAMES:
        raise ValueError("Unsupported cloth reset dataset schema.")
    _validate_arrays(data)
    expected = (env.scene["cloth"].data.default_nodal_state_w.torch[0, :, :3] - env.scene.env_origins[0]).cpu().numpy()
    mesh = data["mesh_default_local"]
    if mesh.shape != expected.shape or not np.allclose(mesh, expected, atol=1.0e-5, rtol=0.0):
        raise ValueError("Reset dataset mesh differs from the task. Regenerate after changing cloth size/refinement.")
    names = data["joint_names"].tolist()
    current = env.scene["robot"].joint_names
    if set(names) != set(current):
        raise ValueError("Reset dataset robot joints differ from the task.")
    order = [names.index(name) for name in current]
    for key in ("joint_pos", "joint_vel", "joint_target"):
        data[key] = data[key][:, order]
    result = {key: torch.as_tensor(data[key], device=env.device) for key in _STATE_KEYS}
    limits = env.scene["robot"].data.soft_joint_pos_limits.torch[0]
    for key in ("joint_pos", "joint_target"):
        if ((result[key] < limits[:, 0] - 0.002) | (result[key] > limits[:, 1] + 0.002)).any():
            raise ValueError(f"Dataset {key} exceeds this robot's soft joint limits.")
    return result


def apply_states(
    env: ManagerBasedEnv, states: dict[str, torch.Tensor], rows: torch.Tensor, env_ids: torch.Tensor
) -> None:
    """Restore only selected environments, with all cloth vertices free and saved drive targets intact."""
    robot, cloth = env.scene["robot"], env.scene["cloth"]
    robot.write_joint_position_to_sim_index(position=states["joint_pos"][rows], env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=states["joint_vel"][rows], env_ids=env_ids)
    robot.actuators.target_command.set_position_index(value=states["joint_target"][rows], env_ids=env_ids)
    robot.actuators.target_command.set_velocity_index(
        value=torch.zeros_like(states["joint_vel"][rows]), env_ids=env_ids
    )
    nodes = states["cloth_nodal_state"][rows].clone()
    nodes[..., :3] += env.scene.env_origins[env_ids, None]
    targets = torch.cat((nodes[..., :3], torch.ones_like(nodes[..., :1])), dim=-1)
    cloth.write_nodal_kinematic_target_to_sim_index(targets, env_ids=env_ids)
    cloth.write_nodal_state_to_sim_index(nodes, env_ids=env_ids)


class ResetFromDataset(ManagerTermBase):
    """Cache snapshots once, then sample rows on GPU during episode resets.

    The normal default reset runs first. With ``dataset_path=None`` this is a
    no-op. Episode counters and latched rewards are reset by the usual managers;
    sampled physical grasps do not pre-award milestones or mark an episode successful.
    Restore the gripper command in the reset hook, after the action manager
    clears history, so the first observation describes the sampled grasp.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        path = cfg.params["dataset_path"]
        self.states = None if path is None else load_dataset(path, env)
        self._sampled_rows = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        if self.states is not None:
            counts = torch.bincount(self.states["phase"].long(), minlength=len(PHASE_NAMES)).tolist()
            print(f"[INFO]: Loaded cloth reset bank {path}: {len(self.states['phase'])} states; phase counts {counts}")
        self.probabilities = None
        weights = cfg.params["phase_weights"]
        if self.states is not None and weights is not None:
            if len(weights) != len(PHASE_NAMES):
                raise ValueError("phase_weights must have one weight per reset phase.")
            weights = torch.tensor(weights, dtype=torch.float32, device=env.device)
            counts = torch.bincount(self.states["phase"].long(), minlength=len(PHASE_NAMES))
            if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
                raise ValueError("phase_weights must be finite, nonnegative, and have positive sum.")
            if ((weights > 0) & (counts == 0)).any():
                raise ValueError("A positively weighted reset phase has no dataset rows.")
            self.probabilities = (weights / counts.clamp_min(1))[self.states["phase"].long()]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice,
        dataset_path: str | None,
        phase_weights: Sequence[float] | None,
    ) -> None:
        if self.states is None:
            return
        ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        if self.probabilities is None:
            rows = torch.randint(len(self.states["phase"]), (len(ids),), device=env.device)
        else:
            rows = torch.multinomial(self.probabilities, len(ids), replacement=True)
        apply_states(env, self.states, rows, ids)
        self._sampled_rows[ids] = rows

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Restore selected gripper commands after action history has been cleared.

        Physical drive targets were already restored in the reset event. Arm
        history remains zero because those commands are relative increments.
        Episode milestones remain under the reward/termination managers.
        """
        if self.states is None:
            return
        ids = torch.arange(self.num_envs, device=self.device)
        if env_ids is not None:
            ids = ids[env_ids]
        rows = self._sampled_rows[ids]
        ids, rows = ids[rows >= 0], rows[rows >= 0]
        command = self.states["gripper_command"][rows, 0]
        self._env.action_manager.action[ids, -1] = command
        self._env.action_manager.prev_action[ids, -1] = command
        self._env.action_manager.get_term("gripper").raw_actions[ids, 0] = command
        self._sampled_rows[ids] = -1
