# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Adapted from isaac-sim/IsaacLabTutorial (Apache-2.0); see ../LICENSE-IsaacLabTutorial.

"""Reset events for validated task-horizon state replay."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

PHASE_NAMES = (
    "approach",
    "pregrasp",
    "grasp",
    "lift",
    "reorient",
    "transport",
    "insert",
    "release",
)
STATE_SHAPES = {
    "joint_position": (6,),
    "joint_target": (6,),
    "vial_pose": (7,),
    "phase": (),
    "difficulty": (),
    "grasped": (),
    "lifted": (),
}
_FLOAT_FIELDS = ("joint_position", "joint_target", "vial_pose", "difficulty")
_BOOL_FIELDS = ("grasped", "lifted")
_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def _validate_state_fields(states: Mapping[str, Any]) -> None:
    if set(states) != set(STATE_SHAPES):
        missing = sorted(set(STATE_SHAPES) - set(states), key=repr)
        extra = sorted(set(states) - set(STATE_SHAPES), key=repr)
        raise ValueError(f"Reset state fields do not match the schema (missing={missing}, extra={extra}).")
    for name, value in states.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Reset field {name!r} must be a tensor.")
    for name in _FLOAT_FIELDS:
        if not states[name].is_floating_point():
            raise ValueError(f"Reset field {name!r} must have a floating-point dtype.")
    if states["phase"].dtype not in _INTEGER_DTYPES:
        raise ValueError("Reset field 'phase' must have an integer dtype.")
    for name in _BOOL_FIELDS:
        if states[name].dtype != torch.bool:
            raise ValueError(f"Reset field {name!r} must have a boolean dtype.")


def validate_reset_states(states: Mapping[str, Any]) -> int:
    """Validate reset tensors and return their common row count."""
    if not isinstance(states, Mapping):
        raise ValueError("Reset states must be a mapping.")
    _validate_state_fields(states)
    row_count = int(states["phase"].shape[0]) if states["phase"].ndim == 1 else 0
    if row_count == 0:
        raise ValueError("Reset dataset must contain at least one row.")
    for name, trailing_shape in STATE_SHAPES.items():
        expected = (row_count, *trailing_shape)
        if tuple(states[name].shape) != expected:
            raise ValueError(f"Reset field {name!r} must have shape {expected}, got {tuple(states[name].shape)}.")
    for name in _FLOAT_FIELDS:
        if not bool(torch.isfinite(states[name]).all()):
            raise ValueError(f"Reset field {name!r} contains non-finite values.")
    if bool(((states["phase"] < 0) | (states["phase"] >= len(PHASE_NAMES))).any()):
        raise ValueError("Reset phase IDs are outside the declared phase table.")
    if bool(((states["difficulty"] < 0.0) | (states["difficulty"] > 1.0)).any()):
        raise ValueError("Reset difficulty must lie in [0, 1].")
    quaternion_norm = torch.linalg.vector_norm(states["vial_pose"][:, 3:7], dim=-1)
    if not bool(torch.allclose(quaternion_norm, torch.ones_like(quaternion_norm), atol=2.0e-3, rtol=0.0)):
        raise ValueError("Reset vial quaternions must be normalized XYZW quaternions.")
    return row_count


def _phase_balanced_row_weights(phase: torch.Tensor, phase_weights: Sequence[float]) -> torch.Tensor:
    """Spread each requested phase probability uniformly over that phase's rows."""
    if phase.ndim != 1 or phase.numel() == 0:
        raise ValueError("phase must be a nonempty one-dimensional tensor")
    if phase.dtype not in _INTEGER_DTYPES or bool((phase < 0).any()):
        raise ValueError("phase must contain nonnegative integers")

    phase_count = int(phase.max().item()) + 1
    weights = torch.as_tensor(phase_weights, device=phase.device, dtype=torch.float32)
    if weights.ndim != 1 or len(weights) != phase_count:
        raise ValueError(f"phase_weights must contain exactly {phase_count} values")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()) or not bool(weights.any()):
        raise ValueError("phase_weights must be finite, nonnegative, and not all zero")

    eligible = weights[phase] > 0.0
    eligible_counts = torch.bincount(phase[eligible], minlength=phase_count)
    missing = (weights > 0.0) & (eligible_counts == 0)
    if bool(missing.any()):
        missing_phases = missing.nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"Reset curriculum has no eligible rows for phases {missing_phases}")
    per_phase_count = eligible_counts.clamp_min(1).to(weights.dtype)
    row_weights = weights[phase] / per_phase_count[phase]
    return torch.where(eligible, row_weights, torch.zeros_like(row_weights))


def _ids(env: ManagerBasedRLEnv, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
    """Normalize event-manager environment indices."""
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    raw_ids = torch.as_tensor(env_ids)
    if raw_ids.ndim > 1 or raw_ids.dtype not in _INTEGER_DTYPES:
        raise ValueError("env_ids must contain integers in a scalar or one-dimensional sequence")
    if raw_ids.device.type == "cpu":
        if bool(((raw_ids < 0) | (raw_ids >= env.num_envs)).any()):
            raise ValueError(f"env_ids must lie in [0, {env.num_envs - 1}]")
        if raw_ids.numel() != torch.unique(raw_ids).numel():
            raise ValueError("env_ids must not contain duplicates")
    return raw_ids.to(device=env.device, dtype=torch.long).reshape(-1)


def _reset_progress_seed(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    *,
    phase: torch.Tensor,
    grasped: torch.Tensor,
    lifted: torch.Tensor,
) -> None:
    """Publish reset-row history for the instance-owned success term."""
    if not hasattr(env, "_so101_reset_phase"):
        env._so101_reset_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._so101_reset_grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._so101_reset_lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._so101_reset_phase[env_ids] = phase
    env._so101_reset_grasped[env_ids] = grasped
    env._so101_reset_lifted[env_ids] = lifted


class ResetFromDataset(ManagerTermBase):
    """Replay physics-validated reset rows, sampled by phase weight or in deterministic order."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        artifact = load_reset_dataset(cfg.params["dataset_path"], device=env.device)
        self.states = artifact["states"]
        self.row_count = int(artifact["row_count"])
        self._cursor = 0
        phase_weights = cfg.params.get("phase_weights")
        self.row_weights = None
        if phase_weights is not None:
            self.row_weights = _phase_balanced_row_weights(self.states["phase"], phase_weights)
        self.sequential_rows = (
            torch.arange(self.row_count, device=env.device)
            if self.row_weights is None
            else self.row_weights.nonzero(as_tuple=False).flatten()
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | torch.Tensor | slice,
        dataset_path: str,
        sequential: bool = False,
        phase_weights: tuple[float, ...] | None = None,
    ) -> None:
        """Write selected joint and vial states into the requested worlds."""
        del dataset_path, phase_weights
        ids = _ids(env, env_ids)
        if ids.numel() == 0:
            return
        if sequential:
            indices = (torch.arange(ids.numel(), device=env.device) + self._cursor).remainder(
                self.sequential_rows.numel()
            )
            rows = self.sequential_rows[indices]
            self._cursor = (self._cursor + ids.numel()) % self.sequential_rows.numel()
        elif self.row_weights is None:
            rows = torch.randint(self.row_count, (ids.numel(),), device=env.device)
        else:
            rows = torch.multinomial(self.row_weights, ids.numel(), replacement=True)

        robot = env.scene["robot"]
        joint_position = self.states["joint_position"][rows]
        joint_target = self.states["joint_target"][rows]
        joint_velocity = torch.zeros_like(joint_position)
        robot.write_joint_position_to_sim_index(position=joint_position, env_ids=ids)
        robot.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=ids)
        robot.set_joint_position_target_index(target=joint_target, env_ids=ids)
        robot.set_joint_velocity_target_index(target=joint_velocity, env_ids=ids)
        vial_pose = self.states["vial_pose"][rows].clone()
        vial_pose[:, :3] += env.scene.env_origins[ids]
        vial = env.scene["vial"]
        vial.write_root_pose_to_sim_index(root_pose=vial_pose, env_ids=ids)
        vial.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((ids.numel(), 6), device=env.device),
            env_ids=ids,
        )

        rack = env.scene["rack"]
        rack_pose = rack.data.default_root_pose.torch[ids].clone()
        rack_pose[:, :3] += env.scene.env_origins[ids]
        rack.write_root_pose_to_sim_index(root_pose=rack_pose, env_ids=ids)
        rack.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((ids.numel(), 6), device=env.device),
            env_ids=ids,
        )
        _reset_progress_seed(
            env,
            ids,
            phase=self.states["phase"][rows],
            grasped=self.states["grasped"][rows],
            lifted=self.states["lifted"][rows],
        )


def save_reset_dataset(
    path: str | Path,
    states: Mapping[str, torch.Tensor],
    *,
    generator: dict,
    validation: dict,
) -> dict:
    """Validate and atomically save generated poses in the example's NumPy format."""
    output = Path(path).expanduser().resolve()
    if output.suffix != ".npz":
        raise ValueError("Reset datasets must use the .npz extension.")
    row_count = validate_reset_states(states)
    metadata = {"generator": generator, "validation": validation, "row_count": row_count}
    arrays = {name: value.detach().cpu().numpy() for name, value in states.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz", delete=False) as file:
            temporary = Path(file.name)
            np.savez_compressed(file, **arrays, _metadata=np.asarray(json.dumps(metadata)))
        os.chmod(temporary, output.stat().st_mode & 0o777 if output.exists() else 0o644)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return metadata


def load_reset_dataset(path: str | Path, device: str = "cpu") -> dict:
    """Load the bundled XYZW reset poses and check their state schema."""
    with np.load(path, allow_pickle=False) as archive:
        states = {name: torch.from_numpy(archive[name]) for name in STATE_SHAPES}
        metadata = json.loads(str(archive["_metadata"])) if "_metadata" in archive else {}
    row_count = validate_reset_states(states)
    return {**metadata, "states": {name: tensor.to(device) for name, tensor in states.items()}, "row_count": row_count}
