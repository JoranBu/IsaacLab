# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Adapted from isaac-sim/IsaacLabTutorial (Apache-2.0); see ../LICENSE-IsaacLabTutorial.

"""Observations, rewards, milestones, and terminations for grasping and lifting a vial."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

from .geometry import cylinder_lowest_offset
from .progress import LiftProgress

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


# ---------------------------------------------------------------------------------------------------------------------
# Task geometry (metres). Heights are measured above each environment's origin.
# ---------------------------------------------------------------------------------------------------------------------

# The mat is centred at 32 mm and is 6 mm thick; the horizontal vial has a 17 mm collision radius.
MAT_TOP_HEIGHT = 0.035
LIFT_TARGET_CLEARANCE = 0.10
VIAL_AXIS_MIN = -0.017
VIAL_AXIS_MAX = 0.100
VIAL_RADIUS = 0.017
# The workshop grasp encloses the enlarged cap and its shoulder, which retains the vial axially.
VIAL_GRASP_OFFSET = (0.0, 0.0, 0.092)
# Both jaws touching a vial that is held this far above its resting height is a load-bearing grasp.
GRASP_PROOF_LIFT = 0.006
FIXED_FINGERTIP_OFFSET = (0.0, 0.0, 0.046)
MOVING_FINGERTIP_OFFSET = (0.0, 0.0, 0.046)


def _finite(value: torch.Tensor) -> torch.Tensor:
    """Replace invalid terminal-state values with finite neutral values."""
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _finite_error(value: torch.Tensor) -> torch.Tensor:
    """Replace invalid geometric errors with a safely distant value."""
    return torch.nan_to_num(value, nan=1.0e3, posinf=1.0e3, neginf=1.0e3)


# ---------------------------------------------------------------------------------------------------------------------
# Physical state helpers
# ---------------------------------------------------------------------------------------------------------------------


def _contact_magnitude(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """Return the maximum filtered contact-force magnitude for a sensor."""
    sensor: ContactSensor = env.scene.sensors[name]
    forces = sensor.data.force_matrix_w.torch
    magnitude = torch.linalg.vector_norm(forces, dim=-1).reshape(env.num_envs, -1).amax(dim=-1)
    return _finite(magnitude)


def _contact(env: ManagerBasedRLEnv, name: str, threshold: float = 0.05) -> torch.Tensor:
    """Return thresholded filtered contact for one sensor."""
    return _contact_magnitude(env, name) > threshold


def bilateral_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Whether the vial has simultaneous physical contact with both jaws."""
    return _contact(env, "fixed_jaw_contact") & _contact(env, "moving_jaw_contact")


def vial_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the vial root height above its environment origin [m]."""
    vial: RigidObject = env.scene["vial"]
    return vial.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]


def vial_clearance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the lowest physical point of the vial above the mat [m], accounting for orientation."""
    vial: RigidObject = env.scene["vial"]
    quaternion = vial.data.root_quat_w.torch
    axis = quaternion.new_tensor((0.0, 0.0, 1.0)).expand(env.num_envs, -1)
    axis_z = quat_apply(quaternion, axis)[:, 2]
    lowest_offset = cylinder_lowest_offset(axis_z, VIAL_AXIS_MIN, VIAL_AXIS_MAX, VIAL_RADIUS)
    return vial_height(env) + lowest_offset - MAT_TOP_HEIGHT


def fingertip_positions_w(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return both simplified contact-pad centres in world coordinates [m]."""
    robot: Articulation = env.scene["robot"]
    fixed_id = robot.find_bodies("panda_leftfinger", preserve_order=True)[0]
    moving_id = robot.find_bodies("panda_rightfinger", preserve_order=True)[0]
    fixed_pos = robot.data.body_pos_w.torch[:, fixed_id].squeeze(1)
    fixed_quat = robot.data.body_quat_w.torch[:, fixed_id].squeeze(1)
    moving_pos = robot.data.body_pos_w.torch[:, moving_id].squeeze(1)
    moving_quat = robot.data.body_quat_w.torch[:, moving_id].squeeze(1)
    fixed = fixed_pos + quat_apply(fixed_quat, fixed_pos.new_tensor(FIXED_FINGERTIP_OFFSET).expand_as(fixed_pos))
    moving = moving_pos + quat_apply(moving_quat, moving_pos.new_tensor(MOVING_FINGERTIP_OFFSET).expand_as(moving_pos))
    return fixed, moving


def grasp_center_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the midpoint between the fingertip contact pads [m]."""
    fixed, moving = fingertip_positions_w(env)
    return 0.5 * (fixed + moving)


def vial_grasp_point_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the centre of the vial cap/shoulder used for grasping [m]."""
    vial: RigidObject = env.scene["vial"]
    root_pos = vial.data.root_pos_w.torch
    root_quat = vial.data.root_quat_w.torch
    offset = root_pos.new_tensor(VIAL_GRASP_OFFSET).expand_as(root_pos)
    return root_pos + quat_apply(root_quat, offset)


def _gripper_openness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return normalized measured jaw opening in [0, 1]."""
    robot: Articulation = env.scene["robot"]
    gripper_id = robot.find_joints("panda_finger_joint1", preserve_order=True)[0]
    position = robot.data.joint_pos.torch[:, gripper_id].squeeze(1)
    limits = robot.data.soft_joint_pos_limits.torch[:, gripper_id].squeeze(1)
    return _finite(((position - limits[:, 0]) / (limits[:, 1] - limits[:, 0])).clamp(0.0, 1.0))


# ---------------------------------------------------------------------------------------------------------------------
# Milestones, success, and terminations
# ---------------------------------------------------------------------------------------------------------------------


def _history(env: ManagerBasedRLEnv) -> LiftProgress:
    """Return the instance-owned episode progress buffer."""
    progress = getattr(env, "_vial_lift_progress", None)
    if progress is None:
        progress = LiftProgress(env.num_envs, env.device, stable_steps=10, grasp_steps=2)
        env._vial_lift_progress = progress
    return progress


class LiftHistoryTerm(ManagerTermBase):
    """Latch physical milestones each step and terminate on a sustained lift.

    Success requires both fingers to hold the vial with its lowest point at least 10 cm above the mat for ten
    consecutive control steps. Orientation is unrestricted.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.progress = _history(env)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.progress.reset(env_ids)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        clearance = vial_clearance(env)
        holding = bilateral_contact(env) & (clearance > GRASP_PROOF_LIFT)
        above_target = clearance >= LIFT_TARGET_CLEARANCE
        success = self.progress.update(holding, above_target, env.episode_length_buf)

        log = env.extras.setdefault("log", {})
        log["Metrics/grasp_rate"] = self.progress.grasped.float().mean()
        log["Metrics/lift_rate"] = self.progress.lifted.float().mean()
        log["Metrics/success_rate"] = success.float().mean()
        log["Diagnostics/vial_height_m"] = _finite(vial_height(env)).mean()
        log["Diagnostics/gripper_openness"] = _gripper_openness(env).mean()
        log["Diagnostics/vial_clearance_m"] = _finite(clearance).mean()
        completed = self.progress.time_to_success >= 0
        if completed.any():
            log["Metrics/time_to_success_s"] = self.progress.time_to_success[completed].float().mean() * env.step_dt

        return success


def vial_lost(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate non-finite, dropped, or unreachable vial states."""
    vial: RigidObject = env.scene["vial"]
    root_pos_w, root_quat_w = _robot_root_pose(env)
    vial_pos = vial.data.root_pos_w.torch
    pos, _ = subtract_frame_transforms(root_pos_w, root_quat_w, vial_pos)
    finite = torch.isfinite(vial_pos).all(dim=-1) & torch.isfinite(vial.data.root_quat_w.torch).all(dim=-1)
    finite &= torch.isfinite(vial.data.root_lin_vel_w.torch).all(dim=-1)
    finite &= torch.isfinite(vial.data.root_ang_vel_w.torch).all(dim=-1)
    dropped = _history(env).lifted & (vial_clearance(env) < 0.002)
    return (~finite) | dropped | (pos[:, 2] < -0.01) | (pos[:, 0].abs() > 0.62) | (pos[:, 1].abs() > 0.50)


def unstable_robot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate non-finite or extreme joint motion."""
    robot: Articulation = env.scene["robot"]
    joint_pos = robot.data.joint_pos.torch
    joint_vel = robot.data.joint_vel.torch
    finite = torch.isfinite(joint_pos).all(dim=-1) & torch.isfinite(joint_vel).all(dim=-1)
    return (~finite) | (joint_vel.abs().amax(dim=1) > 12.0)


# ---------------------------------------------------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------------------------------------------------


class ApproachProgressReward(ManagerTermBase):
    """Pay the decrease in jaw-to-cap distance, in units of ``scale``, until the vial is grasped.

    Undiscounted distance changes telescope: retreating and returning to the same distance pays zero net reward.
    Do not clip individual increments; that makes a large retreat cheaper than several small approach steps.
    The first call after a reset initializes the reference distance without paying a reward.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)
        self._has_previous = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._has_previous[ids] = False

    def __call__(self, env: ManagerBasedRLEnv, scale: float = 0.01) -> torch.Tensor:
        distance = _finite_error(torch.linalg.vector_norm(grasp_center_w(env) - vial_grasp_point_w(env), dim=-1))
        progress = (self._previous_distance - distance) / scale
        active = self._has_previous & ~_history(env).grasped
        self._previous_distance.copy_(distance)
        self._has_previous.fill_(True)
        return torch.where(active, progress, torch.zeros_like(progress))


def lift_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward live bilateral grasps in proportion to clearance, capped at the success height."""
    fraction = (vial_clearance(env) / LIFT_TARGET_CLEARANCE).clamp(0.0, 1.0)
    return _finite(torch.where(bilateral_contact(env), fraction, torch.zeros_like(fraction)))


class PhysicalMilestoneReward(ManagerTermBase):
    """Pay each physical milestone (grasped, lifted) exactly once per episode.

    A milestone latched in the current episode cannot pay again after losing and regaining the grasp.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._previous[ids] = False

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        history = _history(env)
        current = torch.stack((history.grasped, history.lifted), dim=-1)
        newly_reached = current & ~self._previous
        weights = current.new_tensor((1.0, 2.0), dtype=torch.float32)
        reward = (newly_reached.float() * weights).sum(dim=-1)
        self._previous.copy_(current)
        return _finite(reward)


def success_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the one-step confirmed success signal."""
    return _history(env).success.float()


# ---------------------------------------------------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------------------------------------------------


def progress_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the latched grasp and lift milestones that make the milestone reward Markov."""
    history = _history(env)
    return torch.stack((history.grasped, history.lifted), dim=-1).float()


def lift_features(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return clearance, jaw opening, and normalized grasp/hold counters."""
    history = _history(env)
    grasp_fraction = (history.grasp_count.float() / history.grasp_steps).clamp(0.0, 1.0)
    hold_fraction = (history.stable_count.float() / history.stable_steps).clamp(0.0, 1.0)
    clearance = _finite(vial_clearance(env)).clamp(-1.0, 1.0)
    return torch.stack((clearance, _gripper_openness(env), grasp_fraction, hold_fraction), dim=-1)


def contact_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return privileged fixed-jaw, moving-jaw, and bilateral contact flags."""
    fixed = _contact(env, "fixed_jaw_contact")
    moving = _contact(env, "moving_jaw_contact")
    return torch.stack((fixed, moving, fixed & moving), dim=-1).float()


def joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Return finite measured joint positions for the selected robot joints."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    robot: Articulation = env.scene[asset_cfg.name]
    position = _finite(robot.data.joint_pos.torch[:, asset_cfg.joint_ids])
    limits = robot.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids]
    return position.clamp(limits[..., 0], limits[..., 1])


def joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Return finite measured joint velocities for the selected robot joints."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    robot: Articulation = env.scene[asset_cfg.name]
    return _finite(robot.data.joint_vel.torch[:, asset_cfg.joint_ids]).clamp(-12.0, 12.0)


def joint_target(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Return drive targets in the same selected order as the measured joint state."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    robot: Articulation = env.scene[asset_cfg.name]
    return _finite(robot.data.joint_pos_target.torch[:, asset_cfg.joint_ids]).clamp(-4.0, 4.0)


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the previous finite policy action."""
    return _finite(env.action_manager.action).clamp(-1.0, 1.0)


def _robot_root_pose(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the fixed robot base pose used as the task reference frame."""
    robot: Articulation = env.scene["robot"]
    return robot.data.root_link_pos_w.torch, robot.data.root_link_quat_w.torch


def body_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return robot-body pose and velocity in the robot base frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    ids = asset_cfg.body_ids
    root_pos_w, root_quat_w = _robot_root_pose(env)
    body_pos_w = robot.data.body_pos_w.torch[:, ids].reshape(-1, 3)
    body_quat_w = robot.data.body_quat_w.torch[:, ids].reshape(-1, 4)
    bodies_per_env = body_pos_w.shape[0] // env.num_envs
    root_pos_w = root_pos_w.repeat_interleave(bodies_per_env, dim=0)
    root_quat_w = root_quat_w.repeat_interleave(bodies_per_env, dim=0)
    body_pos_b, body_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    body_lin_vel_b = quat_apply_inverse(root_quat_w, robot.data.body_lin_vel_w.torch[:, ids].reshape(-1, 3))
    body_ang_vel_b = quat_apply_inverse(root_quat_w, robot.data.body_ang_vel_w.torch[:, ids].reshape(-1, 3))
    state = torch.cat((body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=-1).reshape(env.num_envs, -1)
    return _finite(state).clamp(-10.0, 10.0)


def rigid_object_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return rigid-object pose and velocity in the robot base frame."""
    obj: RigidObject = env.scene[asset_cfg.name]
    root_pos_w, root_quat_w = _robot_root_pose(env)
    obj_pos_b, obj_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, obj.data.root_pos_w.torch, obj.data.root_quat_w.torch
    )
    state = torch.cat(
        (
            obj_pos_b,
            obj_quat_b,
            quat_apply_inverse(root_quat_w, obj.data.root_lin_vel_w.torch),
            quat_apply_inverse(root_quat_w, obj.data.root_ang_vel_w.torch),
        ),
        dim=-1,
    )
    return _finite(state).clamp(-10.0, 10.0)
