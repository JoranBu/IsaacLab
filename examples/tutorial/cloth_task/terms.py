# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Adapted from isaac-sim/IsaacLabTutorial (Apache-2.0); see ../LICENSE-IsaacLabTutorial.

"""Observations, rewards, and height-based success for lifting a cloth."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

from .geometry import pinch_geometry

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------------------------------------------------
# Task geometry (metres). Heights are measured above each environment's origin.
# ---------------------------------------------------------------------------------------------------------------------

# Match the authored 6 mm mat and cloth particle thickness in env_cfg.py.
MAT_TOP_HEIGHT = 0.035
CLOTH_PARTICLE_RADIUS = 0.003
LIFT_TARGET_CLEARANCE = 0.10
LIFT_HOLD_STEPS = 10
GRASP_PROOF_LIFT = 0.006
PINCH_DISTANCE = 0.015
FIXED_FINGERTIP_OFFSET = (0.0, 0.0, 0.046)
MOVING_FINGERTIP_OFFSET = (0.0, 0.0, 0.046)


def _finite(value: torch.Tensor) -> torch.Tensor:
    """Replace invalid terminal-state values with finite neutral values."""
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------------------------------------------------
# Physical state helpers
# ---------------------------------------------------------------------------------------------------------------------


def _cloth_positions(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Read live simulated cloth vertices in world coordinates [m]."""
    cloth: DeformableObject = env.scene["cloth"]
    return cloth.data.nodal_pos_w.torch


def _pinch_state(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate jaw proximity and pinched-patch clearance from live vertices.

    Coupled Newton solvers do not expose filtered contact forces to ContactSensor. This geometric proxy requires
    the same cloth vertex to lie near both fingertips, with the gripper commanded closed. It is not force sensing.
    """
    fixed, moving = fingertip_positions_w(env)
    fixed_near, moving_near, pinched = pinch_geometry(_cloth_positions(env), fixed, moving, PINCH_DISTANCE)
    pinched &= (env.action_manager.action[:, -1] < 0.0).unsqueeze(-1)
    heights = _cloth_positions(env)[..., 2] - env.scene.env_origins[:, None, 2]
    clearance = heights - MAT_TOP_HEIGHT - CLOTH_PARTICLE_RADIUS
    patch_clearance = torch.where(pinched, clearance, torch.full_like(clearance, -1.0)).amax(dim=-1)
    return fixed_near, moving_near, pinched.any(dim=-1), patch_clearance


def bilateral_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a closed-jaw cloth pinch estimate; this is a geometric contact proxy."""
    return _pinch_state(env)[2]


def cloth_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return mean cloth vertex height above the environment origin [m]."""
    return _cloth_positions(env)[..., 2].mean(dim=-1) - env.scene.env_origins[:, 2]


def cloth_clearance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return clearance of the entire sheet above the mat, including particle thickness [m]."""
    lowest = _cloth_positions(env)[..., 2].amin(dim=-1) - env.scene.env_origins[:, 2]
    return lowest - MAT_TOP_HEIGHT - CLOTH_PARTICLE_RADIUS


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


def cloth_grasp_point_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the live outer corner selected from the undeformed mesh at environment creation."""
    index = getattr(env, "_cloth_grasp_vertex_index", None)
    if index is None:
        cloth: DeformableObject = env.scene["cloth"]
        default = cloth.data.default_nodal_state_w.torch[0, :, :2]
        corner = torch.stack((default[:, 0].amax(), default[:, 1].amin()))
        index = int((default - corner).square().sum(dim=-1).argmin().item())
        env._cloth_grasp_vertex_index = index
    return _cloth_positions(env)[:, index]


def _gripper_openness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return normalized measured jaw opening in [0, 1]."""
    robot: Articulation = env.scene["robot"]
    gripper_id = robot.find_joints("panda_finger_joint1", preserve_order=True)[0]
    position = robot.data.joint_pos.torch[:, gripper_id].squeeze(1)
    limits = robot.data.soft_joint_pos_limits.torch[:, gripper_id].squeeze(1)
    return _finite(((position - limits[:, 0]) / (limits[:, 1] - limits[:, 0])).clamp(0.0, 1.0))


# ---------------------------------------------------------------------------------------------------------------------
# Success and terminations
# ---------------------------------------------------------------------------------------------------------------------


def _hold_count(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Share the success counter with observations, including in the offline generator."""
    count = getattr(env, "_cloth_hold_count", None)
    if count is None:
        count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._cloth_hold_count = count
    return count


class LiftHistoryTerm(ManagerTermBase):
    """Count consecutive control steps above the height target and terminate on success.

    Success requires every cloth vertex (including particle radius) at least 10 cm above the mat for ten
    consecutive control steps. There is no pinch requirement or grasp/lift milestone state.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.hold_count = _hold_count(env)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.hold_count[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        clearance = cloth_clearance(env)
        finite = torch.isfinite(_cloth_positions(env)).all(dim=(1, 2))
        finite &= torch.isfinite(env.scene["cloth"].data.nodal_vel_w.torch).all(dim=(1, 2))
        above_target = finite & (clearance >= LIFT_TARGET_CLEARANCE)
        self.hold_count.copy_(
            torch.where(above_target, self.hold_count + 1, torch.zeros_like(self.hold_count)).clamp_max(LIFT_HOLD_STEPS)
        )
        success = self.hold_count >= LIFT_HOLD_STEPS

        log = env.extras.setdefault("log", {})
        log["Metrics/success_rate"] = success.float().mean()

        return success


def cloth_lost(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate invalid cloth, vertices below the floor, or an unreachable sheet centre."""
    cloth: DeformableObject = env.scene["cloth"]
    positions = _cloth_positions(env)
    velocities = cloth.data.nodal_vel_w.torch
    root_pos_w, root_quat_w = _robot_root_pose(env)
    centre, _ = subtract_frame_transforms(root_pos_w, root_quat_w, positions.mean(dim=1))
    finite = torch.isfinite(positions).all(dim=(1, 2)) & torch.isfinite(velocities).all(dim=(1, 2))
    below_floor = (positions[..., 2] - env.scene.env_origins[:, None, 2]).amin(dim=-1) < -0.05
    return (~finite) | below_floor | (centre[:, 0].abs() > 0.62) | (centre[:, 1].abs() > 0.50)


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
    """Pay signed jaw-to-corner distance progress throughout the episode, in units of ``scale``.

    Undiscounted distance changes telescope: retreating and returning to the same distance pays zero net reward.
    Do not clip individual increments; that makes a large retreat cheaper than several small approach steps.
    The first call after a reset initializes the reference distance without paying a reward.
    Divide by control dt so the reward manager's dt scaling preserves the discrete distance increment.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)
        self._has_previous = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._has_previous[ids] = False

    def __call__(self, env: ManagerBasedRLEnv, scale: float = 0.01) -> torch.Tensor:
        distance = torch.linalg.vector_norm(grasp_center_w(env) - cloth_grasp_point_w(env), dim=-1)
        valid = torch.isfinite(distance)
        distance = _finite(distance)
        progress = (self._previous_distance - distance) / scale
        reward = torch.where(self._has_previous & valid, progress, torch.zeros_like(progress)) / env.step_dt
        self._previous_distance.copy_(distance)
        self._has_previous.copy_(valid)
        return reward


def lift_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward mean sheet clearance, capped at the target, with no pinch or gripper-command requirement.

    This is a reward rate: the manager integrates it over control dt. Mean clearance supplies shaping while
    some vertices remain on the table; success separately requires the lowest vertex to clear the target.
    """
    mean_clearance = cloth_height(env) - MAT_TOP_HEIGHT - CLOTH_PARTICLE_RADIUS
    return (_finite(mean_clearance) / LIFT_TARGET_CLEARANCE).clamp(0.0, 1.0)


def success_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Pay on the actual success termination step, compensating the manager's dt scaling."""
    return env.termination_manager.get_term("success").float() / env.step_dt


def failure_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize lost cloth or an unstable robot once; timeouts and successful lifts are excluded."""
    failed = env.termination_manager.get_term("cloth_lost") | env.termination_manager.get_term("unstable_robot")
    return failed.float() / env.step_dt


def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the squared action change as a finite regularization rate."""
    action = _finite(env.action_manager.action).clamp(-1.0, 1.0)
    previous = _finite(env.action_manager.prev_action).clamp(-1.0, 1.0)
    return (action - previous).square().sum(dim=-1)


def joint_velocity_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return bounded squared joint speeds, keeping invalid terminal-state rewards finite."""
    velocity = _finite(env.scene["robot"].data.joint_vel.torch).clamp(-12.0, 12.0)
    return velocity.square().sum(dim=-1)


# ---------------------------------------------------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------------------------------------------------


def lift_features(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return clearance, jaw opening, and the normalized consecutive-height counter."""
    hold_fraction = (_hold_count(env).float() / LIFT_HOLD_STEPS).clamp(0.0, 1.0)
    clearance = _finite(cloth_clearance(env)).clamp(-1.0, 1.0)
    return torch.stack((clearance, _gripper_openness(env), hold_fraction), dim=-1)


def contact_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return privileged jaw-proximity and closed-jaw pinch estimates."""
    fixed, moving, pinched, _ = _pinch_state(env)
    return torch.stack((fixed, moving, pinched), dim=-1).float()


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


class ClothState(ManagerTermBase):
    """Observe nine material landmarks and the grasp corner in the robot base frame.

    Landmark indices are selected once from a 3 x 3 grid on the default mesh. Tracking material vertices, rather
    than reselecting spatial extrema each step, exposes bending and motion without a mesh-resolution-dependent
    policy input size. This compact observation does not expose every deformation mode.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        cloth: DeformableObject = env.scene["cloth"]
        default = cloth.data.default_nodal_state_w.torch[0, :, :2]
        lower, upper = default.amin(dim=0), default.amax(dim=0)
        fractions = default.new_tensor((0.0, 0.5, 1.0))
        grid = torch.cartesian_prod(fractions, fractions)
        targets = lower + grid * (upper - lower)
        self._indices = (default.unsqueeze(0) - targets.unsqueeze(1)).square().sum(dim=-1).argmin(dim=-1)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        cloth: DeformableObject = env.scene["cloth"]
        positions = _cloth_positions(env)[:, self._indices]
        velocities = cloth.data.nodal_vel_w.torch[:, self._indices]
        root_position, root_quaternion = _robot_root_pose(env)
        rotation = root_quaternion[:, None].expand(-1, len(self._indices), -1).reshape(-1, 4)
        positions_b = quat_apply_inverse(rotation, (positions - root_position[:, None]).reshape(-1, 3))
        velocities_b = quat_apply_inverse(rotation, velocities.reshape(-1, 3))
        state = torch.cat((positions_b, velocities_b), dim=-1).reshape(env.num_envs, -1)
        grasp_error = quat_apply_inverse(root_quaternion, cloth_grasp_point_w(env) - grasp_center_w(env))
        return _finite(torch.cat((state, grasp_error), dim=-1)).clamp(-10.0, 10.0)
