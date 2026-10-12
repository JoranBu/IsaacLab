# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Adapted from isaac-sim/IsaacLabTutorial (Apache-2.0); see ../LICENSE-IsaacLabTutorial.

"""Task-local action bounds for fingertip control and relative gripper targets."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import (
    DifferentialInverseKinematicsActionCfg,
    RelativeJointPositionAction,
    RelativeJointPositionActionCfg,
)
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class BoundedDifferentialIKAction(DifferentialInverseKinematicsAction):
    """Project fingertip goals into the task workspace and bound joint-position targets."""

    cfg: BoundedDifferentialIKActionCfg

    def __init__(self, cfg: BoundedDifferentialIKActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if not math.isfinite(cfg.joint_target_speed_limit) or cfg.joint_target_speed_limit <= 0.0:
            raise ValueError("joint_target_speed_limit must be finite and positive.")
        if len(cfg.workspace_lower) != 3 or len(cfg.workspace_upper) != 3:
            raise ValueError("Workspace bounds must contain three coordinates.")
        if any(
            not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper
            for lower, upper in zip(cfg.workspace_lower, cfg.workspace_upper)
        ):
            raise ValueError("Workspace bounds must be finite with lower < upper on each axis.")
        self._workspace_lower = torch.tensor(cfg.workspace_lower, device=self.device)
        self._workspace_upper = torch.tensor(cfg.workspace_upper, device=self.device)
        self._max_joint_step = cfg.joint_target_speed_limit * env.physics_dt

    def process_actions(self, actions: torch.Tensor) -> None:
        """Bound normalized inputs and project the resulting fingertip position goal."""
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        super().process_actions(actions)
        target = self._ik_controller.ee_pos_des
        bounded_target = target.clamp(self._workspace_lower, self._workspace_upper)
        self._processed_actions[:, :3] += bounded_target - target
        target.copy_(bounded_target)

    def apply_actions(self) -> None:
        """Cap the IK target offset per physics step and keep targets inside soft joint limits."""
        super().apply_actions()
        position = self._asset.data.joint_pos.torch[:, self._joint_ids]
        target = self._asset.actuators.target_command.position.torch[:, self._joint_ids]
        target = torch.where(torch.isfinite(target), target, position)
        target = position + (target - position).clamp(-self._max_joint_step, self._max_joint_step)
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        target.clamp_(limits[..., 0], limits[..., 1])
        self._asset.actuators.target_command.set_position_index(value=target, joint_ids=self._joint_ids)


@configclass
class BoundedDifferentialIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Fingertip workspace and joint target bounds for the Franka lifting task."""

    class_type: str = "{DIR}.actions:BoundedDifferentialIKAction"
    workspace_lower: tuple[float, float, float] = (0.10, -0.35, 0.04)
    """Minimum fingertip position in the robot base frame [m]."""
    workspace_upper: tuple[float, float, float] = (0.65, 0.35, 0.65)
    """Maximum fingertip position in the robot base frame [m]."""
    joint_target_speed_limit: float = 2.0
    """Rate used to cap each IK target offset at ``joint_target_speed_limit * physics_dt`` [rad/s].

    This bounds commanded target changes; it is not a hard bound on measured joint velocity under contact.
    """


class SoftLimitRelativeGripperAction(RelativeJointPositionAction):
    """Apply a bounded incremental jaw-position command.

    This preserves the real controller's ordinary position interface while
    avoiding hidden binary latch state in the policy action. Negative closes,
    positive opens, and zero holds the measured jaw position.
    """

    def process_actions(self, actions: torch.Tensor) -> None:
        """Sanitize the normalized policy command before scaling it."""
        super().process_actions(torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0))

    def apply_actions(self) -> None:
        """Apply the relative target without crossing authored soft limits."""
        target = self._asset.data.joint_pos.torch[:, self._joint_ids] + self.processed_actions
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        target.clamp_(limits[..., 0], limits[..., 1])
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)


@configclass
class SoftLimitRelativeGripperActionCfg(RelativeJointPositionActionCfg):
    """Configuration for :class:`SoftLimitRelativeGripperAction`."""

    class_type: str = "{DIR}.actions:SoftLimitRelativeGripperAction"
