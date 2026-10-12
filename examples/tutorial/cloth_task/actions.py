# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Adapted from isaac-sim/IsaacLabTutorial (Apache-2.0); see ../LICENSE-IsaacLabTutorial.

"""The policy jaw action clamps relative targets to the calibrated soft limits."""

import torch

from isaaclab.envs.mdp.actions import RelativeJointPositionAction, RelativeJointPositionActionCfg
from isaaclab.utils import configclass


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
