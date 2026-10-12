# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Adapted from isaac-sim/IsaacLabTutorial (Apache-2.0); see ../LICENSE-IsaacLabTutorial.

"""Episode-local task milestones with safe partial-reset semantics."""

from collections.abc import Sequence

import torch


class LiftProgress:
    """Track grasp/lift milestones and sustained lift success.

    Each milestone is a physical fact that is latched the first time it is observed in an episode:

    * ``grasped``: both jaws touch the vial while it is held off the mat,
    * ``lifted``: the grasped vial reaches the target clearance above the mat.

    ``success`` latches after a confirmed grasp is held above the target clearance for ``stable_steps`` consecutive
    control steps. Losing the live grasp or falling below the target resets the hold counter.
    """

    def __init__(self, num_envs: int, device: str | torch.device, stable_steps: int = 10, grasp_steps: int = 2):
        self.stable_steps = stable_steps
        self.grasp_steps = grasp_steps
        self.grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.lifted = torch.zeros_like(self.grasped)
        self.success = torch.zeros_like(self.grasped)
        self.grasp_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.stable_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.time_to_success = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    def reset(self, env_ids: Sequence[int] | slice | torch.Tensor | None = None) -> None:
        """Reset only the requested environments."""
        ids = slice(None) if env_ids is None else env_ids
        self.grasped[ids] = False
        self.lifted[ids] = False
        self.success[ids] = False
        self.grasp_count[ids] = 0
        self.stable_count[ids] = 0
        self.time_to_success[ids] = -1

    def update(
        self,
        holding: torch.Tensor,
        above_target: torch.Tensor,
        step: torch.Tensor,
    ) -> torch.Tensor:
        """Latch milestones from the current physical state and return confirmed success."""
        self.grasp_count.copy_(torch.where(holding, self.grasp_count + 1, torch.zeros_like(self.grasp_count)))
        self.grasped |= self.grasp_count >= self.grasp_steps
        held_at_target = holding & above_target
        self.lifted |= self.grasped & held_at_target
        self.stable_count.copy_(torch.where(held_at_target, self.stable_count + 1, torch.zeros_like(self.stable_count)))
        newly_successful = ~self.success & self.grasped & (self.stable_count >= self.stable_steps)
        self.time_to_success.copy_(torch.where(newly_successful, step, self.time_to_success))
        self.success |= newly_successful
        return self.success.clone()
