# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Adapted from isaac-sim/IsaacLabTutorial (Apache-2.0); see ../LICENSE-IsaacLabTutorial.

"""Orientation-aware clearance geometry for the vial-lifting task."""

import torch


def cylinder_lowest_offset(axis_z: torch.Tensor, axial_min: float, axial_max: float, radius: float) -> torch.Tensor:
    """Return the lowest point of an oriented finite cylinder relative to its root."""
    axis_z = axis_z.clamp(-1.0, 1.0)
    axial = torch.where(axis_z >= 0.0, axial_min * axis_z, axial_max * axis_z)
    radial = -radius * torch.sqrt((1.0 - axis_z.square()).clamp_min(0.0))
    return axial + radial
