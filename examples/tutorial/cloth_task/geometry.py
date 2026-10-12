# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Adapted from isaac-sim/IsaacLabTutorial (Apache-2.0); see ../LICENSE-IsaacLabTutorial.

"""Vertex-based pinch geometry for the cloth-lifting task."""

import torch


def pinch_geometry(
    positions: torch.Tensor, fixed: torch.Tensor, moving: torch.Tensor, distance: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-jaw proximity flags and vertices near both fingertip pads.

    Requiring the same vertex near both jaws prevents opposite ends of the sheet touching different fingers from
    being mistaken for a pinch. Proximity alone is not a measurement of contact force.
    """
    fixed_near = (positions - fixed[:, None]).square().sum(dim=-1) <= distance**2
    moving_near = (positions - moving[:, None]).square().sum(dim=-1) <= distance**2
    return fixed_near.any(dim=-1), moving_near.any(dim=-1), fixed_near & moving_near
