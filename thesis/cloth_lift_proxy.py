# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The cloth-lift experiment with an MJWarp robot and a VBD cloth.

Run from the IsaacLab repository root::

    uv run --extra viser python thesis/cloth_lift_proxy.py --viz viser
    uv run python thesis/cloth_lift_proxy.py --viz none --num_steps 900

The scene, materials, drives, IK actions, and scripted trajectory are imported
from cloth_lift.py. Only the physics solver configuration changes. Editing the
original experiment therefore updates both versions together.
"""

from cloth_lift import ClothLiftEnvCfg, main
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCollisionPipelineCfg

from isaaclab.utils import configclass, replace

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg


@configclass
class ClothLiftProxyEnvCfg(ClothLiftEnvCfg):
    """Keep the original experiment and replace its single solver with a coupler."""

    def __post_init__(self):
        super().__post_init__()
        original_physics = self.sim.physics
        robot_bodies = [r"/World/envs/env_[^/]+/Robot"]
        # Copy every shared physics setting. Clear the derived manager so the
        # new NewtonCfg selects the coupler manager instead of the VBD manager.
        self.sim.physics = replace(
            original_physics,
            class_type=None,
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(
                        name="robot",
                        solver_cfg=MJWarpSolverCfg(cone="elliptic", integrator="implicitfast"),
                        bodies=robot_bodies,
                        # Static floor geometry must be available to the rigid
                        # solver too, so the robot can collide with the ground.
                        include_static_shapes=True,
                    ),
                    CouplerEntryCfg(
                        name="cloth",
                        # Retain the same VBD iterations and buffer sizes.
                        solver_cfg=original_physics.solver_cfg,
                        all_particles=True,
                        # No explicit shape ownership: Newton's destination
                        # view includes static shapes (the floor) by default,
                        # alongside the robot bodies added by the proxy below.
                    ),
                ],
                proxies=[
                    CouplerProxyMappingCfg(
                        source="robot",
                        destination="cloth",
                        bodies=robot_bodies,
                        mode="lagged",
                        collide_interval=1,
                        # Match the original vertex/shape contacts. Full-surface
                        # contact would require different robot collision assets.
                        collision_pipeline=NewtonCollisionPipelineCfg(),
                    ),
                ],
                iterations=1,
            ),
        )


if __name__ == "__main__":
    main(cfg=ClothLiftProxyEnvCfg(), solver_name="mjwarp_vbd_proxy", description=__doc__)
