# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prove that generated cloth reset banks survive unassisted archive replay."""

import torch
import warp as wp

wp.config.enable_backward = False

from cloth_task import generator, resets, terms  # noqa: E402
from cloth_task.env_cfg import ClothEnvCFG  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402


def test_generated_banks_survive_reset_and_relocation(tmp_path):
    """Generate all four regions and retain the airborne pinches after restoring saved rows."""
    cfg = ClothEnvCFG()
    cfg.scene.num_envs = 4
    cfg.seed = 42
    # The offline generator uses the base environment so reward milestones
    # cannot automatically reset a proposal while the flat sheet is settling.
    with launch_simulation(cfg, {"visualizer": ["none"]}) as physics:
        cfg.sim.physics = physics
        env = ManagerBasedEnv(cfg)
        try:
            settings = generator.GeneratorCfg(poses_per_phase=1, max_batches=2, ik_iterations=80)
            states, statistics = generator.generate(env, settings, range(4))
            assert sorted(states["phase"].tolist()) == [0, 1, 2, 3]
            assert all(statistics[f"{name}/accepted"] == 1 for name in resets.PHASE_NAMES)
            path = tmp_path / "generated.npz"
            resets.save_dataset(path, env, states, metadata={})
            saved = resets.load_dataset(path, env)

            # Restore the airborne banks into different environment origins.
            # A warm simulation can hide a slipping pinch; this cold replay
            # must hold the cloth using only the normal policy action interface.
            env.reset()
            rows = torch.tensor([2, 3, 2, 3], device=env.device)
            ids = torch.arange(env.num_envs, device=env.device)
            resets.apply_states(env, saved, rows, ids)
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(env.physics_dt)
            corner = terms.cloth_grasp_point_w(env).clone()
            generator._hold(env, closed=True, steps=30)
            assert torch.isfinite(env.scene["cloth"].data.nodal_state_w.torch).all()
            assert (env.scene["cloth"].data.nodal_kinematic_target.torch[..., 3] == 1).all()
            assert terms.bilateral_contact(env).all()
            assert ((terms.cloth_grasp_point_w(env) - corner).norm(dim=-1) < 0.01).all()
            clearance = terms.cloth_clearance(env)
            assert ((clearance[::2] > 0.0) & (clearance[::2] < 0.075)).all()
            assert ((clearance[1::2] >= 0.075) & (clearance[1::2] < terms.LIFT_TARGET_CLEARANCE)).all()
        finally:
            env.close()
