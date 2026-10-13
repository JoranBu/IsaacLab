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
    """Generate supported and airborne grasps and retain both after restoring saved rows."""
    cfg = ClothEnvCFG()
    cfg.events.reset_from_dataset.params["dataset_path"] = None
    cfg.scene.num_envs = 4
    cfg.seed = 42
    # The offline generator uses the base environment so reward milestones
    # cannot automatically reset a proposal during the physical pickup.
    with launch_simulation(cfg, {"visualizer": ["none"]}) as physics:
        cfg.sim.physics = physics
        env = ManagerBasedEnv(cfg)
        try:
            settings = generator.GeneratorCfg(poses_per_phase=2, max_batches=4, ik_iterations=80)
            states, statistics = generator.generate(env, settings, range(4))
            assert torch.bincount(states["phase"].long()).tolist() == [2, 2, 2, 2]
            assert all(statistics[f"{name}/accepted"] == 2 for name in resets.PHASE_NAMES)
            path = tmp_path / "generated.npz"
            resets.save_dataset(path, env, states, metadata={})
            saved = resets.load_dataset(path, env)

            grasp_rows = (saved["phase"] == 2).nonzero().flatten()
            grasp_clearance = saved["cloth_nodal_state"][grasp_rows, :, 2].amin(dim=-1)
            grasp_clearance -= terms.MAT_TOP_HEIGHT + terms.CLOTH_PARTICLE_RADIUS
            supported = grasp_rows[grasp_clearance.abs() < 0.002]
            airborne = grasp_rows[grasp_clearance > 0.002]
            assert len(supported) >= 1, "The grasped bank must include cloth still partially on the table."
            assert len(airborne) >= 1, "The grasped bank must also include lifted cloth."
            near_goal = (saved["phase"] == 3).nonzero().flatten()

            # Restore both grasped categories into different environment origins.
            # A warm simulation can hide a slipping pinch; this cold replay
            # must hold the cloth using only the normal policy action interface.
            env.reset()
            rows = torch.stack((supported[0], airborne[0], near_goal[0], supported[0]))
            ids = torch.arange(env.num_envs, device=env.device)
            resets.apply_states(env, saved, rows, ids)
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(env.physics_dt)
            generator._hold(env, closed=True, steps=30)
            assert torch.isfinite(env.scene["cloth"].data.nodal_state_w.torch).all()
            assert (env.scene["cloth"].data.nodal_kinematic_target.torch[..., 3] == 1).all()
            assert terms.bilateral_contact(env).all()
            clearance = terms.cloth_clearance(env)
            assert (clearance[[0, 3]].abs() < 0.002).all()
            nodes = env.scene["cloth"].data.nodal_pos_w.torch
            height = nodes[..., 2] - env.scene.env_origins[:, None, 2]
            on_mat = (height - terms.MAT_TOP_HEIGHT - terms.CLOTH_PARTICLE_RADIUS).abs() < 0.002
            assert (on_mat[[0, 3]].sum(dim=-1) >= 9).all()
            assert (height[[0, 3]].amax(dim=-1) > terms.MAT_TOP_HEIGHT + 0.02).all()
            assert 0.002 < clearance[1] < 0.075
            assert 0.075 <= clearance[2] < terms.LIFT_TARGET_CLEARANCE
        finally:
            env.close()
