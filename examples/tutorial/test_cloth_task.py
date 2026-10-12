# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cloth lift geometry, progress, reset, and coupled-simulation regression checks."""

import torch
import warp as wp

wp.config.enable_backward = False

from cloth_task import terms  # noqa: E402
from cloth_task.env_cfg import FINGER_JOINTS, ClothEnvCFG  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402


def test_cloth_lift_and_partial_reset():
    """Require the whole sheet to lift, expose cloth motion, and reset only selected episodes."""
    cfg = ClothEnvCFG()
    cfg.scene.num_envs = 2
    cfg.seed = 42
    with launch_simulation(cfg, {"visualizer": ["none"]}) as physics_cfg:
        cfg.sim.physics = physics_cfg
        env = ManagerBasedRLEnv(cfg)
        try:
            obs, _ = env.reset()
            assert env.action_manager.total_action_dim == 7
            assert obs["policy"].shape == (2, 110)
            assert obs["critic"].shape == (2, 113)
            cloth = env.scene["cloth"]
            default = cloth.data.default_nodal_state_w.torch.clone()
            state = default.clone()

            def write_cloth():
                cloth.write_nodal_state_to_sim_index(nodal_state=state)
                env.scene.update(env.physics_dt)

            # Closed approach cycles cannot pay a net distance reward, including after partial resets.
            reward = env.reward_manager.get_term_cfg("approach_progress").func
            point = terms.cloth_grasp_point_w(env).clone()
            centre = terms.grasp_center_w(env).clone()

            def set_distance(distance):
                state.copy_(default)
                state[..., :3] += (centre - point)[:, None]
                state[..., 0] += distance
                write_cloth()

            set_distance(0.10)
            reward.reset()
            torch.testing.assert_close(reward(env), torch.zeros(2, device=env.device))
            total = torch.zeros(2, device=env.device)
            for distance in (0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10):
                set_distance(distance)
                total += reward(env)
            torch.testing.assert_close(total, torch.zeros_like(total), atol=1.0e-4, rtol=0.0)
            set_distance(0.12)
            reward.reset(env_ids=[1])
            torch.testing.assert_close(reward(env), total.new_tensor([-2.0, 0.0]), atol=1.0e-4, rtol=0.0)

            # Close the real finger joints and put a cloth corner between the pads.
            robot = env.scene["robot"]
            fingers, _ = robot.find_joints(FINGER_JOINTS, preserve_order=True)
            q = robot.data.default_joint_pos.torch.clone()
            q[:, fingers] = 0.0
            robot.write_joint_position_to_sim_index(position=q)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
            env.sim.forward()
            env.scene.update(env.physics_dt)
            action = torch.zeros((2, 7), device=env.device)
            action[:, -1] = -1.0
            env.action_manager.process_action(action)
            centre = terms.grasp_center_w(env).clone()
            point = terms.cloth_grasp_point_w(env).clone()
            state[..., :3] += (centre - point)[:, None]
            # Put every vertex in one plane at the held corner height.
            state[..., 2] = centre[:, None, 2]
            state[..., 3:] = 0.0
            write_cloth()
            assert terms.bilateral_contact(env).all()
            assert (terms.cloth_clearance(env) >= terms.LIFT_TARGET_CLEARANCE).all()
            history_term = env.termination_manager.get_term_cfg("success").func
            history_term.reset()

            # Raising the grasp corner while any other vertex stays on the mat is not a completed lift.
            state[:, -1, 2] = env.scene.env_origins[:, 2] + terms.MAT_TOP_HEIGHT + terms.CLOTH_PARTICLE_RADIUS
            write_cloth()
            for _ in range(12):
                assert not history_term(env).any()
            assert not terms.progress_flags(env)[:, 1].any()

            # Full-sheet clearance plus the continuous pinch must persist for ten control steps.
            state[..., 2] = centre[:, None, 2]
            write_cloth()
            for _ in range(9):
                assert not history_term(env).any()
            assert history_term(env).all()
            assert terms.progress_flags(env).bool().all()

            # Released cloth cannot earn dense lift reward, even while still airborne.
            action[:, -1] = 1.0
            env.action_manager.process_action(action)
            torch.testing.assert_close(terms.lift_reward(env), torch.zeros(2, device=env.device))
            action[:, -1] = -1.0
            env.action_manager.process_action(action)

            # An actual environment partial reset restores one mesh and its progress, preserving the other.
            preserved = cloth.data.nodal_pos_w.torch[1].clone()
            env.reset(env_ids=[0])
            torch.testing.assert_close(cloth.data.nodal_pos_w.torch[0], default[0, :, :3])
            torch.testing.assert_close(cloth.data.nodal_pos_w.torch[1], preserved)
            assert not terms.progress_flags(env)[0].any()
            assert terms.progress_flags(env)[1].bool().all()

            # Invalid geometry is a terminal failure but must not leak NaNs into observations/rewards.
            state = cloth.data.nodal_state_w.torch.clone()
            state[0, 0, 0] = float("nan")
            write_cloth()
            assert terms.cloth_lost(env)[0]
            observations = env.observation_manager.compute()
            assert all(torch.isfinite(value).all() for value in observations.values())
            env.reset()
            initial_height = terms.cloth_height(env).clone()
            for _ in range(20):
                obs, value, terminated, truncated, _ = env.step(torch.zeros_like(action))
                assert all(torch.isfinite(item).all() for item in obs.values())
                assert torch.isfinite(value).all()
                assert not terminated.any()
                assert not truncated.any()
            # The cloth must actually advance under gravity and settle on the support surface.
            assert (terms.cloth_height(env) < initial_height).all()
            assert (terms.cloth_clearance(env) > -0.01).all()
        finally:
            env.close()
