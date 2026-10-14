# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cloth lift geometry, progress, reset, and coupled-simulation regression checks."""

import pytest
import torch
import warp as wp

wp.config.enable_backward = False

from cloth_task import resets, terms  # noqa: E402
from cloth_task.env_cfg import FINGER_JOINTS, ClothEnvCFG  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402


def test_cloth_lift_and_partial_reset(tmp_path):
    """Require the whole sheet to lift, expose cloth motion, and reset only selected episodes."""
    cfg = ClothEnvCFG()
    # This fixture exercises the ordinary reset and builds its own archive.
    cfg.events.reset_from_dataset.params["dataset_path"] = None
    cfg.scene.num_envs = 2
    cfg.seed = 42
    with launch_simulation(cfg, {"visualizer": ["none"]}) as physics_cfg:
        cfg.sim.physics = physics_cfg
        env = ManagerBasedRLEnv(cfg)
        try:
            obs, _ = env.reset()
            assert env.action_manager.total_action_dim == 7
            assert obs["policy"].shape == (2, 107)
            assert obs["critic"].shape == (2, 110)
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
            torch.testing.assert_close(reward(env) * env.step_dt, torch.zeros(2, device=env.device))
            total = torch.zeros(2, device=env.device)
            for distance in (0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10):
                set_distance(distance)
                total += reward(env) * env.step_dt
            torch.testing.assert_close(total, torch.zeros_like(total), atol=1.0e-4, rtol=0.0)
            set_distance(0.12)
            reward.reset(env_ids=[1])
            torch.testing.assert_close(reward(env) * env.step_dt, total.new_tensor([-2.0, 0.0]), atol=1.0e-4, rtol=0.0)

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

            def update_success():
                env.termination_manager.compute()
                return env.termination_manager.get_term("success")

            # Raising the grasp corner while any other vertex stays on the mat is not a completed lift.
            state[:, -1, 2] = env.scene.env_origins[:, 2] + terms.MAT_TOP_HEIGHT + terms.CLOTH_PARTICLE_RADIUS
            write_cloth()
            for _ in range(12):
                assert not update_success().any()
            assert not terms.lift_features(env)[:, -1].any()

            # Height alone can complete the task without a pinch; opening does not suppress shaping.
            history_term.reset()
            action[:, -1] = 1.0
            env.action_manager.process_action(action)
            state[..., 2] = centre[:, None, 2]
            write_cloth()
            assert not terms.bilateral_contact(env).any()
            torch.testing.assert_close(terms.lift_reward(env), torch.ones(2, device=env.device))
            for _ in range(9):
                assert not update_success().any()
            # A dip below the height threshold restarts the consecutive-step count.
            state[:, -1, 2] = env.scene.env_origins[:, 2] + terms.MAT_TOP_HEIGHT + terms.CLOTH_PARTICLE_RADIUS
            write_cloth()
            assert not update_success().any()
            assert not terms.lift_features(env)[:, -1].any()
            state[..., 2] = centre[:, None, 2]
            write_cloth()
            for _ in range(9):
                assert not update_success().any()
                assert not terms.success_bonus(env).any()
            reward.reset()
            baseline = env.reward_manager.compute(dt=env.step_dt).clone()
            assert update_success().all()
            terminal_reward = env.reward_manager.compute(dt=env.step_dt)
            # Terminal payouts are discrete bonuses, not reward rates multiplied by dt.
            torch.testing.assert_close(terminal_reward - baseline, baseline.new_full((2,), 200.0))
            assert not terms.failure_penalty(env).any()

            # Keep a closed-jaw snapshot to test reset observations later.
            action[:, -1] = -1.0
            env.action_manager.process_action(action)
            env.action_manager.apply_action()
            grasp_snapshot = {key: value[:1] for key, value in resets.capture_state(env, phase=2).items()}

            # Dense shaping credits a half-height sheet, and caps at the target height.
            action[:, -1] = 1.0
            env.action_manager.process_action(action)
            state[..., 2] = env.scene.env_origins[:, None, 2] + 0.035 + 0.003 + 0.05
            write_cloth()
            torch.testing.assert_close(terms.lift_reward(env), torch.full((2,), 0.5, device=env.device))
            assert not terms.cloth_lost(env).any()
            action[:, -1] = -1.0
            env.action_manager.process_action(action)

            # An actual environment partial reset restores one mesh and its progress, preserving the other.
            preserved = cloth.data.nodal_pos_w.torch[1].clone()
            env.reset(env_ids=[0])
            torch.testing.assert_close(cloth.data.nodal_pos_w.torch[0], default[0, :, :3])
            torch.testing.assert_close(cloth.data.nodal_pos_w.torch[1], preserved)
            assert terms.lift_features(env)[0, -1] == 0
            assert terms.lift_features(env)[1, -1] == 1

            # Invalid geometry is a terminal failure but must not leak NaNs into observations/rewards.
            state = cloth.data.nodal_state_w.torch.clone()
            state[0, 0, 0] = float("nan")
            write_cloth()
            assert terms.cloth_lost(env)[0]
            assert not update_success()[0]
            torch.testing.assert_close(terms.failure_penalty(env) * env.step_dt, baseline.new_tensor([1.0, 0.0]))
            assert torch.isfinite(env.reward_manager.compute(dt=env.step_dt)).all()
            observations = env.observation_manager.compute()
            assert all(torch.isfinite(value).all() for value in observations.values())
            env.reset()
            # An invalid robot velocity terminates with a finite reward and a failure penalty.
            velocity = robot.data.joint_vel.torch.clone()
            velocity[0, 0] = float("nan")
            robot.write_joint_velocity_to_sim_index(velocity=velocity)
            env.scene.update(env.physics_dt)
            update_success()
            assert env.termination_manager.get_term("unstable_robot")[0]
            torch.testing.assert_close(terms.failure_penalty(env) * env.step_dt, baseline.new_tensor([1.0, 0.0]))
            assert torch.isfinite(env.reward_manager.compute(dt=env.step_dt)).all()
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

            # Timeouts are neither successful lifts nor physical failures.
            env.episode_length_buf.fill_(env.max_episode_length)
            update_success()
            assert env.termination_manager.time_outs.all()
            assert not terms.failure_penalty(env).any()
            assert not terms.success_bonus(env).any()
            env.reset()

            # The reset archive must replay nodal geometry in the destination
            # world, release any support, and preserve unrelated environments.
            snapshot = grasp_snapshot
            path = tmp_path / "cloth_resets.npz"
            resets.save_dataset(path, env, snapshot, metadata={"validation": "integration"})
            original_bytes = path.read_bytes()
            invalid = {**snapshot, "cloth_nodal_state": snapshot["cloth_nodal_state"].clone()}
            invalid["cloth_nodal_state"][0, 0, 0] = float("nan")
            with pytest.raises(ValueError, match="cloth_nodal_state"):
                resets.save_dataset(path, env, invalid, metadata={})
            assert path.read_bytes() == original_bytes

            event = env.event_manager.get_term_cfg("reset_from_dataset")
            event.params["dataset_path"] = str(path)
            event.params["phase_weights"] = (0.0, 0.0, 1.0, 0.0)
            event.func = resets.ResetFromDataset(event, env)
            env.event_manager.set_term_cfg("reset_from_dataset", event)
            preserved_nodes = cloth.data.nodal_state_w.torch[0].clone()
            preserved_joints = robot.data.joint_pos.torch[0].clone()
            preserved_action = env.action_manager.action[0].clone()
            preserved_previous_action = env.action_manager.prev_action[0].clone()
            env.reset(env_ids=torch.tensor([1], device=env.device))
            expected = snapshot["cloth_nodal_state"][0].clone()
            expected[:, :3] += env.scene.env_origins[1]
            torch.testing.assert_close(cloth.data.nodal_state_w.torch[1], expected)
            torch.testing.assert_close(cloth.data.nodal_state_w.torch[0], preserved_nodes)
            torch.testing.assert_close(robot.data.joint_pos.torch[0], preserved_joints)
            torch.testing.assert_close(robot.actuators.target_command.position.torch[1], snapshot["joint_target"][0])
            assert (cloth.data.nodal_kinematic_target.torch[1, :, 3] == 1).all()
            assert env.action_manager.action[1, -1] == -1
            assert env.action_manager.prev_action[1, -1] == -1
            assert terms.bilateral_contact(env)[1]
            assert terms.lift_features(env)[1, -1] == 0
            torch.testing.assert_close(env.action_manager.action[0], preserved_action)
            torch.testing.assert_close(env.action_manager.prev_action[0], preserved_previous_action)
        finally:
            env.close()
