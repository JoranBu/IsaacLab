# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression checks for the vial-lifting action interface and approach shaping."""

import newton
import torch
import warp as wp

wp.config.enable_backward = False

from isaaclab_newton.physics import NewtonManager  # noqa: E402
from vial_task import terms  # noqa: E402
from vial_task.env_cfg import ARM_JOINTS, TCP_OFFSET, ClothEnvCFG  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import quat_apply, subtract_frame_transforms  # noqa: E402


def test_approach_cycles_and_relative_ik():
    """A closed distance cycle cannot pay approach reward, and zero arm actions hold the fingertip pose."""
    cfg = ClothEnvCFG()
    cfg.scene.num_envs = 2
    cfg.seed = 42
    with launch_simulation(cfg, {"visualizer": ["none"]}) as physics_cfg:
        cfg.sim.physics = physics_cfg
        env = ManagerBasedRLEnv(cfg)
        try:
            env.reset()
            # Non-adjacent robot colliders must remain eligible for real self-contact in the imported model.
            model = NewtonManager.get_model()
            hand_body = next(index for index, name in enumerate(model.body_label) if name.endswith("/panda_hand"))
            base_body = next(index for index, name in enumerate(model.body_label) if name.endswith("/panda_link0"))
            shape_bodies = model.shape_body.numpy()
            shape_flags = model.shape_flags.numpy()
            hand_shapes = [
                index
                for index, body in enumerate(shape_bodies)
                if body == hand_body and shape_flags[index] & newton.ShapeFlags.COLLIDE_SHAPES
            ]
            base_shapes = [
                index
                for index, body in enumerate(shape_bodies)
                if body == base_body and shape_flags[index] & newton.ShapeFlags.COLLIDE_SHAPES
            ]
            assert hand_shapes and base_shapes
            assert any(
                tuple(sorted((hand, base))) not in model.shape_collision_filter_pairs
                for hand in hand_shapes
                for base in base_shapes
            )
            vial = env.scene["vial"]
            reward = env.reward_manager.get_term_cfg("approach_progress").func
            grasp_center = terms.grasp_center_w(env).clone()
            vial_pose = vial.data.root_pose_w.torch.clone()
            grasp_offset = quat_apply(
                vial_pose[:, 3:7], vial_pose.new_tensor(terms.VIAL_GRASP_OFFSET).expand(env.num_envs, -1)
            )

            def set_distance(distance: float):
                vial_pose[:, :3] = grasp_center - grasp_offset
                vial_pose[:, 0] += distance
                vial.write_root_pose_to_sim_index(root_pose=vial_pose)
                env.sim.forward()
                env.scene.update(env.physics_dt)

            set_distance(0.10)
            reward.reset()
            torch.testing.assert_close(reward(env), torch.zeros(env.num_envs, device=env.device))
            total = torch.zeros(env.num_envs, device=env.device)
            # One 10 cm retreat, followed by ten 1 cm approaches, returns to the starting distance.
            for distance in (0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10):
                set_distance(distance)
                total += reward(env)
            torch.testing.assert_close(total, torch.zeros_like(total), atol=1.0e-4, rtol=0.0)

            # Resetting one reward reference must not suppress progress in the other environment.
            set_distance(0.12)
            reward.reset(env_ids=[1])
            torch.testing.assert_close(reward(env), total.new_tensor([-2.0, 0.0]), atol=1.0e-4, rtol=0.0)

            obs, _ = env.reset()
            assert env.action_manager.total_action_dim == 7
            assert obs["policy"].shape == (2, 66)
            assert obs["critic"].shape == (2, 69)
            robot = env.scene["robot"]
            hand_id = robot.find_bodies("panda_hand")[0][0]

            def fingertip_pose():
                position = robot.data.body_pos_w.torch[:, hand_id]
                quaternion = robot.data.body_quat_w.torch[:, hand_id]
                position = position + quat_apply(quaternion, position.new_tensor(TCP_OFFSET).expand_as(position))
                return position.clone(), quaternion.clone()

            position_before, quaternion_before = fingertip_pose()
            actions = torch.zeros((env.num_envs, 7), device=env.device)
            for _ in range(20):
                obs, reward_value, terminated, truncated, _ = env.step(actions)
                assert all(torch.isfinite(value).all() for value in obs.values())
                assert torch.isfinite(reward_value).all()
                assert not terminated.any()
                assert not truncated.any()
            position_after, quaternion_after = fingertip_pose()
            torch.testing.assert_close(position_after, position_before, atol=0.003, rtol=0.0)
            alignment = (quaternion_after * quaternion_before).sum(dim=-1).abs()
            assert (alignment > 0.999).all()

            # The public action processor maps normalized inputs to small physical increments.
            actions[0, 0] = 1.0
            actions[1, 5] = 1.0
            env.action_manager.process_action(actions)
            expected = actions.new_tensor([[0.02, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.1]])
            torch.testing.assert_close(env.action_manager.get_term("arm").processed_actions, expected)
            obs, reward_value, _, _, _ = env.step(actions)
            assert all(torch.isfinite(value).all() for value in obs.values())
            assert torch.isfinite(reward_value).all()

            # Adversarial IK requests near joint limits must still produce bounded commands.
            arm_ids, _ = robot.find_joints(ARM_JOINTS, preserve_order=True)
            limits = robot.data.soft_joint_pos_limits.torch[:, arm_ids]
            position = robot.data.default_joint_pos.torch.clone()
            position[0, arm_ids] = limits[0, :, 1] - 0.001
            position[1, arm_ids] = limits[1, :, 0] + 0.001
            robot.write_joint_position_to_sim_index(position=position)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(position))
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(env.physics_dt)
            fingertip_position, _ = fingertip_pose()
            position_b, _ = subtract_frame_transforms(
                robot.data.root_link_pos_w.torch, robot.data.root_link_quat_w.torch, fingertip_position
            )
            env.action_manager.process_action(actions.new_full((2, 7), 10.0))
            arm = env.action_manager.get_term("arm")
            goal = position_b + arm.processed_actions[:, :3]
            assert (goal >= goal.new_tensor(arm.cfg.workspace_lower) - 1.0e-6).all()
            assert (goal <= goal.new_tensor(arm.cfg.workspace_upper) + 1.0e-6).all()
            env.action_manager.apply_action()
            target = robot.actuators.target_command.position.torch[:, arm_ids]
            assert torch.isfinite(target).all()
            assert (target >= limits[..., 0] - 1.0e-6).all()
            assert (target <= limits[..., 1] + 1.0e-6).all()
            actual_position = robot.data.joint_pos.torch[:, arm_ids]
            max_joint_step = arm.cfg.joint_target_speed_limit * env.physics_dt
            assert ((target - actual_position).abs() <= max_joint_step + 1.0e-6).all()
        finally:
            env.close()
