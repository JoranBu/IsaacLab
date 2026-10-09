# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check real simulation, partial resets, timeout semantics, and physical placement."""

import gymnasium as gym
import pytest
import torch
import warp as wp

wp.config.enable_backward = False

import vial_task  # noqa: E402
from vial_task.env_cfg import RESET_DATASET  # noqa: E402
from vial_task.resets import load_reset_dataset, save_reset_dataset  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry, resolve_presets  # noqa: E402


def test_reset_step_and_placement(tmp_path):
    """Exercise the example's complete lifecycle in one small CUDA scene."""
    cfg = load_cfg_from_registry(vial_task.TASK_ID, "env_cfg_entry_point")
    resolve_presets(cfg)
    cfg.scene.num_envs = 4
    cfg.seed = 42
    states = load_reset_dataset(RESET_DATASET)["states"]
    dataset = tmp_path / "generated.npz"
    save_reset_dataset(dataset, states, generator={"seed": 42}, validation={"physics": "newton_mjwarp"})
    assert load_reset_dataset(dataset)["generator"]["seed"] == 42
    # Invalid output must preserve the last usable reset archive.
    original_bytes = dataset.read_bytes()
    invalid = {**states, "vial_pose": states["vial_pose"].clone()}
    invalid["vial_pose"][0, 3:7] = 0.0
    with pytest.raises(ValueError, match="normalized"):
        save_reset_dataset(dataset, invalid, generator={}, validation={})
    assert dataset.read_bytes() == original_bytes
    cfg.events.reset_from_dataset.params["dataset_path"] = str(dataset)
    cfg.events.reset_from_dataset.params["phase_weights"] = (1.0,) + (0.0,) * 7
    cfg.episode_length_s = 0.2
    with launch_simulation(cfg, {"visualizer": ["none"]}):
        env = gym.make(vial_task.TASK_ID, cfg=cfg).unwrapped
        try:
            obs, _ = env.reset()
            assert env.action_space.shape == (4, 6)
            assert obs["policy"].shape == (4, 60)
            assert obs["critic"].shape == (4, 63)
            assert all(torch.isfinite(value).all() for value in obs.values())
            robot = env.scene["robot"]
            vial = env.scene["vial"]
            actions = torch.zeros((4, 6), device=env.device)
            for _ in range(2):
                env.step(actions)
            untouched_joints = robot.data.joint_pos.torch[[0, 2]].clone()
            untouched_vials = vial.data.root_pose_w.torch[[0, 2]].clone()
            untouched_steps = env.episode_length_buf[[0, 2]].clone()
            env.reset(env_ids=torch.tensor([3, 1], device=env.device))
            torch.testing.assert_close(robot.data.joint_pos.torch[[0, 2]], untouched_joints)
            torch.testing.assert_close(vial.data.root_pose_w.torch[[0, 2]], untouched_vials)
            assert torch.equal(env.episode_length_buf[[0, 2]], untouched_steps)
            assert not env.episode_length_buf[[1, 3]].any()
            assert not env.action_manager.action[[1, 3]].any()
            assert not obs["policy"][:, -3:].any()
            timeout_count = 0
            for _ in range(12):
                obs, reward, terminated, truncated, _ = env.step(actions)
                assert all(torch.isfinite(value).all() for value in obs.values())
                assert torch.isfinite(reward).all()
                assert not terminated.any()
                timeout_count += int(truncated.sum())
            assert timeout_count >= 4

            # Place a released vial at the centre of the physical rack opening.
            # A longer horizon lets the vial settle and proves success through real contacts.
            env.cfg.episode_length_s = 5.0
            env.reset()
            rack_position = env.scene["rack"].data.root_pos_w.torch[0].clone()
            pose = torch.tensor([[0.0, 0.0, 0.031, 0.0, 0.0, 0.0, 1.0]], device=env.device)
            pose[:, :3] += rack_position
            selected = torch.tensor([0], device=env.device)
            vial.write_root_pose_to_sim_index(root_pose=pose, env_ids=selected)
            vial.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((1, 6), device=env.device), env_ids=selected
            )
            env.scene.write_data_to_sim()
            env.sim.forward()
            found_success = False
            for _ in range(60):
                _, reward, terminated, truncated, _ = env.step(actions)
                assert torch.isfinite(reward).all()
                if env.termination_manager.get_term("success")[0]:
                    assert terminated[0] and not truncated[0]
                    found_success = True
                    break
            assert found_success, "A stable, upright, released vial inside the rack must succeed."
        finally:
            env.close()
