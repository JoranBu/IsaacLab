# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate OmniReset-inspired snapshots for the Franka cloth task.

From the repository root::

    uv run --no-sync python -m examples.tutorial.cloth_task.generator \
        --poses_per_phase 16 --num_envs 4 --output logs/cloth_resets.npz --viz none
    uv run --no-sync python -m examples.tutorial.cloth_task.generator \
        --view logs/cloth_resets.npz --viz viser

The generator constructs four independent reset regions, not demonstration
trajectories. For grasp proposals it lifts the whole undeformed sheet flat
by prescribing temporary flat nodal states while positioning/closing the robot.
This bypasses the difficult exploration needed to pinch a flat sheet. It then
releases EVERY vertex and validates the snapshot under normal table collision,
gravity, and robot dynamics. No kinematic target or relaxed workspace is saved.

``--table_penetration`` only relaxes the generator's IK workspace during offline
pose construction. It does not disable the table or change training physics.
Interpenetrating proposals must survive the same unassisted validation checks.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import warp as wp

wp.config.enable_backward = False

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    compute_pose_error,
    quat_apply,
    quat_apply_inverse,
    quat_mul,
    subtract_frame_transforms,
)

from . import terms  # noqa: E402
from .env_cfg import FINGER_JOINTS, TCP_OFFSET, ClothEnvCFG  # noqa: E402
from .resets import PHASE_NAMES, apply_states, capture_state, load_dataset, save_dataset  # noqa: E402


@dataclass
class GeneratorCfg:
    """Small, task-specific proposal ranges; all distances are metres."""

    poses_per_phase: int = 16
    max_batches: int = 64
    seed: int = 42
    xy_jitter: float = 0.01
    yaw_jitter: float = 0.15
    near_corner_distance: tuple[float, float] = (0.012, 0.045)
    flat_lift_height: tuple[float, float] = (0.29, 0.34)
    table_penetration: float = 0.0
    ik_iterations: int = 160
    close_steps: int = 12
    lift_steps: int = 12
    validation_steps: int = 20  # 0.67 s at this task's 30 Hz control rate.
    near_goal_clearance: tuple[float, float] = (0.075, 0.095)

    def __post_init__(self):
        for name in (
            "poses_per_phase",
            "max_batches",
            "ik_iterations",
            "close_steps",
            "lift_steps",
            "validation_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 <= self.table_penetration <= 0.01:
            raise ValueError("table_penetration must be between 0 and 0.01 m.")


def _tcp_pose(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the same controlled fingertip frame used by the policy."""
    robot = env.scene["robot"]
    hand = robot.find_bodies("panda_hand")[0][0]
    quaternion = robot.data.body_quat_w.torch[:, hand]
    position = robot.data.body_pos_w.torch[:, hand]
    position = position + quat_apply(quaternion, position.new_tensor(TCP_OFFSET).expand_as(position))
    return position, quaternion


def _solve_pose(
    env: ManagerBasedEnv, position: torch.Tensor, quaternion: torch.Tensor, opening: float, iterations: int
):
    """Offline IK: use the real policy controller and FK, without advancing physics.

    Each iteration applies the bounded IK command then writes its proposed joint
    coordinates for the next FK evaluation. This is pose construction, not a
    robot trajectory. Subsequent simulation checks decide whether the pose is usable.
    """
    robot = env.scene["robot"]
    fingers = robot.find_joints(FINGER_JOINTS, preserve_order=True)[0]
    scale = position.new_tensor(env.cfg.actions.arm.scale)
    root_pos, root_quat = robot.data.root_link_pos_w.torch, robot.data.root_link_quat_w.torch
    desired_pos, desired_quat = subtract_frame_transforms(root_pos, root_quat, position, quaternion)
    action = position.new_zeros((env.num_envs, 7))
    action[:, -1] = 1.0 if opening > 0.02 else -1.0
    for _ in range(iterations):
        current_pos, current_quat = _tcp_pose(env)
        current_pos, current_quat = subtract_frame_transforms(root_pos, root_quat, current_pos, current_quat)
        error_pos, error_rot = compute_pose_error(current_pos, current_quat, desired_pos, desired_quat)
        action[:, :6] = (torch.cat((error_pos, error_rot), dim=-1) / scale).clamp(-1.0, 1.0)
        env.action_manager.process_action(action)
        env.action_manager.apply_action()
        q = robot.actuators.target_command.position.torch.clone()
        q[:, fingers] = opening
        robot.write_joint_position_to_sim_index(position=q)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        robot.actuators.target_command.set_position_index(value=q)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(env.physics_dt)
    current_pos, current_quat = _tcp_pose(env)
    pe, re = compute_pose_error(current_pos, current_quat, position, quaternion)
    valid = (pe.norm(dim=-1) < 0.005) & (re.norm(dim=-1) < 0.15)
    return valid


def _hold(
    env: ManagerBasedEnv,
    closed: bool,
    steps: int,
    pose: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Hold a fixed world pose through ordinary policy actions, correcting contact-induced drift."""
    if pose is None:
        pose = tuple(value.clone() for value in _tcp_pose(env))
    robot = env.scene["robot"]
    root_pos, root_quat = robot.data.root_link_pos_w.torch, robot.data.root_link_quat_w.torch
    desired_pos, desired_quat = subtract_frame_transforms(root_pos, root_quat, *pose)
    scale = desired_pos.new_tensor(env.cfg.actions.arm.scale)
    action = desired_pos.new_zeros((env.num_envs, 7))
    action[:, -1] = -1.0 if closed else 1.0
    for _ in range(steps):
        position, quaternion = subtract_frame_transforms(root_pos, root_quat, *_tcp_pose(env))
        pe, re = compute_pose_error(position, quaternion, desired_pos, desired_quat)
        action[:, :6] = (torch.cat((pe, re), dim=-1) / scale).clamp(-1.0, 1.0)
        env.step(action)


def _free_cloth(env: ManagerBasedEnv) -> None:
    """Release every temporary support target; both replay and validation must use free cloth."""
    positions = env.scene["cloth"].data.nodal_pos_w.torch
    targets = torch.cat((positions, torch.ones_like(positions[..., :1])), dim=-1)
    env.scene["cloth"].write_nodal_kinematic_target_to_sim_index(targets)
    env.scene.write_data_to_sim()


def _validate(env: ManagerBasedEnv, phase: int, steps: int) -> torch.Tensor:
    """Replay-style proof under real physics; a supported sheet or a brief flying contact is not a stable grasp."""
    closed = phase >= 2
    valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    _free_cloth(env)
    anchor = tuple(value.clone() for value in _tcp_pose(env))
    initial_corner = terms.cloth_grasp_point_w(env).clone()
    for step in range(steps):
        pose = anchor
        if closed:
            # Allow small settling corrections through the same bounded policy
            # actions, while rejecting a freely falling or escaping corner.
            drift = terms.cloth_grasp_point_w(env) - initial_corner
            valid &= drift.norm(dim=-1) < 0.02
            pose = (anchor[0] + drift.clamp(-0.02, 0.02), anchor[1])
        _hold(env, closed, 1, pose)
        robot, cloth = env.scene["robot"], env.scene["cloth"]
        valid &= torch.isfinite(cloth.data.nodal_state_w.torch).all(dim=(1, 2))
        valid &= torch.isfinite(robot.data.joint_pos.torch).all(dim=-1)
        valid &= ~terms.unstable_robot(env) & ~terms.cloth_lost(env)
        if closed and step >= steps // 2:
            valid &= terms.bilateral_contact(env)
            valid &= terms.cloth_height(env) > terms.MAT_TOP_HEIGHT + terms.GRASP_PROOF_LIFT
        if not closed:
            valid &= terms.cloth_clearance(env).abs() < 0.01
    clearance = terms.cloth_clearance(env)
    if phase == 2:
        valid &= (clearance > -0.002) & (clearance < 0.075)
    elif phase == 3:
        valid &= (clearance >= 0.075) & (clearance < terms.LIFT_TARGET_CLEARANCE)
    if phase == 1:
        distance = (terms.cloth_grasp_point_w(env) - terms.grasp_center_w(env)).norm(dim=-1)
        valid &= (distance >= 0.008) & (distance <= 0.06)
    return valid


def _flat_proposal(env: ManagerBasedEnv, settings: GeneratorCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomize the flat sheet's XY position and yaw while preserving mesh distances."""
    cloth = env.scene["cloth"]
    state = cloth.data.default_nodal_state_w.torch.clone()
    centre = state[..., :3].mean(dim=1)
    yaw = (torch.rand(env.num_envs, device=env.device) * 2.0 - 1.0) * settings.yaw_jitter
    rotation = torch.zeros((env.num_envs, 4), device=env.device)
    rotation[:, 2], rotation[:, 3] = torch.sin(yaw / 2.0), torch.cos(yaw / 2.0)
    delta = state[..., :3] - centre[:, None]
    state[..., :3] = quat_apply(rotation[:, None].expand(-1, delta.shape[1], -1), delta) + centre[:, None]
    state[..., :2] += (torch.rand((env.num_envs, 1, 2), device=env.device) * 2.0 - 1.0) * settings.xy_jitter
    state[..., 3:] = 0.0
    cloth.write_nodal_state_to_sim_index(state)
    env.scene.update(env.physics_dt)
    down = rotation.new_tensor((1.0, 0.0, 0.0, 0.0)).expand_as(rotation)
    return state, quat_mul(rotation, down)


def _construct(env: ManagerBasedEnv, settings: GeneratorCfg, phase: int) -> torch.Tensor:
    """Construct one batched phase proposal, then let normal physics reject bad candidates."""
    _free_cloth(env)
    env.reset()
    flat, down = _flat_proposal(env, settings)
    corner = terms.cloth_grasp_point_w(env).clone()
    if phase == 0:
        target = corner.clone()
        target[:, 0] += torch.rand(env.num_envs, device=env.device) * 0.08
        target[:, 1] += (torch.rand(env.num_envs, device=env.device) - 0.5) * 0.10
        target[:, 2] += 0.12 + torch.rand(env.num_envs, device=env.device) * 0.15
        valid = _solve_pose(env, target, down, 0.04, settings.ik_iterations)
    elif phase == 1:
        target = corner.clone()
        lower, upper = settings.near_corner_distance
        target[:, 2] += lower + torch.rand(env.num_envs, device=env.device) * (upper - lower)
        target[:, :2] += (torch.rand((env.num_envs, 2), device=env.device) - 0.5) * 0.012
        valid = _solve_pose(env, target, down, 0.04, settings.ik_iterations)
    else:
        # Temporary full-sheet support gives a deterministic flat lift proposal.
        # Its flags are cleared before validation, and are never serialized.
        cloth = env.scene["cloth"]
        support = flat.clone()
        target = corner.clone()
        # Seat the corner across the upper inner pad edges. Its finite
        # collision thickness bridges the small closed aperture, avoiding
        # deep particle/pad overlap that disappears on a cold replay.
        target[:, 2] -= 0.014 + settings.table_penetration
        valid = _solve_pose(env, target, down, 0.003, settings.ik_iterations)
        heights = torch.empty(env.num_envs, device=env.device).uniform_(*settings.flat_lift_height)
        start_z = target[:, 2].clone()
        for step in range(1, settings.lift_steps + 1):
            fraction = step / settings.lift_steps
            position = target.clone()
            position[:, 2] = start_z + fraction * (heights - start_z)
            support[..., 2] = flat[..., 2] + fraction * (heights - start_z)[:, None]
            cloth.write_nodal_state_to_sim_index(support)
            _solve_pose(env, position, down, 0.003, max(16, settings.ik_iterations // settings.lift_steps))
        # Close at the final flat height before releasing support, so the sheet
        # cannot fall through an aperture that is still narrowing.
        # Re-seed the final air pose from the collision-free home branch;
        # near-table IK branches can fold the elbow into the robot when raised.
        robot = env.scene["robot"]
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(robot.data.default_joint_pos.torch))
        env.sim.forward()
        env.scene.update(env.physics_dt)
        valid = _solve_pose(env, position, down, 0.003, settings.ik_iterations)
        # Pre-drape the lifted sheet about its held edge, preserving every
        # edge length. This removes the large free-fall transient of a flat
        # airborne sheet before normal-physics validation.
        yaw = quat_mul(down, down.new_tensor((1.0, 0.0, 0.0, 0.0)).expand_as(down))
        local = quat_apply_inverse(yaw[:, None].expand(-1, flat.shape[1], -1), flat[..., :3] - corner[:, None])
        local[..., 2] = -local[..., 1]
        local[..., 1] = 0.0
        lifted_corner = corner.clone()
        lifted_corner[:, 2] += heights - start_z
        support[..., :3] = quat_apply(yaw[:, None].expand(-1, flat.shape[1], -1), local) + lifted_corner[:, None]
        cloth.write_nodal_state_to_sim_index(support)
        anchor = tuple(value.clone() for value in _tcp_pose(env))
        for _ in range(settings.close_steps):
            cloth.write_nodal_state_to_sim_index(support)
            _hold(env, True, 1, anchor)
        _free_cloth(env)
        _hold(env, True, settings.validation_steps)
        if phase == 3:
            # Translate the settled, deformed sheet and robot together toward a
            # near-goal clearance. This is another independent reset proposal.
            nodes = cloth.data.nodal_state_w.torch.clone()
            lower, upper = settings.near_goal_clearance
            desired = torch.empty(env.num_envs, device=env.device).uniform_(lower, upper)
            shift = desired - terms.cloth_clearance(env)
            nodes[..., 2] += shift[:, None]
            position, orientation = _tcp_pose(env)
            position = position.clone()
            position[:, 2] += shift
            cloth.write_nodal_state_to_sim_index(nodes)
            valid &= _solve_pose(env, position, orientation, 0.003, settings.ik_iterations)
    valid &= _validate(env, phase, settings.validation_steps)
    return valid


def generate(env: ManagerBasedEnv, settings: GeneratorCfg, phases: Sequence[int]) -> tuple[dict, dict]:
    """Fill phase quotas with validated, independently replayable snapshots."""
    records = []
    statistics = Counter()
    for phase in phases:
        accepted = 0
        for _ in range(settings.max_batches):
            valid = _construct(env, settings, phase)
            statistics[f"{PHASE_NAMES[phase]}/proposed"] += env.num_envs
            rows = valid.nonzero().flatten()[: settings.poses_per_phase - accepted]
            if len(rows):
                snapshot = capture_state(env, phase)
                # Static reset proposals should not inherit a swinging sheet's
                # transient velocities. Validate these edited candidates afresh.
                snapshot["joint_vel"].zero_()
                snapshot["cloth_nodal_state"][..., 3:].zero_()
                env.reset()
                # Replaying the saved state also clears solver history. Validate
                # again to reject states dependent on generator warm starts.
                apply_states(env, snapshot, rows, rows)
                env.scene.write_data_to_sim()
                env.sim.forward()
                env.scene.update(env.physics_dt)
                replay_valid = _validate(env, phase, settings.validation_steps)
                rows = rows[replay_valid[rows]]
                if len(rows):
                    records.append({key: value[rows].cpu() for key, value in snapshot.items()})
                    accepted += len(rows)
            statistics[f"{PHASE_NAMES[phase]}/accepted"] = accepted
            print(
                f"{PHASE_NAMES[phase]}: {accepted}/{settings.poses_per_phase} accepted; "
                f"{statistics[f'{PHASE_NAMES[phase]}/proposed']} proposed",
                flush=True,
            )
            if accepted >= settings.poses_per_phase:
                break
        if accepted < settings.poses_per_phase:
            raise RuntimeError(
                f"Could not fill {PHASE_NAMES[phase]} quota: {dict(statistics)}. "
                "No dataset was replaced. Adjust proposal ranges/contact settings or increase max_batches."
            )
    return {key: torch.cat([record[key] for record in records]) for key in records[0]}, dict(statistics)


def main(argv: list[str] | None = None) -> int:
    """Generate an atomic archive, or visually replay existing rows under normal physics."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("logs/cloth_resets.npz"))
    parser.add_argument("--poses_per_phase", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=64)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--table_penetration", type=float, default=0.0, help="Offline IK allowance [m], generation only."
    )
    parser.add_argument("--phases", nargs="+", choices=PHASE_NAMES, default=list(PHASE_NAMES))
    parser.add_argument("--view", type=Path, help="Replay this archive instead of generating.")
    parser.add_argument("--steps_per_state", type=int, default=30)
    add_launcher_args(parser)
    args = parser.parse_args(argv)
    settings = GeneratorCfg(
        poses_per_phase=args.poses_per_phase,
        max_batches=args.max_batches,
        seed=args.seed,
        table_penetration=args.table_penetration,
    )
    if args.num_envs <= 0 or args.steps_per_state <= 0:
        parser.error("num_envs and steps_per_state must be positive")
    cfg = ClothEnvCFG()
    cfg.scene.num_envs = 1 if args.view else args.num_envs
    cfg.seed = args.seed
    if not args.view:
        cfg.actions.arm.workspace_lower = (0.10, -0.35, terms.MAT_TOP_HEIGHT - settings.table_penetration)
    with launch_simulation(cfg, args) as physics:
        cfg.sim.physics = physics
        env = ManagerBasedEnv(cfg)
        try:
            env.reset()
            if args.view:
                states = load_dataset(args.view, env)
                for row in range(len(states["phase"])):
                    env.reset()
                    apply_states(
                        env, states, torch.tensor([row], device=env.device), torch.tensor([0], device=env.device)
                    )
                    env.scene.write_data_to_sim()
                    env.sim.forward()
                    env.scene.update(env.physics_dt)
                    phase = int(states["phase"][row])
                    print(f"Viewing row {row}: {PHASE_NAMES[phase]}", flush=True)
                    _hold(env, bool(states["gripper_command"][row, 0] < 0), args.steps_per_state)
            else:
                states, statistics = generate(env, settings, [PHASE_NAMES.index(name) for name in args.phases])
                save_dataset(
                    args.output,
                    env,
                    states,
                    {
                        "paper": "https://arxiv.org/html/2603.15789v3#S3.SS2",
                        "generator": asdict(settings),
                        "validation": statistics,
                        "physics": cfg.sim.physics.to_dict(),
                        "physics_dt": env.physics_dt,
                        "cloth_material": cfg.scene.cloth.spawn.physics_material.to_dict(),
                        "support_removed_before_validation": True,
                    },
                )
                print(f"Saved {len(states['phase'])} states to {args.output.resolve()}", flush=True)
        finally:
            _free_cloth(env)
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
