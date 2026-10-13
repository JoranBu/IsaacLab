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

The generator creates reaching, near-corner, table-supported grasp, airborne
grasp, and near-goal snapshots within the task's four reset banks. Held states
start with a free sheet on the table: the robot approaches with open jaws,
closes around a small patch near the chosen corner, and lifts through normal
policy actions. Contact may bunch several vertices or fold the corner.

Only the initial flat placement and non-contact hover pose use direct state
writes. Once the approach starts, no cloth or joint state is prescribed.
Accepted snapshots must also survive reset and unassisted replay. The grasped
bank reserves a quota for states with part of the sheet still on the table.

``--table_penetration`` relaxes the generation controller's minimum Z goal,
while keeping table collision enabled. Saved states must lie in the normal
workspace and all cloth vertices remain dynamic during training and replay.
"""

from __future__ import annotations

import argparse
import math
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
    # Pinch about one mesh spacing inward: closing can fold a small corner
    # patch into the jaws instead of requiring one perfectly centred vertex.
    grasp_inset: tuple[float, float] = (0.030, 0.033)
    pregrasp_height: tuple[float, float] = (0.0002, 0.0012)
    hover_height: float = 0.06
    approach_steps: int = 60
    approach_settle_steps: int = 90
    lift_speed: float = 0.03  # Vertical command speed [m/s].
    max_lift_steps: int = 600
    partial_lift_height: tuple[float, float] = (0.08, 0.14)
    partial_grasp_fraction: float = 0.5
    table_penetration: float = 0.0
    ik_iterations: int = 160
    close_steps: int = 12
    settle_steps: int = 60  # Hold after pickup so the free sheet can settle.
    validation_steps: int = 30  # 1 s at this task's 30 Hz control rate.
    # Leave settling margin inside the accepted 75–100 mm near-goal region.
    near_goal_clearance: tuple[float, float] = (0.082, 0.092)
    grasped_clearance: tuple[float, float] = (0.025, 0.055)

    def __post_init__(self):
        for name in (
            "poses_per_phase",
            "max_batches",
            "ik_iterations",
            "close_steps",
            "approach_steps",
            "approach_settle_steps",
            "max_lift_steps",
            "settle_steps",
            "validation_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 <= self.table_penetration <= 0.01:
            raise ValueError("table_penetration must be between 0 and 0.01 m.")
        for name in ("xy_jitter", "yaw_jitter", "hover_height", "lift_speed"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name in (
            "near_corner_distance",
            "grasp_inset",
            "pregrasp_height",
            "partial_lift_height",
            "grasped_clearance",
            "near_goal_clearance",
        ):
            lower, upper = getattr(self, name)
            if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 < lower < upper):
                raise ValueError(f"{name} must contain two positive, finite, increasing distances.")
        if (
            not self.grasped_clearance[1]
            < self.near_goal_clearance[0]
            < self.near_goal_clearance[1]
            < terms.LIFT_TARGET_CLEARANCE
        ):
            raise ValueError("Grasped and near-goal clearances must be disjoint and below the success threshold.")
        if self.hover_height <= 0.0 or self.lift_speed <= 0.0:
            raise ValueError("hover_height and lift_speed must be positive.")
        if not 0.0 <= self.partial_grasp_fraction <= 1.0:
            raise ValueError("partial_grasp_fraction must be between zero and one.")
        if self.validation_steps < 10:
            raise ValueError("validation_steps must allow at least ten consecutive control steps.")


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


# to be understood
def _free_cloth(env: ManagerBasedEnv) -> None:
    """Make every vertex dynamic before settling, validation, and replay.

    Nodal kinematic targets have four fields: XYZ and a freedom flag. A flag
    of zero constrains that vertex to XYZ; one lets physics move it normally.
    Writing ones here clears any constraints without moving the vertices.
    No constraints are used during the table pickup.
    """
    positions = env.scene["cloth"].data.nodal_pos_w.torch
    targets = torch.cat((positions, torch.ones_like(positions[..., :1])), dim=-1)
    env.scene["cloth"].write_nodal_kinematic_target_to_sim_index(targets)
    env.scene.write_data_to_sim()


# Tolerance around vertex-centre height on the mat, including collision radius.
_TABLE_TOLERANCE = 0.002
_MIN_TABLE_FRACTION = 0.10


def _table_fraction(env: ManagerBasedEnv) -> torch.Tensor:
    """Fraction of vertices resting within 2 mm of the tabletop support height."""
    nodes = env.scene["cloth"].data.nodal_pos_w.torch
    clearance = nodes[..., 2] - env.scene.env_origins[:, None, 2]
    clearance -= terms.MAT_TOP_HEIGHT + terms.CLOTH_PARTICLE_RADIUS
    return (clearance.abs() <= _TABLE_TOLERANCE).float().mean(dim=-1)


def _validate(env: ManagerBasedEnv, phase: int, steps: int, table_supported: bool | None = None) -> torch.Tensor:
    """Prove a free-cloth reset under normal physics, including any requested table support.

    With no explicit support requirement, a grasped row retains the support
    classification it has at the start. This also verifies archived rows in the
    viewer without adding a new phase or changing the reset archive schema.
    """
    closed = phase >= 2
    valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    _free_cloth(env)
    anchor = tuple(value.clone() for value in _tcp_pose(env))
    robot = env.scene["robot"]
    position_b, _ = subtract_frame_transforms(
        robot.data.root_link_pos_w.torch, robot.data.root_link_quat_w.torch, anchor[0]
    )
    lower = position_b.new_tensor(env.cfg.actions.arm.workspace_lower)
    lower[2] = max(float(lower[2]), terms.MAT_TOP_HEIGHT)
    upper = position_b.new_tensor(env.cfg.actions.arm.workspace_upper)
    valid &= ((position_b >= lower) & (position_b <= upper)).all(dim=-1)
    supported = _table_fraction(env) >= _MIN_TABLE_FRACTION
    if table_supported is not None:
        supported.fill_(table_supported)
    # Track a material point in the gripped patch. The outer corner itself
    # may hang outside the jaws when several nearby vertices are bunched up.
    nodes = env.scene["cloth"].data.nodal_pos_w.torch
    closest = (nodes - terms.grasp_center_w(env)[:, None]).square().sum(dim=-1).argmin(dim=-1)
    ids = torch.arange(env.num_envs, device=env.device)
    initial_patch = nodes[ids, closest].clone()
    for step in range(steps):
        _hold(env, closed, 1, anchor)
        robot, cloth = env.scene["robot"], env.scene["cloth"]
        valid &= torch.isfinite(cloth.data.nodal_state_w.torch).all(dim=(1, 2))
        valid &= torch.isfinite(robot.data.joint_pos.torch).all(dim=-1)
        valid &= ~terms.unstable_robot(env) & ~terms.cloth_lost(env)
        if closed:
            point = cloth.data.nodal_pos_w.torch[ids, closest]
            valid &= (point - initial_patch).norm(dim=-1) < 0.02
            if step >= steps // 2:
                _, _, pinched, patch_clearance = terms._pinch_state(env)
                valid &= pinched & (patch_clearance > terms.GRASP_PROOF_LIFT)
                if phase == 2:
                    valid &= ~supported | (_table_fraction(env) >= _MIN_TABLE_FRACTION)
        else:
            valid &= terms.cloth_clearance(env).abs() < 0.01
    clearance = terms.cloth_clearance(env)
    if phase == 2:
        valid &= (clearance > -0.002) & (clearance < 0.075)
        valid &= supported | (clearance > _TABLE_TOLERANCE)
    elif phase == 3:
        valid &= (clearance >= 0.075) & (clearance < terms.LIFT_TARGET_CLEARANCE)
    elif phase == 1:
        distance = (terms.cloth_grasp_point_w(env) - terms.grasp_center_w(env)).norm(dim=-1)
        valid &= (distance >= 0.008) & (distance <= 0.06)
    return valid


def _move_pose(
    env: ManagerBasedEnv, position: torch.Tensor, quaternion: torch.Tensor, closed: bool, steps: int
) -> None:
    """Interpolate a commanded pose through real simulation; never write joint or cloth state."""
    start = _tcp_pose(env)[0].clone()
    for step in range(steps):
        goal = start + (position - start) * ((step + 1) / steps)
        _hold(env, closed, 1, (goal, quaternion))


def _lift_to_clearance(env: ManagerBasedEnv, desired: torch.Tensor, settings: GeneratorCfg) -> torch.Tensor:
    """Raise a physically held sheet until its lowest vertex reaches the requested clearance."""
    position, orientation = (value.clone() for value in _tcp_pose(env))
    valid = terms.bilateral_contact(env)
    stable = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    for _ in range(settings.max_lift_steps):
        clearance = terms.cloth_clearance(env)
        increment = (desired - clearance).clamp(-settings.lift_speed * env.step_dt, settings.lift_speed * env.step_dt)
        position[:, 2] += torch.where(valid, increment, torch.zeros_like(increment))
        # Stay inside normal commanded workspace even if an unsuccessful
        # candidate loses its cloth or cannot reach the requested clearance.
        position[:, 2] = torch.minimum(
            position[:, 2], env.scene.env_origins[:, 2] + env.cfg.actions.arm.workspace_upper[2]
        )
        _hold(env, True, 1, (position, orientation))
        valid &= terms.bilateral_contact(env) & ~terms.unstable_robot(env) & ~terms.cloth_lost(env)
        at_goal = (terms.cloth_clearance(env) - desired).abs() < 0.002
        stable = torch.where(valid & at_goal, stable + 1, torch.zeros_like(stable))
        if ((stable >= 10) | ~valid).all():
            break
    return valid & (stable >= 10)


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


def _construct(env: ManagerBasedEnv, settings: GeneratorCfg, phase: int, table_supported: bool = False) -> torch.Tensor:
    """Construct one batched phase proposal, then let normal physics reject bad candidates."""
    _free_cloth(env)
    env.reset()
    _, down = _flat_proposal(env, settings)
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
        # Only initialize a non-contact hover pose offline. From the first
        # approach step onwards the mesh and robot advance entirely in physics.
        inset = corner.new_zeros(corner.shape)
        inset[:, :2].uniform_(*settings.grasp_inset)
        inset[:, :2] *= -1.0
        target = corner + quat_apply(down, inset)
        target[:, 2] += torch.empty(env.num_envs, device=env.device).uniform_(*settings.pregrasp_height)
        target[:, 2] -= settings.table_penetration
        hover = target.clone()
        hover[:, 2] += settings.hover_height
        valid = _solve_pose(env, hover, down, 0.04, settings.ik_iterations)
        _move_pose(env, target, down, False, settings.approach_steps)
        # Let the actual hand catch up to the final downward goal before
        # closing. Otherwise a visually near grasp can still miss the sheet.
        _hold(env, False, settings.approach_settle_steps, (target, down))
        _hold(env, True, settings.close_steps)
        valid &= terms.bilateral_contact(env)
        position, orientation = (value.clone() for value in _tcp_pose(env))
        position[:, 2] = env.scene.env_origins[:, 2] + torch.empty(env.num_envs, device=env.device).uniform_(
            *settings.partial_lift_height
        )
        distance = (position - _tcp_pose(env)[0]).norm(dim=-1).amax().item()
        steps = max(1, math.ceil(distance / (settings.lift_speed * env.step_dt)))
        _move_pose(env, position, orientation, True, steps)
        _hold(env, True, settings.settle_steps)
        valid &= terms.bilateral_contact(env)
        if not table_supported:
            lower, upper = settings.near_goal_clearance if phase == 3 else settings.grasped_clearance
            desired = torch.empty(env.num_envs, device=env.device).uniform_(lower, upper)
            valid &= _lift_to_clearance(env, desired, settings)
            _hold(env, True, settings.settle_steps)
    valid &= _validate(env, phase, settings.validation_steps, table_supported if phase == 2 else None)
    return valid


def generate(
    env: ManagerBasedEnv, settings: GeneratorCfg, phases: Sequence[int]
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Fill phase quotas with validated, independently replayable snapshots."""
    if not phases or len(set(phases)) != len(phases) or any(phase not in range(len(PHASE_NAMES)) for phase in phases):
        raise ValueError("Choose a nonempty sequence of distinct reset phase indices.")
    env.reset(seed=settings.seed)
    records = []
    statistics = Counter()
    jobs = []
    for phase in phases:
        partial_quota = math.ceil(settings.poses_per_phase * settings.partial_grasp_fraction) if phase == 2 else 0
        if partial_quota:
            jobs.append((phase, True, partial_quota))
        if partial_quota < settings.poses_per_phase:
            jobs.append((phase, False, settings.poses_per_phase - partial_quota))
    for phase, table_supported, quota in jobs:
        name = PHASE_NAMES[phase]
        region = f"{name}/{'table_supported' if table_supported else 'airborne'}" if phase == 2 else name
        accepted = 0
        for _ in range(settings.max_batches):
            valid = _construct(env, settings, phase, table_supported)
            statistics[f"{region}/proposed"] += env.num_envs
            statistics[f"{region}/constructed"] += int(valid.sum())
            rows = valid.nonzero().flatten()
            if len(rows):
                snapshot = capture_state(env, phase)
                env.reset()
                # Clear solver history, restore the exact candidate, and
                # require its supported/airborne category to survive replay.
                apply_states(env, snapshot, rows, rows)
                env.scene.write_data_to_sim()
                env.sim.forward()
                env.scene.update(env.physics_dt)
                replay_valid = _validate(env, phase, settings.validation_steps, table_supported if phase == 2 else None)
                statistics[f"{region}/replayed"] += int(replay_valid[rows].sum())
                rows = rows[replay_valid[rows]][: quota - accepted]
                if len(rows):
                    records.append({key: value[rows].cpu() for key, value in snapshot.items()})
                    accepted += len(rows)
                    statistics[f"{name}/accepted"] += len(rows)
            statistics[f"{region}/accepted"] = accepted
            print(
                f"{region}: {accepted}/{quota} accepted; "
                f"{statistics[f'{region}/proposed']} proposed, "
                f"{statistics[f'{region}/constructed']} constructed, "
                f"{statistics[f'{region}/replayed']} passed replay",
                flush=True,
            )
            if accepted >= quota:
                break
        if accepted < quota:
            raise RuntimeError(
                f"Could not fill {region} quota: {dict(statistics)}. "
                "No dataset was replaced. Adjust pickup ranges or increase max_batches."
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
    parser.add_argument("--settle_steps", type=int, default=60, help="Unassisted settling steps after pickup.")
    parser.add_argument("--validation_steps", type=int, default=30, help="Physics proof steps, repeated after reset.")
    parser.add_argument(
        "--table_penetration",
        type=float,
        default=0.0,
        help="Downward approach allowance [m], generation only; table collisions remain enabled.",
    )
    parser.add_argument(
        "--partial_grasp_fraction",
        type=float,
        default=0.5,
        help="Fraction of the grasped bank retaining table support (rounded up).",
    )
    parser.add_argument("--phases", nargs="+", choices=PHASE_NAMES, default=list(PHASE_NAMES))
    parser.add_argument("--view", type=Path, help="Replay this archive instead of generating.")
    parser.add_argument("--steps_per_state", type=int, default=30)
    add_launcher_args(parser)
    args = parser.parse_args(argv)
    try:
        settings = GeneratorCfg(
            poses_per_phase=args.poses_per_phase,
            max_batches=args.max_batches,
            seed=args.seed,
            table_penetration=args.table_penetration,
            settle_steps=args.settle_steps,
            validation_steps=args.validation_steps,
            partial_grasp_fraction=args.partial_grasp_fraction,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.num_envs <= 0 or args.steps_per_state <= 0:
        parser.error("num_envs and steps_per_state must be positive")
    if not args.view and args.output.suffix != ".npz":
        parser.error("output must use the .npz extension")
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
                    if not _validate(env, phase, settings.validation_steps).all():
                        raise RuntimeError(f"Archive row {row} failed normal-physics replay. Regenerate the dataset.")
                    print(
                        f"  Replay passed; whole-sheet clearance: {float(terms.cloth_clearance(env)[0]):.4f} m",
                        flush=True,
                    )
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
                        "physical_table_pickup": True,
                    },
                )
                print(f"Saved {len(states['phase'])} states to {args.output.resolve()}", flush=True)
        finally:
            _free_cloth(env)
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
