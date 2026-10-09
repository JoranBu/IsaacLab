# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted Franka grasp attempt on a cloth lying flat on the ground.

Run from the IsaacLab repository root::

    uv run python thesis/cloth_lift.py --viz viser
    # Open http://localhost:8080. Stop the simulation with Ctrl+C.

    uv run python thesis/cloth_lift.py --viz none --num_steps 1200

Both the arm and the deformable cloth use Newton's VBD solver. The gripper
uses physical contact only: no cloth vertices are attached to the fingers.
A flat cloth is difficult to pinch against the floor, so this is a starting
point for grasp experiments, rather than a guaranteed successful grasp.

Distances are in metres, times in seconds, and quaternions are (x, y, z, w).
"""

from __future__ import annotations

import argparse
from functools import partial

import newton
import torch
from isaaclab_newton.physics import NewtonCfg, NewtonSoftContactCfg, VBDSolverCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg, NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.math import combine_frame_transforms, quat_slerp, subtract_frame_transforms
from isaaclab.visualizers import VisualizerCfg

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

# Change these values to adjust the layout without searching through the code.
CLOTH_CENTER = (0.50, 0.0, 0.004)
CLOTH_SIZE = (0.24, 0.24)
TCP_OFFSET = (0.0, 0.0, 0.107)  # From panda_hand to the point between the fingertips.
DOWN_QUATERNION = (1.0, 0.0, 0.0, 0.0)  # Rotate 180 degrees about X: hand +Z points down.


@configclass
class ClothLiftSceneCfg(InteractiveSceneCfg):
    """One fixed-base Franka, one free cloth, a floor, and a light."""

    # A static cuboid gives a visible floor with its upper surface at z = 0.
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.025)),
        spawn=sim_utils.CuboidCfg(
            size=(3.0, 3.0, 0.05),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.4),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.25, 0.28)),
        ),
    )

    robot: ArticulationCfg = replace(FRANKA_PANDA_HIGH_PD_CFG, prim_path="{ENV_REGEX_NS}/Robot")

    cloth: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cloth",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=CLOTH_CENTER),
        spawn=sim_utils.MeshRectangleCfg(
            size=CLOTH_SIZE,
            edge_refinement=16,  # Dense vertices help the fingers contact the thin sheet.
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.65, 0.12)),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=1.0,  # Surface mass density [kg/m^2].
                particle_radius=0.003,  # Finite collision thickness around each vertex.
                tri_ke=500.0,  # Resistance to in-plane stretching.
                tri_ka=0.5,  # Resistance to changes in triangle area. /used to be 500
                tri_kd=0.001,  # Damping of in-plane deformation.
                edge_ke=0.05,  # Bending resistance across triangle edges. smaller means more bendy /used to be 0.5
                edge_kd=0.001,
            ),
        ),
    )

    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))

    def __post_init__(self):
        # The supplied high-PD config disables robot gravity for easier IK tracking.
        # Both fingers have their own drive, so no mimic-joint constraint is needed.
        self.robot.spawn.articulation_props = NewtonArticulationCfg(self_collision_enabled=False)


@configclass
class ActionsCfg:
    """Absolute fingertip pose in the robot base frame, then an open/close command."""

    arm = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint[1-7]"],
        body_name="panda_hand",
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=TCP_OFFSET),
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.05}
        ),
    )
    gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.04},
        close_command_expr={"panda_finger_joint.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Keep joint observations available for later experiments."""

    @configclass
    class RobotCfg(ObservationGroupCfg):
        joint_positions = ObservationTermCfg(func=mdp.joint_pos)
        joint_velocities = ObservationTermCfg(func=mdp.joint_vel)

    robot: RobotCfg = RobotCfg()


@configclass
class ClothLiftEnvCfg(ManagerBasedEnvCfg):
    """Small manager-based environment with scripted control and no RL machinery."""

    scene: ClothLiftSceneCfg = ClothLiftSceneCfg(num_envs=1, env_spacing=2.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    decimation: int = 2  # Send a new scripted target at 60 Hz.
    seed: int = 0

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physics=NewtonCfg(
            # VBD advances rigid links and cloth in the same simulation. Contacts
            # use the cloth vertices, so the robot does not need extra SDF assets.
            solver_cfg=VBDSolverCfg(iterations=20, rigid_body_particle_contact_buffer_size=1024),
            num_substeps=4,
            soft_contact_cfg=NewtonSoftContactCfg(soft_contact_ke=8000.0, soft_contact_kd=0.01, soft_contact_mu=50.0),
        ),
        default_visualizer_cfg=VisualizerCfg(eye=(1.25, -1.1, 0.85), lookat=(0.4, 0.0, 0.15)),
    )


def smoothstep(progress: float) -> float:
    """Interpolate with zero velocity and acceleration at both ends."""
    progress = max(0.0, min(1.0, progress))
    return progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)


def fingertip_pose(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the measured fingertip position and XYZW orientation in the base frame."""
    robot = env.scene["robot"]
    hand_id = robot.find_bodies("panda_hand")[0][0]
    hand_pose = robot.data.body_pose_w.torch[:, hand_id]
    root_pose = robot.data.root_pose_w.torch
    hand_pos, hand_quat = subtract_frame_transforms(
        root_pose[:, :3], root_pose[:, 3:], hand_pose[:, :3], hand_pose[:, 3:]
    )
    offset = torch.tensor([TCP_OFFSET], device=env.device)
    return combine_frame_transforms(hand_pos, hand_quat, offset)


def update_joint_feedback(env: ManagerBasedEnv) -> None:
    """Recover joint angles and velocities from the links moved by VBD."""
    physics = env.sim.physics_manager
    state = physics.get_state_0()
    # VBD integrates body poses, leaving the joint-coordinate arrays unchanged.
    # IK adds its correction to the measured joint angles, so stale angles make
    # the arm stop well above the goal. This reads the bodies without moving them.
    newton.eval_ik(physics.get_model(), state, state.joint_q, state.joint_qd)


def run_trajectory(env: ManagerBasedEnv, args: argparse.Namespace) -> None:
    """Approach an edge, descend, close, lift, then hold until the user stops."""
    env.reset()
    start_position, start_orientation = fingertip_pose(env)
    start_position = start_position.clone()
    start_orientation = start_orientation.clone()

    # Aim just inside the near edge. One open finger is outside the cloth, which
    # gives the other finger a chance to gather the edge during closure.
    #I multiply by 0.9 so I don't grab next to the cloth
    grasp = torch.tensor([[CLOTH_CENTER[0]-0.8*(CLOTH_SIZE[0]/2), -CLOTH_SIZE[1] / 2 , args.grasp_height]], device=env.device)
    above = grasp.clone()
    above[:, 2] = 0.18
    lifted = grasp.clone()
    lifted[:, 2] += args.lift_height
    down = torch.tensor([DOWN_QUATERNION], device=env.device)

    # Each row describes one segment: name, duration, start, end, finger command.
    # +1 opens both fingers; -1 closes them. The last pose is held indefinitely.
    phases = [
        ("settle", 1.0, start_position, start_position, 1.0),
        ("approach edge", 3.0, start_position, above, 1.0),
        ("descend", 3.0, above, grasp, 1.0),
        ("close gripper", 2.0, grasp, grasp, -1.0),
        ("lift", 4.0, grasp, lifted, -1.0),
        ("hold", float("inf"), lifted, lifted, -1.0),
    ]
    action = torch.zeros((1, 8), device=env.device)  # XYZ + XYZW + gripper.
    phase_index = 0
    phase_time = 0.0
    step = 0
    print("[cloth_lift] Newton/VBD: physical grasp attempt; Ctrl+C to stop.", flush=True)
    print(f"[cloth_lift] Phase: {phases[0][0]}", flush=True)

    with torch.inference_mode():
        while env.sim.is_running() and (args.num_steps == 0 or step < args.num_steps):
            name, duration, begin, end, fingers = phases[phase_index]
            blend = smoothstep(phase_time / duration)
            action[:, :3] = torch.lerp(begin, end, blend)
            if phase_index == 0:
                action[:, 3:7] = start_orientation
            elif phase_index == 1:
                action[:, 3:7] = quat_slerp(start_orientation[0], down[0], blend)
            else:
                action[:, 3:7] = down
            action[:, 7] = fingers
            env.step(action)
            step += 1
            phase_time += env.step_dt

            if phase_time >= duration:
                report_state(env, action[:, :3])
                phase_index += 1
                phase_time = 0.0
                print(f"[cloth_lift] Phase: {phases[phase_index][0]}", flush=True)

    report_state(env, action[:, :3])


def report_state(env: ManagerBasedEnv, target: torch.Tensor) -> None:
    """Print measured motion; cloth height alone does not prove a stable grasp."""
    positions = env.scene["cloth"].data.nodal_pos_w.torch
    joints = env.scene["robot"].data.joint_pos.torch
    if not torch.isfinite(positions).all() or not torch.isfinite(joints).all():
        raise RuntimeError("Non-finite physics state detected; stop and inspect the solver settings.")
    actual_position, _ = fingertip_pose(env)
    error = torch.linalg.vector_norm(actual_position - target).item()
    print(
        f"[cloth_lift] Tip height: {actual_position[0, 2].item():.4f} m; tip error: {error:.4f} m; "
        f"cloth height min/mean/max: {positions[..., 2].min().item():.4f} / "
        f"{positions[..., 2].mean().item():.4f} / {positions[..., 2].max().item():.4f} m",
        flush=True,
    )


def main() -> None:
    """Parse options, start the simulator, and release resources on exit."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num_steps", type=int, default=0, help="Control steps to run; 0 holds until Ctrl+C.")
    parser.add_argument("--grasp_height", type=float, default=0.007, help="Fingertip target above the ground [m].")
    parser.add_argument("--lift_height", type=float, default=0.15, help="Vertical motion after closing [m].")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["viser"])
    args = parser.parse_args()
    if args.num_steps < 0 or args.grasp_height < 0.0 or args.lift_height < 0.0:
        parser.error("Step count and heights must be nonnegative.")

    cfg = ClothLiftEnvCfg()
    cfg.sim.device = args.device
    with launch_simulation(cfg, args):
        env = ManagerBasedEnv(cfg=cfg)
        # Register before the first step so CUDA graph capture includes the
        # feedback update and the IK controller sees live joints every tick.
        joint_feedback = partial(update_joint_feedback, env)
        env.sim.physics_manager.register_post_step_callback(joint_feedback)
        try:
            run_trajectory(env, args)
        except KeyboardInterrupt:
            print("\n[cloth_lift] Stopped.")
        finally:
            env.sim.physics_manager.unregister_post_step_callback(joint_feedback)
            env.close()


if __name__ == "__main__":
    main()
