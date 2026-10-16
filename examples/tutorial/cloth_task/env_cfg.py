# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Adapted from isaac-sim/IsaacLabTutorial (Apache-2.0); see ../LICENSE-IsaacLabTutorial.

"""Manager-based Franka cloth-lifting task."""

from __future__ import annotations

import newton
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonBuilderCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonManager,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg, NewtonArticulationCfg
from isaaclab_newton.sim.schemas.schemas_cfg import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials.physics_materials_cfg import NewtonSurfaceDeformableBodyMaterialCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.paths import ISAACLAB_ROOT
from isaaclab.physics import PhysicsEvent
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, replace
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.visualizers import VisualizerCfg

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

from . import terms
from .actions import BoundedDifferentialIKActionCfg
from .resets import ResetFromDataset

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Objects/Vial_Rack"  # Tabletop assets shared with the vial task.
TABLETOP_CLOTH_POSITION = (0.231, -0.017, terms.MAT_TOP_HEIGHT + terms.CLOTH_PARTICLE_RADIUS + 0.001)
CLOTH_SIZE = (0.24, 0.24)
# Resolve from the checkout, so launching from examples/tutorial also works.
RESET_DATASET_PATH = str(ISAACLAB_ROOT / "logs" / "cloth_table_resets.npz")
ARM_JOINTS = [f"panda_joint{index}" for index in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
# Keep positions, velocities, and drive targets in the same explicit policy order.
JOINTS = ARM_JOINTS + FINGER_JOINTS


# VBD mixes particle-side and shape-side material values. The vial's
# 157 kN/m / 1120 Ns/m contact model makes thin-sheet pinches unstable.
_CONTACT_STIFFNESS = 8.0e3
_CONTACT_DAMPING = 0.01
_FRICTION = 0.7
_ROLLING_FRICTION = 0.05
_TORSIONAL_FRICTION = 0.005
_SOLIMP = (0.7, 0.95, 0.0001, 0.5, 2.0)
_SOLREF = (0.002, 1.5)
_contact_model_registered = False

TCP_OFFSET = (0.0, 0.0, 0.107)  # From panda_hand to the point between the fingertips.


def _initialize_contacts(_event: PhysicsEvent) -> None:
    """Apply cloth contact settings without changing the rigid-only vial scene."""
    sim = SimulationContext.instance()
    builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))

    # run.py registers both local tasks. Keep this callback harmless when the
    # selected scene is the rigid-only vial task.
    if not builder.particle_q:
        return
    num_shapes = len(builder.shape_body)
    for shape_index in range(num_shapes):
        builder.shape_material_ke[shape_index] = _CONTACT_STIFFNESS
        builder.shape_material_kd[shape_index] = _CONTACT_DAMPING
        builder.shape_material_mu[shape_index] = _FRICTION
        builder.shape_material_mu_rolling[shape_index] = _ROLLING_FRICTION
        builder.shape_material_mu_torsional[shape_index] = _TORSIONAL_FRICTION

    # These coarse primitive approximations overlap at the trained home pose.
    # Filter only this pair per robot; resolving it otherwise launches the arm
    # into the unstable-robot termination before the policy can approach the cloth.
    link1_boxes = {
        label.rsplit("/panda_link1/", 1)[0]: index
        for index, label in enumerate(builder.shape_label)
        if label.endswith("/link1_box")
    }
    for index, label in enumerate(builder.shape_label):
        if label.endswith("/link3_capsule"):
            link1_box = link1_boxes.get(label.rsplit("/panda_link1/", 1)[0])
            if link1_box is not None:
                builder.add_shape_collision_filter_pair(link1_box, index)

    # Prototype builders register these attributes, but Newton's cloner does
    # not currently carry that registration to the main builder.
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    for name, value in (("mujoco:geom_solimp", _SOLIMP), ("mujoco:geom_solref", _SOLREF)):
        attribute = builder.custom_attributes.get(name)
        if attribute is None:
            continue
        if attribute.values is None:
            attribute.values = {}
        for shape_index in range(num_shapes):
            attribute.values[shape_index] = value


def _register_contact_model() -> None:
    """Register the contact initializer once per process."""
    global _contact_model_registered
    if _contact_model_registered:
        return
    NewtonManager.register_callback(
        _initialize_contacts,
        PhysicsEvent.MODEL_INIT,
        name="cloth_task_contact_model",
    )
    _contact_model_registered = True


_register_contact_model()


@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    """One Franka, one cloth, and a collision mat."""

    robot: ArticulationCfg = replace(
        FRANKA_PANDA_HIGH_PD_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
        soft_joint_pos_limit_factor=0.95,
        spawn=replace(
            FRANKA_PANDA_HIGH_PD_CFG.spawn,
            # MJWarp needs its own gravity compensation for zero-action relative IK to hold the arm.
            rigid_props=[FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props, MujocoRigidBodyCfg(gravcomp=1.0)],
        ),
    )

    def __post_init__(self):
        # This task uses Newton/MJWarp; the shared asset defaults to the PhysX payload.
        self.robot.spawn.variants["Physics"] = "mujoco"
        # Analytic boxes/capsules support full-surface cloth contact without mesh SDF generation.
        self.robot.spawn.variants["Colliders"] = "primitives"
        self.robot.spawn.activate_contact_sensors = False
        # The flat asset's 350 N/m finger drive lets a 6 mm thick cloth
        # particle open the jaws until the pinch cannot support the sheet.
        self.robot.actuators["panda_hand"].stiffness = 2000.0
        self.robot.actuators["panda_hand"].damping = 100.0
        for properties in self.robot.spawn.articulation_props:
            if isinstance(properties, NewtonArticulationCfg):
                properties.self_collision_enabled = True
            elif isinstance(properties, PhysxArticulationCfg):
                properties.enabled_self_collisions = True

    cloth: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cloth",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=TABLETOP_CLOTH_POSITION),
        spawn=sim_utils.MeshRectangleCfg(
            size=CLOTH_SIZE,
            edge_refinement=8,  # Dense vertices help the fingers contact the thin sheet.
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.65, 0.12)),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=1.0,  # Surface mass density [kg/m^2].
                particle_radius=terms.CLOTH_PARTICLE_RADIUS,  # Finite collision thickness around each vertex.
                tri_ke=500.0,  # Resistance to in-plane stretching.
                tri_ka=0.5,  # Resistance to changes in triangle area.
                tri_kd=0.001,  # Damping of in-plane deformation.
                edge_ke=0.05,  # Bending resistance across triangle edges.
                edge_kd=0.001,
            ),
        ),
    )

    mat = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Mat",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ASSET_DIR}/mat.usda"),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.22, 0.0, 0.032),
            rot=(0.0, 0.0, 0.7071068, 0.7071068),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1200.0, color=(0.9, 0.9, 0.9)),
    )


@configclass
class ActionsCfg:
    """Small fingertip pose increments in the robot base frame, then an open/close command."""

    arm = BoundedDifferentialIKActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        body_name="panda_hand",
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=TCP_OFFSET),
        # Thin-sheet contact requires a lower fingertip goal than the vial's 4 cm boundary.
        workspace_lower=(0.10, -0.35, terms.MAT_TOP_HEIGHT),
        # Keep bounded commands while permitting the longer descent within a 20 s cloth episode.
        joint_target_speed_limit=4.0,
        # Normalized arm actions map to metres and rotation-vector radians per control step.
        scale=(0.02, 0.02, 0.02, 0.1, 0.1, 0.1),
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": 0.05},
            joint_limit_avoidance_gain=0.05,
            joint_limit_avoidance_margin=0.1,
        ),
    )
    gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        # The flat Franka asset drives finger 1; finger 2 follows its mimic constraint.
        joint_names=[FINGER_JOINTS[0]],
        open_command_expr={FINGER_JOINTS[0]: 0.04},
        # Leave a 4 mm jaw aperture for 6 mm particle collision thickness,
        # rather than crushing both pads through the same cloth vertices.
        close_command_expr={FINGER_JOINTS[0]: 0.002},
    )


@configclass
class PolicyStateGroupCfg(ObsGroup):
    """Robot state and a compact cloth-shape observation."""

    joint_pos = ObsTerm(
        func=terms.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS, preserve_order=True)}
    )
    joint_vel = ObsTerm(
        func=terms.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS, preserve_order=True)}
    )
    joint_target = ObsTerm(
        func=terms.joint_target, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS, preserve_order=True)}
    )
    previous_action = ObsTerm(func=terms.last_action)
    end_effector = ObsTerm(
        func=terms.body_state, params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_hand")}
    )
    cloth = ObsTerm(func=terms.ClothState)
    lift = ObsTerm(func=terms.lift_features)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CriticStateGroupCfg(PolicyStateGroupCfg):
    """Privileged training critic inputs."""

    contact = ObsTerm(func=terms.contact_state)


@configclass
class ObservationsCfg:
    policy: PolicyStateGroupCfg = PolicyStateGroupCfg()
    critic: CriticStateGroupCfg = CriticStateGroupCfg()


@configclass
class DefaultEventsCfg:
    """Restore the robot home pose, cloth mesh, velocities, and drive targets."""

    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    reset_from_dataset = EventTerm(
        func=ResetFromDataset,
        mode="reset",
        # The default reset runs first, then this restores a sampled snapshot.
        params={"dataset_path": RESET_DATASET_PATH, "phase_weights": None},
    )


@configclass
class RewardsCfg:
    """Approach and height shaping, elapsed-time cost, terminal rewards, and regularization."""

    approach_progress = RewTerm(func=terms.ApproachProgressReward, weight=0.5)
    lift = RewTerm(func=terms.lift_reward, weight=2)
    success = RewTerm(func=terms.success_bonus, weight=200.0)
    # The manager integrates this rate: -0.02 per second until termination, including timeouts.
    time_penalty = RewTerm(func=mdp.is_alive, weight=-0.02)
    action_rate = RewTerm(func=terms.action_rate_penalty, weight=-0.002)
    joint_velocity = RewTerm(func=terms.joint_velocity_penalty, weight=-0.0002)
    failure = RewTerm(func=terms.failure_penalty, weight=-0.5)


@configclass
class TerminationsCfg:
    success = DoneTerm(func=terms.LiftHistoryTerm)
    cloth_lost = DoneTerm(func=terms.cloth_lost)
    unstable_robot = DoneTerm(func=terms.unstable_robot)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class PhysicsCfg(PresetCfg):
    newton_mjwarp_vbd = NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="robot",
                    solver_cfg=MJWarpSolverCfg(
                        solver="newton",
                        integrator="implicitfast",
                        njmax=300,
                        nconmax=200,
                        cone="elliptic",
                        impratio=10.0,
                        iterations=50,  # used to be 100
                        ls_iterations=50,
                        use_mujoco_contacts=False,
                        ccd_iterations=35,
                    ),
                    bodies=[r"/World/envs/env_[^/]+/Robot"],
                    include_static_shapes=True,
                ),
                CouplerEntryCfg(
                    name="cloth",
                    solver_cfg=VBDSolverCfg(
                        iterations=15,  # used to be 20
                        # Reduce slow sliding through the pads under cloth load.
                        friction_epsilon=1.0e-4,
                        rigid_body_particle_contact_buffer_size=1024,
                        particle_enable_self_contact=True,
                        particle_self_contact_radius=terms.CLOTH_PARTICLE_RADIUS,
                        particle_self_contact_margin=2.0 * terms.CLOTH_PARTICLE_RADIUS,
                    ),
                    all_particles=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="robot",
                    destination="cloth",
                    bodies=[r"/World/envs/env_[^/]+/Robot"],
                    mode="lagged",
                    collide_interval=1,
                    collision_pipeline=NewtonCollisionPipelineCfg(
                        enable_rigid_soft_full_surface_contact=True,
                    ),
                ),
            ],
            iterations=1,
        ),
        soft_contact_cfg=NewtonSoftContactCfg(soft_contact_ke=8000.0, soft_contact_kd=0.01, soft_contact_mu=0.7),
        num_substeps=4,
        debug_mode=False,
    )
    default = newton_mjwarp_vbd


@configclass
class ClothEnvCFG(ManagerBasedRLEnvCfg):
    """Franka state task starting from the asset's configured home pose."""

    scene: FrankaSceneCfg = FrankaSceneCfg(num_envs=64, env_spacing=0.9, replicate_physics=True)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: DefaultEventsCfg = DefaultEventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0  # in seconds - used to be 20
        self.is_finite_horizon = False
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(1.5, 0.0, 0.8), lookat=(-0.2, 0.02, 0.075))

    def play_mode(self):
        """Play complete episodes from the Franka home pose."""
        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 16)
