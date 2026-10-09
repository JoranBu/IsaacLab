# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Adapted from isaac-sim/IsaacLabTutorial (Apache-2.0); see ../LICENSE-IsaacLabTutorial.

"""Manager-based SO-101 vial placement task with physical reset replay."""

from __future__ import annotations

import math
from pathlib import Path

import newton
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonBuilderCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonManager,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysicsEvent
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.so101 import SO101_CFG

from . import resets, terms
from .actions import SoftLimitRelativeGripperActionCfg

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Objects/Vial_Rack"
RESET_DATASET = Path(__file__).with_name("reset_poses.npz")
CANONICAL_START = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
ALL_PHASES = (1.0,) * 8
TABLETOP_VIAL_POSITION = (0.231, -0.017, 0.06)
TABLETOP_VIAL_HEADING_RANGE = (-0.35, 0.35)
PREGRASP_GRIPPER_POSITION = math.radians(-10.0 + 1.1 * 22.4)
GRASP_GRIPPER_POSITION = math.radians(-10.0 + 1.1 * 1.0)
RELEASE_GRIPPER_POSITION = math.radians(-10.0 + 1.1 * 42.7)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINTS[:-1]
WORKSHOP_INITIAL_JOINT_POSITION = tuple(SO101_CFG.init_state.joint_pos[name] for name in JOINTS)


_CONTACT_STIFFNESS = 1.57e5
_CONTACT_DAMPING = 1.12e3
_FRICTION = 0.7
_ROLLING_FRICTION = 0.05
_TORSIONAL_FRICTION = 0.005
_SOLIMP = (0.7, 0.95, 0.0001, 0.5, 2.0)
_SOLREF = (0.002, 1.5)
_contact_model_registered = False


def _initialize_contacts(_event: PhysicsEvent) -> None:
    """Apply the workshop-validated contact model to every Newton shape."""
    sim = SimulationContext.instance()
    builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))

    num_shapes = len(builder.shape_body)
    for shape_index in range(num_shapes):
        builder.shape_material_ke[shape_index] = _CONTACT_STIFFNESS
        builder.shape_material_kd[shape_index] = _CONTACT_DAMPING
        builder.shape_material_mu[shape_index] = _FRICTION
        builder.shape_material_mu_rolling[shape_index] = _ROLLING_FRICTION
        builder.shape_material_mu_torsional[shape_index] = _TORSIONAL_FRICTION

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
        name="so101_workshop_contact_model",
    )
    _contact_model_registered = True


_register_contact_model()


@configclass
class SO101SceneCfg(InteractiveSceneCfg):
    """One SO-101, one vial, one rack, and a collision mat."""

    robot = SO101_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=SO101_CFG.spawn.replace(
            activate_contact_sensors=True,
        ),
        init_state=SO101_CFG.init_state.replace(
            pos=(-0.05, 0.0, 0.0),
            # Isaac Lab 3 uses XYZW quaternions: +90 degrees about world Z.
            rot=(0.0, 0.0, 0.7071068, 0.7071068),
        ),
        soft_joint_pos_limit_factor=0.98,
    )

    vial = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Vial",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ASSET_DIR}/vial.usda"),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=TABLETOP_VIAL_POSITION,
            # Horizontal vial: +90 degrees about world Y (XYZW).
            rot=(0.0, 0.7071068, 0.0, 0.7071068),
        ),
    )

    rack = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Rack",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ASSET_DIR}/rack.usda"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.18, 0.08, 0.04)),
    )

    mat = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Mat",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ASSET_DIR}/mat.usda"),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.22, 0.0, 0.032),
            rot=(0.0, 0.0, 0.7071068, 0.7071068),
        ),
    )

    fixed_jaw_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Vial"],
        history_length=4,
    )
    moving_jaw_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Vial"],
        history_length=4,
    )

    vial_rack_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Vial",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Rack"],
        history_length=4,
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1200.0, color=(0.9, 0.9, 0.9)),
    )


@configclass
class ActionsCfg:
    """Bounded relative joint targets matching the real SO-101 interface."""

    arm_action: RelativeJointPositionActionCfg = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        preserve_order=True,
        # Larger steps increased failures and rack forces in evaluation.
        scale=0.033,
        use_zero_offset=True,
    )
    gripper_action: SoftLimitRelativeGripperActionCfg = SoftLimitRelativeGripperActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        # Avoid opening a grasp rapidly from a small policy bias.
        scale=0.02,
        use_zero_offset=True,
    )


@configclass
class PolicyStateGroupCfg(ObsGroup):
    """Fully observed state actor inputs."""

    joint_pos = ObsTerm(func=terms.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS)})
    joint_vel = ObsTerm(func=terms.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS)})
    joint_target = ObsTerm(func=terms.joint_target)
    previous_action = ObsTerm(func=terms.last_action)
    end_effector = ObsTerm(func=terms.body_state, params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper")})
    vial = ObsTerm(func=terms.rigid_object_state, params={"asset_cfg": SceneEntityCfg("vial")})
    rack_target = ObsTerm(func=terms.rack_relative_target)
    placement = ObsTerm(func=terms.placement_features)
    # Latched milestones make the once-per-episode milestone rewards Markov.
    progress = ObsTerm(func=terms.progress_flags)

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
class DatasetEventsCfg:
    """Task-horizon resets plus modest physical domain randomization."""

    vial_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("vial"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.02),
            "num_buckets": 32,
        },
    )
    vial_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("vial"),
            # Newton cannot reliably infer mass from the detailed mesh.
            "mass_distribution_params": (0.015, 0.025),
            "operation": "abs",
        },
    )
    reset_from_dataset = EventTerm(
        func=resets.ResetFromDataset,
        mode="reset",
        params={"dataset_path": str(RESET_DATASET), "sequential": False, "phase_weights": ALL_PHASES},
    )


@configclass
class RewardsCfg:
    """Sparse physical milestones, a success bonus, two dense shaping terms, and light regularization."""

    approach_progress = RewTerm(func=terms.ApproachProgressReward, weight=1.0)
    held_goal = RewTerm(func=terms.held_goal_reward, weight=0.1)
    milestones = RewTerm(func=terms.PhysicalMilestoneReward, weight=10.0)
    success = RewTerm(func=terms.success_bonus, weight=200.0)
    vial_lost = RewTerm(func=terms.vial_lost, weight=-50.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002)


@configclass
class TerminationsCfg:
    success = DoneTerm(func=terms.PlacementHistoryTerm)
    vial_lost = DoneTerm(func=terms.vial_lost)
    unstable_robot = DoneTerm(func=terms.unstable_robot)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class PhysicsCfg(PresetCfg):
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=300,
            nconmax=200,
            cone="elliptic",
            impratio=10.0,
            update_data_interval=2,
            iterations=100,
            ls_iterations=15,
            use_mujoco_contacts=False,
            ccd_iterations=35,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(),
        num_substeps=2,
        debug_mode=False,
    )
    default = newton_mjwarp


@configclass
class SO101VialEnvCfg(ManagerBasedRLEnvCfg):
    """State task trained from physics-validated reset poses."""

    scene: SO101SceneCfg = SO101SceneCfg(num_envs=4096, env_spacing=0.9, replicate_physics=True)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: DatasetEventsCfg = DatasetEventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.is_finite_horizon = False
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.64, 0.0, 0.36), lookat=(0.19, 0.02, 0.075))

    def play_mode(self):
        """Play and evaluate complete episodes from the canonical home-pose starts, in dataset order."""
        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 16)
        self.events.reset_from_dataset.params["sequential"] = True
        self.events.reset_from_dataset.params["phase_weights"] = CANONICAL_START


@configclass
class GeneratorEventsCfg:
    """Give the generator a fixed 20 g vial; it initializes candidate poses explicitly."""

    vial_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("vial"),
            "mass_distribution_params": (0.02, 0.02),
            "operation": "abs",
        },
    )


@configclass
class SO101VialGeneratorEnvCfg(SO101VialEnvCfg):
    """Reuse the task scene and physics for directly commanded candidate rollouts."""

    events: GeneratorEventsCfg = GeneratorEventsCfg()
    rewards = None
    terminations = None
