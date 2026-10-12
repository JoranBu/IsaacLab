# Franka robot configuration

This task uses Franka Panda to grasp and lift the tutorial vial from a collision
mat. Adding cloth is a separate change.

## Lift objective

Success requires a live bilateral finger grasp with the vial's **lowest point
at least 0.10 m above the mat for ten consecutive control steps** (about 0.33 s).
The vial can have any orientation. `terms.py` computes the lowest point from
the vial's axis and cylinder dimensions, so rotating the vial cannot satisfy
the goal just by moving its root upward.

- Approach shaping rewards decreasing fingertip-to-vial grasp-point distance.
  Distance changes are not individually clipped: a retreat followed by a return
  to the starting distance has zero net undiscounted approach reward.
- Dense lift shaping rewards live bilateral contact in proportion to clearance,
  capped at the success height. A released or airborne vial gets no lift reward.
- Grasp and lift milestones pay once per episode; sustained lift gives the
  success bonus and terminates the episode.
- Invalid/unreachable vial states, a return to the mat after the lift milestone,
  extreme joint motion, and timeout remain termination conditions.
- The actor observes clearance, jaw opening, normalized grasp/hold counters,
  and two latched progress flags. The critic additionally observes finger contacts.

The old insertion/placement observations, rewards, metrics, reset generator,
and dataset were removed. Resets use the configured robot and vial defaults.
The original placement workflow is available in `../vial_task/`.

## Arm boundaries

`FrankaSceneCfg` enables robot self-collision and selects `convex_hulls` arm
colliders. The primitive collider variant overlaps at the default home pose
when self-collision is enabled. The joint soft-limit range is 95% of the
authored physical range; the IK controller also biases joints away from limits
within a 0.1 rad margin.

`BoundedDifferentialIKActionCfg` in `actions.py` projects fingertip position
goals into this robot-base-frame workspace:

| Axis | Minimum (m) | Maximum (m) |
| --- | ---: | ---: |
| X | 0.10 | 0.65 |
| Y | -0.35 | 0.35 |
| Z | 0.04 | 0.65 |

These bounds include the configured home pose and vial grasp point. Adjust
`workspace_lower` and `workspace_upper` when moving the robot, mat, or vial.
Inputs are sanitized and clipped to `[-1, 1]` before scaling. Every physics
step clips the IK joint targets to the soft limits and caps their offset from
measured positions to `2.0 * physics_dt` radians (about 0.0167 rad here).
This is a command bound; contact and inertia can still cause measured motion
to overshoot. Self-collision is handled by the contact solver using approximate
collision geometry, with neighboring-link exclusions retained by Newton.

The seven-action interface and observation dimensions stay the same. These
constraints change the dynamics, so assess any resumed policy with a short
playback before extending training. Hull contacts and joint-limit avoidance
add runtime work; rerun the throughput benchmark for this configuration.

## When changing robots

Changing `scene.robot` does not update joint selectors, body selectors, actions,
or reset states. Check each of them:

- `env_cfg.py`: `ARM_JOINTS`, `FINGER_JOINTS`, and `JOINTS` define the policy's
  observation order: `panda_joint1` through `panda_joint7`, then
  `panda_finger_joint1` and `panda_finger_joint2`. Positions, velocities, and
  targets all use `SceneEntityCfg(..., preserve_order=True)`.
- `PolicyStateGroupCfg` selects `panda_hand` for the end-effector observation.
  `ActionsCfg.arm` controls the fingertip frame 0.107 m along hand-local Z.
  These are different frames: the observation is the hand pose.
- `ActionsCfg` takes seven actions: three fingertip position increments,
  three rotation-vector increments, and a gripper command. The increments are
  expressed in the robot base frame. Arm actions clipped to `[-1, 1]` map to
  at most 0.02 m per translation axis and 0.1 rad per rotation-vector component
  per control step. With the configured initial exploration standard deviation
  of 0.2, that is 4 mm and 0.02 rad of noise per component.
  A negative gripper command closes; zero or positive opens. Zero arm actions
  hold the current fingertip pose inside the workspace; an out-of-bounds pose
  receives a correction toward the boundary. An all-zero action holds the arm and opens
  the gripper. The robot has MJWarp body gravity compensation (`gravcomp=1.0`)
  so repeatedly commanding its current pose does not let the arm sag. Scene
  gravity remains active for the vial.
- The flat Panda asset drives finger 1 and uses a mimic constraint for finger 2.
  Both finger states are observed, but the gripper action commands finger 1.
  Arm positions are radians; finger positions are metres.
- `DefaultEventsCfg.reset_scene` restores the asset's `init_state` and drive
  targets. Robot-specific reset data must match the robot's joint names, order,
  and task objective before it can replace the default reset.
- `terms.py` uses Panda finger bodies and a 0.046 m local-Z fingertip offset.
  Both finger contact sensors filter contacts against the vial.

Relative IK changes the action count from eight to seven and the previous-action
observation from eight values to seven. Train a fresh policy; absolute-pose
checkpoints cannot be resumed with this interface. `agent_cfg.py` uses
`cloth_franka_lift_relative_ik` to keep new runs separate from the earlier runs.

From `examples/tutorial`, check the training pipeline and then start training:

```bash
uv run --no-sync python run.py train --task Adapted_Cloth_Task \
  --num_envs 32 --max_iterations 2 --run_name smoke \
  agent.num_steps_per_env=8 agent.algorithm.num_mini_batches=2

uv run --no-sync python run.py train --task Adapted_Cloth_Task \
  --num_envs 4096 --max_iterations 800 --device cuda:0
```

## Sources in this checkout

- [Franka asset configuration](../../../source/isaaclab_assets/isaaclab_assets/robots/franka.py):
  initial joint positions, actuators, finger mimic behavior, and USD variants.
  This Newton task selects the `mujoco` physics variant.
- [SceneEntityCfg](../../../source/isaaclab/isaaclab/managers/scene_entity_cfg.py):
  joint/body selection and `preserve_order`.
- [Franka relative IK example](../../../source/isaaclab_tasks/isaaclab_tasks/contrib/lift/config/franka/ik_rel_env_cfg.py):
  arm joints, hand body, and fingertip offset.
- [Franka cabinet configuration](../../../source/isaaclab_tasks/isaaclab_tasks/core/cabinet/config/franka/joint_pos_env_cfg.py):
  finger body paths and fingertip offsets.
- [Articulation configuration guide](../../../docs/source/how-to/write_articulation_cfg.rst).
- [Manager-based RL environment guide](../../../docs/source/how-to/create_manager_rl_env.rst).

For a new asset, inspect `env.scene["robot"].joint_names` and `.body_names` after
creating a single environment. Resolve named selectors once through managers;
do not assume the backend's raw joint order equals your policy's desired order.

Run the local task with the tutorial launcher:

```bash
uv run --no-sync python examples/tutorial/run.py random_agent \
  --task Adapted_Cloth_Task --num_envs 1 --max_steps 10 --visualizer none
```

Random actions only check that the environment can step. They do not validate
reachability or successful lifting.
