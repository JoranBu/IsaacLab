# Franka robot configuration

This task currently keeps the vial and rack from the SO-101 tutorial. Its robot
and live policy inputs use Franka Panda; adding cloth is a separate change.

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
- `ActionsCfg` takes eight actions: absolute fingertip position XYZ in the
  robot base frame, orientation quaternion XYZW, and a gripper command.
  A negative gripper command closes; zero or positive opens. An all-zero action
  is not a hold command for absolute-pose IK. To hold, supply the current
  fingertip pose and a valid unit quaternion.
- The flat Panda asset drives finger 1 and uses a mimic constraint for finger 2.
  Both finger states are observed, but the gripper action commands finger 1.
  Arm positions are radians; finger positions are metres.
- `DefaultEventsCfg.reset_scene` restores the asset's `init_state` and drive
  targets. The copied `reset_poses.npz` and `generator.py` are SO-101-specific
  and are not used by the Franka task. Port the generator and dataset schema
  before enabling dataset replay; six SO-101 coordinates cannot be mapped to
  nine Franka coordinates by renaming them.
- `terms.py` uses Panda finger bodies and a 0.046 m local-Z fingertip offset.
  Both finger contact sensors filter contacts against the vial.

The observation shape and action meaning changed, so train a new policy rather
than resuming an SO-101 checkpoint. `agent_cfg.py` uses `cloth_franka_state` as
the experiment name to keep new runs separate from the original tutorial.

## Sources in this checkout

- [Franka asset configuration](../../../source/isaaclab_assets/isaaclab_assets/robots/franka.py):
  initial joint positions, actuators, finger mimic behavior, and USD variants.
  This Newton task selects the `mujoco` physics variant.
- [SceneEntityCfg](../../../source/isaaclab/isaaclab/managers/scene_entity_cfg.py):
  joint/body selection and `preserve_order`.
- [Franka absolute IK example](../../../source/isaaclab_tasks/isaaclab_tasks/contrib/lift/config/franka/ik_abs_env_cfg.py):
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
reachability or successful vial placement.
