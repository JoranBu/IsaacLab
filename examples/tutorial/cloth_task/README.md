# Franka cloth lifting

`Adapted_Cloth_Task` uses the same Franka, mat, bounded relative IK actions,
PPO settings, and approach → grasp → lift structure as `../vial_task/`.
The object is now a free 0.24 × 0.24 m deformable sheet. No vertices are attached
or teleported by the gripper during simulation.

## Task terms

- **Approach:** decrease fingertip-midpoint distance to a material vertex at the
  outer corner (maximum X, minimum Y). The corner is selected from the default mesh and tracked
  as it deforms. Signed distance changes remain unclipped, so a closed distance
  cycle has zero net undiscounted approach reward.
- **Grasp:** estimate a pinch when the same cloth vertex lies within 15 mm of
  both fingertip pad centres and the gripper is commanded closed. The patch
  must be more than 6 mm above the mat for two consecutive control steps to
  latch the grasp milestone. This is a geometric proxy, not a measured force:
  the coupled solver does not support the vial task's filtered contact sensors.
- **Dense lift:** reward mean sheet clearance during a lifted pinch, capped at
  the target height. This gives credit while part of the cloth still touches
  the mat. Opening the gripper suppresses this reward immediately.
- **Success:** the entire sheet's lowest vertex, minus particle radius, must
  clear the mat by 0.10 m while the pinch persists for ten consecutive control
  steps (about 0.33 s). Lifting only a corner does not complete the task.
- **Failure:** non-finite nodal state, vertices falling below the floor,
  unreachable sheet centre, or a return to the mat after the lift milestone.
  Extreme robot joint motion and the 20 s timeout retain their vial-task roles.

`terms.py` owns the mat height, particle radius, grasp thresholds, lift goal,
observations, rewards, and failure conditions. `geometry.py` owns the shared
vertex-proximity calculation. `progress.py` retains episode-local counters
and once-per-episode grasp/lift milestones, including partial resets.

## Observations and resets

The actor observes robot joint positions, velocities, targets, previous
commands, hand pose/velocity, nine cloth material landmark positions and
velocities in the robot base frame, the fingertip-to-corner vector, whole-sheet
clearance, gripper opening, grasp/hold counters, and milestone flags. The
critic also observes three jaw-proximity/pinch flags. The landmark indices
are selected once from a 3 × 3 grid on the undeformed mesh; this compact state
omits some deformation modes.

The interface is **7 actions, 110 actor observations, 113 critic observations**.
`mdp.reset_scene_to_default` restores the cloth's default nodal positions and
velocities alongside the robot home pose and actuator targets. Resetting a
subset of environments preserves the other meshes and episode progress.

## Physics and robot boundaries

`env_cfg.py` uses the coupled solver pattern from
[`thesis/cloth_lift_proxy.py`](../../../thesis/cloth_lift_proxy.py): MJWarp
advances the robot, VBD advances the cloth, and lagged robot collider proxies
provide cloth–robot contact. Cloth forces are not fed back as robot loads in
this proxy configuration. Cloth self-contact is enabled. The original mat
supports the sheet, and its initial vertex height includes the 3 mm particle
radius plus 1 mm of clearance.

The robot retains gravity compensation, convex-hull self-collision geometry,
95% joint soft limits, joint-limit avoidance, and the bounds in `actions.py`:

| Fingertip coordinate in robot base frame | Minimum (m) | Maximum (m) |
| --- | ---: | ---: |
| X | 0.10 | 0.65 |
| Y | -0.35 | 0.35 |
| Z | 0.035 | 0.65 |

Arm inputs in `[-1, 1]` map to 0.02 m translation and 0.1 rad rotation-vector
increments per component. The IK joint target offset is capped at
`4.0 * physics_dt` radians per physics step and clipped to soft limits. These
are command bounds; contact and inertia can still cause measured motion to
overshoot. The cloth configuration lowers the minimum Z goal from 0.04 m to
mat height and raises the target-offset cap from 2.0 to 4.0 rad/s: the scripted
probe otherwise stopped above the sheet or descended too slowly for the
20 s episode. Negative gripper input closes; zero or positive opens.

Tune cloth placement, mesh refinement, material parameters, and physics in
`env_cfg.py`. Keep `CLOTH_PARTICLE_RADIUS` consistent with clearance geometry.
A flat sheet is harder to pinch than the vial; a successful PPO run is not
established by launch or training smoke checks.

## Run and train

Run from `examples/tutorial` using `run.py` so the local task is registered:

```bash
uv run --no-sync python run.py zero_agent --task Adapted_Cloth_Task \
  --num_envs 2 --max_steps 100 --viz none

uv run --no-sync python run.py train --task Adapted_Cloth_Task \
  --num_envs 32 --max_iterations 2 --run_name smoke \
  agent.num_steps_per_env=8 agent.algorithm.num_mini_batches=2

uv run --no-sync python run.py train --task Adapted_Cloth_Task \
  --num_envs 64 --max_iterations 800 --device cuda:0

uv run --no-sync python run.py play --task Adapted_Cloth_Task \
  --checkpoint latest --num_envs 1 --viz viser
```

Start fresh: vial checkpoints have different observation dimensions.
`agent_cfg.py` stores new runs under `cloth_franka_deformable_lift`.
The default is 64 environments because the coupled deformable simulation
costs more memory and compute than the rigid vial task. Measure before scaling:

```bash
uv run --no-sync python run.py benchmark training --rl_library rsl_rl \
  --task Adapted_Cloth_Task --num_envs 64 --max_iterations 20 \
  --device cuda:0 --viz none --seed 42 --run_name benchmark \
  --output_path logs/benchmarks/cloth/training_64
```

From the repository root, run the local regression checks:

```bash
uv run --no-sync --with pytest python -m pytest examples/tutorial/test_cloth_task.py
uv run --no-sync --with pytest python -m pytest examples/tutorial/test_vial_task.py
```

## Source references

- [Cloth contact experiment](../../../thesis/cloth_lift.py): mesh, material, and VBD setup.
- [Coupler implementation](../../../source/isaaclab_contrib/isaaclab_contrib/coupling/coupler.py):
  solver ownership and contact-sensor limitations.
- [Default scene reset](../../../source/isaaclab/isaaclab/envs/mdp/events.py): nodal reset alongside articulations.
- [Franka asset](../../../source/isaaclab_assets/isaaclab_assets/robots/franka.py): joint ordering,
  actuators, mimic finger, and collider variants.
- [Manager-based RL environment guide](../../../docs/source/how-to/create_manager_rl_env.rst).

## Reset generator work in progress

`generator.py` and `resets.py` preserve the interrupted OmniReset-inspired work.
The proposed banks are reaching, near-corner, grasped, and near-goal. The generator
constructs flat-lift proposals, pre-drapes the sheet, removes its assistance, and
checks normal-physics replay before saving full nodal and robot states.

**Status:** grasped and near-goal banks have not passed replay validation; this
is unfinished work, not a ready-to-use reset dataset. No valid complete archive
has been produced. The cloth-specific contact stiffness/damping and 2 mm closed
finger target in `env_cfg.py` are experimental adjustments from that work.

The dataset reset event is disabled by default (`dataset_path=None`). The existing
home-pose reset remains active. A working archive can later be enabled through
`env.events.reset_from_dataset.params.dataset_path=/absolute/path/to/archive.npz`.

The script's options and generation-only table-penetration allowance are documented
in its module docstring and `--help`:

```bash
uv run --no-sync python -m examples.tutorial.cloth_task.generator --help
```
