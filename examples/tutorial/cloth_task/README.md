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
provide cloth–robot contact with lagged impulse feedback to the robot. Cloth
self-contact is enabled. The original mat
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

The finger drive uses an explicit stiffness of 2,000 N/m and damping of
100 Ns/m. Its 2 mm closed target leaves a nominal 4 mm aperture for the cloth’s
6 mm particle collision thickness. VBD uses `friction_epsilon=1e-4` m/s to
reduce slow sliding through the pads; the Newton default of `1e-2` m/s was too
large for a sustained cloth pinch. These settings apply during training too.

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

## OmniReset-inspired reset banks

[`generator.py`](generator.py) builds independent reset states for four regions,
inspired by [OmniReset, section 3.2](https://arxiv.org/html/2603.15789v3#S3.SS2).
This adapts its diverse-reset idea to this task's fixed cloth corner. The banks
contain physical snapshots rather than demonstration trajectories.

| Bank | Initial situation |
| --- | --- |
| `reaching` | Open jaws, roughly 12–27 cm above the selected corner |
| `near_corner` | Open jaws, 12–45 mm above the corner with small XY offsets |
| `grasped` | Closed jaws holding a folded corner patch; half the bank retains table support, half targets 25–55 mm whole-sheet clearance |
| `near_goal` | Closed jaws holding a corner patch; whole-sheet clearance sampled at 82–92 mm, below the 100 mm success threshold |

Every proposal randomizes the flat sheet's XY position and yaw. The same material
corner selected in `terms.py` defines the pickup region. The saved clearance can differ
slightly from its sampled value because the cloth continues settling.

For the held banks, the script starts with the free sheet on the table and
constructs an open-jaw hover pose with offline IK. It then descends through real
simulation, closes the jaws, and raises the arm using the normal policy action
interface. **Every cloth vertex stays dynamic throughout the pickup.** Direct
joint and mesh writes are limited to initial pose/flat-sheet initialization and
archive reset; the pickup never prescribes cloth motion or attaches vertices.

The grasp centre is sampled slightly inside the selected corner. Closing the
jaws can gather neighbouring vertices and produce a small fold. A grasp need
not contain only the exact corner vertex. The generator first raises the patch
a modest distance and lets it settle. For airborne and near-goal candidates,
it continues raising the arm while measuring the lowest cloth vertex's clearance.
It never translates the settled mesh to obtain the desired clearance.

By default, `--partial_grasp_fraction 0.5` reserves half the grasped bank for
**table-supported** states, rounded up. A supported candidate must have at least
10% of its vertices within 2 mm of the tabletop support height while a pinched
patch is more than 6 mm above the mat. Its support must persist through physics
validation and reset replay. These are geometric checks backed by simulated
stability, rather than measured table contact forces. The remaining grasped rows
must clear the table by more than 2 mm. Both categories use the existing `grasped`
phase label, so the archive format and training reset event remain compatible.
A quota of one produces one supported grasp; use two or more to obtain both kinds.

Each candidate must pass 30 control steps (1 s) with free vertices, normal
gravity, table collision, and ordinary pose-holding actions. Checks reject
non-finite/lost states, unstable robot motion, escaping material in the held
patch, lost pinches, and incorrect final clearance. Validation tracks a material
vertex near the jaws, allowing the ungripped outer corner to move as the sheet
folds. The generator then resets the environment, restores the exact snapshot,
and repeats the proof, including its table-supported/airborne category. Only
passing rows are saved. Failed quotas leave an existing archive intact.

The generator uses `ManagerBasedEnv`, so success/failure rewards cannot reset a
proposal during pickup. Training uses the usual RL environment.
`--table_penetration` permits up to 10 mm of extra downward commanded reach
**only during generation**, while table collision stays enabled. The default is
zero and the validated saved hand pose must be inside the normal workspace.
No vertex attachments, constrained support flags, solver warm-start data, or
reward milestones are saved. Actual joint and nodal velocities are retained.

From the **repository root**:

```bash
# Four accepted states per bank; grasped includes two supported and two airborne.
uv run --no-sync python -m examples.tutorial.cloth_task.generator \
  --poses_per_phase 4 --num_envs 8 --seed 42 \
  --output logs/cloth_table_resets.npz --device cuda:0 --viz none

# Independently restore and validate each row, then display it.
uv run --no-sync python -m examples.tutorial.cloth_task.generator \
  --view logs/cloth_table_resets.npz --device cuda:0 --viz viser

# The same fresh-simulator replay checks without a viewer.
uv run --no-sync python -m examples.tutorial.cloth_task.generator \
  --view logs/cloth_table_resets.npz --device cuda:0 --viz none

# Optional downward allowance during the physical table approach.
uv run --no-sync python -m examples.tutorial.cloth_task.generator \
  --poses_per_phase 16 --num_envs 8 --partial_grasp_fraction 0.5 \
  --table_penetration 0.003 \
  --output logs/cloth_table_resets.npz --device cuda:0 --viz none
```

`GeneratorCfg` at the top of `generator.py` owns the proposal ranges. The CLI
exposes the seed, bank selection (`--phases reaching near_corner`, for example),
quota, supported-grasp fraction, batch limit, settling duration, and replay
duration. Increase the quota and environment count for a larger bank; `--max_batches` bounds rejection sampling.
Archive metadata records these settings, acceptance counts, solver configuration,
material values, and simulation timestep. `resets.py` owns archive validation and
restoration: all cloth vertices, named robot joints, drive targets, gripper command,
and phase are stored in compressed `.npz` arrays. Positions are relative to each
environment origin, so snapshots can be moved between vectorized environments.
The loader rejects changed mesh geometry or incompatible robot joints. Regenerate
when changing cloth refinement, placement, materials, robot, or physics.

## Train with the reset bank

The dataset event is enabled by default. `RESET_DATASET_PATH` in `env_cfg.py`
points to `logs/cloth_table_resets.npz` under the checkout, independently of the
current working directory. Startup logs report the loaded archive and bank counts.
The archive must exist; use the generator to create it when setting up another
checkout. The generator disables automatic sampling while constructing or viewing
its own states.

Train from the **repository root**:

```bash
uv run --no-sync python examples/tutorial/run.py train \
  --task Adapted_Cloth_Task --resets uniform --num_envs 64 --device cuda:0 \
  --max_iterations 800 --run_name cloth_diverse_resets
```

`run.py` accepts `--resets uniform` to sample uniformly from the fixed archive,
or `--resets none` to start with the robot at its home pose and the cloth flat on
the table. The option works for training, playback, zero/random agents, and
benchmarks of the cloth task. Omitting it keeps the task configuration's default
(currently uniform archive sampling). `uniform` also clears explicit phase
weights; rows are equally likely, so larger banks are sampled more often.

To use another bank, override
`env.events.reset_from_dataset.params.dataset_path=/absolute/path/to/archive.npz`
without `--resets`. An explicit `--resets` selection takes precedence over raw
reset overrides.

The archive is cached on the environment device once; each episode samples a row.
Equal bank quotas give equal bank probabilities with the default uniform row
sampling. Optional explicit bank weights follow the table's order:

```bash
# Append this override to the training command for more near-corner resets:
'env.events.reset_from_dataset.params.phase_weights=[0.2,0.4,0.2,0.2]'
```

A partial episode reset only restores the selected environments. Reward history
starts fresh, including for already held cloth. The saved open/closed gripper command is restored after action history resets,
so the first observation correctly identifies a held state. The policy controls
the next command; it must keep the gripper action negative to retain that grasp.
Passing generation/replay checks establishes usable initial states, not that PPO
has learned the complete task.

The generator regression test covers producing all four banks, requiring both
supported and airborne grasped rows, and unassisted replay at different environment
origins:

```bash
uv run --no-sync --with pytest python -m pytest examples/tutorial/test_cloth_generator.py
```
