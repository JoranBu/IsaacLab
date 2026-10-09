# SO101 vial placement

A minimal local adaptation of `IsaacTutorial-Place-Vial-SO101` from
[IsaacLabTutorial](https://github.com/isaac-sim/IsaacLabTutorial/tree/c64024e).
It uses this Isaac Lab checkout and its existing uv environment. No separate
tutorial installation or sibling checkout is needed.

The example registers one state-based task and includes its scene, actions,
observations, rewards, contact-based success detection, PPO settings, and all
1,024 reset poses. It imports the SO101 robot, simulation, managers, contact
sensors, standard MDP terms, and agent runners from Isaac Lab. USD assets use
the framework's normal asset server and cache; the first run needs network
access. The backend is Newton with MJWarp and requires an NVIDIA GPU.

## Run

Run from this folder so logs, checkpoints, and exports stay under `examples/tutorial`:

```bash
cd examples/tutorial

# Exercise the environment without a trained policy; exits after 120 steps.
uv run --no-sync python run.py zero_agent --num_envs 4 --viz none --max_steps 120
uv run --no-sync python run.py random_agent --num_envs 8 --viz none --max_steps 120

# Two PPO updates to check the training pipeline.
uv run --no-sync python run.py train --num_envs 32 --viz none --max_iterations 2 \
  agent.num_steps_per_env=8 agent.algorithm.num_mini_batches=2

# Train the state policy.
uv run --no-sync python run.py train --num_envs 4096 --viz none

# View the latest trained checkpoint in the Newton viewer.
uv run --no-sync python run.py play --num_envs 4 --viz newton_gl --checkpoint latest
```

The launcher supplies `--task IsaacTutorial-Place-Vial-SO101` and forwards other
arguments to Isaac Lab's shared runners. Use `uv run --no-sync python run.py train --help`
to see their options. These commands reuse the installed checkout environment;
`--no-sync` skips dependency synchronization.

For a browser viewer, install the repository's `viser` extra and select `--viz viser`:

```bash
uv run --extra viser python run.py zero_agent --num_envs 1 --viz viser
```

Training writes to `logs/rsl_rl/so101_vial_state/`. A smoke checkpoint proves
that the runner works; it has not learned the complete pick-and-place task.

## Task and resets

The policy has six actions: five measured-relative arm joint increments scaled
by 0.033 rad and one gripper increment scaled by 0.02 rad. Negative gripper
commands close the jaw; positive commands open it. Policy inputs contain 60
state values; the critic adds three contact flags. Simulation runs at 120 Hz
with two physics substeps and a 30 Hz control rate. Episodes last at most 20 seconds.

Training samples all eight reset phases uniformly: approach, pregrasp, grasp,
lift, reorient, transport, insert, and release. Playback uses sequential canonical
approach starts. Resets restore robot joints and controller targets, vial pose,
rack pose, zero velocities, and milestone history. Isaac Lab's managers clear
previous actions, reward state, success counters, and episode buffers, including
on partial and automatic resets.

Success requires the vial to remain upright, released, nearly stationary, and
seated in the target rack opening for ten consecutive control steps. Lost vials
and unstable robot motion terminate an episode; the time limit truncates it.
Object poses are written only during resets. Grasping, carrying, insertion, and
release use physical contacts.

The small `vial_task/reset_poses.npz` archive contains the original tutorial's
state tensors, converted without changing their values. It includes all eight
phases (128 poses each), so reset generation is unnecessary. Changing robot or
rack geometry requires generating matching poses.

## Generate reset poses

The local generator uses Isaac Lab's Newton IK solver and the same scene and
contact model as training. It executes grasp, lift, reorientation, transport,
insertion, and release with the vial fully dynamic, then rejects candidates
that fail the phase's contact, clearance, stability, or placement checks. Object
state writes initialize a candidate or restore a previously validated branch;
they do not carry the vial through a rollout.

From this folder, regenerate the bundled 1,024-pose archive:

```bash
uv run --no-sync python generate_resets.py --poses_per_phase 128 --batch_size 128 --viz none
```

The default output is `vial_task/reset_poses.npz`. The generator replaces it
atomically after all eight phases have met their quotas and passed schema
validation. Use `--output logs/reset_poses.npz` to write a separate archive.
Generation uses rejection sampling and can take substantially longer than an
environment smoke run. Progress includes accepted counts, rejections, and physical
diagnostics; a phase that exhausts `--max_attempts_per_phase` raises an error.

For a small trial, reduce both the row quota and the required independent branch seeds:

```bash
uv run --no-sync python generate_resets.py --output logs/reset_smoke.npz \
  --poses_per_phase 1 --batch_size 16 --branch_seed_count 2 --viz none
```

Inspect generated poses, or use a separate archive for training:

```bash
uv run --no-sync python generate_resets.py --view --dataset logs/reset_smoke.npz \
  --steps_per_pose 45 --viz viser
uv run --no-sync python run.py train --num_envs 32 --viz none \
  env.events.reset_from_dataset.params.dataset_path=logs/reset_smoke.npz
```

Both the bundled and generated archives use XYZW quaternions and the same state
fields. Newly generated archives also record generator settings and rejection
counts as JSON metadata, readable without Python pickle.

## Validate

From this folder:

```bash
uv run --no-sync --with pytest python -m pytest test_tutorial.py -q
```

The test uses one four-environment CUDA scene to check finite observations and
rewards, full and partial resets, automatic timeout resets, and physical success
from an upright vial placed in the rack. It also writes and replays the generated
archive format and checks that invalid poses preserve the previous archive.
It does not measure learned-policy
success from the home pose.

## Files

- `run.py`: task registration and dispatch to the existing Isaac Lab runners.
- `generate_resets.py`, `vial_task/generator.py`: physical reset generation and inspection.
- `vial_task/env_cfg.py`: scene, manager terms, and tuned Newton contact settings.
- `vial_task/actions.py`: gripper targets constrained to the robot's soft limits.
- `vial_task/terms.py`, `geometry.py`, `progress.py`: placement-specific logic.
- `vial_task/resets.py`, `reset_poses.npz`: reset sampling, state replay, and data.
- `vial_task/agent_cfg.py`: state PPO configuration.

Adapted tutorial code and reset data originate from IsaacLabTutorial commit
`c64024e` and retain its Apache 2.0 license in `LICENSE-IsaacLabTutorial`. This
adaptation uses Isaac Lab's current backend and IK APIs and supports slice resets.
Camera, distillation, and exact-evaluation workflows are omitted.

## Pull Isaac Lab updates

The checkout has two remotes:

- `origin`: your fork, `git@github.com:JoranBu/IsaacLab.git`.
- `upstream`: official Isaac Lab, `https://github.com/isaac-sim/IsaacLab.git`.

From the repository root, choose which remote's `develop` branch to merge into
your current branch:

```bash
git pull origin develop
git pull upstream develop
```

Commit or stash pending work before pulling. `git fetch upstream` refreshes
upstream references without merging; inspect `git log --oneline HEAD..upstream/develop`
before pulling. The official `main` branch is also available as `upstream/main`.
Adding the remote does not change your fork's push destination or merge updates.
