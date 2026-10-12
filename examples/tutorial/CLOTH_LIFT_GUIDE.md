# From vial placement to cloth lifting

This is a development guide, not an implemented cloth environment. It starts
from the structure of this tutorial and points to working framework components
you can reuse. Commands below run existing tasks; the proposed `cloth_task`
package is a future addition. No training or benchmarks were run to write this
guide.

## Recommended starting point

Adapt existing code. Keep the tutorial's manager-based organization, and use
Isaac Lab's `Isaac-Lift-Cloth-Franka` task as the reference for cloth physics.
Create a separate cloth task so the vial example remains available for
comparison. You do not need to implement a simulator, PPO trainer, environment
manager, or IK controller yourself.

The decision is about which task components to reuse. Writing a direct
environment from scratch does not automatically give better performance.
For this project, first measure the cloth solver, collisions, resets, and
observation costs. Consider changing the environment workflow only if profiling
shows the managers are a significant bottleneck.

The vial tutorial uses MJWarp to simulate rigid bodies. Cloth is a surface
deformable: its state is the position and velocity of every mesh node. In this
checkout, Newton cloth requires VBD. The existing Franka cloth task couples an
MJWarp robot with VBD cloth using `CouplerProxyCfg`. Your
[scripted prototype](../../thesis/cloth_lift.py) instead runs the robot and cloth
together in VBD. These are different solver architectures; neither should be
assumed faster without measurement. See the official
[deformables guide](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/deformables.html)
and [coupled-solvers guide](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/coupled_solvers.html).

For learning the framework, start with Franka because the maintained cloth task
already configures it. If SO101 is your final robot, migrate to it after the
cloth grasp and lift work. If learning specifically on SO101 is essential,
keep that robot from the beginning, but budget extra work for its gripper,
reachable poses, and coupling selectors. A Franka policy cannot be reused
unchanged with SO101.

## 1. Understand what one environment step does

An RL environment repeats this loop:

```text
observation -> policy -> action -> physics -> reward and termination
     ^                                             |
     +-------------- next observation -------------+
                   reset if episode ends
```

Training runs many copies of that loop in parallel. PPO is the algorithm that
updates the policy from the collected experience.

In `env_cfg.py`, a configuration describes how to construct the environment;
it does not itself perform every calculation. `@configclass` makes these
configuration objects usable by Isaac Lab. Managers call the functions
referenced by their configured terms.

| Tutorial component | What it controls | Cloth-task decision |
| --- | --- | --- |
| Scene configuration | Robot, objects, surfaces, sensors | Replace vial and rack with cloth and a simple support |
| Actions | How policy numbers move the robot | Start with a small, bounded action space |
| Observations | What the policy can see | Replace rigid-object pose with cloth features |
| Events and resets | Initial state and randomization | Restore all cloth nodes and robot state |
| Rewards | Which behavior is encouraged | Approach, grasp, lift, hold |
| Terminations | Success, failure, and timeout | Use measurable cloth and robot conditions |
| Physics configuration | Solver, timestep, contacts | Reuse the maintained cloth configuration |
| Agent configuration | PPO and neural-network settings | Start with a small state-based policy |
| `run.py` | Registration and runner dispatch | Reuse shared runners; allow the new task ID |

Read [the tutorial config](vial_task/env_cfg.py) alongside
[the maintained cloth config](../../source/isaaclab_tasks/isaaclab_tasks/core/lift/config/franka_soft/franka_cloth_env_cfg.py).
Follow one action, observation, reward, and reset term to its implementation.

**Checkpoint:** You can explain what changes when you edit an action scale,
a reward weight, the simulation timestep, or the episode duration.

## 2. Record two baselines before changing anything

From `examples/tutorial`, exercise and benchmark the vial task:

```bash
uv run --no-sync python run.py zero_agent --num_envs 4 --viz none --max_steps 120
uv run --no-sync python run.py benchmark runtime \
  --num_envs 128 --viz none --num_steps 1000 --warmup_steps 100 \
  --seed 42 --output_path logs/benchmarks/vial_128
```

Also inspect the already registered cloth task. You can run the standard CLI
from this folder so reports remain under `examples/tutorial/logs`:

```bash
uv run --no-sync isaaclab zero_agent --task Isaac-Lift-Cloth-Franka \
  --num_envs 1 --viz viser --max_steps 300

uv run --no-sync isaaclab benchmark runtime --task Isaac-Lift-Cloth-Franka \
  --num_envs 32 --viz none --num_steps 1000 --warmup_steps 100 \
  --seed 42 --output_path logs/benchmarks/cloth_reference_32
```

The viewer command requires the `viser` extra. After an upstream update,
synchronize the repository environment, including the extras you use, before
using `--no-sync` to repeat experiments. That flag reuses installed packages;
it does not update them to a changed lockfile.

The tutorial launcher currently restricts `--task` to the vial task, so use
the standard CLI for the built-in cloth task. A zero-action run checks physics
and initialization; it does not prove that grasping works.

**Checkpoint:** Both scenes launch and reset. You have recorded hardware,
revision, environment count, and benchmark settings. Do not compare the two
throughput numbers as equivalent workloads: their physics differs.

## 3. Define an easy first task and a measurable goal

Start with one square cloth in a fixed pose, one robot, and an accessible edge
draped over a simple support. The existing cloth task has support blocks for
this purpose. Leave cameras, variable cloth topology, and material
randomization for later.

A first learning objective can be: approach a known edge, pinch it, lift the
cloth, and hold it. Picking up cloth lying completely flat on a table is a
later stage: the fingers must get underneath or gather the edge, which can be
physically difficult even with a good controller.

Define success before designing rewards. For example, use these initial
targets, then adjust them to your scene and intended meaning of "lift":

- Cloth mean height increases at least 8 cm above its reset mean height.
- At least 80% of nodes are above the support surface by 2 cm.
- The selected grasp region stays near the closed gripper, and cloth motion
  has settled sufficiently to exclude a brief throw.
- Those conditions remain true for one simulated second.

These numbers are proposed task definitions, not validated solver settings.
Use support height and environment-local coordinates, rather than assuming
the table is at world `z = 0`. Convert the hold duration to control steps using
`ceil(hold_seconds / env.step_dt)`.

Track approach distance, cloth height, fraction of lifted nodes, retained grasp,
and hold duration separately. Maximum node height alone can reward lifting
one corner or launching the cloth while most of it stays on the table.

**Checkpoint:** A recorded trajectory can be classified as success or failure
without relying on visual judgment or its total reward.

## 4. Create a small cloth task using existing components

Keep additions under the example folder. A possible layout is:

```text
examples/tutorial/
  run.py
  vial_task/              existing example
  cloth_task/             proposed new task
    __init__.py           registers a distinct Gym task ID
    env_cfg.py            cloth scene and manager configuration
    terms.py              only custom cloth features/rewards/success
    resets.py             only reset behavior absent from shared terms
    agent_cfg.py          small PPO configuration
```

Reuse or subclass `FrankaClothEnvCfg` for your initial configuration, importing
it from Isaac Lab rather than copying its solver setup. Keep inherited asset
names such as `deformable` when reusing terms that expect those names. When
you replace the scene or support geometry, also replace the inherited reset
terms and coupling selectors that refer to them.

For a locally registered task, import `cloth_task` in the launcher and extend
its allowed task IDs. Register the new configuration and its PPO configuration
using the same Gym registration pattern as `vial_task/__init__.py`. Do this
after the new package exists. No launcher change is needed to experiment with
the already registered `Isaac-Lift-Cloth-Franka` task.

Replace the vial-specific scene, observations, milestone logic, rewards,
terminations, and reset dataset. The original eight vial phases and
`reset_poses.npz` do not describe deformable cloth states. Reuse framework
action terms, manager classes, PPO runners, and appropriate deformable MDP
terms. Copying all of `vial_task/terms.py` would bring irrelevant placement
logic into the new task.

**Checkpoint:** The new task is independently registered and can run one
environment without rewards or randomization obscuring physics problems.

## 5. Validate cloth physics and a scripted grasp

First run cloth with gravity and no robot motion. Check that it settles on
the support, stays finite, and has plausible stretch and thickness. Start
from the maintained task's material and contact configuration. Do not transfer
MJWarp's vial contact constants to VBD cloth.

Then drive a slow scripted sequence: approach, descend beside the accessible
edge, close the fingers, lift, and hold. Your
[existing prototype](../../thesis/cloth_lift.py) illustrates this sequence,
but its docstring correctly calls it a grasp attempt rather than a guaranteed
successful lift. Its high friction values and modified material parameters
are experiment settings, not general defaults.

Validate measured fingertip motion as well as commanded targets. In the
prototype's all-VBD architecture, a post-step `newton.eval_ik` callback updates
joint coordinates from the moved rigid links. That feedback matters to its
IK controller. Do not omit it when reusing that architecture, or add it blindly
to the maintained MJWarp/VBD coupled architecture.

In a coupled scene, the robot belongs to the rigid solver and cloth nodes to
the deformable solver. Proxies expose the gripper colliders to cloth contact.
If switching to SO101, inspect its actual body paths and update the selectors;
Franka finger selectors will not match SO101. Coupling is experimental, so
validate grasp retention and force response for the pinned checkout.

Use physical gripper contact for the intended task. Pinning cloth vertices
to the fingers can help isolate another problem, but an attached-cloth lift
is a different task. Do not teleport cloth nodes during a rollout to make a
grasp succeed.

**Checkpoint:** The script can repeatedly lift and retain the cloth through
a hold. If it cannot, investigate contact geometry, edge access, controller
tracking, thickness, and materials before training PPO.

## 6. Choose simple actions and observations

For Franka, a useful first action space is bounded XYZ fingertip displacement
plus a gripper command, with a fixed downward orientation. Use the framework's
IK action implementation. Alternatively, retain bounded joint increments if
they already provide reliable control. On SO101, respect its reachable poses
and available degrees of freedom; do not assume arbitrary six-dimensional
end-effector commands are feasible.

Start with small movements and verify their units, reference frame, joint
limits, and control rate using a script. A controller operates on many physics
steps between policy decisions. Decimation controls that ratio;
`control_dt = simulation_dt * decimation`. Solver substeps subdivide each
simulation step further and do not increase the policy decision rate.

For the first state-based policy, expose:

- Robot joint positions and velocities, fingertip pose, and gripper opening.
- Previous action, so smooth-motion penalties have useful context.
- A few fixed cloth landmarks, including the target grasp patch and corners.
- Landmark velocities, cloth centroid, and target height.
- Any phase flag or hold counter needed by stateful reward or success logic.

Express positions relative to the robot or environment origin. Keep landmark
indices consistent across environments; mesh topology must remain fixed for
that index scheme. Compute the dense cloth state once when needed and reuse
features where practical. Reading simulator state gives privileged information:
a future camera-based or real-robot policy will need a different observation
pipeline.

**Checkpoint:** Observations and actions have fixed shapes, finite values,
and correct frames. Switching robots, actions, or observations may invalidate
existing checkpoints; start a fresh training experiment.

## 7. Implement complete resets before adding a curriculum

A rigid object can be restored using its pose and velocity. Cloth needs every
node's position and velocity restored. In the current Newton implementation,
`DeformableObject.reset()` itself is a no-op; use the framework's nodal-state
write methods or appropriate deformable reset terms through the event manager.

For each resetting environment:

1. Restore robot joints and velocities and their controller targets.
2. Restore the cloth nodal state from a valid flat or settled reference state.
3. Restore movable support state if your scene uses it.
4. Clear temporary node pins or kinematic targets if your experiment used them.
5. Clear custom grasp flags, hold counters, and reward history.
6. Let the environment managers reset their action, episode, and term state.

Use the provided subset of environment IDs. A reset of environment 0 must not
alter environments 1 through N. Framework nodal references may already include
world offsets; inspect the reset implementation before adding environment
origins, so you do not apply offsets twice.

Prefer a saved settled reference and small valid pose changes over simulating
many settling steps inside every reset. If settling is required, make it
explicit and keep it from advancing unrelated active episodes. Measure reset
cost and ensure post-reset observations describe the restored state.

Begin with one deterministic start. Later, collect a small cloth-specific bank
of physically validated states. Such a bank must include node positions and
velocities, robot state, controller state where required, and support state;
metadata should identify mesh topology and physics settings. A bank made with
one mesh cannot be reused blindly after changing its resolution.

**Checkpoint:** Repeated full, partial, and automatic resets reproduce valid
states without moving unaffected environments or retaining previous progress.

## 8. Add rewards one behavior at a time

Start with a dense approach reward toward the chosen grasp patch. Reward
lifting only when the grasp geometry supports a retained grasp. Add a modest
penalty for abrupt action changes and excessive joint motion. Define a success
bonus using the same height, retention, and hold conditions as the success
termination.

Do not reward gripper height alone: an empty gripper can rise. Do not reward
finger closure alone: the robot can close away from the cloth. A gripper-contact
signal also needs checking against cloth state; rigid-body contact observations
from the vial task do not automatically measure cloth-particle contacts.

Reuse the existing deformable approach and lifting reward functions when they
match your objective. Their generic centroid/nearest-node criteria may need
a task-specific grasp-patch term. Keep success metrics stricter than useful
dense shaping and log each component to detect reward shortcuts.

Separate failure from timeout. Non-finite state, leaving the permitted
workspace, or an unrecoverable loss can terminate an episode; reaching its
time limit truncates it. A missed grasp does not need immediate termination if
retrying is part of the intended behavior.

**Checkpoint:** The scripted successful lift earns more useful reward than
doing nothing, waving an empty gripper, or briefly flinging the cloth.

## 9. Start PPO on an easy distribution, then widen it

Before your custom task is ready, this checks the existing cloth task's
training pipeline from `examples/tutorial`:

```bash
uv run --no-sync isaaclab train --task Isaac-Lift-Cloth-Franka \
  --num_envs 32 --viz none --max_iterations 2 \
  agent.num_steps_per_env=8 agent.algorithm.num_mini_batches=2
```

Two PPO updates establish that collection and optimization execute; they do
not establish a learned lift. For your registered custom task, run the same
kind of smoke check and then train long enough to measure actual behavior.

Progress through these training distributions:

| Stage | Start condition | What the agent must learn |
| --- | --- | --- |
| Lift and hold | Physically validated grasped cloth | Maintain a grasp while lifting |
| Close, lift, hold | Open fingers near an accessible edge | Acquire and retain the grasp |
| Approach and lift | Robot farther from the same fixed cloth | Complete the full sequence |
| Varied accessible edges | Small cloth position and angle changes | Robust approach and grasp |
| Flat cloth pickup | Edge closer to the support surface | Gather or pinch a harder edge |
| Wider variation | Material, size, and initial-shape changes | Generalize within validated physics |

A grasped reset must be produced by real contact, not a hidden attachment.
Increase difficulty when evaluated behavior meets a chosen success target,
not merely when training reward rises. Keep evaluation starts independent of
the training reset bank, including full approach starts. Evaluate several
seeds and report success rate and time to lift separately.

**Checkpoint:** The policy lifts cloth from a full start distribution, rather
than succeeding only when reset into an almost-complete lift.

## 10. Measure performance as part of development

Benchmark the actual cloth task. Do not use vial throughput to choose the
cloth training batch size. Start with 1, 8, 32, and 128 environments, then
increase only while throughput and memory use remain useful. Thousands of
cloth environments are not guaranteed to be practical.

Keep a record like this for each experiment:

| Setting or result | Record |
| --- | --- |
| Reproducibility | Revision, dependency lock, GPU, seed |
| Workload | Environment count, actual node/triangle counts, observations |
| Physics | Solver architecture, dt, decimation, substeps, iterations, contacts |
| Runtime | Warmup and measured steps, transitions/s, control-step timing |
| Resources | GPU memory and utilization, startup duration |
| Task quality | Scripted retention, penetration/stretch, policy success rate |
| Training | Collection time, optimizer time, wall time to a target success rate |

Prioritize these changes:

1. Disable viewers and cameras for the state-based training baseline.
2. Use the coarsest mesh that preserves the intended grasp and deformation.
   `edge_refinement` is a mesh-resolution parameter, not a node count; measure
   the resulting mesh. A mesh too coarse for finger contact can change the task.
3. Use compact observations and batched tensor operations on the GPU. Avoid
   Python loops per environment and `.cpu()`, `.numpy()`, or `.item()` in
   every-step reward/observation code. Occasional diagnostic reads are useful.
4. Keep CUDA graph compatibility where the chosen solver supports it. Register
   required callbacks before capture and avoid changing topology during a run.
5. Reduce solver iterations, substeps, contact complexity, or self-contact
   only after a repeated grasp-and-hold check still passes. More parallel
   environments do not compensate for invalid physics.
6. Benchmark frequent-reset workloads as well as stable rollouts. Cloth reset
   and dataset selection can dominate short episodes.
7. Profile physics, actions, observations, rewards, and PPO separately before
   rewriting the environment. Compare architectures with equivalent task quality.

Self-contact may be deferred for an easy unfolded-sheet task, but enable and
validate it when folds and cloth-on-cloth interaction matter. Check capacities
when contacts overflow. Hiding solver warnings changes output, not physics
quality. Record the warning policy when comparing timings.

The runtime benchmark measures environment stepping with random actions;
it does not include PPO optimization or prove task success. Use training
timings and time-to-success to judge the final learning system. There is no
measured performance winner for your scene in this guide.

**Checkpoint:** You can explain a performance change using both timing and
physical/task metrics. Choose the fastest setup that still performs the
intended lift reliably.

## Suggested first session

Read the vial and maintained cloth configurations, view one instance of the
built-in cloth task, and record its headless benchmark. Then try the scripted
approach-close-lift sequence with an accessible edge. Create the custom RL task
only after that sequence works. This keeps each new problem small enough to
identify whether it belongs to physics, control, resets, or learning.

Further references:

- [VBD configuration and tuning](../../docs/source/how-to/solver_tuning_vbd.rst)
- [Deformable state and supported backends](../../docs/source/concepts/deformables.rst)
- [Coupled solver configuration](../../docs/source/concepts/coupled_solvers.rst)
- [Manager-based RL environment tutorial](../../docs/source/how-to/create_manager_rl_env.rst)
- [Maintained cloth PPO config](../../source/isaaclab_tasks/isaaclab_tasks/core/lift/config/franka_soft/agents/rsl_rl_ppo_cfg.py)
