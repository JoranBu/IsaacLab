# Comparing the cloth-lift solvers

Run these commands from the IsaacLab repository root.

The original environment uses VBD for both robot and cloth:

```bash
uv run --extra viser python thesis/cloth_lift.py --viz viser
```

The coupled environment uses MJWarp for the robot and VBD for the cloth:

```bash
uv run --extra viser python thesis/cloth_lift_proxy.py --viz viser
```

Open the Viser URL printed in the terminal, or its forwarded address in VS Code.
Run one visualized simulation at a time so both do not compete for port 8080.
Both scripts accept `--num_steps`, `--grasp_height`, `--lift_height`, and `--device`.
The default scripted sequence settles, approaches, descends, closes, lifts, and
then holds until Ctrl+C. A 900-step rollout lasts 15 simulated seconds.

The proxy version inherits the original scene, materials, robot drives, action
configuration, and trajectory. Changes to those values in `cloth_lift.py` affect
both environments. VBD retains its 20 iterations; both configurations use four
substeps, a physics timestep of 1/120 s, and control decimation of two. Cloth–robot
contacts use the same vertex/shape collision method. All collidable robot bodies
are exposed to the cloth solver as virtual proxies; cloth contact impulses feed
back to MJWarp with lagged coupling and one coupling pass per substep. The floor
is visible to both solver views. No cloth vertices are attached to the gripper.

MJWarp uses an elliptic friction cone and the `implicitfast` integrator, following
the repository's Franka cloth proxy example. The all-VBD version reconstructs
joint coordinates after each physics step for IK feedback. MJWarp maintains those
coordinates directly and does not need that callback. The solvers can therefore
produce different motion and grasp outcomes even with identical task settings.

Run the comparison:

```bash
uv run python thesis/benchmark_cloth_lift.py --runs 3 --num_steps 900
```

Optional settings:

```bash
uv run python thesis/benchmark_cloth_lift.py \
    --runs 5 --num_steps 1200 --device cuda:0 \
    --grasp_height 0.007 --lift_height 0.15 \
    --output_dir /tmp/cloth_comparison
```

Use an output directory that does not already exist. Results default to
`thesis/benchmarks/<timestamp>/`. Each trial runs in a fresh process with no
visualizer, and the order alternates between repetitions. The first 60 control
steps are settling steps excluded from timing; they allow initial kernel
compilation and CUDA graph capture to finish. Warm-up does not skip any trajectory
phase. `--warmup_steps` accepts 1–60; keep the default for comparisons. Use at least
900 total steps to include lifting and a short hold.

Timing synchronizes the GPU at the start and end of the measured rollout. It
includes control target generation, IK, physics, scene updates, and observations.
Per-phase printing and final state measurements happen outside the timed loop.
This measures the complete environment rollout, rather than physics alone.

The output contains:

- `summary.json`: all trials, configuration snapshots, median milliseconds per
  control step, and the ratio `VBD time / proxy time`. A ratio above one means the
  proxy version is faster.
- `trials.csv`: per-trial timing and final fingertip error, cloth min/mean/max
  height, and the fraction of cloth vertices above 3 cm.
- One JSON result and terminal log per trial, including setup time and device.
  The summary also records total process time, including imports and teardown.

The benchmark checks that scene, actions, observations, material settings, and
shared timing settings match. Final cloth height describes motion; it does not
prove a stable grasp. Compare the outcomes alongside speed and use Viser to
inspect the pinch. This benchmark uses the original single environment; it does
not measure large-batch training throughput.

For one individual measurement:

```bash
uv run python thesis/cloth_lift_proxy.py --viz none --num_steps 900 \
    --benchmark_output /tmp/cloth_proxy_trial.json
```
