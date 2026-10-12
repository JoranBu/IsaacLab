# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark both cloth-lift solvers using identical scripted, headless rollouts.

    uv run python thesis/benchmark_cloth_lift.py --runs 3 --num_steps 900

Each trial uses a fresh process to isolate simulator state. The execution order
alternates between repetitions. Initial settling is untimed to exclude JIT and
CUDA graph capture; each trial still executes the complete trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def main() -> None:
    """Run both programs, retain their logs, and summarize timings and outcomes."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=3, help="Repetitions per solver.")
    parser.add_argument("--num_steps", type=int, default=900, help="Control steps in each full rollout.")
    parser.add_argument("--warmup_steps", type=int, default=60, help="Initial untimed settling steps (1 to 60).")
    parser.add_argument("--device", default="cuda:0", help="Isaac Lab simulation device.")
    parser.add_argument("--grasp_height", type=float, default=0.007)
    parser.add_argument("--lift_height", type=float, default=0.15)
    parser.add_argument(
        "--output_dir", type=Path, help="Results folder; defaults to a timestamped thesis/benchmarks folder."
    )
    args = parser.parse_args()
    if args.runs < 1 or not 1 <= args.warmup_steps <= 60 or args.num_steps <= args.warmup_steps:
        parser.error("Use runs >= 1, 1 <= warmup_steps <= 60, and num_steps > warmup_steps.")
    if args.grasp_height < 0.0 or args.lift_height < 0.0:
        parser.error("Heights must be nonnegative.")

    thesis_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir or thesis_dir / "benchmarks" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir.resolve()
    # Refuse to overwrite an earlier experiment accidentally.
    output_dir.mkdir(parents=True, exist_ok=False)
    scripts = {"vbd": "cloth_lift.py", "mjwarp_vbd_proxy": "cloth_lift_proxy.py"}
    results = []
    for repetition in range(args.runs):
        order = list(scripts) if repetition % 2 == 0 else list(reversed(scripts))
        for solver in order:
            stem = f"{solver}_{repetition + 1:02d}"
            result_path = output_dir / f"{stem}.json"
            log_path = output_dir / f"{stem}.log"
            command = [
                sys.executable,
                str(thesis_dir / scripts[solver]),
                "--viz",
                "none",
                "--device",
                args.device,
                "--num_steps",
                str(args.num_steps),
                "--warmup_steps",
                str(args.warmup_steps),
                "--grasp_height",
                str(args.grasp_height),
                "--lift_height",
                str(args.lift_height),
                "--benchmark_output",
                str(result_path),
            ]
            print(f"Running {solver}, repetition {repetition + 1}/{args.runs} ...", flush=True)
            started = time.perf_counter()
            with log_path.open("w") as log:
                process = subprocess.run(command, cwd=thesis_dir.parent, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode != 0:
                raise RuntimeError(f"{solver} exited with {process.returncode}. See {log_path}")
            result = json.loads(result_path.read_text())
            if results and result["experiment_configuration"] != results[0]["experiment_configuration"]:
                raise RuntimeError("The trials have different scene, controller, or shared physics settings.")
            result["process_seconds"] = time.perf_counter() - started
            result["repetition"] = repetition + 1
            results.append(result)
            print(f"  {result['milliseconds_per_control_step']:.3f} ms/control step", flush=True)

    # CSV keeps the final cloth motion beside each timing. A high cloth vertex
    # is a useful outcome measurement, but does not establish a stable grasp.
    configuration_fields = {"outcome", "experiment_configuration", "physics_configuration"}
    rows = [
        {**{k: v for k, v in result.items() if k not in configuration_fields}, **result["outcome"]}
        for result in results
    ]
    with (output_dir / "trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    medians = {
        solver: statistics.median(r["milliseconds_per_control_step"] for r in results if r["solver"] == solver)
        for solver in scripts
    }
    speedup = medians["vbd"] / medians["mjwarp_vbd_proxy"]
    summary = {"median_ms_per_control_step": medians, "proxy_speedup_over_vbd": speedup, "trials": results}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nMedian ms/control step: VBD {medians['vbd']:.3f}; proxy {medians['mjwarp_vbd_proxy']:.3f}")
    print(f"Proxy speedup over VBD: {speedup:.2f}x (greater than 1 means proxy is faster).")
    print(f"Results and per-trial logs: {output_dir}")


if __name__ == "__main__":
    main()
