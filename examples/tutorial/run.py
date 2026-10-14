# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the tutorial through Isaac Lab's existing agent and RL entry points."""

import argparse
import math
import sys

import warp as wp

wp.config.enable_backward = False

import cloth_task
import vial_task  # noqa: E402

from isaaclab_rl.entrypoints import (  # noqa: E402
    run_play_cli,
    run_random_agent_cli,
    run_train_cli,
    run_zero_agent_cli,
)


def main() -> int:
    """Register the local example and forward arguments to the shared runner."""
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("command", choices=("zero_agent", "random_agent", "train", "play", "benchmark"))
    parser.add_argument("--task", choices=(vial_task.TASK_ID, cloth_task.TASK_ID), default=vial_task.TASK_ID)
    parser.add_argument(
        "--resets",
        choices=("none", "uniform", "weighted"),
        default=None,
        help="Cloth resets: home pose (none), uniform archive rows, or weighted phases. Omit to use task config.",
    )
    parser.add_argument(
        "--reset_probabilities",
        type=float,
        nargs=4,
        metavar=("REACHING", "NEAR_CORNER", "GRASPED", "NEAR_GOAL"),
        help="Four phase probabilities summing to one; requires --resets weighted.",
    )
    args, remaining = parser.parse_known_args()
    help_requested = any(arg in ("-h", "--help") for arg in remaining)
    if help_requested:
        parser.print_help()
    if args.reset_probabilities is not None and not help_requested:
        if args.resets != "weighted":
            parser.error("--reset_probabilities requires --resets weighted.")
        probabilities = args.reset_probabilities
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in probabilities):
            parser.error("Reset probabilities must be finite numbers between zero and one.")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1.0e-6):
            parser.error("Reset probabilities must sum to one.")
    elif args.resets == "weighted" and not help_requested:
        parser.error("--resets weighted requires --reset_probabilities with four numbers.")
    reset_overrides = []
    if args.resets is not None and not help_requested:
        if args.task != cloth_task.TASK_ID:
            parser.error(f"--resets is only supported with --task {cloth_task.TASK_ID}.")
        from cloth_task.env_cfg import RESET_DATASET_PATH

        dataset_path = "null" if args.resets == "none" else RESET_DATASET_PATH
        reset_overrides.append(f"env.events.reset_from_dataset.params.dataset_path={dataset_path}")
        if args.resets == "uniform":
            reset_overrides.append("env.events.reset_from_dataset.params.phase_weights=null")
        elif args.resets == "weighted":
            reset_overrides.append(f"env.events.reset_from_dataset.params.phase_weights={args.reset_probabilities}")
    if args.command == "benchmark":
        from isaaclab.benchmark import run_benchmark_cli

        if not remaining or remaining[0] in ("-h", "--help"):
            return run_benchmark_cli(remaining)
        return run_benchmark_cli([remaining[0], "--task", args.task, *remaining[1:], *reset_overrides])

    runners = {
        "zero_agent": run_zero_agent_cli,
        "random_agent": run_random_agent_cli,
        "train": run_train_cli,
        "play": run_play_cli,
    }
    return runners[args.command](["--task", args.task, *remaining, *reset_overrides])


if __name__ == "__main__":
    sys.exit(main())
