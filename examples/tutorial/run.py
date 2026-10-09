# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the tutorial through Isaac Lab's existing agent and RL entry points."""

import argparse
import sys

import warp as wp

wp.config.enable_backward = False

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
    parser.add_argument("command", choices=("zero_agent", "random_agent", "train", "play"))
    parser.add_argument("--task", choices=(vial_task.TASK_ID,), default=vial_task.TASK_ID)
    args, remaining = parser.parse_known_args()
    runners = {
        "zero_agent": run_zero_agent_cli,
        "random_agent": run_random_agent_cli,
        "train": run_train_cli,
        "play": run_play_cli,
    }
    return runners[args.command](["--task", args.task, *remaining])


if __name__ == "__main__":
    sys.exit(main())
