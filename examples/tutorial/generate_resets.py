# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate or inspect the local SO101 vial reset dataset."""

import argparse

import warp as wp

wp.config.enable_backward = False

from vial_task.generator import generate_main, view_main  # noqa: E402


def main() -> int:
    """Forward generator options to the selected local workflow."""
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--view", action="store_true")
    args, remaining = parser.parse_known_args()
    return view_main(remaining) if args.view else generate_main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
