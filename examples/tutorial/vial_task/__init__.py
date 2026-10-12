# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register the state-based Franka vial-lifting task."""

import gymnasium as gym

TASK_ID = "Vial_Pickup_Task"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ClothEnvCFG",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:SO101StatePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
