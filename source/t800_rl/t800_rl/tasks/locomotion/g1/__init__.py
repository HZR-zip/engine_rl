"""Unitree G1 12-DoF CMoE terrain locomotion task for Isaac Lab."""

import gymnasium as gym

from . import agents
from .env_cfg import G1CMoEEnvCfg

gym.register(
    id="G1-CMoE-Velocity-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1CMoEEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:G1CMoEPPORunnerCfg",
    },
)
