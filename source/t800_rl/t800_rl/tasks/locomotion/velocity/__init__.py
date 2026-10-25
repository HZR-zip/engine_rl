"""Velocity-tracking task registration."""

import gymnasium as gym

from . import agents
from .env_cfg import T800FlatEnvCfg

gym.register(
    id="T800-Velocity-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:T800FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:T800PPORunnerCfg",
    },
)
