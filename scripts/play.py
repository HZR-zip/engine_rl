"""Evaluate a trained T800 policy checkpoint."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a T800 Isaac Lab checkpoint.")
parser.add_argument("--task", default="T800-Velocity-Flat-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from t800_rl.cmoe.runner import runner_class

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import t800_rl.tasks  # noqa: F401


def main():
    if not args_cli.checkpoint:
        parser.error("--checkpoint is required")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.observations.policy.enable_corruption = False
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = runner_class(agent_cfg.to_dict())(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.path.abspath(args_cli.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs = env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
