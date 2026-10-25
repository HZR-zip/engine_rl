"""使用 RSL-RL 的 PPO 算法训练 T800 速度跟踪策略。

可以把本脚本理解成训练的“总调度入口”：启动 Isaac Sim，读取环境和算法配置，
创建并包装环境，然后让 RSL-RL 的 runner 与环境交互、更新策略并保存训练结果。
机器人模型、奖励、观测和 PPO 超参数等具体内容不都写在这里，需查看任务配置文件。
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# 先定义本脚本提供的命令行参数。比如：
#   --num_envs 16       同时运行 16 份并行环境
#   --max_iterations 1  只训练 1 次迭代，适合做启动/配置冒烟测试
#   --checkpoint xxx.pt 从已有模型权重继续加载
# default=None 表示不从命令行覆盖，后面沿用配置文件里的值。
parser = argparse.ArgumentParser(description="Train the T800 Isaac Lab task.")
parser.add_argument("--task", default="T800-Velocity-Flat-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
# Isaac Lab / Isaac Sim 自己也需要命令行参数（例如 --device、--enable_cameras）。
AppLauncher.add_app_launcher_args(parser)
# parse_known_args 会把本脚本认识的参数放进 args_cli，其他参数留给 Hydra。
args_cli, hydra_args = parser.parse_known_args()
# Hydra 会在后续加载配置时解析它的参数；暂时把它们从原始 argv 中取出交给 Hydra。
sys.argv = [sys.argv[0]] + hydra_args

# 必须先启动 Isaac Sim，再导入/创建依赖仿真器的任务环境。
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 以下导入放在 Simulation App 启动之后是 Isaac Lab 脚本的常见约定。
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import t800_rl.tasks  # noqa: F401
# 上面的 tasks 导入虽然没有在本文件中直接使用名字，但它有注册副作用：
# 让 Gymnasium 知道 "T800-Velocity-Flat-v0" 这个环境 ID 对应哪个环境类和配置。


def main():
    # 读取环境配置。num_envs/device 等命令行选项可以覆盖任务配置中的默认值。
    # env_cfg 主要描述仿真场景、机器人、观测、动作、奖励、终止条件等。
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # 从 Gym 注册信息中读取该任务对应的 RSL-RL 配置（这里的配置入口名为
    # rsl_rl_cfg_entry_point）。agent_cfg 包含 PPO、网络、学习率、训练轮数等设置。
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    # 用户显式传入 seed/最大迭代次数时覆盖配置文件；否则保留配置里的默认值。
    agent_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    # 训练日志和模型检查点放在项目当前目录下的 logs/rsl_rl/<实验名>。
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    os.makedirs(log_root, exist_ok=True)

    # gym.make 根据 task ID 创建环境；开启相机时请求 RGB 图像渲染，否则不创建相机渲染输出。
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.enable_cameras else None)

    # RSL-RL 需要特定的向量环境接口。包装器负责适配 Isaac Lab 环境与 RSL-RL，
    # clip_actions 则按算法配置限制策略输出的动作范围。
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # OnPolicyRunner 是 RSL-RL 的训练循环管理器：接收环境、算法配置、日志目录和设备。
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_root, device=agent_cfg.device)
    if args_cli.checkpoint:
        # 加载已有 checkpoint 中的策略/训练状态；不传该参数时从头开始训练。
        runner.load(args_cli.checkpoint)

    # 开始 PPO 训练。init_at_random_ep_len=True 让各并行环境从随机 episode 进度开始，
    # 避免所有环境在训练开始时都处于完全相同的 episode 时刻。
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    # 释放环境及其仿真资源。
    env.close()


if __name__ == "__main__":
    try:
        # 只有直接运行 python scripts/train.py 时才启动训练；被其他 Python 文件导入时不会自动训练。
        main()
    finally:
        # 无论训练正常结束还是抛出异常，都尽量关闭 Isaac Sim 应用。
        simulation_app.close()
