# SPDX-License-Identifier: BSD-3-Clause
"""RSL-RL runner integration with complete CMoE training checkpoint state."""
from copy import deepcopy
import os
from pathlib import Path
import random
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from .contract import JOINT_NAMES
from .policy import CMoEActorCritic
from .ppo import CMoEPPO

CHECKPOINT_VERSION = 1


class CMoEOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        self.checkpoint_config = deepcopy(train_cfg)
        super().__init__(env, deepcopy(train_cfg), log_dir=log_dir, device=device)
        self.logger_type = self.cfg.get('logger', 'tensorboard')

    def _configure_multi_gpu(self):
        if int(os.getenv('WORLD_SIZE', '1')) > 1:
            raise ValueError('Distributed CMoE estimator synchronization is not implemented.')
        super()._configure_multi_gpu()

    def _construct_algorithm(self, obs):
        policy_cfg = dict(self.policy_cfg)
        algorithm_cfg = dict(self.alg_cfg)
        if policy_cfg.pop('class_name') != 'CMoEActorCritic' or algorithm_cfg.pop('class_name') != 'CMoEPPO':
            raise ValueError('CMoE runner requires CMoEActorCritic and CMoEPPO configurations.')
        policy = CMoEActorCritic(obs, self.cfg['obs_groups'], self.env.num_actions, **policy_cfg).to(self.device)
        algorithm = CMoEPPO(policy, device=self.device, **algorithm_cfg)
        algorithm.init_storage('rl', self.env.num_envs, self.cfg['num_steps_per_env'], obs, [self.env.num_actions])
        return algorithm

    def save(self, path, infos=None):
        checkpoint = {
            'cmoe_checkpoint_version': CHECKPOINT_VERSION,
            'joint_names': JOINT_NAMES,
            'model_state_dict': self.alg.policy.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'state_optimizer_state_dict': self.alg.state_optimizer.state_dict(),
            'terrain_optimizer_state_dict': self.alg.terrain_optimizer.state_dict(),
            'learning_rate': self.alg.learning_rate,
            # Base runner calls save after a completed iteration indexed from zero.
            'next_iteration': self.current_learning_iteration + 1,
            'tot_timesteps': self.tot_timesteps, 'tot_time': self.tot_time,
            'config': self.checkpoint_config, 'infos': infos,
            'rng_torch': torch.get_rng_state(), 'rng_python': random.getstate(),
            'rng_numpy': np.random.get_state(),
        }
        if torch.cuda.is_initialized():
            checkpoint['rng_cuda'] = torch.cuda.get_rng_state_all()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
        if self.writer is not None and self.logger_type in ('neptune', 'wandb') and not self.disable_logs:
            self.writer.save_model(str(path), self.current_learning_iteration)

    def load(self, path, load_optimizer=True, map_location=None):
        # Load only checkpoints you trust: optimizer/RNG state includes Python objects.
        state = torch.load(path, map_location=map_location or self.device, weights_only=False)
        if state.get('cmoe_checkpoint_version') != CHECKPOINT_VERSION:
            raise ValueError('Not a compatible Lab CMoE checkpoint; upstream Gym weights need explicit conversion.')
        if tuple(state['joint_names']) != JOINT_NAMES:
            raise ValueError('Checkpoint joint ordering does not match the environment.')
        if state['config']['policy'] != self.checkpoint_config['policy']:
            raise ValueError('Checkpoint policy configuration differs from the current model.')
        if load_optimizer and state['config']['algorithm'] != self.checkpoint_config['algorithm']:
            raise ValueError('Resume requires the same algorithm configuration.')
        self.alg.policy.load_state_dict(state['model_state_dict'], strict=True)
        if load_optimizer:
            self.alg.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.alg.state_optimizer.load_state_dict(state['state_optimizer_state_dict'])
            self.alg.terrain_optimizer.load_state_dict(state['terrain_optimizer_state_dict'])
            self.alg.learning_rate = state['learning_rate']
            self.current_learning_iteration = state['next_iteration']
            self.tot_timesteps = state['tot_timesteps']
            self.tot_time = state['tot_time']
            torch.set_rng_state(state['rng_torch'].cpu())
            random.setstate(state['rng_python'])
            np.random.set_state(state['rng_numpy'])
            if 'rng_cuda' in state and torch.cuda.is_initialized():
                if len(state['rng_cuda']) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all([item.cpu() for item in state['rng_cuda']])
        return state['infos']


def runner_class(config):
    """Resolve task-owned runner selection without global eval/monkeypatching."""
    name = config.get('class_name', 'OnPolicyRunner')
    if name == 'CMoEOnPolicyRunner':
        return CMoEOnPolicyRunner
    if name == 'OnPolicyRunner':
        return OnPolicyRunner
    raise ValueError(f'Unsupported runner: {name}')
