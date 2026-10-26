# SPDX-License-Identifier: BSD-3-Clause
# Adapted from CMoE (2026, Fudan University) and RSL-RL (ETH Zurich/NVIDIA).
"""CMoE PPO with separate estimator optimizers and episode-safe auxiliary targets."""
import torch
from rsl_rl.algorithms import PPO
from .policy import LATENTS_KEY, NEXT_FRAME_KEY, NEXT_VALID_KEY


class CMoEPPO(PPO):
    def __init__(self, policy, contrastive_coef=1.0, kld_weight=0.005,
                 estimator_max_grad_norm=10.0, **kwargs):
        if any(kwargs.get(key) is not None for key in ('rnd_cfg', 'symmetry_cfg', 'multi_gpu_cfg')):
            raise ValueError('CMoE port supports single-device PPO without RND or symmetry augmentation.')
        super().__init__(policy, **kwargs)
        self.ppo_params = list(policy.ppo_parameters())
        self.optimizer = torch.optim.Adam(self.ppo_params, lr=self.learning_rate)
        self.state_optimizer = torch.optim.Adam(policy.state_estimator.parameters(), lr=self.learning_rate)
        self.terrain_optimizer = torch.optim.Adam(policy.terrain_estimator.parameters(), lr=self.learning_rate)
        self.contrastive_coef = contrastive_coef
        self.kld_weight = kld_weight
        self.estimator_max_grad_norm = estimator_max_grad_norm

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        batch = num_envs * num_transitions_per_env
        if batch < 2 or self.num_mini_batches < 1 or batch % self.num_mini_batches:
            raise ValueError('Rollout size must be >= 2 and divisible by the number of minibatches.')
        augmented = obs.clone()
        augmented[LATENTS_KEY] = obs['policy'].new_zeros((num_envs, 35))
        augmented[NEXT_FRAME_KEY] = obs['policy'].new_zeros((num_envs, 45))
        augmented[NEXT_VALID_KEY] = obs['policy'].new_zeros((num_envs, 1))
        super().init_storage(training_type, num_envs, num_transitions_per_env, augmented, actions_shape)

    def act(self, obs):
        augmented = obs.clone()
        # Freeze the sampled latent inputs with each transition: unchanged weights must give ratio=1.
        augmented[LATENTS_KEY] = self.policy.encode_latents(obs['policy'], stochastic=self.policy.training)
        augmented[NEXT_FRAME_KEY] = torch.zeros_like(obs['policy'][:, :45])
        augmented[NEXT_VALID_KEY] = torch.zeros_like(obs['policy'][:, :1])
        return super().act(augmented)

    def process_env_step(self, obs, rewards, dones, extras):
        self.transition.observations[NEXT_FRAME_KEY] = obs['critic'][:, :45].detach().clone()
        # Lab may already have reset done environments. Never learn reset states as next-state targets.
        self.transition.observations[NEXT_VALID_KEY] = (~dones.bool()).reshape(-1, 1).float()
        super().process_env_step(obs, rewards, dones, extras)

    def update(self):
        if self.storage.step != self.storage.num_transitions_per_env:
            raise RuntimeError('Collect a full rollout before calling CMoE PPO update.')
        totals = {}
        updates = 0
        for obs, actions, old_value, advantage, returns, old_log_prob, old_mu, old_std, _, _ in (
            self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        ):
            advantage = advantage.detach().reshape(-1)
            if self.normalize_advantage_per_mini_batch:
                advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
            self.policy.update_distribution(obs)
            log_prob = self.policy.get_actions_log_prob(actions)
            values = self.policy.evaluate(obs)
            entropy = self.policy.entropy.mean()
            with torch.no_grad():
                sigma = self.policy.action_std
                kl = (torch.log(sigma / old_std) + (old_std.square()
                      + (old_mu - self.policy.action_mean).square()) / (2 * sigma.square()) - 0.5).sum(-1).mean()
                if self.schedule == 'adaptive' and self.desired_kl is not None:
                    if kl > self.desired_kl * 2:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0 < kl < self.desired_kl / 2:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                for optimizer in (self.optimizer, self.state_optimizer, self.terrain_optimizer):
                    for group in optimizer.param_groups:
                        group['lr'] = self.learning_rate
            ratio = (log_prob - old_log_prob.detach().reshape(-1)).exp()
            surrogate = -torch.minimum(ratio * advantage,
                                        ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * advantage).mean()
            value_errors = (values - returns.detach()).square()
            if self.use_clipped_value_loss:
                clipped = old_value + (values - old_value).clamp(-self.clip_param, self.clip_param)
                value_errors = torch.maximum(value_errors, (clipped - returns.detach()).square())
            value_loss = value_errors.mean()
            contrastive = self.policy.contrastive_loss(obs)
            loss = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            loss = loss + self.contrastive_coef * contrastive
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite CMoE PPO loss.')
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ppo_params, self.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()

            state_loss, aux = self.policy.state_estimator.losses(
                obs['policy'], obs['critic'], obs[NEXT_FRAME_KEY], obs[NEXT_VALID_KEY], self.kld_weight)
            terrain_loss = self.policy.terrain_estimator.loss(obs['policy'])
            for optimizer, module, auxiliary_loss in (
                (self.state_optimizer, self.policy.state_estimator, state_loss),
                (self.terrain_optimizer, self.policy.terrain_estimator, terrain_loss),
            ):
                optimizer.zero_grad(set_to_none=True)
                auxiliary_loss.backward()
                torch.nn.utils.clip_grad_norm_(module.parameters(), self.estimator_max_grad_norm,
                                               error_if_nonfinite=True)
                optimizer.step()
            metrics = dict(aux, surrogate=surrogate, value=value_loss, entropy=entropy,
                           contrastive=contrastive, terrain_reconstruction=terrain_loss, kl=kl)
            for name, metric in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(metric.detach())
            updates += 1
        self.storage.clear()
        return {name: value / updates for name, value in totals.items()}
