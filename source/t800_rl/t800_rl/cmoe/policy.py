# SPDX-License-Identifier: BSD-3-Clause
# Adapted from CMoE (2026, Fudan University); attribution in CMOE_PORT.md.
"""Five-expert CMoE policy with the RSL-RL 3.0 TensorDict interface."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from .contract import ACTION_DIM, CRITIC_DIM, POLICY_DIM

LATENTS_KEY = '_cmoe_latents'
NEXT_FRAME_KEY = '_cmoe_next_frame'
NEXT_VALID_KEY = '_cmoe_next_valid'


def mlp(input_dim, hidden_dims, output_dim):
    layers = []
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim, width), nn.ELU()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class StateEstimator(nn.Module):
    def __init__(self, noise='uniform'):
        super().__init__()
        if noise not in ('uniform', 'normal'):
            raise ValueError('State latent noise must be uniform or normal.')
        self.noise = noise
        self.encoder = nn.Sequential(mlp(450, (128, 64), 32), nn.ELU())
        self.fc_mu, self.fc_var, self.fc_explicit = nn.Linear(32, 16), nn.Linear(32, 16), nn.Linear(32, 3)
        self.decoder = mlp(19, (32, 64, 128), 45)

    def encode(self, actor_obs, stochastic=True):
        hidden = self.encoder(actor_obs[..., :450].detach())
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden).clamp(-20, 20)
        explicit = self.fc_explicit(hidden)
        if stochastic:
            noise = torch.rand_like(mu) if self.noise == 'uniform' else torch.randn_like(mu)
            latent = mu + noise * (0.5 * log_var).exp()
        else:
            latent = mu
        return explicit, latent, mu, log_var

    def losses(self, actor_obs, critic_obs, next_frame, valid, kld_weight):
        explicit, latent, mu, log_var = self.encode(actor_obs)
        prediction = self.decoder(torch.cat((latent, explicit), -1))
        error = (prediction - next_frame.detach()).square().mean(-1)
        valid = valid.reshape(-1).to(error.dtype)
        reconstruction = (error * valid).sum() / valid.sum().clamp_min(1)
        velocity = F.mse_loss(explicit, critic_obs[:, 45:48].detach())
        kl = (-0.5 * (1 + log_var - mu.square() - log_var.exp()).sum(-1)).mean()
        return velocity + reconstruction + kld_weight * kl, {
            'velocity': velocity, 'state_reconstruction': reconstruction, 'state_kl': kl,
        }


class TerrainEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(mlp(77, (128, 64), 32), nn.ELU())
        self.fc_mu = nn.Linear(32, 16)
        self.decoder = mlp(16, (32, 64, 128), 77)

    def forward(self, actor_obs):
        return self.fc_mu(self.encoder(actor_obs[..., 450:527].detach()))

    def loss(self, actor_obs):
        return F.mse_loss(self.decoder(self(actor_obs)), actor_obs[..., 450:527].detach())


@torch.no_grad()
def sinkhorn(scores, eps=0.05, iterations=3):
    """Balanced prototype assignments, computed in log space to avoid overflow."""
    log_q = scores.T / eps
    prototypes, batch = log_q.shape
    log_q = log_q - torch.logsumexp(log_q.flatten(), 0)
    for _ in range(iterations):
        log_q -= torch.logsumexp(log_q, dim=1, keepdim=True) + math.log(prototypes)
        log_q -= torch.logsumexp(log_q, dim=0, keepdim=True) + math.log(batch)
    return (log_q + math.log(batch)).exp().T


class CMoEActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, obs, obs_groups, num_actions, actor_hidden_dims=(512, 256, 128),
                 critic_hidden_dims=(512, 256, 128), activation='elu', init_noise_std=1.0,
                 actor_obs_normalization=False, critic_obs_normalization=False,
                 noise_std_type='log', num_experts=5, num_prototypes=32, temperature=0.2,
                 state_latent_noise='uniform'):
        super().__init__()
        if obs_groups != {'policy': ['policy'], 'critic': ['critic']}:
            raise ValueError('CMoE requires separate policy and privileged critic groups.')
        if obs['policy'].shape[-1] != POLICY_DIM or obs['critic'].shape[-1] != CRITIC_DIM:
            raise ValueError('CMoE requires 527 actor and 128 critic observation values.')
        if num_actions != ACTION_DIM or num_experts != 5:
            raise ValueError('The reference G1 contract requires 12 actions and five experts.')
        if activation != 'elu' or actor_obs_normalization or critic_obs_normalization:
            raise ValueError('CMoE uses ELU and explicit observation scaling, without running normalization.')
        if noise_std_type != 'log' or init_noise_std <= 0 or temperature <= 0:
            raise ValueError('Use positive temperature/noise and log-parameterized action standard deviation.')
        self.state_latent_noise = state_latent_noise
        self.temperature = temperature
        self.state_estimator = StateEstimator(state_latent_noise)
        self.terrain_estimator = TerrainEstimator()
        self.actors = nn.ModuleList(mlp(157, actor_hidden_dims, num_actions) for _ in range(num_experts))
        self.critics = nn.ModuleList(mlp(CRITIC_DIM, critic_hidden_dims, 1) for _ in range(num_experts))
        self.gating_network = nn.Sequential(mlp(157, (128,), num_experts), nn.Softmax(dim=-1))
        self.gate_projector = mlp(num_experts, (128, 64), 16)
        self.terrain_projector = mlp(93, (128, 64), 16)
        self.prototypes = nn.Embedding(num_prototypes, 16)
        self.log_std = nn.Parameter(torch.full((num_actions,), math.log(init_noise_std)))
        self.distribution = None

    def ppo_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith(('state_estimator.', 'terrain_estimator.')):
                yield parameter

    @torch.no_grad()
    def encode_latents(self, actor_obs, stochastic=False):
        velocity, state, _, _ = self.state_estimator.encode(actor_obs, stochastic=stochastic)
        return torch.cat((velocity, state, self.terrain_estimator(actor_obs)), -1).detach()

    def features(self, obs):
        actor = obs['policy']
        latent = obs.get(LATENTS_KEY)
        if latent is None:
            latent = self.encode_latents(actor, stochastic=False)
        return torch.cat((actor[:, :45], latent[:, :19].detach(), actor[:, 450:527],
                          latent[:, 19:].detach()), -1)

    def mean_and_gate(self, obs):
        features = self.features(obs)
        gate = self.gating_network(features)
        means = torch.stack([actor(features) for actor in self.actors], dim=1)
        return (means * gate.unsqueeze(-1)).sum(1), gate

    def update_distribution(self, obs):
        mean, self.gate_weights = self.mean_and_gate(obs)
        self.distribution = Normal(mean, self.log_std.exp().expand_as(mean))

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        # Always infer from the estimator means, without changing global train/eval state.
        return self.mean_and_gate(obs)[0]

    def evaluate(self, obs, **kwargs):
        # Recompute from this observation: never reuse stale gates from a previous act call.
        gate = self.gating_network(self.features(obs)).detach()
        values = torch.stack([critic(obs['critic']) for critic in self.critics], dim=1)
        return (values * gate.unsqueeze(-1)).sum(1)

    def contrastive_loss(self, obs):
        features = self.features(obs)
        gate = self.gating_network(features.detach())
        terrain = features[:, -93:].detach()
        gate_z = F.normalize(self.gate_projector(gate), dim=-1)
        terrain_z = F.normalize(self.terrain_projector(terrain), dim=-1)
        prototypes = F.normalize(self.prototypes.weight, dim=-1)
        scores_g, scores_t = gate_z @ prototypes.T, terrain_z @ prototypes.T
        q_g, q_t = sinkhorn(scores_g), sinkhorn(scores_t)
        # Preserve the upstream mean over batch AND prototypes (not sum over prototypes).
        return -0.5 * (q_g * F.log_softmax(scores_t / self.temperature, -1)
                       + q_t * F.log_softmax(scores_g / self.temperature, -1)).mean()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(-1)

    def update_normalization(self, obs):
        pass

    def reset(self, dones=None):
        pass
