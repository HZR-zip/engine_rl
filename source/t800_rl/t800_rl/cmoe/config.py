"""Simulator-independent defaults shared by the Lab config and CPU integration tests."""
from copy import deepcopy

POLICY_CONFIG = dict(class_name='CMoEActorCritic', init_noise_std=1.0, noise_std_type='log',
                     actor_obs_normalization=False, critic_obs_normalization=False,
                     actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
                     activation='elu', num_experts=5, num_prototypes=32, temperature=0.2,
                     state_latent_noise='uniform')
ALGORITHM_CONFIG = dict(class_name='CMoEPPO', value_loss_coef=1.0, use_clipped_value_loss=True,
                        clip_param=0.2, entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4,
                        learning_rate=1e-3, schedule='adaptive', gamma=0.99, lam=0.95,
                        desired_kl=0.01, max_grad_norm=1.0, normalize_advantage_per_mini_batch=False,
                        rnd_cfg=None, symmetry_cfg=None, contrastive_coef=1.0, kld_weight=0.005,
                        estimator_max_grad_norm=10.0)


def training_config():
    return dict(class_name='CMoEOnPolicyRunner', num_steps_per_env=24, max_iterations=50000,
                save_interval=200, seed=42, device='cpu', clip_actions=100.0, logger='tensorboard',
                obs_groups={'policy': ['policy'], 'critic': ['critic']},
                experiment_name='g1_cmoe_lab', policy=deepcopy(POLICY_CONFIG),
                algorithm=deepcopy(ALGORITHM_CONFIG))
