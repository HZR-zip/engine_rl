"""CMoE configuration for the G1 port, targeting RSL-RL 3.0.1."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


from t800_rl.cmoe.config import POLICY_CONFIG, ALGORITHM_CONFIG

@configclass
class CMoEPolicyCfg(RslRlPpoActorCriticCfg):
    class_name: str = "CMoEActorCritic"
    noise_std_type: str = "log"
    state_latent_noise: str = "uniform"
    num_experts: int = 5
    num_prototypes: int = 32
    temperature: float = 0.2


@configclass
class CMoEAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "CMoEPPO"
    contrastive_coef: float = 1.0
    kld_weight: float = 0.005
    estimator_max_grad_norm: float = 10.0


@configclass
class G1CMoEPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "CMoEOnPolicyRunner"
    clip_actions = 100.0
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 200
    experiment_name = "g1_cmoe_lab"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = CMoEPolicyCfg(**POLICY_CONFIG)
    algorithm = CMoEAlgorithmCfg(**ALGORITHM_CONFIG)
