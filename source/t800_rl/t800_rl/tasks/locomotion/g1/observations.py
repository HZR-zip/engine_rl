"""Isaac Lab adapter for synchronized CMoE policy and privileged observations."""
import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.sensors.ray_caster.patterns import grid_pattern
from t800_rl.cmoe.contract import JOINT_NAMES
from t800_rl.cmoe.observations import ObservationHistory, corrupt_observations, proprioception


def height_pattern(cfg, device):
    starts, directions = grid_pattern(cfg, device)
    starts[:, 0] += 0.4
    return starts, directions


class CMoEObservation(ManagerTermBase):
    """Share one noisy snapshot/history between actor and critic in each environment step."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not hasattr(env, '_cmoe_observations'):
            env._cmoe_observations = {
                'history': ObservationHistory(env.num_envs, env.device), 'step': None, 'snapshot': None,
            }
        self.shared = env._cmoe_observations
        self.joint_ids, names = env.scene['robot'].find_joints(list(JOINT_NAMES), preserve_order=True)
        if tuple(names) != JOINT_NAMES:
            raise ValueError('USD articulation joints do not match the G1 action/observation contract.')

    def reset(self, env_ids=None):
        self.shared['history'].reset(env_ids)
        self.shared['step'] = None

    def __call__(self, env, group='policy'):
        step = getattr(env, 'common_step_counter', 0)
        if self.shared['step'] != step:
            robot = env.scene['robot']
            frame = proprioception(
                env.command_manager.get_command('base_velocity'), robot.data.root_ang_vel_b,
                robot.data.projected_gravity_b,
                robot.data.joint_pos[:, self.joint_ids] - robot.data.default_joint_pos[:, self.joint_ids],
                robot.data.joint_vel[:, self.joint_ids], env.action_manager.action,
            )
            # The author uses world terrain elevation, not base-to-ground clearance.
            hits = env.scene['height_scanner'].data.ray_hits_w[..., 2]
            if not torch.isfinite(hits).all():
                raise ValueError('Height scanner missed terrain; refusing invalid CMoE observations.')
            heights = hits.clamp(-100, 100) * 5.0
            if env.cfg.observations.policy.enable_corruption:
                frame, heights = corrupt_observations(frame, heights)
            # This task has no external-force randomizer; zero is its actual applied force.
            force = getattr(env, 'cmoe_external_force_b', torch.zeros_like(robot.data.root_lin_vel_b))
            self.shared['snapshot'] = self.shared['history'].push(
                frame, robot.data.root_lin_vel_b * 2.0, force, heights)
            self.shared['step'] = step
        return self.shared['snapshot'][0 if group == 'policy' else 1]
