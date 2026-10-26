"""CPU contracts: observations, learning gradients, rollout targets, and resume continuity."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import torch
from tensordict import TensorDict
from t800_rl.cmoe.config import training_config
from t800_rl.cmoe.contract import URDF_PATH, joint_parameters
from t800_rl.cmoe.observations import ObservationHistory, proprioception
from t800_rl.cmoe.policy import LATENTS_KEY, NEXT_FRAME_KEY, NEXT_VALID_KEY, sinkhorn
from t800_rl.cmoe.runner import CMoEOnPolicyRunner


class TensorEnvironment:
    """Small CPU vector-environment interface fixture; NOT a robot simulator."""
    num_envs, num_actions, max_episode_length = 4, 12, 3
    device = 'cpu'

    def __init__(self):
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.frame = torch.zeros(self.num_envs, 45)
        self.heights = torch.linspace(-0.1, 0.1, 77).repeat(self.num_envs, 1)
        self.history = ObservationHistory(self.num_envs)
        self.obs = self._observe()

    def _observe(self):
        actor, critic = self.history.push(self.frame, self.frame[:, :3],
                                           torch.zeros(self.num_envs, 3), self.heights)
        return TensorDict({'policy': actor, 'critic': critic}, batch_size=[self.num_envs])

    def get_observations(self):
        return self.obs

    def step(self, actions):
        self.frame = self.frame.clone()
        self.frame[:, :12] += 0.05 * actions
        self.frame[:, -12:] = actions
        self.episode_length_buf += 1
        dones = self.episode_length_buf >= self.max_episode_length
        rewards = 1.0 - self.frame[:, :12].square().mean(-1)
        self.frame[dones] = 0
        self.episode_length_buf[dones] = 0
        self.history.reset(dones.nonzero().flatten())
        self.obs = self._observe()
        return self.obs, rewards, dones, {'time_outs': dones.clone()}


def make_runner(log_dir=None):
    cfg = training_config()
    cfg['num_steps_per_env'] = 4
    cfg['save_interval'] = 1
    cfg['algorithm']['num_learning_epochs'] = 2
    cfg['algorithm']['num_mini_batches'] = 2
    return CMoEOnPolicyRunner(TensorEnvironment(), cfg, log_dir=log_dir, device='cpu')


def collect(runner):
    obs = runner.env.get_observations()
    with torch.inference_mode():
        for _ in range(runner.num_steps_per_env):
            actions = runner.alg.act(obs)
            obs, rewards, dones, extras = runner.env.step(actions)
            runner.alg.process_env_step(obs, rewards, dones, extras)
        runner.alg.compute_returns(obs)


class CMoECPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(21)

    def assert_nested_equal(self, left, right):
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_reference_asset_limits_and_meshes(self):
        joints = joint_parameters()
        self.assertEqual(len(joints), 12)
        self.assertEqual(next(iter(joints)), 'left_hip_pitch_joint')
        self.assertEqual(joints['left_hip_roll_joint']['effort'], 139)
        self.assertEqual(joints['right_ankle_pitch_joint']['effort'], 50)
        self.assertEqual(joints['left_knee_joint']['stiffness'], 150)
        for mesh in ET.parse(URDF_PATH).findall('.//mesh'):
            self.assertTrue((URDF_PATH.parent / mesh.get('filename')).is_file())

    def test_observation_scaling_history_and_partial_reset(self):
        ones3, ones12 = torch.ones(2, 3), torch.ones(2, 12)
        frame = proprioception(ones3, ones3, ones3, ones12, ones12, ones12)
        torch.testing.assert_close(frame[0, :6], torch.tensor([2., 2., .25, .25, .25, .25]))
        self.assertAlmostEqual(frame[0, 21].item(), .05)
        history = ObservationHistory(2)
        a, c = history.push(frame, ones3 * 2, ones3 * 0, torch.zeros(2, 77))
        self.assertEqual(a.shape, (2, 527))
        self.assertEqual(c.shape, (2, 128))
        torch.testing.assert_close(a[:, :45], a[:, 405:450])
        history.reset(torch.tensor([0]))
        a, _ = history.push(frame + 1, ones3, ones3, torch.zeros(2, 77))
        torch.testing.assert_close(a[0, :45], a[0, 45:90])
        torch.testing.assert_close(a[1, 45:90], frame[1])
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            history.push(frame, ones3, ones3, torch.full((2, 77), float('inf')))

    def test_privilege_isolation_and_gradient_boundaries(self):
        runner = make_runner()
        policy = runner.alg.policy
        obs = runner.env.get_observations().clone()
        obs[LATENTS_KEY] = policy.encode_latents(obs['policy'], stochastic=True)
        before = policy.act_inference(obs)
        changed = obs.clone()
        changed['critic'] += 10
        torch.testing.assert_close(before, policy.act_inference(changed))
        policy.zero_grad(set_to_none=True)
        policy.evaluate(obs).sum().backward()
        self.assertTrue(all(p.grad is None for p in policy.gating_network.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.state_estimator.parameters()))
        self.assertTrue(all(p.grad is not None for p in policy.critics.parameters()))
        policy.zero_grad(set_to_none=True)
        (policy.act_inference(obs).square().mean() + policy.contrastive_loss(obs)).backward()
        for expert in policy.actors:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in expert.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.gating_network.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.state_estimator.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.terrain_estimator.parameters()))
        self.assertTrue(torch.isfinite(sinkhorn(torch.randn(4, 32) * 1000)).all())
        torch.testing.assert_close(sinkhorn(torch.randn(4, 32)).sum(-1), torch.ones(4))

    def test_stored_probabilities_and_reset_target_mask(self):
        runner = make_runner()
        collect(runner)
        storage = runner.alg.storage
        obs = storage.observations.flatten(0, 1)
        runner.alg.policy.update_distribution(obs)
        log_prob = runner.alg.policy.get_actions_log_prob(storage.actions.flatten(0, 1))
        torch.testing.assert_close(log_prob, storage.actions_log_prob.flatten(), atol=2e-6, rtol=1e-6)
        self.assertTrue((storage.observations[NEXT_VALID_KEY][2] == 0).all())
        self.assertTrue((storage.observations[NEXT_VALID_KEY][0] == 1).all())
        estimator = runner.alg.policy.state_estimator
        torch.manual_seed(17)
        _, losses1 = estimator.losses(obs['policy'], obs['critic'], obs[NEXT_FRAME_KEY], obs[NEXT_VALID_KEY], .005)
        targets = obs[NEXT_FRAME_KEY].clone()
        targets[obs[NEXT_VALID_KEY].flatten() == 0] = 10000
        torch.manual_seed(17)
        _, losses2 = estimator.losses(obs['policy'], obs['critic'], targets, obs[NEXT_VALID_KEY], .005)
        torch.testing.assert_close(losses1['state_reconstruction'], losses2['state_reconstruction'])

    def test_full_update_and_exact_optimizer_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = make_runner(log_dir=directory)
            before = deepcopy(runner.alg.policy.state_dict())
            runner.learn(1)
            for prefix in ('actors.', 'critics.', 'gating_network.', 'state_estimator.',
                           'terrain_estimator.', 'gate_projector.', 'terrain_projector.', 'prototypes.'):
                self.assertTrue(any(not torch.equal(before[k], v) for k, v in runner.alg.policy.state_dict().items()
                                    if k.startswith(prefix)), prefix)
            path = Path(directory) / 'model_0.pt'
            self.assertTrue(path.is_file())
            restored = make_runner()
            restored.load(path)
            self.assertEqual(restored.current_learning_iteration, 1)
            for name in ('optimizer', 'state_optimizer', 'terrain_optimizer'):
                self.assert_nested_equal(getattr(runner.alg, name).state_dict(), getattr(restored.alg, name).state_dict())
            self.assert_nested_equal(runner.alg.policy.state_dict(), restored.alg.policy.state_dict())
            # Feed identical restarted environments and RNG to compare the NEXT real optimizer update.
            runner.load(path)
            runner.env = TensorEnvironment()
            collect(runner)
            first_metrics = runner.alg.update()
            restored.load(path)
            restored.env = TensorEnvironment()
            collect(restored)
            second_metrics = restored.alg.update()
            self.assertEqual(first_metrics, second_metrics)
            self.assert_nested_equal(runner.alg.policy.state_dict(), restored.alg.policy.state_dict())
            for value in first_metrics.values():
                self.assertTrue(torch.isfinite(torch.tensor(value)))
            if runner.writer is not None:
                runner.writer.close()

    def test_inference_is_deterministic_and_bad_checkpoint_rejected(self):
        runner = make_runner()
        policy = runner.get_inference_policy('cpu')
        obs = runner.env.get_observations()
        torch.testing.assert_close(policy(obs), policy(obs), rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.pt'
            torch.save({'model_state_dict': runner.alg.policy.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, 'compatible Lab CMoE'):
                runner.load(path)


if __name__ == '__main__':
    unittest.main()
