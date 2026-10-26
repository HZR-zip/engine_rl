# CMoE Isaac Gym Reference

This directory vendors the G1 training framework from the official CMoE
repository so it can be inspected and reproduced alongside this Isaac Lab
project. It is an isolated Isaac Gym codebase, not an Isaac Lab port.

- Upstream: https://github.com/Hoshi-No-Ai/CMoE
- Imported commit: `4575d6ae762adf0043b4eabf1b8224a227d0ce38`
- Upstream repository license: `LICENSE`
- Third-party notices: `NOTICE`
- Component licenses: `legged_gym/LICENSE`, `rsl_rl/LICENSE`, and their
  respective `licenses/` directories

The imported `legged_gym/` contains the G1 CMoE task, humanoid environment,
terrain generation and height sampling, train/play launchers, and robot URDF
and mesh assets. The imported `rsl_rl/` contains the CMoE actor-critic,
state and terrain estimators, PPO loss, and runner.

## Runtime

The upstream implementation targets Isaac Gym Preview 4, Python 3.8,
PyTorch 1.13.1 with CUDA 11.7, and a compatible NVIDIA driver. Install and run
it in a separate environment from the Isaac Lab project environment. Follow
the upstream `README.md` for Isaac Gym installation and the exact dependency
setup.

From this directory, after activating that environment:

```bash
python -m pip install -e rsl_rl
python -m pip install -e legged_gym
python legged_gym/legged_gym/scripts/train.py --task=g1cmoe --alg=cmoe --run_name=baseline
```

The task is registered as `g1cmoe` in
`legged_gym/legged_gym/envs/g1/g1_cmoe_config.py`. Training outputs are
written under `legged_gym/logs/`.

The source was copied without adapting its simulator APIs. GPU training and
visual replay have not been run in this workspace; treat this as the pinned
upstream G1 reproduction reference until validated in the Isaac Gym runtime.
