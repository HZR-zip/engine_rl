# CMoE Isaac Lab CPU Port — implementation record

当前结论：三项代码工作已实现，六项 CPU 测试通过。训练测试使用张量测试
环境，不是机器人仿真；不能据此宣称机器人会走路或论文已复现。
下一台训练主机仍需完成模型转换、真实关节控制、传感器和训练启动验收。

Source: Hoshi-No-Ai/CMoE, commit 4575d6ae762adf0043b4eabf1b8224a227d0ce38.
The reference tree remains unchanged. The new policy/estimators/PPO are adapted
from its BSD-3-Clause sources (2026 CMoE authors, Fudan University), with the
RSL-RL 3.0.1 PPO/runner interface (ETH Zurich and NVIDIA). See the original
licenses under references/cmoe_isaacgym/legged_gym and rsl_rl. Copies of the CMoE, RSL-RL and legged_gym notices are included in the installed cmoe/licenses directory.

## Scope and version boundary

Implementing the requested three stages on CPU: G1 asset/action/observation
alignment, CMoE networks, and PPO plus full learning-state checkpointing.
This project retains its existing Isaac Lab 2.3.x / Isaac Sim 5.1 API surface
and targets rsl-rl-lib 3.0.1. Installing the upstream skills did not migrate
the project to Lab 3.x. No GPU simulation or locomotion success is claimed.

## Stage 1 — robot and observations

- Use the exact fixed-upper-body URDF selected by the author, under
  references/cmoe_isaacgym/legged_gym/resources/robots/g1/29dof_urdf/.
  Despite the filename it has 12 movable leg joints. All referenced meshes exist.
- Explicit order, left then right: hip pitch, hip roll, hip yaw, knee,
  ankle pitch, ankle roll. Bind actions and joint observations by name.
- Explicit ideal PD: hip 100/2, knee 150/4, ankle 40/2 (stiffness/damping).
  Torque/velocity limits are read from that URDF, not generic constants.
- Target angle = default angle + 0.25 * clipped action; action clip = 100.
  Physics dt = 0.005 s, decimation = 4. Enable robot self-collision.
- Conversion disables implicit USD drives, merges fixed bodies, and outputs
  assets/g1/g1_cmoe.usd. G1_USD_PATH overrides the asset location.
  ENGINE_RL_ROOT can point to a checkout containing the reference assets.
- Frame = command(3), body angular velocity(3), gravity(3), joint offset(12),
  joint velocity(12), previous applied action(12): total 45.
- Scale commands by [2, 2, 0.25], angular velocity by 0.25, joint velocity
  by 0.05; repeat initial frame at reset, newest-first 10-frame history.
- Actor = 450 history + 77 heights. Critic = current 45 + scaled body linear
  velocity 3 + external force 3 + heights 77 = 128 (not 139).
- Height grid has x=-0.1..0.9, y=-0.3..0.3, x outer/y inner ordering,
  pelvis-yaw alignment. Inputs are world terrain elevation times 5, not
  base-to-ground clearance. Non-finite ray hits raise an error.
- Share one noisy snapshot across actor/critic; preserve upstream uniform
  proprio noise and height noise plus four high/four low outliers.
- No external force randomizer exists in the current task; the force slots
  are zero. Future force randomization must populate cmoe_external_force_b.

## Stage 2 — networks

- State encoder: 450 -> 128 -> 64 -> 32, with velocity(3), mean(16),
  log-variance(16) heads; decoder 19 -> 32 -> 64 -> 128 -> 45.
- Terrain autoencoder: 77 -> 128 -> 64 -> 32 -> 16 and
  16 -> 32 -> 64 -> 128 -> 77. No unused terrain variance head.
- Five independent actors: 157 -> 512 -> 256 -> 128 -> 12.
  Features = current frame 45 + velocity 3 + state latent 16 + heights 77
  + terrain latent 16. ELU hidden activations.
- Gate: 157 -> 128 -> 5 softmax. Weighted means define ONE Normal policy,
  with shared learned standard deviation; this is not sampling an expert.
- Five independent privileged critics: 128 -> 512 -> 256 -> 128 -> 1.
  Their weights are the detached gates computed for the same observation.
- Gate/terrain projectors and 32 normalized prototypes implement Sinkhorn
  swapped assignments. Preserve the original batch-and-prototype mean loss.

## Stage 3 — training and checkpoints

- RSL-RL TensorDict rollout/GAE, clipped PPO and value losses, entropy, KL
  learning-rate adaptation, contrastive loss, and separate state/terrain Adam
  optimizers. Estimator parameters are excluded from the PPO optimizer.
- Every rollout stores detached latent features with its observation. PPO
  reuses those inputs, avoiding a new random latent on each probability check.
- Next-frame reconstruction targets use next critic[:45]. Done transitions
  are masked because Lab may already have returned the reset observation.
- Save/load model, all three optimizer states, adaptive learning rate, next
  iteration index, counters, policy/algorithm config, joint order, and RNG state.
  Atomic replacement prevents exposing a partially written checkpoint.
- Playback loads weights without restoring optimizers/RNG. Author Gym
  checkpoints are rejected pending an explicit weight conversion tool.
- Resume restarts environments; physical simulator state is not checkpointed.
  RNG restoration does not promise bit-identical cross-host trajectories.
- This port rejects multi-GPU, RND and symmetry augmentation explicitly.

## Deliberate corrections versus upstream

- State latent noise defaults to uniform to retain the released code behavior,
  even though its KL term has a Gaussian form. A normal-noise option is explicit.
- Fix the upstream reconstruction slice [3:48] to the full next frame [0:45].
- Cache rollout latents, prevent reset-target contamination, and recompute
  critic gates instead of relying on stale gates from a previous act call.
- Use log-standard-deviation parameters for positive action noise; bound
  state log variance and compute Sinkhorn in log space for numerical safety.
- Use normalized prototype weights in the computation graph rather than
  mutating parameters between forward and backward.
- Auxiliary optimizers step after the main PPO backward, without sharing
  parameters or stale estimator gradients with the PPO optimizer.

These fixes are documented behavior changes, not bitwise reproduction of the
released Gym training trajectory. Compare their effects during later ablations.

## Still outside this CPU implementation

Ray casts are not the author's four-sample heightfield averaging with pose/XY
noise. Terrain types/proportions/curriculum, full reward terms, command heading
logic, and physical randomization remain simplified. The fixed-joint USD,
self-collisions, standing pose, joint control and live sensor behavior require
validation on the user's GPU training host. Do not call this paper-result
reproduction until those differences and measured evaluation results are addressed.

## Validation

Validated on Python 3.11.16, torch 2.7.0+cu128 (CPU tensors only),
rsl-rl-lib 3.0.1. The six unittest cases passed:

1. Reference URDF movable joints, representative limits and all mesh paths.
2. Observation scaling, dimensions, newest-first history and partial resets.
3. Privileged data cannot leak into actor outputs; critic gate and estimator
   gradient boundaries are respected; expert and contrastive gradients exist.
4. Re-evaluating rollout actions before updates preserves log probabilities;
   resetting environments cannot contaminate next-frame reconstruction.
5. The real RSL-RL runner collects a rollout, updates every intended module,
   saves all optimizers, reloads and reproduces the next CPU optimizer update
   with identical inputs and RNG (not identical physical trajectories).
6. Deterministic inference and rejection of incompatible checkpoints.

Commands from the project root in a Python environment containing PyTorch/RSL-RL:

    python -m pip install -e '.[cmoe]'
    python -m unittest discover -s tests -p test_cmoe_cpu.py -v

Without installing this extension (as used in the current CPU validation):

    PYTHONPATH=source/t800_rl python -m unittest discover -s tests -p test_cmoe_cpu.py -v

Python AST/syntax checks and git diff --check passed. A no-dependency wheel
build passed; its module layout is t800_rl/cmoe rather than a nested namespace,
and generated bytecode caches are excluded.

A one-iteration launcher check with --device cpu and CUDA_VISIBLE_DEVICES empty
stopped at importing isaaclab.app: ModuleNotFoundError: No module named isaaclab.
No simulator was started. GPU training, USD conversion, standing and physical
joint/sensor checks are still pending on the user's training host.

## GPU-host handoff

Use the compatible Isaac Lab 2.3.x / Isaac Sim 5.1 environment, validate its
runtime imports, and install this package with its pinned RSL-RL extra. Do not
install the reference tree's older rsl_rl into that environment.

    python -m pip install -e '.[cmoe]'
    python scripts/convert_g1.py --headless --force
    python scripts/train.py --task G1-CMoE-Velocity-Rough-v0 --headless --num_envs 4 --max_iterations 1

Verify standing pose and joint control before scaling. Logs are timestamped,
with agent_config.json, env_config.yaml and CMoE checkpoints.

    python scripts/train.py --task G1-CMoE-Velocity-Rough-v0 --headless --num_envs 4 --max_iterations 1 --checkpoint /path/to/model.pt
    python scripts/play.py --task G1-CMoE-Velocity-Rough-v0 --num_envs 1 --checkpoint /path/to/model.pt

Resume iterations are additional iterations. Inspect actual stability during
playback; a saved checkpoint alone is not success.
