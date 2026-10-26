# Work Log and Host Handoff

Keep this file current when making a significant project change so work can
resume after switching machines. Record source revisions, environment
requirements, runnable commands, validation performed, and remaining blockers.
Keep machine-specific paths and credentials out of this file.

## 2026-10-25: Vendor CMoE G1 Isaac Gym Reference

### Change

- Added the authors' G1 training framework under `references/cmoe_isaacgym/`.
- Included the CMoE-modified `legged_gym` and `rsl_rl`, G1 URDF/meshes,
  terrain code, license files, upstream notices, and the original README.
- Pinned the source to upstream commit
  `4575d6ae762adf0043b4eabf1b8224a227d0ce38` from
  `https://github.com/Hoshi-No-Ai/CMoE`.
- See `references/cmoe_isaacgym/UPSTREAM.md` for the source and runtime notes.

### Runtime Setup

This reference tree uses Isaac Gym Preview 4 and must stay in an environment
separate from the Isaac Lab project environment. The authors document Ubuntu
20.04, Python 3.8, PyTorch 1.13.1 with CUDA 11.7, and a compatible NVIDIA
driver. Install Isaac Gym from NVIDIA's package first, then from this repository
root run in the Isaac Gym environment:

```bash
cd references/cmoe_isaacgym
python -m pip install -e rsl_rl
python -m pip install -e legged_gym
python legged_gym/legged_gym/scripts/train.py --task=g1cmoe --alg=cmoe --run_name=baseline
```

For visual policy evaluation, use the upstream play command documented in
`references/cmoe_isaacgym/README.md` after training.

### Validation and Status

- The vendored `legged_gym/` and `rsl_rl/` trees were compared file by file
  with the pinned upstream checkout and matched.
- Python syntax parsing passed for all 43 vendored Python files.
- `git diff --check` passed.
- Isaac Gym training and visual replay have not been run in this workspace.
- This is the original Isaac Gym implementation, not an Isaac Lab port. The
  current T800 task remains a separate Isaac Lab package.
- Training still requires an Isaac Gym Preview 4 installation and a host with a
  compatible NVIDIA GPU/driver.

### Next Steps

1. On a host with the supported Isaac Gym stack, install dependencies and run a
   short `g1cmoe` training startup check.
2. Run the authors' policy playback and retain the command, host environment,
   checkpoint path relative to the run, and result here.
3. Complete the Isaac Lab CMoE policy and PPO auxiliary-loss port after the
   initial G1 task and terrain-sensor startup checks are validated.

## 2026-10-25: Initial G1 Isaac Lab Task Port

### Change

- Added the independent Gym task `G1-CMoE-Velocity-Rough-v0` under
  `source/t800_rl/t800_rl/tasks/locomotion/g1/`.
- Added `scripts/convert_g1.py` to convert the authors' 12-DoF URDF into the
  local generated asset `assets/g1/g1_12dof.usd`.
- Deferred the T800 USD existence check from task-module import so the shared
  package can register the G1 task on hosts that have not generated T800 USD.
- Added flat, stair, gap, and random-grid terrains, an 11x7 yaw-aligned ray
  scan, 12 leg position actions, and the 45-D frame observation with 10-frame
  history expected by the reference policy.
- Configured current Lab RSL-RL PPO as a baseline. This does not yet implement
  the CMoE experts, gating, estimator auxiliary losses, or terrain contrastive
  objective.

### Host Setup and Commands

Use the Isaac Lab environment matching this project's README, install the
extension in editable mode, and convert the URDF on each host:

```bash
python -m pip install -e .
python scripts/convert_g1.py --headless
python scripts/train.py --task G1-CMoE-Velocity-Rough-v0 --headless --num_envs 4 --max_iterations 1
python scripts/play.py --task G1-CMoE-Velocity-Rough-v0 --checkpoint /path/to/model.pt --num_envs 1
```

### Validation and Remaining Work

- `python3 -m compileall -q source/t800_rl/t800_rl scripts/convert_g1.py` and
  `git diff --check` passed. XML parsing confirmed the source URDF has exactly
  the 12 expected actuated leg joints.
- Isaac Lab conversion/training are not yet validated. A config import attempt
  using the known shared Isaac Lab environment stopped because that environment
  lacks `gymnasium`; the system Python also lacks the Isaac Lab runtime. The
  README records that this host has no usable NVIDIA driver.
- On a supported Lab/GPU host, verify conversion, joint order/limits, one
  robot's standing pose, the 77-ray scan, and one PPO iteration.
- Port CMoE routing and estimator/terrain alignment losses into the current
  RSL-RL policy and PPO lifecycle. Keep the vendored Isaac Gym source as the
  reference implementation.

## 2026-10-26: Isaac Lab Agent Skills Installation

- Resolved the supplied develop/skills GitHub page to main/skills and pinned
  upstream commit a9fb50e5388724a680bbc5c1f836917271605a99.
- Added scripts/install_isaaclab_skills.py and ISAACLAB_SKILLS.md so another
  training host can repeat the installation with Python 3 and Git.
- Preserve upstream skill and documentation paths; expose all 24 skill names
  through user-level .agents/skills symlinks. The installer refuses conflicting
  existing skills and reuses matching installations.
- Reinstall from this repository root: python3 scripts/install_isaaclab_skills.py
- Installed and verified all 24 user-level skill links. The upstream validator
  passed: Validated 24 skills. Syntax validation and git diff --check passed.
  The prepared-checkout installation succeeded; a fresh-host network install
  has not been separately exercised. Final installation checks are recorded in
  ISAACLAB_SKILLS.md. These instructions do not install an Isaac Lab runtime
  or establish that the existing G1/CMoE task is ready for training.

## 2026-10-26: G1 Contract, CMoE Networks and RSL-RL Training Port

### Scope

- Followed the user's CPU-only host constraint. No GPU environment changes
  or simulation launch were performed. Tool-environment GPU discovery from
  an earlier turn does not establish the user's machine capability.
- Targeted the existing Isaac Lab 2.3.x / Isaac Sim 5.1 API, with RSL-RL 3.0.1.
  Installed Lab 3.x guidance skills do not automatically change this target.
- Full implementation notes, corrections and reproduction gaps: CMOE_PORT.md.

### Robot and observation alignment

- Selected the author's fixed-upper-body G1 URDF with 12 movable joints;
  corrected pitch/roll/yaw ordering, named action mapping and default USD path.
- Read actuator limits from URDF and applied explicit hip/knee/ankle PD gains.
  USD conversion disables implicit drives and enables self-collisions.
- Implemented shared 527-D actor / 128-D critic observations, newest-first
  history with reset repetition, scaling, and upstream-style input corruption.
- Documented remaining ray/heightfield sampling, terrain, reward, command
  and randomization differences rather than claiming full environment parity.

### Networks and training

- Added CPU-importable t800_rl.cmoe: five expert actors/critics, softmax gate,
  state estimator, terrain autoencoder, prototype/Sinkhorn contrastive loss.
- Added CMoEPPO with separate state/terrain optimizers and episode-safe
  next-state targets. Cached latent rollout inputs keep probability checks stable.
- Added CMoEOnPolicyRunner with all optimizer states, LR, counters, configuration,
  joint order and RNG in atomic checkpoints; wired train/play task selection.
- Training uses unique run directories and exports agent/environment configs.
- Fixed setuptools package discovery; kept task registration explicit so CPU
  imports do not launch/import Isaac Sim. Added the cmoe dependency extra.
- The authors' reference code and license files remain unchanged.

### Validation

- Six CPU tests passed, including actual RSL-RL collection/update/save/load and
  matching the next optimizer update after restore with identical test inputs.
- Syntax/AST checks and git diff --check passed. Wheel build and layout
  verification passed; no generated bytecode files are included.
- CPU one-iteration startup attempt stopped before AppLauncher because the
  selected Python environment cannot import isaaclab. No physical validation.
- Repeat on a host with PyTorch/RSL-RL: python -m pip install -e '.[cmoe]', then
  python -m unittest discover -s tests -p test_cmoe_cpu.py -v.
- On a configured GPU host: convert G1, verify standing/joints/sensors, then
  run the four-environment one-iteration check documented in CMOE_PORT.md.
- Distributed training, original Gym weight conversion, full environment
  parity, long training and paper-metric evaluation remain out of scope here.
