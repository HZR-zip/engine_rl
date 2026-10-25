# T800 model source

- Repository: https://github.com/engineai-robotics/engineai_robotics_native_sdk
- Revision: `335c60e88772c26c7852d0abd6b3c7439037dd8f`
- Upstream model: `assets/resource/robot/t800/`
- Included: original URDF, its 25 referenced DAE meshes, model and motor configs, walking-example config, and the official walking MNN policy.
- License: see `upstream/LICENSE.txt` (BSD 3-Clause, EngineAI copyright 2025).

`t800.urdf` is a local import copy with only the upstream `package://resource/robot/t800/meshes/` URIs rewritten to relative paths. The original file is preserved under `upstream/assets/resource/robot/t800/urdf/serial_t800.urdf`.

The official walking policy is at `upstream/assets/config/t800/rl_walking_example/policy/t800_260618_165257_30000.mnn` (SHA-256: `cbcb90f86dbb2fde39bdc5a25c8d0530d5c79c7a8f84b1f90863d8c9065b6427`). It is an EngineAI Native SDK deployment artifact, not an Isaac Lab/RSL-RL checkpoint.
