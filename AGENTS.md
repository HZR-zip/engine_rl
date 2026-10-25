# Repository Guidelines

## Project Structure

This repository is an Isaac Lab extension for T800 velocity-tracking reinforcement learning. The installable Python package lives in `source/t800_rl/t800_rl/`; the registered task configuration is under `tasks/locomotion/velocity/`. Use `scripts/train.py` and `scripts/play.py` for training and checkpoint evaluation, and `scripts/convert_t800.py` to convert the robot URDF. Robot descriptions, meshes, and upstream policy/configuration artifacts are under `assets/t800/`. Generated USD files and run outputs are local artifacts, not source code.

## Build and Development

Use the project’s Isaac Lab environment (documented in `README.md`), then install the extension in editable mode:

```bash
python -m pip install -e .
python scripts/convert_t800.py --headless
python scripts/train.py --task T800-Velocity-Flat-v0 --headless --num_envs 16 --max_iterations 1
python scripts/play.py --task T800-Velocity-Flat-v0 --checkpoint /path/to/model.pt --num_envs 1
```

The short training invocation is a startup smoke test, not a quality evaluation. Set `T800_USD_PATH` when the converted model is not at `assets/t800/t800.usd`. Training logs and checkpoints are written under `logs/rsl_rl/`.

## Coding Style

Follow the existing Python style: four-space indentation, descriptive `snake_case` names, and task-specific configuration in the task package rather than duplicated in launcher scripts. Isaac Sim-dependent imports may need to remain after `AppLauncher` starts the application. Keep asset paths relative to the repository or configurable; do not embed machine-specific paths. No formatter or linter is configured in `pyproject.toml`.

## Testing

There is no dedicated automated test suite or coverage threshold currently configured. For changes to task registration or configuration, run the one-iteration headless training command above in the supported Isaac Lab environment. For robot or actuator changes, also convert/load the USD and verify a single robot’s standing pose and joint control before scaling up environments. Hardware/GPU availability affects whether simulation can run.

## Commits and Pull Requests

The available Git history contains only an initial commit, so no established commit-message convention can be inferred. Use a short imperative subject (for example, `Clarify T800 actuator configuration`). Pull requests should describe the behavior/configuration change, identify the Isaac Lab environment used, include smoke-test results, and attach screenshots or logs when they help demonstrate simulation behavior.

## Asset and Configuration Care

Preserve upstream license and attribution files. Treat the included MNN walking policy as a deployment/reference artifact; `scripts/play.py` expects an RSL-RL checkpoint. Mark placeholder model, actuator, and reward assumptions clearly, and validate robot specifications before presenting a configuration as reproduction-ready.
