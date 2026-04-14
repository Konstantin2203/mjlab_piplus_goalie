# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a [mjlab](https://github.com/mujocolab/mjlab) project for training RL policies for the PiPlus humanoid robot. It uses MuJoCo Warp (GPU-accelerated MuJoCo) with a manager-based API similar to Isaac Lab, and RSL-RL for PPO training.

## Commands

Always use `uv run`, not `python` directly.

```sh
# Train — launch as a background Task (TaskCreate), auto-select a free GPU via nvidia-smi
# After launch, poll wandb/ dir for new run-YYYYMMDD_HHMMSS-* directories
# (verify timestamp matches current time, as runs take time to initialize).
# Inside the dir, find the run-<id>.wandb file — the ID is in the filename.
# Read the first few lines of that file to find a path like
# logs/rsl_rl/piplus_velocity/2026-04-12_23-56-38 — the last segment is the
# run name. Then prepend a line to runs.log (newest first):
# <name>: <id>: <short summary>
uv run train Mjlab-Velocity-PiPlus --env.scene.num-envs 4096 --enable-nan-guard True --gpu-ids '[GPU]'

# Play/evaluate a trained policy — launch as a background Task, pick a free GPU via nvidia-smi
# Export CUDA_VISIBLE_DEVICES to a single GPU before launching.
# Prefer unused GPUs; if all busy, pick one with low utilization and free memory.
CUDA_VISIBLE_DEVICES=GPU uv run play Mjlab-Velocity-PiPlus --wandb-run-path jgueldenstein/mjlab_piplus/<id>

# Sanity-check with dummy agents (no training needed)
uv run play Mjlab-Velocity-PiPlus --agent zero
uv run play Mjlab-Velocity-PiPlus --agent random

# Format and lint
uv run ruff format
uv run ruff check --fix

# Type check
uv run ty check    # fast
uv run pyright     # thorough
```

## Architecture

### Package layout

- `src/mjlab_piplus/robot/` — Robot definition: MuJoCo XML spec, actuator configs (PD gains, effort limits, delay), collision configs, and home keyframe pose.
- `src/mjlab_piplus/tasks/` — Task registration and configs: environment config (observations, rewards, terminations, curriculum) and RL runner config (PPO hyperparameters).
- `mjlab/` — Upstream mjlab framework (editable install). Contains the manager-based env API, sensors, MDP building blocks, and task infrastructure.
- `rsl_rl/` — Upstream RSL-RL library (editable install). PPO implementation and runner.

### Task registration

Tasks are registered in `src/mjlab_piplus/tasks/__init__.py` via `register_mjlab_task()`. The package entry point `mjlab.tasks` (declared in `pyproject.toml`) causes mjlab's CLI to discover the task automatically. The registered task ID is `"Mjlab-Velocity-PiPlus"`.

### Robot definition (`piplus_constants.py`)

Three motor series are used:
- **3536** — head joints (currently disabled in `PIPLUS_ARTICULATION`)
- **4438** — arm joints
- **5036** — leg joints (hip pitch/roll, thigh, calf, ankle pitch/roll), with per-joint tuned PD gains

Each actuator series has a `BuiltinPositionActuatorCfg` and a `DelayedActuatorCfg` wrapper (2–3 step lag). The `get_piplus_robot_cfg()` function assembles the full `EntityCfg` with the MuJoCo spec, collision model, and actuator list.

Collision configs (`FULL_COLLISION`, `LEG_ONLY_COLLISION`, `FEET_ONLY_COLLISION`) use regex matching on geom names ending in `_collision[0-9]`.

### Environment config (`piplus_velocity_env_cfg.py`)

The function `piplus_velocity_env_cfg(play=False)` builds a `ManagerBasedRlEnvCfg`. Key design choices:

- **Actor obs**: base linear/angular velocity, projected gravity, joint pos/vel relative to default, last action, velocity command. All noised.
- **Critic obs**: actor obs + foot height (from `TerrainHeightSensorCfg`), foot air time, foot contact, foot contact forces. No noise.
- **Sensors**: `ContactSensorCfg` for feet ground contact (ankle roll link subtrees vs. terrain), self-collision sensor, terrain height scan anchored at foot sites.
- **Curriculum**: velocity command ranges ramp up over ~1200 × 24 steps; `air_time` reward weight ramps from 0 to 0.5 over 600 × 24 steps.
- **Terminations**: time out (20 s episodes), orientation > 70°, root height < 0.25 m.
- **Simulation**: 5 ms timestep, decimation 4 (→ 200 Hz policy), MuJoCo Warp with 10 solver iterations.
- In `play` mode: curriculum disabled, push events removed.

### RL config (`piplus_velocity_rl_cfg.py`)

PPO with asymmetric actor-critic (actor/critic each 512-256-128 MLP, ELU activation). Logs to W&B project `mjlab_piplus`. 24 steps per env per rollout, up to 30 000 iterations, checkpoint every 50 iterations.

## Code style

- Line length: 88 columns.
- Avoid local imports unless necessary for circular import resolution.
- Joint name patterns use regex strings (e.g., `".*_hip_pitch_joint"`) — these are matched by MuJoCo name filters throughout mjlab.
