# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a [mjlab](https://github.com/mujocolab/mjlab) project for training RL policies for the PiPlus humanoid robot. It uses MuJoCo Warp (GPU-accelerated MuJoCo) with a manager-based API similar to Isaac Lab, and RSL-RL for PPO training.

## Commands

Always use `uv run`, not `python` directly.

```sh
# Train — launch as a background process via Bash run_in_background=true (NOT tmux), auto-select a free GPU via nvidia-smi
# After launch, poll wandb/ dir for new run-YYYYMMDD_HHMMSS-* directories
# (verify timestamp matches current time, as runs take time to initialize).
# Inside the dir, find the run-<id>.wandb file — the ID is in the filename.
# Read the first few lines of that file to find a path like
# logs/rsl_rl/piplus_velocity/2026-04-12_23-56-38 — the last segment is the
# run name. Then prepend a line to runs.log (newest first):
# <name>: <id>: <short summary>
uv run train Mjlab-Velocity-PiPlus --env.scene.num-envs 4096 --gpu-ids '[GPU]'

# Play/evaluate a trained policy — launch as a background process via Bash run_in_background=true (NOT tmux), pick a free GPU via nvidia-smi
# Export CUDA_VISIBLE_DEVICES to a single GPU before launching.
# Prefer unused GPUs; if all busy, pick one with low utilization and free memory.
# When playing multiple runs, always run them SEQUENTIALLY — wait for each to
# finish before starting the next, not in parallel.
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


### Task registration

Tasks are registered in `src/mjlab_piplus/tasks/__init__.py` via `register_mjlab_task()`. The package entry point `mjlab.tasks` (declared in `pyproject.toml`) causes mjlab's CLI to discover the task automatically. The registered task ID is `"Mjlab-Velocity-PiPlus"`.

### Robot definition (`piplus_constants.py`)

Three motor series are used:
- **3536** — head joints (still commented out in the MJCF and in the actuator list)
- **4438** — arm joints, enabled via `get_piplus_robot_cfg(with_arms=True)`
- **5036** — leg joints (hip pitch/roll, thigh, calf, ankle pitch/roll), with per-joint tuned PD gains

Each actuator series uses `BuiltinPositionActuatorCfg` with built-in action lag controlled by `ACTUATOR_LAG_MIN` and `ACTUATOR_LAG_MAX` constants (same lag applied to all actuators). The `get_piplus_robot_cfg()` function assembles the full `EntityCfg` with the MuJoCo spec, collision model, and actuator list.

> ⚠️ **`with_arms` changes the DoF count (12 → 20)** and therefore the action
> dim and the joint blocks of the observation. Only the **multi** task uses it.
> The velocity and path tasks must stay at 12 because `mdp/symmetry.py` mirrors
> their observations by hard-coded index. `get_spec(with_arms=False)` deletes the
> arm joints and their actuators from the spec, so the arms stay welded to the
> torso exactly as before.

`get_spec(collision_inflate=...)` scales named collision geoms — see
`GETUP_COLLISION_INFLATE`, the "bigger collision model" hack the getup task uses.
Adding a link to that map can create *permanent* self-collisions (the thighs do,
against the trunk); re-run the pose sweep described in its docstring before
changing it.

Collision configs (`FULL_COLLISION`, `LEG_ONLY_COLLISION`, `FEET_ONLY_COLLISION`) use regex matching on geom names ending in `_collision[0-9]`.

### Environment config (`piplus_velocity_env_cfg.py`)

The function `piplus_velocity_env_cfg(play=False)` builds a `ManagerBasedRlEnvCfg`. Key design choices:

- **Actor obs**: base angular velocity, projected gravity, joint pos/vel relative to default, last action, gait phase (sin/cos), velocity command. The sensor-derived terms (ang vel, gravity, joint pos/vel) are noised; actions, phase, and command are not.
- **Critic obs**: actor obs + base linear velocity, foot height (from `TerrainHeightSensorCfg`), foot air time, foot contact, foot contact forces. No noise.

> ⚠️ **Symmetry depends on the actor obs layout.** The left-right mirror in
> `symmetry.py` (`_transform_obs` / `_transform_obs_common`) is hard-coded by
> index, and each term has its own mirror rule (swap left/right joints, plus a
> sign flip for axial vs. polar quantities — e.g. ang-vel `[-1,1,-1]`, gravity
> `[1,-1,1]`, command `[1,-1,-1]`, phase half-period shift `[-1,-1]`). **Any
> change to the actor obs — term order, dimensions, or what a term measures —
> requires updating `_transform_obs` and `check_symmetry.py` to match.** The
> **critic obs does NOT need a mirror transform**: under data augmentation the
> mirrored sample reuses the original's value/return (`V(mirror(s)) == V(s)`),
> so the value function is only evaluated on the original obs.
- **Sensors**: `ContactSensorCfg` for feet ground contact (ankle roll link subtrees vs. terrain), self-collision sensor, terrain height scan anchored at foot sites.
- **Curriculum**: velocity command ranges ramp up in two stages, from step 0 to step 800 × 24 (`command_vel`).
- **Terminations**: time out (20 s episodes), orientation > 45° (`bad_orientation`). The root-height termination is currently commented out.
- **Simulation**: 5 ms timestep, decimation 4 (→ 50 Hz policy), MuJoCo Warp with 10 solver iterations.
- In `play` mode: curriculum disabled, episode length unbounded, extra air-time metrics added. Push events stay active.

### Getup (fall recovery), in the multi task only

Terms live in `src/mjlab_piplus/tasks/mdp/getup.py` and are wired into
`piplus_multi_env_cfg.py`. Design, and why each piece is the way it is:

- **No getup command.** The terms are unconditional, so recovery is available
  under every command type rather than being something the operator requests.
- **No `fell_over` termination.** It is commented out with a warning — with it
  on, the robot never experiences the state it is meant to recover from and
  getup is unlearnable. The pressure not to fall is now the `on_ground` reward
  (a rate charged while tilted past `FALLEN_ANGLE`) instead of ending the
  episode. `is_terminated` is consequently a dead term (only `time_out` remains,
  and that is flagged `time_out=True`).
- **No end pose.** `variable_posture` already owns the standing pose; getup
  stops at "upright and off the floor" and hands over.
- **The penalty is gated, the shaping is not.** `getup_orientation` and
  `getup_height` are ungated on purpose: gating them on "is fallen" would make
  the reward step *down* the instant the robot stands up.
- **Spawns.** `reset_fallen_or_standing` replaces the old `reset_base` +
  `reset_robot_joints` pair. `FALL_PROBABILITY` of envs start on the floor at a
  uniformly random orientation with the default pose plus Gaussian joint noise
  (not uniform over the joint range — that samples mostly unreachable tangles).
- **Camera.** `camera_link_collision0` sits ~5 mm proud of the head shell, so a
  faceplant hits it first; `camera_ground_impact` charges force-graded contact
  heavily. Do not add the head to `GETUP_COLLISION_INFLATE` — that buries the
  camera geom inside the head shell and the penalty stops firing.
- **`nconmax` is 120, not 45.** A robot lying flat peaks at ~50 contacts;
  overflow is silent and looks like sinking through the floor.

Watch `Metrics/getup_success` (latched, should stay near 1.0) against
`Metrics/fallen` (episode fraction spent down, starts around 0.2).

### Viewer GUI (`mdp/push_gui.py`)

mjlab's viser viewer only hands *command terms* a GUI hook
(`CommandTerm.create_gui`), so anything the project wants in the panel has to
ride along on one. `PushGui` (a "Push" folder: six velocity-delta sliders,
±3 m/s linear / ±6 rad/s angular, a heading-frame toggle, and a *Push robot*
button) is built by whichever term owns the panel — `MultiCommand` in the multi
task, `UniformVelocityCommandWithStop` when it runs standalone (velocity and
path). A sub-term under a composite command must skip it (`if
self.joystick_gate is None:`), or the folder appears twice.

The button click arrives on viser's thread, so it only sets a flag; the write
happens in `PushGui.apply`, called from the owning term's `compute`. The push is
the same mass-independent root-velocity delta as the training-time
`push_by_setting_velocity` event. It shows up in `asset.data` only after the
next forward, which is what makes it look like a no-op if you assert on the same
step.

### RL config (`piplus_velocity_rl_cfg.py`)

PPO with asymmetric actor-critic (actor/critic each 512-256-128 MLP, ELU activation). Left-right symmetry is enabled via `PiPlusSymmetry` (data augmentation + mirror loss, coeff 0.5). Logs to W&B project `mjlab_piplus`. 24 steps per env per rollout, up to 6 400 iterations, checkpoint every 50 iterations.

## Editing conventions

- All custom MDP terms (rewards, metrics, curriculums, etc.) go in
  `src/mjlab_piplus/tasks/mdp/` and are wired into the env cfg via
  `pi_mdp.<name>`. Do **not** edit `mjlab/` sources unless the user
  explicitly says to.

## Code style

- Line length: 88 columns.
- Avoid local imports unless necessary for circular import resolution.
- Joint name patterns use regex strings (e.g., `".*_hip_pitch_joint"`) — these are matched by MuJoCo name filters throughout mjlab.
