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

The `*_selfpad` geoms are declared in `piplus.xml` under the
`self_collision_pad` class, the way the reference declares its own — they are
always in the spec. `get_piplus_robot_cfg(self_pads=True)` is what makes them
live, by adding `SELF_PAD_COLLISION` to put them on contype/conaffinity **bit
2**, so they can only ever touch each other: never the terrain, never the ball,
never the real (bit 1) collision geoms. They are what the multi task's
`self_collision` sensor scores. Because a pad cannot reach the ground, its size
is free — which is the point. Without `self_pads`, `FULL_COLLISION`'s
`disable_other_geoms` leaves them at contype 0 and they cost nothing, which is
how velocity and path see them.

> ⚠️ **A pad the same size as its collision geom is not a bug.** The pad set's
> first job is to *select* which link pairs count as a self-collision; only where
> near-misses matter is the copy enlarged — torso 1.2×, hands 2×, feet 1.3×, the
> rest 1:1, and the upper arms not padded at all. The MJCF says the same next to
> the class definition.

Collision configs (`FULL_COLLISION`, `LEG_ONLY_COLLISION`, `FEET_ONLY_COLLISION`) use regex matching on geom names ending in `_collision[0-9]`. `SELF_PAD_COLLISION` must be applied **after** `FULL_COLLISION`, whose `disable_other_geoms` default would otherwise zero the pads out again.

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
- **No `fell_over` termination, but a deadline.** `fell_over` stays commented
  out with a warning — firing on the *instant* of a fall means the robot never
  experiences the state it is meant to recover from, and getup is unlearnable.
  The pressure not to fall is the `on_ground` reward (a rate charged while
  tilted past `FALLEN_ANGLE` or below `STANDING_MIN_HEIGHT`, weight `-1.0`, a
  flat -1 per step while down), plus `fallen_too_long`: a real termination that
  fires once the robot has been *continuously* down for `GETUP_TIMEOUT_S`
  (6 s). Falling is survivable, staying down is not. The timer counts
  consecutive down-steps by the same test `on_ground` uses and resets the moment
  the robot is standing, so each fall gets a fresh window; envs that spawn
  fallen get the full one. `is_terminated` is live again at `-200.0` — with
  `time_out` flagged `time_out=True` and excluded by mjlab's `is_terminated`,
  it fires on failed getups and nothing else. It costs `200 × dt = 4.0` once,
  about a second of standing; the forfeited rest of the episode (up to ~75) is
  the real cost, so the term mainly puts a discontinuity at the deadline.
- **"Not fallen" needs height as well as angle.** Upright-only is a hole a
  policy will find: sitting on the floor with a vertical trunk passes the angle
  test, so nothing charges it and it never has to do the hard part. `on_ground`
  therefore also requires the base above `STANDING_MIN_HEIGHT` (0.25 m), and the
  reward gates below multiply in `HEIGHT_GATE` (0.29 → 0.23 m). All of these
  derive from `STANDING_HEIGHT` and must move together.
  **`STANDING_HEIGHT` is 0.34, not the keyframe's 0.39.** 0.39 is
  `default_root_state` z, measured by FK with the feet flat, and the actuators do
  not hold it: from a keyframe spawn under a zero action the base sags to 0.333
  within ten steps (mean joint error 0.05 rad, calf off by 0.14) and the robot
  topples within about three seconds — zero action is not a standing controller,
  so 0.333 is a floor on the standing height, not a measurement of it. Held
  actively upright it reaches ~0.357. With 0.39 as the target a visibly standing
  robot failed `getup_success` and collected ~85% of the height gate; at 0.34
  both height terms saturate at 1.000 there. Reference points, base link above
  its own env's ground: ~0.357 held upright, 0.333 settled, ~0.17 sitting with
  the legs folded, 0.10 flat. If a policy that is visibly walking starts showing
  a high `Metrics/fallen`, `STANDING_MIN_HEIGHT` is too high for its gait.
- **No end pose.** `variable_posture` already owns the standing pose; getup
  stops at "upright and off the floor" and hands over.
- **Height is paid twice: base and head.** `head_height` is `getup_height`
  again with `head_yaw_link` and `STANDING_HEAD_HEIGHT` (0.53 = `STANDING_HEIGHT`
  + the rigid 0.198 m head-over-base offset). Ported from the reference, which
  pays pelvis *and* head and deletes its orientation term outright on the
  grounds that both being high already implies an upright trunk. Here it is an
  addition, not a replacement — `getup_orientation` and `upright` stay. It buys
  about +0.42/step of standing-vs-sitting gap, and it is what prices a *folded*
  posture (tall at the base, short at the head); against pure sitting the base
  term is still the stronger discriminator, since a sitting robot's trunk is
  vertical and its head is already 0.37 of the 0.53. The reference runs the
  pelvis at 2:1 over the head, and so does this env.
- **The penalty is gated, the shaping is not.** `getup_orientation` and
  `getup_height` are ungated on purpose: gating them on "is fallen" would make
  the reward step *down* the instant the robot stands up.
- **Walking terms are gated the other way.** `GatedRewardTermCfg` (in
  `getup.py`) wraps a term and fades it out with tilt, linearly in `cos(tilt)`
  between its two angles — a hard cutoff would be a cliff in the return, which a
  policy will go and sit on. `pose` uses the tight `POSTURE_GATE` (20°→40°): the
  standing pose is the wrong target for a robot on the floor, which has to bend
  where posture says not to. `foot_gait`, `feet_phase` and `foot_slip` use
  `GAIT_GATE` (30°→`FALLEN_ANGLE`), so they are off exactly where `on_ground`
  starts charging; both are multiplied by the `HEIGHT_GATE` ramp above — face down the gait clock means nothing, and the slip penalty
  would charge for the ground contact the arms need to push off with. The inner
  term is still *evaluated* while gated (`feet_phase` and `feet_slip` are
  stateful), just multiplied by zero. It does not compose with
  `MultiRewardTermCfg`: a term is gated or command-weighted, not both.
- **Penalties ramp in.** Following the reference's curriculum: `action_rate_l2`,
  `joint_acc_l2`, `body_ang_vel`, `angular_momentum`, `foot_slip` and
  `foot_too_close_to_base` all start reduced (or at zero) and reach their configured values
  by ~1500 iterations, via `reward_weight` terms in the curriculum. Standing up
  from flat is violent, whippy and self-touching; charged full price from step 0
  these make lying still the cheaper option. Each reward carries a `# -> value`
  comment naming its endpoint. `joint_pos_limits` (-1.0) and `self_collisions`
  (-0.1) are deliberately flat: both read ~0 at a settled standing pose, so
  there is no early-training tax to curriculum away.
- **Foot placement is fenced.** `foot_distance` (-1.0, flat past 0.40 m between
  the feet) and `foot_too_close_to_base` (ramped to -0.5 by 900 iterations, fires
  inside 0.15 m) are the reference's two flat geometric penalties. They price
  standing shapes the height terms would otherwise pass: a straddle wide enough
  to be stable without extending the legs, and a foot tucked under the pelvis.
  At the settled pose the measurements are 0.12-0.17 m and 0.31 m, so both read
  zero with margin.
- **Spawns.** `reset_fallen_or_standing` replaces the old `reset_base` +
  `reset_robot_joints` pair. `FALL_PROBABILITY` of envs start on the floor at a
  uniformly random orientation with the default pose plus Gaussian joint noise
  (not uniform over the joint range — that samples mostly unreachable tangles).
- **Camera.** `camera_link_collision0` sits ~5 mm proud of the head shell, so a
  faceplant hits it first; `camera_ground_impact` charges force-graded contact
  heavily. Never grow `head_pitch_link_collision0` — burying
  `camera_link_collision0` inside the head shell stops the penalty firing.
- **Self-collision is scored on the pads, with no `secondary`.** A secondary
  *pattern* silently collapses to one geom — `ContactSensorCfg.secondary_policy`
  defaults to `"first"` — so matching `.*_selfpad` on both sides scores every pad
  against the base pad alone and reads ~0 however folded the robot is. Dropping
  the secondary is correct here, not a shortcut: bit-2 isolation already means a
  pad's only possible contact partner is another pad.
- **`nconmax` is 120, not 45.** Overflow is silent and looks like sinking through
  the floor. 256 fallen envs peak at ~1.6k contacts total against a 120×256
  allocation, so there is ample headroom.

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
