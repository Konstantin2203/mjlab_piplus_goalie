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
  `getup.py`) wraps a term and multiplies it by
  `gate_func(env, **gate_params)`, a `[0, 1]` factor evaluated every step. The
  default `gate_func` is `upright_gate`, which fades the term out with tilt,
  linearly in `cos(tilt)` between the two angles of its `angles` pair — a hard
  cutoff would be a cliff in the return, which a policy will go and sit on. Pass
  a `heights` pair for a second ramp on base height. Any callable of that shape
  can be substituted, and a *class* is instantiated with `(cfg, env)` like a
  class-valued `func`, so a gate may carry state.
  `pose` uses the tight `POSTURE_GATE` (20°→40°): the
  standing pose is the wrong target for a robot on the floor, which has to bend
  where posture says not to. `foot_gait`, `feet_phase` and `foot_slip` use
  `GAIT_GATE` (30°→`FALLEN_ANGLE`), so they are off exactly where `on_ground`
  starts charging; both are multiplied by the `HEIGHT_GATE` ramp above — face down the gait clock means nothing, and the slip penalty
  would charge for the ground contact the arms need to push off with. The inner
  term is still *evaluated* while gated (`feet_phase` and `feet_slip` are
  stateful), just multiplied by zero. It does not compose with
  `MultiRewardTermCfg`: a term is gated or command-weighted, not both.
- **Penalties are doubled and oppositely gated.** Each of `body_ang_vel`,
  `joint_pos_limits`, `action_rate_l2`, `self_collisions` and `foot_slip` exists
  twice: the plain name carries this env's standing weight through
  `upright_gate`, and a `*_fallen` twin carries the **reference
  implementation's** weight and curriculum through `fallen_gate`, which is
  `1 - upright_gate`. Both halves use the same band, `REGIME_GATE`
  (`GAIT_GATE` x `HEIGHT_GATE`), so the two gates sum to exactly 1 at every tilt
  and height and the effective weight is a linear blend between the two
  regimes — verified `max|sum-1| = 0` at standing, sitting and flat.
  **That summing property is the point.** A lone gated penalty is *free*
  wherever its gate is 0, and free is a thing a policy will go and sit in; the
  twin means the robot is always paying one weight or a blend, never neither.
  Endpoints, standing -> fallen: `body_ang_vel` -0.25 -> -0.1,
  `joint_pos_limits` -10.0 -> -1.0, `action_rate_l2` -0.05 -> -0.1,
  `self_collisions` -1.0 -> -0.1, `foot_slip` -0.25 -> -1.5. Only the fallen
  copies are ramped (`fallen_*` curriculum terms, reference schedules); the
  standing copies apply to a robot already up, where full price is the point.
  `joint_acc_l2` and `angular_momentum` are **not** doubled — the reference has
  no counterpart to take a fallen value from — so they are gated to standing
  alone and waived while down.
- **Two penalties are gated the same way, minus the height ramp.**
  `body_ang_vel` and `angular_momentum` use `GAIT_GATE` **with no `heights`**.
  Standing up from flat is necessarily whippy and high-momentum, so ungated
  these are a standing charge on the recovery itself; gated, they are waived
  exactly where `on_ground` takes over. The height ramp is deliberately omitted:
  for a *reward* it stops a sitting robot collecting the term, but for a
  *penalty* it would make sitting the cheaper place to be. That asymmetry is the
  rule — gate rewards on tilt **and** height, penalties on tilt alone.
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

### The action map is signed-quadratic, not linear

`QuadraticJointPositionActionCfg` (`mdp/actions.py`) replaces mjlab's linear
`JointPositionActionCfg` in the multi task:

```
target = default + sign(a) * 0.2 * a^2
```

One fixed function, so the deployed policy needs no notion of whether it is
walking or getting up — the reason a conditional action scale was rejected.
It subclasses `JointPositionAction` and overrides only `process_actions`, so the
default offset, encoder-bias correction, `clip` bounds and dict-valued `scale`
all behave as before. `scale` is the **quadratic coefficient**, not a gain.

**Reach is the win.** Every joint reaches its full range inside `|a| = 4.6`
(shoulder_pitch 4.12 rad → 4.54; hip_pitch → 4.09; calf → 3.85; ankle_roll →
1.90), against `|a| = 41` for the same excursion under the old linear 0.1.

> ⚠️ **Resolution is not the win, at the current entropy setting.** Local gain
> is `d(target)/da = 2*0.2*|a| = 0.4|a|`, so the map is finer than the old
> linear 0.1 only for `|a| < 0.25` and finer than 0.5 only for `|a| < 1.25`.
> Measured against where `ewxi1s0n` actually operates, with std ≈ 1.0–1.5:
>
> | joint | walking swing | noise @lin0.1 | @lin0.5 | @quad0.2 |
> |---|---|---|---|---|
> | hip_pitch | 1.98 | 0.147 | 0.735 | **1.308** |
> | calf | 1.49 | 0.111 | 0.555 | **0.857** |
> | ankle_roll | 0.44 | 0.118 | 0.590 | **0.495** |
> | hip_roll | 0.28 | 0.094 | 0.470 | **0.315** |
>
> The quadratic is *better than linear 0.5* on the small joints and *worse* on
> the big ones, and worse than linear 0.1 everywhere. The fine end exists but
> `entropy_coef=0.01` pins std near 1.0–1.5 and the policy cannot sit in it.
> **Lowering `entropy_coef` (and `init_std`) is what unlocks the quadratic's
> resolution**; without that it buys reach only. Std will self-correct somewhat
> — a coarser map makes a given std cost more reward — so treat these as an
> upper bound, not a prediction.

> ⚠️ `action_rate_l2` is charged on **raw** policy output, so under a nonlinear
> map its physical meaning varies with `|a|`: the same raw step moves the joint
> `0.4|a|` times as far, so big excursions are under-charged for smoothness.

### Getup by imitation (`mdp/imitation.py`)

Instead of discovering a stand-up from shaped rewards, the fallen robot is paid
for reproducing what an already-trained getup policy would do from the same
state. The teacher is a checkpoint from the reference env in `mjlab_playground`
(`getup_model.pt` at the repo root); `GETUP_TEACHER_CHECKPOINT` names it.

Two things are bridged, and they are the whole module:

- **Observation.** The teacher's actor is a 66 → 20 MLP taking
  `[base_ang_vel(3) | projected_gravity(3) | joint_pos_rel(20) |
  joint_vel_rel(20) | last_action(20)]`. The first four are the *same mjlab
  functions* this env already uses, so they are rebuilt by calling them
  directly — **not** by slicing this env's actor obs, which has extra terms
  (phase, command) in a different layout and is noised. The fifth is the
  **teacher's own** previous output, kept in a buffer here and zeroed on reset;
  carrying it across an episode boundary would feed it the tail of a different
  getup. The reference runner sets `obs_normalization=False` on its actor, so
  there is no normalizer to apply — the loader raises if a checkpoint has one.
- **Action.** The teacher is a `RelativeJointPositionAction`
  (`target = joint_pos + a*0.6`); this env is default-relative
  (`target = default + f(a)`). Comparing raw action vectors would be
  meaningless. Both are converted to a **joint position target in radians** and
  the reward is `exp(-mean_j (Δq)^2 / std^2)` over that. This also makes the
  term indifferent to this env's action scale or to swapping in a nonlinear map.
  `_processed_actions` is read *before* the encoder-bias subtraction, which is
  where the teacher's expression sits too, so the bias cancels.

Joint order is resolved **by name** (`TEACHER_JOINT_ORDER`), not assumed — it
happens to be the identity for all 20 actuated joints, but a reorder on either
side would otherwise scramble the mapping silently.

> ⚠️ **The teacher's `joint_pos` observation is relative to the TEACHER's
> default pose, not this env's.** The reference robot's home keyframe
> ("walkready") differs from ours on 9 of 20 joints — elbows by 0.7 rad, calves
> by 0.2, hips and ankles by 0.1, RMS 0.235 — so subtracting our default (what
> mjlab's `joint_pos_rel` does) offsets the teacher's input by that much.
> Measured: it changed the teacher's action by mean 0.30 (max 1.55), i.e. a
> 0.179 rad error in the joint target it asks for, and cost ~0.036 m of the
> height its getup reaches. `TEACHER_DEFAULT_JOINT_POS` holds the correct pose.
> Note the consequence: following the teacher drives the robot toward *its*
> standing pose, not ours — which in the multi task pulls against `posture`.

> ⚠️ **Rewards run after the physics step, but the action being judged was
> chosen from the pre-step state.** The teacher's target is therefore cached one
> call earlier, from the state the student actually acted on, rather than
> recomputed at reward time. During a getup the joints move ~0.06 rad per 20 ms
> step, and comparing across that gap scored a *perfect* teacher-following
> oracle **0.618** instead of 0.997. An env's first step after a reset has no
> cached target from its own episode and scores 0.

> ⚠️ `std` is 0.3, and should not go tighter. An untrained policy sits ~0.39 rad
> from the teacher; at `std=0.2` that is a reward of 0.018 with gradient 0.44,
> versus 0.17 and 1.60 at 0.3 — the same exponential-saturation dead zone that
> kills `posture` when its std is too tight.

The term is gated to the fallen half with `fallen_gate`: the teacher's env has
no velocity command and it was trained to stand up and *stop*, so ungated it
would fight the walking controller. `on_ground` is now the mirror image — a flat
`always_one` at weight -1.0 through the same `fallen_gate`, so the "is it
fallen" test is the gate rather than something the term re-derives.

### `Mjlab-StandupImitation-PiPlus` — the imitation validation task

`piplus_standup_imitation_env_cfg.py`. A stripped env whose only job is to
answer "does the teacher bridge work": no commands, no gait shaping, no
penalties, no ball, no curriculum, no terminations but the time-out. The
**imitation reward is the entire objective**, so the return is a direct readout
of how closely the student tracks the teacher, uncontaminated by anything else.

Deliberate differences from the multi task:

- **The imitation term is ungated here.** In the multi task it is confined to
  the fallen half so it cannot fight the walking controller; here there is
  nothing to fight, and gating would blind the reward exactly when the robot
  succeeds and stands up.
- **The three shaping terms come from the reference** at its own weights:
  `torso_height` 2.0, `head_height` 1.0, `orientation` 1.0, all ungated (gating
  them on "is fallen" would step the return *down* the instant the robot stands
  up). They sum to 4.0/step; `IMITATION_WEIGHT` matches that total at 4.0.
- **No `on_ground` and no terminations but the time-out.** Being on the floor is
  priced by the height terms falling toward zero there, which is a gradient
  rather than a flat rate.
- **Every env spawns fallen** (`fall_probability=1.0`).
- **The actor obs is exactly the teacher's 66-vector**, so the student can in
  principle represent the teacher and a poor score is evidence about the bridge
  rather than a missing input.

It carries the multi task's **domain randomization** (all 11 startup terms) and
**observation noise + 1-step delay**, verbatim, so a policy that works here is
answering the same sensing and dynamics question it will face there. No
`push_robot`: that term gates on the stand command and this env has no command
manager to gate on.

> ⚠️ **The teacher does not see any of the noise.** `imitation.py` builds its
> observation from true state, so the target stays a clean function of where the
> robot actually is while the student reproduces it from noisy, delayed sensing.
> A teacher fed the same noise would emit a noisy target — variance in the
> reward with no added information.

That extends to the encoder bias, and it matters. `teacher_biased` defaults
**False**. Teacher getup success (fraction of envs above 0.29 m after 300 steps,
256 envs, teacher-following oracle):

| configuration | final base h | success |
|---|---|---|
| no DR, no obs noise | 0.329 | 1.00 |
| no DR, obs noise on | 0.329 | 1.00 |
| full DR minus `encoder_bias` | 0.327 | 0.98 |
| full DR, `encoder_bias` at the reference's ±0.0698 | 0.310 | 0.85 |
| full DR, teacher sees biased position | 0.292 | 0.65 |
| **full DR, teacher sees true position (default)** | **0.312** | **0.84–0.89** |

Two readings. **Observation noise costs the teacher nothing** — it reads true
state, so only the student is affected. **`encoder_bias` is the entire DR
effect**: it is the one randomization that reaches the teacher's input, this env
runs it at ±0.1 against the ±0.0698 the reference trained on, and a persistent
per-joint offset is not something a policy can average away. Feeding the teacher
the true position recovers most of the loss. The residual (0.84 vs 1.00) is the
physical DR — masses, gains, friction — which the teacher genuinely never saw,
and `getup_model.pt` is an early (iteration 200) checkpoint.

Measured, 128 envs x 300 steps, no DR (values are per-step, i.e. `reward_buf / dt`):

| policy | total/step | imitation value | base height start → end |
|---|---|---|---|
| zero action | 1.233 | 0.0003 | 0.398 → 0.092 |
| random action | 1.216 | 0.0003 | 0.398 → 0.089 |
| **teacher-following oracle** | **7.627** | **0.9967** | 0.397 → **0.348** |

The oracle scores 0.997 of a possible 1.0 on the imitation term, ~2500x above
chance, and the teacher driven through this env's default-relative action term
genuinely stands the robot up (0.09 collapsed → 0.31). The bridge is exact: the
student's commanded target matches the teacher's to 2e-8 rad.

Per-term payout, robot held in each pose (max 4.036/step of shaping):

| state | torso_height | head_height | orientation | shaping total |
|---|---|---|---|---|
| flat on the floor | 0.519 | 0.150 | 0.500 | 1.171 |
| standing upright | 2.000 | 1.000 | 1.000 | 4.036 |

So the shaping alone offers 2.87/step for standing up, against up to 2.0/step
for imitating well. A policy that stands *without* imitating out-scores one that
imitates on the floor — intended, but worth knowing before reading a run.

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
