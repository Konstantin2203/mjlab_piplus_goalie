"""Fall recovery: spawn a fraction of envs on the ground and pay them to get up.

Adapted from the bit-bots ``mjlab_playground`` getup task, which is itself a port
of MuJoCo Playground's getup. Two things are deliberately different here, because
getup is folded into the multi-command walking task rather than trained on its
own:

* **No command type of its own.** Getting up is not something the operator asks
  for -- it is what the robot owes you whenever it ends up horizontal. The terms
  below are plain reward terms, live under every command type, and simply read
  as "already satisfied" while the robot is upright.
* **No end pose.** The reference gates a posture reward on being upright and
  drives the robot back to a fixed standing keyframe. Here the walking task
  already owns the standing pose (``variable_posture``) and the velocity command
  that follows it, so a second pose target would only fight them. Getup stops at
  "upright and off the floor" and hands over.

The shaping is deliberately split:

* :func:`on_ground` is **gated** -- it pays a flat penalty only while the robot
  counts as fallen. Together with :class:`fallen_too_long`, which ends the
  episode after a fixed window spent down, this is the pressure that replaces
  the ``fell_over`` termination: falling is survivable, staying down is not.
* :func:`getup_orientation` and :func:`getup_height` are **not gated**. Gating
  them on "is fallen" would make the reward jump *down* at the instant the robot
  finally stands up, which is exactly the wrong sign. Ungated they are monotone
  in how upright and how tall the robot is, so every centimetre off the floor
  pays, and they saturate near 1 once it is standing -- a constant offset that
  the walking terms do not care about.

A third piece is the mirror image of the same idea: :class:`GatedRewardTermCfg`
fades a *walking* term out as the robot tips over. Posture, gait phase, foot
clearance and foot slip all describe a robot on its feet; scored against one
lying on the floor they pay for holding the standing pose it has to leave, for
kicking the ground on the beat, and they charge for the very contact the arms
need to push off with. Getup and walking are the same policy here, so the terms
of each have to switch off where the other's begin.

Both getup shaping terms use a broad, non-saturating shape on purpose. The
task's existing ``upright`` reward is a narrow Gaussian: at 90 degrees over it
is numerically zero and offers no gradient at all, which is fine for keeping a walker upright
and useless for a robot starting flat on its back.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.metrics_manager import MetricsTermCfg
  from mjlab.managers.termination_manager import TerminationTermCfg

__all__ = [
  "GatedRewardTermCfg",
  "upright_gate",
  "fallen_gate",
  "reset_fallen_or_standing",
  "on_ground",
  "getup_orientation",
  "getup_height",
  "ground_contact_cost",
  "fallen_too_long",
  "getup_success",
]

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _base_height(
  env: ManagerBasedRlEnv, asset: Entity, body_ids: Any = None
) -> torch.Tensor:
  """Height of the base (or of a named body) above its own env's ground."""
  if isinstance(body_ids, list):
    height = asset.data.body_link_pos_w[:, body_ids, 2].squeeze(-1)
  else:
    height = asset.data.root_link_pos_w[:, 2]
  return height - env.scene.env_origins[:, 2]


def _fallen_mask(
  env: ManagerBasedRlEnv,
  asset: Entity,
  limit_angle: float,
  min_height: float | None = None,
) -> torch.Tensor:
  """True where the robot is tilted past ``limit_angle`` **or** below ``min_height``.

  ``projected_gravity_b`` is ``(0, 0, -1)`` when upright, so ``-g_z`` is the
  cosine of the tilt angle. The angle test alone is the one the ``fell_over``
  termination used, and on its own it has an obvious hole: sitting on the floor
  with the trunk vertical passes it. That is a local optimum a policy will find
  and stay in -- it is upright, so nothing charges it, and it never has to do
  the hard part. Requiring a minimum base height as well closes it.
  """
  tipped = -asset.data.projected_gravity_b[:, 2] < math.cos(limit_angle)
  if min_height is None:
    return tipped
  return tipped | (_base_height(env, asset) < min_height)


def _upright_gate(
  env: ManagerBasedRlEnv,
  asset: Entity,
  full_angle: float,
  zero_angle: float,
  full_height: float | None = None,
  zero_height: float | None = None,
) -> torch.Tensor:
  """Ramped "is the robot standing" factor, in ``[0, 1]``.

  1.0 below ``full_angle`` of tilt, 0.0 above ``zero_angle``, ramped between; if
  heights are given, multiplied by the same ramp on base height. A hard on/off
  gate at a single threshold is a cliff in the return: at the boundary a
  fraction of a degree (or a millimetre) is worth the whole term, which is a
  thing a policy will happily go and exploit. The tilt ramp is linear in
  ``cos(tilt)`` rather than in the angle -- same monotone shape, one clamp.

  Both factors are needed for the same reason :func:`_fallen_mask` needs both:
  a robot sitting on the floor is upright, and paying it the walking terms is
  what makes sitting worth doing.
  """
  cos_tilt = -asset.data.projected_gravity_b[:, 2]
  full, zero = math.cos(full_angle), math.cos(zero_angle)
  gate = ((cos_tilt - zero) / (full - zero)).clamp(0.0, 1.0)
  if full_height is None or zero_height is None:
    return gate
  height = _base_height(env, asset)
  return gate * ((height - zero_height) / (full_height - zero_height)).clamp(0.0, 1.0)


def upright_gate(
  env: ManagerBasedRlEnv,
  angles: tuple[float, float],
  heights: tuple[float, float] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """The default gate for :class:`GatedRewardTermCfg`: "is the robot standing".

  A thin, config-facing wrapper over :func:`_upright_gate` that takes the
  ``(full, zero)`` pairs the env cfg already keeps as constants
  (``GAIT_GATE``, ``POSTURE_GATE``, ``HEIGHT_GATE``) rather than four loose
  floats.

  Args:
    env: The environment.
    angles: ``(full_angle, zero_angle)`` in radians -- paid in full at or below
      ``full_angle`` of tilt, off at or above ``zero_angle``.
    heights: Optional ``(full_height, zero_height)`` in metres, applied as a
      second multiplicative ramp on base height. Give it wherever the term
      would otherwise pay a robot sitting on the floor, which is upright and so
      passes the angle ramp on its own.
    asset_cfg: Entity whose base tilt and height drive the gate.
  """
  full_angle, zero_angle = angles
  if zero_angle <= full_angle:
    raise ValueError(f"angles must be (full, zero) with zero > full; got {angles}.")
  if heights is not None and heights[1] >= heights[0]:
    raise ValueError(f"heights must be (full, zero) with zero < full; got {heights}.")
  asset: Entity = env.scene[asset_cfg.name]
  full_height, zero_height = heights if heights is not None else (None, None)
  return _upright_gate(env, asset, full_angle, zero_angle, full_height, zero_height)


def fallen_gate(
  env: ManagerBasedRlEnv,
  angles: tuple[float, float],
  heights: tuple[float, float] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Exact complement of :func:`upright_gate`: 1 while down, 0 while standing.

  Pair it with :func:`upright_gate` on the *same* ``angles``/``heights`` to
  split one penalty into two independently-priced copies -- a standing one and
  a fallen one. Because the two gates sum to exactly 1 everywhere, the robot is
  always paying one weight or a linear blend of the two, never neither and
  never both at full price. That is what makes the split safe where a lone
  gated penalty is not: a term gated off while down is *free* while down, and
  free is a thing a policy will go and sit in.
  """
  return 1.0 - upright_gate(env, angles, heights, asset_cfg)


##
# Events.
##


def reset_fallen_or_standing(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  fall_probability: float = 0.3,
  fall_height: float = 0.4,
  fall_velocity_range: float = 0.5,
  joint_pos_std: float = 0.3,
  stand_pose_range: dict[str, tuple[float, float]] | None = None,
  stand_joint_pos_range: tuple[float, float] = (-0.1, 0.1),
  stand_joint_vel_range: tuple[float, float] = (-0.5, 0.5),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset each env either fallen or standing.

  Replaces the usual ``reset_root_state_uniform`` + ``reset_joints_by_offset``
  pair: the standing branch reproduces what those two did, the fallen branch
  drops the robot in from ``fall_height`` with a uniformly random orientation.

  Fallen joint positions are the **default pose plus Gaussian noise**, clamped to
  the soft limits -- not uniform over the joint range. Uniform sampling spends
  most of its mass on configurations the robot cannot reach on the way down (and
  often cannot untangle itself from), which makes the fallen distribution much
  harder than the falls the policy will actually take.

  Args:
    env: The environment.
    env_ids: Environments to reset; ``None`` means all of them.
    fall_probability: Fraction of the reset envs that start on the ground.
    fall_height: Drop height (m) for fallen envs.
    fall_velocity_range: Root and joint velocities for fallen envs are sampled
      uniformly from ``[-range, range]``.
    joint_pos_std: Std (rad) of the Gaussian around the default pose.
    stand_pose_range: Root pose ranges for the standing branch, keyed
      ``x``/``y``/``z``/``roll``/``pitch``/``yaw``. Missing keys are zero.
    stand_joint_pos_range: Uniform joint offset for the standing branch.
    stand_joint_vel_range: Uniform joint velocity for the standing branch.
    asset_cfg: Which entity to reset.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = len(env_ids)
  if n == 0:
    return

  asset: Entity = env.scene[asset_cfg.name]
  device = env.device
  origins = env.scene.env_origins[env_ids]

  default_root_state = asset.data.default_root_state[env_ids]
  default_joint_pos = asset.data.default_joint_pos[env_ids]
  default_joint_vel = asset.data.default_joint_vel[env_ids]
  soft_limits = asset.data.soft_joint_pos_limits[env_ids]

  fall_mask = torch.rand(n, device=device) < fall_probability
  mask = fall_mask.unsqueeze(-1)

  # -- Fallen: dropped from fall_height with a uniformly random orientation. --
  fallen_pos = origins.clone()
  fallen_pos[:, 2] += fall_height
  # A normalized Gaussian 4-vector is uniform on the quaternion sphere, i.e. a
  # uniformly random orientation -- so the policy sees face-down, back-down and
  # everything between in the same proportions.
  fallen_quat = F.normalize(torch.randn(n, 4, device=device), dim=-1)
  fallen_vel = sample_uniform(-fall_velocity_range, fall_velocity_range, (n, 6), device)

  # -- Standing: the previous reset_root_state_uniform behaviour. -------------
  ranges = stand_pose_range or {}
  keys = ("x", "y", "z", "roll", "pitch", "yaw")
  samples = torch.stack(
    [sample_uniform(*ranges.get(key, (0.0, 0.0)), (n,), device) for key in keys],
    dim=-1,
  )
  standing_pos = default_root_state[:, 0:3] + origins + samples[:, 0:3]
  standing_quat = _quat_mul(default_root_state[:, 3:7], _rpy_to_quat(samples[:, 3:6]))
  standing_vel = default_root_state[:, 7:13]

  root_pos = torch.where(mask, fallen_pos, standing_pos)
  root_quat = torch.where(mask, fallen_quat, standing_quat)
  root_vel = torch.where(mask, fallen_vel, standing_vel)

  asset.write_root_link_pose_to_sim(
    torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

  # -- Joints. ---------------------------------------------------------------
  fallen_joint_pos = torch.clamp(
    default_joint_pos + joint_pos_std * torch.randn_like(default_joint_pos),
    soft_limits[..., 0],
    soft_limits[..., 1],
  )
  fallen_joint_vel = sample_uniform(
    -fall_velocity_range, fall_velocity_range, default_joint_vel.shape, device
  )
  standing_joint_pos = torch.clamp(
    default_joint_pos
    + sample_uniform(*stand_joint_pos_range, default_joint_pos.shape, device),
    soft_limits[..., 0],
    soft_limits[..., 1],
  )
  standing_joint_vel = default_joint_vel + sample_uniform(
    *stand_joint_vel_range, default_joint_vel.shape, device
  )

  asset.write_joint_state_to_sim(
    torch.where(mask, fallen_joint_pos, standing_joint_pos),
    torch.where(mask, fallen_joint_vel, standing_joint_vel),
    env_ids=env_ids,
  )


def _rpy_to_quat(rpy: torch.Tensor) -> torch.Tensor:
  """(N, 3) roll/pitch/yaw -> (N, 4) wxyz quaternion."""
  half = 0.5 * rpy
  cr, cp, cy = torch.cos(half).unbind(-1)
  sr, sp, sy = torch.sin(half).unbind(-1)
  return torch.stack(
    [
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
    ],
    dim=-1,
  )


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Hamilton product of two (N, 4) wxyz quaternions."""
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack(
    [
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ],
    dim=-1,
  )


##
# Rewards.
##


def on_ground(
  env: ManagerBasedRlEnv,
  limit_angle: float = math.radians(70.0),
  min_height: float | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """1.0 while the robot is not standing (tipped over or too low), else 0.0.

  Pair with a negative weight. This is the whole reason the ``fell_over``
  termination can be dropped: falling still has to cost something, but it now
  costs a rate rather than ending the episode, so the robot keeps living and
  keeps being asked to get back up.

  ``min_height`` is what stops "sitting up" from counting as recovered: a robot
  parked on its backside is upright by the angle test, collects nothing for the
  effort of actually standing, and pays nothing for staying there. Set it below
  any crouch the walking policy needs and above where the pelvis ends up when it
  is resting on the floor.

  Keep the weight comfortably below what :func:`getup_orientation` and
  :func:`getup_height` pay back on the way up, or lying still beats trying.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return _fallen_mask(env, asset, limit_angle, min_height).float()


def getup_orientation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """How upright the base is, on a flat 0..1 scale.

  1.0 upright, 0.5 on its side, 0.0 fully inverted -- linear in the projected
  gravity, so the gradient is just as strong lying flat as it is nearly
  standing. Ungated by design (see the module docstring).
  """
  asset: Entity = env.scene[asset_cfg.name]
  return 0.5 * (1.0 - asset.data.projected_gravity_b[:, 2].clamp(-1.0, 1.0))


def getup_height(
  env: ManagerBasedRlEnv,
  desired_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for raising the base toward ``desired_height``, saturating there.

  ``(exp(h) - 1) / (exp(h*) - 1)`` with ``h`` clamped at ``h*``: 0 on the floor,
  1 at standing height, and mildly convex in between so the last part of the
  push-up is worth more than the first centimetre. Taken from the reference
  implementation. Ungated by design (see the module docstring).
  """
  asset: Entity = env.scene[asset_cfg.name]
  height = _base_height(env, asset, asset_cfg.body_ids)
  clamped = torch.clamp(height, min=0.0, max=desired_height)
  return (torch.exp(clamped) - 1.0) / (math.exp(desired_height) - 1.0)


def ground_contact_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_scale: float = 50.0,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """Impact cost for a sensor's geoms touching the ground, graded by force.

  Used for the camera: the head bar sticks ~5 mm out of the head's collision
  shell, so a face-first landing hits the lens before anything else. On the real
  robot that is the one part of a fall that is not recoverable, hence a weight
  far heavier than any other contact penalty -- landing on the shoulder, the
  back or the arms is fine, landing on the camera must never be worth it.

  The cost is the contact force normalized by ``force_scale`` and saturated at
  1 per substep, summed over the substeps in the sensor's history (one control
  step's worth at ``history_length == decimation``). Grading by force is what
  separates the two things that should not cost the same: *resting* on the lens
  carries roughly the weight of the head and lands near the bottom of the range,
  while *slamming* into it saturates every substep and costs an order of
  magnitude more. A flat per-substep count would price a slow topple the same as
  a faceplant, and a raw unsaturated force would let one hard hit swamp the
  entire episode return.

  Args:
    env: The environment.
    sensor_name: A :class:`ContactSensor` with ``"force"`` in its fields and a
      non-zero ``history_length``.
    force_scale: Contact force (N) at which one substep costs the full 1.0.
    force_threshold: Deadband (N) below which contact is ignored, so solver
      noise on a resting geom does not accumulate a cost.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  assert data.force_history is not None, (
    f"Sensor '{sensor_name}' needs 'force' in fields and history_length > 0."
  )
  # force_history: [B, N, H, 3] -> per-substep saturated cost, summed over H.
  force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
  above = (force_mag - force_threshold).clamp(min=0.0)
  return (above / force_scale).clamp(max=1.0).amax(dim=1).sum(dim=-1)


##
# Upright gating.
##


class _GatedReward:
  """Evaluates a reward term, then scales it by the cfg's gate function."""

  def __init__(self, cfg: GatedRewardTermCfg, env: ManagerBasedRlEnv) -> None:
    inner = cfg.inner_func
    assert inner is not None, "GatedRewardTermCfg.__post_init__ did not run."
    self._inner = inner(cfg=cfg, env=env) if inspect.isclass(inner) else inner
    gate = cfg.gate_func
    # A class-valued gate is instantiated the way the RewardManager instantiates
    # a class-valued ``func``, so a gate may carry state (a hysteresis latch, a
    # timer) exactly as an inner term may.
    self._gate = gate(cfg=cfg, env=env) if inspect.isclass(gate) else gate
    self._gate_params = dict(cfg.gate_params)

  def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
    # The inner term is evaluated even where the gate is 0: the gait terms are
    # stateful (``feet_phase`` tracks the phase clock, ``feet_slip`` the previous
    # foot position), and skipping them while the robot is down would emit a
    # large spurious value on the step it stands back up.
    value = self._inner(env, **params)
    return value * self._gate(env, **self._gate_params)

  def __getattr__(self, item: str) -> Any:
    # Delegate ``reset`` / ``debug_vis`` so the RewardManager's ``hasattr``
    # probes see exactly what the unwrapped term would expose.
    if item.startswith("_"):
      raise AttributeError(item)
    inner = self.__dict__.get("_inner")
    if inner is None:
      raise AttributeError(item)
    return getattr(inner, item)


@dataclass(kw_only=True)
class GatedRewardTermCfg(RewardTermCfg):
  """A reward term multiplied by a gate function every step.

  The gate is ``gate_func(env, **gate_params) -> Tensor`` in ``[0, 1]``. It
  defaults to :func:`upright_gate`, which fades a term out as the robot tips
  over; supplying a different callable is how a term gets gated on something
  else entirely without touching this class.

  Two kinds of term need the default gate once a fall is something the robot is
  expected to recover from rather than an episode ending:

  * **Terms that describe standing.** The posture reward pays for holding the
    default standing joint pose; charged to a robot lying on its back it pays for
    keeping the very configuration it has to leave to get up. Gate it tightly, so
    it is a walking reward and nothing else.
  * **Terms that describe walking.** Gait phase, foot clearance and foot slip all
    presuppose a robot standing on its feet. Face down they score whatever the
    legs happen to be doing against a gait clock that means nothing, and the slip
    penalty in particular charges for the ground contact the arms and legs need
    in order to push off. Gate those at the fallen angle, so they switch off
    exactly where :func:`on_ground` starts charging.

  A third kind is the mirror image: **penalties that price a motion the robot
  only makes while getting up.** Standing up from flat is necessarily whippy and
  high-momentum, so an ungated angular-velocity penalty is a standing charge on
  the recovery itself. Gating it on tilt alone -- deliberately *without* the
  height ramp -- charges it at full price while upright and waives it while
  down. The height ramp is right for rewards, where it stops a sitting robot
  collecting them, and wrong for penalties, where it would make sitting cheaper
  than standing.

  Pass ``heights`` to :func:`upright_gate` wherever a *reward* would otherwise
  pay a robot that is sitting on the floor: sitting is upright, so the angle
  ramp alone hands it the walking rewards for free and makes the local optimum
  worth staying in.

  Note this does not compose with ``MultiRewardTermCfg`` -- a term can be gated
  or command-weighted, not both. Nothing needs both yet: the gated terms are all
  command-independent.
  """

  gate_func: Any = upright_gate
  """The gate: ``(env, **gate_params) -> Tensor`` in ``[0, 1]``, shape
  ``(num_envs,)``, multiplied into the term's value every step.

  Defaults to :func:`upright_gate`. Any callable with that signature works, and
  a *class* is instantiated with ``(cfg, env)`` the way the RewardManager
  instantiates a class-valued ``func``, so a gate may carry state."""

  gate_params: dict[str, Any] = field(default_factory=dict)
  """Keyword arguments forwarded to ``gate_func`` on every call."""

  inner_func: Any = field(default=None, init=False)
  """The user's reward callable. ``func`` is replaced by the gating wrapper in
  ``__post_init__``; the original is kept here for it to call."""

  def __post_init__(self) -> None:
    if self.func is _GatedReward:
      return  # Already wrapped (e.g. re-constructed from an existing cfg).
    self.inner_func = self.func
    self.func = _GatedReward


##
# Terminations.
##


class fallen_too_long:
  """Terminate an env that has been down continuously for ``timeout_s``.

  This is *not* the ``fell_over`` termination that had to be removed: that one
  fired the instant the robot tipped past an angle, so the policy never spent a
  step in the state it was supposed to recover from and getup was unlearnable.
  This one gives the robot a fixed window to do something about it and only ends
  the episode if nothing happens, which is the difference between "falling is
  fatal" and "staying down is".

  The timer counts *consecutive* steps down and resets the moment the robot is
  standing again by the same test :func:`on_ground` uses, so a robot that gets
  up, walks, and falls again gets a fresh window each time. An env that spawns
  fallen starts the count at zero and so gets the full window as well.

  Pair it with a terminal penalty (``mdp.is_terminated``): without one the
  termination is close to free early in training, when the remaining episode is
  not worth much to a policy that cannot walk yet.
  """

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv) -> None:
    del cfg
    self._fallen_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      self._fallen_steps[:] = 0
    else:
      self._fallen_steps[env_ids] = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    timeout_s: float = 6.0,
    limit_angle: float = math.radians(45.0),
    min_height: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    fallen = _fallen_mask(env, asset, limit_angle, min_height)
    self._fallen_steps = torch.where(
      fallen, self._fallen_steps + 1, torch.zeros_like(self._fallen_steps)
    )
    return self._fallen_steps * env.step_dt >= timeout_s


##
# Metrics.
##


class getup_success:
  """1 once the robot has been upright and at standing height, 0 otherwise.

  Latched for the rest of the episode, so averaged over envs this reads as
  "fraction of episodes in which the robot was standing at some point". Envs
  that spawn standing latch immediately, so read it against the ``fallen``
  metric (which is :func:`on_ground`): the
  number that matters is this staying near 1.0 while ``fall_probability`` is
  high.
  """

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv) -> None:
    del cfg
    self._stood_up = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      self._stood_up[:] = 0.0
    else:
      self._stood_up[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    desired_height: float,
    limit_angle: float = math.radians(20.0),
    height_tolerance: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    upright = ~_fallen_mask(env, asset, limit_angle)
    height = _base_height(env, asset)
    tall_enough = height > desired_height - height_tolerance
    self._stood_up = torch.maximum(self._stood_up, (upright & tall_enough).float())
    return self._stood_up
