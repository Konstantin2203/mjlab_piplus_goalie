from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class track_angular_velocity_filtered:
  """Reward yaw-rate tracking against a low-pass filtered base angular velocity.

  The commanded yaw rate is compared against an EMA of the measured yaw rate so
  that short oscillations (e.g. torso swing) do not yield reward. XY angular
  velocity is still penalized instantaneously.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    tau: float = cfg.params["tau"]
    dt = env.step_dt
    self._alpha = dt / (tau + dt)
    self._yaw_ema = torch.zeros(env.num_envs, device=env.device)
    self._last_yaw_raw = torch.zeros(env.num_envs, device=env.device)
    self._last_yaw_cmd = torch.zeros(env.num_envs, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    tau: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del tau
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_ang_vel_b
    self._yaw_ema = (1.0 - self._alpha) * self._yaw_ema + self._alpha * actual[:, 2]
    self._last_yaw_raw = actual[:, 2]
    self._last_yaw_cmd = command[:, 2]
    z_error = torch.square(command[:, 2] - self._yaw_ema)
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    return torch.exp(-(z_error + xy_error) / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    self._yaw_ema[env_ids] = 0.0


class upright_relaxed:
  """Like mjlab's ``upright`` but with a cubic exponent for a flatter plateau.

  Uses ``exp(-|tilt|^3 / std^2)`` instead of ``exp(-|tilt|^2 / std^2)`` so small
  tilts are barely penalized and large ones fall off sharply. Useful when
  natural gait-induced torso sway shouldn't be punished.

  Assumes flat ground (no terrain sensor support).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg, env

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    if asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    else:
      body_quat_w = asset.data.root_link_quat_w
    gravity_w = asset.data.gravity_vec_w
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    tilt_cubed = xy_squared * torch.sqrt(xy_squared.clamp(min=1e-12))
    return torch.exp(-tilt_cubed / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids


def alive(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  return ~env.termination_manager.terminated


def feet_air_time_zero_vel(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)  # number of feet in the air
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      standing = command[:, 3] > 0.5  # stand flag
      reward *= (~standing).float()
      # Standing still: reward max value (zero support goal).
      reward[standing] = 1.0
  return reward




def feet_gait(
  env: ManagerBasedRlEnv,
  frequency: float,
  offset: list[float],
  threshold: float,
  command_name: str,
  sensor_name: str,
) -> torch.Tensor:
  period = 1 / frequency
  sensor: ContactSensor = env.scene[sensor_name]
  is_contact = sensor.data.current_contact_time > 0
  global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
  offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(
    1, -1
  )
  leg_phase = (global_phase + offsets) % 1.0
  is_stance = leg_phase < threshold
  reward = (is_stance == is_contact).float().mean(dim=1)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      # Gate off when the robot should stand still (stand flag at index 3).
      scale = (command[:, 3] < 0.5).float()
      reward *= scale
  return reward


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    # Gate off when the robot should stand still (stand flag at index 3).
    active = (command[:, 3] < 0.5).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


class feet_swing_height_fixed:
  """Reward reaching a fixed swing height, evaluated at landing.

  Gives a reward of 1 per foot whose peak height during the swing phase was
  at or above ``target_height``, granted at the moment of touchdown.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    # Gate off when the robot should stand still (stand flag at index 3).
    active = (command[:, 3] < 0.5).float()
    reached = (self.peak_heights >= target_height).float()
    reward = torch.sum(reached * first_contact.float(), dim=1) * active
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return reward


def _get_rz(
  phase01: torch.Tensor, swing_height: float, shape: str = "cosine"
) -> torch.Tensor:
  """Target foot height over a normalized gait phase in [0, 1).

  Both shapes rise from 0 to ``swing_height`` and back to 0 with the peak at
  phase 0.5:

  - ``"cosine"`` (default): a raised cosine-squared bump
    ``h * cos(pi * (t - 0.5))**2`` (== ``h * sin(pi * t)**2``): 0 at both ends,
    peak ``h`` at phase 0.5. Smooth (zero slope at both ends) and symmetric.
  - ``"parabola"``: the original cubic-Bezier bump (two half-cycles), kept for
    backward compatibility.
  """
  if shape == "cosine":
    return swing_height * torch.cos(torch.pi * (phase01 - 0.5)) ** 2
  if shape == "parabola":

    def bezier(y0: float, y1: float, t: torch.Tensor) -> torch.Tensor:
      return y0 + (y1 - y0) * (t**3 + 3.0 * t**2 * (1.0 - t))

    stance = bezier(0.0, swing_height, 2.0 * phase01)
    swing = bezier(swing_height, 0.0, 2.0 * phase01 - 1.0)
    return torch.where(phase01 <= 0.5, stance, swing)
  raise ValueError(f"Unknown rz shape {shape!r}; expected 'cosine' or 'parabola'.")


class feet_phase:
  """Reward feet tracking a gait-phase-aligned swing-height target.

  Port of ``_reward_feet_phase`` from Flova's mujoco_playground piplus task,
  with two adaptations:

  - The target height (see :func:`_get_rz`) is zero during stance and a
    swing-height bump centered in the swing window ``[stance_threshold, 1)``;
    ``rz_shape`` selects the bump shape (``"cosine"`` default, or ``"parabola"``).
  - The foot height is measured as the **minimum z over the foot's collision
    geoms** (its lowest point / ground clearance) rather than a single foot
    site. A single site can be raised just by tilting the ankle, which the
    policy exploits to "reach" the target without lifting the foot; the lowest
    collision point only rises when the whole foot actually leaves the ground,
    so it can't be gamed by rotation.

  ``foot_geom_names`` is a list of geom-name tuples, one entry per foot, in the
  same order as ``offset`` (e.g. right foot first, left foot second).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.asset_name = cfg.params["asset_cfg"].name
    # Resolve each foot's collision geoms once, preserving the given foot order.
    self.geom_id_groups: list[list[int]] = []
    for names in cfg.params["foot_geom_names"]:
      sc = SceneEntityCfg(self.asset_name, geom_names=tuple(names))
      sc.resolve(env.scene)
      self.geom_id_groups.append(list(sc.geom_ids))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    frequency: float,
    offset: list[float],
    foot_height: float,
    command_name: str,
    foot_geom_names: list[tuple[str, ...]],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    baseline_height: float = 0.0,
    stance_threshold: float = 0.56,
    std: float = 0.1,
    rz_shape: str = "cosine",
  ) -> torch.Tensor:
    del foot_geom_names  # resolved to geom ids in __init__.
    command = env.command_manager.get_command(command_name)
    assert command is not None
    asset: Entity = env.scene[asset_cfg.name]

    period = 1.0 / frequency
    global_phase = (env.episode_length_buf * env.step_dt) % period / period  # [B]
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype)
    leg_phase = (global_phase.unsqueeze(1) + offsets.unsqueeze(0)) % 1.0  # [B, N]

    # Target height is zero during stance (leg_phase < stance_threshold) and a
    # cubic-Bezier bump over the swing window [stance_threshold, 1), remapped so
    # the peak is centered in swing. This matches ``feet_gait``'s stance/swing
    # split (same ``offset`` and ``threshold``).
    swing_phase = (leg_phase - stance_threshold) / (1.0 - stance_threshold)  # [B, N]
    rz = torch.where(
      leg_phase < stance_threshold,
      torch.zeros_like(leg_phase),
      _get_rz(swing_phase, swing_height=foot_height, shape=rz_shape),
    )  # [B, N]

    # Foot clearance = lowest collision geom per foot (tilt-invariant). Subtract
    # the resting clearance ``baseline_height`` (collision geom centers sit above
    # the ground when planted) so a planted foot reads ~0 and the target refers
    # to clearance above the ground.
    geom_z = asset.data.geom_pos_w[:, :, 2]  # [B, num_geoms]
    foot_z = torch.stack(
      [geom_z[:, ids].amin(dim=1) for ids in self.geom_id_groups], dim=1
    )  # [B, N]
    foot_z = foot_z - baseline_height
    error = torch.sum(torch.square(torch.clamp(rz - foot_z, min=0.0)), dim=1)  # [B]

    walk_in_place = (torch.norm(command[:, :3], dim=1) < 0.05).float()
    reward = torch.exp(-error / std**2) * (1.0 + 0.25 * walk_in_place)
    # Gate off when the robot should stand still (stand flag at index 3).
    reward = reward * (command[:, 3] < 0.5).float()
    return reward


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  # Gate off when the robot should stand still (stand flag at index 3).
  active = (command[:, 3] < 0.5).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      # Gate off when the robot should stand still (stand flag at index 3).
      active = (command[:, 3] < 0.5).float()
      cost = cost * active
  return cost


def stand_still(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  diff_angle = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  reward = torch.sum(torch.square(diff_angle), dim=1)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      # Only penalize pose deviation when the robot should stand still (stand
      # flag at index 3). Walk-in-place envs (flag 0) are exempt so they can step.
      scale = (command[:, 3] > 0.5).float()
      reward *= scale
  return reward


class variable_posture:
  """Penalize deviation from default pose with a stand/walk/run-dependent std.

  Copy of mjlab's ``variable_posture``, but the standing regime is selected by
  the command's stand flag (``command[:, 3]`` of ``UniformVelocityCommandWithStop``)
  rather than a velocity ``walking_threshold``. This way a "walk in place" env
  (zero velocity, stand flag 0) gets the looser ``std_walking`` tolerance so it
  can step, while a "stand still" env (flag 1) gets the tight ``std_standing``
  tolerance to hold the default pose. The walking/running split still uses the
  command speed via ``running_threshold``.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])

    # Standing is driven by the command's stand flag, not the speed.
    standing = command[:, 3] > 0.5
    running = (total_speed >= running_threshold) & ~standing
    walking = ~standing & ~running

    standing_mask = standing.float()
    walking_mask = walking.float()
    running_mask = running.float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def feet_clearance_positive(
  *args, **kwargs
):
  r = feet_clearance(*args, **kwargs)
  return 1 - r

def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      # Gate off when the robot should stand still (stand flag at index 3).
      active = (command[:, 3] < 0.5).float()
      cost = cost * active
  return cost


class goal_progress:
  """Reward progress toward the goal: the per-step reduction in Euclidean
  distance to the goal position.

  Normalized so that approaching the goal at ``max_vel`` yields a reward of 1.
  Moving away yields a proportional negative reward (unbounded below). The
  reward is capped at 1 so the policy is not encouraged to exceed ``max_vel``,
  which can be unstable.

  Within ``distance_threshold`` of the goal the reward is always 1, giving the
  policy a reason to hold position at the goal instead of oscillating once no
  more progress can be made.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    term = env.command_manager.get_term(cfg.params["command_name"])
    self.last_dist = torch.linalg.norm(term.command[:, :2], dim=1)
    self.last_counter = term.command_counter.clone()

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    max_vel: float = 1.0,
    distance_threshold: float = 0.5,
  ) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    assert term is not None, f"Command '{command_name}' not found."
    dist = torch.linalg.norm(term.command[:, :2], dim=1)  # [B]
    # Positive when the robot got closer to the goal since the previous step.
    progress = self.last_dist - dist
    # On a goal change (periodic resample or episode reset) the distance delta
    # is meaningless, so zero those steps out.
    goal_changed = term.command_counter != self.last_counter
    progress = torch.where(goal_changed, torch.zeros_like(progress), progress)
    # step_dt already includes decimation (physics_dt * decimation), so the max
    # distance reachable in one policy step is max_vel * step_dt.
    reward = torch.clamp(progress / (max_vel * env.step_dt), max=1.0)
    # Hold reward at the goal so the policy does not oscillate near it.
    reward = torch.where(dist <= distance_threshold, torch.ones_like(reward), reward)
    self.last_dist = dist
    self.last_counter = term.command_counter.clone()
    return reward


def close_to_goal(
  env: ManagerBasedRlEnv,
  command_name: str,
  epsilon: float = 0.5,
) -> torch.Tensor:
  """Reward proximity to the goal position: ``exp(-epsilon * d)``.

  ``d`` is the Euclidean distance to the goal position, read from the base-frame
  goal command ``[dx_b, dy_b, yaw_err]``.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  d = torch.norm(command[:, :2], dim=1)
  return torch.exp(-epsilon * d)


def align_with_goal(
  env: ManagerBasedRlEnv,
  command_name: str,
  pos_coeff: float = 5.0,
  ori_coeff: float = 1.0,
) -> torch.Tensor:
  """Reward reaching the goal pose: ``exp(-pos_coeff * d) * exp(-ori_coeff * |alpha|)``.

  ``d`` is the Euclidean distance to the goal position and ``alpha`` is the
  angular distance to the goal orientation (wrapped yaw error).
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  d = torch.norm(command[:, :2], dim=1)
  alpha = torch.abs(command[:, 2])
  return torch.exp(-pos_coeff * d) * torch.exp(-ori_coeff * alpha)


def orient_towards_goal(
  env: ManagerBasedRlEnv,
  command_name: str,
  coeff: float = 1.0,
  distance_threshold: float = 0.3,
) -> torch.Tensor:
  """Reward facing the goal position while far away.

  When farther than ``distance_threshold`` from the goal, returns
  ``exp(-coeff * |beta|)`` where ``beta`` is the angle between the robot's
  forward (base x) axis and the vector from the base to the goal position
  (i.e. the goal bearing in the base frame). When within
  ``distance_threshold``, returns 1.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  d = torch.norm(command[:, :2], dim=1)
  beta = torch.atan2(command[:, 1], command[:, 0])
  reward = torch.exp(-coeff * torch.abs(beta))
  return torch.where(d <= distance_threshold, torch.ones_like(reward), reward)


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return xy_squared


def flat_foot_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize non-flat feet via each foot site's projected gravity.

  Replacement for ``mdp.flat_orientation_l2``, which only reads the *root link*
  quaternion (ignoring the passed selection) and so penalizes base tilt, not
  foot tilt. This uses the foot sites in ``asset_cfg.site_names`` and sums the
  squared xy projected-gravity components over the feet — matching the
  ``foot_tilt`` metrics. Yaw-invariant.
  """
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.site_quat_w[:, asset_cfg.site_ids]  # (E, n_feet, 4)
  gravity_w = asset.data.gravity_vec_w.unsqueeze(1).expand(
    -1, quat.shape[1], -1
  )  # (E, n_feet, 3)
  proj_g = quat_apply_inverse(quat, gravity_w)  # (E, n_feet, 3)
  return torch.sum(torch.square(proj_g[..., :2]), dim=(-2, -1))
