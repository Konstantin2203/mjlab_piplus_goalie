from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.contact_sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab_piplus.tasks.mdp.rewards import _get_rz

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.metrics_manager import MetricsTermCfg


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class feet_swing_peak_height:
  """Mean peak swing height of the feet, evaluated at landing (meters).

  Tracks the highest point each foot reaches while airborne and, on touchdown,
  latches it as that foot's most-recent swing peak. The metric returns the mean
  latched peak across feet, held between landings (like ``last_air_time``), so
  the per-step time average reflects the typical swing height. This is the same
  quantity ``feet_swing_height`` logs as a side effect, extracted as a
  standalone metric.
  """

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    self.site_names = cfg.params["asset_cfg"].site_names
    n_feet = len(self.site_names)
    self.peak_heights = torch.zeros(
      (env.num_envs, n_feet), device=env.device, dtype=torch.float32
    )
    self.last_peak_heights = torch.zeros_like(self.peak_heights)
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    # Latch the swing peak on touchdown; hold it until the next landing.
    self.last_peak_heights = torch.where(
      first_contact, self.peak_heights, self.last_peak_heights
    )
    # Reset the running peak for feet that just landed.
    self.peak_heights = torch.where(
      first_contact, torch.zeros_like(self.peak_heights), self.peak_heights
    )
    return self.last_peak_heights.mean(dim=-1)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.peak_heights[env_ids] = 0.0
    self.last_peak_heights[env_ids] = 0.0


def _get_filtered_reward(env: ManagerBasedRlEnv, reward_name: str):
  term_cfg = env.reward_manager.get_term_cfg(reward_name)
  return term_cfg.func


def yaw_tracking_error_filtered(
  env: ManagerBasedRlEnv, reward_name: str = "track_angular_velocity"
) -> torch.Tensor:
  """Abs yaw-rate tracking error using the EMA-filtered measurement."""
  r = _get_filtered_reward(env, reward_name)
  return torch.abs(r._last_yaw_cmd - r._yaw_ema)


def yaw_tracking_error_raw(
  env: ManagerBasedRlEnv, reward_name: str = "track_angular_velocity"
) -> torch.Tensor:
  """Abs yaw-rate tracking error against the raw (unfiltered) measurement."""
  r = _get_filtered_reward(env, reward_name)
  return torch.abs(r._last_yaw_cmd - r._last_yaw_raw)


def torso_yaw_swing(
  env: ManagerBasedRlEnv, reward_name: str = "track_angular_velocity"
) -> torch.Tensor:
  """How much raw yaw rate deviates from its EMA — i.e. the high-freq
  component the filter is absorbing. High values suggest torso swinging."""
  r = _get_filtered_reward(env, reward_name)
  return torch.abs(r._last_yaw_raw - r._yaw_ema)


def step_frequency(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """Steps per second, summed across feet.

  For each foot, the most recent completed stride takes
  ``last_air_time + last_contact_time`` seconds; one strike per stride.
  Summing 1/stride over feet gives total foot-strikes per second.
  Zero before any stride has completed.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.last_air_time is not None
  assert sensor.data.last_contact_time is not None
  stride = sensor.data.last_air_time + sensor.data.last_contact_time
  freq_per_foot = torch.where(
    stride > 0.0, 1.0 / stride.clamp(min=1e-6), torch.zeros_like(stride)
  )
  return freq_per_foot.sum(dim=-1)


def current_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """Mean ongoing air time across feet and envs (seconds)."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.current_air_time is not None
  return sensor.data.current_air_time.mean(dim=-1)


def last_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """Mean most recently completed air time across feet and envs (seconds)."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.last_air_time is not None
  return sensor.data.last_air_time.mean(dim=-1)


def base_ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """L2 magnitude of base roll/pitch rates — catches torso wobble."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def joint_pos_deviation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Mean absolute deviation from default joint position, for the selected joint(s)."""
  asset: Entity = env.scene[asset_cfg.name]
  jnt_ids = asset_cfg.joint_ids
  deviation = (
    asset.data.joint_pos[:, jnt_ids] - asset.data.default_joint_pos[:, jnt_ids]
  ).abs()
  return deviation.mean(dim=-1)


def foot_contact_force(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """Mean contact force magnitude across feet (N), averaged over feet."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  # force: [B, N, 3] global frame (netforce reduce)
  return torch.linalg.norm(sensor.data.force, dim=-1).mean(dim=-1)


def foot_tilt(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  axis: str = "roll",
) -> torch.Tensor:
  """Absolute roll/pitch tilt of a single foot site (radians), yaw-invariant.

  Derived from the foot site's projected gravity, so it measures deviation from
  level independent of heading. ``axis="roll"`` uses the lateral (foot-x) tilt,
  ``"pitch"`` the fore-aft (foot-y) tilt — matching the ankle roll/pitch axes.
  Pairs with the ``flat_foot`` reward. Reported as a magnitude so it doesn't
  average toward zero over a symmetric gait.
  """
  asset: Entity = env.scene[asset_cfg.name]
  site_id = asset_cfg.site_ids[0]
  quat = asset.data.site_quat_w[:, site_id]  # (E, 4)
  proj_g = quat_apply_inverse(quat, asset.data.gravity_vec_w)  # (E, 3)
  comp = 1 if axis == "roll" else 0  # roll <- g_y, pitch <- g_x
  return torch.atan2(proj_g[:, comp], -proj_g[:, 2]).abs()


def phase_metric(
  env: ManagerBasedRlEnv,
  frequency: float = 1.75,
  command_name: str = "twist",
) -> torch.Tensor:
  """phase from 0 to 1"""
  period = 1 / frequency
  global_phase = (env.episode_length_buf * env.step_dt) % period / period
  # Zero out phase when the command asks the robot to stand still (stand flag at
  # index 3). Matches the ``phase`` observation.
  command = env.command_manager.get_command(command_name)
  stand = command[:, 3] > 0.5
  global_phase = torch.where(stand, torch.zeros_like(global_phase), global_phase)
  return global_phase


def feet_phase_target(
  env: ManagerBasedRlEnv,
  frequency: float,
  offset: float,
  foot_height: float,
  stance_threshold: float = 0.56,
  rz_shape: str = "cosine",
) -> torch.Tensor:
  """Target swing height ``rz`` for a single foot from the ``feet_phase`` gait.

  Mirrors ``pi_mdp.feet_phase``'s target: zero during stance, a swing-height bump
  centered in the swing window (shape set by ``rz_shape``; see :func:`_get_rz`).
  ``offset`` selects the foot (e.g. 0.0 for the left, 0.5 for the right).
  Intended as a play-mode visualization of the gait target the reward is
  tracking, so keep ``rz_shape`` matched to the ``feet_phase`` reward.
  """
  period = 1.0 / frequency
  global_phase = (env.episode_length_buf * env.step_dt) % period / period  # [B]
  leg_phase = (global_phase + offset) % 1.0
  swing_phase = (leg_phase - stance_threshold) / (1.0 - stance_threshold)
  return torch.where(
    leg_phase < stance_threshold,
    torch.zeros_like(leg_phase),
    _get_rz(swing_phase, swing_height=foot_height, shape=rz_shape),
  )
