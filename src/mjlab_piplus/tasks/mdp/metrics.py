from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.contact_sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


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


def base_ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """L2 magnitude of base roll/pitch rates — catches torso wobble."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
