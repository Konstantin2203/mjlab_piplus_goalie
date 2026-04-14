from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

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


def base_ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """L2 magnitude of base roll/pitch rates — catches torso wobble."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
