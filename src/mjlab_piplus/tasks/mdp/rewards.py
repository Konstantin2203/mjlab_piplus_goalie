from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

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
