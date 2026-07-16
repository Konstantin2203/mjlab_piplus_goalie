from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

from mjlab_piplus.tasks.mdp.path_following.utils import (
  angle_distance,
  heading_from_quat,
)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class path_progress_reward:
  """Rewards progressing along the path by comparing the index of the closest
  waypoint to the previous step."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.prev_closest_waypoint_indices = torch.zeros(
      env.num_envs, dtype=int, device=env.device
    )
    self.last_command_counter = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    root_position = asset.data.root_link_pos_w[..., :2]

    is_new_command = command.command_counter != self.last_command_counter
    self.prev_closest_waypoint_indices[is_new_command] = 0  # reset for new paths
    self.last_command_counter = command.command_counter.clone()

    # Compute closest waypoint index
    waypoints = command.paths[..., :2]  # (num_envs, num_waypoints, 2)
    dists = torch.norm(
      waypoints - root_position.unsqueeze(1), dim=-1
    )  # (num_envs, num_waypoints)
    closest_waypoint_indices = torch.argmin(dists, dim=-1)  # (num_envs,)

    progress = closest_waypoint_indices - self.prev_closest_waypoint_indices
    env.extras["log"]["Metrics/mean_path_progress"] = torch.mean(
      progress.float()
    ).item()

    self.prev_closest_waypoint_indices = closest_waypoint_indices.clone()

    # Reward is the progress along the path (can be negative if going backwards)
    reward = progress.float()
    return reward


def close_to_path_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  std: float = 0.2,
) -> torch.Tensor:
  """Rewards being close to the path via a Gaussian on distance to the path."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  root_position = asset.data.root_link_pos_w[..., :2]
  waypoints = command.paths[..., :2]  # (num_envs, num_waypoints, 2)
  dists = torch.norm(waypoints - root_position.unsqueeze(1), dim=-1)
  closest_dists = torch.min(dists, dim=-1).values  # (num_envs,)
  reward = torch.exp(-closest_dists / std**2)
  env.extras["log"]["Metrics/mean_close_to_path_reward"] = torch.mean(reward).item()
  return reward


def align_with_path_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  std: float = 0.2,
) -> torch.Tensor:
  """Rewards aligning the heading with the path tangent."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  root_position = asset.data.root_link_pos_w[..., :2]
  root_quat = asset.data.root_link_quat_w  # (num_envs, 4)
  waypoints = command.paths[..., :2]  # (num_envs, num_waypoints, 2)
  dists = torch.norm(waypoints - root_position.unsqueeze(1), dim=-1)
  closest_indices = torch.argmin(dists, dim=-1)  # (num_envs,)
  is_last_waypoint: torch.Tensor = closest_indices == (waypoints.shape[1] - 1)
  closest_indices -= is_last_waypoint.long()
  closest_waypoints = waypoints[
    torch.arange(env.num_envs), closest_indices
  ]  # (num_envs, 2)
  next_waypoints = waypoints[
    torch.arange(env.num_envs), closest_indices + 1
  ]  # (num_envs, 2)
  path_tangents = next_waypoints - closest_waypoints  # (num_envs, 2)
  path_headings = torch.atan2(
    path_tangents[..., 1], path_tangents[..., 0]
  )  # (num_envs,)

  headings = heading_from_quat(root_quat)  # (num_envs,)

  abs_angle_diff = torch.abs(angle_distance(headings, path_headings))  # (num_envs,)
  reward = torch.exp(-abs_angle_diff / std**2)
  env.extras["log"]["Metrics/mean_abs_angle_diff"] = torch.mean(abs_angle_diff).item()
  env.extras["log"]["Metrics/mean_align_with_path_reward"] = torch.mean(reward).item()
  return reward


def goal_reached(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  threshold: float = 0.1,
) -> torch.Tensor:
  """Rewards reaching the goal: distance to the final waypoint below threshold."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  root_position = asset.data.root_link_pos_w[..., :2]
  waypoints = command.paths[..., :2]  # (num_envs, num_waypoints, 2)
  final_waypoints = waypoints[..., -1, :]  # (num_envs, 2)
  dists = torch.norm(final_waypoints - root_position, dim=-1)  # (num_envs,)
  reached = dists < threshold
  reward = reached.float()
  env.extras["log"]["Metrics/goal_reached"] = torch.mean(reward).item()
  return reward
