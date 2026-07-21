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
_DEFAULT_FOOT_CFG = SceneEntityCfg("robot", site_names=("l_foot", "r_foot"))


def base_footprint_xy(
  env: ManagerBasedRlEnv, foot_asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_CFG
) -> torch.Tensor:
  """World-frame xy of the humanoid ``base_footprint``.

  Faithful to ros-sports/humanoid_base_footprint: ``base_footprint`` is the
  ground point midway between the two foot frames. In the support-foot frame it
  is placed at ``(non_support.x / 2, non_support.y / 2, 0)``, which in world xy
  is simply the mean of the two feet positions (independent of which foot is the
  support foot). z is dropped (the frame lies on the ground) and its orientation
  is the base yaw — which the path rewards already read from the root quaternion.

  Returns a ``(num_envs, 2)`` tensor.
  """
  asset: Entity = env.scene[foot_asset_cfg.name]
  foot_xy = asset.data.site_pos_w[:, foot_asset_cfg.site_ids, :2]  # (B, n_feet, 2)
  return foot_xy.mean(dim=1)  # (B, 2)


class path_progress_reward:
  """Rewards progress speed along the path, in meters/second.

  Each step it compares the closest fine-path index to the previous step's;
  scaling that index delta by ``point_spacing`` gives meters advanced along the
  path, and dividing by the policy timestep gives a speed in m/s (invariant to
  the path sampling resolution). The reward is capped at ``max_speed`` so there
  is no bonus for exceeding it; it stays negative when moving backwards."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.prev_closest_waypoint_indices = torch.zeros(
      env.num_envs, dtype=int, device=env.device
    )
    self.last_command_counter = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )
    # Set by reset() for envs reset since the last __call__. Needed because the
    # command counter alone does NOT flag an episode reset here: on reset the
    # command manager zeroes command_counter then resamples (+1), and the path
    # command never resamples mid-episode (resampling_time_range ~1e6), so the
    # counter is pinned at 1 every episode. Without this flag a reset would leave
    # prev pointing at the far end of the old path and yield a large negative
    # progress against the new path.
    self._was_reset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      self._was_reset[:] = True
    else:
      self._was_reset[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    foot_asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_CFG,
    max_speed: float = 1.0,
  ) -> torch.Tensor:
    del asset_cfg  # Position comes from the base footprint (feet), not the root.
    command = env.command_manager.get_term(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    root_position = base_footprint_xy(env, foot_asset_cfg)

    # New path if the command resampled OR the env was reset last step.
    is_new_command = (
      command.command_counter != self.last_command_counter
    ) | self._was_reset
    self.prev_closest_waypoint_indices[is_new_command] = 0  # reset for new paths
    self.last_command_counter = command.command_counter.clone()
    self._was_reset[:] = False

    # Compute closest waypoint index
    waypoints = command.paths[..., :2]  # (num_envs, num_waypoints, 2)
    dists = torch.norm(
      waypoints - root_position.unsqueeze(1), dim=-1
    )  # (num_envs, num_waypoints)
    closest_waypoint_indices = torch.argmin(dists, dim=-1)  # (num_envs,)

    # Index delta -> meters along the path (points are spaced by point_spacing)
    # -> divide by the policy timestep to get progress speed in meters/second.
    # This makes the reward invariant to the path sampling resolution.
    progress_idx = closest_waypoint_indices - self.prev_closest_waypoint_indices
    speed = progress_idx.float() * command.cfg.point_spacing / env.step_dt  # m/s
    # Cap the reward: no bonus for exceeding max_speed (still negative if moving
    # backwards, which the policy should be free to correct without a floor).
    # Raw clamped speed (no near-goal saturation): this is telescoping, so it is
    # not farmable (forward gives +delta, backward -delta, a back-and-forth nets
    # ~0). Advancing the final stretch keeps earning progress, so the robot walks
    # all the way to the goal; once stopped there, closest_index is pinned at the
    # end and progress is simply 0 (no penalty, no reward to oscillate for).
    reward = torch.clamp(speed, max=max_speed)
    env.extras["log"]["Metrics/mean_path_progress_mps"] = torch.mean(speed).item()

    self.prev_closest_waypoint_indices = closest_waypoint_indices.clone()

    return reward


def close_to_path_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  foot_asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_CFG,
  std: float = 0.2,
) -> torch.Tensor:
  """Rewards being close to the path via a Gaussian on distance to the path."""
  del asset_cfg  # Position comes from the base footprint (feet), not the root.
  command = env.command_manager.get_term(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  root_position = base_footprint_xy(env, foot_asset_cfg)
  waypoints = command.paths[..., :2]  # (num_envs, num_waypoints, 2)
  dists = torch.norm(waypoints - root_position.unsqueeze(1), dim=-1)
  closest_dists = torch.min(dists, dim=-1).values  # (num_envs,)
  reward = torch.exp(-closest_dists**2 / std**2)
  env.extras["log"]["Metrics/mean_close_to_path_distance"] = torch.mean(closest_dists).item()
  return reward


def align_with_path_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  foot_asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_CFG,
  std: float = 0.2,
) -> torch.Tensor:
  """Rewards aligning the heading with the path tangent."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  # Position from the base footprint (feet midpoint); heading from the base yaw
  # (the base_footprint orientation is the base yaw about vertical).
  root_position = base_footprint_xy(env, foot_asset_cfg)
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
  # Near the goal the path tangent is ill-defined; switch the alignment target
  # from the path tangent to the goal orientation so there is a dense gradient to
  # turn onto the goal heading. This is state-based (a function of yaw error), so
  # it is not farmable and reaches its max only when actually facing the goal.
  goal_angle_diff = command.yaw_err.abs()  # (num_envs,)
  angle_err = torch.where(command.near_goal, goal_angle_diff, abs_angle_diff)
  reward = torch.exp(-angle_err / std**2)
  env.extras["log"]["Metrics/mean_abs_angle_diff"] = torch.mean(abs_angle_diff).item()
  env.extras["log"]["Metrics/mean_align_with_path_reward"] = torch.mean(reward).item()
  return reward


def goal_reached(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  foot_asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_CFG,
  pos_std: float = 0.25,
  yaw_std: float = 0.5,
  success_threshold: float = 0.1,
  success_yaw_threshold: float = 0.2,
) -> torch.Tensor:
  """Dense goal-pose reward: a product of Gaussians on the distance to the final
  waypoint and on the orientation error to the goal yaw.

      exp(-dist^2 / pos_std^2) * exp(-yaw_err^2 / yaw_std^2)

  The product gates orientation by proximity: far from the goal the position
  term is ~0 (so facing the goal yaw is not rewarded and does not fight
  path-following), while near the goal it gives a smooth gradient to both slide
  onto the goal position and turn to the goal heading. ``success_*`` thresholds
  only drive the logged success-rate metric, not the reward."""
  del asset_cfg  # Position comes from the base footprint (feet), not the root.
  command = env.command_manager.get_term(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  root_position = base_footprint_xy(env, foot_asset_cfg)
  final_waypoints = command.paths[..., -1, :2]  # (num_envs, 2)
  dist = torch.norm(final_waypoints - root_position, dim=-1)  # (num_envs,)
  # yaw_err is wrap(goal_yaw - heading), updated by the command each step.
  yaw_err = command.yaw_err.abs()  # (num_envs,)
  reward = torch.exp(-dist**2 / pos_std**2) * torch.exp(-yaw_err**2 / yaw_std**2)
  success = (dist < success_threshold) & (yaw_err < success_yaw_threshold)
  env.extras["log"]["Metrics/goal_reached_reward"] = torch.mean(reward).item()
  env.extras["log"]["Metrics/goal_success_rate"] = torch.mean(success.float()).item()
  env.extras["log"]["Metrics/mean_dist_to_goal"] = torch.mean(dist).item()
  return reward
