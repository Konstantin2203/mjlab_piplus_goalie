from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

from mjlab_piplus.tasks.mdp.multi_command import MultiCommand, resolve_goal_term
from mjlab_piplus.tasks.mdp.path_following.path_command import PathCommand
from mjlab_piplus.tasks.mdp.path_following.utils import (
  angle_distance,
  heading_from_quat,
)


def resolve_path_command(
  env: ManagerBasedRlEnv, command_name: str
) -> tuple[PathCommand, torch.Tensor]:
  """Resolve the path command term and its per-env active mask.

  ``command_name`` may name the path command directly (standalone path task) or
  a ``MultiCommand`` that holds it as a sub-command. The mask is all ones in the
  former case and the path type's active mask in the latter.
  """
  term = env.command_manager.get_term(command_name)
  if isinstance(term, PathCommand):
    return term, torch.ones(env.num_envs, device=env.device)
  if isinstance(term, MultiCommand):
    for name, sub in term.terms.items():
      if isinstance(sub, PathCommand):
        return sub, term.active_mask(name).float()
  raise TypeError(
    f"Command '{command_name}' does not resolve to a PathCommand "
    f"(got {type(term).__name__})."
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
    command, _ = resolve_path_command(env, command_name)
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
    # Average only over envs the path command is actually driving: under a
    # MultiCommand the term still runs on every env (to keep its state fresh),
    # so an unrestricted mean would be diluted by velocity/stand envs.
    _, active = resolve_path_command(env, command_name)
    env.extras["log"]["Metrics/mean_path_progress_mps"] = (
      (speed * active).sum() / active.sum().clamp(min=1)
    ).item()

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
  command, _ = resolve_path_command(env, command_name)
  root_position = base_footprint_xy(env, foot_asset_cfg)
  waypoints = command.paths[..., :2]  # (num_envs, num_waypoints, 2)
  dists = torch.norm(waypoints - root_position.unsqueeze(1), dim=-1)
  closest_dists = torch.min(dists, dim=-1).values  # (num_envs,)
  reward = torch.exp(-(closest_dists**2) / std**2)
  env.extras["log"]["Metrics/mean_close_to_path_distance"] = torch.mean(
    closest_dists
  ).item()
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
  command, _ = resolve_path_command(env, command_name)
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


def _goal_dist(env: ManagerBasedRlEnv, command, foot_asset_cfg: SceneEntityCfg):
  """Base-footprint distance to the goal position.

  Reads ``goal_pos_w``, which both ``PathCommand`` (its final waypoint) and
  ``GoalPoseCommand`` (its sampled target) provide.
  """
  root_position = base_footprint_xy(env, foot_asset_cfg)
  return torch.norm(command.goal_pos_w - root_position, dim=-1)  # (num_envs,)


def goal_reached_linear(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  foot_asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_CFG,
  pos_std: float = 0.25,
  success_threshold: float = 0.1,
  metric_prefix: str = "goal",
) -> torch.Tensor:
  """Dense reward for reaching the goal position: ``exp(-dist^2 / pos_std^2)``.

  Works against any command exposing ``goal_pos_w`` -- the path command (its
  final waypoint) or the goal-pose command. Split out from the orientation term
  so each can be weighted independently. ``success_threshold`` only drives the
  logged success-rate metric; ``metric_prefix`` keeps the log keys distinct when
  both commands use this term."""
  del asset_cfg  # Position comes from the base footprint (feet), not the root.
  command, active = resolve_goal_term(env, command_name, sub_command_name)
  dist = _goal_dist(env, command, foot_asset_cfg)
  reward = torch.exp(-(dist**2) / pos_std**2)
  # Log only over the envs this command is actually driving.
  n = active.sum().clamp(min=1)
  log = env.extras["log"]
  log[f"Metrics/{metric_prefix}_reached_linear"] = ((reward * active).sum() / n).item()
  log[f"Metrics/{metric_prefix}_success_rate_linear"] = (
    ((dist < success_threshold).float() * active).sum() / n
  ).item()
  log[f"Metrics/mean_dist_to_{metric_prefix}"] = ((dist * active).sum() / n).item()
  return reward


def goal_reached_angular(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  foot_asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_CFG,
  pos_std: float = 0.25,
  yaw_std: float = 0.5,
  success_yaw_threshold: float = 0.2,
  metric_prefix: str = "goal",
) -> torch.Tensor:
  """Dense reward for reaching the goal orientation, gated by proximity so it
  only applies near the goal:

      exp(-dist^2 / pos_std^2) * exp(-yaw_err^2 / yaw_std^2)

  The position gate (same ``pos_std`` as the linear term) means facing the goal
  yaw far away is not rewarded and does not fight path-following; near the goal
  it gives a smooth gradient to turn onto the goal heading. ``success_yaw_*``
  only drives the logged success-rate metric."""
  del asset_cfg  # Position comes from the base footprint (feet), not the root.
  command, active = resolve_goal_term(env, command_name, sub_command_name)
  dist = _goal_dist(env, command, foot_asset_cfg)
  # yaw_err is wrap(goal_yaw - heading), updated by the command each step.
  yaw_err = command.yaw_err.abs()  # (num_envs,)
  pos_gate = torch.exp(-(dist**2) / pos_std**2)
  reward = pos_gate * torch.exp(-(yaw_err**2) / yaw_std**2)
  n = active.sum().clamp(min=1)
  log = env.extras["log"]
  log[f"Metrics/{metric_prefix}_reached_angular"] = ((reward * active).sum() / n).item()
  log[f"Metrics/mean_yaw_err_to_{metric_prefix}"] = (
    (yaw_err * active).sum() / n
  ).item()
  return reward
