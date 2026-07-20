from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.viewer.debug_visualizer import DebugVisualizer

from mjlab_piplus.tasks.mdp.path_following.path_gen import generate_parametric_paths
from mjlab_piplus.tasks.mdp.path_following.utils import (
  angle_distance,
  heading_from_quat,
  inv_mat,
  xy_yaw_to_mat,
)


class PathCommand(CommandTerm):
  def __init__(self, cfg: PathCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.align = False
    self.global_command_counter = 0
    if cfg.use_pathlib:
      self.all_paths, self.all_waypoints = self._load_or_generate_pathlib(cfg)
      # randomly select a path for each environment
      indices = torch.randint(0, cfg.pathlib_size, (self.num_envs,), device=self.device)
      self.paths = self.all_paths[indices]
      self.waypoints = self.all_waypoints[indices]
    else:
      self.paths, self.waypoints = generate_parametric_paths(
        num_paths=self.num_envs,
        max_point_dxy=cfg.max_point_dxy,
        num_waypoints=cfg.num_waypoints,
        curvature_limit=cfg.curvature_limit,
        point_spacing=cfg.point_spacing,
        device=self.device,
      )
    self.command_b = torch.zeros(
      self.num_envs, cfg.num_observations * 2, device=self.device
    )
    self._update_robot_pose()

  def _load_or_generate_pathlib(
    self, cfg: PathCommandCfg
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the pre-generated path library from disk, or generate and cache it.

    Path generation is deterministic (``generate_parametric_paths`` seeds numpy
    with 0), so the library is fully determined by the generation parameters.
    We key the cache file on those params; a cache hit skips regenerating all
    ``pathlib_size`` paths on every training run.
    """
    cache_dir = Path(cfg.cache_dir or (Path(__file__).parent / "path_cache"))
    # max_points is not exposed on the cfg; keep it in sync with the default
    # used by generate_parametric_paths so the key stays correct.
    fname = (
      f"pathlib_n{cfg.pathlib_size}_wp{cfg.num_waypoints}"
      f"_dxy{cfg.max_point_dxy}_curv{cfg.curvature_limit}"
      f"_sp{cfg.point_spacing}_mp1500.pt"
    )
    cache_path = cache_dir / fname

    if cache_path.exists():
      print(f"Loading cached path library from {cache_path}")
      data = torch.load(cache_path, map_location=self.device)
      return data["paths"], data["waypoints"]

    all_paths, all_waypoints = generate_parametric_paths(
      num_paths=cfg.pathlib_size,
      max_point_dxy=cfg.max_point_dxy,
      num_waypoints=cfg.num_waypoints,
      curvature_limit=cfg.curvature_limit,
      point_spacing=cfg.point_spacing,
      device=self.device,
      progress=True,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
      {"paths": all_paths.cpu(), "waypoints": all_waypoints.cpu()}, cache_path
    )
    print(f"Cached path library to {cache_path}")
    return all_paths, all_waypoints

  @property
  def command(self) -> torch.Tensor:
    return self.command_b

  def _resample_command(self, env_ids):
    self.starting_step = self._env.common_step_counter
    self._update_robot_pose()
    if self.cfg.use_pathlib:
      if self.cfg.eval_mode:
        indices = torch.tensor(
          [self.global_command_counter % self.cfg.pathlib_size], device=self.device
        ).repeat(len(env_ids))
      else:
        indices = torch.randint(
          0, self.cfg.pathlib_size, (len(env_ids),), device=self.device
        )
      if self.align:
        self.paths[env_ids] = self.all_paths[indices]
        self.paths[env_ids, :, :2] += self.current_pose[env_ids, :2].unsqueeze(1)
        self.waypoints[env_ids] = (
          self.all_waypoints[indices] + self.current_pose[env_ids, :2].unsqueeze(1)
        )
      else:
        self.paths[env_ids] = self.all_paths[indices]
        self.waypoints[env_ids] = self.all_waypoints[indices]
    else:
      paths, waypoints = generate_parametric_paths(
        num_paths=len(env_ids),
        max_point_dxy=self.cfg.max_point_dxy,
        num_waypoints=self.cfg.num_waypoints,
        curvature_limit=self.cfg.curvature_limit,
        point_spacing=self.cfg.point_spacing,
        device=self.device,
      )
      if self.align:
        self.paths[env_ids] = paths
        self.paths[env_ids, :, :2] += self.current_pose[env_ids, :2].unsqueeze(1)
        self.waypoints[env_ids] = (
          waypoints + self.current_pose[env_ids, :2].unsqueeze(1)
        )
      else:
        self.paths[env_ids] = paths
        self.waypoints[env_ids] = waypoints

    if self.cfg.eval_mode:
      # write waypoints to csv
      with open("commands_path.csv", "a") as f:
        for i in range(self.waypoints.shape[0]):
          waypoint_str = ",".join(
            [f"{x:.2f}" for x in self.waypoints[i].cpu().numpy().flatten()]
          )
          f.write(f"{waypoint_str},")
        f.write(f"{self.global_command_counter}\n")
    self.global_command_counter += 1

  def _update_metrics(self):
    pass

  def _update_robot_pose(self) -> None:
    root_pos = self.robot.data.root_link_pos_w
    root_quat = self.robot.data.root_link_quat_w
    heading = heading_from_quat(root_quat).unsqueeze(-1)
    self.current_pose = torch.cat([root_pos[..., :2], heading], dim=-1)

  def _update_command(self) -> None:
    # get closest point on path to robot
    self._update_robot_pose()
    root_pos = self.current_pose[..., :2]  # (n_envs, 2)
    dists = torch.norm(
      self.paths[..., :2] - root_pos.unsqueeze(1), dim=-1
    )  # (num_envs, resolution)
    closest_indices = torch.argmin(dists, dim=-1)  # (num_envs,)

    obs_offsets = (
      torch.arange(self.cfg.num_observations, device=self.device)
      * self.cfg.observation_spacing
    )
    obs_indices = closest_indices.unsqueeze(1) + obs_offsets.unsqueeze(
      0
    )  # (num_envs, num_observations)
    # Clamp indices to be within path bounds
    obs_indices = torch.clamp(obs_indices, 0, self.paths.shape[1] - 1)

    future_points_in_world = torch.gather(
      self.paths[..., :2], 1, obs_indices.unsqueeze(-1).expand(-1, -1, 2)
    )  # (num_envs, num_observations, 2)
    # make into homogenous coordinates
    future_points_in_world_hom = torch.cat(
      [
        future_points_in_world,
        torch.zeros(
          (self.num_envs, self.cfg.num_observations, 1), device=self.device
        ),
        torch.ones(
          (self.num_envs, self.cfg.num_observations, 1), device=self.device
        ),
      ],
      dim=-1,
    ).unsqueeze(-1)  # (num_envs, num_observations, 4, 1)

    world_to_robot = xy_yaw_to_mat(self.current_pose)  # (num_envs, 4, 4)
    robot_to_world = inv_mat(world_to_robot)  # (num_envs, 4, 4)
    future_points_relative = torch.matmul(
      robot_to_world.unsqueeze(1), future_points_in_world_hom
    ).squeeze(-1)  # (num_envs, num_observations, 4)
    self.command_b = future_points_relative[..., :2].reshape(
      self.num_envs, -1
    )  # (num_envs, num_observations*2)

    if self.cfg.eval_mode:
      # hacky calculate metrics
      closest_dists = torch.min(dists, dim=-1).values  # (num_envs,)
      closest_waypoints = self.paths[:, closest_indices]  # (num_envs, 2)
      next_indices = torch.clamp(closest_indices + 1, max=self.paths.shape[1] - 1)
      next_waypoints = self.paths[:, next_indices]  # (num_envs, 2)
      path_tangents = next_waypoints - closest_waypoints  # (num_envs, 2)
      path_headings = torch.atan2(
        path_tangents[..., 1], path_tangents[..., 0]
      )  # (num_envs,)
      robot_headings = self.current_pose[:, 2]
      abs_angle_diff = torch.abs(angle_distance(robot_headings, path_headings))
      distances_to_goal = torch.norm(self.waypoints[:, -1, :] - root_pos, dim=-1)
      with open(f"path_logs{self.global_command_counter:03d}.csv", "a") as f:
        f.write(
          f"{self.starting_step},{self._env.common_step_counter},"
          f"{closest_dists.item()},{abs_angle_diff.item()},"
          f"{distances_to_goal.item()}\n"
        )

      if distances_to_goal.item() < 0.15:
        print("Reached goal, resetting env")
        self._env.reset()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    self._update_robot_pose()
    env_indices = visualizer.get_env_indices(self.num_envs)
    for batch in env_indices:
      for i in range(self.paths.shape[1]):
        path_point_with_z = np.concatenate(
          [self.paths[batch, i, :2].cpu().numpy(), [0.0]]
        )
        visualizer.add_sphere(path_point_with_z, radius=0.02, color=(0, 0, 1, 0.5))
      for i in range(self.waypoints[batch].shape[0]):
        waypoint_with_z = np.concatenate(
          [self.waypoints[batch, i].cpu().numpy(), [0.0]]
        )
        visualizer.add_sphere(waypoint_with_z, radius=0.05, color=(0, 1, 0, 1))

      world_to_robot = (
        xy_yaw_to_mat(self.current_pose[batch : batch + 1]).cpu().numpy()
      )  # (1, 4, 4)
      for i in range(self.cfg.num_observations):
        command_point_hom = np.concatenate(
          [self.command_b[batch, i * 2 : i * 2 + 2].cpu().numpy(), [0.0, 1.0]]
        )
        command_point_in_world_hom = world_to_robot @ command_point_hom
        command_point_with_z = command_point_in_world_hom[0, :3]
        visualizer.add_sphere(command_point_with_z, radius=0.05, color=(1, 0, 0, 1))


@dataclass(kw_only=True)
class PathCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  num_waypoints: int = 4  # Number of waypoints in the generated path.
  curvature_limit: float = 10.0  # Max curvature; higher allows sharper turns.
  max_point_dxy: float = 3.0  # Max change in x/y between consecutive waypoints.
  point_spacing: float = 0.01  # Euclidean (arc-length) spacing between path samples, meters.
  num_observations: int = 20  # Number of future path points in the command obs.
  observation_spacing: int = 15  # Spacing between observed path points.
  use_pathlib: bool = True
  pathlib_size: int = 50_000
  cache_dir: str | None = None  # Directory for the cached path library
  eval_mode: bool = False  # Use a fixed set of paths for evaluation if True.

  @dataclass
  class VizCfg:
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> PathCommand:
    return PathCommand(self, env)

  def __post_init__(self):
    pass
