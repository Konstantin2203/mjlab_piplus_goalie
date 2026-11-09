"""Shot command: automatic shots on goal for training the goalie.

The ball is placed 2-4 meters in front of the goal at a random position 
and launched towards a random point on the goal line at a random speed.

The goalie gets no command, so the policy learns nothing about target or
speed in advance, so it only sees the ball through ``shot_ball_state`` like
it would see it via a camera (Lags are currently not being used).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from mjlab_piplus.objects import BALL_RADIUS

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

__all__ = [
  "ShotCommand",
  "ShotCommandCfg",
  "shot_ball_state",
]

class ShotCommand(CommandTerm):
  cfg: ShotCommandCfg

  def __init__(self, cfg: ShotCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.ball: Entity = env.scene[cfg.ball_entity_name]

    # Initializing by creating tensors 
    self.ball_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.ball_pos_b = torch.zeros(self.num_envs, 2, device=self.device)
    self.ball_vel_b = torch.zeros(self.num_envs, 2, device=self.device)

    self.target_y = torch.zeros(self.num_envs, device=self.device)
    self.shot_speed = torch.zeros(self.num_envs, device=self.device)

    self._pending_place = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    self._wrote = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._wrote_pos = torch.zeros(self.num_envs, 2, device=self.device)
    self._wrote_vel = torch.zeros(self.num_envs, 2, device=self.device)

    self.metrics["shots"] = torch.zeros(self.num_envs, device=self.device)

  # Command for the goalie (empty on purpose because he should just see the ball through
  # shot_ball_state)
  @property
  def command(self) -> torch.Tensor:
    return torch.zeros(self.num_envs, 0, device=self.device)

  # Combines position and velocity of the ball into a tensor (4 columns)
  @property
  def ball_state_b(self) -> torch.Tensor:
    return torch.cat([self.ball_pos_b, self.ball_vel_b], dim=1)

  # Teleports ball to position with velocity
  def _write_ball(
    self, env_ids: torch.Tensor, pos_xy: torch.Tensor, vel_xy: torch.Tensor
  ) -> None:
    if len(env_ids) == 0:
      return
    r = self.cfg.ball_radius
    state = torch.zeros(len(env_ids), 13, device=self.device)
    state[:, :2] = pos_xy # every line, columns 0 and 1
    state[:, 2] = r # every line, column 2
    state[:, 3] = 1.0  # every line, column 3 (not turned)
    state[:, 7:9] = vel_xy # ..., columns 7 and 8
    state[:, 10] = -vel_xy[:, 1] / r # ..., column 10 (Turning speed of the Ball)
    state[:, 11] = vel_xy[:, 0] / r # ..., column 11 (e.g. 10)
    self.ball.write_root_state_to_sim(state, env_ids)
    self._wrote[env_ids] = True
    self._wrote_pos[env_ids] = pos_xy
    self._wrote_vel[env_ids] = vel_xy

  # Creates a new shot for envs
  def _apply_pending_placement(self) -> None:
    env_ids = self._pending_place.nonzero().flatten() # List of envs that need a shot
    if len(env_ids) == 0:
      return
    n = len(env_ids) # Number of shots
    origin = self._env.scene.env_origins[env_ids, :2] # Center of each needed env

    dist = torch.empty(n, device=self.device).uniform_(*self.cfg.distance_range)
    side = torch.empty(n, device=self.device).uniform_(*self.cfg.lateral_range)
    start = torch.stack([self.cfg.goal_x + dist, side], dim=1)

    target_y = torch.empty(n, device=self.device).uniform_(*self.cfg.target_y_range)
    target = torch.stack(
      [torch.full_like(target_y, self.cfg.goal_x), target_y], dim=1
    )

    direction = target - start
    direction = direction / torch.norm(direction, dim=1, keepdim=True) # Dividing through vectors lenght to keep direction but get length 1 
    speed = torch.empty(n, device=self.device).uniform_(*self.cfg.speed_range)
    vel = direction * speed.unsqueeze(1)

    self._write_ball(env_ids, origin + start, vel)
    self.target_y[env_ids] = target_y
    self.shot_speed[env_ids] = speed
    self.metrics["shots"][env_ids] += 1.0
    self._pending_place[env_ids] = False

  # Called if new command is needed in some envs
  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._pending_place[env_ids] = True

  # Runs through every control step, reads ball state and converts into robot frame
  def _update_command(self) -> None:
    self._apply_pending_placement()

    pos_w = self.ball.data.root_link_pos_w[:, :2]
    vel_w = self.ball.data.root_link_lin_vel_w[:, :2]
    if bool(self._wrote.any()):
      fresh = self._wrote.unsqueeze(1)
      pos_w = torch.where(fresh, self._wrote_pos, pos_w)
      vel_w = torch.where(fresh, self._wrote_vel, vel_w)
      self._wrote[:] = False
    self.ball_pos_w = pos_w

    base_xy = self.robot.data.root_link_pos_w[:, :2]
    heading = self.robot.data.heading_w
    cos_h, sin_h = torch.cos(heading), torch.sin(heading)
    delta = pos_w - base_xy # Vektor robot to ball
    # x becomes direction of robot
    self.ball_pos_b = torch.stack(
      [
        cos_h * delta[:, 0] + sin_h * delta[:, 1],
        -sin_h * delta[:, 0] + cos_h * delta[:, 1],
      ],
      dim=1,
    )
    # velocity from robots perspective
    self.ball_vel_b = torch.stack(
      [
        cos_h * vel_w[:, 0] + sin_h * vel_w[:, 1],
        -sin_h * vel_w[:, 0] + cos_h * vel_w[:, 1],
      ],
      dim=1,
    )

  def _update_metrics(self) -> None:
    pass

  # Arrow for ball in simulation
  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    import numpy as np

    pos = self.ball.data.root_link_pos_w[:, :2].cpu().numpy()
    vel = self.ball.data.root_link_lin_vel_w[:, :2].cpu().numpy()
    for i in env_indices:
      start = np.array([pos[i, 0], pos[i, 1], BALL_RADIUS])
      end = start + 0.3 * np.array([vel[i, 0], vel[i, 1], 0.0])
      visualizer.add_arrow(start, end, color=(1.0, 0.4, 0.0, 0.9), width=0.02)

@dataclass(kw_only=True)
class ShotCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  ball_entity_name: str = "ball"
  ball_radius: float = BALL_RADIUS

  goal_x: float = -0.2
  distance_range: tuple[float, float] = (2.0, 4.0)
  lateral_range: tuple[float, float] = (-1.5, 1.5)
  target_y_range: tuple[float, float] = (-1.1, 1.1)
  speed_range: tuple[float, float] = (1.0, 3.0)

  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> ShotCommand:
    return ShotCommand(self, env)

##
# Observation.
##
def shot_ball_state(env: ManagerBasedRlEnv, command_name: str = "shot") -> torch.Tensor:
  term = env.command_manager.get_term(command_name)
  return term.ball_state_b
