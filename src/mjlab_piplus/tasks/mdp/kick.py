"""Kick command: send the ball off in a commanded direction at a commanded speed.

The command owns the ball. While it is driving an env it places the ball near
the robot; while it is not, it parks the ball far away so the other command
types never trip over it. That makes the ball's presence in the scene entirely
this command's business.

The policy sees, in the robot base frame, where the ball is, how fast it is
moving, which way to kick it, and how hard:

    [ball_x, ball_y, ball_vx, ball_vy, cos(dir_err), sin(dir_err), strength]

``dir_err`` is the commanded kick direction relative to the robot's heading,
encoded as a unit vector because it wraps. ``MultiCommand`` zeroes the whole
block on envs the kick command is not driving.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.lab_api.math import wrap_to_pi

from mjlab_piplus.objects import BALL_RADIUS
from mjlab_piplus.tasks.mdp.multi_command import resolve_command_term

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

__all__ = [
  "KickCommand",
  "KickCommandCfg",
  "kick_approach",
  "kick_velocity_match",
]


class KickCommand(CommandTerm):
  cfg: KickCommandCfg

  def __init__(self, cfg: KickCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.ball: Entity = env.scene[cfg.ball_entity_name]

    # Commanded kick: a world-frame heading and a target ball speed.
    self.kick_dir_w = torch.zeros(self.num_envs, device=self.device)
    self.strength = torch.zeros(self.num_envs, device=self.device)
    # Latched once the ball has been set moving, cleared on resample. Gates the
    # approach and velocity-match rewards into their two disjoint phases.
    self.has_been_kicked = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    # Ball state in the robot base frame, refreshed every step.
    self.ball_pos_b = torch.zeros(self.num_envs, 2, device=self.device)
    self.ball_vel_b = torch.zeros(self.num_envs, 2, device=self.device)
    self.ball_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.dir_err = torch.zeros(self.num_envs, device=self.device)

    # Set each step by MultiCommand; None means "this term drives every env".
    self.active_env_mask: torch.Tensor | None = None
    self._was_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # write_root_state_to_sim does not show up in the entity's derived state
    # until the next forward, so remember what we wrote and use that for this
    # step's observation. Without it a freshly placed ball would be observed at
    # its parked position for one step -- a 100 m spike into the policy.
    self._wrote = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._wrote_pos = torch.zeros(self.num_envs, 2, device=self.device)

    # Placement is sampled on resample but applied in _update_command. At an
    # episode reset the command manager runs *before* sim.forward(), so the
    # robot pose visible during _resample_command is still the pre-reset one;
    # placing there would put the ball relative to wherever the robot used to
    # be. _update_command runs after the forward, so it sees the real pose.
    self._pending_place = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._pending_offset = torch.zeros(self.num_envs, 2, device=self.device)

    self.metrics["kick_speed_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["kick_dir_error"] = torch.zeros(self.num_envs, device=self.device)

  # Properties.

  @property
  def command(self) -> torch.Tensor:
    return torch.cat(
      [
        self.ball_pos_b,
        self.ball_vel_b,
        torch.cos(self.dir_err).unsqueeze(1),
        torch.sin(self.dir_err).unsqueeze(1),
        self.strength.unsqueeze(1),
      ],
      dim=1,
    )

  @property
  def target_vel_w(self) -> torch.Tensor:
    """Commanded ball velocity in world xy, ``(num_envs, 2)``."""
    return self.strength.unsqueeze(1) * torch.stack(
      [torch.cos(self.kick_dir_w), torch.sin(self.kick_dir_w)], dim=1
    )

  # The GoalPoseTerm contract, with the ball as the "goal": lets the shared
  # goal-bearing rewards be pointed at this command too.

  @property
  def goal_pos_w(self) -> torch.Tensor:
    return self.ball_pos_w

  @property
  def goal_command_b(self) -> torch.Tensor:
    return torch.cat([self.ball_pos_b, self.dir_err.unsqueeze(1)], dim=1)

  @property
  def yaw_err(self) -> torch.Tensor:
    return self.dir_err

  # Command lifecycle.

  def _active(self) -> torch.Tensor:
    if self.active_env_mask is None:
      return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    return self.active_env_mask

  def _write_ball(self, env_ids: torch.Tensor, pos_xy: torch.Tensor) -> None:
    """Teleport the ball to ``pos_xy`` at rest."""
    if len(env_ids) == 0:
      return
    state = torch.zeros(len(env_ids), 13, device=self.device)
    state[:, :2] = pos_xy
    state[:, 2] = self.cfg.ball_radius
    state[:, 3] = 1.0  # Identity quaternion; velocities stay zero.
    self.ball.write_root_state_to_sim(state, env_ids)
    self._wrote[env_ids] = True
    self._wrote_pos[env_ids] = pos_xy

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    r = torch.empty(n, device=self.device)

    self.kick_dir_w[env_ids] = r.uniform_(*self.cfg.direction_range)
    self.strength[env_ids] = torch.empty(n, device=self.device).uniform_(
      *self.cfg.strength_range
    )
    self.has_been_kicked[env_ids] = False

    # Offset from the robot at a random bearing, distance_range metres away.
    # Only the offset is decided here; _update_command anchors it on the robot.
    distance = torch.empty(n, device=self.device).uniform_(*self.cfg.distance_range)
    bearing = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
    self._pending_offset[env_ids] = torch.stack(
      [distance * torch.cos(bearing), distance * torch.sin(bearing)], dim=1
    )
    self._pending_place[env_ids] = True

  def _update_command(self) -> None:
    active = self._active()

    # Apply any placement sampled since the last update, now that the robot
    # pose is current.
    pending = self._pending_place.nonzero().flatten()
    if len(pending) > 0:
      base_xy = self.robot.data.root_link_pos_w[pending, :2]
      self._write_ball(pending, base_xy + self._pending_offset[pending])
      self._pending_place[pending] = False

    # Park the ball wherever this command is not driving, so it cannot get in
    # the way of whatever command took over. Normally only on the transition --
    # a parked ball rests where it is and needs no further writes -- but also
    # for freshly reset envs, because scene.reset() restores every entity to its
    # default pose and so undoes the parking.
    just_reset = self._env.episode_length_buf <= 1
    newly_inactive = (~active & (self._was_active | just_reset)).nonzero().flatten()
    if len(newly_inactive) > 0:
      park = torch.tensor(self.cfg.park_position, device=self.device)
      park_xy = self._env.scene.env_origins[newly_inactive, :2] + park
      self._write_ball(newly_inactive, park_xy)
    self._was_active = active.clone()

    # Ball state, preferring what we just wrote where the sim has not caught up.
    pos_w = self.ball.data.root_link_pos_w[:, :2]
    vel_w = self.ball.data.root_link_lin_vel_w[:, :2]
    if bool(self._wrote.any()):
      fresh = self._wrote.unsqueeze(1)
      pos_w = torch.where(fresh, self._wrote_pos, pos_w)
      vel_w = torch.where(fresh, torch.zeros_like(vel_w), vel_w)
      self._wrote[:] = False
    self.ball_pos_w = pos_w

    # Into the base frame.
    base_xy = self.robot.data.root_link_pos_w[:, :2]
    heading = self.robot.data.heading_w
    cos_h, sin_h = torch.cos(heading), torch.sin(heading)
    delta = pos_w - base_xy
    self.ball_pos_b = torch.stack(
      [
        cos_h * delta[:, 0] + sin_h * delta[:, 1],
        -sin_h * delta[:, 0] + cos_h * delta[:, 1],
      ],
      dim=1,
    )
    self.ball_vel_b = torch.stack(
      [
        cos_h * vel_w[:, 0] + sin_h * vel_w[:, 1],
        -sin_h * vel_w[:, 0] + cos_h * vel_w[:, 1],
      ],
      dim=1,
    )
    self.dir_err = wrap_to_pi(self.kick_dir_w - heading)

    # The ball counts as kicked once it is properly moving. Latched, so the
    # approach reward cannot be re-earned by walking back to a ball the robot
    # itself nudged away.
    speed = torch.norm(vel_w, dim=1)
    self.has_been_kicked |= active & (speed > self.cfg.kicked_speed_threshold)

  def _update_metrics(self) -> None:
    # Only meaningful once the ball is moving; averaged over the episode.
    scale = self._env.step_dt / self._env.max_episode_length_s
    vel_w = self.ball.data.root_link_lin_vel_w[:, :2]
    counted = (self._active() & self.has_been_kicked).float()
    speed_err = (torch.norm(vel_w, dim=1) - self.strength).abs()
    ball_dir = torch.atan2(vel_w[:, 1], vel_w[:, 0])
    dir_err = wrap_to_pi(ball_dir - self.kick_dir_w).abs()
    self.metrics["kick_speed_error"] += speed_err * counted * scale
    self.metrics["kick_dir_error"] += dir_err * counted * scale

  # Visualization.

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    import numpy as np

    ball_xy = self.ball_pos_w.cpu().numpy()
    dirs = self.kick_dir_w.cpu().numpy()
    strengths = self.strength.cpu().numpy()
    for batch in env_indices:
      # Arrow from the ball along the commanded kick, scaled by strength.
      length = 0.15 * float(strengths[batch])
      start = np.array([ball_xy[batch, 0], ball_xy[batch, 1], BALL_RADIUS])
      end = start + np.array(
        [length * math.cos(dirs[batch]), length * math.sin(dirs[batch]), 0.0]
      )
      visualizer.add_arrow(start, end, color=(1.0, 0.4, 0.0, 0.9), width=0.02)


@dataclass(kw_only=True)
class KickCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  ball_entity_name: str = "ball"
  ball_radius: float = BALL_RADIUS

  distance_range: tuple[float, float] = (0.2, 2.0)
  """How far from the robot the ball is placed when the command starts, in
  metres, at a uniformly random bearing."""

  direction_range: tuple[float, float] = (-math.pi, math.pi)
  """Commanded kick heading in the world frame; the full circle by default."""

  strength_range: tuple[float, float] = (1.0, 5.0)
  """Commanded ball speed in m/s."""

  park_position: tuple[float, float] = (100.0, 100.0)
  """Where the ball is sent, relative to the env origin, while this command is
  not driving the env -- far enough that it cannot interact with anything."""

  kicked_speed_threshold: float = 0.5
  """Ball speed (m/s) above which the ball counts as kicked. Latches for the
  rest of the command period."""

  resampling_time_range: tuple[float, float] = (6.0, 12.0)

  @dataclass
  class VizCfg:
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> KickCommand:
    return KickCommand(self, env)


##
# Rewards.
##


class kick_approach:
  """Reward closing on the ball, in metres/second, before it has been kicked.

  Telescoping like the other progress terms, so walking out and back nets ~0.
  It switches off for good once the ball has been kicked -- otherwise the robot
  could nudge the ball away (no penalty, the term is inactive while the ball
  rolls) and then re-earn the approach by walking to it again, over and over.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.last_dist = torch.zeros(env.num_envs, device=env.device)
    self.last_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sub_command_name: str | None = None,
    max_speed: float = 1.0,
  ) -> torch.Tensor:
    term, _ = resolve_command_term(env, command_name, sub_command_name)
    assert isinstance(term, KickCommand)
    dist = torch.norm(term.ball_pos_b, dim=1)
    speed = (self.last_dist - dist) / env.step_dt
    # A new command means a new ball placement; that delta is meaningless.
    changed = term.command_counter != self.last_counter
    speed = torch.where(changed, torch.zeros_like(speed), speed)
    reward = torch.clamp(speed, max=max_speed)
    reward = torch.where(term.has_been_kicked, torch.zeros_like(reward), reward)
    self.last_dist = dist
    self.last_counter = term.command_counter.clone()
    return reward


def kick_velocity_match(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
  std: float = 1.0,
) -> torch.Tensor:
  """Reward the ball moving at the commanded velocity, once it has been kicked.

  ``exp(-|v_ball - v_target|^2 / std^2)`` on the world-frame xy velocity, so
  direction and speed are scored together. Zero until the ball is actually
  moving, which is what ``kick_approach`` is there to bridge.
  """
  term, _ = resolve_command_term(env, command_name, sub_command_name)
  assert isinstance(term, KickCommand)
  vel_w = term.ball.data.root_link_lin_vel_w[:, :2]
  error = torch.sum(torch.square(vel_w - term.target_vel_w), dim=1)
  reward = torch.exp(-error / std**2)
  return torch.where(term.has_been_kicked, reward, torch.zeros_like(reward))
