"""Goalkeeper functions: 
Observations (goal in relation to the robot), terminations, rewards and helper classes.
 
Coordinates as in shot.py: x points from the goal into the field, the goal line
lies at GOAL_X (relative to the origin of each environment), y is sideways.
"""

import torch

from mjlab_piplus.objects import BALL_RADIUS

__all__ = [
  "ball_missed",
  "ball_saved",
  "ball_proximity",
  "goal_conceded",
  "goal_pos_b",
  "goal_proximity",
  "end_after_event",
  "event_once",
  "kick_away",
  "ball_away"
]

# x position of the goal line. Must match the pos of goal line / posts
# and ShotCommandCfg.goal_x.
GOAL_X = -0.2

# Half of the inner goal width (1.8 m goal).
GOAL_HALF_WIDTH = 0.9

# Half of the goal line width (5 cm line).
LINE_HALF_WIDTH = 0.025

class end_after_event:
  """Ends the episode delay_s seconds after the condition was true for the first time."""

  def __init__(self, cfg, env):
    # Step at which the event happened, -1 = not happened yet.
    self.event_step = torch.full(
      (env.num_envs,), -1, dtype=torch.long, device=env.device
    )

  def __call__(self, env, condition, delay_s=1.0):
    step = env.episode_length_buf      
    self.event_step[step <= 1] = -1 # new episode: reset       
    first = condition(env) & (self.event_step < 0)
    self.event_step[first] = step[first] # remember when the event happened   
    delay_steps = int(round(delay_s / env.step_dt))   
    return (self.event_step >= 0) & (step - self.event_step >= delay_steps)

class event_once:
  """Returns 1 when the condition becomes true for the first time in this episode, else 0."""

  def __init__(self, cfg, env):
    self.fired = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(self, env, condition):
    self.fired[env.episode_length_buf <= 1] = False # new episode: reset
    now = condition(env) & ~self.fired
    self.fired |= now
    return now.float()

##
# Observation.
##

def goal_pos_b(env, goal_pos=(GOAL_X, 0.0)):
  """Goal center relative to the robot: [in front of me, left of me]."""
  robot = env.scene["robot"] 
  goal = torch.tensor(goal_pos, device=env.device) + env.scene.env_origins[:, :2]
 
  # Rotate into the base frame (like kick.py)
  base_xy = robot.data.root_link_pos_w[:, :2]
  heading = robot.data.heading_w
  cos_h, sin_h = torch.cos(heading), torch.sin(heading)
  delta = goal - base_xy
  return torch.stack(
    [
      cos_h * delta[:, 0] + sin_h * delta[:, 1],
      -sin_h * delta[:, 0] + cos_h * delta[:, 1],
    ],
    dim=1,
  )

##
# Terminations.
##

def _ball_xy_local(env):
  """Ball position relative to the origin of its own environment."""
  ball = env.scene["ball"]
  return ball.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]

def goal_conceded(env, goal_x=GOAL_X, half_width=GOAL_HALF_WIDTH):
  """Goal: the whole ball is behind the goal line, between the posts."""
  xy = _ball_xy_local(env)
  behind_line = xy[:, 0] < goal_x - LINE_HALF_WIDTH - BALL_RADIUS
  between_posts = xy[:, 1].abs() < half_width
  return behind_line & between_posts

def ball_missed(env, goal_x=GOAL_X, half_width=GOAL_HALF_WIDTH):
  """Miss: the ball is past the goal line, but outside the posts."""
  xy = _ball_xy_local(env)
  behind_line = xy[:, 0] < goal_x - LINE_HALF_WIDTH - BALL_RADIUS
  outside_posts = xy[:, 1].abs() >= half_width
  return behind_line & outside_posts


def ball_saved(
  env, goal_x=GOAL_X, away_speed=0.1, stop_speed=0.1, near_dist=0.5
):
  """Save: the ball is still in front of the goal and rolls away from it,
  or lies still close to the robot.
  """
  # TODO: A rebound off the post also counts as a save. Discuss in the team.
  
  ball = env.scene["ball"]
  robot = env.scene["robot"]
  xy = _ball_xy_local(env)
  vel = ball.data.root_link_lin_vel_w[:, :2]
  speed = torch.norm(vel, dim=1)
 
  in_front = xy[:, 0] > goal_x
  moving_away = vel[:, 0] > away_speed
 
  dist_to_robot = torch.norm(
    ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2], dim=1
  )
  stopped_near = (speed < stop_speed) & (dist_to_robot < near_dist)
 
  return in_front & (moving_away | stopped_near)

##
# Rewards.
##

def ball_proximity(
    env, near_dist=1.0, std= 0.3):
    """Reward for being close to the ball (only within near_dist), between 0 and 1."""
    ball = env.scene["ball"]
    robot = env.scene["robot"]
    d = torch.norm( ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2], dim=1)
    return torch.where(d< near_dist, torch.exp(-(d/std)** 2), torch.zeros_like(d))

def goal_proximity(env, goal_x=GOAL_X, max_forward=1, max_side=0.9):
    """Penalty (use a negative weight!) when the goalkeeper moves too far in front of or beside the goal.
       Returns how far he is outside the allowed area.
    """
    robot = env.scene["robot"]
    xy = robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    dist=xy[:,0] - goal_x
    too_far = (dist - max_forward).clamp(min=0.0)
    too_wide = (xy[:,1].abs() - max_side).clamp(min=0.0)
    return too_far + too_wide

def kick_away(env, near_dist=3.0, target_x=8.8, target_y=0, std= 0.3):
    """Reward when the ball rolls away from our goal to a specific point in the opponent's half"""

    ball = env.scene["ball"]
    target = torch.tensor(
        [target_x, target_y], 
        device=ball.data.root_link_pos_w.device, 
        dtype=ball.data.root_link_pos_w.dtype
    )
    d = torch.norm(ball.data.root_link_pos_w[:, :2] - target, dim=1)
    return torch.where(d< near_dist, torch.exp(-(d/std)** 2), torch.zeros_like(d))

def ball_away(env):
    ball = env.scene["ball"]
    vel = ball.data.root_link_lin_vel_w[:, :2]
    moving_away = vel[:, 0] < 0.1
    return moving_away
    

