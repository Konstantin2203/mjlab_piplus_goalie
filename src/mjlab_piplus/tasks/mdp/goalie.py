"""Torwart-Funktionen: 
Observation (Tor relativ zum Roboter) und Terminations.
 
Koordinaten wie in shot.py: x zeigt vom Tor ins Feld, die Torlinie liegt bei
GOAL_X (relativ zum Ursprung der jeweiligen Umgebung), y ist seitlich.
"""

import torch

from mjlab_piplus.objects import BALL_RADIUS

__all__ = [
  "ball_missed",
  "ball_saved",
  "goal_scored",
  "goal_pos_b",
]

"""Torlinie. Muss zu pos von Torlinie/Pfosten und zu ShotCommandCfg.goal_x passen."""
GOAL_X = -0.2

"""Halbe lichte Torbreite (1,8 m Tor)."""
GOAL_HALF_WIDTH = 0.9

"""Halbe Breite der Torlinie (5 cm Linie)."""
LINE_HALF_WIDTH = 0.025


##
# Observation.
##

"""Tormitte relativ zum Roboter: [vor mir, links von mir]."""
def goal_pos_b(env, goal_pos=(GOAL_X, 0.0)):
  
  """den Roboter aus der Szene holen"""
  robot = env.scene["robot"]
  goal = torch.tensor(goal_pos, device=env.device) + env.scene.env_origins[:, :2]
 
  # Into the base frame (wie kick.py).
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

"""Ballposition relativ zum Ursprung der jeweiligen Umgebung."""
def _ball_xy_local(env):
  ball = env.scene["ball"]
  """Ballposition relativ zur eigenen Umgebung"""
  return ball.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]

"""Tor: der ganze Ball ist über die Torlinie, zwischen den Pfosten."""
def goal_scored(env, goal_x=GOAL_X, half_width=GOAL_HALF_WIDTH):
  xy = _ball_xy_local(env)
  behind_line = xy[:, 0] < goal_x - LINE_HALF_WIDTH - BALL_RADIUS
  between_posts = xy[:, 1].abs() < half_width
  return behind_line & between_posts

"""Daneben: Ball ist an der Torlinie vorbei, aber neben den Pfosten."""
def ball_missed(env, goal_x=GOAL_X, half_width=GOAL_HALF_WIDTH):
  xy = _ball_xy_local(env)
  behind_line = xy[:, 0] < goal_x - LINE_HALF_WIDTH - BALL_RADIUS
  outside_posts = xy[:, 1].abs() >= half_width
  return behind_line & outside_posts

"""Gehalten: Ball ist noch vor dem Tor und rollt vom Tor weg
  oder liegt still in der Nähe des Roboters.
 
  Achtung: Ein Abpraller vom Pfosten zählt damit auch als gehalten. 
  Wollen wir das so lassen? lol 
  """
def ball_saved(
  env, goal_x=GOAL_X, away_speed=0.1, stop_speed=0.1, near_dist=0.5
):
  
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