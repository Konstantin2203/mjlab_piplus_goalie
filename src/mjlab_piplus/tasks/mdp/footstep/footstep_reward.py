"""Footstep-tracking rewards, one axis group per term.

The deviation of a foot from the footstep it was asked to land on is scored in
four independent terms so each can be weighted (and curricularized) on its own:

- :func:`footstep_tracking_xy` -- Euclidean distance in the ground plane.
- :func:`footstep_tracking_z` -- height error. Always ~0 on flat ground;
  the term is here so it is already wired when terrain lands.
- :func:`footstep_tracking_yaw` -- foot heading error.
- :func:`footstep_tracking_roll_pitch` -- foot tilt error. Like ``z``, a
  placeholder until the targets stop being flat.

All four score a foot only while it is **in stance**, against the target of the
footstep it most recently landed. Scoring the swing foot instead would pay for
hovering over the target; scoring stance pays for landing on it and staying
there, and is dense (a foot is planted for most of the gait cycle) rather than
a single-step impulse at touchdown.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import wrap_to_pi

from mjlab_piplus.tasks.mdp.footstep.footstep_command import FootstepCommand
from mjlab_piplus.tasks.mdp.multi_command import resolve_command_term

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

__all__ = [
  "footstep_tracking_roll_pitch",
  "footstep_tracking_xy",
  "footstep_tracking_yaw",
  "footstep_tracking_z",
]


def _footstep_term(
  env: ManagerBasedRlEnv, command_name: str, sub_command_name: str | None
) -> FootstepCommand:
  term, _ = resolve_command_term(env, command_name, sub_command_name)
  if not isinstance(term, FootstepCommand):
    raise TypeError(
      f"Command '{sub_command_name or command_name}' is not a FootstepCommand "
      f"(got {type(term).__name__})."
    )
  return term


def _stance_reward(term: FootstepCommand, error: torch.Tensor, std: float):
  """Mean of ``exp(-error^2 / std^2)`` over the feet currently in stance.

  Zero when no foot is planted (both in swing), which the gait makes rare but
  not impossible.
  """
  in_stance = term.foot_in_stance.float()
  count = in_stance.sum(dim=1)
  reward = (torch.exp(-torch.square(error) / std**2) * in_stance).sum(dim=1)
  return torch.where(count > 0, reward / count.clamp(min=1.0), torch.zeros_like(reward))


def footstep_tracking_xy(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  sub_command_name: str | None = None,
) -> torch.Tensor:
  """Reward planted feet for sitting on their footstep target in the xy plane."""
  term = _footstep_term(env, command_name, sub_command_name)
  pos, _ = term.foot_pose_w()
  error = torch.norm(pos[..., :2] - term.stance_target_w[..., :2], dim=-1)
  return _stance_reward(term, error, std)


def footstep_tracking_z(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  sub_command_name: str | None = None,
) -> torch.Tensor:
  """Reward planted feet for sitting at their footstep target's height.

  Flat ground makes every target height the resting height, so this is close to
  its maximum throughout; it only starts saying something once the footstep
  plan carries terrain heights.
  """
  term = _footstep_term(env, command_name, sub_command_name)
  pos, _ = term.foot_pose_w()
  error = pos[..., 2] - term.stance_target_w[..., 2]
  return _stance_reward(term, error, std)


def footstep_tracking_yaw(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  sub_command_name: str | None = None,
) -> torch.Tensor:
  """Reward planted feet for pointing along their footstep target's yaw."""
  term = _footstep_term(env, command_name, sub_command_name)
  _, rpy = term.foot_pose_w()
  error = wrap_to_pi(rpy[..., 2] - term.stance_target_w[..., 5])
  return _stance_reward(term, error, std)


def footstep_tracking_roll_pitch(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  sub_command_name: str | None = None,
) -> torch.Tensor:
  """Reward planted feet for matching their footstep target's roll and pitch.

  The targets are flat until terrain lands, so for now this rewards keeping the
  feet level -- overlapping with ``flat_foot_l2``. Keep its weight low (or zero)
  until the plan actually tilts the footsteps.
  """
  term = _footstep_term(env, command_name, sub_command_name)
  _, rpy = term.foot_pose_w()
  error = torch.norm(wrap_to_pi(rpy[..., :2] - term.stance_target_w[..., 3:5]), dim=-1)
  return _stance_reward(term, error, std)
