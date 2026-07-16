from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch

from mjlab_piplus.tasks.mdp.commands import GoalPoseCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "EventCurriculumStage",
  "event_curriculum",
  "GoalRadiusStage",
  "commands_goal_radius",
  "RewardWeightStage",
  "reward_weight",
]


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Ramp a reward term's weight over training steps.

  Later stages take precedence once ``env.common_step_counter`` passes their
  ``step`` threshold. Mirrors the ``reward_weight`` curriculum shipped with some
  mjlab builds, which this project's mjlab version does not include.
  """
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])


class GoalRadiusStage(TypedDict):
  step: int
  radius_range: tuple[float, float]


def commands_goal_radius(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  radius_stages: list[GoalRadiusStage],
) -> dict[str, torch.Tensor]:
  """Ramp the goal-pose sampling radius over training steps.

  Later stages take precedence once ``env.common_step_counter`` reaches their
  ``step`` threshold.
  """
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(GoalPoseCommandCfg, command_term.cfg)
  for stage in radius_stages:
    if env.common_step_counter >= stage["step"]:
      cfg.radius_range = stage["radius_range"]
  return {
    "radius_min": torch.tensor(cfg.radius_range[0]),
    "radius_max": torch.tensor(cfg.radius_range[1]),
  }


class _EventCurriculumStageOptional(TypedDict, total=False):
  params: dict[str, Any]


class EventCurriculumStage(_EventCurriculumStageOptional):
  step: int


def event_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str,
  stages: list[EventCurriculumStage],
) -> dict[str, torch.Tensor]:
  """Update an event term's params based on training steps.

  Each stage specifies a ``step`` threshold and a ``params`` dict. When
  ``env.common_step_counter`` reaches a stage's ``step``, the params are
  applied. Later stages take precedence. Params are deep-merged: nested
  dicts are updated recursively rather than replaced wholesale.
  """
  del env_ids
  event_term_cfg = env.event_manager.get_term_cfg(event_name)
  for stage in stages:
    if env.common_step_counter >= stage["step"]:
      if "params" in stage:
        _deep_update(event_term_cfg.params, stage["params"])
  result: dict[str, torch.Tensor] = {}
  for k, v in event_term_cfg.params.items():
    if isinstance(v, (int, float)):
      result[k] = torch.tensor(v)
    elif isinstance(v, tuple) and len(v) == 2:
      result[k + "_min"] = torch.tensor(v[0])
      result[k + "_max"] = torch.tensor(v[1])
    elif isinstance(v, torch.Tensor):
      result[k] = v
  return result


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
  for k, v in source.items():
    if k in target and isinstance(target[k], dict) and isinstance(v, dict):
      _deep_update(target[k], v)
    else:
      target[k] = v
