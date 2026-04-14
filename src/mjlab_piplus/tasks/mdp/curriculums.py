from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["EventCurriculumStage", "event_curriculum"]


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
