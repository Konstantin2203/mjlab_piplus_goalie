"""Reward terms whose weight depends on which command type is active.

Under a :class:`~mjlab_piplus.tasks.mdp.multi_command.MultiCommand` a reward is
often only meaningful for some command types: velocity tracking says nothing
under a path command, and the path rewards say nothing under a velocity command.
:class:`MultiRewardTermCfg` adds a ``command_weights`` field for exactly that.

The gating is self-contained -- no custom ``RewardManager`` is needed, because
``ManagerBase._resolve_common_term_cfg`` instantiates any class-valued ``func``
as ``func(cfg=term_cfg, env=env)``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.reward_manager import RewardTermCfg

from mjlab_piplus.tasks.mdp.multi_command import MultiCommand

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

__all__ = [
  "MultiRewardTermCfg",
  "with_command_weights",
]


class _CommandWeightedReward:
  """Evaluates a reward term, then scales it by the active command's weight."""

  def __init__(self, cfg: MultiRewardTermCfg, env: ManagerBasedRlEnv) -> None:
    inner = cfg.inner_func
    assert inner is not None, "MultiRewardTermCfg.__post_init__ did not run."
    self._inner = inner(cfg=cfg, env=env) if inspect.isclass(inner) else inner

    multi = env.command_manager.get_term(cfg.multi_command_name)
    if not isinstance(multi, MultiCommand):
      raise TypeError(
        f"MultiRewardTermCfg.multi_command_name='{cfg.multi_command_name}' does "
        f"not resolve to a MultiCommand (got {type(multi).__name__})."
      )
    self._multi = multi

    weights = torch.ones(multi.num_types, device=env.device)
    for name, value in (cfg.command_weights or {}).items():
      if name not in multi.term_names:
        raise KeyError(
          f"command_weights key '{name}' is not a configured command type. "
          f"Configured: {list(multi.term_names)}."
        )
      weights[multi.term_names.index(name)] = float(value)
    self._weights = weights

  def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
    # The inner term is evaluated unconditionally, even where its weight is 0.
    # Stateful terms (e.g. ``path_progress_reward``, which tracks the previous
    # closest path index) would otherwise go stale while inactive and emit a
    # large spurious value the step they are re-activated.
    value = self._inner(env, **params)
    return value * self._weights[self._multi.active_type]

  def __getattr__(self, item: str) -> Any:
    # Delegate ``reset`` / ``debug_vis`` so the RewardManager's ``hasattr``
    # probes see exactly what the unwrapped term would expose.
    if item.startswith("_"):
      raise AttributeError(item)
    inner = self.__dict__.get("_inner")
    if inner is None:
      raise AttributeError(item)
    return getattr(inner, item)


@dataclass(kw_only=True)
class MultiRewardTermCfg(RewardTermCfg):
  """A reward term that is weighted per active command type."""

  command_weights: dict[str, float] | None = None
  """Per-command-type weight multiplier. Types not listed default to ``1.0``;
  ``None`` means the term is always at full weight. Use ``0.0`` to switch a term
  off for a command type entirely."""

  multi_command_name: str = "command"
  """Name of the ``MultiCommand`` term whose active type selects the weight."""

  inner_func: Any = field(default=None, init=False)
  """The user's reward callable. ``func`` is replaced by the gating wrapper in
  ``__post_init__``; the original is kept here for it to call."""

  def __post_init__(self) -> None:
    if self.func is _CommandWeightedReward:
      return  # Already wrapped (e.g. re-constructed from an existing cfg).
    self.inner_func = self.func
    self.func = _CommandWeightedReward


def with_command_weights(
  term_cfg: RewardTermCfg,
  command_weights: dict[str, float],
  multi_command_name: str = "command",
) -> MultiRewardTermCfg:
  """Clone ``term_cfg`` into a :class:`MultiRewardTermCfg` with the given weights.

  Lets the multi env cfg reweight a reward inherited from the velocity env in a
  single line, without restating its func and params.
  """
  inner = getattr(term_cfg, "inner_func", None) or term_cfg.func
  return MultiRewardTermCfg(
    func=inner,
    weight=term_cfg.weight,
    params=term_cfg.params,
    command_weights=command_weights,
    multi_command_name=multi_command_name,
  )
