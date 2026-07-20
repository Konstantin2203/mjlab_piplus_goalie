"""Custom termination terms for PiPlus."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.termination_manager import TerminationTermCfg

__all__ = ["time_out_jittered"]


class time_out_jittered:
  """Episode time-out at a per-env random horizon (mark with ``time_out=True``).

  Drop-in replacement for ``mdp.time_out``. The plain version fires exactly at
  ``max_episode_length`` for every env, so once the policy can survive a full
  episode all envs time out in lockstep and reset together. That synchronized
  reset shows up as a periodic dip in rewards (and a spike in ``is_terminated``).
  
  """

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv) -> None:
    self._env = env
    self._min_fraction: float = cfg.params.get("min_fraction", 0.8)
    self._horizon = torch.empty(
      env.num_envs, dtype=torch.long, device=env.device
    )
    # The framework issues a reset over ALL envs immediately after construction,
    # which calls reset() below. Without special-casing, that first reset would
    # resample every horizon into the narrow [min_fraction*max, max] band, so all
    # envs would time out within one short window and only slowly decorrelate
    # (a damped ring at ~episode-length period). This flag makes that first reset
    # seed the horizons across the FULL [1, max] range instead, so episodes are
    # staggered from step 0.
    self._seeded = False
    self._resample(
      torch.arange(env.num_envs, device=env.device), full_range=True
    )

  def _resample(self, env_ids: torch.Tensor, full_range: bool = False) -> None:
    if full_range:
      lo, hi = 1, int(self._env.max_episode_length)
    else:
      lo = max(1, int(self._min_fraction * self._env.max_episode_length))
      hi = int(self._env.max_episode_length)
    self._horizon[env_ids] = torch.randint(
      lo, hi + 1, (len(env_ids),), device=self._env.device
    )

  def __call__(self, env: ManagerBasedRlEnv, min_fraction: float = 0.8) -> torch.Tensor:
    del min_fraction  # Consumed in __init__.
    return env.episode_length_buf >= self._horizon

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self._env.num_envs, device=self._env.device)
    # First reset (the startup all-env reset): keep the full-range staggering so
    # the initial episodes don't all end together. Subsequent per-env resets use
    # the tight [min_fraction*max, max] band.
    if not self._seeded:
      self._seeded = True
      self._resample(env_ids, full_range=True)
      return
    self._resample(env_ids)
