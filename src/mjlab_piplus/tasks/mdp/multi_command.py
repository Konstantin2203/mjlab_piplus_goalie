"""Composite command that switches between several command types per env.

``MultiCommand`` owns a set of sub-command terms (e.g. ``velocity``, ``path``,
``stand``) and assigns exactly one of them to each environment. The policy sees
every sub-command concatenated, with the inactive blocks zeroed and a one-hot
prefix naming the active type:

    [one_hot(K) | sub_0.command | sub_1.command | ...]

Sub-commands that expose no observation (``stand``) contribute nothing to the
tail; the one-hot is their whole signal.

Resampling
----------
The type is redrawn on the MultiCommand's own timer, but the *next* tick is
armed from the selected sub-command's ``resampling_time_range``. A velocity
command therefore keeps its short 3-8 s cadence while a path is given the much
longer window it needs to actually be walked.

At an episode reset there is no command to "keep", so the type is drawn straight
from ``fractions``. Mid-episode the type changes with probability
``change_command_probability`` (drawing from ``fractions`` renormalized over the
*other* types), and is otherwise kept.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  import viser
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

__all__ = [
  "MultiCommand",
  "MultiCommandCfg",
  "StandCommand",
  "StandCommandCfg",
  "is_command_active",
  "stand_mask",
  "velocity_active",
  "velocity_command",
]


##
# Stand command.
##


class StandCommand(CommandTerm):
  """A command that says nothing except "stand still".

  Contributes zero observation dimensions -- the MultiCommand one-hot already
  tells the policy that stand is active. Its only content is ``wants_stand``,
  which the reward terms query through :func:`stand_mask`.
  """

  cfg: StandCommandCfg

  def __init__(self, cfg: StandCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self._empty = torch.zeros(self.num_envs, 0, device=self.device)
    self._wants_stand = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._empty

  @property
  def wants_stand(self) -> torch.Tensor:
    return self._wants_stand

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    pass

  def _update_command(self) -> None:
    pass


@dataclass(kw_only=True)
class StandCommandCfg(CommandTermCfg):
  resampling_time_range: tuple[float, float] = (2.0, 5.0)

  def build(self, env: ManagerBasedRlEnv) -> StandCommand:
    return StandCommand(self, env)


##
# Multi command.
##


class MultiCommand(CommandTerm):
  cfg: MultiCommandCfg

  def __init__(self, cfg: MultiCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)

    self.term_names: tuple[str, ...] = tuple(cfg.terms.keys())
    self.terms: dict[str, CommandTerm] = {
      name: term_cfg.build(env) for name, term_cfg in cfg.terms.items()
    }
    self.num_types = len(self.term_names)
    self._type_index = {name: i for i, name in enumerate(self.term_names)}

    fractions = torch.tensor(
      [float(cfg.fractions[name]) for name in self.term_names],
      device=self.device,
      dtype=torch.float32,
    )
    self._fractions = fractions / fractions.sum()

    # Start the population at the configured fractions rather than all-type-0,
    # so the very first episode is already balanced.
    self.active_type = torch.multinomial(
      self._fractions.unsqueeze(0).expand(self.num_envs, -1), 1
    ).squeeze(1)
    # Envs whose next resample must ignore the current type (episode reset).
    self._redraw_from_fractions = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    for name in self.term_names:
      self.metrics[f"active_{name}"] = torch.zeros(self.num_envs, device=self.device)

    # Set by create_gui() when the viewer is active.
    self._gui_override: viser.GuiCheckboxHandle | None = None
    self._gui_type: viser.GuiDropdownHandle | None = None
    self._gui_get_env_idx: Callable[[], int] | None = None

  # Properties.

  @property
  def command(self) -> torch.Tensor:
    one_hot = torch.nn.functional.one_hot(
      self.active_type, num_classes=self.num_types
    ).float()
    parts = [one_hot]
    for i, name in enumerate(self.term_names):
      sub = self.terms[name].command
      if sub.shape[1] == 0:
        continue
      mask = (self.active_type == i).unsqueeze(1).to(sub.dtype)
      parts.append(sub * mask)
    return torch.cat(parts, dim=1)

  def active_mask(self, name: str) -> torch.Tensor:
    """Boolean ``(num_envs,)`` mask: is ``name`` the active type?"""
    if name not in self._type_index:
      raise KeyError(
        f"Unknown command type '{name}'. Configured: {list(self.term_names)}."
      )
    return self.active_type == self._type_index[name]

  @property
  def stand_mask(self) -> torch.Tensor:
    """Per-env "the robot should stand still" mask.

    True where the active sub-command asks to stand. ``StandCommand`` always
    does; other terms opt in with a ``wants_stand`` property (e.g. the path
    command once it has settled on its goal pose).
    """
    mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    for i, name in enumerate(self.term_names):
      wants = getattr(self.terms[name], "wants_stand", None)
      if wants is None:
        continue
      mask |= wants & (self.active_type == i)
    return mask

  @property
  def velocity_command(self) -> torch.Tensor:
    """Commanded ``[vx, vy, wz]``; zeros where velocity is not the active type."""
    out = torch.zeros(self.num_envs, 3, device=self.device)
    name = self.cfg.velocity_command_name
    if name not in self.terms:
      return out
    sub = self.terms[name].command
    return sub[:, :3] * self.active_mask(name).unsqueeze(1).float()

  # Command lifecycle.

  def compute(self, dt: float) -> None:
    self._update_metrics()
    for term in self.terms.values():
      term._update_metrics()
    self.time_left -= dt
    resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
    if len(resample_env_ids) > 0:
      self._resample(resample_env_ids)
    self._update_command()
    self._apply_gui_override()

  def _update_metrics(self) -> None:
    # Fraction of episode time spent in each type (mean over the episode gives
    # the realized time share, which is what ``fractions`` is meant to control).
    share = self._env.step_dt / self._env.max_episode_length_s
    for i, name in enumerate(self.term_names):
      self.metrics[f"active_{name}"] += (self.active_type == i).float() * share

  def _update_command(self) -> None:
    for term in self.terms.values():
      term._update_command()

  def _resample(self, env_ids: torch.Tensor) -> None:
    """Redraw the type, resample the selected sub-command, arm its timer.

    Overrides the base implementation, which would arm ``time_left`` from this
    term's own ``resampling_time_range``. Here the interval comes from whichever
    sub-command was selected.
    """
    if len(env_ids) == 0:
      return

    self._sample_type(env_ids)
    self.command_counter[env_ids] += 1

    active = self.active_type[env_ids]
    for i, name in enumerate(self.term_names):
      selected = env_ids[active == i]
      if len(selected) == 0:
        continue
      # ``_resample`` (not ``_resample_command``) so the sub-term's
      # ``command_counter`` increments -- ``path_progress_reward`` keys its
      # state reset off that counter.
      self.terms[name]._resample(selected)
      low, high = self.cfg.terms[name].resampling_time_range
      self.time_left[selected] = torch.empty(
        len(selected), device=self.device
      ).uniform_(low, high)

  def _sample_type(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    current = self.active_type[env_ids]
    fresh = self._redraw_from_fractions[env_ids]

    probs = self._fractions.unsqueeze(0).repeat(n, 1)
    drawn_fresh = torch.multinomial(probs, 1).squeeze(1)

    if self.num_types > 1:
      # Draw a *different* type: zero out the current one and renormalize, so
      # ``change_command_probability`` is literally the probability of changing.
      other = probs.clone()
      other.scatter_(1, current.unsqueeze(1), 0.0)
      total = other.sum(dim=1, keepdim=True)
      degenerate = total.squeeze(1) <= 0.0
      other = torch.where(total > 0.0, other / total.clamp(min=1e-12), probs)
      drawn_other = torch.multinomial(other, 1).squeeze(1)
      change = (
        torch.rand(n, device=self.device) < self.cfg.change_command_probability
      ) & ~degenerate
      kept_or_switched = torch.where(change, drawn_other, current)
    else:
      kept_or_switched = current

    self.active_type[env_ids] = torch.where(fresh, drawn_fresh, kept_or_switched)
    self._redraw_from_fractions[env_ids] = False

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Unused: ``_resample`` is overridden and drives the sub-terms directly.
    raise AssertionError("MultiCommand._resample_command should never be called.")

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    extras: dict[str, float] = {}
    for metric_name, metric_value in self.metrics.items():
      extras[metric_name] = torch.mean(metric_value[env_ids]).item()
      metric_value[env_ids] = 0.0
    for name, term in self.terms.items():
      for metric_name, metric_value in term.metrics.items():
        extras[f"{name}/{metric_name}"] = torch.mean(metric_value[env_ids]).item()
        metric_value[env_ids] = 0.0
      term.command_counter[env_ids] = 0
    self.command_counter[env_ids] = 0
    # A fresh episode has no command to keep: draw the type from ``fractions``.
    self._redraw_from_fractions[env_ids] = True
    self._resample(env_ids)
    return extras

  # Visualization / GUI.

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    active = self.active_type
    for i, name in enumerate(self.term_names):
      term = self.terms[name]
      if not term.cfg.debug_vis:
        continue
      # Only draw a sub-command that is active for one of the visualized envs,
      # so a stale path is not rendered under a velocity command.
      if not any(int(active[batch]) == i for batch in env_indices):
        continue
      term._debug_vis_impl(visualizer)

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    with server.gui.add_folder(f"{name.capitalize()} type"):
      self._gui_override = server.gui.add_checkbox("Override", initial_value=False)
      self._gui_type = server.gui.add_dropdown(
        "Active", options=self.term_names, initial_value=self.term_names[0]
      )
    self._gui_get_env_idx = get_env_idx
    for sub_name, term in self.terms.items():
      term.create_gui(
        sub_name,
        server,
        get_env_idx,
        on_change=on_change,
        request_action=request_action,
      )

  def _apply_gui_override(self) -> None:
    if self._gui_override is None or not self._gui_override.value:
      return
    assert self._gui_type is not None and self._gui_get_env_idx is not None
    idx = self._gui_get_env_idx()
    self.active_type[idx] = self._type_index[self._gui_type.value]

  def on_viewer_pause(self, paused: bool) -> None:
    for term in self.terms.values():
      term.on_viewer_pause(paused)

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    applied = False
    for term in self.terms.values():
      applied |= term.apply_gui_reset(env_ids)
    return applied


@dataclass(kw_only=True)
class MultiCommandCfg(CommandTermCfg):
  terms: dict[str, CommandTermCfg] = field(default_factory=dict)
  """The sub-commands, in observation order. Each contributes its own
  ``command`` block to the tail (a zero-width block contributes nothing)."""

  fractions: dict[str, float] = field(default_factory=dict)
  """Relative sampling weight per type; normalized internally. Keys must match
  ``terms``."""

  change_command_probability: float = 0.3
  """Probability that a mid-episode resample switches to a *different* type.
  With probability ``1 - change_command_probability`` the type is kept (its
  content is resampled either way). At an episode reset the type is always
  drawn fresh from ``fractions``."""

  velocity_command_name: str = "velocity"
  """Which sub-command :func:`velocity_command` reads the commanded twist from."""

  resampling_time_range: tuple[float, float] = (0.0, 0.0)
  """Unused. The next resample is armed from the *selected* sub-command's
  ``resampling_time_range`` -- see ``MultiCommand._resample``."""

  def build(self, env: ManagerBasedRlEnv) -> MultiCommand:
    return MultiCommand(self, env)

  def __post_init__(self) -> None:
    if not self.terms:
      raise ValueError("MultiCommandCfg.terms must not be empty.")
    if set(self.fractions) != set(self.terms):
      raise ValueError(
        "MultiCommandCfg.fractions keys must match terms keys: "
        f"{sorted(self.fractions)} != {sorted(self.terms)}."
      )
    if any(v < 0.0 for v in self.fractions.values()):
      raise ValueError("MultiCommandCfg.fractions must be non-negative.")
    if sum(self.fractions.values()) <= 0.0:
      raise ValueError("MultiCommandCfg.fractions must not sum to zero.")
    if not 0.0 <= self.change_command_probability <= 1.0:
      raise ValueError(
        "MultiCommandCfg.change_command_probability must be in [0, 1], got "
        f"{self.change_command_probability}."
      )


##
# Accessors used by the reward / observation terms.
##


def is_command_active(
  env: ManagerBasedRlEnv, command_name: str, sub_command_name: str
) -> torch.Tensor:
  """``(num_envs,)`` bool: is ``sub_command_name`` the active type?

  For a non-composite command term every env is "active", so this returns all
  True -- which keeps terms that use it working in the standalone velocity task.
  """
  term = env.command_manager.get_term(command_name)
  if isinstance(term, MultiCommand):
    return term.active_mask(sub_command_name)
  return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)


def stand_mask(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """``(num_envs,)`` bool: should the robot stand still (freeze the gait)?

  Queries the stand command when ``command_name`` is a :class:`MultiCommand`,
  and otherwise falls back to the stand flag at index 3 of
  ``UniformVelocityCommandWithStop``.
  """
  term = env.command_manager.get_term(command_name)
  if isinstance(term, MultiCommand):
    return term.stand_mask
  assert term is not None, f"Command '{command_name}' not found."
  return term.command[:, 3] > 0.5


def velocity_active(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """``(num_envs,)`` bool: is the velocity command the active type?

  All True for a non-composite command term, so terms that use this keep their
  behavior in the standalone velocity task.
  """
  term = env.command_manager.get_term(command_name)
  if isinstance(term, MultiCommand):
    name = term.cfg.velocity_command_name
    if name not in term.terms:
      return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return term.active_mask(name)
  return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)


def velocity_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """``(num_envs, 3)`` commanded ``[vx, vy, wz]``.

  Zeros on envs whose active command type is not the velocity command, so a
  path- or stand-commanded env reads a zero twist.
  """
  term = env.command_manager.get_term(command_name)
  if isinstance(term, MultiCommand):
    return term.velocity_command
  assert term is not None, f"Command '{command_name}' not found."
  return term.command[:, :3]
