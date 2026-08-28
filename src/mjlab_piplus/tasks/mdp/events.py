"""Project-local event terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.events import push_by_setting_velocity, resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_piplus.tasks.mdp.multi_command import stand_mask

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["push_by_setting_velocity_unless_standing"]

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def push_by_setting_velocity_unless_standing(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  velocity_range: dict[str, tuple[float, float]],
  command_name: str = "command",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """``push_by_setting_velocity``, skipped for envs under a stand command.

  A push is a disturbance the policy is meant to *walk off*. Under a stand
  command the robot is being scored for holding still, so a kick there charges
  it for a displacement it did not choose and then charges it again for the
  correction -- the same event means "recover" in one regime and "you failed" in
  the other.

  Filtering happens on ``env_ids`` rather than by zeroing the velocity delta, so
  a standing env is left completely untouched: no sampling, no write to sim.
  Note this is evaluated at *fire* time against the command active right then, so
  an env that is pushed and then switched to stand still has to deal with the
  push it already took.
  """
  env_ids = resolve_env_ids(env, env_ids)
  standing = stand_mask(env, command_name)[env_ids.long()]
  env_ids = env_ids[~standing]
  if env_ids.numel() == 0:
    return
  push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg)
