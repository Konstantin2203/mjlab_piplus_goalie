"""Custom domain-randomization terms for PiPlus.

These wrap mjlab's generic per-world model-field randomizer for fields that
mjlab does not expose a dedicated `dr.*` helper for (e.g. contact softness).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.dr._core import Ranges, _randomize_model_field
from mjlab.envs.mdp.dr._types import Distribution, Operation
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@requires_model_fields("geom_solref")
def geom_solref(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize contact softness via ``geom_solref``.

  ``geom_solref = (timeconst, dampratio)``. Axis 0 (``timeconst``) is the main
  softness knob: larger -> softer ground. Must stay ``>= 2 * sim.timestep``.
  Default axes is [0] (timeconst only). Pass ``axes=[0, 1]`` to also vary the
  damping ratio.
  """
  _randomize_model_field(
    env,
    env_ids,
    "geom_solref",
    entity_type="geom",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0],
    valid_axes=[0, 1],
  )


@requires_model_fields("geom_solimp")
def geom_solimp(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize contact impedance via ``geom_solimp``.

  ``geom_solimp = (dmin, dmax, width, midpoint, power)``. Axis 1 (``dmax``,
  the max impedance) is the relevant softness/penetration knob.
  """
  _randomize_model_field(
    env,
    env_ids,
    "geom_solimp",
    entity_type="geom",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[1],
    valid_axes=[0, 1, 2, 3, 4],
  )
