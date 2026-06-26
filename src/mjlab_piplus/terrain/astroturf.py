"""Astroturf terrain: a soft contact surface over a hard base.

Models a compliant surface (astroturf, foam, soft ground) as two stacked planes:

- A **soft top plane** at ``z = 0`` — the surface the robot rests on. Its contact
  ``solref`` is configurable (and ramped by a curriculum), so under load the foot
  sinks into the turf. It is given a higher contact ``priority`` than the robot's
  foot geoms (priority 1) so that MuJoCo uses *its* ``solref`` for the foot-turf
  contact rather than the foot's (see the contact-priority note in the
  ``ground_softness`` event — without this the surface solref is ignored).
- A **hard base plane** at ``z = -turf_thickness`` — a rigid backstop. Once the
  foot sinks ``turf_thickness`` into the soft layer it hits this and stops, giving
  the "soft for the first cm, then firm" feel of real astroturf.

mjlab's ``Scene`` constructs ``TerrainEntity`` directly (no class hook), and its
``terrain_type`` only knows ``"plane"`` / ``"generator"``. Rather than edit the
installed package, this module subclasses ``TerrainEntity`` to add an
``"astroturf"`` type (delegating all other types to the base implementation) and
rebinds the name the scene resolves so the subclass is used. The rebind is a safe
superset: behaviour is unchanged for every non-astroturf terrain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg

_PLANE = mujoco.mjtGeom.mjGEOM_PLANE

# Geom names. The base keeps the name ``terrain`` so the default checker material
# (regex ``terrain$``) and terrain-only sensors still match it.
BASE_GEOM_NAME = "terrain"
TOP_GEOM_NAME = "terrain_top"


@dataclass
class AstroturfTerrainCfg(TerrainEntityCfg):
  """Two-plane astroturf terrain (soft top over a hard base)."""

  terrain_type: Literal["generator", "plane", "astroturf"] = "astroturf"  # type: ignore[assignment]

  top_solref: tuple[float, float] = (0.02, 1.0)
  """Contact ``solref`` (timeconst, dampratio) of the soft top plane. Larger
  timeconst -> softer. Must stay >= 2 * sim.timestep. Ramped by the curriculum."""
  turf_thickness: float = 0.02
  """Vertical gap (m) between the soft top plane (z=0) and the hard base plane."""
  top_priority: int = 2
  """Contact priority of the top plane. Must exceed the robot foot geom priority
  (1) so the top plane's solref/friction govern the foot-turf contact."""
  top_friction: tuple[float, float, float] = (0.6, 0.005, 0.0001)
  """Friction of the top plane. Because ``top_priority`` wins the contact, this
  (not the per-foot friction DR) sets the turf friction."""
  top_condim: int = 3
  """Contact dimensionality of the top plane (match the robot feet)."""


class AstroturfTerrain(TerrainEntity):
  """``TerrainEntity`` that additionally supports ``terrain_type="astroturf"``."""

  cfg: AstroturfTerrainCfg

  def _build_spec(self) -> None:
    if self.cfg.terrain_type != "astroturf":
      super()._build_spec()
      return

    self._spec = mujoco.MjSpec()
    body = self._spec.worldbody.add_body(name="terrain")

    # Hard base plane, a rigid backstop below the turf.
    base = body.add_geom(
      name=BASE_GEOM_NAME,
      type=_PLANE,
      size=(0.0, 0.0, 0.01),
      pos=(0.0, 0.0, -self.cfg.turf_thickness),
    )
    base.priority = self.cfg.top_priority  # backstop should also win vs the feet

    # Soft top plane: the surface the robot rests on (z=0).
    top = body.add_geom(
      name=TOP_GEOM_NAME,
      type=_PLANE,
      size=(0.0, 0.0, 0.01),
      pos=(0.0, 0.0, 0.0),
    )
    top.priority = self.cfg.top_priority
    top.condim = self.cfg.top_condim
    top.solref = list(self.cfg.top_solref)
    top.friction = list(self.cfg.top_friction)
    # Semi-transparent green so the checker base still shows through.
    top.rgba = (0.25, 0.6, 0.25, 0.45)

    self._configure_env_origins()
    self._flat_patches = {}
    self._flat_patch_radii = {}

    if self.cfg.debug_vis:
      self._add_env_origin_sites()


def _install() -> None:
  """Rebind the name ``Scene`` resolves so it builds :class:`AstroturfTerrain`."""
  import mjlab.scene.scene as _scene_module

  _scene_module.TerrainEntity = AstroturfTerrain


_install()
