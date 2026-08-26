"""Viewer control for shoving the robot around by hand.

The training-time disturbance is ``push_by_setting_velocity`` on an interval
timer: it adds a uniformly sampled 6D delta straight to the root velocity, a
mass-independent instantaneous kick. :class:`PushGui` is the manual version of
exactly that -- six sliders for the delta and a button that applies it once to
the env the viewer is looking at -- so what you can provoke by hand is the same
kind of disturbance the policy was trained against.

Wiring
------
mjlab's viser viewer only offers *command terms* a hook into the GUI
(``CommandTerm.create_gui``), and pushing is not a command, so this rides along
on whichever command term owns the viewer panel: :class:`MultiCommand` in the
multi task, :class:`UniformVelocityCommand` when it runs standalone. A term that
is owned by a composite command must not build a second panel -- the same rule
its own joystick controls already follow -- so the owner builds it and the
sub-terms skip it.

The button callback runs on viser's server thread while the sim steps on the
main one, so the click only sets a flag; :meth:`apply` does the write, from the
command term's ``compute``. A push requested while the viewer is paused lands on
the first step after it resumes rather than being dropped.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  import viser
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["PushGui"]

# Slider extents, in m/s and rad/s of root-velocity delta.
LIN_RANGE = 3.0
ANG_RANGE = 6.0

_LIN_LABELS = ("lin x [m/s]", "lin y [m/s]", "lin z [m/s]")
_ANG_LABELS = ("ang x [rad/s]", "ang y [rad/s]", "ang z [rad/s]")


class PushGui:
  """A "Push" folder: 6 sliders for the velocity delta, and a button.

  Args:
    asset_name: scene entity to push.
    lin_range: linear slider extent, +/- m/s.
    ang_range: angular slider extent, +/- rad/s.
  """

  def __init__(
    self,
    asset_name: str = "robot",
    lin_range: float = LIN_RANGE,
    ang_range: float = ANG_RANGE,
  ) -> None:
    self._asset_name = asset_name
    self._lin_range = lin_range
    self._ang_range = ang_range
    self._sliders: list[viser.GuiSliderHandle] = []
    self._heading_frame: viser.GuiCheckboxHandle | None = None
    self._get_env_idx: Callable[[], int] | None = None
    self._pending = False

  def create_gui(
    self,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    folder_name: str = "Push",
  ) -> None:
    from viser import Icon

    axes = [(label, self._lin_range) for label in _LIN_LABELS]
    axes += [(label, self._ang_range) for label in _ANG_LABELS]

    with server.gui.add_folder(folder_name):
      for label, extent in axes:
        self._sliders.append(
          server.gui.add_slider(
            label,
            min=-extent,
            max=extent,
            step=0.05 if extent <= self._lin_range else 0.1,
            initial_value=0.0,
          )
        )
      # Sliders read as "from the front / from the left" rather than "along
      # world +x", which is what you actually want when the robot has turned
      # away from you. Yaw only: after a fall the full body frame is arbitrary,
      # and a push meant to be horizontal would come out pointing at the floor.
      self._heading_frame = server.gui.add_checkbox("Heading frame", initial_value=True)

      push_btn = server.gui.add_button("Push robot", icon=Icon.KARATE)

      @push_btn.on_click
      def _(_) -> None:
        self._pending = True

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for slider in self._sliders:
          slider.value = 0.0

    self._get_env_idx = get_env_idx

  def apply(self, env: ManagerBasedRlEnv) -> None:
    """Apply a requested push to the viewed env. Call once per step."""
    if not self._pending:
      return
    self._pending = False
    if not self._sliders or self._get_env_idx is None:
      return

    asset = env.scene[self._asset_name]
    idx = self._get_env_idx()
    device = env.device

    delta = torch.tensor([s.value for s in self._sliders], device=device)
    if self._heading_frame is not None and self._heading_frame.value:
      heading = asset.data.heading_w[idx]
      cos_h, sin_h = torch.cos(heading), torch.sin(heading)
      for offset in (0, 3):  # linear block, then angular block
        x, y = delta[offset].clone(), delta[offset + 1].clone()
        delta[offset] = cos_h * x - sin_h * y
        delta[offset + 1] = sin_h * x + cos_h * y

    env_ids = torch.tensor([idx], device=device, dtype=torch.long)
    vel_w = asset.data.root_link_vel_w[env_ids] + delta
    asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)
