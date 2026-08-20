"""Goal-pose command for the PiPlus go-to-pose task.

Samples a 2D target position (relative to the robot's current base position,
at a random distance/bearing) plus a random target yaw. The command exposed to
the policy is the goal expressed in the robot base frame:

    [dx_b, dy_b, yaw_err]

where ``(dx_b, dy_b)`` is the goal position in the base frame and ``yaw_err`` is
the wrapped yaw difference to the goal orientation. This 3-vector mirrors with
signs ``[1, -1, -1]`` under a left-right reflection, matching the velocity
command layout used by ``symmetry.py`` (so the actor obs layout is unchanged).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, wrap_to_pi

if TYPE_CHECKING:
  import viser
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class GoalPoseCommand(CommandTerm):
  cfg: GoalPoseCommandCfg

  def __init__(self, cfg: GoalPoseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]

    # Goal expressed in the world frame.
    self.goal_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.goal_yaw_w = torch.zeros(self.num_envs, device=self.device)

    # Goal expressed in the robot base frame: [dx_b, dy_b, yaw_err].
    self.goal_command_b = torch.zeros(self.num_envs, 3, device=self.device)

    self.is_standing_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_yaw"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    # 4-dim [dx_b, dy_b, yaw_err, stand_flag] so the shared gait terms
    # (``phase``/``phase_metric``) can read the stand flag at index 3 uniformly.
    # Standing envs (goal == current pose) freeze the gait; the flag is
    # reflection-invariant, so ``goal_pose_command`` (which only reads indices
    # 0-2) and the symmetry layout are unaffected.
    stand_flag = self.is_standing_env.float().unsqueeze(1)
    return torch.cat([self.goal_command_b, stand_flag], dim=1)

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_pos"] += (
      torch.norm(self.goal_command_b[:, :2], dim=1) / max_command_step
    )
    self.metrics["error_yaw"] += torch.abs(self.goal_command_b[:, 2]) / max_command_step

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)

    base_pos_w = self.robot.data.root_link_pos_w[env_ids, :2]

    radius = r.uniform_(*self.cfg.radius_range)
    bearing = torch.empty(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)
    offset = torch.stack(
      [radius * torch.cos(bearing), radius * torch.sin(bearing)], dim=1
    )
    self.goal_pos_w[env_ids] = base_pos_w + offset
    self.goal_yaw_w[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
      *self.cfg.yaw_range
    )

    # Standing envs: goal == current pose so the robot learns to hold still.
    is_standing = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
    self.is_standing_env[env_ids] = is_standing
    stand_ids = env_ids[is_standing]
    if len(stand_ids) > 0:
      self.goal_pos_w[stand_ids] = self.robot.data.root_link_pos_w[stand_ids, :2]
      self.goal_yaw_w[stand_ids] = self.robot.data.heading_w[stand_ids]

  def _update_command(self) -> None:
    base_pos_w = self.robot.data.root_link_pos_w[:, :2]
    heading_w = self.robot.data.heading_w

    delta_w = self.goal_pos_w - base_pos_w  # [N, 2]
    cos_h = torch.cos(heading_w)
    sin_h = torch.sin(heading_w)
    # Rotate world-frame delta into the base frame (rotation by -heading).
    self.goal_command_b[:, 0] = cos_h * delta_w[:, 0] + sin_h * delta_w[:, 1]
    self.goal_command_b[:, 1] = -sin_h * delta_w[:, 0] + cos_h * delta_w[:, 1]
    self.goal_command_b[:, 2] = wrap_to_pi(self.goal_yaw_w - heading_w)

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    goal_pos_ws = self.goal_pos_w.cpu().numpy()
    goal_yaw_ws = self.goal_yaw_w.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()

    arrow_len = self.cfg.viz.arrow_length
    z = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      gx, gy = goal_pos_ws[batch]
      gyaw = goal_yaw_ws[batch]
      goal_xyz = np.array([gx, gy, z])

      # Line from robot to goal (yellow).
      robot_xyz = np.array([base_pos_w[0], base_pos_w[1], z])
      visualizer.add_arrow(robot_xyz, goal_xyz, color=(0.9, 0.9, 0.1, 0.5), width=0.01)

      # Goal orientation arrow (red).
      heading_to = goal_xyz + np.array(
        [arrow_len * math.cos(gyaw), arrow_len * math.sin(gyaw), 0.0]
      )
      visualizer.add_arrow(goal_xyz, heading_to, color=(0.9, 0.1, 0.1, 0.9), width=0.02)


@dataclass(kw_only=True)
class GoalPoseCommandCfg(CommandTermCfg):
  entity_name: str
  radius_range: tuple[float, float] = (0.5, 3.0)
  """Distance range (meters) from the robot's current base position at which the
  goal position is sampled."""
  yaw_range: tuple[float, float] = (-math.pi, math.pi)
  """Goal yaw sampling range (radians)."""
  rel_standing_envs: float = 0.0
  """Fraction of environments whose goal is set to the current pose (stand)."""

  @dataclass
  class VizCfg:
    z_offset: float = 0.05
    arrow_length: float = 0.3

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> GoalPoseCommand:
    return GoalPoseCommand(self, env)


class UniformVelocityCommandWithStop(CommandTerm):
  """Uniform velocity command with an explicit stand/walk-in-place flag.

  This is a standalone copy of mjlab's ``UniformVelocityCommand`` (no
  inheritance) extended with a 4th command element, ``stand_flag``:

      [vx, vy, wz, stand_flag]

  ``stand_flag`` decouples "should the feet keep stepping" from "is the
  commanded velocity zero". Standing envs (zero velocity) are split into two
  behaviors:

  - ``stand_flag == 1``: stand still (freeze the gait, plant the feet).
  - ``stand_flag == 0``: walk in place (keep marching despite zero velocity).

  Moving envs always get ``stand_flag == 0``. The flag is exposed to the policy
  as part of the command and is consumed by the ``phase`` observation to decide
  whether to zero the gait phase. The mirror sign for the flag is ``+1`` (a
  left-right reflection leaves it unchanged) — see ``symmetry.py``.
  """

  cfg: UniformVelocityCommandWithStopCfg

  def __init__(
    self, cfg: UniformVelocityCommandWithStopCfg, env: ManagerBasedRlEnv
  ) -> None:
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)
    self.is_walk_in_place_env = torch.zeros_like(self.is_heading_env)
    self.is_world_env = torch.zeros_like(self.is_heading_env)
    self.is_forward_env = torch.zeros_like(self.is_heading_env)

    # 1.0 = stand still (no stepping), 0.0 = keep stepping.
    self.stand_flag = torch.zeros(self.num_envs, 1, device=self.device)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None
    self._stand_checkbox: viser.GuiCheckboxHandle | None = None

  @property
  def command(self) -> torch.Tensor:
    if not self.cfg.expose_stand_flag:
      return self.vel_command_b
    return torch.cat([self.vel_command_b, self.stand_flag], dim=1)

  @property
  def wants_stand(self) -> torch.Tensor:
    """Per-env "stand still" request, read by ``MultiCommand.stand_mask``.

    Always False when the stand flag is not exposed, which is how the multi
    command configures this term: standing is its own command type there, so
    this term only ever produces walk / walk-in-place.
    """
    return self.stand_flag[:, 0] > 0.5

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    # Stand-still and walk-in-place are sampled as two independent probabilities.
    # Both zero the velocity command; they differ only in the gait: stand-still
    # freezes it (stand_flag=1), walk-in-place keeps stepping (stand_flag=0). On
    # the rare overlap (an env drawn as both) stand-still wins.
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
    self.is_walk_in_place_env[env_ids] = (
      r.uniform_(0.0, 1.0) <= self.cfg.rel_walk_in_place_envs
    ) & ~self.is_standing_env[env_ids]
    self.stand_flag[env_ids, 0] = self.is_standing_env[env_ids].float()

    # Randomly assign world-frame envs.
    self.is_world_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_world_envs
    # Copy sampled velocities as world-frame reference for world envs.
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

    # Forward-only envs: positive lin_vel_x, zero lateral and angular.
    self.is_forward_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs
    fwd_ids = env_ids[self.is_forward_env[env_ids]]
    if len(fwd_ids) > 0:
      self.vel_command_b[fwd_ids, 0] = (
        self.vel_command_b[fwd_ids, 0].abs().clamp(min=0.3)
      )
      self.vel_command_b[fwd_ids, 1] = 0.0
      self.vel_command_b[fwd_ids, 2] = 0.0

    init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    init_vel_env_ids = env_ids[init_vel_mask]
    if len(init_vel_env_ids) > 0:
      root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
      lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
      root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

  def _update_command(self) -> None:
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    # World-frame envs: rotate world-frame linear vel into body frame.
    if self.is_world_env.any():
      w_ids = self.is_world_env.nonzero(as_tuple=False).flatten()
      heading = self.robot.data.heading_w[w_ids]
      cos_h = torch.cos(heading)
      sin_h = torch.sin(heading)
      vx_w = self.vel_command_w[w_ids, 0]
      vy_w = self.vel_command_w[w_ids, 1]
      self.vel_command_b[w_ids, 0] = cos_h * vx_w + sin_h * vy_w
      self.vel_command_b[w_ids, 1] = -sin_h * vx_w + cos_h * vy_w

    # Zero the velocity for both stand-still and walk-in-place envs.
    zero_vel_env_ids = (
      (self.is_standing_env | self.is_walk_in_place_env)
      .nonzero(as_tuple=False)
      .flatten()
    )
    self.vel_command_b[zero_vel_env_ids, :] = 0.0
    self.vel_command_w[zero_vel_env_ids, :] = 0.0

  # GUI.

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create velocity joystick sliders in the Viser viewer."""
    from viser import Icon

    ranges = self.cfg.ranges

    axes = [
      ("lin_vel_x", ranges.lin_vel_x[1]),
      ("lin_vel_y", ranges.lin_vel_y[1]),
      ("ang_vel_z", ranges.ang_vel_z[1]),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, max_val in axes:
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=max_val,
          step=0.1,
          min=0.0,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-max_val,
          max=max_val,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      # Walk-in-place vs. stand-still toggle for the joystick env.
      self._stand_checkbox = server.gui.add_checkbox("Stand still", initial_value=False)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    # Store GUI state for compute() override.
    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, s in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = s.value
      if self._stand_checkbox is not None:
        self.stand_flag[idx, 0] = 1.0 if self._stand_checkbox.value else 0.0

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      # Skip if robot appears uninitialized (at origin).
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      # Helper to transform local to world coordinates.
      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      # Command linear velocity arrow (blue).
      cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      cmd_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
      )
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # Command angular velocity arrow (green).
      cmd_ang_from = cmd_lin_from
      cmd_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
      )
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      # Actual linear velocity arrow (cyan).
      act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      act_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
      )
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      # Actual angular velocity arrow (light green).
      act_ang_from = act_lin_from
      act_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
      )
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class UniformVelocityCommandWithStopCfg(CommandTermCfg):
  entity_name: str
  expose_stand_flag: bool = True
  """Whether ``command`` carries the stand flag as a 4th element.

  True (default) gives the standalone velocity task its ``[vx, vy, wz,
  stand_flag]`` layout. Set False when this term is a sub-command of a
  ``MultiCommand``, where standing is its own command type: the command is then
  the bare 3-dim twist and a zero-velocity command means walk-in-place."""
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  """Fraction of environments commanded to stand still: zero velocity with the
  gait frozen (stand_flag=1)."""
  rel_walk_in_place_envs: float = 0.5
  """Fraction of environments commanded to walk in place: zero velocity but the
  gait keeps stepping (stand_flag=0). Sampled independently of
  ``rel_standing_envs``; on the rare overlap, stand-still takes precedence."""
  rel_heading_envs: float = 1.0
  rel_world_envs: float = 0.0
  """Fraction of environments that use world-frame velocity commands.
  World-frame envs sample linear velocity in world frame and rotate to body
  frame each step, so the command direction stays fixed in the world."""
  rel_forward_envs: float = 0.0
  """Fraction of environments that receive forward-only commands (positive
  lin_vel_x, zero lin_vel_y and ang_vel_z). Increases training coverage for
  straight-line walking, which is important for stair climbing."""
  init_velocity_prob: float = 0.0

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommandWithStop:
    return UniformVelocityCommandWithStop(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
