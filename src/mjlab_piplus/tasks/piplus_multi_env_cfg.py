"""Multi-command env: one policy driven by a velocity, path, or stand command.

A single ``MultiCommand`` assigns each env one of three command types and
switches between them during the episode, so the policy learns the hand-over
between modes rather than being three separate policies.

The policy sees every sub-command concatenated with the inactive blocks zeroed,
prefixed by a one-hot naming the active type:

    [one_hot(5) | velocity(3) | path(2N + 2) | goal_pose(4) | kick(7)]

Standing is its own command type here, not a flag on the velocity command, so a
zero-velocity *velocity* command means walk-in-place. Reward terms that only
apply to some command types carry ``command_weights``; terms without the field
apply at full weight everywhere.

This config is standalone -- it deliberately does not derive from
``piplus_velocity_env_cfg`` -- so the two tasks can be tuned independently.
Shared settings (gait frequency, DR ranges, sensors, sim) are therefore
duplicated rather than inherited.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import NoiseModelWithAdditiveBiasCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab_piplus.objects import get_ball_cfg
from mjlab_piplus.robot.piplus_constants import get_piplus_robot_cfg
from mjlab_piplus.tasks import mdp as pi_mdp
from mjlab_piplus.tasks.mdp.multi_rewards import MultiRewardTermCfg
from mjlab_piplus.tasks.mdp.path_following import (
  PathCommandCfg,
  align_with_path_reward,
  close_to_path_reward,
  goal_reached_angular,
  goal_reached_linear,
  path_progress_reward,
)

DELAY_MIN = 1
DELAY_MAX = 1
WALK_FREQUENCY = 1.75

# Name of the composite command term. Every command-consuming reward,
# observation and metric points at this one name; the accessors in
# ``mdp/multi_command.py`` resolve the active sub-command behind it.
COMMAND = "command"

# Weight presets, spelled out so a reward's applicability reads at a glance.
_OFF = {"velocity": 0.0, "path": 0.0, "goal_pose": 0.0, "kick": 0.0, "stand": 0.0}
VELOCITY_AND_STAND = {**_OFF, "velocity": 1.0, "stand": 1.0}
PATH_ONLY = {**_OFF, "path": 1.0}
GOAL_POSE_ONLY = {**_OFF, "goal_pose": 1.0}
KICK_ONLY = {**_OFF, "kick": 1.0}

# Number of future path points in the command observation.
NUM_PATH_OBSERVATIONS = 8


def piplus_multi_env_cfg(
  play: bool = False, static: bool = False
) -> ManagerBasedRlEnvCfg:
  """Velocity + path + stand env driven by a single ``MultiCommand``.

  Reference paths are planned with A* through a random field of cubes and
  cylinders, so the robot has to walk corners and detours. The obstacles are
  virtual -- drawn in the viewer, never simulated -- so nothing about the scene,
  contacts or terminations changes.
  """
  del static  # Unused; kept for signature parity with the sibling tasks.
  foot_sites = ("l_foot", "r_foot")
  # Terrain height under each foot, registered in the scene below. The
  # "foot_height" critic observation that reads it is still commented out, so
  # nothing consumes the scan yet.
  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=(
      ObjRef(type="site", name="l_foot", entity="robot"),
      ObjRef(type="site", name="r_foot", entity="robot"),
    ),
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=False,
    viz=TerrainHeightSensorCfg.VizCfg(
      show_rays=True,
      hit_color=(1.0, 0.0, 1.0, 0.8),  # Magenta rays.
      hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
    ),
  )

  ##
  # Observations
  ##
  actor_terms = {  # careful: when changing, also change symmetry config
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/gyro"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
      delay_min_lag=DELAY_MIN,
      delay_max_lag=DELAY_MAX,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=NoiseModelWithAdditiveBiasCfg(
        noise_cfg=Unoise(n_min=-0.01, n_max=0.01),
        bias_noise_cfg=Unoise(n_min=-0.03, n_max=0.03, operation="add"),
        sample_bias_per_component=True,
      ),
      delay_min_lag=DELAY_MIN,
      delay_max_lag=DELAY_MAX,
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      # biased=True so the actor reads the encoder-biased position (true + bias),
      # matching the action manager which always compensates the target by
      # -encoder_bias. Without this the encoder_bias DR is only half-applied (the
      # actor sees the true position) and models no real calibration error.
      params={"biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
      delay_min_lag=DELAY_MIN,
      delay_max_lag=DELAY_MAX,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      delay_min_lag=DELAY_MIN,
      delay_max_lag=DELAY_MAX,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "phase": ObservationTermCfg(
      func=pi_mdp.phase,
      params={"frequency": WALK_FREQUENCY, "command_name": COMMAND},
    ),
    # [one_hot(5) | velocity(3) | path(2N+2) | goal_pose(4) | kick(7)],
    # inactive blocks zeroed.
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": COMMAND},
    ),
  }
  critic_terms = {
    **actor_terms,
    # Critic is privileged: override the (biased) actor joint_pos with the true
    # position so the value function sees ground truth.
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": False},
      delay_min_lag=DELAY_MIN,
      delay_max_lag=DELAY_MAX,
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/local_linvel"},
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"sensor_name": "foot_height_scan"},
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }
  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.1,
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  # The multi command redraws the type on each resample, then arms the next
  # resample from the *selected* sub-command's own ``resampling_time_range``:
  # velocity re-rolls every few seconds, a path gets long enough to be walked,
  # a stand gets a short dwell.
  path_cfg = PathCommandCfg(
    entity_name="robot",
    debug_vis=True,
    resampling_time_range=(20.0, 30.0),
    num_observations=NUM_PATH_OBSERVATIONS,
    # A path can be handed to a robot that is already mid-episode and away from
    # its spawn, so anchor each new path at the robot's current position.
    align=True,
  )

  commands: dict[str, CommandTermCfg] = {
    COMMAND: pi_mdp.MultiCommandCfg(
      debug_vis=True,
      change_command_probability=0.3,
      fractions={
        "velocity": 0.35,
        "path": 0.25,
        "goal_pose": 0.15,
        "kick": 0.15,
        "stand": 0.10,
      },
      terms={
        "velocity": pi_mdp.UniformVelocityCommandWithStopCfg(
          entity_name="robot",
          resampling_time_range=(3.0, 8.0),
          debug_vis=True,
          heading_command=True,
          heading_control_stiffness=0.0,
          # Standing is its own command type, so this term never stands: a zero
          # velocity here means walk-in-place, and the command drops to 3 dims.
          expose_stand_flag=False,
          rel_standing_envs=0.0,
          rel_walk_in_place_envs=0.075,
          rel_heading_envs=0.3,
          rel_forward_envs=0.15,
          ranges=pi_mdp.UniformVelocityCommandWithStopCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.25, 0.25),
            ang_vel_z=(-1.5, 1.5),
            heading=(-math.pi, math.pi),
          ),
        ),
        "path": path_cfg,
        # Go to a pose sampled relative to the robot: no reference path, just
        # the goal. Standing is its own type here, so rel_standing_envs stays 0.
        "goal_pose": pi_mdp.GoalPoseCommandCfg(
          entity_name="robot",
          debug_vis=True,
          resampling_time_range=(1.0, 5.0),
          radius_range=(0.5, 8.0),
          yaw_range=(-math.pi, math.pi),
          rel_standing_envs=0.0,
        ),
        # Owns the ball outright: places it near the robot while driving an
        # env, parks it at park_position while not.
        "kick": pi_mdp.KickCommandCfg(
          entity_name="robot",
          debug_vis=True,
          resampling_time_range=(6.0, 12.0),
          distance_range=(0.2, 2.0),
          direction_range=(-math.pi, math.pi),
          strength_range=(1.0, 5.0),
          park_position=(100.0, 100.0),
        ),
        "stand": pi_mdp.StandCommandCfg(resampling_time_range=(2.0, 5.0)),
      },
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.0, 0.0),
          "y": (-0.0, 0.0),
          "z": (0.01, 0.01),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.1, 0.1),
        "velocity_range": (-0.5, 0.5),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(2.0, 6.0),
      params={
        "velocity_range": {
          "x": (-0.25, 0.25),
          "y": (-0.25, 0.25),
          "z": (-0.15, 0.15),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          geom_names=(
            "l_ankle_roll_link_collision.*",
            "r_ankle_roll_link_collision.*",
          ),
        ),
        "operation": "scale",
        "ranges": (0.4, 1.0),
      },
    ),
    "joint_frictionloss": EventTermCfg(
      mode="startup",
      func=dr.joint_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "ranges": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    # DR on startup, enough variance over 4096 envs, much better performance than per-step.
    "joint_armature": EventTermCfg(
      mode="startup",
      func=dr.joint_armature,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "ranges": (1.0, 1.05),
        "operation": "scale",
      },
    ),
    "joint_damping": EventTermCfg(
      mode="startup",
      func=dr.joint_damping,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "ranges": (0.95, 1.05),
        "operation": "scale",
      },
    ),
    "actuator_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=(r".*",)),
        "kp_range": (0.9, 1.1),
        "kd_range": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    "link_masses": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
        "ranges": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    "torso_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        "ranges": (-0.25, 0.25),
        "operation": "add",
      },
    ),
    "link_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
        "operation": "add",
        "ranges": {
          0: (-0.01, 0.01),
          1: (-0.01, 0.01),
          2: (-0.01, 0.01),
        },
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        "operation": "add",
        "ranges": {
          0: (-0.03, 0.03),
          1: (-0.03, 0.03),
          2: (-0.03, 0.03),
        },
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "bias_range": (-0.1, 0.1),
      },
    ),
    "joint_torque": EventTermCfg(
      mode="startup",
      func=dr.effort_limits,
      params={
        "effort_limit_range": (0.85, 1.0),
        "operation": "scale",
      },
    ),
  }

  ##
  # Rewards
  ##

  # Path rewards evaluate the robot's position at the base footprint (the ground
  # midpoint between the two feet), so ``foot_asset_cfg`` carries the foot sites.
  foot_asset_cfg = SceneEntityCfg("robot", site_names=foot_sites)
  base_cfg = SceneEntityCfg("robot", body_names=("base_link",))

  rewards = {
    # -- Command tracking (velocity + stand) ----------------------------------
    # These are the pi_mdp copies, not mjlab's: mjlab's index the command tensor
    # directly, which under a MultiCommand would slice into the one-hot prefix.
    # They stay active under a stand command, where the commanded twist is zero
    # -- tracking zero velocity is exactly what standing still means.
    "track_linear_velocity": MultiRewardTermCfg(
      func=pi_mdp.track_linear_velocity,
      weight=2.0,
      params={"command_name": COMMAND, "std": math.sqrt(0.125)},
      command_weights=VELOCITY_AND_STAND,
    ),
    "track_angular_velocity": MultiRewardTermCfg(
      func=pi_mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": COMMAND, "std": math.sqrt(0.5)},
      command_weights=VELOCITY_AND_STAND,
    ),
    # -- Path following -------------------------------------------------------
    # Dense goal-pose reward split into independently-weighted position and
    # orientation terms (the combined product was too weak). Linear:
    # exp(-dist^2). Angular: same position gate * exp(-yaw_err^2), so
    # orientation only counts near the goal and does not fight path-following.
    "goal_reached_linear_reward": MultiRewardTermCfg(
      func=goal_reached_linear,
      weight=1.0,
      params={
        "command_name": COMMAND,
        "sub_command_name": "path",
        "metric_prefix": "path_goal",
        "asset_cfg": base_cfg,
        "foot_asset_cfg": foot_asset_cfg,
        "pos_std": 0.25,  # Gaussian width on goal distance (m).
        "success_threshold": 0.05,  # logged success-rate metric only.
      },
      command_weights=PATH_ONLY,
    ),
    "goal_reached_angular_reward": MultiRewardTermCfg(
      func=goal_reached_angular,
      weight=1.0,
      params={
        "command_name": COMMAND,
        "sub_command_name": "path",
        "metric_prefix": "path_goal",
        "asset_cfg": base_cfg,
        "foot_asset_cfg": foot_asset_cfg,
        "pos_std": 0.25,  # position gate width (m); matches the linear term.
        "yaw_std": 0.5,  # Gaussian width on goal-yaw error (rad).
        "success_yaw_threshold": 0.2,  # logged success-rate metric only.
      },
      command_weights=PATH_ONLY,
    ),
    "close_to_path_reward": MultiRewardTermCfg(
      func=close_to_path_reward,
      weight=1.0,
      params={
        "command_name": COMMAND,
        "asset_cfg": base_cfg,
        "foot_asset_cfg": foot_asset_cfg,
        "std": math.sqrt(0.25),
      },
      command_weights=PATH_ONLY,
    ),
    "path_progress_reward": MultiRewardTermCfg(
      func=path_progress_reward,
      weight=2.0,
      params={
        "command_name": COMMAND,
        "asset_cfg": base_cfg,
        "foot_asset_cfg": foot_asset_cfg,
        # Reward is progress speed in m/s, capped here (no bonus above this).
        "max_speed": 0.8,
      },
      command_weights=PATH_ONLY,
    ),
    "alignment_with_path_reward": MultiRewardTermCfg(
      func=align_with_path_reward,
      weight=1.0,
      params={
        "command_name": COMMAND,
        "asset_cfg": base_cfg,
        "foot_asset_cfg": foot_asset_cfg,
        "std": math.sqrt(1.0),
      },
      command_weights=PATH_ONLY,
    ),
    # -- Goal pose ------------------------------------------------------------
    # Walk to a sampled pose. Progress carries the robot there, the look-at term
    # keeps it facing where it is going, and the two goal-reached terms (shared
    # with the path task) settle it onto the position and then the orientation.
    "goal_pose_progress": MultiRewardTermCfg(
      func=pi_mdp.goal_progress,
      weight=2.0,
      params={
        "command_name": COMMAND,
        "sub_command_name": "goal_pose",
        # Progress speed in m/s, capped (no bonus above this).
        "max_speed": 0.8,
      },
      command_weights=GOAL_POSE_ONLY,
    ),
    "goal_pose_look_at": MultiRewardTermCfg(
      func=pi_mdp.orient_towards_goal,
      weight=0.5,
      params={
        "command_name": COMMAND,
        "sub_command_name": "goal_pose",
        "coeff": 1.0,
        # Inside this radius the bearing is ill-conditioned, so the look-at
        # saturates to full reward and goal_pose_reached_angular takes over.
        "distance_threshold": 0.3,
      },
      command_weights=GOAL_POSE_ONLY,
    ),
    "goal_pose_reached_linear": MultiRewardTermCfg(
      func=goal_reached_linear,
      weight=1.0,
      params={
        "command_name": COMMAND,
        "sub_command_name": "goal_pose",
        "metric_prefix": "goal_pose",
        "asset_cfg": base_cfg,
        "foot_asset_cfg": foot_asset_cfg,
        "pos_std": 0.25,
        "success_threshold": 0.05,
      },
      command_weights=GOAL_POSE_ONLY,
    ),
    "goal_pose_reached_angular": MultiRewardTermCfg(
      func=goal_reached_angular,
      weight=1.0,
      params={
        "command_name": COMMAND,
        "sub_command_name": "goal_pose",
        "metric_prefix": "goal_pose",
        "asset_cfg": base_cfg,
        "foot_asset_cfg": foot_asset_cfg,
        "pos_std": 0.25,
        "yaw_std": 0.5,
        "success_yaw_threshold": 0.2,
      },
      command_weights=GOAL_POSE_ONLY,
    ),
    # -- Kick -----------------------------------------------------------------
    # Two disjoint phases, split by the command's latched "has been kicked":
    # walk to the ball, then send it off at the commanded velocity. The latch is
    # what stops the robot nudging the ball away and re-earning the approach.
    "kick_approach": MultiRewardTermCfg(
      func=pi_mdp.kick_approach,
      weight=2.0,
      params={
        "command_name": COMMAND,
        "sub_command_name": "kick",
        # Closing speed in m/s, capped (no bonus above this).
        "max_speed": 0.8,
      },
      command_weights=KICK_ONLY,
    ),
    "kick_quality": MultiRewardTermCfg(
      func=pi_mdp.kick_quality,
      # Paid once per kick rather than every step, so this is the whole value of
      # a good kick, not a rate. Watch Episode_Reward/kick_quality against the
      # approach term when tuning.
      weight=5.0,
      params={
        "command_name": COMMAND,
        "sub_command_name": "kick",
        # Peak ball velocity over this window after first contact: the latch
        # can fire mid-contact while the ball is still accelerating.
        "window_s": 0.3,
        "direction_std": 0.35,  # rad, ~20 deg
        # Relative to the commanded strength, so 5 m/s is not held to the same
        # absolute precision as 1 m/s.
        "speed_rel_std": 0.35,
        # Aim is weighted above pace, and pace only counts once the aim is good.
        "direction_weight": 3.0,
        "speed_weight": 2.0,
      },
      command_weights=KICK_ONLY,
    ),
    # -- Stability ------------------------------------------------------------
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=1.0,
      params={"std": math.sqrt(0.2), "asset_cfg": base_cfg},
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.25,
      params={"asset_cfg": base_cfg},
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.1,
      params={"sensor_name": "robot/root_angmom"},
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    # -- Safety ---------------------------------------------------------------
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={
        "sensor_name": "self_collision",
        "force_threshold": 2.0,
      },  # we dont want even soft collisions
    ),
    "pose": RewardTermCfg(
      func=pi_mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": COMMAND,
        "std_standing": {r".*": 0.05},
        "std_walking": {
          r".*_hip_pitch_joint": 0.35,
          r".*_hip_roll_joint": 0.15,
          r".*_thigh_joint": 0.15,
          r".*_calf_joint": 0.35,
          r".*_ankle_pitch_joint": 0.35,
          r".*_ankle_roll_joint": 0.1,
        },
        "std_running": {
          r".*_hip_pitch_joint": 0.5,
          r".*_hip_roll_joint": 0.25,
          r".*_thigh_joint": 0.25,
          r".*_calf_joint": 0.5,
          r".*_ankle_pitch_joint": 0.5,
          r".*_ankle_roll_joint": 0.1,
        },
        # NOTE: under a path command the commanded twist reads zero, so this
        # term sees the walking regime; only a velocity command can select
        # ``std_running``.
        "running_threshold": 0.5,
      },
    ),
    # -- Gait shaping ---------------------------------------------------------
    # Encourage feet to be at target height when the foot velocity is high
    # (i.e. during swing).
    "foot_gait": RewardTermCfg(
      func=pi_mdp.feet_gait,
      weight=1.0,
      params={
        "frequency": WALK_FREQUENCY,
        "offset": [0.0, 0.5],
        "threshold": 0.56,
        "command_name": COMMAND,
        "sensor_name": "feet_ground_contact",
      },
    ),
    "feet_phase": RewardTermCfg(
      func=pi_mdp.feet_phase,
      weight=1.0,
      params={
        "frequency": WALK_FREQUENCY,
        "offset": [0.0, 0.5],
        "foot_height": 0.03,
        "stance_threshold": 0.56,
        "std": 0.025,
        "command_name": COMMAND,
        "asset_cfg": SceneEntityCfg("robot"),
        # Clearance is the lowest collision geom per foot (tilt-invariant).
        # Order matches `offset`: right (0.0) first, left (0.5) second.
        "foot_geom_names": [
          ("r_ankle_roll_link_collision.*",),
          ("l_ankle_roll_link_collision.*",),
        ],
        # Resting clearance of the lowest geom center when the foot is planted;
        # subtracted so a planted foot reads ~0.
        "baseline_height": 0.019,
      },
    ),
    "foot_slip": RewardTermCfg(
      func=pi_mdp.feet_slip,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": COMMAND,
        "asset_cfg": foot_asset_cfg,
      },
    ),
    # "soft_landing": RewardTermCfg(
    #   func=pi_mdp.soft_landing,
    #   weight=-1e-3,
    #   params={
    #     "sensor_name": "feet_ground_contact",
    #     "command_name": COMMAND,
    #   },
    # ),
    # "flat_foot": RewardTermCfg(
    #   func=pi_mdp.flat_foot_l2,
    #   weight=-0.0,
    #   params={
    #     "asset_cfg": foot_asset_cfg,
    #   },
    # ),
  }

  ##
  # Metrics
  ##
  metrics = {
    "feet_swing_peak_height": MetricsTermCfg(
      func=pi_mdp.feet_swing_peak_height,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": foot_asset_cfg,
      },
    ),
  }

  ##
  # Terminations
  ##
  terminations = {
    # Jittered per-env timeout keeps episode resets decorrelated
    "time_out": TerminationTermCfg(
      func=pi_mdp.time_out_jittered,
      time_out=True,
      params={"min_fraction": 0.5},
    ),
    # "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    # "root_height_below_minimum": TerminationTermCfg(
    #   func=mdp.root_height_below_minimum,
    #   params={"minimum_height": 0.25},
    # ),
  }

  ##
  # Curriculum
  ##
  curriculum = {
    # The velocity command lives inside the composite term, so it cannot be
    # reached by name through the command manager -- hence commands_vel_multi.
    "command_vel": CurriculumTermCfg(
      func=pi_mdp.commands_vel_multi,
      params={
        "command_name": COMMAND,
        "sub_command_name": "velocity",
        "velocity_stages": [
          {
            "step": 0 * 24,
            "lin_vel_x": (-0.25, 0.35),
            "lin_vel_y": (-0.15, 0.15),
            "ang_vel_z": (-math.pi / 4, math.pi / 4),
          },
          {
            "step": 800 * 24,
            "lin_vel_x": (-0.5, 0.8),
            "lin_vel_y": (-0.4, 0.4),
            "ang_vel_z": (-1.5, 1.5),
          },
        ],
      },
    ),
  }

  ##
  # Sensors
  ##

  feet_contact_sensor_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(l_ankle_roll_link|r_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
    history_length=4,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  scene = SceneCfg(
    num_envs=4096,
    env_spacing=0.0,
    entities={"robot": get_piplus_robot_cfg(), "ball": get_ball_cfg()},
    sensors=(feet_contact_sensor_cfg, self_collision_cfg, foot_height_scan),
    terrain=TerrainEntityCfg(
      terrain_type="plane",
      terrain_generator=None,
      max_init_terrain_level=0,
      env_spacing=0.0,
    ),
  )

  ##
  # Viewer
  ##
  viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="base_link",
    distance=3.0,
    elevation=-5.0,
    azimuth=90.0,
  )

  ##
  # Simulation
  ##
  sim = SimulationCfg(
    nconmax=45,
    njmax=1500,
    contact_sensor_maxmatch=500,
    mujoco=MujocoCfg(
      timestep=0.005,
      iterations=10,
      ls_iterations=20,
      ccd_iterations=500,
    ),
  )
  episode_length_s = 20.0

  if play:
    scene.num_envs = 1
    events = {  # remove other dr in play mode
      "reset_base": events["reset_base"],
      "reset_robot_joints": events["reset_robot_joints"],
    }
    # Unbounded episodes in play: plain time_out (never fires at 1e10 s).
    terminations["time_out"] = TerminationTermCfg(func=mdp.time_out, time_out=True)
    episode_length_s = 1e10
    curriculum = {}
    # With no episode resets in play, the type only changes on the active
    # sub-command's own timer (up to 30 s for a path). Use the viewer's
    # command-type dropdown to switch on demand.
    # Keep the play library small enough to generate in a few seconds, but not
    # so small that every episode shows the same worlds.
    path_cfg.pathlib_size = 200
    metrics["phase_metric"] = MetricsTermCfg(
      func=pi_mdp.phase_metric,
      params={"frequency": WALK_FREQUENCY, "command_name": COMMAND},
    )
    # Kick diagnostics: how fast the ball is actually going, and which half of
    # the task the robot is in. Play only -- in training these would just be
    # averaged over thousands of envs in different phases.
    metrics["kick_ball_speed"] = MetricsTermCfg(
      func=pi_mdp.kick_ball_speed,
      params={"command_name": COMMAND, "sub_command_name": "kick"},
    )
    metrics["kick_phase"] = MetricsTermCfg(
      func=pi_mdp.kick_phase,
      params={"command_name": COMMAND, "sub_command_name": "kick"},
      # 0/1/2 is a state, not a quantity: averaging it over an episode says
      # nothing, so report where the episode ended up.
      reduce="last",
    )
    metrics["foot_air_time"] = MetricsTermCfg(
      func=pi_mdp.current_air_time,
      params={"sensor_name": "feet_ground_contact"},
    )
    metrics["last_air_time"] = MetricsTermCfg(
      func=pi_mdp.last_air_time,
      params={"sensor_name": "feet_ground_contact"},
    )
    for side in ("l", "r"):
      for axis in ("roll", "pitch"):
        metrics[f"{side}_foot_{axis}"] = MetricsTermCfg(
          func=pi_mdp.foot_tilt,
          params={
            "asset_cfg": SceneEntityCfg("robot", site_names=(f"{side}_foot",)),
            "axis": axis,
          },
        )
    # NOTE: SceneEntityCfg.resolve() sorts sites by model index, so the reward's
    # site order is (r_foot=idx0, l_foot=idx1) -- i.e. offset[0]=0.0 drives the
    # RIGHT foot and offset[1]=0.5 the LEFT foot. The metric offsets below are
    # matched to that so each label tracks the correct physical foot.
    metrics["l_foot_rz_target"] = MetricsTermCfg(
      func=pi_mdp.feet_phase_target,
      params={
        "frequency": WALK_FREQUENCY,
        "offset": 0.5,
        "foot_height": 0.045,
        "stance_threshold": 0.56,
      },
    )
    metrics["r_foot_rz_target"] = MetricsTermCfg(
      func=pi_mdp.feet_phase_target,
      params={
        "frequency": WALK_FREQUENCY,
        "offset": 0.0,
        "foot_height": 0.045,
        "stance_threshold": 0.56,
      },
    )

  ##
  # Assemble the config
  ##
  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    metrics=metrics,
    terminations=terminations,
    curriculum=curriculum,
    viewer=viewer,
    sim=sim,
    decimation=4,
    episode_length_s=episode_length_s,
  )
