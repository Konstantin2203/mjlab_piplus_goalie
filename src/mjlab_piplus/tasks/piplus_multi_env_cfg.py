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

Fall recovery is folded in rather than being a command of its own: a fraction of
envs respawn on the floor, there is no fall termination, and the getup terms are
live under every command type. See ``mdp/getup.py`` and the "Getup" constants
below. This is also the only task that actuates the arms (20 DoF), which is why
it is the only one without a symmetry transform to keep in sync.

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
from mjlab_piplus.robot.piplus_constants import (
  GETUP_COLLISION_INFLATE,
  get_piplus_robot_cfg,
)
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

# Angular tolerance (Gaussian sigma, radians) of the kick's signed aim reward,
# narrowed over training by the ``narrow_kick_aim`` curriculum. Both kick aim
# terms are signed: outside the tolerance they pay *negative*, so a tolerance
# that tightens faster than the policy can follow turns kicking into a net loss
# and the robot stops attempting it. Hence a wide start and a long runway.
KICK_DIR_SIGMA_START = math.radians(60.0)
KICK_DIR_SIGMA_END = math.radians(20.0)

# Weight presets, spelled out so a reward's applicability reads at a glance.
_OFF = {"velocity": 0.0, "path": 0.0, "goal_pose": 0.0, "kick": 0.0, "stand": 0.0}
VELOCITY_AND_STAND = {**_OFF, "velocity": 1.0, "stand": 1.0}
PATH_ONLY = {**_OFF, "path": 1.0}
GOAL_POSE_ONLY = {**_OFF, "goal_pose": 1.0}
KICK_ONLY = {**_OFF, "kick": 1.0}


##
# Getup.
##
# Base height (m) of the robot standing in the home keyframe. The getup height
# reward saturates here and the success metric is measured against it.
STANDING_HEIGHT = 0.39

# Tilt from vertical past which the robot counts as "on the ground". This is the
# angle the removed ``fell_over`` termination used, so the penalty takes over
# exactly where the termination left off.
FALLEN_ANGLE = math.radians(45.0)

def piplus_multi_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  foot_sites = ("l_foot", "r_foot")
  # only for critic
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
      # biased=True for encoder-biased position
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
      params={"frequency": WALK_FREQUENCY, "command_name": "command"},
    ),
    # [one_hot(5) | velocity(3) | path(2N+2) | goal_pose(4) | kick(7)],
    # inactive blocks zeroed.
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "command"},
    ),
  }
  critic_terms = {
    **actor_terms,
    # enable_corruption=False does not work for biased parameter
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
    resampling_time_range=(10.0, 20.0),
    num_observations=8,
    # A path can be handed to a robot that is already mid-episode and away from
    # its spawn, so anchor each new path at the robot's current position.
    align=True,
  )

  commands: dict[str, CommandTermCfg] = {
    "command": pi_mdp.MultiCommandCfg(
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
          resampling_time_range=(1.0, 15.0),
          radius_range=(0.5, 8.0),
          yaw_range=(-math.pi, math.pi),
          rel_standing_envs=0.0,
        ),
        # Owns the ball outright: places it near the robot while driving an
        # env, parks it at park_position while not.
        "kick": pi_mdp.KickCommandCfg(
          entity_name="robot",
          debug_vis=True,
          # Long enough for several ball lives: the ball comes back after every
          # kick, so this only bounds how long the kick task is handed out for.
          resampling_time_range=(8.0, 16.0),
          # Placed within a 60 degree fan either side of where the robot is
          # facing, at most 1.2 m away whatever the bearing.
          distance_range=(0.3, 1.2),
          placement_cone_deg=60.0,
          # Target direction within this cone of the robot's heading at
          # placement -- a direction it can plausibly hit from where it stands.
          # Widen towards 180 once kicking works to demand turning the ball.
          direction_cone_deg=135.0,
          strength_range=(1.0, 5.0),
          park_position=(100.0, 100.0),
          # Ball speed that counts as a strike, and how long the kick is scored
          # for afterwards before the ball is re-placed.
          contact_speed=0.5,
          window_s=1.0,
        ),
        "stand": pi_mdp.StandCommandCfg(resampling_time_range=(2.0, 5.0)),
      },
    )
  }

  ##
  # Events
  ##

  events = {
    # Replaces the old reset_base + reset_robot_joints pair. The standing branch
    # is byte-for-byte what those two did; the fallen branch drops the robot in
    # from fall_height at a uniformly random orientation with the default pose
    # plus Gaussian joint noise. Getting up is therefore never commanded -- it is
    # simply the situation a third of the envs wake up in.
    "reset_fallen_or_standing": EventTermCfg(
      func=pi_mdp.reset_fallen_or_standing,
      mode="reset",
      params={
        "fall_probability": 0.3,
        "fall_height": 0.4,
        "fall_velocity_range": 0.5,
        "joint_pos_std": 0.3,
        "stand_pose_range": {
          "x": (-0.0, 0.0),
          "y": (-0.0, 0.0),
          "z": (0.01, 0.01),
          "yaw": (-3.14, 3.14),
        },
        "stand_joint_pos_range": (-0.1, 0.1),
        "stand_joint_vel_range": (-0.5, 0.5),
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
      params={"command_name": "command", "std": math.sqrt(0.125)},
      command_weights=VELOCITY_AND_STAND,
    ),
    "track_angular_velocity": MultiRewardTermCfg(
      func=pi_mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": "command", "std": math.sqrt(0.5)},
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
        "command_name": "command",
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
        "command_name": "command",
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
        "command_name": "command",
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
        "command_name": "command",
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
        "command_name": "command",
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
        "command_name": "command",
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
        "command_name": "command",
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
        "command_name": "command",
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
        "command_name": "command",
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
    # One repeating cycle per ball: approach, strike, then a short window over
    # which the kick is scored densely before the ball is re-placed and the next
    # approach begins (see ``mdp/kick.py``). The window and the respawn are what
    # make kicking worth more than loitering next to the ball -- the previous
    # design paid a single lump at contact and then nothing, so the approach was
    # worth more than the kick it was supposed to set up.
    #
    # Rough budget for one good kick, at these weights: contact +2.5 (until the
    # curriculum drops it), impact +6.0 over the 1 s window, direction +2.0 --
    # against roughly 1.5/s of approach reward on the way in. Keep that ordering
    # when retuning: Episode_Reward/kick_impact should dominate
    # Episode_Reward/kick_approach once the policy is kicking at all.
    "kick_approach": MultiRewardTermCfg(
      func=pi_mdp.kick_approach,
      weight=3.0,
      params={
        "command_name": "command",
        "sub_command_name": "kick",
        # Speed toward the ball in m/s, capped: no bonus for charging it.
        "max_speed": 1.0,
        # Cubed cosine, so only a well-aimed approach gets most of the credit.
        "alignment_power": 3.0,
      },
      command_weights=KICK_ONLY,
    ),
    "kick_face_ball": MultiRewardTermCfg(
      func=pi_mdp.kick_face_ball,
      weight=0.5,
      params={"command_name": "command", "sub_command_name": "kick"},
      command_weights=KICK_ONLY,
    ),
    "kick_idle": MultiRewardTermCfg(
      func=pi_mdp.kick_idle,
      # Kills the do-nothing optimum: standing still with a ball to fetch has to
      # cost something, or refusing to engage is free.
      weight=-1.0,
      params={
        "command_name": "command",
        "sub_command_name": "kick",
        "vel_threshold": 0.1,
        # Inside this radius the robot is planting to swing, not loitering.
        "near_dist": 0.35,
      },
      command_weights=KICK_ONLY,
    ),
    "kick_foot_alignment": MultiRewardTermCfg(
      func=pi_mdp.kick_foot_alignment,
      weight=-0.25,
      params={
        "command_name": "command",
        "sub_command_name": "kick",
        # Start turning the foot to line up well before the last half metre.
        "close_dist": 0.9,
      },
      command_weights=KICK_ONLY,
    ),
    "kick_contact": MultiRewardTermCfg(
      func=pi_mdp.kick_contact,
      # Bootstrap only -- dropped to 0 by ``drop_kick_contact`` below. Left on,
      # a flat payout per touch makes a stream of weak taps competitive with
      # real kicks, since every touch buys a fresh ball.
      weight=2.5,
      params={
        "command_name": "command",
        "sub_command_name": "kick",
        "wrong_foot_penalty": 0.5,
      },
      command_weights=KICK_ONLY,
    ),
    "kick_impact": MultiRewardTermCfg(
      func=pi_mdp.kick_impact,
      # The payload: paid every step of the window, so this is ~weight x
      # window_s for a perfect kick and as negative for a well-struck one sent
      # the wrong way.
      weight=6.0,
      params={
        "command_name": "command",
        "sub_command_name": "kick",
        # Aim tolerance in radians, narrowed by ``narrow_kick_aim`` below.
        "dir_sigma": KICK_DIR_SIGMA_START,
        # Speed tolerance relative to the commanded strength, so 5 m/s is not
        # held to the same absolute precision as 1 m/s.
        "speed_rel_std": 0.35,
        "min_ball_speed": 0.2,
        # A toe-poke still pays a quarter of a clean side-foot contact: it has
        # to be discoverable before it can be refined away.
        "style_floor": 0.25,
      },
      command_weights=KICK_ONLY,
    ),
    "kick_direction": MultiRewardTermCfg(
      func=pi_mdp.kick_direction,
      weight=2.0,
      params={
        "command_name": "command",
        "sub_command_name": "kick",
        "dir_sigma": KICK_DIR_SIGMA_START,
        "min_ball_speed": 0.2,
      },
      command_weights=KICK_ONLY,
    ),
    # -- Getup ----------------------------------------------------------------
    # Not gated on a command: whatever the robot was asked to do, being on the
    # floor is wrong and standing back up is right. See ``mdp/getup.py`` for why
    # only the penalty is gated and the two shaping terms are not.
    #
    # Budget per second on the floor, at these weights: -2.0 for lying there,
    # against up to +0.5 orientation and +1.0 height earned back on the way up,
    # plus everything the walking terms start paying again once upright. Lying
    # still must stay a clear loss, and it does -- but only by about 0.5/s, so
    # the climb is worth attempting rather than something to be endured.
    "on_ground": RewardTermCfg(
      func=pi_mdp.on_ground,
      weight=-2.0,
      params={"limit_angle": FALLEN_ANGLE, "asset_cfg": base_cfg},
    ),
    "getup_orientation": RewardTermCfg(
      func=pi_mdp.getup_orientation,
      weight=0.5,
      params={"asset_cfg": base_cfg},
    ),
    "getup_height": RewardTermCfg(
      func=pi_mdp.getup_height,
      weight=1.0,
      params={"desired_height": STANDING_HEIGHT, "asset_cfg": base_cfg},
    ),
    # The camera bar sits ~5 mm proud of the head's collision shell, so this only
    # fires when the lens itself reaches the floor. Graded by force: resting on
    # it costs a few times what lying on your back costs, while a real faceplant
    # saturates all four substeps at -40/step -- twenty seconds' worth of
    # on_ground penalty spent in a fifth of a second. On the real robot the
    # camera is the one thing a fall can destroy, so there is no trajectory where
    # hitting it hard should look like an acceptable price for standing up
    # faster. It is deliberately not a termination: the robot has to learn to
    # fall away from its face, which it cannot do if the episode ends on impact.
    "camera_ground_impact": RewardTermCfg(
      func=pi_mdp.ground_contact_cost,
      weight=-10.0,
      params={
        "sensor_name": "camera_ground_contact",
        "force_scale": 50.0,
        "force_threshold": 1.0,
      },
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
        "command_name": "command",
        # NOTE: every actuated joint must be matched here or
        # ``resolve_matching_names_values`` raises -- hence the arm entries,
        # which are new with the 20-DoF robot. They are loose on purpose: the
        # arms are the main tool for getting off the floor and for balancing a
        # kick, so pinning them to the home pose would work against both. The
        # standing regime still holds them near the default, since a standing
        # robot has no reason to flail.
        "std_standing": {r".*": 0.05},
        "std_walking": {
          r".*_hip_pitch_joint": 0.35,
          r".*_hip_roll_joint": 0.15,
          r".*_thigh_joint": 0.15,
          r".*_calf_joint": 0.35,
          r".*_ankle_pitch_joint": 0.35,
          r".*_ankle_roll_joint": 0.1,
          r".*_shoulder_pitch_joint": 0.6,
          r".*_shoulder_roll_joint": 0.4,
          r".*_upper_arm_joint": 0.6,
          r".*_elbow_joint": 0.6,
        },
        "std_running": {
          r".*_hip_pitch_joint": 0.5,
          r".*_hip_roll_joint": 0.25,
          r".*_thigh_joint": 0.25,
          r".*_calf_joint": 0.5,
          r".*_ankle_pitch_joint": 0.5,
          r".*_ankle_roll_joint": 0.1,
          r".*_shoulder_pitch_joint": 0.8,
          r".*_shoulder_roll_joint": 0.5,
          r".*_upper_arm_joint": 0.8,
          r".*_elbow_joint": 0.8,
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
        "command_name": "command",
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
        "command_name": "command",
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
        "command_name": "command",
        "asset_cfg": foot_asset_cfg,
      },
    ),
    # "soft_landing": RewardTermCfg(
    #   func=pi_mdp.soft_landing,
    #   weight=-1e-3,
    #   params={
    #     "sensor_name": "feet_ground_contact",
    #     "command_name": "command",
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
    "fallen": MetricsTermCfg(
      func=pi_mdp.on_ground,
      params={"limit_angle": FALLEN_ANGLE, "asset_cfg": base_cfg},
    ),
    # Latched: did this env ever end up standing? Read against "fallen" -- this
    # is the number that says whether getup actually works.
    "getup_success": MetricsTermCfg(
      func=pi_mdp.getup_success,
      reduce="last",
      params={
        "desired_height": STANDING_HEIGHT,
        "limit_angle": math.radians(20.0),
        "height_tolerance": 0.05,
        "asset_cfg": base_cfg,
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
    # NOTE: no "fell_over" termination. Ending the episode on a fall makes getup
    # unlearnable -- the robot never experiences the state it is supposed to
    # recover from, and the -200 is_terminated penalty teaches it that being
    # horizontal is the end of the world rather than a situation with a way out.
    # The pressure not to fall now comes from the "on_ground" reward instead,
    # which charges a rate for lying there and stops charging the moment the
    # robot is upright again. Re-enabling this line disables getup training:
    # "fell_over": TerminationTermCfg(
    #   func=mdp.bad_orientation,
    #   params={"limit_angle": FALLEN_ANGLE},
    # ),
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
        "command_name": "command",
        "sub_command_name": "velocity",
        # Pushed back from (0, 800) to make room for getup. With the fall
        # termination gone, the first few hundred iterations are spent with a
        # third of the envs on the floor; opening the velocity ranges over that
        # stretch would have the policy chasing fast commands it cannot serve
        # while it still cannot stand up. Standing up first, then walking.
        "velocity_stages": [
          {
            "step": 0 * 24,
            "lin_vel_x": (-0.15, 0.25),
            "lin_vel_y": (-0.1, 0.1),
            "ang_vel_z": (-math.pi / 6, math.pi / 6),
          },
          {
            "step": 400 * 24,
            "lin_vel_x": (-0.25, 0.35),
            "lin_vel_y": (-0.15, 0.15),
            "ang_vel_z": (-math.pi / 4, math.pi / 4),
          },
          {
            "step": 1500 * 24,
            "lin_vel_x": (-0.5, 0.8),
            "lin_vel_y": (-0.4, 0.4),
            "ang_vel_z": (-1.5, 1.5),
          },
        ],
      },
    ),
    # Kick bootstrap off: the flat payout per contact exists only to get the
    # policy to touch the ball at all. Once it does, the quality terms should
    # drive the rest, or weak taps stay competitive with real kicks.
    "drop_kick_contact": CurriculumTermCfg(
      func=pi_mdp.reward_weight,
      params={
        "reward_name": "kick_contact",
        "weight_stages": [{"step": 1500 * 24, "weight": 0.0}],
      },
    ),
    # Then tighten the aim tolerance, on both aim terms at once so they keep
    # scoring the same kick the same way.
    "narrow_kick_aim": CurriculumTermCfg(
      func=pi_mdp.reward_param_linear,
      params={
        "reward_names": ["kick_impact", "kick_direction"],
        "param_name": "dir_sigma",
        "start_step": 1500 * 24,
        "end_step": 8000 * 24,
        "start_value": KICK_DIR_SIGMA_START,
        "end_value": KICK_DIR_SIGMA_END,
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
  # Fires only when the camera bar itself touches the ground -- see the
  # "camera_ground_impact" reward. Geom-mode, not subtree: the head shell is
  # allowed to take a landing, the lens is not.
  camera_ground_contact_cfg = ContactSensorCfg(
    name="camera_ground_contact",
    primary=ContactMatch(
      mode="geom", pattern=r"^camera_link_collision[0-9]$", entity="robot"
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
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
    entities={
      # 20 DoF: the arms are actuated here, unlike in the velocity and path
      # tasks. A robot that cannot use its arms cannot roll over and push
      # itself up, and they are worth having for balance while walking anyway.
      # The inflated upper-body collision shell is the getup hack -- see
      # GETUP_COLLISION_INFLATE.
      "robot": get_piplus_robot_cfg(
        with_arms=True, collision_inflate=GETUP_COLLISION_INFLATE
      ),
      "ball": get_ball_cfg(),
    },
    sensors=(
      feet_contact_sensor_cfg,
      self_collision_cfg,
      camera_ground_contact_cfg,
      foot_height_scan,
    ),
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
    # Raised from 45: a walker touches the ground with two feet, a robot lying
    # on its back touches it with the whole trunk, both arms and both legs at
    # once. Settling the robot flat from 400 random falls peaks at 50 contacts
    # with the inflated shell (43 without), so 45 was already clipping. Contacts
    # past nconmax are dropped silently, which shows up as the robot sinking
    # through the floor exactly when it is trying to push off it -- the rest is
    # headroom for the ball and for terrain that is not a plane.
    nconmax=120,
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
      "reset_fallen_or_standing": events["reset_fallen_or_standing"],
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
      params={"frequency": WALK_FREQUENCY, "command_name": "command"},
    )
    # Kick diagnostics: how fast the ball is actually going, and where in the
    # ball cycle the robot is (0 = not kicking, 1 = approaching, 2 = inside the
    # post-kick scoring window). Play only -- in training these would just be
    # averaged over thousands of envs in different phases. The per-kick numbers
    # (peak speed, aim error, style) are logged by the command itself and show
    # up in training under Metrics/kick_*.
    metrics["kick_ball_speed"] = MetricsTermCfg(
      func=pi_mdp.kick_ball_speed,
      params={"command_name": "command", "sub_command_name": "kick"},
    )
    metrics["kick_phase"] = MetricsTermCfg(
      func=pi_mdp.kick_phase,
      params={"command_name": "command", "sub_command_name": "kick"},
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
