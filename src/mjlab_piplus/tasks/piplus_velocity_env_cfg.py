from copy import deepcopy
import math
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.noise import NoiseModelWithAdditiveBiasCfg
from mjlab.envs.mdp import dr
from mjlab.tasks.velocity import mdp
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab_piplus.robot.piplus_constants import get_piplus_robot_cfg
from mjlab_piplus.tasks import mdp as pi_mdp

from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, TerrainHeightSensorCfg, ObjRef

def piplus_velocity_env_cfg(play: bool = False, static: bool = False) -> ManagerBasedRlEnvCfg:
  foot_sites = ("l_foot", "r_foot")
  #foot_geoms = ("l_foot00", "r_foot00")
  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=(ObjRef(type="site", name="l_foot", entity="robot"), ObjRef(type="site", name="r_foot", entity="robot")),
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
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
    func=mdp.builtin_sensor,
    params={"sensor_name": "robot/gyro"},
    noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
    func=mdp.projected_gravity,
    noise=NoiseModelWithAdditiveBiasCfg(
      noise_cfg=Unoise(n_min=-0.05, n_max=0.05,),
      bias_noise_cfg=Unoise(n_min=-0.03, n_max=0.03, operation="add"),
      sample_bias_per_component=True,
    ),
    ),
    "joint_pos": ObservationTermCfg(
    func=mdp.joint_pos_rel,
    noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
    func=mdp.joint_vel_rel,
    noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "twist"},
    ),
  }
  critic_terms = {
    **actor_terms,
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
    scale=0.5,
    use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    debug_vis=True,
    heading_command=False,
    heading_control_stiffness=0.0,
    rel_standing_envs=0.1,
    rel_heading_envs=0.3,
    rel_forward_envs=0.15,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-0.5, 0.75),
      lin_vel_y=(-0.35, 0.35),
      ang_vel_z=(-math.pi/2, math.pi/2),
    ),
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
        "z": (0.01, 0.05),
        "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
        "x": (-0.1, 0.1),
        "y": (-0.1, 0.1),
        "z": (-0.2, 0.2),
        "roll": (-0.52, 0.52),
        "pitch": (-0.52, 0.52),
        "yaw": (-0.78, 0.78),
        },
      },
    ),
    # push with impulse
    "push_robot_impulse": EventTermCfg(
      func=mdp.apply_body_impulse,
      mode="step", # required as class keeps track of length of impule itself
      params={
        "force_range": (1.0, 5.0),
        "torque_range": (0.0, 0.0),
        "duration": (0.1, 0.3),
        "cooldown": (1.0, 3.0),
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      }
    ),
    # "foot_friction": EventTermCfg(
    #   mode="reset",
    #   func=dr.geom_friction,
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot", geom_names=("l_ankle_roll_link_collision.*", "r_ankle_roll_link_collision.*")),
    #     "operation": "scale",
    #     "ranges": (0.4, 1.0),
    #     "shared_random": True,
    #   },
    # ),
    # "joint_frictionloss": EventTermCfg(
    #   mode="reset",
    #   func=dr.joint_friction,
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    #     "ranges": (0.9, 1.1),
    #     "operation": "scale",
    #   },
    # ),
    # "joint_armature": EventTermCfg(
    #   mode="reset",
    #   func=dr.joint_armature,
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    #     "ranges": (1.0, 1.05),
    #     "operation": "scale",
    #   },
    # ),
    # "joint_damping": EventTermCfg(
    #   mode="reset",
    #   func=dr.joint_damping,
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    #     "ranges": (0.95, 1.05),
    #     "operation": "scale",
    #   },
    # ),
    # "actuator_gains": EventTermCfg(
    #   mode="reset",
    #   func=dr.pd_gains,
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot", actuator_names=(r".*",)),
    #     "kp_range": (0.75, 1.25),
    #     "kd_range": (0.75, 1.25),
    #     "operation": "scale",
    #   },
    # ),
    # "link_masses": EventTermCfg(
    #   mode="reset",
    #   func=dr.body_mass,
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
    #     "ranges": (0.9, 1.1),
    #     "operation": "scale",
    #   },
    # ),
    # "torso_mass": EventTermCfg(
    #   mode="reset",
    #   func=dr.body_mass,
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
    #     "ranges": (-0.5, 0.5),
    #     "operation": "add",
    #   },
    # ),
    # "link_com": EventTermCfg(
    #   mode="reset",
    #   func=dr.body_com_offset,
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
    #     "operation": "add",
    #     "ranges": {
    #       0: (-0.01, 0.01),
    #       1: (-0.01, 0.01),
    #       2: (-0.01, 0.01),
    #     },
    #   },
    # ),
    # "base_com": EventTermCfg(
    #   mode="reset",
    #   func=dr.body_com_offset,
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
    #     "operation": "add",
    #     "ranges": {
    #       0: (-0.03, 0.03),
    #       1: (-0.03, 0.03),
    #       2: (-0.03, 0.03),
    #     },
    #   },
    # ),
    # "encoder_bias": EventTermCfg(
    #   mode="reset",
    #   func=dr.encoder_bias,
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    #     "bias_range": (-0.05, 0.05),
    #   },
    # ),
  }

  ##
  # Rewards
  ##
  rewards = {
    # task rewards
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_linear_velocity_fine": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.03)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "track_angular_velocity_fine": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.05)},
    ),
    # stability rewards
    "upright": RewardTermCfg(
      func=pi_mdp.upright_relaxed,
      weight=1.0,
      params={
        "std": 0.025,
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,  # Override per-robot
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))}, 
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.02,
      params={"sensor_name": "robot/root_angmom"},
    ),
    # safety rewards
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05), # weight overwritten by curriculum
    "torque_penalty": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-1e-4, # weight overwritten by curriculum
      params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",))},
    ),
    "joint_vel_l2": RewardTermCfg(
      func=mdp.joint_vel_l2,
      weight=-1e-3, # weight overwritten by curriculum
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "self_collision", "force_threshold": 2.0}, # we dont want even soft collisions
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "twist",
        "std_standing": {r".*": 0.05},
        "std_walking": {
          r".*_hip_pitch_joint": 0.3,
          r".*_hip_roll_joint": 0.15,
          r".*_thigh_joint": 0.15,
          r".*_calf_joint": 0.35,
          r".*_ankle_pitch_joint": 0.25,
          r".*_ankle_roll_joint": 0.1,
          #r".*_shoulder_pitch_joint": 0.3,
          #r".*_shoulder_roll_joint": 0.05,
          #r".*_upper_arm_joint": 0.05,
          #r".*_elbow_joint": 0.2,
        },
        "std_running": {
          r".*_hip_pitch_joint": 0.3,
          r".*_hip_roll_joint": 0.15,
          r".*_thigh_joint": 0.15,
          r".*_calf_joint": 0.35,
          r".*_ankle_pitch_joint": 0.25,
          r".*_ankle_roll_joint": 0.1,
          #r".*_shoulder_pitch_joint": 0.3,
          #r".*_shoulder_roll_joint": 0.05,
          #r".*_upper_arm_joint": 0.05,
          #r".*_elbow_joint": 0.2,
        },
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    # shaping rewards
    "air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=2.0,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": "twist",
        "command_threshold": 0.01,
      },
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-20.0,
      params={
        "target_height": 0.04,
        "height_sensor_name": "foot_height_scan",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=foot_sites),
      },
    ),
    "foot_swing_height": RewardTermCfg(
      func=mdp.feet_swing_height,
      weight=-2.5,
      params={
        "sensor_name": "feet_ground_contact",
        "height_sensor_name": "foot_height_scan",
        "target_height": 0.04,
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=foot_sites),
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
  }
  ##
  # Metrics
  ##
  metrics = {
    #"yaw_tracking_error_filtered": MetricsTermCfg(
    #  func=pi_mdp.yaw_tracking_error_filtered,
    #),
    #"yaw_tracking_error_raw": MetricsTermCfg(
    #  func=pi_mdp.yaw_tracking_error_raw,
    #),
    #"torso_yaw_swing": MetricsTermCfg(
    #  func=pi_mdp.torso_yaw_swing,
    #),
    "base_ang_vel_xy_l2": MetricsTermCfg(
      func=pi_mdp.base_ang_vel_xy_l2,
    ),
    "step_frequency": MetricsTermCfg(
      func=pi_mdp.step_frequency,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_force": MetricsTermCfg(
      func=pi_mdp.foot_contact_force,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }
  if play:
    _all_joints = [
      #"r_shoulder_pitch_joint", "r_shoulder_roll_joint", "r_upper_arm_joint", "r_elbow_joint",
      #"l_shoulder_pitch_joint", "l_shoulder_roll_joint", "l_upper_arm_joint", "l_elbow_joint",
      "r_hip_pitch_joint", "r_hip_roll_joint", "r_thigh_joint", "r_calf_joint",
      "r_ankle_pitch_joint", "r_ankle_roll_joint",
      "l_hip_pitch_joint", "l_hip_roll_joint", "l_thigh_joint", "l_calf_joint",
      "l_ankle_pitch_joint", "l_ankle_roll_joint",
    ]
    for _jname in _all_joints:
      _short = _jname.replace("_joint", "")
      metrics[f"jdev/{_short}"] = MetricsTermCfg(
        func=pi_mdp.joint_pos_deviation,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(_jname,))},
      )

  ##
  # Terminations
  ##
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "root_height_below_minimum": TerminationTermCfg(
      func=mdp.root_height_below_minimum,
      params={"minimum_height": 0.25},
    ),
  }

  ##
  # Curriculum
  ##
  curriculum = {
    "command_vel": CurriculumTermCfg(
    func=mdp.commands_vel,
    params={
          "command_name": "twist",
          "velocity_stages": [
          #{"step": 0, "lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
          {"step": 0 * 24, "lin_vel_x": (-0.08, 0.1), "lin_vel_y": (-0.04, 0.04), "ang_vel_z": (-math.pi/6, math.pi/6)},
          #{"step": 200 * 24, "lin_vel_x": (-0.08, 0.1), "lin_vel_y": (-0.04, 0.04), "ang_vel_z": (-math.pi/6, math.pi/6)},
          {"step": 200 * 24, "lin_vel_x": (-0.15, 0.2), "lin_vel_y": (-0.1, 0.1), "ang_vel_z": (-math.pi/4, math.pi/4)},
          {"step": 400 * 24, "lin_vel_x": (-0.35, 0.5), "lin_vel_y": (-0.25, 0.25), "ang_vel_z": (-math.pi/3, math.pi/3)},
          {"step": 600 * 24, "lin_vel_x": (-0.5, 0.75), "lin_vel_y": (-0.35, 0.35), "ang_vel_z": (-math.pi/2, math.pi/2)},
          ],
      },
   ),
   #"feet_air_time_coef": CurriculumTermCfg(
   # func=mdp.reward_curriculum,
   # params={
   #   "reward_name": "air_time",
   #   "stages": [
   #     {"step": 0, "weight": 0.0},
   #     {"step": 200 * 24, "weight": 1.25},
   #     {"step": 600 * 24, "weight": 2.5},
   #   ],
   # },
   #),
   "action_rate_l2_coef": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
      "reward_name": "action_rate_l2",
      "stages": [
        {"step": 0, "weight": -0.005},
        {"step": 3200 * 24, "weight": -0.02},
      ],
    },
   ),
   "joint_vel_l2_coef": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
      "reward_name": "joint_vel_l2",
      "stages": [
        {"step": 0, "weight": -1e-4},
        {"step": 3200 * 24, "weight": -1e-3},
      ],
    },
   ),
   "torque_penalty_coef": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
      "reward_name": "torque_penalty",
      "stages": [
        {"step": 0, "weight": -1e-5},
        {"step": 3200 * 24, "weight": -1e-4},
      ],
    },
   ),
   "air_time_threshold_min": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "air_time",
        "stages": [
          {"step": 2000 * 24, "params": {"threshold_min": 0.06}},
          {"step": 2100 * 24, "params": {"threshold_min": 0.07}},
          {"step": 2200 * 24, "params": {"threshold_min": 0.08}},
          {"step": 2300 * 24, "params": {"threshold_min": 0.09}},
          {"step": 2400 * 24, "params": {"threshold_min": 0.10}},
          {"step": 2500 * 24, "params": {"threshold_min": 0.11}},
          {"step": 2600 * 24, "params": {"threshold_min": 0.12}},
          {"step": 2700 * 24, "params": {"threshold_min": 0.13}},
          {"step": 2800 * 24, "params": {"threshold_min": 0.14}},
          {"step": 2900 * 24, "params": {"threshold_min": 0.15}},
        ],
      },
   ),
   # "dr_foot_friction": CurriculumTermCfg(
   #  func=mdp.event_curriculum,
   #  params={
   #    "event_name": "foot_friction",
   #    "stages": [
   #      {"step": 0, "params": {"ranges": (1.0, 1.0)}},
   #      {"step": 1600 * 24, "params": {"ranges": (0.4, 1.0)}},
   #    ],
   #  },
   # ),
   # "dr_joint_frictionloss": CurriculumTermCfg(
   #  func=mdp.event_curriculum,
   #  params={
   #    "event_name": "joint_frictionloss",
   #    "stages": [
   #      {"step": 0, "params": {"ranges": (1.0, 1.0)}},
   #      {"step": 1600 * 24, "params": {"ranges": (0.9, 1.1)}},
   #    ],
   #  },
   # ),
   # "dr_joint_armature": CurriculumTermCfg(
   #  func=mdp.event_curriculum,
   #  params={
   #    "event_name": "joint_armature",
   #    "stages": [
   #      {"step": 0, "params": {"ranges": (1.0, 1.0)}},
   #      {"step": 1600 * 24, "params": {"ranges": (1.0, 1.05)}},
   #    ],
   #  },
   # ),
   # "dr_joint_damping": CurriculumTermCfg(
   #  func=mdp.event_curriculum,
   #  params={
   #    "event_name": "joint_damping",
   #    "stages": [
   #      {"step": 0, "params": {"ranges": (1.0, 1.0)}},
   #      {"step": 1600 * 24, "params": {"ranges": (0.95, 1.05)}},
   #    ],
   #  },
   # ),
   # "dr_actuator_gains": CurriculumTermCfg(
   #  func=mdp.event_curriculum,
   #  params={
   #    "event_name": "actuator_gains",
   #    "stages": [
   #      {"step": 0, "params": {"kp_range": (1.0, 1.0), "kd_range": (1.0, 1.0)}},
   #      {"step": 1600 * 24, "params": {"kp_range": (0.75, 1.25), "kd_range": (0.75, 1.25)}},
   #    ],
   #  },
   # ),
   # "dr_link_masses": CurriculumTermCfg(
   #  func=mdp.event_curriculum,
   #  params={
   #    "event_name": "link_masses",
   #    "stages": [
   #      {"step": 0, "params": {"ranges": (1.0, 1.0)}},
   #      {"step": 1600 * 24, "params": {"ranges": (0.9, 1.1)}},
   #    ],
   #  },
   # ),
   # "dr_torso_mass": CurriculumTermCfg(
   #  func=mdp.event_curriculum,
   #  params={
   #    "event_name": "torso_mass",
   #    "stages": [
   #      {"step": 0, "params": {"ranges": (0.0, 0.0)}},
   #      {"step": 1600 * 24, "params": {"ranges": (-0.5, 0.5)}},
   #    ],
   #  },
   # ),
   # "dr_link_com": CurriculumTermCfg(
   #  func=mdp.event_curriculum,
   #  params={
   #    "event_name": "link_com",
   #    "stages": [
   #      {"step": 0, "params": {"ranges": {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0)}}},
   #      {"step": 1600 * 24, "params": {"ranges": {0: (-0.01, 0.01), 1: (-0.01, 0.01), 2: (-0.01, 0.01)}}},
   #    ],
   #  },
   # ),
   # "dr_base_com": CurriculumTermCfg(
   #  func=mdp.event_curriculum,
   #  params={
   #    "event_name": "base_com",
   #    "stages": [
   #      {"step": 0, "params": {"ranges": {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0)}}},
   #      {"step": 1600 * 24, "params": {"ranges": {0: (-0.03, 0.03), 1: (-0.03, 0.03), 2: (-0.03, 0.03)}}},
   #    ],
   #  },
   # ),
   # "dr_encoder_bias": CurriculumTermCfg(
   #  func=mdp.event_curriculum,
   #  params={
   #    "event_name": "encoder_bias",
   #    "stages": [
   #      {"step": 0, "params": {"bias_range": (0.0, 0.0)}},
   #      {"step": 1600 * 24, "params": {"bias_range": (-0.05, 0.05)}},
   #    ],
   #  },
   # ),
  }

  ##
  # Sensors
  ##

  feet_contact_sensor_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(r_ankle_roll_link|l_ankle_roll_link)$",
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
    entities={"robot": get_piplus_robot_cfg()},
    sensors=(feet_contact_sensor_cfg, self_collision_cfg, foot_height_scan),
    terrain=TerrainEntityCfg(
      terrain_type="plane",
      terrain_generator=None,
      max_init_terrain_level=0,
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
  if play:
    curriculum = {}
    #del events["push_robot"]
    metrics["foot_air_time"] = MetricsTermCfg(
      func=pi_mdp.current_air_time,
      params={"sensor_name": "feet_ground_contact"},
    )
    metrics["last_air_time"] = MetricsTermCfg(
      func=pi_mdp.last_air_time,
      params={"sensor_name": "feet_ground_contact"},
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
    episode_length_s=20.0,
  )
