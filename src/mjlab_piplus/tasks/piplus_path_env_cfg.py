import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_piplus.tasks import mdp as pi_mdp
from mjlab_piplus.tasks.mdp.path_following import (
  ObstacleWorldCfg,
  PathCommandCfg,
  align_with_path_reward,
  close_to_path_reward,
  goal_reached,
  path_progress_reward,
)
from mjlab_piplus.tasks.piplus_velocity_env_cfg import piplus_velocity_env_cfg


def piplus_path_env_cfg(
  play: bool = False, static: bool = False, obstacles: bool = False
) -> ManagerBasedRlEnvCfg:
  """Path-following env.

  Args:
    obstacles: If True, reference paths are planned with A* through a random
      field of cubes and cylinders instead of being free-space b-splines, so the
      robot has to walk corners and detours. The obstacles themselves are
      virtual — drawn in the viewer, never simulated — so nothing about the
      scene, contacts or terminations changes.
  """
  # Start from the velocity task; change only what the path task needs.
  cfg = piplus_velocity_env_cfg(play=play, static=static)

  ##
  # Commands: add a path command alongside the existing twist command. The twist
  # command is kept because the gait phase (and phase-based rewards) read its
  # stand flag; the path rewards read the path from this new command.
  ##
  cfg.commands["path"] = PathCommandCfg(
    entity_name="robot",
    debug_vis=True,
    resampling_time_range=(1.0e6, 1.0e6),
    num_observations=8,
  )
  if obstacles:
    cfg.commands["path"].generator = "astar"
    cfg.commands["path"].obstacle_world = ObstacleWorldCfg()
  # fake a command twist command for the rewards to work
  cfg.commands["twist"] = pi_mdp.UniformVelocityCommandWithStopCfg(
      entity_name="robot",
      resampling_time_range=(99.0, 99.0),
      debug_vis=False,
      heading_command=False,
      heading_control_stiffness=0.0,
      rel_standing_envs=0.0,
      rel_walk_in_place_envs=0.0,
      rel_heading_envs=0.0,
      rel_forward_envs=0.0,
      ranges=pi_mdp.UniformVelocityCommandWithStopCfg.Ranges(
        lin_vel_x=(0.25, 0.25), # use tighter pose rewards first
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.0, 0.0),
      ),
  )

  ##
  # Observations
  ##
  cfg.observations["actor"].terms["command"].params["command_name"] = "path"
  cfg.observations["critic"].terms["command"].params["command_name"] = "path"
  

  ##
  # Rewards: drop the two velocity tracking rewards, add the four path rewards.
  # Every other velocity reward is kept unchanged. The path rewards start at
  # weight 0 and are ramped in by the curriculum below.
  ##
  del cfg.rewards["track_linear_velocity"]
  del cfg.rewards["track_angular_velocity"]

  # Path rewards evaluate the robot's position at the base footprint (the ground
  # midpoint between the two feet), so ``foot_asset_cfg`` carries the foot sites.
  foot_asset_cfg = SceneEntityCfg("robot", site_names=("l_foot", "r_foot"))
  # Path rewards are active from the start (full weight, no ramp-in curriculum).
  cfg.rewards["goal_reached_reward"] = RewardTermCfg(
    func=goal_reached,
    weight=1.0,
    params={
      "command_name": "path",
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      "foot_asset_cfg": foot_asset_cfg,
      # Dense goal-pose reward: Gaussian widths on position (m) and goal-yaw (rad).
      "pos_std": 0.25,
      "yaw_std": 0.5,
      # Thresholds for the logged success-rate metric only (not the reward).
      "success_threshold": 0.05,
      "success_yaw_threshold": 0.2,
    },
  )
  cfg.rewards["close_to_path_reward"] = RewardTermCfg(
    func=close_to_path_reward,
    weight=1.0,
    params={
      "command_name": "path",
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      "foot_asset_cfg": foot_asset_cfg,
      "std": math.sqrt(0.25),
    },
  )
  cfg.rewards["path_progress_reward"] = RewardTermCfg(
    func=path_progress_reward,
    weight=2.0,
    params={
      "command_name": "path",
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      "foot_asset_cfg": foot_asset_cfg,
      # Reward is progress speed in m/s, capped here (no bonus above this speed).
      "max_speed": 0.8,
    },
  )
  cfg.rewards["alignment_with_path_reward"] = RewardTermCfg(
    func=align_with_path_reward,
    weight=1.0,
    params={
      "command_name": "path",
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      "foot_asset_cfg": foot_asset_cfg,
      "std": math.sqrt(1.0),
    },
  )

  ##
  # Curriculum: delete the velocity command curriculum. The path rewards are
  # active from the start (no ramp-in), so only the push magnitude is ramped.
  # (The velocity cfg already clears the curriculum in play.)
  ##

  cfg.curriculum.pop("command_vel", None)
  if not play:
    # Ramp the x/y push magnitude over training: start at +/-0.25 (the push
    # event's default), then widen at 800 / 1200 / 1600 iterations. Only x and y
    # are touched; z/roll/pitch/yaw are left at their defaults via deep-merge.
    cfg.curriculum["push_velocity"] = CurriculumTermCfg(
      func=pi_mdp.event_curriculum,
      params={
        "event_name": "push_robot",
        "stages": [
          {"step": 0, "params": {"velocity_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25)}}},
          {"step": 800 * 24, "params": {"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}},
          {"step": 1200 * 24, "params": {"velocity_range": {"x": (-0.75, 0.75), "y": (-0.75, 0.75)}}},
          {"step": 1600 * 24, "params": {"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}}},
        ],
      },
    )

  if play:
    # No curriculum in play: set the path rewards to their final (curriculum)
    # weights so play shows the trained behavior.
    cfg.commands["path"].pathlib_size = 1000
    cfg.commands["path"].resampling_time_range = (30.0, 30.0)
    if obstacles:
      # Keep the play library small enough to generate in a few seconds, but
      # not so small that every episode shows the same worlds.
      cfg.commands["path"].pathlib_size = 200

  return cfg
