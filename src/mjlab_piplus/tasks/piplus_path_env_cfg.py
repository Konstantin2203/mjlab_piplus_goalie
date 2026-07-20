import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_piplus.tasks import mdp as pi_mdp
from mjlab_piplus.tasks.mdp.path_following import (
  PathCommandCfg,
  align_with_path_reward,
  close_to_path_reward,
  goal_reached,
  path_progress_reward,
)
from mjlab_piplus.tasks.piplus_velocity_env_cfg import piplus_velocity_env_cfg


def piplus_path_env_cfg(
  play: bool = False, static: bool = False
) -> ManagerBasedRlEnvCfg:
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

  cfg.rewards["goal_reached_reward"] = RewardTermCfg(
    func=goal_reached,
    weight=0.0,  # overwritten by curriculum
    params={
      "command_name": "path",
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      "threshold": 0.075,
    },
  )
  # Path rewards evaluate the robot's position at the base footprint (the ground
  # midpoint between the two feet), so ``foot_asset_cfg`` carries the foot sites.
  foot_asset_cfg = SceneEntityCfg("robot", site_names=("l_foot", "r_foot"))
  cfg.rewards["close_to_path_reward"] = RewardTermCfg(
    func=close_to_path_reward,
    weight=0.0,  # overwritten by curriculum
    params={
      "command_name": "path",
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      "foot_asset_cfg": foot_asset_cfg,
      "std": math.sqrt(0.25),
    },
  )
  cfg.rewards["path_progress_reward"] = RewardTermCfg(
    func=path_progress_reward,
    weight=0.0,  # overwritten by curriculum
    params={
      "command_name": "path",
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      "foot_asset_cfg": foot_asset_cfg,
    },
  )
  cfg.rewards["alignment_with_path_reward"] = RewardTermCfg(
    func=align_with_path_reward,
    weight=0.0,  # overwritten by curriculum
    params={
      "command_name": "path",
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      "foot_asset_cfg": foot_asset_cfg,
      "std": math.sqrt(1.0),
    },
  )

  ##
  # Curriculum: delete the velocity command curriculum and ramp the path rewards
  # in over training. (The velocity cfg already clears the curriculum in play.)
  ##
  
  cfg.curriculum.pop("command_vel", None)
  if not play:
    cfg.curriculum["goal_reached_weight"] = CurriculumTermCfg(
      func=pi_mdp.reward_weight,
      params={
        "reward_name": "goal_reached_reward",
        "weight_stages": [
          {"step": 0, "weight": 0.0},
          {"step": 400 * 24, "weight": 1.0},
        ],
      },
    )
    cfg.curriculum["path_progress_weight"] = CurriculumTermCfg(
      func=pi_mdp.reward_weight,
      params={
        "reward_name": "path_progress_reward",
        "weight_stages": [
          {"step": 0, "weight": 0.0},
          {"step": 400 * 24, "weight": 1.5},
        ],
      },
    )
    cfg.curriculum["close_to_path_weight"] = CurriculumTermCfg(
      func=pi_mdp.reward_weight,
      params={
        "reward_name": "close_to_path_reward",
        "weight_stages": [
          {"step": 0, "weight": 0.0},
          {"step": 400 * 24, "weight": 1.0},
        ],
      },
    )
    cfg.curriculum["alignment_with_path_weight"] = CurriculumTermCfg(
      func=pi_mdp.reward_weight,
      params={
        "reward_name": "alignment_with_path_reward",
        "weight_stages": [
          {"step": 0, "weight": 0.0},
          {"step": 400 * 24, "weight": 1.0},
        ],
      },
    )

  if play:
    # No curriculum in play: set the path rewards to their final (curriculum)
    # weights so play shows the trained behavior.
    cfg.commands["path"].pathlib_size = 1000
    cfg.commands["path"].resampling_time_range = (30.0, 30.0)
    cfg.rewards["goal_reached_reward"].weight = 1.0
    cfg.rewards["path_progress_reward"].weight = 2.0
    cfg.rewards["close_to_path_reward"].weight = 1.0
    cfg.rewards["alignment_with_path_reward"].weight = 1.0

  return cfg
