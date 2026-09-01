"""Minimal standup-by-imitation environment, for validating the teacher bridge.

This exists to answer one question: does ``pi_mdp.getup_imitation`` actually
reproduce the teacher's behaviour? Everything that could confound that answer is
removed -- no commands, no gait shaping, no penalties, no ball, no curriculum.
The imitation term is the primary reward, alongside the reference
implementation's three shaping terms -- base height, head height and
orientation. Nothing else: no gait, no penalties, no command tracking.

Consequences of that choice, all deliberate:

* **The imitation reward is ungated here**, unlike in the multi task. There the
  teacher must not fight the walking controller, so it is confined to the fallen
  half; here there is nothing to fight and gating would blind the reward exactly
  when the robot succeeds and stands up.
* **No ``on_ground``, no terminations but the time-out.** Being on the floor is
  priced by the height terms falling to zero there -- a gradient rather than a
  flat rate -- and there is no deadline to fail.
* **Every env spawns fallen** (``fall_probability=1.0``): the standing half
  would spend its episode in a state the teacher was barely trained on.
* **The actor observation is exactly the teacher's 66-vector**, in the same
  order. That is not required by the reward -- ``imitation.py`` builds the
  teacher's input itself -- but it means the student is at least *able* to
  represent the teacher's policy, so a poor score is evidence about the bridge
  rather than about a missing input.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import NoiseModelWithAdditiveBiasCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_piplus.robot.piplus_constants import get_piplus_robot_cfg
from mjlab_piplus.tasks import mdp as pi_mdp

from mjlab.tasks.velocity import mdp


# Kept identical to the multi task so a policy validated here is validated for
# the environment it will actually be used in.
STANDING_HEIGHT = 0.34
STANDING_HEAD_HEIGHT = 0.53
"""``STANDING_HEIGHT`` plus the rigid 0.198 m head-over-base offset."""
FALLEN_ANGLE = math.radians(45.0)
STANDING_MIN_HEIGHT = 0.25

# One step of observation lag, as in the multi task.
DELAY_MIN = 1
DELAY_MAX = 1

# Reward weights, gathered here because the balance between them is the only
# real tuning knob in this env. The three shaping terms carry the reference
# implementation's own weights (torso 2:1 over head, orientation 1.0), summing
# to 4.0 per step; the imitation weight matches that total, so "match the
# teacher" and "be tall and upright" are worth the same at their maxima and
# neither can be ignored in favour of the other.
IMITATION_WEIGHT = 2.0
TORSO_HEIGHT_WEIGHT = 2.0
HEAD_HEIGHT_WEIGHT = 1.0
ORIENTATION_WEIGHT = 1.0

GETUP_TEACHER_CHECKPOINT = "getup_model.pt"
IMITATION_STD = 0.3
"""Radians of joint-target error at which the reward falls to 1/e. See the
warning in CLAUDE.md: tighter than ~0.3 and an untrained policy starts inside
the exponential's dead zone with no usable gradient."""


def piplus_standup_imitation_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  base_cfg = SceneEntityCfg("robot", body_names=("base_link",))

  ##
  # Scene
  ##

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="geom", pattern=r".*_selfpad", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  scene = SceneCfg(
    num_envs=4096,
    env_spacing=0.0,
    entities={"robot": get_piplus_robot_cfg(with_arms=True, self_pads=True)},
    sensors=(self_collision_cfg,),
    terrain=TerrainEntityCfg(
      terrain_type="plane",
      terrain_generator=None,
      max_init_terrain_level=0,
      env_spacing=0.0,
    ),
  )

  ##
  # Observations -- the teacher's 66-vector, same order.
  ##

  # Noise and one-step lag copied verbatim from the multi task, so a policy that
  # survives here is being asked the same sensing question it will face there.
  #
  # NOTE the *teacher* does not see any of this. ``imitation.py`` builds its
  # observation from the true state, so the imitation target stays a clean
  # function of where the robot actually is while the student has to reproduce
  # it from noisy, delayed sensing. That asymmetry is the point -- a teacher fed
  # the same noise would emit a noisy target, which adds variance to the reward
  # without adding any information. The one thing it does keep is the encoder
  # bias (``joint_pos_biased``), because the reference policy was trained with
  # biased position input and expects it.
  actor_terms = {
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
  }
  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=True
    ),
    "critic": ObservationGroupCfg(
      terms={
        **actor_terms,
        "base_lin_vel": ObservationTermCfg(
          func=mdp.builtin_sensor, params={"sensor_name": "robot/local_linvel"}
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions -- default-relative, the map imitation.py bridges away from.
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
  # Rewards -- the imitation term and nothing else.
  ##

  rewards = {
    "getup_imitation": RewardTermCfg(
      func=pi_mdp.getup_imitation,
      weight=IMITATION_WEIGHT,
      params={
        "checkpoint": GETUP_TEACHER_CHECKPOINT,
        "std": IMITATION_STD,
        "teacher_scale": 0.6,
      },
    ),
    # The reference's three shaping terms, all ungated by design: gating them on
    # "is fallen" would step the return *down* the instant the robot stands up.
    # Height is paid twice, at the base and at the head, because either alone
    # has a hole -- a folded posture is tall at the base and short at the head,
    # and a robot sitting with a vertical trunk passes the orientation test.
    "torso_height": RewardTermCfg(
      func=pi_mdp.getup_height,
      weight=TORSO_HEIGHT_WEIGHT,
      params={"desired_height": STANDING_HEIGHT, "asset_cfg": base_cfg},
    ),
    "head_height": RewardTermCfg(
      func=pi_mdp.getup_height,
      weight=HEAD_HEIGHT_WEIGHT,
      params={
        "desired_height": STANDING_HEAD_HEIGHT,
        "asset_cfg": SceneEntityCfg("robot", body_names=("head_yaw_link",)),
      },
    ),
    "orientation": RewardTermCfg(
      func=pi_mdp.getup_orientation,
      weight=ORIENTATION_WEIGHT,
      params={"asset_cfg": base_cfg},
    ),
    # "upright": RewardTermCfg(
    #   func=mdp.upright,
    #   weight=1.0,
    #   params={"std": math.sqrt(0.2), "asset_cfg": base_cfg},
    # ),
  }

  ##
  # Events -- every env starts on the floor.
  ##

  # Domain randomization, copied from the multi task. All startup-mode: sampled
  # once per env at build time rather than per step, which over thousands of
  # envs gives the same coverage far more cheaply.
  domain_rands = {
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
        "ranges": {0: (-0.01, 0.01), 1: (-0.01, 0.01), 2: (-0.01, 0.01)},
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        "operation": "add",
        "ranges": {0: (-0.03, 0.03), 1: (-0.03, 0.03), 2: (-0.03, 0.03)},
      },
    ),
    # Feeds ``joint_pos_biased``, which is what BOTH the student's joint_pos
    # observation and the teacher's read. The reference trained its policy on
    # biased input too (at a slightly narrower +-0.0698), so this is the one
    # randomization the teacher is meant to see.
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
      params={"effort_limit_range": (0.85, 1.0), "operation": "scale"},
    ),
  }

  # NOTE no ``push_robot``. The multi task's push term is gated on the stand
  # command, and this env has no command manager at all to gate on; an ungated
  # push would also knock the robot over mid-getup, which is a disturbance the
  # teacher never saw and cannot advise on.
  events = {
    "reset_fallen_or_standing": EventTermCfg(
      func=pi_mdp.reset_fallen_or_standing,
      mode="reset",
      params={
        "fall_probability": 1.0,
        "fall_height": 0.4,
        "fall_velocity_range": 0.5,
        "joint_pos_std": 0.3,
      },
    ),
  } | domain_rands

  terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}

  ##
  # Metrics -- how the run is judged, since nothing but imitation is paid.
  ##

  metrics = {
    "fallen": MetricsTermCfg(
      func=pi_mdp.on_ground,
      params={
        "limit_angle": FALLEN_ANGLE,
        "min_height": STANDING_MIN_HEIGHT,
        "asset_cfg": base_cfg,
      },
    ),
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

  sim = SimulationCfg(
    nconmax=120,
    njmax=1500,
    contact_sensor_maxmatch=500,
    mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
  )
  episode_length_s = 8.0

  if play:
    scene.num_envs = 1
    episode_length_s = 1e10

  return ManagerBasedRlEnvCfg(
    decimation=4,
    scene=scene,
    observations=observations,
    actions=actions,
    rewards=rewards,
    events=events,
    terminations=terminations,
    metrics=metrics,
    sim=sim,
    episode_length_s=episode_length_s,
  )
