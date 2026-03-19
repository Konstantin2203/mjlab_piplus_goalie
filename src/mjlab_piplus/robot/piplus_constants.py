
from pathlib import Path
import os
import mujoco

from mjlab.actuator import DelayedActuatorCfg, BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg
##
# MJCF and assets.
##

PIPLUS_XML: Path = (
  Path(os.path.dirname(__file__)) / "piplus/piplus.xml"
)

def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, PIPLUS_XML.parent / "assets", meshdir)
  return assets

def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PIPLUS_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


EFFORT_LIMIT_4438 = 20.0
EFFORT_LIMIT_5047 = 30.0

ARMATURE_4438 = 0.008419
ARMATURE_5047 = 0.044277

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_4438 = 30
STIFFNESS_5047 = 80

DAMPING_4438 = 0.6
DAMPING_5047 = 1.1

# Motors used in Arms
ACTUATOR_4438 = BuiltinPositionActuatorCfg(
  target_names_expr=(
      ".*_shoulder_pitch_joint",
      ".*_shoulder_roll_joint",
      ".*_upper_arm_joint",
      ".*_elbow_joint",
  ),
  stiffness=STIFFNESS_4438,
  damping=DAMPING_4438,
  armature=ARMATURE_4438,
  effort_limit=EFFORT_LIMIT_4438,
  frictionloss=0.2, # TODO
)
DELAYED_ACTUATOR_4438 = DelayedActuatorCfg(
  base_cfg=ACTUATOR_4438,
  delay_target="position",
  delay_min_lag=2,
  delay_max_lag=3,
)

ACTUATOR_5047 = BuiltinPositionActuatorCfg(
  target_names_expr=(
      ".*_thigh_joint",
      ".*_hip_roll_joint",
      ".*_hip_pitch_joint",
      ".*_calf_joint",
      ".*_ankle_pitch_joint",
      ".*_ankle_roll_joint",
  ),
  stiffness=STIFFNESS_5047,
  damping=DAMPING_5047,
  armature=ARMATURE_5047,
  effort_limit=EFFORT_LIMIT_5047,
  frictionloss=0.2, # TODO
)

# Motors used in Legs
DELAYED_ACTUATOR_5047 = DelayedActuatorCfg(
  base_cfg=ACTUATOR_5047,
  delay_target="position",
  delay_min_lag=2,
  delay_max_lag=3,
)


HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.85),
  joint_pos={
    ".*_hip_pitch_joint": -0.25,
    ".*_hip_roll_joint": 0.0,
    ".*_thigh_joint": 0.0,
    ".*_calf_joint": 0.65,
    ".*_ankle_pitch_joint": -0.4,
    ".*_ankle_roll_joint": 0.0,
    ".*_shoulder_roll_joint": 0.0,
    ".*_shoulder_pitch_joint": 0.0,
    ".*_upper_arm_joint": 0.0,
    ".*_elbow_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(r|l)_ankle_roll_link_collision$",),
  contype=0, # disable self collisions by default
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

PIPLUS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(ACTUATOR_4438, ACTUATOR_5047),
  soft_joint_pos_limit_factor=0.9
)

def get_piplus_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=PIPLUS_ARTICULATION,
  )