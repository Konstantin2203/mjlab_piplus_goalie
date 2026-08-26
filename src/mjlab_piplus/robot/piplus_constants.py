import os
import re
from collections.abc import Mapping
from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

PIPLUS_XML: Path = Path(os.path.dirname(__file__)) / "piplus/piplus.xml"

VISUAL_RGBA = (0.2, 0.2, 0.2, 1.0)
"""Dark gray (80% gray) for the robot's visual meshes. The MJCF paints them
light gray (0.75); they are recolored here so the vendored file stays as
exported. Collision geoms are named ``*_collision[0-9]`` and are left alone."""


ARM_JOINT_EXPR = r"^[lr]_(shoulder_pitch|shoulder_roll|upper_arm|elbow)_joint$"
"""Arm joints, present in the MJCF but deleted from the spec unless a task asks
for them (see :func:`get_spec`)."""

GETUP_COLLISION_INFLATE: dict[str, float] = {
  r"^base_link_collision[0-9]$": 1.25,
  r"^torso_link_collision[0-9]$": 1.25,
  r"^[lr]_upper_arm_link_collision[0-9]$": 1.25,
  r"^[lr]_wrist_link_collision[0-9]$": 1.25,
  r"^[lr]_calf_link_collision[0-9]$": 1.25,
}
"""The "bigger collision model" getup hack.

A robot that has to roll over and push itself up spends most of the episode
resting on its trunk and arms, which the exported MJCF represents with a couple
of thin boxes and cylinders that the meshes poke straight through. Scaling those
primitives up gives the fallen robot a convex shell to lever against, and keeps
the solver away from the near-degenerate mesh-corner contacts that make getup
look like it is clipping through the floor.

Two links are deliberately left out.

``head_pitch_link_collision0`` already encloses the camera bar, so inflating it
would bury ``camera_link_collision0`` inside the shell and the camera-strike
penalty could never fire. The camera geom is sized to sit ~5 mm proud of the head
shell for exactly that reason.

The thighs are excluded because inflating them *and* the trunk makes the two
overlap in poses the robot holds all the time -- at the home keyframe the thigh
cylinders penetrate ``base_link_collision0`` by 6 mm, and at a deep hip pitch
they reach ``torso_link_collision0`` by 9 mm. That is a permanent contact the
solver has to fight and a permanent ``self_collisions`` penalty the policy can do
nothing about. Sweeping hip pitch, knee and shoulder roll over their ranges with
the list below gives zero self-contacts, the same as the uninflated model; adding
the thighs back gives four penetrating pairs. Re-check that sweep before adding
anything here.
"""


def get_spec(
  with_arms: bool = False,
  collision_inflate: Mapping[str, float] | None = None,
) -> mujoco.MjSpec:
  """Build the PiPlus spec.

  Args:
    with_arms: Keep the arm joints and their position actuators. When False they
      are deleted from the spec, leaving the arms welded to the torso -- the
      12-DoF layout the velocity and path tasks (and the index-hard-coded
      mirror in ``mdp/symmetry.py``) are built around.
    collision_inflate: Map of collision-geom name regex -> size scale factor,
      applied to every matching geom. See :data:`GETUP_COLLISION_INFLATE`.
  """
  spec = mujoco.MjSpec.from_file(str(PIPLUS_XML))
  for geom in spec.geoms:
    if geom.type == mujoco.mjtGeom.mjGEOM_MESH and not geom.name:
      geom.rgba = VISUAL_RGBA

  if not with_arms:
    # Actuators first: an actuator whose target joint is gone fails to compile.
    for actuator in list(spec.actuators):
      if re.match(ARM_JOINT_EXPR, actuator.target):
        spec.delete(actuator)
    for joint in list(spec.joints):
      if re.match(ARM_JOINT_EXPR, joint.name):
        spec.delete(joint)

  if collision_inflate:
    for geom in spec.geoms:
      for expr, scale in collision_inflate.items():
        if geom.name and re.match(expr, geom.name):
          geom.size = geom.size * scale
          break

  return spec


ACTUATOR_LAG_MIN = 0
ACTUATOR_LAG_MAX = 3

# ACTUATOR_3536 = XmlActuatorCfg(
#   target_names_expr=(
#     "head_yaw_joint",
#     "head_pitch_joint"
#   ),
#   delay_min_lag=ACTUATOR_LAG_MIN,
#   delay_max_lag=ACTUATOR_LAG_MAX,
# )

# Motors used in Arms
ACTUATOR_4438 = XmlActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_upper_arm_joint",
    ".*_elbow_joint",
  ),
  delay_min_lag=ACTUATOR_LAG_MIN,
  delay_max_lag=ACTUATOR_LAG_MAX,
)

# Motors used in Legs
ACTUATOR_5036 = XmlActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_thigh_joint",
    ".*_calf_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
  ),
  delay_min_lag=ACTUATOR_LAG_MIN,
  delay_max_lag=ACTUATOR_LAG_MAX,
)


LEG_HOME_JOINT_POS: dict[str, float] = {
  "r_hip_pitch_joint": 0.5,
  "l_hip_pitch_joint": -0.5,
  ".*_hip_roll_joint": 0.0,
  ".*_thigh_joint": 0.0,
  "r_calf_joint": 1.0,
  "l_calf_joint": -1.0,
  "r_ankle_pitch_joint": 0.5,
  "l_ankle_pitch_joint": -0.5,
  ".*_ankle_roll_joint": 0.0,
}

ARM_HOME_JOINT_POS: dict[str, float] = {
  # The shoulder joints carry a ``ref`` of +-1.5708 that sits outside their own
  # range, so "zero" is not a reachable pose: these are the arms-down values
  # that the MJCF's reference configuration actually corresponds to, clamped
  # into range for the roll joints.
  "r_shoulder_pitch_joint": 1.5708,
  "l_shoulder_pitch_joint": -1.5708,
  "r_shoulder_roll_joint": 1.2,
  "l_shoulder_roll_joint": -1.2,
  ".*_upper_arm_joint": 0.0,
  ".*_elbow_joint": 0.0,
}


def get_home_keyframe(with_arms: bool = False) -> EntityCfg.InitialStateCfg:
  joint_pos = dict(LEG_HOME_JOINT_POS)
  if with_arms:
    joint_pos.update(ARM_HOME_JOINT_POS)
  return EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.39),
    joint_pos=joint_pos,
    joint_vel={".*": 0.0},
  )


FULL_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_collision[0-9]$",),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

LEG_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"(r|l)_(thigh|calf|ankle_pitch|ankle_roll)_link_collision[0-9]$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"(r|l)_ankle_roll_link_collision[0-9]$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

def get_piplus_articulation(with_arms: bool = False) -> EntityArticulationInfoCfg:
  actuators = (ACTUATOR_5036, ACTUATOR_4438) if with_arms else (ACTUATOR_5036,)
  return EntityArticulationInfoCfg(
    actuators=actuators,
    soft_joint_pos_limit_factor=0.9,
  )


def get_piplus_robot_cfg(
  with_arms: bool = False,
  collision_inflate: Mapping[str, float] | None = None,
) -> EntityCfg:
  """Assemble the PiPlus entity config.

  Args:
    with_arms: Actuate the eight arm joints (20 DoF instead of 12). Off by
      default: the velocity and path tasks mirror their observations by
      hard-coded index in ``mdp/symmetry.py`` and assume 12 joints.
    collision_inflate: Passed through to :func:`get_spec`; see
      :data:`GETUP_COLLISION_INFLATE`.
  """
  return EntityCfg(
    init_state=get_home_keyframe(with_arms),
    collisions=(FULL_COLLISION,),
    spec_fn=lambda: get_spec(with_arms, collision_inflate),
    articulation=get_piplus_articulation(with_arms),
  )
