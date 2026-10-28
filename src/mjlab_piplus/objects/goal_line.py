"""A goal. With a width 

Parameters follow the RoboCup kid-size field model in bit-bots/bitbots_main
(``bitbots_mujoco_sim/xml/kid_field.xml``): a 7 cm sphere at 205 g with a full
friction cone and a stiff contact. The texture there is a cube map loaded from
PNGs; we skip it and use a flat colour, since nothing in training reads pixels.
"""

import mujoco from mjlab.entity import EntityCfg


GOAL_WIDTH = 1.8
GOAL_LINE_WIDTH = 0.05


def get_goal_line_spec(
  width: float = GOAL_WIDTH,
  length: float = GOAL_LINE_WIDTH,
  rgba: tuple[float, float, float, float] = (0.0, 0.1, 0.4, 1.0),
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="goal_line")
  body.add_freejoint(name="goal_line_root")
  body.add_geom(
    name="goal_line",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(length / 2, width / 2, 0.01),
    contype=0,
    conaffinity=0,
    rgba=rgba,
  )
  return spec


def get_goal_line_cfg(
  width: float = GOAL_WIDTH,
  length: float = GOAL_LINE_WIDTH,
) -> EntityCfg:
  """Scene entity for the ball, resting on the ground at the world origin.

  It is a floating-base, non-articulated entity: one body, one freejoint, one
  geom. Place it per episode with a ``reset_root_state_uniform`` event.
  """
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(4.0, 0.0, 0.000),  # Resting on the plane, not dropped onto it.
      # The freejoint is the only joint, so there is nothing for the joint
      # dicts to match; leaving them at the defaults keeps the entity builder
      # from looking for a keyframe that this spec does not have.
      joint_pos={},
      joint_vel={},
    ),
    spec_fn=lambda: get_goal_line_spec(width=width, length=length),
  )
