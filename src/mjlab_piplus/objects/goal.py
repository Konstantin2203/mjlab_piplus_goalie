"""A goal. With a width 

Parameters follow the RoboCup kid-size field model in bit-bots/bitbots_main
(``bitbots_mujoco_sim/xml/kid_field.xml``): a 7 cm sphere at 205 g with a full
friction cone and a stiff contact. The texture there is a cube map loaded from
PNGs; we skip it and use a flat colour, since nothing in training reads pixels.
"""

import mujoco from mjlab.entity import EntityCfg


GOAL_WIDTH = 1.8
GOAL_LINE_WIDTH = 0.05