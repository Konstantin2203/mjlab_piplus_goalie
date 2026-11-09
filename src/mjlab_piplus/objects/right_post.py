"""A post with a height. 

Parameters follow the RoboCup kid-size field model in bit-bots/bitbots_main
(``bitbots_mujoco_sim/xml/kid_field.xml``): a 1.2 m high with a radius of 2.5 cm goal post that 
blocks the ball when both collide.
"""
import mujoco 
from mjlab.entity import EntityCfg

RIGHT_POST_HEIGHT = 1.2
RIGHT_POST_RADIUS = 0.025


def get_right_post_spec(
    radius: float = RIGHT_POST_RADIUS,
    height: float = RIGHT_POST_HEIGHT,
    rgba: tuple[float, float, float, float] = (0.0, 0.1, 0.4, 1.0),
    ) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="right_post")
    body.add_geom(
        name="right_post",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(radius, height / 2),
        rgba=rgba,
    )
    return spec

def get_right_post_cfg(
    radius: float = RIGHT_POST_RADIUS,
    height: float = RIGHT_POST_HEIGHT,
    ) -> EntityCfg:
    """Scene entity for the left post, resting on the ground at the world origin.

    It is a floating-base, non-articulated entity: one body, one freejoint, one
    geom. Place it per episode with a ``reset_root_state_uniform`` event.
    """
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(-0.2, -0.925, height / 2),  # Resting on the plane, not dropped onto it.
            # The freejoint is the only joint, so there is nothing for the joint
            # dicts to match; leaving them at the defaults keeps the entity builder
            # from looking for a keyframe that this spec does not have.
            joint_pos={},
            joint_vel={},
        ),
        spec_fn=lambda: get_right_post_spec(radius=radius, height=height),
    )