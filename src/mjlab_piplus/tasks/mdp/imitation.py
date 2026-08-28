"""Imitation of a pre-trained getup policy.

Rather than discovering a stand-up from shaped rewards, the fallen robot is
paid for reproducing what an already-trained getup policy would do from the
same state. The teacher is the policy from the reference environment in
``mjlab_playground`` (``getup_env_cfg.py``), loaded from an rsl_rl ``.pt``
checkpoint.

Two things have to be bridged, and they are the whole content of this module:

* **The observation.** The teacher's actor takes a 66-vector, in its own order:
  ``[base_ang_vel(3) | projected_gravity(3) | joint_pos_rel(20) |
  joint_vel_rel(20) | last_action(20)]``. The first four terms are the *same
  functions* this env already uses, so they are rebuilt here by calling them
  directly rather than slicing this env's actor observation -- which carries
  extra terms (phase, command) in a different layout and is noised. The fifth
  is the **teacher's own** previous output, not this policy's, so it is kept in
  a buffer here and reset with the episode.
* **The action.** The teacher uses ``RelativeJointPositionAction``:
  ``target = joint_pos + a_teacher * 0.6``. This env uses a default-relative
  map: ``target = default + f(a)``. Comparing raw action vectors would
  therefore be meaningless. Both are converted to the thing they actually
  mean -- a **joint position target, in radians** -- and the reward is the
  distance between those. That also makes the term indifferent to this env's
  action scale or to swapping in a nonlinear map.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from mjlab.entity import Entity
from mjlab.envs.mdp.observations import (
  builtin_sensor,
  joint_vel_rel,
  projected_gravity,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg

__all__ = ["getup_imitation"]

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# The teacher's joint order, from the reference robot's MJCF. Resolved to this
# env's action-term indices by name in __init__, so a reordering on either side
# is caught rather than silently scrambling the mapping.
TEACHER_JOINT_ORDER: tuple[str, ...] = (
  "r_shoulder_pitch_joint",
  "r_shoulder_roll_joint",
  "r_upper_arm_joint",
  "r_elbow_joint",
  "l_shoulder_pitch_joint",
  "l_shoulder_roll_joint",
  "l_upper_arm_joint",
  "l_elbow_joint",
  "r_hip_pitch_joint",
  "r_hip_roll_joint",
  "r_thigh_joint",
  "r_calf_joint",
  "r_ankle_pitch_joint",
  "r_ankle_roll_joint",
  "l_hip_pitch_joint",
  "l_hip_roll_joint",
  "l_thigh_joint",
  "l_calf_joint",
  "l_ankle_pitch_joint",
  "l_ankle_roll_joint",
)


# The teacher's own home pose, from the reference robot's HOME_KEYFRAME
# ("walkready"), in TEACHER_JOINT_ORDER. This env's default pose is NOT the
# same -- the elbows differ by 0.7 rad, the calves by 0.2, hips and ankles by
# 0.1 -- and the teacher's ``joint_pos`` observation is measured against *its*
# default, so feeding it a position relative to ours offsets its input by that
# difference. Its action is relative to the *measured* pose, so only the
# observation needs this; getting it wrong is silently wrong, not an error.
TEACHER_DEFAULT_JOINT_POS: tuple[float, ...] = (
  math.radians(-90.0),  # r_shoulder_pitch
  math.radians(-70.0),  # r_shoulder_roll
  0.0,  # r_upper_arm
  0.0,  # r_elbow
  math.radians(90.0),  # l_shoulder_pitch
  math.radians(70.0),  # l_shoulder_roll
  0.0,  # l_upper_arm
  0.0,  # l_elbow
  0.6,  # r_hip_pitch
  0.0,  # r_hip_roll
  0.0,  # r_thigh
  1.2,  # r_calf
  0.6,  # r_ankle_pitch
  0.0,  # r_ankle_roll
  -0.6,  # l_hip_pitch
  0.0,  # l_hip_roll
  0.0,  # l_thigh
  -1.2,  # l_calf
  -0.6,  # l_ankle_pitch
  0.0,  # l_ankle_roll
)


def _build_actor(state_dict: dict[str, torch.Tensor], device: str) -> nn.Sequential:
  """Rebuild the teacher's actor MLP from an rsl_rl ``actor_state_dict``.

  The reference runner sets ``obs_normalization=False`` on the actor, so the
  checkpoint carries no normalizer and the MLP consumes raw observations. If a
  checkpoint ever does carry one, that assumption is wrong and we say so rather
  than silently feeding unnormalized input to a normalized network.
  """
  mlp_sd = {k[len("mlp.") :]: v for k, v in state_dict.items() if k.startswith("mlp.")}
  if any(k.startswith("obs_normalizer") for k in state_dict):
    raise ValueError(
      "Teacher checkpoint carries an obs_normalizer; this loader assumes the "
      "reference's obs_normalization=False actor and would feed it raw input."
    )
  dims = [mlp_sd[k].shape for k in sorted(mlp_sd) if k.endswith("weight")]
  layers: list[nn.Module] = []
  for i, (out_f, in_f) in enumerate(dims):
    layers.append(nn.Linear(in_f, out_f))
    if i < len(dims) - 1:
      layers.append(nn.ELU())
  net = nn.Sequential(*layers).to(device)
  net.load_state_dict(mlp_sd)
  net.eval()
  net.requires_grad_(False)
  return net


class getup_imitation:
  """Reward for matching the teacher's joint position target.

  ``exp(-mean_j (q_target_ours - q_target_teacher)^2 / std^2)``, in radians.

  Gate this with ``fallen_gate`` so it only pays while the robot is down --
  ungated it would also fight the walking controller, which the teacher knows
  nothing about (its env has no velocity command and it was trained to stand up
  and stop).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    device = env.device
    path = Path(cfg.params["checkpoint"]).expanduser()
    if not path.is_file():
      raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    self._net = _build_actor(ckpt["actor_state_dict"], device)

    self._action_term = env.action_manager._terms[
      cfg.params.get("action_name", "joint_pos")
    ]
    our_names = list(self._action_term.target_names)
    missing = set(TEACHER_JOINT_ORDER) - set(our_names)
    if missing:
      raise ValueError(
        f"Joints the teacher controls are absent here: {sorted(missing)}"
      )
    if len(our_names) != len(TEACHER_JOINT_ORDER):
      raise ValueError(
        f"This env actuates {len(our_names)} joints, the teacher {len(TEACHER_JOINT_ORDER)}."
      )
    # ours[perm] == teacher order;  teacher[inv] == our order.
    perm = [our_names.index(n) for n in TEACHER_JOINT_ORDER]
    self._to_teacher = torch.tensor(perm, device=device, dtype=torch.long)
    self._to_ours = torch.argsort(self._to_teacher)

    in_features = self._net[0].in_features
    expected = 6 + 3 * len(TEACHER_JOINT_ORDER)
    if in_features != expected:
      raise ValueError(
        f"Teacher expects {in_features} inputs, this bridge builds {expected} "
        "([ang_vel 3 | gravity 3 | joint_pos | joint_vel | last_action])."
      )
    self._teacher_default = torch.tensor(
      TEACHER_DEFAULT_JOINT_POS, device=device, dtype=torch.float32
    ).unsqueeze(0)
    self._last_action = torch.zeros(
      env.num_envs, len(TEACHER_JOINT_ORDER), device=device
    )
    # The teacher target for the action the student is *about to* take, cached
    # from the state the student will see. See __call__ for why this lag matters.
    self._pending_target = torch.zeros(
      env.num_envs, len(TEACHER_JOINT_ORDER), device=device
    )
    self._primed = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    # The teacher's obs is recursive in its own last action; carrying it across
    # an episode boundary would feed it the tail of someone else's getup.
    if env_ids is None:
      self._last_action.zero_()
      self._primed.zero_()
    else:
      self._last_action[env_ids] = 0.0
      self._primed[env_ids] = False

  def _teacher_obs(
    self, env: ManagerBasedRlEnv, cfg: SceneEntityCfg, biased: bool = False
  ) -> torch.Tensor:
    asset: Entity = env.scene[cfg.name]
    ids = self._action_term.target_ids
    p = self._to_teacher
    # ``joint_pos`` is measured against the TEACHER's default pose, not this
    # env's. mjlab's ``joint_pos_rel`` would subtract ours, so the absolute
    # position is taken and the teacher's own default subtracted instead.
    # ``joint_vel_rel`` needs no such care: both defaults are zero velocity.
    #
    # ``biased`` defaults False -- the teacher gets the TRUE joint position.
    # The reference policy was trained on encoder-biased input, so passing the
    # bias through is the faithful choice, but the teacher is already privileged
    # in every other input (clean, undelayed gyro / gravity / joint velocity)
    # and the bias is the one place a handicap costs something: under this env's
    # +-0.1 rad encoder_bias it drops the teacher's getup success from 0.98 to
    # 0.65, and feeding it the true position restores 0.89. The student still
    # sees the bias in its own observation and still has to cope with it; there
    # is just no reason for the *target* to be corrupted by a nuisance parameter.
    raw = asset.data.joint_pos_biased if biased else asset.data.joint_pos
    joint_pos = raw[:, ids][:, p] - self._teacher_default
    return torch.cat(
      [
        builtin_sensor(env, "robot/gyro"),
        projected_gravity(env, cfg),
        joint_pos,
        joint_vel_rel(env, asset_cfg=cfg)[:, p],
        self._last_action,
      ],
      dim=-1,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    checkpoint: str,
    std: float = 0.2,
    teacher_scale: float = 0.6,
    teacher_biased: bool = False,
    action_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del checkpoint, action_name  # consumed in __init__.
    asset: Entity = env.scene[asset_cfg.name]
    ids = self._action_term.target_ids

    # -- Score the action that was just applied. --------------------------------
    # Rewards run *after* the physics step, but the action being judged was
    # chosen from the pre-step state. Recomputing the teacher here would compare
    # a pre-step decision against a post-step target: during a getup the joints
    # move ~0.06 rad in a single 20 ms step, which alone drove an *exact*
    # teacher-following oracle down to 0.018 of the 1.0 it should score. So the
    # teacher's target is cached one call earlier, from the same state the
    # student acted on.
    #
    # ``_processed_actions`` is this env's target *before* the encoder-bias
    # subtraction, which is exactly where the teacher's expression sits too, so
    # the bias cancels and never enters the comparison.
    err = torch.square(self._action_term._processed_actions - self._pending_target)
    reward = torch.exp(-err.mean(dim=1) / (std * std))
    # An env on its first step after a reset has no cached target from this
    # episode; scoring it against the previous episode's would be noise.
    reward = torch.where(self._primed, reward, torch.zeros_like(reward))

    # -- Cache the teacher's target for the *next* action. ----------------------
    # The teacher is a SettleRelativeJointPositionAction: it integrates off the
    # measured pose, so its target is joint_pos + a*scale. Ours is
    # default-relative. Both become the same thing -- a joint position target in
    # radians -- which is what makes the two parameterizations commensurable.
    with torch.no_grad():
      a_teacher = self._net(self._teacher_obs(env, asset_cfg, teacher_biased))
    self._last_action = a_teacher
    self._pending_target = (
      asset.data.joint_pos[:, ids] + teacher_scale * a_teacher[:, self._to_ours]
    )
    self._primed = torch.ones_like(self._primed)
    return reward
