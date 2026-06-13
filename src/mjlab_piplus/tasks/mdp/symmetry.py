"""Left-right bilateral symmetry augmentation for the PiPlus robot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from rsl_rl.env import VecEnv

# (right_joint, left_joint, sign)
# sign = +1 when the same value encodes a symmetric posture;
# sign = -1 when the partner joint's value must be negated to get the
# symmetric posture (opposite axis direction or negated range convention).
_JOINT_SYMMETRY = [
    #("r_shoulder_pitch_joint", "l_shoulder_pitch_joint", -1),  # axes (0,-1,0) vs (0,1,0)
    #("r_shoulder_roll_joint",  "l_shoulder_roll_joint",  -1),  # same axis, ranges negated
    #("r_upper_arm_joint",      "l_upper_arm_joint",      +1),  # same axis, same range
    #("r_elbow_joint",          "l_elbow_joint",          -1),  # axes (0,-1,0) vs (0,1,0)
    ("r_hip_pitch_joint",      "l_hip_pitch_joint",      -1),  # axes (0,-1,0) vs (0,1,0), defaults ±0.25
    ("r_hip_roll_joint",       "l_hip_roll_joint",       -1),  # same axis, ranges negated
    ("r_thigh_joint",          "l_thigh_joint",          +1),  # same axis, same range
    ("r_calf_joint",           "l_calf_joint",           -1),  # axes (0,1,0) vs (0,-1,0)
    ("r_ankle_pitch_joint",    "l_ankle_pitch_joint",    -1),  # axes (0,-1,0) vs (0,1,0)
    ("r_ankle_roll_joint",     "l_ankle_roll_joint",     +1),  # same axis, symmetric range
]


class PiPlusSymmetry:
    """Callable left-right symmetry function for the PiPlus robot.

    Actor observation layout assumed (concatenated in this order):
      base_lin_vel [3], base_ang_vel [3], projected_gravity [3],
      joint_pos [N], joint_vel [N], actions [N], command [3]

    Passed as ``data_augmentation_func`` to rsl_rl's ``Symmetry`` extension.
    Returns (obs, actions) with the original batch prepended to the mirrored
    batch along dim 0 (num_aug = 2).
    """

    def __init__(self) -> None:
        self._ready = False
        self._joint_perm: torch.Tensor
        self._joint_sign: torch.Tensor
        self._n_joints: int

    def _build(self, env: VecEnv) -> None:
        mjenv = env.unwrapped
        robot = mjenv.scene["robot"]
        names = list(robot.joint_names)
        n = len(names)
        name_to_i = {name: i for i, name in enumerate(names)}

        perm = list(range(n))
        joint_sign = [1.0] * n
        for r_name, l_name, s in _JOINT_SYMMETRY:
            r_i = name_to_i[r_name]
            l_i = name_to_i[l_name]
            perm[r_i] = l_i
            perm[l_i] = r_i
            joint_sign[r_i] = float(s)
            joint_sign[l_i] = float(s)

        dev = env.device
        self._joint_perm = torch.tensor(perm, dtype=torch.long, device=dev)
        self._joint_sign = torch.tensor(joint_sign, dtype=torch.float32, device=dev)
        self._n_joints = n

        # Per-term sign rules.  Joint-space terms are handled separately via _joint_perm.
        _TERM_SIGNS: dict[str, list[float]] = {
            "base_lin_vel":      [1., -1., 1.],         # flip y
            "base_ang_vel":      [-1., 1., -1.],        # flip roll & yaw
            "projected_gravity": [1., -1., 1.],         # flip y
            "command":           [1., -1., -1.],         # flip vy & wz
        }
        _JOINT_TERMS = {"joint_pos", "joint_vel", "actions"}

        # Walk active actor terms in order; compute per-element sign vector.
        obs_sign: list[float] = []
        self._joint_slices: list[tuple[int, int]] = []  # (start, end) for joint-space terms

        obs_manager = mjenv.observation_manager
        offset = 0
        for name in obs_manager.active_terms["actor"]:
            cfg = obs_manager.get_term_cfg("actor", name)
            with torch.no_grad():
                size = int(cfg.func(mjenv, **cfg.params).shape[-1])
            if name in _JOINT_TERMS:
                self._joint_slices.append((offset, offset + size))
                obs_sign.extend([1.0] * size)
            elif name in _TERM_SIGNS:
                obs_sign.extend(_TERM_SIGNS[name])
            else:
                obs_sign.extend([1.0] * size)
            offset += size

        self._obs_sign = torch.tensor(obs_sign, dtype=torch.float32, device=dev)
        self._ready = True

    def _mirror_joints(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self._joint_perm] * self._joint_sign

    def _mirror_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # Apply per-element sign flips (non-joint terms already handled by _obs_sign).
        m = obs * self._obs_sign
        # Overwrite joint-space blocks with permuted + signed values.
        for start, end in self._joint_slices:
            m[:, start:end] = obs[:, start:end][:, self._joint_perm] * self._joint_sign
        return m

    def __call__(
        self,
        env: VecEnv,
        obs: TensorDict | torch.Tensor | None,
        actions: torch.Tensor | None,
    ) -> tuple[TensorDict | torch.Tensor | None, torch.Tensor | None]:
        if not self._ready:
            self._build(env)

        out_obs = None
        out_actions = None

        if obs is not None:
            if isinstance(obs, TensorDict):
                # Mirror actor group; repeat other groups (e.g. critic) unchanged.
                new_tensors = {}
                for key, val in obs.items():
                    mirrored = self._mirror_obs(val) if key == "actor" else val
                    new_tensors[key] = torch.cat([val, mirrored], dim=0)
                out_obs = TensorDict(new_tensors, batch_size=[obs.batch_size[0] * 2], device=obs.device)
            else:
                mirrored = self._mirror_obs(obs)
                out_obs = torch.cat([obs, mirrored], dim=0)

        if actions is not None:
            mirrored = self._mirror_joints(actions)
            out_actions = torch.cat([actions, mirrored], dim=0)

        return out_obs, out_actions
