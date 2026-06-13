"""Left-right bilateral symmetry augmentation for the PiPlus robot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from rsl_rl.env import VecEnv

class PiPlusSymmetry:
    """Callable left-right symmetry function for the PiPlus robot.

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
        dev = env.device
        self._actor_obs_sign = torch.tensor([
            -1., 1., -1.,      # base_ang_vel
            1., -1., 1.,      # projected_gravity
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, # joint_pos
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, # joint vel
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, # actions
            1., -1., -1.,     # command
        ], dtype=torch.float32, device=dev)
        self._critic_obs_sign = torch.hstack([
            self._actor_obs_sign,
            torch.tensor([
                1., -1., 1.,   # base_lin_vel
                1, 1, # foot height
                1, 1, # foot air time
                1, 1, # foot contact
                1, -1, 1, 1, -1, 1 # foot contact forces
            ], dtype=torch.float32, device=dev),
        ])
        self._action_sign = torch.ones(12, dtype=torch.float32, device=dev) * -1

        self._actor_obs_perm = torch.arange(len(self._actor_obs_sign), dtype=torch.int64, device=dev)
        for k in [6, 18, 30]: # joint pos, vel, actions
            self._actor_obs_perm[k:k+6] = torch.arange(k+6, k+12, device=dev)
            self._actor_obs_perm[k+6:k+12] = torch.arange(k, k+6, device=dev)

        critic_only_perm = torch.arange(start=len(self._actor_obs_sign),
                                        end=len(self._critic_obs_sign)+len(self._actor_obs_sign),
                                        dtype=torch.int64, device=dev)
        for k in [3, 5, 7]: # foot height, air time, contact
            a = critic_only_perm[k].clone()
            critic_only_perm[k] = critic_only_perm[k+1]
            critic_only_perm[k+1] = a

        a = critic_only_perm[9:12].clone() # swap foot forces left-right
        critic_only_perm[9:12] = critic_only_perm[12:15]
        critic_only_perm[12:15] = a

        self._critic_obs_perm = torch.hstack([
            self._actor_obs_perm,
            critic_only_perm,
        ])

        self._action_perm = torch.arange(12, dtype=torch.int64, device=dev)
        self._action_perm[0:6] = torch.arange(6, 12, device=dev)
        self._action_perm[6:12] = torch.arange(0, 6, device=dev)

        self._ready = True

    def _mirror_actions(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self._action_perm] * self._action_sign

    def _mirror_obs(self, obs: torch.Tensor, key: str) -> torch.Tensor:
        if key == "actor":
            m = obs * self._actor_obs_sign
        elif key == "critic":
            m = obs * self._critic_obs_sign
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
            out_obs = TensorDict({}, batch_size=[obs.batch_size[0] * 2], device=obs.device, names=obs.names)
            for key, val in obs.items():
                mirrored = self._mirror_obs(val, key)
                out_obs[key] = torch.cat([val, mirrored], dim=0)

        if actions is not None:
            mirrored = self._mirror_actions(actions)
            out_actions = torch.cat([actions, mirrored], dim=0)
        return out_obs, out_actions
