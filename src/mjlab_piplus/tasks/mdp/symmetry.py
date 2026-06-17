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

    @torch.no_grad()
    def __call__(
        self,
        env: VecEnv,
        obs: TensorDict | torch.Tensor | None,
        actions: torch.Tensor | None,
    ) -> tuple[TensorDict | torch.Tensor | None, torch.Tensor | None]:
        if obs is not None:
            batch_size = obs.batch_size[0]
            obs_aug = obs.repeat(2)
            obs_aug["actor"][:batch_size] = obs["actor"][:]
            obs_aug["actor"][batch_size:] = _transform_obs(obs["actor"][:])
        else:
            obs_aug = None

        if actions is not None:
            batch_size = actions.shape[0]
            action_aug = torch.zeros((batch_size * 2, actions.shape[1]), device=actions.device, dtype=actions.dtype)
            action_aug[:batch_size] = actions[:]
            action_aug[batch_size:] = _transform_actions(actions[:])
        else:
            action_aug = None

        return obs_aug, action_aug

def _transform_actions(actions: torch.Tensor) -> torch.Tensor:
    actions = actions.clone()
    actions[:] = _switch_left_right(actions[:]) * -1
    return actions

def _transform_obs(obs: TensorDict) -> TensorDict:
    obs = obs.clone()
    device = obs.device
    obs[:, :3] = obs[:, :3] * torch.tensor([-1., 1., -1.,], device=device) # base ang vel
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([1., -1., 1.], device=device) # projected gravity
    obs[:, 6:18] = _switch_left_right(obs[:, 6:18]) * -1 # joint pos
    obs[:, 18:30] = _switch_left_right(obs[:, 18:30]) * -1 # joint vel
    obs[:, 30:42] = _switch_left_right(obs[:, 30:42]) * -1 # actions
    obs[:, 42:45] = obs[:, 42:45] * torch.tensor([1., -1., -1.], device=device) # command
    return obs

def _switch_left_right(joint_data : torch.Tensor) -> torch.Tensor:
    joint_data_switched = joint_data.clone()
    joint_data_switched[..., [0,1,2,3,4,5]] = joint_data[..., [6,7,8,9,10,11]]
    joint_data_switched[..., [6,7,8,9,10,11]] = joint_data[..., [0,1,2,3,4,5]]
    return joint_data_switched