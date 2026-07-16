import torch
from mjlab.utils.lab_api.math import wrap_to_pi


def xy_yaw_to_mat(xy_yaw: torch.Tensor) -> torch.Tensor:
  mat = torch.eye(4, device=xy_yaw.device).repeat(xy_yaw.shape[0], 1, 1)
  cos_yaw = torch.cos(xy_yaw[:, 2])
  sin_yaw = torch.sin(xy_yaw[:, 2])
  mat[:, 0, 0] = cos_yaw
  mat[:, 0, 1] = -sin_yaw
  mat[:, 1, 0] = sin_yaw
  mat[:, 1, 1] = cos_yaw
  mat[:, 0, 3] = xy_yaw[:, 0]
  mat[:, 1, 3] = xy_yaw[:, 1]
  return mat


def inv_mat(T: torch.Tensor) -> torch.Tensor:
  R = T[..., :3, :3]  # Extract rotation (3x3)
  t = T[..., :3, 3:4]  # Extract translation (3x1)
  R_inv = R.transpose(-1, -2)
  t_inv = -torch.matmul(R_inv, t)
  inv_T = torch.zeros_like(T)
  inv_T[..., :3, :3] = R_inv
  inv_T[..., :3, 3:4] = t_inv
  inv_T[..., 3, 3] = 1.0
  return inv_T


def angle_distance(x, y):
  return torch.atan2(torch.sin(y - x), torch.cos(y - x))


def heading_from_quat(q_wxyz: torch.Tensor):
  return wrap_to_pi(2.0 * torch.atan2(q_wxyz[..., 3], q_wxyz[..., 0]))
