
from mjlab.envs import ManagerBasedRlEnv
import torch

def phase(env: ManagerBasedRlEnv, frequency: float, command_name: str, cmd_threshold: float) -> torch.Tensor:
    period = 1 / frequency
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < cmd_threshold
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase

def zeros(env: ManagerBasedRlEnv, shape) -> torch.Tensor:
    return torch.zeros((env.num_envs, *shape), device=env.device)