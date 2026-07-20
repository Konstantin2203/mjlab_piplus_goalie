from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from mjlab_piplus.tasks.mdp import PiPlusPathSymmetry


@dataclass
class PiPlusPathPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  # Left-right symmetry for the path task. PiPlusPathSymmetry mirrors the actor
  # obs whose command tail is N base-frame path points [x, y] (each -> [x, -y]);
  # see symmetry.py. Both enforcement paths are on: data augmentation appends
  # mirrored samples to every mini-batch, and the mirror loss adds a soft
  # symmetry penalty.
  symmetry_cfg: dict[str, Any] | None = field(
    default_factory=lambda: {
      "data_augmentation_func": PiPlusPathSymmetry(),
      "use_data_augmentation": True,
      "use_mirror_loss": True,
      "mirror_loss_coeff": 0.5,
    }
  )


def get_piplus_path_rl_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=PiPlusPathPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    wandb_project="mjlab_piplus",
    experiment_name="piplus_path",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=3_200,
  )
