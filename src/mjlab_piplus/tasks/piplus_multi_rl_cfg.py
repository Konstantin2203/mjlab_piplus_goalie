from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def get_piplus_multi_rl_cfg() -> RslRlOnPolicyRunnerCfg:
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
    # NOTE: no symmetry_cfg. The multi-command actor obs tail is
    # ``[one_hot(3) | velocity(3) | path(2N + 2)]``, which none of the existing
    # transforms in ``mdp/symmetry.py`` mirror correctly -- they are hard-coded
    # by index against the old layouts. A wrong mirror is worse than none (it
    # trains the policy toward a false invariance), so symmetry stays off until
    # a PiPlusMultiSymmetry is written and checked with ``check_symmetry.py``.
    algorithm=RslRlPpoAlgorithmCfg(
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
    experiment_name="piplus_multi",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=6_400,
  )
