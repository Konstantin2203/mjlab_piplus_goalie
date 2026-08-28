"""PPO config for the standup-imitation validation task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def get_piplus_standup_imitation_rl_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    # Same shape as the teacher's actor, so the student can in principle
    # represent it exactly. A poor score then says something about the bridge
    # rather than about capacity.
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
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
    experiment_name="piplus_standup_imitation",
    save_interval=100,
    num_steps_per_env=24,
    # A validation run, not a production one: imitation is a dense supervised-ish
    # signal and should move within a few hundred iterations. If it has not, the
    # bridge is wrong and more iterations will not fix it.
    max_iterations=1_000,
  )
