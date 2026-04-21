"""NPP-14: RewardConfig BSRS / beta_blend schedule."""

from nclone.gym_environment.reward_calculation.reward_config import RewardConfig


def test_beta_blend_disabled_is_one():
    cfg = RewardConfig(total_timesteps=1_000_000, enable_bsrs_blend=False)
    assert cfg.beta_blend() == 1.0


def test_beta_blend_anneals_over_horizon_fraction():
    cfg = RewardConfig(
        total_timesteps=1000,
        enable_bsrs_blend=True,
        bsrs_blend_horizon_fraction=0.7,
    )
    cfg.sync_training_timesteps(0)
    assert cfg.beta_blend() == 1.0
    cfg.sync_training_timesteps(350)
    assert abs(cfg.beta_blend() - 0.5) < 1e-6
    cfg.sync_training_timesteps(700)
    assert cfg.beta_blend() == 0.0
    cfg.sync_training_timesteps(800)
    assert cfg.beta_blend() == 0.0


def test_eta_default_is_theorem_scale_order():
    cfg = RewardConfig()
    bound = (1.0 - cfg.bsrs_gamma) / (1.0 + cfg.bsrs_gamma)
    assert cfg.eta_bsrs <= bound + 1e-4
