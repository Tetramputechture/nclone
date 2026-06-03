"""L1.4: game_state includes switch_activated (idx 27) and gold-remaining (idx 28)."""

from nclone.gym_environment.config import EnvironmentConfig
from nclone.gym_environment.constants import (
    GAME_STATE_CHANNELS,
    GAME_STATE_GOLD_REMAINING_IDX,
    GAME_STATE_SWITCH_ACTIVATED_IDX,
)
from nclone.gym_environment.npp_environment import NppEnvironment


def test_game_state_shape_and_switch_tail_match_obs_flag():
    cfg = EnvironmentConfig.for_training(seed=123)
    env = NppEnvironment(cfg)
    obs, _ = env.reset(seed=123)

    gs = obs["game_state"]
    assert gs.shape == (GAME_STATE_CHANNELS,) == (29,)
    assert GAME_STATE_SWITCH_ACTIVATED_IDX == 27
    assert GAME_STATE_GOLD_REMAINING_IDX == 28
    sw = bool(obs.get("switch_activated", False))
    assert float(gs[GAME_STATE_SWITCH_ACTIVATED_IDX]) == float(sw)
    assert gs[GAME_STATE_SWITCH_ACTIVATED_IDX] in (0.0, 1.0)
    # Gold-remaining fraction is in [0, 1]; at reset no gold is collected yet, so
    # it is 1.0 on any gold-bearing level (and 0.0 on a gold-free level).
    assert 0.0 <= float(gs[GAME_STATE_GOLD_REMAINING_IDX]) <= 1.0


def test_observation_space_exposes_29_dim_game_state():
    cfg = EnvironmentConfig.for_training(seed=42)
    env = NppEnvironment(cfg)
    assert env.observation_space["game_state"].shape == (GAME_STATE_CHANNELS,)
