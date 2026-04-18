"""L1.4: game_state includes switch_activated as the 28th dimension."""

from nclone.gym_environment.config import EnvironmentConfig
from nclone.gym_environment.constants import (
    GAME_STATE_CHANNELS,
    GAME_STATE_SWITCH_ACTIVATED_IDX,
)
from nclone.gym_environment.npp_environment import NppEnvironment


def test_game_state_shape_and_switch_tail_match_obs_flag():
    cfg = EnvironmentConfig.for_training(seed=123)
    env = NppEnvironment(cfg)
    obs, _ = env.reset(seed=123)

    gs = obs["game_state"]
    assert gs.shape == (GAME_STATE_CHANNELS,) == (28,)
    assert GAME_STATE_SWITCH_ACTIVATED_IDX == 27
    sw = bool(obs.get("switch_activated", False))
    assert float(gs[GAME_STATE_SWITCH_ACTIVATED_IDX]) == float(sw)
    assert gs[GAME_STATE_SWITCH_ACTIVATED_IDX] in (0.0, 1.0)


def test_observation_space_exposes_28_dim_game_state():
    cfg = EnvironmentConfig.for_training(seed=42)
    env = NppEnvironment(cfg)
    assert env.observation_space["game_state"].shape == (GAME_STATE_CHANNELS,)
