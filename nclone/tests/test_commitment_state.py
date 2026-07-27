"""v26 W6c: the commitment / irreversibility block.

Impact deaths are committed at TAKEOFF, and the per-arc fatality bits the policy
needs to answer "is every option from here fatal?" already exist — spread across
five 3-arc bundles of 45 dims each, whose validity depends on the contact regime.
This block computes the conjunction directly.

These tests pin: (1) the per-arc fatality classifier matches the sim's two kill
paths, (2) only regime-valid arc rows are counted (a zero-filled row reads as a
perfectly safe landing and would make ``no_escape`` unreachable), (3) zero is the
safe reading in every dim, (4) the block is emitted and sized correctly.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest

from nclone.gym_environment.config import EnvironmentConfig, ReachabilityConfig
from nclone.gym_environment.constants import (
    GAME_STATE_GHOST_ARC_ARCS,
    GAME_STATE_GHOST_ARC_IDX,
    GAME_STATE_GHOST_ARC_PER_ARC,
    GAME_STATE_JUMP_ARC_IDX,
    GAME_STATE_PASSIVE_ARC_IDX,
    GAME_STATE_TAP_ARC_IDX,
    GAME_STATE_WALL_ARC_IDX,
)
from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.map_generation.map_jump_platforms import MapJumpPlatforms


PA = GAME_STATE_GHOST_ARC_PER_ARC


class _Ninja:
    def __init__(self, airborn=False, walled=False):
        self.airborn = airborn
        self.walled = walled


def _make_env(**reach):
    cfg = EnvironmentConfig.for_training(
        frame_skip=1,
        sim_backend="python",
        reachability=ReachabilityConfig(**reach),
    )
    return NppEnvironment(config=cfg)


def _mine_arc():
    a = [0.0] * PA
    a[4] = 1.0  # deadly-mine landing
    return a


def _lethal_impact_arc():
    a = [0.0] * PA
    a[5] = 0.0  # contact happened
    a[7] = 0.8  # impact_ratio/2 = 0.8 -> ratio 1.6 > 1 -> fatal
    return a


def _safe_arc(ratio_half=0.1):
    a = [0.0] * PA
    a[7] = ratio_half
    return a


def _no_contact_arc():
    a = [0.0] * PA
    a[5] = 1.0
    return a


# ---------------------------------------------------------------------------
# The per-arc classifier
# ---------------------------------------------------------------------------
def test_arc_classifier_matches_both_sim_kill_paths():
    f = NppEnvironment._arc_fatal_and_margin
    assert f(_mine_arc())[0] is True
    assert f(_lethal_impact_arc())[0] is True
    assert f(_safe_arc())[0] is False
    # Exactly at the threshold (ratio == 1) survives: the sim kills on ">".
    at_threshold = [0.0] * PA
    at_threshold[7] = 0.5
    assert f(at_threshold)[0] is False


def test_arc_margin_is_one_minus_impact_ratio():
    f = NppEnvironment._arc_fatal_and_margin
    assert f(_safe_arc(0.1))[1] == pytest.approx(0.8)
    assert f(_safe_arc(0.0))[1] == pytest.approx(1.0)
    assert f(_no_contact_arc())[1] == pytest.approx(1.0)
    assert f(_mine_arc())[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Regime gating and the conjunction
# ---------------------------------------------------------------------------
def _fill(env, idx0, arcs):
    for i, arc in enumerate(arcs):
        env._game_state_buffer[idx0 + i * PA : idx0 + (i + 1) * PA] = arc


def _blank(env):
    env._game_state_buffer[:] = [0.0] * len(env._game_state_buffer)


def test_grounded_reads_the_jump_rows_not_the_committed_row():
    env = _make_env()
    _blank(env)
    # A fatal COMMITTED row must be ignored while grounded (that row is
    # zero-filled by the real fill path and carries no meaning on the ground).
    _fill(env, GAME_STATE_GHOST_ARC_IDX, [_mine_arc()] * GAME_STATE_GHOST_ARC_ARCS)
    _fill(env, GAME_STATE_JUMP_ARC_IDX, [_safe_arc()] * GAME_STATE_GHOST_ARC_ARCS)
    _fill(env, GAME_STATE_TAP_ARC_IDX, [_safe_arc()] * GAME_STATE_GHOST_ARC_ARCS)

    out = env._compute_commitment_state(_Ninja(airborn=False))
    assert out[0] == 0.0, "committed row must not be counted while grounded"
    assert out[1] == 0.0, "all jump arcs safe -> no takeoff trap"
    assert out[4] == 0.0


def test_grounded_takeoff_trap_when_every_jump_arc_is_fatal():
    env = _make_env()
    _blank(env)
    _fill(env, GAME_STATE_JUMP_ARC_IDX, [_mine_arc()] * GAME_STATE_GHOST_ARC_ARCS)
    _fill(
        env, GAME_STATE_TAP_ARC_IDX,
        [_lethal_impact_arc()] * GAME_STATE_GHOST_ARC_ARCS,
    )

    out = env._compute_commitment_state(_Ninja(airborn=False))
    assert out[1] == pytest.approx(1.0)
    assert out[4] == 1.0, "no jump option survives -> point of no return"
    assert out[5] == pytest.approx(1.0)


def test_partial_takeoff_trap_is_graded_not_binary():
    env = _make_env()
    _blank(env)
    _fill(
        env, GAME_STATE_JUMP_ARC_IDX,
        [_mine_arc(), _mine_arc(), _safe_arc(0.25)],
    )
    _fill(env, GAME_STATE_TAP_ARC_IDX, [_mine_arc()] * GAME_STATE_GHOST_ARC_ARCS)

    out = env._compute_commitment_state(_Ninja(airborn=False))
    assert out[1] == pytest.approx(5 / 6)
    assert out[4] == 0.0, "one surviving arc means there IS an escape"
    assert out[5] == pytest.approx(0.5), "deficit tracks the BEST option"


def test_airborne_counts_committed_and_passive():
    env = _make_env()
    _blank(env)
    _fill(
        env, GAME_STATE_GHOST_ARC_IDX,
        [_lethal_impact_arc()] * GAME_STATE_GHOST_ARC_ARCS,
    )
    env._game_state_buffer[GAME_STATE_PASSIVE_ARC_IDX + 4] = 1.0  # passive fatal

    out = env._compute_commitment_state(_Ninja(airborn=True))
    assert out[0] == pytest.approx(1.0)
    assert out[3] == 1.0
    assert out[4] == 1.0
    assert out[1] == 0.0, "jump rows are invalid while airborne"


def test_airborne_survivable_passive_fall_is_an_escape():
    env = _make_env()
    _blank(env)
    _fill(env, GAME_STATE_GHOST_ARC_IDX, [_mine_arc()] * GAME_STATE_GHOST_ARC_ARCS)
    env._game_state_buffer[GAME_STATE_PASSIVE_ARC_IDX + 4] = 0.0  # survives
    env._game_state_buffer[GAME_STATE_PASSIVE_ARC_IDX + 5] = 1.0  # no contact yet

    out = env._compute_commitment_state(_Ninja(airborn=True))
    assert out[0] == pytest.approx(1.0)
    assert out[3] == 0.0
    assert out[4] == 0.0, "letting go still works -> not committed"


def test_wall_row_only_counts_while_walled():
    env = _make_env()
    _blank(env)
    _fill(env, GAME_STATE_GHOST_ARC_IDX, [_safe_arc()] * GAME_STATE_GHOST_ARC_ARCS)
    _fill(env, GAME_STATE_WALL_ARC_IDX, [_mine_arc()] * GAME_STATE_GHOST_ARC_ARCS)

    unwalled = env._compute_commitment_state(_Ninja(airborn=True, walled=False))
    assert unwalled[2] == 0.0

    walled = env._compute_commitment_state(_Ninja(airborn=True, walled=True))
    assert walled[2] == pytest.approx(1.0)


def test_all_safe_state_is_all_zero():
    """Zero must be the safe reading in every dim, so the missing-block fallback
    is correct rather than merely conventional."""
    env = _make_env()
    _blank(env)
    _fill(env, GAME_STATE_JUMP_ARC_IDX, [_safe_arc(0.0)] * GAME_STATE_GHOST_ARC_ARCS)
    _fill(env, GAME_STATE_TAP_ARC_IDX, [_safe_arc(0.0)] * GAME_STATE_GHOST_ARC_ARCS)
    out = env._compute_commitment_state(_Ninja(airborn=False))
    assert np.allclose(out, 0.0), out


# ---------------------------------------------------------------------------
# Environment wiring
# ---------------------------------------------------------------------------
def test_block_grows_game_state_and_matches_the_space():
    base = _make_env()
    with_block = _make_env(expose_commitment_state=True)
    md = list(MapJumpPlatforms(seed=3).generate(seed=3).map_data())

    obs_a, _ = base.reset(
        options={"map_data": md, "map_name": "jp3", "new_level": True}
    )
    obs_b, _ = with_block.reset(
        options={"map_data": md, "map_name": "jp3", "new_level": True}
    )

    grew = obs_b["game_state"].shape[0] - obs_a["game_state"].shape[0]
    assert grew == NppEnvironment.COMMITMENT_STATE_DIM
    space = with_block.observation_space.spaces["game_state"]
    assert space.shape == obs_b["game_state"].shape
    assert space.contains(obs_b["game_state"])


def test_block_stays_in_range_over_a_rollout():
    env = _make_env(expose_commitment_state=True)
    md = list(MapJumpPlatforms(seed=3).generate(seed=3).map_data())
    obs, _ = env.reset(
        options={"map_data": md, "map_name": "jp3", "new_level": True}
    )
    d = NppEnvironment.COMMITMENT_STATE_DIM
    for _ in range(150):
        obs, _, term, trunc, info = env.step(np.random.randint(0, 6))
        block = obs["game_state"][-d:]
        assert np.isfinite(block).all()
        assert (block >= 0.0).all() and (block <= 1.0).all(), block
        assert "commitment_no_escape" in info
        if term or trunc:
            obs, _ = env.reset(
                options={"map_data": md, "map_name": "jp3", "new_level": True}
            )
