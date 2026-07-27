"""L1: game_state tail = switch_activated (27) + goal vectors (28-31) +
surface raycasts (32-39) + free-space nav hint (40-43) + nearest-mine (44) +
closing-speed (45)."""

import numpy as np

from nclone.gym_environment.config import EnvironmentConfig
from nclone.gym_environment.constants import (
    GAME_STATE_CHANNELS,
    GAME_STATE_SWITCH_ACTIVATED_IDX,
    GAME_STATE_SWITCH_VEC_IDX,
    GAME_STATE_EXIT_VEC_IDX,
    GAME_STATE_RAYCAST_IDX,
    GAME_STATE_RAYCAST_DIM,
    GAME_STATE_FREE_SPACE_NAV_IDX,
    GAME_STATE_NEAREST_MINE_IDX,
    GAME_STATE_CLOSING_SPEED_IDX,
    LEVEL_WIDTH,
    LEVEL_HEIGHT,
)
from nclone.gym_environment.npp_environment import NppEnvironment


def _make_env_resilient():
    """Construct + reset an env, retrying seeds to dodge the unrelated pre-existing
    ``extremely_simple`` curriculum-category crash (a default-config map-gen bug,
    not an observation issue). Returns (env, obs) from the single reset so callers
    read positions consistent with the returned observation."""
    last = None
    for seed in range(123, 220):
        try:
            env = NppEnvironment(EnvironmentConfig.for_training(seed=seed))
            obs, _ = env.reset(seed=seed)
            return env, obs
        except ValueError as e:  # noqa: PERF203
            last = e
            continue
    raise AssertionError(f"could not construct env on any seed: {last}")


def test_game_state_shape_and_tail_layout():
    env, obs = _make_env_resilient()

    gs = obs["game_state"]
    # Base game_state is GAME_STATE_CHANNELS (261: v16 15-wide arcs + passive + goal),
    # plus the ALWAYS-ON gold-fraction-remaining tail scalar at index
    # GAME_STATE_CHANNELS (docs/GOLD_AND_SCORE_OPTIMIZATION.md part b).
    assert GAME_STATE_CHANNELS == 261
    assert gs.shape == (GAME_STATE_CHANNELS + 1,)
    gold_frac = float(gs[GAME_STATE_CHANNELS])
    assert 0.0 <= gold_frac <= 1.0
    assert GAME_STATE_SWITCH_ACTIVATED_IDX == 27
    assert GAME_STATE_SWITCH_VEC_IDX == 28
    assert GAME_STATE_EXIT_VEC_IDX == 30
    assert GAME_STATE_RAYCAST_IDX == 32
    assert GAME_STATE_FREE_SPACE_NAV_IDX == 40
    assert GAME_STATE_NEAREST_MINE_IDX == 44
    assert GAME_STATE_CLOSING_SPEED_IDX == 45

    sw = bool(obs.get("switch_activated", False))
    assert float(gs[GAME_STATE_SWITCH_ACTIVATED_IDX]) == float(sw)
    assert gs[GAME_STATE_SWITCH_ACTIVATED_IDX] in (0.0, 1.0)

    # Goal vectors are clipped to [-1, 1].
    goal_vecs = gs[GAME_STATE_SWITCH_VEC_IDX:GAME_STATE_EXIT_VEC_IDX + 2]
    assert goal_vecs.shape == (4,)
    assert np.all(goal_vecs >= -1.0) and np.all(goal_vecs <= 1.0)

    # Surface ray-casts are normalized distances in [0, 1].
    rays = gs[GAME_STATE_RAYCAST_IDX:GAME_STATE_RAYCAST_IDX + GAME_STATE_RAYCAST_DIM]
    assert rays.shape == (8,)
    assert np.all(rays >= 0.0) and np.all(rays <= 1.0)

    # Free-space nav hint: [dx_hop, dy_hop, cost_norm, phase], all in [-1, 1].
    fsn = gs[GAME_STATE_FREE_SPACE_NAV_IDX:GAME_STATE_FREE_SPACE_NAV_IDX + 4]
    assert fsn.shape == (4,)
    assert np.all(fsn >= -1.0) and np.all(fsn <= 1.0)

    # Nearest-mine distance in [0, 1]; closing speed in [-1, 1].
    assert 0.0 <= float(gs[GAME_STATE_NEAREST_MINE_IDX]) <= 1.0
    assert -1.0 <= float(gs[GAME_STATE_CLOSING_SPEED_IDX]) <= 1.0


def test_goal_vectors_match_relative_geometry():
    env, obs = _make_env_resilient()
    gs = obs["game_state"]

    nx, ny = env.nplay_headless.ninja_position()
    sx, sy = env.nplay_headless.exit_switch_position()
    ex, ey = env.nplay_headless.exit_door_position()

    exp = np.clip(
        np.array(
            [
                (sx - nx) / LEVEL_WIDTH,
                (sy - ny) / LEVEL_HEIGHT,
                (ex - nx) / LEVEL_WIDTH,
                (ey - ny) / LEVEL_HEIGHT,
            ],
            dtype=np.float32,
        ),
        -1.0,
        1.0,
    )
    np.testing.assert_allclose(
        gs[GAME_STATE_SWITCH_VEC_IDX:GAME_STATE_EXIT_VEC_IDX + 2], exp, atol=1e-5
    )


def test_observation_space_exposes_full_dim_game_state():
    env, _obs = _make_env_resilient()
    # GAME_STATE_CHANNELS base + the always-on gold-fraction-remaining tail (+1).
    assert env.observation_space["game_state"].shape == (GAME_STATE_CHANNELS + 1,)
    # v16 blocks: 69 head + 45 committed + 45 jump-now + 6 arc-phi + 45 tap-jump
    # + 45 wall-jump-now + 6 passive-release (arcs are 15-wide: v15 13 + v16
    # goal_approach/goal_hit).
    assert GAME_STATE_CHANNELS == 261


def test_tap_arc_block_layout():
    from nclone.gym_environment.constants import (
        GAME_STATE_TAP_ARC_IDX,
        GAME_STATE_TAP_ARC_DIM,
        GAME_STATE_WALL_ARC_IDX,
        GAME_STATE_WALL_ARC_DIM,
        GAME_STATE_ARC_PHI_IDX,
        GAME_STATE_ARC_PHI_DIM,
        GAME_STATE_PASSIVE_ARC_IDX,
        GAME_STATE_PASSIVE_ARC_DIM,
    )

    # Tap row follows arc-phi; wall row follows tap; v16 passive-release arc is
    # the final block.
    assert GAME_STATE_TAP_ARC_IDX == GAME_STATE_ARC_PHI_IDX + GAME_STATE_ARC_PHI_DIM
    assert GAME_STATE_WALL_ARC_IDX == GAME_STATE_TAP_ARC_IDX + GAME_STATE_TAP_ARC_DIM
    assert GAME_STATE_PASSIVE_ARC_IDX == GAME_STATE_WALL_ARC_IDX + GAME_STATE_WALL_ARC_DIM
    assert GAME_STATE_PASSIVE_ARC_IDX + GAME_STATE_PASSIVE_ARC_DIM == GAME_STATE_CHANNELS

    env, obs = _make_env_resilient()
    gs = obs["game_state"]
    tap = gs[GAME_STATE_TAP_ARC_IDX:GAME_STATE_TAP_ARC_IDX + GAME_STATE_TAP_ARC_DIM]
    assert tap.shape == (GAME_STATE_TAP_ARC_DIM,)
    # All tap-arc dims are normalized to [-1, 1] by the projector contract.
    assert np.all(tap >= -1.0) and np.all(tap <= 1.0)
    # On the reset frame the ninja is grounded: the tap row must be populated
    # (a projected takeoff arc always terminates somewhere -> at least one of
    # the contact-normal / mine / none dims per arc is nonzero), mirroring the
    # jump-now row.
    n = getattr(env.nplay_headless.sim, "ninja", None)
    if n is not None and not bool(getattr(n, "airborn", False)):
        from nclone.gym_environment.constants import GAME_STATE_GHOST_ARC_PER_ARC
        per = np.asarray(tap).reshape(3, GAME_STATE_GHOST_ARC_PER_ARC)
        assert all(np.any(np.abs(a[2:6]) > 0.0) for a in per)


def test_wall_arc_block_zero_when_unwalled():
    from nclone.gym_environment.constants import (
        GAME_STATE_WALL_ARC_IDX,
        GAME_STATE_WALL_ARC_DIM,
    )

    env, obs = _make_env_resilient()
    gs = obs["game_state"]
    wall = gs[GAME_STATE_WALL_ARC_IDX:GAME_STATE_WALL_ARC_IDX + GAME_STATE_WALL_ARC_DIM]
    assert wall.shape == (GAME_STATE_WALL_ARC_DIM,)
    assert np.all(wall >= -1.0) and np.all(wall <= 1.0)
    # On the reset frame the ninja is grounded and un-walled: the row is zeros.
    n = getattr(env.nplay_headless.sim, "ninja", None)
    if n is not None and not bool(getattr(n, "airborn", False)):
        assert np.all(wall == 0.0)
