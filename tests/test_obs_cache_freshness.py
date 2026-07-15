"""P0-1 regression: observation caches must be FRESH across level changes.

``NppEnvironment.reset()`` never calls ``BaseNppEnvironment.reset()``; on the
standard (multi-level) reset path it relies on
``cache_management.clear_all_caches_for_new_level`` /
``clear_all_caches_for_reset``. Those historically did NOT clear the lazily
populated observation caches (``_cached_tile_grid``, the mine-overlay
signature/channels, the nearest-deadly-mine feature cache, ``_prev_obs_cache``),
so after each level switch the policy saw the PREVIOUS level's tiles and mines
— and the hazard-PBRS reward read the previous level's nearest-mine distance.

These tests load two DIFFERENT mine levels through the exact reset path the
benchmark/curriculum training wrapper uses (``reset(options={"map_data": ...,
"new_level": True})``) and assert every previously-stale stream now describes
level B. Also covers P1-8: a relevant-mine cull installed for one level must
not leak into the next.
"""

from __future__ import annotations

import numpy as np
import pytest

from nclone.gym_environment.config import EnvironmentConfig
from nclone.gym_environment.constants import GAME_STATE_NEAREST_MINE_IDX
from nclone.gym_environment.entity_overlay_utils import (
    ENTITY_OVERLAY_CH_MINE_DEADLY,
    ENTITY_OVERLAY_CH_MINE_SAFE,
)
from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.map_generation.map import Map

# Normalizer for the scalar nearest-deadly-mine feature (see npp_environment).
_NEAREST_MINE_NORM_PX = 168.0


def _mine_level(floor_y: int, mine_xs: tuple, exit_x: int = 38, switch_x: int = 30):
    """Flat corridor at ``floor_y`` with DEADLY (type 21) mines above it.

    Both fixture levels use the SAME mine count so the historical count-only
    mine signature ``(len_t1, len_t21, n_deadly)`` collides across them — the
    exact aliasing that reproduced the stale-overlay bug.
    """
    m = Map()
    for x in range(2, 41):
        m.set_tile(x, floor_y, 1)
    m.set_ninja_spawn(4, floor_y - 1)
    m.add_entity(3, exit_x, floor_y - 1, 0, 0, switch_x=switch_x, switch_y=floor_y - 1)
    for mx in mine_xs:
        # Type 1 = toggle mine: starts TOGGLED (state 0, deadly) in this sim.
        m.add_entity(1, mx, floor_y - 1)
    return [int(v) for v in m.map_data()]


# Same mine COUNT, different tiles + mine positions.
LEVEL_A = _mine_level(floor_y=12, mine_xs=(20, 24, 28))
LEVEL_B = _mine_level(floor_y=8, mine_xs=(10, 14, 34))


def _make_env() -> NppEnvironment:
    return NppEnvironment(
        config=EnvironmentConfig(defer_initial_map_load=True, sim_backend="native")
    )


def _reset(env, md, name):
    obs, _ = env.reset(
        options={"map_data": list(md), "map_name": name, "new_level": True}
    )
    return obs


def _fresh_tile_grid(env) -> np.ndarray:
    return (
        np.array(env.nplay_headless.sim.get_tile_grid(), dtype=np.int8)
        .reshape(23, 42)
        .T
    )


def _deadly_mine_xy_from_sim(env):
    toggle, toggled = env.nplay_headless.get_mine_entities()
    return [
        (float(m.xpos), float(m.ypos))
        for m in (*toggle, *toggled)
        if getattr(m, "state", 1) == 0
    ]


@pytest.fixture(scope="module")
def env():
    e = _make_env()
    yield e
    try:
        e.close()
    except Exception:  # noqa: BLE001 - closing is best-effort in tests
        pass


def test_obs_streams_fresh_after_level_change(env):
    obs_a = _reset(env, LEVEL_A, "cache_fresh_level_a")
    tile_a = np.asarray(obs_a["tile_grid"]).copy()
    deadly_a = np.asarray(
        obs_a["entity_overlays"][ENTITY_OVERLAY_CH_MINE_DEADLY]
    ).copy()

    # Step a few frames on level A so every lazy cache is populated.
    for _ in range(5):
        _, _, term, trunc, _ = env.step(2)  # run right
        if term or trunc:
            break

    obs_b = _reset(env, LEVEL_B, "cache_fresh_level_b")

    # (a) tile_grid reflects level B's actual tiles (fresh from the sim) and
    # differs from level A's.
    tile_b = np.asarray(obs_b["tile_grid"])
    np.testing.assert_array_equal(
        tile_b, _fresh_tile_grid(env),
        err_msg="tile_grid after level switch is not the live sim's tile grid",
    )
    assert not np.array_equal(tile_b, tile_a), (
        "fixture levels must have different tiles"
    )

    # (b) entity_overlays mine channels describe level B, not level A. The two
    # levels share a mine COUNT, so a count-only signature would have reused
    # level A's rasterized channels here.
    deadly_b = np.asarray(obs_b["entity_overlays"][ENTITY_OVERLAY_CH_MINE_DEADLY])
    assert not np.array_equal(deadly_b, deadly_a), (
        "deadly-mine overlay channel still shows level A's mines after the "
        "level switch (stale _entity_mine_sig / _entity_mine_channels_cache)"
    )
    # Match a fresh recompute: rasterizing level B's live mine set must equal
    # what the observation returned.
    from nclone.gym_environment.entity_overlay_utils import (
        classify_mines_by_state,
        compute_entity_overlays,
    )

    toggle, toggled = env.nplay_headless.get_mine_entities()
    deadly_mines, safe_mines = classify_mines_by_state((*toggle, *toggled))
    fresh = compute_entity_overlays(
        ninja_pos=env.nplay_headless.ninja_position(),
        switch_pos=env.nplay_headless.exit_switch_position(),
        switch_activated=env.nplay_headless.exit_switch_activated(),
        exit_pos=env.nplay_headless.exit_door_position(),
        deadly_mines=deadly_mines,
        safe_mines=safe_mines,
    )
    np.testing.assert_allclose(
        deadly_b, fresh[ENTITY_OVERLAY_CH_MINE_DEADLY],
        err_msg="deadly-mine overlay differs from a fresh recompute",
    )
    np.testing.assert_allclose(
        np.asarray(obs_b["entity_overlays"][ENTITY_OVERLAY_CH_MINE_SAFE]),
        fresh[ENTITY_OVERLAY_CH_MINE_SAFE],
        err_msg="safe-mine overlay differs from a fresh recompute",
    )

    # (c) game_state nearest-deadly-mine feature reflects level B's mines
    # (brute-force nearest over the LIVE sim mine set).
    nx, ny = env.nplay_headless.ninja_position()
    deadly_xy = _deadly_mine_xy_from_sim(env)
    assert deadly_xy, "fixture level B must contain deadly mines"
    d_min = min(((mx - nx) ** 2 + (my - ny) ** 2) ** 0.5 for mx, my in deadly_xy)
    expected = min(d_min / _NEAREST_MINE_NORM_PX, 1.0)
    got = float(np.asarray(obs_b["game_state"])[GAME_STATE_NEAREST_MINE_IDX])
    assert got == pytest.approx(expected, abs=1e-4), (
        f"nearest-mine feature {got:.4f} != brute-force level-B value "
        f"{expected:.4f} (stale _deadly_xy_sig / _deadly_mine_xy)"
    )


def test_first_step_reward_has_fresh_prev_obs(env):
    """(d) The first step of a new episode must not compare against the
    previous episode's terminal observation — that pays a spurious
    switch-activation bonus when a curriculum seeds post-switch starts."""
    _reset(env, LEVEL_A, "cache_fresh_level_a")
    for _ in range(3):
        env.step(2)

    # New level: _prev_obs_cache must be dropped by the reset itself.
    _reset(env, LEVEL_B, "cache_fresh_level_b")
    assert env._prev_obs_cache is None, (
        "_prev_obs_cache survived a new-level reset"
    )
    assert not env.nplay_headless.exit_switch_activated()
    env.step(0)
    assert env.reward_calculator.episode_sparse_switch_reward == 0.0, (
        "spurious switch-activation reward on the first step after reset"
    )

    # Same-level reset (standard path, new_level False): also cleared.
    env.step(2)
    env.reset(
        options={
            "map_data": list(LEVEL_B),
            "map_name": "cache_fresh_level_b",
            "new_level": False,
        }
    )
    assert env._prev_obs_cache is None, (
        "_prev_obs_cache survived a same-level reset"
    )


def test_relevant_mine_cull_does_not_leak_across_levels(env):
    """P1-8: a mine cull installed for level A must not hide level B's mines."""
    _reset(env, LEVEL_A, "cache_fresh_level_a")

    # Manually install a cull (same API the env uses on >=256-mine levels)
    # whose key set matches NONE of level B's mines.
    env.nplay_headless.set_relevant_mine_keys({(0, 0)})
    toggle, toggled = env.nplay_headless.get_mine_entities()
    assert len(toggle) + len(toggled) == 0, "cull install did not take effect"

    obs_b = _reset(env, LEVEL_B, "cache_fresh_level_b")

    # All of level B's mines are visible again in the entity dic...
    toggle, toggled = env.nplay_headless.get_mine_entities()
    assert len(toggle) + len(toggled) == 3, (
        f"level B shows {len(toggle) + len(toggled)}/3 mines — the previous "
        "level's relevant-mine cull leaked across the map load"
    )
    assert env.nplay_headless.sim._relevant_mine_keys is None

    # ...and in the overlay channels.
    deadly_b = np.asarray(obs_b["entity_overlays"][ENTITY_OVERLAY_CH_MINE_DEADLY])
    assert float(deadly_b.sum()) > 0.0, (
        "mine overlay empty on level B after a leaked cull"
    )
