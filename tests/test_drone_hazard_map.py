"""Patrol-drone hazard map (``drone_hazard_map``) tests.

Before this key existed the whole hazard pipeline was mine-shaped, so a
patrolling drone was indistinguishable from empty space in the policy
observation. These tests cover the three layers that had to line up:

- the Rust ``HazardSdf`` engine (radius-parameterized, no change-gate),
- the backend accessor that surfaces drones at all -- ``_build_entity_dic``
  whitelists only the static entity types, so drones are read from the live
  snapshot instead,
- the env's ``_compute_drone_hazard_map`` merge + the flag-gated obs key.

They also pin the two invariants that would fail silently: the min-merge of the
two drone classes (which have different kill radii) and the min-pooling on the
way to the policy, either of which would average a lethal cell away if it
became a mean.
"""

from __future__ import annotations

import numpy as np
import pytest

from nclone_native import HazardSdf, MineThreatMap

from nclone.gym_environment.constants import (
    DRONE_HAZARD_MAP_CHANNELS,
    DRONE_KILL_RADIUS_PX,
    HAZARD_MAP_HEIGHT,
    HAZARD_MAP_WIDTH,
    MINI_DRONE_KILL_RADIUS_PX,
)

SDF_SCALE = float(MineThreatMap.hazard_sdf_scale())
CELL = float(MineThreatMap.hazard_cell_size())
OFFSET = float(MineThreatMap.hazard_playfield_offset())


def _grid(raw):
    """`HazardSdf.compute` returns raw f32 bytes (see its docstring) — the
    Python-list return costs ~20x the SDF math. View it as float32."""
    return np.frombuffer(raw, dtype=np.float32)


def _reference_sdf(positions, kill_radius):
    """Python reference for the drone SDF. Returns `(H, W)`, matching the Rust
    row-major flat buffer before the env's transpose to `(W, H)`."""
    h, w = HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH
    if not positions:
        return np.ones((h, w), dtype=np.float32)
    ys = OFFSET + (np.arange(h, dtype=np.float64) + 0.5) * CELL
    xs = OFFSET + (np.arange(w, dtype=np.float64) + 0.5) * CELL
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    dists = np.full((h, w), np.inf, dtype=np.float64)
    for mx, my in positions:
        dists = np.minimum(dists, np.sqrt((grid_x - mx) ** 2 + (grid_y - my) ** 2))
    return np.clip((dists - kill_radius) / SDF_SCALE, -1.0, 1.0).astype(np.float32)


@pytest.fixture
def drone_fixture_20():
    """Deterministic 20-drone fixture inside the playable area."""
    rng = np.random.default_rng(20260802)
    xs = rng.uniform(OFFSET + 24.0, OFFSET + 984.0, size=20)
    ys = rng.uniform(OFFSET + 24.0, OFFSET + 528.0, size=20)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


# --------------------------------------------------------------------------
# Rust engine
# --------------------------------------------------------------------------


def test_empty_field_saturates_to_one():
    """A level with no drones must be indistinguishable from one where every
    drone is far away -- both read 1.0, matching the mine map's saturation."""
    out = _grid(HazardSdf(DRONE_KILL_RADIUS_PX).compute([]))
    assert out.shape == (HAZARD_MAP_HEIGHT * HAZARD_MAP_WIDTH,)
    assert np.all(out == 1.0)


def test_parity_with_mine_engine_at_same_radius():
    """The drone engine is the mine engine generalized over the kill radius:
    at r=14 the two must agree bit-for-bit, so the drone channel is in exactly
    the same units as `hazard_map` and their encoders are interchangeable."""
    pts = [(120.0, 120.0), (400.0, 300.0), (900.0, 500.0)]
    mine = np.asarray(MineThreatMap().compute_hazard_map(pts), dtype=np.float32)
    drone = _grid(HazardSdf(14.0).compute(pts))
    assert np.array_equal(mine, drone)


@pytest.mark.parametrize(
    "kill_radius", [DRONE_KILL_RADIUS_PX, MINI_DRONE_KILL_RADIUS_PX]
)
def test_matches_python_reference(drone_fixture_20, kill_radius):
    got = _grid(HazardSdf(kill_radius).compute(drone_fixture_20)).reshape(
        HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH
    )
    np.testing.assert_allclose(got, _reference_sdf(drone_fixture_20, kill_radius),
                               rtol=0, atol=1e-6)


def test_sign_convention_around_the_kill_boundary():
    """Negative strictly inside the contact-death disc, positive outside it.

    A cell center is placed exactly on a drone, one just inside the kill radius
    and one well outside; the boundary itself is the zero crossing.
    """
    # A drone sitting exactly on a cell center makes the arithmetic exact.
    col, row = 40, 30
    dx = OFFSET + (col + 0.5) * CELL
    dy = OFFSET + (row + 0.5) * CELL
    grid = _grid(HazardSdf(DRONE_KILL_RADIUS_PX).compute([(dx, dy)])).reshape(
        HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH
    )

    # d = 0 -> (0 - 17.5) / 24
    assert grid[row, col] == pytest.approx(-DRONE_KILL_RADIUS_PX / SDF_SCALE, abs=1e-6)
    # Two cells away = 12 px < 17.5 px kill radius -> still lethal, still negative.
    assert grid[row, col + 2] < 0.0
    # Far away -> saturated safe.
    assert grid[row, col + 40] == pytest.approx(1.0)


def test_kill_radius_is_per_class():
    """Zap (r=17.5) and mini (r=14) must not share a radius: at the same
    distance the larger drone reads as more dangerous."""
    pts = [(500.0, 300.0)]
    zap = _grid(HazardSdf(DRONE_KILL_RADIUS_PX).compute(pts))
    mini = _grid(HazardSdf(MINI_DRONE_KILL_RADIUS_PX).compute(pts))
    assert np.all(zap <= mini)
    assert zap.min() < mini.min()
    assert HazardSdf(DRONE_KILL_RADIUS_PX).kill_radius == DRONE_KILL_RADIUS_PX


def test_recompute_tracks_movement():
    """There is no change-gate: a moved drone must change the field.

    The mine engine caches on an unchanged position set, which is sound only
    because mines never move. Regressing the drone path onto that cache would
    freeze the field at the drone's spawn position.
    """
    engine = HazardSdf(DRONE_KILL_RADIUS_PX)
    a = _grid(engine.compute([(300.0, 300.0)]))
    b = _grid(engine.compute([(306.0, 300.0)]))
    assert not np.array_equal(a, b)
    # ...and an unchanged input still returns a full valid grid, never None.
    c = _grid(engine.compute([(306.0, 300.0)]))
    assert np.array_equal(b, c)


# --------------------------------------------------------------------------
# Env integration
# --------------------------------------------------------------------------


def _make_env(enable_drone_hazard, sim_backend="python"):
    from nclone.gym_environment.config import EnvironmentConfig
    from nclone.gym_environment.npp_environment import NppEnvironment

    cfg = EnvironmentConfig.for_training(seed=0, sim_backend=sim_backend)
    cfg.reachability.enable_drone_hazard = enable_drone_hazard
    return NppEnvironment(cfg)


def test_key_absent_by_default():
    """Default-off: the strict Layer 1 contract and existing checkpoints must
    be unaffected by this work landing."""
    env = _make_env(False)
    try:
        assert "drone_hazard_map" not in env.observation_space.spaces
        obs, _ = env.reset()
        assert "drone_hazard_map" not in obs
        # The mine hazard map is untouched.
        assert env.observation_space.spaces["hazard_map"].shape[0] == 2
    finally:
        env.close()


def test_key_present_and_shaped_when_enabled():
    env = _make_env(True)
    try:
        space = env.observation_space.spaces["drone_hazard_map"]
        assert space.shape[0] == DRONE_HAZARD_MAP_CHANNELS
        assert space.low.min() == -1.0 and space.high.max() == 1.0
        obs, _ = env.reset()
        assert obs["drone_hazard_map"].shape == space.shape
        assert obs["drone_hazard_map"].dtype == np.float32
        # The default map carries no drones -> saturated safe everywhere.
        assert np.all(obs["drone_hazard_map"] == 1.0)
    finally:
        env.close()


@pytest.mark.parametrize("sim_backend", ["python", "native"])
def test_backend_accessor_and_merge_on_a_real_drone_map(sim_backend):
    """The end-to-end path on an official 28-zap-drone level, BOTH backends.

    This is the case that was silently broken: the Rust sim simulates and
    snapshots drones, but they were dropped one layer up, so the policy saw 28
    lethal patrols as empty space. Training runs the native backend, so a test
    that only covered the Python one would not have caught it -- the two reach
    drones by different routes (entity_dic vs. the live snapshot).
    """
    from pathlib import Path
    from nclone.dev_tool.loaders import load_map_data_into_env

    map_path = (
        Path(__file__).resolve().parent.parent
        / "nclone" / "maps" / "official" / "R" / "173 variational mechanics"
    )
    if not map_path.exists():
        pytest.skip(f"official map corpus not available at {map_path}")

    env = _make_env(True, sim_backend=sim_backend)
    try:
        obs = load_map_data_into_env(env, [int(b) for b in map_path.read_bytes()])
        zap, mini = env.nplay_headless.get_drone_positions()
        assert len(zap) == 28, "zap drones must reach the backend accessor"
        assert len(mini) == 0

        field = env._compute_drone_hazard_map()
        assert field.shape == (
            DRONE_HAZARD_MAP_CHANNELS, HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT
        )
        # Some cells must read lethal, and the field must not be all-saturated
        # (which is exactly what "the policy cannot see drones" looked like).
        assert field.min() < 0.0
        assert not np.all(field == 1.0)

        # Matches the reference for this drone set, after the (H, W) -> (W, H)
        # transpose the env applies for co-registration with tile_grid.
        np.testing.assert_allclose(
            field[0], _reference_sdf(zap, DRONE_KILL_RADIUS_PX).T,
            rtol=0, atol=1e-6,
        )

        # Min-pooling to the policy grid never averages a threat away.
        assert obs["drone_hazard_map"].min() == pytest.approx(field.min())
    finally:
        env.close()


def test_class_merge_takes_the_worst_case(monkeypatch):
    """With both classes present the merge is an element-wise min, so a cell
    lethal to either drone type reads lethal."""
    env = _make_env(True)
    try:
        zap = [(300.0, 300.0)]
        mini = [(700.0, 300.0)]
        monkeypatch.setattr(
            env.nplay_headless, "get_drone_positions", lambda: (zap, mini)
        )
        merged = env._compute_drone_hazard_map()[0]
        expected = np.minimum(
            _reference_sdf(zap, DRONE_KILL_RADIUS_PX).T,
            _reference_sdf(mini, MINI_DRONE_KILL_RADIUS_PX).T,
        )
        np.testing.assert_allclose(merged, expected, rtol=0, atol=1e-6)
        # Both drones are visible in the merge.
        assert merged.min() < 0.0
        assert merged[:84].min() < 0.0 and merged[84:].min() < 0.0
    finally:
        env.close()


def test_mirror_rule_registered():
    """The mirror spec hard-errors on a present-but-unlisted key, so a missing
    rule would only surface once mirror augmentation was switched on."""
    from nclone.map_augmentation.obs_mirror import SPATIAL_MIRROR_SPEC

    rule = SPATIAL_MIRROR_SPEC["drone_hazard_map"]
    assert rule.flip_x is True
    assert not getattr(rule, "tile_remap", False)
    assert not getattr(rule, "negate_channels", ())


# --------------------------------------------------------------------------
# Forecast channels
#
# A present-frame field says only "a drone is there now". It cannot tell a
# corridor whose drone is walking away from one whose drone is walking in, and
# those demand opposite actions from an identical frame. Patrol drones are
# ninja-independent, so their future is a lookup rather than a prediction --
# which is what these tests pin: EXACT, not approximate.
# --------------------------------------------------------------------------


def _official_map(name):
    from pathlib import Path

    path = (
        Path(__file__).resolve().parent.parent / "nclone" / "maps" / "official" / name
    )
    if not path.exists():
        pytest.skip(f"official map corpus not available at {path}")
    return [int(b) for b in path.read_bytes()]


def _load(env, name):
    from nclone.dev_tool.loaders import load_map_data_into_env

    return load_map_data_into_env(env, _official_map(name))


@pytest.mark.parametrize("sim_backend", ["python", "native"])
def test_forecast_matches_the_sim_it_predicts(sim_backend):
    """The whole premise, checked against ground truth: the position reported
    for horizon k is the position the drone actually occupies after k ticks.

    Exact equality, not a tolerance. A drone's patrol is a pure function of the
    tile grid and the frame counter, so any discrepancy at all means the
    forecast has drifted from the simulator's own motion rules -- and a forecast
    that disagrees with what happens is worse than no forecast, because the
    policy would learn to trust it.
    """
    horizons = [12, 24, 48]
    env = _make_env(True, sim_backend=sim_backend)
    try:
        _load(env, "R/173 variational mechanics")
        headless = env.nplay_headless
        predicted = headless.get_predicted_drone_positions(horizons)

        for k in range(1, max(horizons) + 1):
            headless.tick(0, 0)
            if k not in horizons:
                continue
            zap, mini = headless.get_drone_positions()
            pred_zap, pred_mini = predicted[horizons.index(k)]
            assert pred_zap == zap, f"zap forecast wrong at t+{k}"
            assert pred_mini == mini, f"mini forecast wrong at t+{k}"
    finally:
        env.close()


@pytest.mark.parametrize("sim_backend", ["python", "native"])
def test_forecast_does_not_disturb_the_live_sim(sim_backend):
    """Rolling drones forward must not touch live state.

    The roll-out drives the simulator's REAL motion code (rather than a second
    copy of the patrol rules, which would drift), so it has to save and restore
    around it: the Python side shadows out ``grid_move`` and the collision log,
    the Rust side works on a cloned drone. A leak here would corrupt the
    episode itself, not just the observation.
    """
    env = _make_env(True, sim_backend=sim_backend)
    try:
        _load(env, "R/173 variational mechanics")
        headless = env.nplay_headless
        before = headless.get_drone_positions()
        headless.get_predicted_drone_positions([1, 7, 60])
        assert headless.get_drone_positions() == before

        # ...and the episode still evolves identically afterwards.
        headless.tick(0, 0)
        stepped = headless.get_drone_positions()
        headless.get_predicted_drone_positions([3])
        assert headless.get_drone_positions() == stepped
        assert stepped != before
    finally:
        env.close()


def test_backends_agree_on_the_forecast():
    """The Python roll-out and the Rust one are independent implementations of
    the same seam; training runs only the native path, so they are compared
    directly rather than each trusted alone."""
    horizons = [1, 9, 33]
    envs = {b: _make_env(True, sim_backend=b) for b in ("python", "native")}
    try:
        out = {}
        for backend, env in envs.items():
            _load(env, "R/173 variational mechanics")
            out[backend] = env.nplay_headless.get_predicted_drone_positions(horizons)
        assert out["python"] == out["native"]
    finally:
        for env in envs.values():
            env.close()


@pytest.mark.parametrize("sim_backend", ["python", "native"])
def test_horizon_zero_is_the_present_position(sim_backend):
    """The forecast and the present-frame read are the same function at k=0 --
    they must not disagree about the origin they are measured from."""
    env = _make_env(True, sim_backend=sim_backend)
    try:
        _load(env, "R/173 variational mechanics")
        headless = env.nplay_headless
        assert headless.get_predicted_drone_positions([0])[0] == (
            headless.get_drone_positions()
        )
    finally:
        env.close()


def test_chaser_drones_are_excluded():
    """The chaser (15) reads the ninja, so forecasting it would amount to
    predicting the agent's own future. It is excluded from BOTH the present-frame
    read and the forecast -- on a map carrying 5 chasers alongside 20 mini
    drones, the counts must be 20, never 25."""
    env = _make_env(True, sim_backend="native")
    try:
        _load(env, "R/034 oh, they'll swarm ya")
        headless = env.nplay_headless
        zap, mini = headless.get_drone_positions()
        assert (len(zap), len(mini)) == (0, 20)
        for pred_zap, pred_mini in headless.get_predicted_drone_positions([5, 30]):
            assert (len(pred_zap), len(pred_mini)) == (0, 20)
    finally:
        env.close()


def test_channels_are_the_field_at_their_horizon():
    """Channel i+1 of the map now must equal channel 0 of the map computed
    DRONE_HAZARD_HORIZONS[i] frames later -- the forecast channels carry the
    same encoding as the present one, just displaced in time."""
    from nclone.gym_environment.constants import DRONE_HAZARD_HORIZONS

    env = _make_env(True, sim_backend="native")
    try:
        _load(env, "R/173 variational mechanics")
        forecast = env._compute_drone_hazard_map().copy()
        assert forecast.shape[0] == 1 + len(DRONE_HAZARD_HORIZONS)

        for k in range(1, max(DRONE_HAZARD_HORIZONS) + 1):
            env.nplay_headless.tick(0, 0)
            if k not in DRONE_HAZARD_HORIZONS:
                continue
            i = DRONE_HAZARD_HORIZONS.index(k)
            actual_now = env._compute_drone_hazard_map()[0]
            np.testing.assert_array_equal(forecast[i + 1], actual_now)
    finally:
        env.close()


def test_forecast_channels_actually_differ_from_the_present():
    """A forecast that duplicates channel 0 would pass every exactness check
    above while adding nothing. On a real drone level the horizons must move
    the field, including cells that flip between lethal and safe."""
    from nclone.gym_environment.constants import DRONE_HAZARD_HORIZONS

    env = _make_env(True, sim_backend="native")
    try:
        _load(env, "R/173 variational mechanics")
        field = env._compute_drone_hazard_map()
        now = field[0]
        for i, horizon in enumerate(DRONE_HAZARD_HORIZONS):
            later = field[i + 1]
            assert not np.array_equal(now, later), f"t+{horizon} duplicates the present"
            # Somewhere a cell that is inside a kill radius now is safe later,
            # or vice versa -- the signal the present frame cannot express.
            assert np.any((now < 0.0) != (later < 0.0)), f"t+{horizon} flips no cell"
    finally:
        env.close()


def test_no_drones_saturates_every_channel():
    """The forecast must not leave stale values in the later channels on a level
    with no drones -- the whole map, all channels, reads 'safe'."""
    env = _make_env(True, sim_backend="native")
    try:
        env.reset()
        field = env._compute_drone_hazard_map()
        assert field.shape[0] == DRONE_HAZARD_MAP_CHANNELS
        assert np.all(field == 1.0)
    finally:
        env.close()
