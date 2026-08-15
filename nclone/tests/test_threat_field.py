"""Contract and exactness tests for the Rust `threat_field` engine.

The engine replaces `HazardSdf` + `predict_drone_positions` for the policy's
threat observation. Two claims have to hold, and both are easy to break silently:

1. **It is exact, not an approximation.** Two separate restatements are involved
   -- scattering straight into the 12 px grid instead of pooling a 6 px grid, and
   evaluating one analytically-chosen sub-centre per output cell instead of all
   four. Either could plausibly be "close enough" while being wrong, and a
   half-right hazard field is worse than none: the policy would learn to trust a
   safety margin that does not exist. So the tier-C channels are compared
   BIT-FOR-BIT against the legacy path on real maps at real densities.

2. **Cost does not scale with entity count across FFI.** The whole reason the
   engine exists is that the corpus reaches 770 movers on one map and 165 at p99,
   where marshalling positions into Python cost more than the physics tick.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nclone.gym_environment.constants import (
    DRONE_HAZARD_HORIZONS,
    DRONE_KILL_RADIUS_PX,
    HAZARD_MAP_HEIGHT,
    HAZARD_MAP_WIDTH,
    MINI_DRONE_KILL_RADIUS_PX,
)
from nclone.gym_environment.obs_map_pooling import pool_channels
from nclone.native_headless import NativePlayHeadless

OUT_W, OUT_H = 84, 46
N_TIERS = 2
TIER_CERTAIN, TIER_REACTIVE = 0, 1
_OFFICIAL = Path(__file__).resolve().parents[1] / "maps" / "official"


def _load(name):
    """Load an official map by basename, skipping if the corpus is absent."""
    if not _OFFICIAL.is_dir():
        pytest.skip("official map corpus not present")
    hits = list(_OFFICIAL.rglob(name))
    if not hits:
        pytest.skip(f"map {name!r} not in corpus")
    sim = NativePlayHeadless()
    sim.load_map(str(hits[0]))
    return sim


def _read(sim, horizons):
    """threat_field as (n_channels, OUT_W, OUT_H), matching the obs axis order."""
    raw = sim._native_sim.threat_field(list(horizons))
    n_ch = N_TIERS * (1 + len(horizons))
    # Rust emits row-major (row, col) per channel; transpose to (W, H) like the
    # rest of the spatial observation stack.
    return np.frombuffer(raw, dtype=np.float32).reshape(n_ch, OUT_H, OUT_W).transpose(0, 2, 1)


def _legacy_pooled(zap, mini):
    """The path this replaces: 6 px scatter per drone class, min-merge, min-pool."""
    from nclone_native import HazardSdf

    field = None
    for points, engine in (
        (zap, HazardSdf(DRONE_KILL_RADIUS_PX)),
        (mini, HazardSdf(MINI_DRONE_KILL_RADIUS_PX)),
    ):
        if not points:
            continue
        chan = (
            np.frombuffer(engine.compute(points), dtype=np.float32)
            .reshape(HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH)
            .T
        )
        field = chan if field is None else np.minimum(field, chan)
    if field is None:
        field = np.ones((HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT), np.float32)
    return pool_channels(field[None], 2, ["min"])[0]


# Real maps spanning the corpus density range: p50, mid, and the two densest.
_DENSITY_CASES = [
    "000 the basics",  # 0 movers
    "205 upward over the mountain",  # 256
    "589 escape is at hand",  # 261
    "109 out of the micro",  # 334
    "097 compendium for literates",  # 737 -- densest in the corpus
]


@pytest.mark.parametrize("map_name", _DENSITY_CASES)
def test_certain_tier_is_bit_identical_to_the_legacy_path(map_name):
    """Tier C must equal min-pool(6 px scatter) exactly, at every horizon.

    Run over many ticks so the drones actually move through their patrols and
    the comparison covers a wide spread of sub-cell offsets -- the analytic
    sub-centre pick is exactly where an off-by-half-a-cell error would hide.
    """
    sim = _load(map_name)
    for _ in range(40):
        sim.tick(0, 0)
        got = _read(sim, DRONE_HAZARD_HORIZONS)
        zap, mini = sim.get_drone_positions()
        expected = [_legacy_pooled(zap, mini)] + [
            _legacy_pooled(z, m)
            for z, m in sim.get_predicted_drone_positions(DRONE_HAZARD_HORIZONS)
        ]
        for h_idx, exp in enumerate(expected):
            np.testing.assert_array_equal(
                got[TIER_CERTAIN * (1 + len(DRONE_HAZARD_HORIZONS)) + h_idx],
                exp,
                err_msg=f"{map_name}: tier-C horizon {h_idx} diverged from the 6px path",
            )


def test_shape_and_range_contract():
    sim = _load("000 the basics")
    sim.tick(0, 0)
    field = _read(sim, DRONE_HAZARD_HORIZONS)
    assert field.shape == (N_TIERS * (1 + len(DRONE_HAZARD_HORIZONS)), OUT_W, OUT_H)
    assert field.dtype == np.float32
    assert field.min() >= -1.0 and field.max() <= 1.0


def test_horizon_count_is_variable():
    """The channel count follows the requested horizons, including zero of them."""
    sim = _load("589 escape is at hand")
    sim.tick(0, 0)
    for horizons in ([], [12], [12, 24], [6, 12, 24, 48]):
        assert _read(sim, horizons).shape[0] == N_TIERS * (1 + len(horizons))


def test_threat_free_level_saturates_to_one():
    """No threats must read exactly 1.0 everywhere -- the same "far away" value a
    distant threat saturates to, so a threat-free level is not a special case the
    encoder has to learn separately."""
    sim = _load("000 the basics")
    sim.tick(0, 0)
    zap, mini = sim.get_drone_positions()
    assert not zap and not mini, "expected a drone-free map for this test"
    field = _read(sim, DRONE_HAZARD_HORIZONS)
    assert np.all(field == 1.0)


@pytest.mark.parametrize(
    "map_name,expected_kind",
    [("044 one got loose", "death_ball"), ("166 tutorial flashback", "chaser")],
)
def test_reactive_tier_is_populated_and_dilates(map_name, expected_kind):
    """Tier R must (a) actually fire for reactive threats and (b) grow with the
    horizon.

    A reactive threat cannot be forecast -- rolling it forward would be
    predicting the agent's own future -- so its forecast channels are worst-case
    reachability envelopes. Two properties follow and both are load-bearing: the
    envelope must be monotonically NON-DECREASING in danger with the horizon (a
    longer horizon can only add reachable ground), and it must never be tighter
    than the present frame, which would claim a safety that does not exist.
    """
    sim = _load(map_name)
    for _ in range(30):
        sim.tick(0, 0)
    horizons = list(DRONE_HAZARD_HORIZONS)
    field = _read(sim, horizons)
    base = TIER_REACTIVE * (1 + len(horizons))
    present = field[base]
    assert present.min() < 1.0, f"tier R never fired on a {expected_kind} map"
    prev = present
    for h_idx in range(1, 1 + len(horizons)):
        chan = field[base + h_idx]
        assert np.all(chan <= prev + 1e-7), (
            f"horizon {h_idx} is SAFER than horizon {h_idx - 1}; a worst-case "
            "envelope may only grow"
        )
        prev = chan


def test_buffer_is_refilled_not_accumulated():
    """The engine reuses one buffer across calls. If it failed to reset, stale
    threat would persist after the source moved away -- a phantom hazard that no
    shape or range assertion would catch."""
    sim = _load("589 escape is at hand")
    for _ in range(10):
        sim.tick(0, 0)
    first = _read(sim, DRONE_HAZARD_HORIZONS).copy()
    for _ in range(40):
        sim.tick(0, 0)
    moved = _read(sim, DRONE_HAZARD_HORIZONS).copy()
    assert not np.array_equal(first, moved), "field never changed as drones moved"
    # Recomputing without stepping must be idempotent.
    again = _read(sim, DRONE_HAZARD_HORIZONS).copy()
    np.testing.assert_array_equal(moved, again)


def test_cost_is_independent_of_entity_count_across_ffi():
    """The engine's whole purpose: no per-entity Python object crosses FFI.

    Compared against `predict_drone_positions`, which marshals one tuple per
    drone per horizon and is the call this replaces. On the densest map that
    marshal alone costs more than five physics ticks.
    """
    import time

    sim = _load("097 compendium for literates")
    for _ in range(20):
        sim.tick(0, 0)
    horizons = list(DRONE_HAZARD_HORIZONS)
    zap, mini = sim.get_drone_positions()
    assert len(zap) + len(mini) > 500, "expected the densest map"

    def timed(fn, reps=40):
        for _ in range(3):
            fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps

    marshal = timed(lambda: sim._native_sim.predict_drone_positions(horizons))
    field = timed(lambda: sim._native_sim.threat_field(horizons))
    # The field does strictly more work (8 channels of SDF, vs positions only),
    # yet must not be dominated by marshalling. Generous bound: this is a
    # regression guard against reintroducing per-entity FFI, not a benchmark.
    assert field < marshal * 6, (
        f"threat_field {field * 1e6:.0f}us vs bare position marshal "
        f"{marshal * 1e6:.0f}us -- per-entity FFI may have crept back in"
    )
