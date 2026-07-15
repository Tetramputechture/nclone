"""NPP-46: 1-channel deadly-mine hazard_map tests.

These tests exercise the Rust `MineThreatMap.compute_hazard_map*` API and a
Python numpy reference implementation of the same SDF. They validate:

- shape / dtype contract of the `hazard_map` key,
- sign convention (positive outside the 14 px kill zone, zero at boundary,
  negative inside),
- response to a mine toggling safe -> deadly,
- numerical parity with a Python reference over a 100-mine fixture,
- compute_hazard_map_if_changed caching (returns None when inputs repeat).
"""

from __future__ import annotations

import numpy as np
import pytest

from nclone_native import MineThreatMap

from nclone.gym_environment.constants import (
    HAZARD_MAP_HEIGHT,
    HAZARD_MAP_WIDTH,
)


KILL_RADIUS = float(MineThreatMap.hazard_kill_radius())
SDF_SCALE = float(MineThreatMap.hazard_sdf_scale())
CELL = float(MineThreatMap.hazard_cell_size())
# Raw sim px offset of hazard cell (0, 0): the sim's 1-tile solid border means
# the playable area starts at 24 px; the 168x92 grid covers exactly the
# playable region (co-registered with tile_grid's border-stripped frame).
OFFSET = float(MineThreatMap.hazard_playfield_offset())


def _reference_hazard_map(deadly_positions):
    """Python reference for the 1-channel deadly SDF used by `hazard_map`.

    Output shape `(H, W)` matches the Rust row-major flat buffer before the
    `(1, H, W)` reshape applied by the Python wrapper.
    """
    h, w = HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH
    out = np.ones((h, w), dtype=np.float32)
    if not deadly_positions:
        return out
    ys = OFFSET + (np.arange(h, dtype=np.float64) + 0.5) * CELL
    xs = OFFSET + (np.arange(w, dtype=np.float64) + 0.5) * CELL
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    mines = np.asarray(deadly_positions, dtype=np.float64)
    dists = np.full((h, w), np.inf, dtype=np.float64)
    for (mx, my) in mines:
        d = np.sqrt((grid_x - mx) ** 2 + (grid_y - my) ** 2)
        dists = np.minimum(dists, d)
    normalized = np.clip((dists - KILL_RADIUS) / SDF_SCALE, -1.0, 1.0)
    return normalized.astype(np.float32)


@pytest.fixture
def mine_fixture_100():
    """Deterministic 100-mine fixture inside the playable area."""
    rng = np.random.default_rng(seed=46)
    xs = rng.uniform(24.0, HAZARD_MAP_WIDTH * CELL - 24.0, size=100)
    ys = rng.uniform(24.0, HAZARD_MAP_HEIGHT * CELL - 24.0, size=100)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def _compute(map_rs: MineThreatMap, deadly) -> np.ndarray:
    # The Rust engine emits ONE class's SDF per call (H*W floats); the env stacks
    # two such calls into the 2-channel hazard_map. These tests exercise the
    # single-class engine, so reshape to a literal 1-channel (H, W) block.
    flat = map_rs.compute_hazard_map(list(deadly))
    arr = np.asarray(flat, dtype=np.float32).reshape(
        1, HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH
    )
    return arr


def test_hazard_map_shape_and_dtype_empty():
    m = MineThreatMap()
    arr = _compute(m, [])
    assert arr.shape == (1, HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH)
    assert arr.dtype == np.float32
    assert np.all(arr == 1.0)


def test_hazard_map_constants_match_rust():
    assert MineThreatMap.hazard_map_width() == HAZARD_MAP_WIDTH
    assert MineThreatMap.hazard_map_height() == HAZARD_MAP_HEIGHT
    assert KILL_RADIUS == 14.0
    assert SDF_SCALE == 24.0
    assert CELL == 6.0
    assert OFFSET == 24.0


def _cell_of(raw_px: float) -> int:
    """Hazard cell index containing a raw-sim-px coordinate."""
    return int((raw_px - OFFSET) / CELL)


def test_hazard_map_sign_convention_single_mine():
    m = MineThreatMap()
    mine_pos = (200.0, 100.0)
    arr = _compute(m, [mine_pos])[0]

    mx, my = mine_pos
    col = _cell_of(mx)
    row = _cell_of(my)
    assert arr[row, col] < 0.0

    far_col = _cell_of(mx + KILL_RADIUS + SDF_SCALE + 12.0)
    assert arr[row, far_col] == pytest.approx(1.0, abs=1e-5)

    boundary_col = _cell_of(mx + KILL_RADIUS)
    assert abs(arr[row, boundary_col]) <= (CELL / SDF_SCALE) + 1e-3


def test_hazard_map_toggle_flips_sign(mine_fixture_100):
    m = MineThreatMap()
    deadly = list(mine_fixture_100)
    arr_with = _compute(m, deadly)[0]

    mx, my = deadly[0]
    col = _cell_of(mx)
    row = _cell_of(my)
    assert arr_with[row, col] < 0.0

    # Toggle that mine to safe: SDF at its center should no longer be negative
    # (assuming no other deadly mine sits within one kill radius in the fixture).
    arr_without = _compute(m, deadly[1:])[0]
    assert arr_without[row, col] >= 0.0 - 1e-3


def test_hazard_map_playable_frame_co_registration():
    """The grid must cover exactly the playable region [24, 1032)x[24, 576).

    A mine centered on playable-frame tile (tx, ty) — raw px
    ((tx+1)*24 + 12, (ty+1)*24 + 12) — must put the SDF minimum within 1 cell
    of ((mine_x_raw - 24)/6, (mine_y_raw - 24)/6) = (tx*4 + 2, ty*4 + 2).
    Before the frame fix the grid was laid over [0, 1008)x[0, 552) raw px:
    everything was shifted by one tile and the last playable tile column/row
    had no hazard cells at all.
    """
    m = MineThreatMap()
    for tx, ty in [(0, 0), (5, 5), (41, 22), (20, 10)]:
        mine_x = (tx + 1) * 24.0 + 12.0
        mine_y = (ty + 1) * 24.0 + 12.0
        arr = _compute(m, [(mine_x, mine_y)])[0]
        row, col = np.unravel_index(np.argmin(arr), arr.shape)
        exp_col = (mine_x - OFFSET) / CELL
        exp_row = (mine_y - OFFSET) / CELL
        assert abs(col - exp_col) <= 1.0, (
            f"tile ({tx},{ty}): argmin col {col} vs expected {exp_col} "
            f"(= {tx * 4 + 2})"
        )
        assert abs(row - exp_row) <= 1.0, (
            f"tile ({tx},{ty}): argmin row {row} vs expected {exp_row} "
            f"(= {ty * 4 + 2})"
        )
        # The mine's own cell must read INSIDE the kill zone (negative SDF) —
        # in the old frame the last playable tile column/row (tx=41 / ty=22)
        # fell outside the grid entirely.
        assert arr[_cell_of(mine_y), _cell_of(mine_x)] < 0.0


def test_hazard_map_rust_python_parity(mine_fixture_100):
    m = MineThreatMap()
    arr = _compute(m, mine_fixture_100)[0]
    ref = _reference_hazard_map(mine_fixture_100)
    max_abs = float(np.max(np.abs(arr - ref)))
    assert max_abs <= 1e-5, f"Rust vs Python parity max |Δ| = {max_abs:.3e}"


def test_hazard_map_if_changed_returns_none_on_repeat(mine_fixture_100):
    m = MineThreatMap()
    deadly = list(mine_fixture_100)
    first = m.compute_hazard_map_if_changed(deadly)
    assert first is not None
    assert len(first) == HAZARD_MAP_HEIGHT * HAZARD_MAP_WIDTH

    unchanged = m.compute_hazard_map_if_changed(deadly)
    assert unchanged is None

    # Mutate one mine; expect a fresh buffer.
    mutated = list(deadly)
    mx, my = mutated[0]
    mutated[0] = (mx + 6.0, my)
    changed = m.compute_hazard_map_if_changed(mutated)
    assert changed is not None


def test_hazard_map_cache_stats_increment(mine_fixture_100):
    """NPP-48: cache hit/miss counters track hazard_map_if_changed behaviour."""
    m = MineThreatMap()
    m.reset_hazard_cache_stats()
    deadly = list(mine_fixture_100)
    m.compute_hazard_map_if_changed(deadly)
    h1, miss1, cells1 = m.get_hazard_cache_stats()
    assert miss1 >= 1
    assert cells1 == HAZARD_MAP_HEIGHT * HAZARD_MAP_WIDTH

    m.compute_hazard_map_if_changed(deadly)
    h2, miss2, _ = m.get_hazard_cache_stats()
    assert h2 > h1
    assert miss2 == miss1


def _shape(buf):
    return np.asarray(buf, dtype=np.float32).reshape(
        HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH
    )


def test_hazard_map_incremental_matches_full_on_single_toggle(mine_fixture_100):
    """NPP-48: incremental update is bit-parity with a full recompute.

    Single-mine toggle is the most common runtime event. The incremental
    path must produce the exact same SDF as a full recompute.
    """
    deadly = list(mine_fixture_100)
    m_inc = MineThreatMap()
    _ = m_inc.compute_hazard_map_if_changed(deadly)

    toggled = deadly[:-1]
    inc = _shape(m_inc.compute_hazard_map_if_changed(toggled))

    m_full = MineThreatMap()
    m_full.set_hazard_force_full(True)
    _ = m_full.compute_hazard_map_if_changed(deadly)
    full = _shape(m_full.compute_hazard_map_if_changed(toggled))

    np.testing.assert_allclose(inc, full, atol=0.0, rtol=0.0)


def test_hazard_map_incremental_matches_full_on_add_and_move(mine_fixture_100):
    """NPP-48: adding a mine and moving one must also match full recompute."""
    deadly = list(mine_fixture_100)
    m_inc = MineThreatMap()
    _ = m_inc.compute_hazard_map_if_changed(deadly)

    # Add a new mine in an empty region
    added = deadly + [(900.0, 500.0)]
    inc_add = _shape(m_inc.compute_hazard_map_if_changed(added))

    m_full = MineThreatMap()
    m_full.set_hazard_force_full(True)
    _ = m_full.compute_hazard_map_if_changed(deadly)
    full_add = _shape(m_full.compute_hazard_map_if_changed(added))

    np.testing.assert_allclose(inc_add, full_add, atol=0.0, rtol=0.0)

    # "Move" a mine = remove old position + add new position
    moved = list(added)
    moved[0] = (moved[0][0] + 18.0, moved[0][1] + 6.0)
    inc_moved = _shape(m_inc.compute_hazard_map_if_changed(moved))
    full_moved = _shape(m_full.compute_hazard_map_if_changed(moved))
    np.testing.assert_allclose(inc_moved, full_moved, atol=0.0, rtol=0.0)


def test_hazard_map_incremental_recomputes_fewer_cells_than_full(mine_fixture_100):
    """NPP-48: incremental update must touch < full grid after a small delta."""
    deadly = list(mine_fixture_100)
    m = MineThreatMap()
    m.reset_hazard_cache_stats()
    _ = m.compute_hazard_map_if_changed(deadly)
    _, _, cells_cold = m.get_hazard_cache_stats()
    assert cells_cold == HAZARD_MAP_HEIGHT * HAZARD_MAP_WIDTH

    # Toggle a single mine off
    _ = m.compute_hazard_map_if_changed(deadly[:-1])
    _, _, cells_inc = m.get_hazard_cache_stats()
    # At most ~pi * (38/6)^2 ~= 126 cells per toggled mine; allow generous slack.
    assert cells_inc < HAZARD_MAP_HEIGHT * HAZARD_MAP_WIDTH // 10, (
        f"incremental touched {cells_inc} cells (expected << full grid)"
    )
