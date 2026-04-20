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
    HAZARD_MAP_CHANNELS,
    HAZARD_MAP_HEIGHT,
    HAZARD_MAP_WIDTH,
)


KILL_RADIUS = float(MineThreatMap.hazard_kill_radius())
SDF_SCALE = float(MineThreatMap.hazard_sdf_scale())
CELL = float(MineThreatMap.hazard_cell_size())


def _reference_hazard_map(deadly_positions):
    """Python reference for the 1-channel deadly SDF used by `hazard_map`.

    Output shape `(H, W)` matches the Rust row-major flat buffer before the
    `(1, H, W)` reshape applied by the Python wrapper.
    """
    h, w = HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH
    out = np.ones((h, w), dtype=np.float32)
    if not deadly_positions:
        return out
    ys = (np.arange(h, dtype=np.float64) + 0.5) * CELL
    xs = (np.arange(w, dtype=np.float64) + 0.5) * CELL
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
    flat = map_rs.compute_hazard_map(list(deadly))
    arr = np.asarray(flat, dtype=np.float32).reshape(
        HAZARD_MAP_CHANNELS, HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH
    )
    return arr


def test_hazard_map_shape_and_dtype_empty():
    m = MineThreatMap()
    arr = _compute(m, [])
    assert arr.shape == (HAZARD_MAP_CHANNELS, HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH)
    assert arr.dtype == np.float32
    assert np.all(arr == 1.0)


def test_hazard_map_constants_match_rust():
    assert MineThreatMap.hazard_map_width() == HAZARD_MAP_WIDTH
    assert MineThreatMap.hazard_map_height() == HAZARD_MAP_HEIGHT
    assert KILL_RADIUS == 14.0
    assert SDF_SCALE == 24.0
    assert CELL == 6.0


def test_hazard_map_sign_convention_single_mine():
    m = MineThreatMap()
    mine_pos = (200.0, 100.0)
    arr = _compute(m, [mine_pos])[0]

    mx, my = mine_pos
    col = int(mx / CELL)
    row = int(my / CELL)
    assert arr[row, col] < 0.0

    far_col = int((mx + KILL_RADIUS + SDF_SCALE + 12.0) / CELL)
    assert arr[row, far_col] == pytest.approx(1.0, abs=1e-5)

    boundary_col = int((mx + KILL_RADIUS) / CELL)
    assert abs(arr[row, boundary_col]) <= (CELL / SDF_SCALE) + 1e-3


def test_hazard_map_toggle_flips_sign(mine_fixture_100):
    m = MineThreatMap()
    deadly = list(mine_fixture_100)
    arr_with = _compute(m, deadly)[0]

    mx, my = deadly[0]
    col = int(mx / CELL)
    row = int(my / CELL)
    assert arr_with[row, col] < 0.0

    # Toggle that mine to safe: SDF at its center should no longer be negative
    # (assuming no other deadly mine sits within one kill radius in the fixture).
    arr_without = _compute(m, deadly[1:])[0]
    assert arr_without[row, col] >= 0.0 - 1e-3


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
