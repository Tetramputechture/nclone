"""Tests for the geometry-morphing curriculum primitive (nclone/map_generation/mine_morph.py).

Invariants (see docs/GEOMETRY_MORPHING_CURRICULUM.md §4 Phase 0):
  * delta=0 is byte-identical to the input map (terminal rung == the real level).
  * carve widens the channel monotonically with delta (clearance non-decreasing).
  * morphed maps stay in-bounds and load faithfully in the native sim.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nclone.map_generation.map import Map
from nclone.map_generation.mine_morph import (
    morph_mines, route_centerline, min_clearance_along, iter_mines, spawn_world,
    MineMorphError, DEADLY_MINE_TYPE,
)

_REAL_MAP = Path(__file__).resolve().parents[1] / "nclone" / "test_maps" / "mines-extremely-difficult"


def _corridor_with_mines() -> list:
    """A wide floor corridor with a row of deadly mines along the ceiling, so the
    route threads beneath them and there is real clearance to widen."""
    m = Map()
    for x in range(2, 41):
        for y in (9, 10, 11, 12):
            m.set_tile(x, y, 0)      # open
    for x in range(2, 41):
        m.set_tile(x, 13, 1)         # floor
    m.set_ninja_spawn(4, 11)
    m.add_entity(3, 38, 11, 0, 0, switch_x=10, switch_y=11)  # exit + switch
    # deadly toggle mines (type 1) on the 6px grid, up near the ceiling
    for ux in range(16, 34, 2):
        m.add_entity(DEADLY_MINE_TYPE, ux, 16, 0, 0)
    return [int(v) for v in m.map_data()]


# --------------------------------------------------------------------------- #
# Fast synthetic-map tests
# --------------------------------------------------------------------------- #
def test_delta0_is_identity():
    md = _corridor_with_mines()
    assert morph_mines(md, 0) == list(md)
    assert morph_mines(md, -5) == list(md)


def test_centerline_spawn_to_exit():
    md = _corridor_with_mines()
    cl = route_centerline(md)
    assert len(cl) >= 2
    sx, sy = spawn_world(md)
    # leg A starts at the spawn
    assert abs(cl[0][0] - sx) < 12 and abs(cl[0][1] - sy) < 12


def test_carve_monotonic_and_removes_mines():
    md = _corridor_with_mines()
    cl = route_centerline(md)
    base = min_clearance_along(md, cl)
    n0 = sum(1 for _ in iter_mines(md, deadly_only=True))
    prev_clear, prev_mines = base, n0
    widened = False
    for d in (6, 12, 24, 48):
        mm = morph_mines(md, d, centerline=cl, mode="carve")
        clr = min_clearance_along(mm, cl)
        nm = sum(1 for _ in iter_mines(mm, deadly_only=True))
        assert clr >= prev_clear - 1e-6, f"clearance regressed at δ={d}"
        assert nm <= prev_mines, f"mine count grew at δ={d}"
        widened = widened or clr > base + 1e-6
        prev_clear, prev_mines = clr, nm
    assert widened, "carve never widened the channel"


def test_carve_delta0_keeps_all_mines():
    md = _corridor_with_mines()
    cl = route_centerline(md)
    assert (sum(1 for _ in iter_mines(morph_mines(md, 0, centerline=cl), deadly_only=True))
            == sum(1 for _ in iter_mines(md, deadly_only=True)))


def test_displace_preserves_mine_count_and_bounds():
    md = _corridor_with_mines()
    cl = route_centerline(md)
    n0 = sum(1 for _ in iter_mines(md, deadly_only=True))
    mm = morph_mines(md, 12, centerline=cl, mode="displace")
    mines = list(iter_mines(mm, deadly_only=True))
    assert len(mines) == n0                       # count preserved
    for _i, _t, wx, wy in mines:                  # in playfield interior
        assert 24 <= wx <= 1032 and 24 <= wy <= 576


def test_unknown_mode_raises():
    md = _corridor_with_mines()
    with pytest.raises(ValueError):
        morph_mines(md, 6, mode="bogus")


def test_no_route_raises():
    """A map without a switch/exit can't resolve a threading route -> the morph
    refuses (rather than silently producing a bogus centerline)."""
    m = Map()
    for x in range(2, 41):
        m.set_tile(x, 13, 1)
    m.set_ninja_spawn(4, 11)        # no exit/switch entity at all
    md = [int(v) for v in m.map_data()]
    with pytest.raises(MineMorphError):
        route_centerline(md)


def test_mineless_map_morph_is_noop():
    """A mine-free map has a valid centerline; carving it changes nothing."""
    m = Map()
    for x in range(2, 41):
        for y in (9, 10, 11, 12):
            m.set_tile(x, y, 0)
    for x in range(2, 41):
        m.set_tile(x, 13, 1)
    m.set_ninja_spawn(4, 11)
    m.add_entity(3, 38, 11, 0, 0, switch_x=10, switch_y=11)
    md = [int(v) for v in m.map_data()]
    assert morph_mines(md, 24) == md      # no deadly mines -> identity


# --------------------------------------------------------------------------- #
# Real-map invariants (the actual curriculum target)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _REAL_MAP.exists(), reason="mines-extremely-difficult map not present")
def test_real_map_delta0_byte_identical():
    raw = [int(b) for b in _REAL_MAP.read_bytes()]
    assert morph_mines(raw, 0) == raw


@pytest.mark.skipif(not _REAL_MAP.exists(), reason="mines-extremely-difficult map not present")
def test_real_map_carve_monotonic_and_native_load():
    import nclone_native
    raw = [int(b) for b in _REAL_MAP.read_bytes()]
    cl = route_centerline(raw)
    base = min_clearance_along(raw, cl)
    prev = base
    for d in (0, 12, 24, 48):
        mm = morph_mines(raw, d, centerline=cl, mode="carve")
        clr = min_clearance_along(mm, cl)
        assert clr >= prev - 1e-6
        assert clr >= base + d - 1.0, f"δ={d}: clearance {clr:.1f} < target {base+d:.1f}"
        prev = clr
        # loads faithfully in the native sim with switch+exit intact
        sim = nclone_native.Simulator.load(list(mm)); sim.fast_reset()
        types = {e.entity_type for e in sim.get_state().entities}
        assert 3 in types and 4 in types, f"δ={d}: lost switch/exit"
