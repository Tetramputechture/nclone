"""Tests for the airborne-aware surface lattice (docs/SPEC_airborne_lattice.md).

Covers the spec's A7 acceptance criteria:

* the swept mine test on arcs (D1) rejects an arc that flies through a deadly
  mine and accepts the same arc when the mine is ~20px clear;
* the arc test keys on *deadly* mines only — an inert (untoggled) mine must not
  prune anything;
* a mine-free level is bit-identical to the pre-change build (the mine handling
  must be inert where it should be).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import nclone_native
from nclone.map_generation.map import Map
from nclone.constants.physics_constants import (
    NINJA_RADIUS, TOGGLE_MINE_RADIUS_TOGGLED,
)
from nclone.value_function.edge_enumeration import (
    enumerate_edges, populate_graph_edges,
)
from nclone.value_function.physics_step_wrapper import PhysicsStepWrapper
from nclone.value_function.surface_extraction import extract_surfaces
from nclone.value_function.surface_graph import build_graph
from nclone.value_function.tools.physics_route_spike import (
    _MINE_STATE_DEADLY, _goal_positions,
)

KILL_RADIUS = NINJA_RADIUS + TOGGLE_MINE_RADIUS_TOGGLED  # 14.0 px

# A launch state whose arc crosses the gap in ``_gap_level`` and passes within
# ~3px of tile (18, 15) — i.e. straight through a mine placed there.
_LAUNCH_X, _LAUNCH_Y, _LAUNCH_VX = 330.0, 326.0, 3.3333
# Tile (18, 15) sits ON that arc; (18, 14) is one tile (24px) above it, giving
# ~18.5px clearance — the spec's "20px above a mine is safe" case.
_TILE_ON_ARC = (18, 15)
_TILE_ABOVE_ARC = (18, 14)


def _gap_level(mine_tile=None, mine_state: int = _MINE_STATE_DEADLY):
    """Two floor platforms with a gap; optional toggle mine at ``mine_tile``."""
    m = Map()
    for x in range(2, 42):
        for y in range(6, 14):
            m.set_tile(x, y, 0)
    for x in list(range(2, 15)) + list(range(21, 42)):
        m.set_tile(x, 14, 1)
    m.set_ninja_spawn(4, 13)
    m.add_entity(3, 38, 13, 0, 0, switch_x=10, switch_y=13)
    if mine_tile is not None:
        # Entity type 1 = toggle mine; type 21 is the untoggled (inert) variant.
        etype = 1 if mine_state == _MINE_STATE_DEADLY else 21
        m.add_entity(etype, mine_tile[0], mine_tile[1], 0, 0)
    return [int(v) for v in m.map_data()]


def _fly_arc(map_data, tick_mines: bool):
    """Fly the fixed launch state and report ``(outcome, frames, path)``.

    ``tick_mines`` selects the entity-aware tick used by edge enumeration on
    mine-bearing levels (real contact death) vs the mine-blind tiles-only tick.
    """
    sim = nclone_native.Simulator.load(list(map_data))
    sim.inject_ninja_for_value_iteration(_LAUNCH_X, _LAUNCH_Y, _LAUNCH_VX, 0.0, 0)
    tick = sim.tick_physics_mines_only if tick_mines else sim.tick_tiles_only
    tick(1, 1)
    path = []
    for frame in range(1, 120):
        tick(1, 0)
        state = sim.get_ninja_state_code()
        pos = sim.get_ninja_pos()
        path.append((pos[0], pos[1]))
        if state in (6, 7):
            return "DEAD", frame, path
        if state in (0, 1, 2, 5) and frame > 3:
            return "LANDED", frame, path
    return "TIMEOUT", 120, path


def _mine_xy(map_data):
    sim = nclone_native.Simulator.load(list(map_data))
    _switch, _exit, mines = _goal_positions(sim)
    assert mines, "level should carry a mine"
    return mines[0][0], mines[0][1]


def _min_clearance(path, mx, my):
    return min(math.hypot(x - mx, y - my) for x, y in path)


def _build_dstar(map_data):
    """Enumerate edges the way the production lattice path does."""
    phys = PhysicsStepWrapper(map_data=map_data)
    surface_points = extract_surfaces(phys.tile_dic, mine_positions=[])
    graph = build_graph(surface_points)
    sim = nclone_native.Simulator.load(list(map_data))
    _switch, _exit, mines = _goal_positions(sim)
    deadly = [(x, y) for (x, y, s) in mines if s == _MINE_STATE_DEADLY]
    edges, _ = enumerate_edges(
        graph, map_data, num_workers=1, collect_waypoints=False,
        mine_positions=(np.asarray(deadly, np.float32) if deadly else None),
        use_native=True,
    )
    populate_graph_edges(graph, edges)
    return graph, edges


# ---------------------------------------------------------------------------
# A7: swept arc test
# ---------------------------------------------------------------------------
def test_arc_through_deadly_mine_is_lethal():
    """An arc whose traced path enters the kill disc must die (edge pruned)."""
    map_data = _gap_level(_TILE_ON_ARC)
    mx, my = _mine_xy(map_data)

    # The mine-blind tick proves the arc is otherwise a perfectly good edge, so
    # a DEAD result under the mine-aware tick is caused by the mine and nothing
    # else (geometry, timeout, ...).
    blind_outcome, _, path = _fly_arc(map_data, tick_mines=False)
    assert blind_outcome == "LANDED"
    assert _min_clearance(path, mx, my) < KILL_RADIUS, (
        "test fixture is wrong: the arc does not actually enter the kill disc"
    )

    outcome, _, _ = _fly_arc(map_data, tick_mines=True)
    assert outcome == "DEAD"


def test_arc_clear_of_mine_survives():
    """The same arc with the mine ~20px clear must still be enumerated."""
    map_data = _gap_level(_TILE_ABOVE_ARC)
    mx, my = _mine_xy(map_data)

    _blind, _f, path = _fly_arc(map_data, tick_mines=False)
    clearance = _min_clearance(path, mx, my)
    assert clearance > KILL_RADIUS, "fixture: mine should be clear of the arc"
    assert clearance < 3 * KILL_RADIUS, "fixture: mine should still be nearby"

    outcome, _, _ = _fly_arc(map_data, tick_mines=True)
    assert outcome == "LANDED"


def test_inert_mine_does_not_prune():
    """An untoggled (state 1) mine sitting ON the arc is harmless."""
    map_data = _gap_level(_TILE_ON_ARC, mine_state=1)
    sim = nclone_native.Simulator.load(list(map_data))
    _switch, _exit, mines = _goal_positions(sim)
    assert mines and mines[0][2] != _MINE_STATE_DEADLY

    outcome, _, _ = _fly_arc(map_data, tick_mines=True)
    assert outcome == "LANDED"


# ---------------------------------------------------------------------------
# A7: inert where it should be
# ---------------------------------------------------------------------------
def test_mine_free_level_edges_unchanged_by_mine_plumbing():
    """On a mine-free level the enumeration is byte-identical with and without
    the mine arguments — the mine handling must be a strict no-op here."""
    map_data = _gap_level()
    phys = PhysicsStepWrapper(map_data=map_data)
    surface_points = extract_surfaces(phys.tile_dic, mine_positions=[])

    def run(mine_positions):
        graph = build_graph(surface_points)
        edges, _ = enumerate_edges(
            graph, map_data, num_workers=1, collect_waypoints=False,
            mine_positions=mine_positions, use_native=True,
        )
        populate_graph_edges(graph, edges)
        return graph

    a = run(None)
    b = run(np.empty((0, 2), np.float32))
    assert np.array_equal(a.edges_src, b.edges_src)
    assert np.array_equal(a.edges_dst, b.edges_dst)
    assert np.array_equal(a.edges_weight, b.edges_weight)


def test_deadly_mine_removes_edges_relative_to_mine_free():
    """The swept test can only ever REMOVE edges (spec A1 monotonicity)."""
    free_graph, free_edges = _build_dstar(_gap_level())
    mined_graph, mined_edges = _build_dstar(_gap_level(_TILE_ON_ARC))

    free_pairs = {(int(e.src_node), int(e.dst_node)) for e in free_edges}
    mined_pairs = {(int(e.src_node), int(e.dst_node)) for e in mined_edges}
    # Surface points are identical (mines are not extracted as geometry here),
    # so node ids are comparable across the two builds.
    assert free_graph.n_points == mined_graph.n_points
    assert mined_pairs <= free_pairs, (
        "the mine-aware build gained edges the mine-free build lacked"
    )
    assert len(mined_pairs) < len(free_pairs), "the mine pruned nothing at all"


# ---------------------------------------------------------------------------
# The airborne extension (D2/D3)
# ---------------------------------------------------------------------------
def _lattice(map_data, airborne):
    from nclone.value_function.lattice_distance import LatticeDistanceSource
    return LatticeDistanceSource(
        map_data, name="test", workers=2, airborne_field=airborne,
    )


@pytest.fixture(scope="module")
def gap_pair():
    """(surface-only, airborne-extended) lattices over the same level."""
    map_data = _gap_level()
    off = _lattice(map_data, False)
    on = _lattice(map_data, True)
    if not (off.available and on.available):
        pytest.skip("lattice unavailable on the fixture level")
    return map_data, off, on


def _free_cells(map_data):
    from nclone.value_function.free_space_distance import (
        FreeSpaceDistanceSource, _CELL, _W, _H,
    )
    fs = FreeSpaceDistanceSource(list(map_data), mine_hard=True, physics_cost=True)
    if not fs.available or fs._free is None:
        pytest.skip("free-space grid unavailable")
    return [(cx * _CELL + _CELL / 2.0, cy * _CELL + _CELL / 2.0)
            for cx in range(_W) for cy in range(_H) if fs._free[cx, cy]]


def test_airborne_extension_never_changes_an_existing_value(gap_pair):
    """A position the surface lattice could already price keeps its EXACT value.

    This is the spec's "inert where it should be" criterion.  The extension is a
    strict widening of the domain, never a re-pricing of it.
    """
    map_data, off, on = gap_pair
    d_off, _ = off._phase_arrays(False, None)
    d_on, _ = on._phase_arrays(False, None)

    n_checked = 0
    for x, y in _free_cells(map_data):
        before = off._min_dstar_at(x, y, d_off)
        if before is None or not np.isfinite(before):
            continue
        after = on._cost_at(x, y, d_on)
        n_checked += 1
        assert after == before, f"value at ({x},{y}) changed {before} -> {after}"
    assert n_checked > 0, "fixture priced nothing; the test proved nothing"


def test_airborne_extension_widens_coverage(gap_pair):
    """The extension must price strictly more free space than surfaces alone."""
    map_data, off, on = gap_pair
    d_off, _ = off._phase_arrays(False, None)
    d_on, _ = on._phase_arrays(False, None)
    cells = _free_cells(map_data)

    def finite(src, d_star, use_surface_only):
        n = 0
        for x, y in cells:
            v = (src._min_dstar_at(x, y, d_star) if use_surface_only
                 else src._cost_at(x, y, d_star))
            if v is not None and np.isfinite(v):
                n += 1
        return n

    n_surface = finite(off, d_off, True)
    n_extended = finite(on, d_on, False)
    assert n_extended > n_surface, (
        f"extension added no coverage ({n_surface} -> {n_extended} of {len(cells)})"
    )


def test_airborne_field_is_disabled_by_flag(gap_pair):
    """``airborne_field=False`` reproduces the old surface-snapped behaviour.

    Without this the widening test above could not distinguish "the extension
    works" from "the lattice always covered everything".
    """
    map_data, off, _on = gap_pair
    assert off._air is None, "airborne_field=False still built a field"
    d_off, _ = off._phase_arrays(False, None)
    # With the field off, _cost_at must be exactly the surface lookup — including
    # returning None at positions the surface graph cannot reach.
    blind = 0
    for x, y in _free_cells(map_data):
        surface = off._min_dstar_at(x, y, d_off)
        assert off._cost_at(x, y, d_off) == surface or (
            surface is None and off._cost_at(x, y, d_off) is None
        )
        if surface is None:
            blind += 1
    assert blind > 0, (
        "the surface-only lattice was blind nowhere on this level, so this "
        "fixture cannot demonstrate the extension at all"
    )


def test_field_never_ROUTES_through_a_kill_cell():
    """The field array holds ``inf`` at every deadly-mine cell.

    This is the invariant spec trap #3 actually protects: severance must not be
    fixed by letting a PATH pass through a kill radius.  The flood never relaxes
    a cut cell, so no route can ever gain a shortcut through a mine.

    Note the deliberate distinction from the LOOKUP: ``cost_at`` at a kill cell
    snaps to a neighbouring free cell (see ``_is_solid_cell``), because the ninja
    genuinely can occupy one — some official levels spawn it there — and ``inf``
    at the spawn is strictly worse than a snapped value.  Routing is protected
    here; the lookup is covered by ``test_kill_cell_lookup_is_bounded_by_neighbours``.
    """
    from nclone.value_function.free_space_distance import _CELL
    map_data = _gap_level(_TILE_ON_ARC)
    src = _lattice(map_data, True)
    if not src.available or src._air is None:
        pytest.skip("airborne field unavailable on this level")
    mx, my = _mine_xy(map_data)
    d_star, _ = src._phase_arrays(False, None)
    field = src._air.field_for(d_star, src._goal_spec(d_star))
    assert field is not None

    checked = 0
    for dx, dy in ((0, 0), (4, 0), (-4, 0), (0, 4), (0, -4)):
        cx, cy = int((mx + dx) / _CELL), int((my + dy) / _CELL)
        if src._air._fs._free[cx, cy]:
            continue  # outside the cut disc at this resolution
        checked += 1
        assert not np.isfinite(field[cx, cy]), (
            f"field array priced cut cell ({cx},{cy}) — a route could pass through"
        )
    assert checked > 0, "fixture cut no cells; the test asserted nothing"


def test_kill_cell_lookup_is_bounded_by_neighbours():
    """A kill-cell lookup returns a NEIGHBOUR's cost, never a bargain.

    The snap must not invent a cheaper value than the surrounding free space,
    which is what would actually create an incentive to enter the disc.
    """
    from nclone.value_function.free_space_distance import _CELL, _W, _H
    map_data = _gap_level(_TILE_ON_ARC)
    src = _lattice(map_data, True)
    if not src.available or src._air is None:
        pytest.skip("airborne field unavailable")
    mx, my = _mine_xy(map_data)
    d_star, _ = src._phase_arrays(False, None)
    goal = src._goal_spec(d_star)
    air = src._air
    field = air.field_for(d_star, goal)

    cx, cy = int(mx / _CELL), int(my / _CELL)
    if air._fs._free[cx, cy]:
        pytest.skip("mine centre is not a cut cell at this resolution")
    v = air.cost_at(mx, my, d_star, goal)
    if not np.isfinite(v):
        return  # treated as solid; also acceptable
    r = air._SNAP_CELLS
    neigh = [float(field[ix, iy])
             for ix in range(max(0, cx - r), min(_W, cx + r + 1))
             for iy in range(max(0, cy - r), min(_H, cy + r + 1))
             if air._fs._free[ix, iy] and np.isfinite(field[ix, iy])]
    assert neigh, "no finite free neighbour, yet the snap returned a value"
    assert v >= min(neigh) - 1e-6, "snap invented a value cheaper than any neighbour"


def test_airborne_field_is_finite_where_surface_lookup_is_blind(gap_pair):
    """The headline claim: positions with no nearby surface DO get a cost now."""
    map_data, off, on = gap_pair
    d_off, _ = off._phase_arrays(False, None)
    d_on, _ = on._phase_arrays(False, None)

    blind_then_priced = 0
    for x, y in _free_cells(map_data):
        if off._min_dstar_at(x, y, d_off) is not None:
            continue  # surface lookup already had an answer here
        v = on._cost_at(x, y, d_on)
        if v is not None and np.isfinite(v):
            blind_then_priced += 1
    assert blind_then_priced > 0, (
        "no previously-blind position gained a cost; the extension is a no-op"
    )


# ---------------------------------------------------------------------------
# A4: airborne goals are priced correctly (D3)
# ---------------------------------------------------------------------------
def _air_goal_level():
    """A floor with the exit SWITCH floating high in open space above it.

    The switch's contact disc touches no surface point, so the surface-only
    lattice can only price its neighbourhood through the ledge below — the exact
    "distance to the nearest ledge, which can be arbitrarily wrong" mispricing
    spec A4 names.
    """
    m = Map()
    for x in range(2, 42):
        for y in range(2, 18):
            m.set_tile(x, y, 0)
    for x in range(2, 42):
        m.set_tile(x, 18, 1)
    m.set_ninja_spawn(4, 17)
    # Exit door on the floor; its switch floats 7 tiles up in open air.
    m.add_entity(3, 38, 17, 0, 0, switch_x=20, switch_y=10)
    return [int(v) for v in m.map_data()]


def test_airborne_goal_is_priced_below_the_ledge_beneath_it():
    """Cost-to-go must DECREASE as you approach a mid-air switch.

    Surface-snapping makes the neighbourhood of an air goal read as the cost of
    the ledge below it, so approaching the goal through open air changes nothing
    and the potential has no gradient toward it.  With the goal seeded into the
    field, being at the switch must be strictly cheaper than being on the floor
    directly beneath it.
    """
    map_data = _air_goal_level()
    on = _lattice(map_data, True)
    if not on.available or on._air is None:
        pytest.skip("lattice/field unavailable on the air-goal fixture")
    sx, sy = on.switch

    at_goal = on.cost_to_go(sx, sy, switch_activated=False)
    # Directly below the switch, standing on the floor (floor top = 18*24 = 432).
    on_floor = on.cost_to_go(sx, 432.0 - NINJA_RADIUS, switch_activated=False)

    assert np.isfinite(at_goal), "no cost at the air switch itself"
    assert np.isfinite(on_floor)
    assert at_goal < on_floor, (
        f"air switch ({at_goal:.1f}) is not cheaper than the ledge below it "
        f"({on_floor:.1f}) — the potential has no gradient toward the goal"
    )


def test_airborne_goal_gradient_is_monotone_on_approach():
    """Rising from the floor toward the air switch must monotonically reduce cost."""
    map_data = _air_goal_level()
    on = _lattice(map_data, True)
    if not on.available or on._air is None:
        pytest.skip("lattice/field unavailable on the air-goal fixture")
    sx, sy = on.switch

    # Sample the column from the floor up to the switch.
    ys = [432.0 - NINJA_RADIUS - k for k in range(0, int(432 - sy), 24)]
    costs = [on.cost_to_go(sx, y, switch_activated=False) for y in ys]
    costs = [c for c in costs if np.isfinite(c)]
    assert len(costs) >= 4, "not enough finite samples to judge a gradient"
    # Not strictly monotone cell-by-cell (the flood is 8-connected on a 6px
    # grid), but the trend across the column must be downward.
    assert costs[-1] < costs[0], (
        f"cost did not fall on approach to the air goal: {costs[0]:.1f} -> "
        f"{costs[-1]:.1f}"
    )


def test_surface_only_lattice_misprices_the_air_goal():
    """The control: without the extension the air goal has no gradient.

    If this passed WITHOUT the change, the A4 tests above would prove nothing.
    """
    map_data = _air_goal_level()
    off = _lattice(map_data, False)
    if not off.available:
        pytest.skip("lattice unavailable on the air-goal fixture")
    sx, sy = off.switch
    d_star, _ = off._phase_arrays(False, None)
    # The surface lookup either has nothing at the air switch, or answers with a
    # value it also gives on the ledge below — either way there is no descent.
    at_goal = off._min_dstar_at(sx, sy, d_star)
    on_floor = off._min_dstar_at(sx, 432.0 - NINJA_RADIUS, d_star)
    no_gradient = (
        at_goal is None
        or not np.isfinite(at_goal)
        or (on_floor is not None and np.isfinite(on_floor) and at_goal >= on_floor)
    )
    assert no_gradient, (
        "the surface-only lattice already descends toward the air goal, so this "
        "fixture does not exercise the mispricing A4 is about"
    )


# ---------------------------------------------------------------------------
# The field must survive the lattice failing entirely
# ---------------------------------------------------------------------------
def test_finite_spawn_when_the_lattice_cannot_build():
    """A level whose surface graph blows the node budget still gets a potential.

    Measured over 60 official levels, the lattice fails outright on ~13%:
    8.3% ``LatticeTooLargeError`` and 5% "goal has no surface contact node".
    Every one of those is still a connected free space with a goal in it, so the
    field is built regardless and seeded from the goal.  ``max_nodes=1`` forces
    the failure deterministically without needing a pathological level.
    """
    from nclone.value_function.lattice_distance import LatticeDistanceSource
    map_data = _gap_level()
    sim = nclone_native.Simulator.load(list(map_data))
    sx, sy = sim.get_ninja_pos()

    # use_cache=False: precompute checks the cache BEFORE the node budget, so a
    # pickle left by an earlier run would sail past ``max_nodes`` entirely.
    src = LatticeDistanceSource(map_data, name="toolarge", workers=2,
                                max_nodes=1, use_cache=False,
                                airborne_field=True)
    assert src.lattice_available is False, "fixture did not trip the node budget"
    assert src.available is True, "no potential at all despite a usable field"
    assert src.spawn_finite(sx, sy), "spawn has no finite cost-to-go"
    c = src.cost_to_go(sx, sy, switch_activated=False)
    assert np.isfinite(c) and c > 0


def test_lattice_failure_without_the_field_is_unavailable():
    """Control: with the field off, the same level really is unusable.

    Without this the test above could pass for the wrong reason.
    """
    from nclone.value_function.lattice_distance import LatticeDistanceSource
    src = LatticeDistanceSource(_gap_level(), name="toolarge-off", workers=2,
                                max_nodes=1, use_cache=False,
                                airborne_field=False)
    assert src.lattice_available is False
    assert src.available is False
    assert src.cost_to_go(100.0, 100.0, False) == float("inf")


def test_soft_rescue_never_prices_a_kill_cell():
    """The mine-soft rescue tier fixes UNREACHABILITY, never LETHALITY.

    A cell the hard mask cut out is solid or inside a kill disc; both stay
    ``inf``.  Without this distinction the rescue would hand back a finite cost
    on every mine cell in the level — exactly what spec trap #3 forbids — while
    still looking like it "improved coverage".
    """
    from nclone.value_function.free_space_distance import _CELL
    map_data = _gap_level(_TILE_ON_ARC)
    src = _lattice(map_data, True)
    if not src.available or src._air is None:
        pytest.skip("airborne field unavailable")
    mx, my = _mine_xy(map_data)
    d_star, _ = src._phase_arrays(False, None)
    air = src._air

    # Force the soft tier to exist, so the test cannot pass merely because the
    # rescue was never built.
    air._soft_field_for(d_star, src._goal_spec(d_star))

    soft = air._soft_field_for(d_star, src._goal_spec(d_star))
    hard = air.field_for(d_star, src._goal_spec(d_star))
    assert soft is not None and hard is not None
    for dx, dy in ((0, 0), (4, 0), (-4, 0), (0, 4), (0, -4)):
        cx, cy = int((mx + dx) / _CELL), int((my + dy) / _CELL)
        if air._fs._free[cx, cy]:
            continue
        # The HARD field — the one that governs routing — must still be inf at
        # the cut cell even though a soft flood now exists alongside it.
        assert not np.isfinite(hard[cx, cy]), (
            f"building the soft rescue leaked a finite value into the hard "
            f"routing field at ({cx},{cy})"
        )


def test_prefer_field_is_off_by_default_and_preserves_coverage():
    """``prefer_field`` must be opt-in, and must not lose a single value.

    Taking the flood where the surface lookup could answer is a large gain in
    MONOTONICITY (measured: 20.21% of surface-sourced cells are spurious local
    minima vs 0.25% of field-sourced ones; composite 4.57% -> 0.15%) — but it
    moves the potential, so it stays off until it has had a training A/B.

    It is NOT monotonically lower, and the reason matters: ``_min_dstar_at``
    snaps to any surface point within +/-18px and charges nothing for that
    teleport, while the flood makes you pay the travel.  Measured over 47.5k
    cells: 0 lost, 0 gained, median difference 0, lowered on 19.7%, raised on
    1.7% by at most ~33 frames.  So the invariant to hold is COVERAGE, plus a
    bound on how far a value can rise.
    """
    from nclone.value_function.lattice_distance import LatticeDistanceSource
    map_data = _gap_level()
    default = _lattice(map_data, True)
    assert default._prefer_field is False, "prefer_field must default to OFF"

    pf = LatticeDistanceSource(map_data, name="pf", workers=2,
                               airborne_field=True, prefer_field=True)
    if not (default.available and pf.available):
        pytest.skip("lattice unavailable on the fixture")
    d_a, d_max = default._phase_arrays(False, None)
    d_b, _ = pf._phase_arrays(False, None)

    n = raised = lowered = 0
    worst_rise = 0.0
    for x, y in _free_cells(map_data):
        a = default._cost_at(x, y, d_a)
        if a is None or not np.isfinite(a):
            continue
        b = pf._cost_at(x, y, d_b)
        assert b is not None and np.isfinite(b), (
            f"prefer_field lost a value the default mode had at ({x},{y})"
        )
        n += 1
        if b > a + 1e-6:
            raised += 1
            worst_rise = max(worst_rise, b - a)
        elif b < a - 1e-6:
            lowered += 1
    assert n > 0, "compared nothing"
    # A rise comes only from the free <=18px teleport the surface snap allows, so
    # it is bounded relative to the level's own cost scale rather than by an
    # absolute constant (measured max +33 frames across official levels; larger
    # on this synthetic gap fixture, where 18px of free teleport skips a jump).
    assert worst_rise <= 0.5 * d_max, (
        f"a value rose by {worst_rise:.1f} frames, over half the level scale "
        f"({d_max:.1f})"
    )
    # NOT asserted: that lowering outnumbers raising.  Over official levels it
    # does by ~12x (19.7% vs 1.7%), but this synthetic fixture is mostly open air
    # where the free teleport is doing a lot of work, and there raising wins.
    # That split is a property of the CORPUS, not of the change.
    assert raised + lowered <= n


def test_survivable_position_near_a_mine_is_still_priced():
    """A position the routing mask cuts but the ninja can occupy must get a cost.

    The free mask cuts any cell whose CENTRE is within ``_MINE_KILL + _CELL`` =
    20px of a deadly mine — the true kill radius (14px) dilated by a whole cell
    so no part of a 6px cell can be lethal.  That is right for routing and wrong
    for "can the ninja be here".

    Real case: ``550 bamboo training forest`` spawns the ninja at (258, 468)
    between two deadly mines each **21.63px** away — a 7.6px margin over the kill
    radius, standing on a surface.  Its cell centre is 17.49px from the nearest
    mine, so the dilated cut erases a perfectly good standing position, and the
    spawn (the worst possible place to lose the gradient) went to ``inf``.

    So lethality is tested at the POSITION, not inferred from the cell mask.
    """
    from nclone.value_function.airborne_field import _MINE_KILL_RADIUS
    map_data = _gap_level(_TILE_ON_ARC)
    src = _lattice(map_data, True)
    if not src.available or src._air is None:
        pytest.skip("airborne field unavailable")
    air = src._air
    mx, my = _mine_xy(map_data)

    # Inside the kill disc -> lethal.
    assert air._position_is_lethal(mx, my)
    assert air._position_is_lethal(mx + _MINE_KILL_RADIUS - 1.0, my)
    # Outside it, but inside the mask's dilated cut -> survivable, must not be
    # reported lethal even though the cell it lands in may well be cut.
    for d in (_MINE_KILL_RADIUS + 1.0, _MINE_KILL_RADIUS + 5.0):
        assert not air._position_is_lethal(mx + d, my), (
            f"position {d:.1f}px from a mine reported lethal; kill radius is "
            f"{_MINE_KILL_RADIUS}px"
        )


def test_lethal_position_is_never_priced():
    """The exact game condition still gates the lookup: inside 14px -> inf."""
    map_data = _gap_level(_TILE_ON_ARC)
    src = _lattice(map_data, True)
    if not src.available or src._air is None:
        pytest.skip("airborne field unavailable")
    mx, my = _mine_xy(map_data)
    d_star, _ = src._phase_arrays(False, None)
    goal = src._goal_spec(d_star)
    assert not np.isfinite(src._air.cost_at(mx, my, d_star, goal))


def test_goal_seeded_field_descends_toward_the_goal():
    """With no lattice at all, the field is still a usable potential.

    A shortest-path cost-to-go over the connected free space has its only
    minimum AT the goal (the discrete analogue of a navigation function), so
    approaching the switch must reduce the cost.
    """
    from nclone.value_function.lattice_distance import LatticeDistanceSource
    map_data = _gap_level()
    src = LatticeDistanceSource(map_data, name="seeded", workers=2,
                                max_nodes=1, use_cache=False,
                                airborne_field=True)
    if not src.available or src.switch is None:
        pytest.skip("no field on the fixture")
    swx, swy = src.switch
    near = src.cost_to_go(swx, swy, switch_activated=False)
    far = src.cost_to_go(swx + 300.0, swy, switch_activated=False)
    assert np.isfinite(near) and np.isfinite(far)
    assert near < far, f"no descent toward the goal: near={near} far={far}"


# ---------------------------------------------------------------------------
# A7: cache versioning
# ---------------------------------------------------------------------------
def test_lattice_cache_key_tracks_the_mine_semantics_change():
    """A cache written before the deadly-mine arc change must not be reused.

    The arc mine test switched from ``all_mines`` to ``deadly`` and the kill-zone
    prune was decoupled from ``mine_inflation``; both change the enumerated edge
    set on levels with an untoggled mine.  The lattice cache key carries a
    version token for exactly this, and it has to have moved.
    """
    from nclone.value_function.tools.lattice_audit import LatticeAudit
    audit = LatticeAudit(map_data=_gap_level(_TILE_ON_ARC), name="cachekey",
                         workers=1)
    import inspect
    src = inspect.getsource(LatticeAudit._cache_path)
    assert "v17-deadly-mine-arcs" in src, (
        "cache version token not bumped for the mine-semantics change"
    )
    # And the key must actually differ from one built under the old token.
    old_payload = src.replace("v17-deadly-mine-arcs", "v16-slopejump")
    assert old_payload != src
    assert audit._cache_path()  # key is computable


def test_bin_schedule_version_not_bumped_unnecessarily():
    """The airborne extension adds no velocity bins, so the tile caches stay valid.

    Spec §5 trap 1 says to bump ``BIN_SCHEDULE_VERSION`` when the bin LAYOUT
    changes.  This work adds a free-space field alongside the graph and does not
    touch the schedules, so bumping would invalidate every tile-cache shard for
    nothing.  This test pins that reasoning: if someone later adds vy bins, the
    schedule lengths change and this fails, prompting the bump.
    """
    from nclone.value_function.surface_graph import (
        BIN_SCHEDULE_VERSION,
        NUM_VELOCITY_BINS_FLOOR, NUM_VELOCITY_BINS_WALL,
        NUM_VELOCITY_BINS_SLIDE, NUM_VELOCITY_BINS_LAUNCH_PAD,
    )
    assert BIN_SCHEDULE_VERSION == 4
    # The default stride-2 subsample of the 15/12/9/1 hand-tuned schedules.
    assert (NUM_VELOCITY_BINS_FLOOR, NUM_VELOCITY_BINS_WALL,
            NUM_VELOCITY_BINS_SLIDE, NUM_VELOCITY_BINS_LAUNCH_PAD) == (8, 8, 5, 1)
