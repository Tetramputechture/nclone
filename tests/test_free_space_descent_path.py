"""Tests for ``FreeSpaceDistanceSource.descent_path`` (the overlay's route).

The route is what the "Physics cost-to-go" dev overlay draws, and it is also the
clearest executable statement of what the physics cost model believes: it must
descend the same field the PBRS potential reads, using only moves the flood was
willing to charge for.
"""

import os

import numpy as np
import pytest

from nclone.value_function.free_space_distance import FreeSpaceDistanceSource

MAPS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "nclone", "test_maps")


def _load(name):
    from nclone.value_function.tools.physics_route_spike import _load_map_data

    return list(_load_map_data(os.path.join(MAPS, name)))


@pytest.fixture(scope="module")
def stairs():
    return _load("open-platform-stairs")


@pytest.mark.parametrize("physics", [False, True])
def test_descent_path_reaches_goal_and_is_monotone(stairs, physics):
    src = FreeSpaceDistanceSource(stairs, physics_cost=physics)
    assert src.available
    sx, sy = src.switch
    # Start far from the goal so the route is non-trivial.
    route = src.descent_path(sx + 300.0, sy + 100.0, False)
    assert route is not None
    assert route["reached_goal"], "steepest descent must terminate at the goal"
    assert not route["truncated"]
    assert len(route["points"]) == len(route["moves"]) + 1

    # Every step must strictly reduce cost: that is what makes this a descent,
    # and what makes the PBRS shaping it mirrors non-circular.
    assert all(c > 0.0 for c in route["costs"])
    # The per-class budget must account for the whole drop from start to goal.
    assert sum(route["by_class"].values()) == pytest.approx(sum(route["costs"]))
    assert sum(route["costs"]) == pytest.approx(route["total"], rel=1e-9)


def test_geometric_route_does_not_claim_physics_move_classes(stairs):
    """A geometric field distinguishes no physics, so it must not label moves
    'jump'/'wall'/'air' -- those names would imply a gating it never applied."""
    src = FreeSpaceDistanceSource(stairs, physics_cost=False)
    sx, sy = src.switch
    route = src.descent_path(sx + 300.0, sy + 100.0, False)
    assert route is not None and not route["physics"]
    assert set(route["moves"]) <= {"run", "fall", "up"}


def test_physics_route_is_gated_by_climb_reach(stairs):
    """Physics rising steps are only legal into wall-adjacent cells or within a
    floor jump's reach -- the rule that stops the field flooding up open air."""
    src = FreeSpaceDistanceSource(stairs, physics_cost=True)
    sx, sy = src.switch
    route = src.descent_path(sx + 300.0, sy + 100.0, False)
    assert route is not None and route["physics"]
    assert set(route["moves"]) <= {"run", "air", "fall", "jump", "wall", "arc"}

    from nclone.value_function.free_space_distance import (
        _CELL, _physics_cost_constants,
    )

    reach = _physics_cost_constants()["jump_reach_cells"]
    pts, moves = route["points"], route["moves"]
    for i, cls in enumerate(moves):
        if cls not in ("jump", "wall"):
            continue
        nx, ny = int(pts[i + 1][0] / _CELL), int(pts[i + 1][1] / _CELL)
        assert src._wall_adj[nx, ny] or src._floor_rise[nx, ny] <= reach


def test_physics_route_costs_more_frames_than_geometric_grid_units(stairs):
    """Sanity on units: the physics field is FRAMES and prices climbing, so on a
    level that must be climbed it is much larger than the geometric grid count.
    Guards against the two models being silently swapped."""
    geo = FreeSpaceDistanceSource(stairs, physics_cost=False)
    phys = FreeSpaceDistanceSource(stairs, physics_cost=True)
    sx, sy = geo.switch
    g = geo.descent_path(sx + 300.0, sy + 100.0, False)
    p = phys.descent_path(sx + 300.0, sy + 100.0, False)
    assert g is not None and p is not None
    assert p["total"] > g["total"] * 2.0


def test_descent_path_returns_none_off_grid(stairs):
    src = FreeSpaceDistanceSource(stairs, physics_cost=True)
    assert src.descent_path(-500.0, -500.0, False) is None


def test_descent_path_handles_start_inside_geometry(stairs):
    """The ninja can momentarily sit inside a tile sliver; the route should nudge
    to a nearby finite cell rather than give up."""
    src = FreeSpaceDistanceSource(stairs, physics_cost=True)
    solid = np.argwhere(~src._free)
    assert solid.size, "test level has no solid cells"
    # A solid cell adjacent to reachable space.
    for cx, cy in solid:
        route = src.descent_path(float(cx) * 6.0 + 3.0, float(cy) * 6.0 + 3.0, False)
        if route is not None:
            assert route["points"]
            return
    pytest.skip("no solid cell within nudge range of the flooded region")


# --------------------------------------------------------------------------
# Wall-launch arc edges
# --------------------------------------------------------------------------
def test_arc_edges_unseal_a_level_the_step_model_severs():
    """``long-jump-reachable``'s lower chamber is only exitable by a wall jump.

    Every cell of the gap in the divider has ``floor_rise = 48`` (the chamber
    below) and ``wall_adj = 0``, so the up-gate refuses all of them and the
    8-connected step model cannot express the one move that works.
    """
    import nclone.value_function.free_space_distance as fsd

    md = _load("long-jump-reachable")
    prev = fsd._ARC_EDGES
    try:
        fsd._ARC_EDGES = False
        fsd._FS_MEM_CACHE.clear()
        without = fsd.FreeSpaceDistanceSource(md, physics_cost=True)
        cov_without = (np.isfinite(without._d_switch).sum()
                       / max(int(without._free.sum()), 1))
        fsd._ARC_EDGES = True
        fsd._FS_MEM_CACHE.clear()
        with_arcs = fsd.FreeSpaceDistanceSource(md, physics_cost=True)
        cov_with = (np.isfinite(with_arcs._d_switch).sum()
                    / max(int(with_arcs._free.sum()), 1))
    finally:
        fsd._ARC_EDGES = prev
        fsd._FS_MEM_CACHE.clear()

    assert cov_without < 0.6, f"expected the step model to sever it, got {cov_without}"
    assert cov_with > 0.99, f"arcs should reach the whole level, got {cov_with}"


def test_arc_edges_only_target_cells_the_step_model_refuses():
    """Arc edges exist to unlock cells no climb can enter — not to shortcut
    ordinary movement. Every landing must fail the up-gate."""
    from nclone.value_function.free_space_distance import _physics_cost_constants

    src = FreeSpaceDistanceSource(_load("long-jump-reachable"), physics_cost=True)
    assert src._arc_rev_start is not None and src._arc_rev_src.size
    reach = _physics_cost_constants()["jump_reach_cells"]
    counts = np.diff(src._arc_rev_start)
    landings = np.repeat(np.arange(counts.size), counts)
    H = src._free.shape[1]
    lx, ly = landings // H, landings % H
    assert src._free[lx, ly].all()
    assert not src._wall_adj[lx, ly].any()
    assert (src._floor_rise[lx, ly] > reach).all()


def test_arc_route_uses_arc_hops_and_reaches_goal():
    """The drawn route must be able to take the same edges the field did,
    otherwise the overlay shows a stalled path over a solved field."""
    from nclone.value_function.free_space_distance import _CELL

    src = FreeSpaceDistanceSource(_load("long-jump-reachable"), physics_cost=True)
    route = src.descent_path(60 * _CELL + 3, 80 * _CELL + 3, False)
    assert route is not None
    assert route["reached_goal"]
    assert route["moves"].count("arc") > 0
    assert sum(route["costs"]) == pytest.approx(route["total"], rel=1e-9)


def test_arc_edges_absent_where_nothing_is_gated():
    """squiggly is climbable everywhere (max floor_rise 11 < reach), so it must
    emit no arc edges at all -- a guard against the builder over-connecting."""
    src = FreeSpaceDistanceSource(_load("squiggly"), physics_cost=True)
    assert src._arc_rev_src is None or src._arc_rev_src.size == 0
