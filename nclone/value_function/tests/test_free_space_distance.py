"""Tests for the free-space grid cost-to-go potential (free_space_distance.py).

It must (a) give a finite, goal-decreasing cost-to-go on a normal corridor, and
(b) cover airborne/mine-threading levels (mine-difficult-two) where the surface
lattice is severed — the whole point of this fallback potential.
"""

from __future__ import annotations

import numpy as np
import pytest

from nclone.constants.physics_constants import (
    NINJA_RADIUS, EXIT_SWITCH_RADIUS, EXIT_DOOR_RADIUS,
)
from nclone.map_generation.map import Map
from nclone.value_function.free_space_distance import FreeSpaceDistanceSource


def _corridor_map() -> list:
    m = Map()
    for x in range(2, 41):
        m.set_tile(x, 12, 1)  # floor
    m.set_ninja_spawn(4, 11)
    m.add_entity(3, 38, 11, 0, 0, switch_x=10, switch_y=11)  # exit + switch
    return [int(v) for v in m.map_data()]


def test_available_and_goal_decreasing():
    src = FreeSpaceDistanceSource(_corridor_map(), name="corridor")
    assert src.available
    # cost-to-switch decreases as we approach the switch (grid 10 ~ x=... ).
    sx, sy = src.switch
    far = src.cost_to_go(sx + 240, sy, switch_activated=False)
    near = src.cost_to_go(sx + 24, sy, switch_activated=False)
    assert np.isfinite(far) and np.isfinite(near)
    assert near < far  # closer to the switch => lower cost-to-go


def test_cost_zero_within_switch_activation_reach():
    """Cost-to-go bottoms out at the activation boundary: the ninja triggers the
    switch once its CENTER is within (NINJA_RADIUS + EXIT_SWITCH_RADIUS) of the
    switch center (ninja edge touches switch edge), not at the exact center."""
    src = FreeSpaceDistanceSource(_corridor_map(), name="corridor")
    sx, sy = src.switch
    reach = NINJA_RADIUS + EXIT_SWITCH_RADIUS  # 16px
    # Center and any point inside the activation disc -> cost 0.
    assert src.cost_to_go(sx, sy, switch_activated=False) == 0.0
    assert src.cost_to_go(sx + (reach - 4), sy, switch_activated=False) == 0.0
    # Well outside the disc -> strictly positive.
    assert src.cost_to_go(sx + (reach + 24), sy, switch_activated=False) > 0.0


def test_cost_zero_within_exit_activation_reach():
    src = FreeSpaceDistanceSource(_corridor_map(), name="corridor")
    ex, ey = src.exit
    reach = NINJA_RADIUS + EXIT_DOOR_RADIUS  # 22px
    assert src.cost_to_go(ex, ey, switch_activated=True) == 0.0
    assert src.cost_to_go(ex - (reach - 4), ey, switch_activated=True) == 0.0
    assert src.cost_to_go(ex - (reach + 24), ey, switch_activated=True) > 0.0


def test_activation_reach_is_symmetric():
    """No left/right snap bias: points equidistant on either side of the switch
    center share the same (zero) cost inside the activation disc."""
    src = FreeSpaceDistanceSource(_corridor_map(), name="corridor")
    sx, sy = src.switch
    inside = NINJA_RADIUS + EXIT_SWITCH_RADIUS - 4
    left = src.cost_to_go(sx - inside, sy, switch_activated=False)
    right = src.cost_to_go(sx + inside, sy, switch_activated=False)
    assert left == right == 0.0


def test_cost_to_completion_composes_phase0():
    """Phase-0 completion cost = cost-to-switch + the constant exit tail."""
    src = FreeSpaceDistanceSource(_corridor_map(), name="corridor")
    sx, sy = src.switch
    x = sx + 240
    expected = src.cost_to_go(x, sy, False) + src.exit_cost_at_switch
    assert src.cost_to_completion(x, sy, False) == pytest.approx(expected)
    # Phase 1 is just cost-to-exit (no tail).
    assert src.cost_to_completion(x, sy, True) == src.cost_to_go(x, sy, True)


def test_cost_to_completion_continuous_at_switch():
    """The composed potential must NOT jump at switch activation: at the switch
    position the phase-0 completion cost (≈ exit_cost_at_switch, since cost-to-
    switch is ~0 there) equals the phase-1 cost-to-exit. This continuity is what
    keeps PBRS from punishing the activation step under the free-space fallback.
    """
    src = FreeSpaceDistanceSource(_corridor_map(), name="corridor")
    sx, sy = src.switch
    phase0_at_switch = src.cost_to_completion(sx, sy, False)
    phase1_at_switch = src.cost_to_completion(sx, sy, True)
    # cost-to-switch is 0 inside the activation disc, so phase0 == exit tail.
    assert phase0_at_switch == pytest.approx(src.exit_cost_at_switch)
    # The two phases agree at the boundary to within one grid step (the exit tail
    # constant is sampled at the nearest free cell to the switch).
    assert abs(phase0_at_switch - phase1_at_switch) <= 1.5


def test_cost_to_completion_monotone_into_switch():
    """Phase-0 completion cost strictly decreases as the ninja nears the switch."""
    src = FreeSpaceDistanceSource(_corridor_map(), name="corridor")
    sx, sy = src.switch
    far = src.cost_to_completion(sx + 240, sy, False)
    near = src.cost_to_completion(sx + 48, sy, False)
    assert near < far


def test_nav_feature_shape_and_phase():
    src = FreeSpaceDistanceSource(_corridor_map(), name="corridor")
    nav = src.nav_feature(src.switch[0] + 100, src.switch[1], switch_activated=False)
    assert nav.shape == (4,) and nav.dtype == np.float32
    assert nav[3] == 0.0  # phase
    assert src.nav_feature(src.exit[0], src.exit[1], switch_activated=True)[3] == 1.0


def test_free_mask_follows_subtile_geometry():
    """The free/solid mask must use the canonical 6px tile-geometry table, NOT
    treat any non-empty tile as a fully-solid 24px block. A 45-degree slope tile
    therefore has a MIX of free and solid 6px sub-cells, each matching
    SUBNODE_VALIDITY_TABLE — something a whole-tile mask could never produce.
    """
    from nclone.graph.reachability.graph_builder import SUBNODE_VALIDITY_TABLE

    m = Map()
    for x in range(2, 41):
        m.set_tile(x, 12, 1)            # floor
    slope_tx, slope_ty, slope_type = 20, 11, 6  # 45-degree slope above the floor
    m.set_tile(slope_tx, slope_ty, slope_type)
    m.set_ninja_spawn(4, 11)
    m.add_entity(3, 38, 11, 0, 0, switch_x=10, switch_y=11)
    src = FreeSpaceDistanceSource([int(v) for v in m.map_data()], name="slope")

    table = SUBNODE_VALIDITY_TABLE[slope_type]
    seen = set()
    for sx in range(4):
        for sy in range(4):
            # Free-space grid index of sub-cell (sx, sy) of tile (tx, ty).
            ix, iy = slope_tx * 4 + sx, slope_ty * 4 + sy
            assert bool(src._free[ix, iy]) == bool(table[(sx, sy)])
            seen.add(bool(table[(sx, sy)]))
    # A 45-degree slope is genuinely partial: both free and solid sub-cells exist.
    assert seen == {True, False}


def test_unavailable_is_safe():
    # Degenerate map_data -> build fails gracefully, not available, inf cost.
    src = FreeSpaceDistanceSource([0] * 1300, name="empty")
    # (no switch/exit entities -> not available)
    assert not src.available
    assert src.cost_to_go(100, 100, False) == float("inf")


@pytest.mark.slow
def test_covers_mine_difficult_two_where_lattice_is_severed():
    """The headline case: mine-difficult-two's surface lattice gives inf at spawn,
    but the free-space grid gives a finite, corridor-following cost-to-go."""
    import nclone_native
    from nclone.value_function.tools.physics_route_spike import _load_map_data, _goal_positions
    md = _load_map_data("nclone/test_maps/mine-difficult-two")
    sim = nclone_native.Simulator.load(list(md))
    spawn = sim.get_ninja_pos()
    src = FreeSpaceDistanceSource(list(md), name="mine-difficult-two")
    assert src.available
    assert src.spawn_finite(*spawn)
    c = src.cost_to_go(spawn[0], spawn[1], switch_activated=False)
    assert np.isfinite(c) and c > 0
