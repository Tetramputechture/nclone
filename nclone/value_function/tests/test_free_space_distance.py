"""Tests for the free-space grid cost-to-go potential (free_space_distance.py).

It must (a) give a finite, goal-decreasing cost-to-go on a normal corridor, and
(b) cover airborne/mine-threading levels (mine-difficult-two) where the surface
lattice is severed — the whole point of this fallback potential.
"""

from __future__ import annotations

import numpy as np
import pytest

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


def test_nav_feature_shape_and_phase():
    src = FreeSpaceDistanceSource(_corridor_map(), name="corridor")
    nav = src.nav_feature(src.switch[0] + 100, src.switch[1], switch_activated=False)
    assert nav.shape == (4,) and nav.dtype == np.float32
    assert nav[3] == 0.0  # phase
    assert src.nav_feature(src.exit[0], src.exit[1], switch_activated=True)[3] == 1.0


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
