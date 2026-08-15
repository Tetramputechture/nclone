"""Tests for the mine-threading cost-to-go (mine_threading_distance.py).

It must give a CONNECTED, MINE-FAITHFUL cost-to-go on a mine-saturated level where
the surface lattice is severed and the free-space grid is only finite by passing
SOFTLY through lethal cells: the route here threads a narrow channel that stays
>= KILL_RADIUS (14px) from every deadly mine. The headline case is
mines-extremely-difficult, whose ninja-free region is a single connected component
spanning spawn / switch / exit.
"""

from __future__ import annotations

import numpy as np
import pytest

from nclone.map_generation.map import Map
from nclone.value_function.mine_threading_distance import (
    MineThreadingDistanceSource,
    _KILL_RADIUS,
)


def _corridor_with_mine() -> list:
    """A wide floor corridor with a single deadly mine mid-span; the threading
    route must keep the ninja centre >= 14px from it."""
    m = Map()
    for x in range(2, 41):
        for y in (10, 11, 12):
            m.set_tile(x, y, 0)  # open
    for x in range(2, 41):
        m.set_tile(x, 13, 1)  # floor
    m.set_ninja_spawn(4, 9)
    m.add_entity(3, 38, 9, 0, 0, switch_x=10, switch_y=9)  # exit + switch
    # A deadly toggle mine (type 1, state 0) on the 6px grid mid-corridor.
    m.add_entity(1, 22, 10, 0, 0)
    return [int(v) for v in m.map_data()]


def test_corridor_available_and_goal_decreasing():
    src = MineThreadingDistanceSource(_corridor_with_mine(), name="corridor")
    assert src.available
    sx, sy = src.switch
    far = src.cost_to_go(sx + 240, sy, switch_activated=False)
    near = src.cost_to_go(sx + 48, sy, switch_activated=False)
    assert np.isfinite(far) and np.isfinite(near)
    assert near < far  # closer to the switch => lower cost-to-go


def test_cost_field_is_mine_faithful():
    """Every cell with a finite cost-to-go is >= KILL_RADIUS from all mines:
    the implied route NEVER passes through a deadly mine's kill-disc."""
    src = MineThreadingDistanceSource(_corridor_with_mine(), name="corridor")
    finite = np.isfinite(src._d_switch) & src._safe
    assert finite.any()
    assert bool((src._clear[finite] >= _KILL_RADIUS).all())


def test_cost_to_completion_continuous_at_switch():
    src = MineThreadingDistanceSource(_corridor_with_mine(), name="corridor")
    sx, sy = src.switch
    phase0 = src.cost_to_completion(sx, sy, False)
    phase1 = src.cost_to_completion(sx, sy, True)
    assert phase0 == pytest.approx(src.exit_cost_at_switch, abs=1.0)
    assert abs(phase0 - phase1) <= src.exit_cost_at_switch + 1.0


def test_unavailable_is_safe():
    src = MineThreadingDistanceSource([0] * 1300, name="empty")
    assert not src.available
    assert src.cost_to_go(100, 100, False) == float("inf")
    assert not src.spawn_finite(100, 100)


def test_mines_extremely_difficult_threads_the_channel():
    """The headline case: the surface lattice is severed (available=False) but the
    mine-threading source gives a connected, mine-faithful cost-to-go with spawn,
    switch and exit all finite."""
    from nclone.nplay_headless import NPlayHeadless
    from .conftest import require_intact_map_fixture

    # This fixture is a replay payload committed under a map filename, so the
    # level it names does not exist to test against.  See the guard's docstring;
    # the source itself is verified healthy there against the repo's real mine
    # levels.
    require_intact_map_fixture("nclone/test_maps/mines-extremely-difficult")

    nph = NPlayHeadless(render_mode="rgb_array")
    nph.load_map("nclone/test_maps/mines-extremely-difficult")
    md = nph.current_map_data
    spawn = nph.ninja_position()

    src = MineThreadingDistanceSource(md, name="mines-extremely-difficult")
    assert src.available
    assert src.spawn_finite(*spawn)
    # spawn -> switch -> exit all finite (connected threadable route).
    assert np.isfinite(src.cost_to_go(spawn[0], spawn[1], False))
    assert np.isfinite(src.cost_to_completion(spawn[0], spawn[1], False))
    assert np.isfinite(src.cost_to_go(*src.switch, switch_activated=False))
    assert np.isfinite(src.cost_to_go(*src.exit, switch_activated=True))
    # Mine-faithful: no finite-cost cell is inside any 14px mine kill-disc.
    finite = np.isfinite(src._d_switch) & src._safe
    assert bool((src._clear[finite] >= _KILL_RADIUS).all())
