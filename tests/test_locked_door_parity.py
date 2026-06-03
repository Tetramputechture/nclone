"""Locked-door (entity type 6) Rust↔Python parity + behaviour tests.

Covers the native locked-door implementation added to nclone-native:

* The closed door segment physically BLOCKS the ninja.
* Touching the switch (radius 5) OPENS the door permanently and bumps
  ``ninja.doors_opened``.
* The opened door no longer blocks.
* ``fast_reset`` re-closes the door (the segment is re-activated simulator-side).
* The native engine is byte-identical to the pure-Python ``nclone.nsim``
  simulator over the same input sequence (trajectory + doors_opened).

These exercise end-to-end physics on a minimal hand-built corridor rather than
a procgen level, so the locked-door mechanic is isolated.
"""

from __future__ import annotations

import nclone_native
import pytest

from nclone.map_generation.map import Map
from nclone.nsim import Simulator
from nclone.sim_config import SimConfig


def _corridor_with_locked_door(door_switch_grid_x: int) -> list[int]:
    """A flat corridor (solid floor at y=12) with a vertical locked door at grid
    x=20 and an exit at grid x=38. The door's switch sits at ``door_switch_grid_x``
    (use a value < 20 to make it reachable before the door, > 20 to place it
    behind the closed door)."""
    m = Map()
    for x in range(2, 41):
        m.set_tile(x, 12, 1)  # solid floor
    m.set_ninja_spawn(4, 11)
    m.add_entity(3, 38, 11, 0, 0, switch_x=36, switch_y=11)  # exit + its switch
    m.add_entity(6, 20, 11, 0, 0, switch_x=door_switch_grid_x, switch_y=11)
    return [int(v) for v in m.map_data()]


def _run_native(map_data: list[int], n: int):
    sim = nclone_native.Simulator.load(list(map_data))
    xs, doors = [], []
    for _ in range(n):
        sim.tick(1, 0)  # hold right
        st = sim.get_state()
        xs.append(round(st.ninja.xpos, 4))
        doors.append(st.ninja.doors_opened)
    return sim, xs, doors


def _run_python(map_data: list[int], n: int):
    sim = Simulator(SimConfig())
    sim.load(list(map_data))
    xs, doors = [], []
    for _ in range(n):
        sim.tick(1, 0)
        xs.append(round(sim.ninja.xpos, 4))
        doors.append(sim.ninja.doors_opened)
    return sim, xs, doors


def test_locked_door_loads_with_switch_and_barrier():
    """The native loader creates a type-6 entity whose (x, y) is the switch and
    whose door_* fields describe the (initially closed) barrier."""
    sim = nclone_native.Simulator.load(_corridor_with_locked_door(10))
    doors = [e for e in sim.get_state().entities if e.entity_type == 6]
    assert len(doors) == 1
    d = doors[0]
    assert d.active is True  # switch uncollected
    assert d.door_closed is True
    # Switch (entity position) is distinct from the door barrier.
    assert (d.x, d.y) != (d.door_x, d.door_y)


def test_closed_door_blocks_ninja():
    """With the switch placed BEHIND the closed door, the ninja cannot reach the
    exit and never opens the door."""
    sim, xs, doors = _run_native(_corridor_with_locked_door(30), 600)
    assert doors[-1] == 0
    # Blocked well short of the exit (exit switch ~ grid 36 -> ~864px).
    assert xs[-1] < 600


def test_reachable_switch_opens_door_and_lets_ninja_through():
    """With the switch reachable, the ninja hits it, the door opens once, and the
    ninja proceeds past the (former) barrier toward the exit."""
    sim, xs, doors = _run_native(_corridor_with_locked_door(10), 600)
    assert doors[-1] == 1
    assert xs[-1] > 900  # passed the barrier (grid 20) on toward the exit


def test_fast_reset_recloses_door():
    """Opening the door then fast_reset must restore the closed/blocking state."""
    sim, _, _ = _run_native(_corridor_with_locked_door(10), 120)
    opened = [e for e in sim.get_state().entities if e.entity_type == 6][0]
    assert opened.door_closed is False and opened.active is False
    sim.fast_reset()
    st = sim.get_state()
    reset_door = [e for e in st.entities if e.entity_type == 6][0]
    assert reset_door.door_closed is True
    assert reset_door.active is True
    assert st.ninja.doors_opened == 0


@pytest.mark.parametrize("door_switch_grid_x", [10, 30])
def test_native_matches_python_simulator(door_switch_grid_x):
    """The native engine reproduces the pure-Python simulator byte-for-byte over
    600 frames: identical ninja x-trajectory and identical doors_opened sequence,
    for both the reachable-switch and blocked-switch layouts."""
    md = _corridor_with_locked_door(door_switch_grid_x)
    _, nx, ndoors = _run_native(md, 600)
    _, px, pdoors = _run_python(md, 600)
    max_dx = max(abs(a - b) for a, b in zip(nx, px))
    assert max_dx == 0.0, f"trajectory diverged by {max_dx}px"
    assert ndoors == pdoors, "doors_opened sequence diverged"
