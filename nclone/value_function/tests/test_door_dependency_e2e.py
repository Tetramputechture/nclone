"""End-to-end locked-door planner tests through the real native enumeration.

Builds small tile maps with locked doors, runs the full LatticeAudit precompute
(native edge enumeration with doors forced open) + door-dependency solver, and
asserts the resolved dependency structure. These are slower than the synthetic
solver unit tests (they enumerate physics edges) but validate the integration:
Rust door extraction -> doors-open enumeration -> edge/node masking -> N-phase
sequenced cost-to-go exposed on LatticeDistanceSource.

A 1-tile-tall tunnel is used so doors are unavoidable (the ninja cannot jump over
a door that fills the passage), which is what makes a door "necessary".
"""

from __future__ import annotations

import numpy as np
import pytest

from nclone.map_generation.map import Map
from nclone.value_function.lattice_distance import LatticeDistanceSource


def _tunnel_base() -> Map:
    """Fully solid map with a 1-tile tunnel carved at grid row 11 (x 3..39)."""
    m = Map()
    for x in range(0, 44):
        for y in range(0, 25):
            m.set_tile(x, y, 1)
    for x in range(3, 40):
        m.set_tile(x, 11, 0)
    m.set_ninja_spawn(4, 11)
    return m


def _build(m: Map) -> LatticeDistanceSource:
    md = [int(v) for v in m.map_data()]
    return LatticeDistanceSource(md, use_cache=False, use_native=True, workers=2)


@pytest.mark.slow
def test_single_necessary_door_creates_a_prerequisite_phase():
    m = _tunnel_base()
    m.add_entity(3, 38, 11, 0, 0, switch_x=36, switch_y=11)
    # Door barrier at grid 20; switch reachable before it (grid 10).
    m.add_entity(6, 20, 11, 0, 0, switch_x=10, switch_y=11)
    src = _build(m)
    assert src.available
    plan = src._door_plan
    assert plan is not None and plan.solvable
    assert plan.necessary_order == [0]
    assert len(plan.door_leg_dstar) == 1
    assert plan.exit_leg_dstar is not None
    # While the door is shut -> door leg 0; once open -> None (normal switch/exit).
    assert src.door_phase(False, set()) == 0
    assert src.door_phase(False, {0}) is None
    assert src.door_phase(True, {0}) is None


@pytest.mark.slow
def test_self_gated_door_is_unsolvable():
    m = _tunnel_base()
    m.add_entity(3, 38, 11, 0, 0, switch_x=36, switch_y=11)
    # Door barrier at grid 20 with its own switch BEHIND it (grid 30): deadlock.
    m.add_entity(6, 20, 11, 0, 0, switch_x=30, switch_y=11)
    src = _build(m)
    # Unsolvable plans are not stored; the source falls back to the two-phase d*.
    assert src._door_plan is None


@pytest.mark.slow
def test_nested_doors_resolve_in_dependency_order():
    """Door B's switch sits behind Door A: the solver must order A before B."""
    m = _tunnel_base()
    m.add_entity(3, 38, 11, 0, 0, switch_x=36, switch_y=11)
    m.add_entity(6, 14, 11, 0, 0, switch_x=8, switch_y=11)   # door A (index 0)
    m.add_entity(6, 26, 11, 0, 0, switch_x=20, switch_y=11)  # door B (index 1)
    src = _build(m)
    assert src.available
    plan = src._door_plan
    assert plan is not None and plan.solvable
    assert plan.necessary_order == [0, 1]  # A before B
    assert len(plan.door_leg_dstar) == 2
    # Phase climbs as doors open in order; None once both necessary doors are open.
    assert src.door_phase(False, set()) == 0
    assert src.door_phase(False, {0}) == 1
    assert src.door_phase(False, {0, 1}) is None
    assert src.door_phase(True, {0, 1}) is None


@pytest.mark.slow
def test_phase_cost_to_go_is_finite_on_the_route():
    m = _tunnel_base()
    m.add_entity(3, 38, 11, 0, 0, switch_x=36, switch_y=11)
    m.add_entity(6, 20, 11, 0, 0, switch_x=10, switch_y=11)
    src = _build(m)
    # The exit switch position is on the graph; in the all-doors-open phase its
    # cost-to-go (to the exit door) must be finite.
    ex, ey = src.switch
    c = src.cost_to_go(ex, ey, switch_activated=False, open_door_indices={0})
    assert np.isfinite(c)
