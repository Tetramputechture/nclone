"""Unit tests for the faithful per-config locked-door dependency solver.

A synthetic single-bin (LAUNCH_PAD) corridor where each door, when CLOSED, removes
the edge across its barrier (severing the corridor there) and, when OPEN, restores
it — exactly the per-configuration physics the real solver enumerates. This checks
closure / minimal-necessity / dependency-order / per-leg d* without the heavy
native edge enumeration.
"""

from __future__ import annotations

import numpy as np

from nclone.value_function.surface_extraction import SurfacePoint, SURFACE_LAUNCH_PAD
from nclone.value_function.surface_graph import SurfaceGraph, num_velocity_bins
from nclone.value_function.door_dependency import DoorDescriptor, solve_door_dependencies

STEP = 24.0
Y = 100.0
N_POINTS = 16  # x = 0, 24, ..., 360


def _x(point: int) -> float:
    return STEP * point


def _make_corridor(cuts_when_closed):
    """Return enumerate_graph(open_set) for a 16-point corridor. ``cuts_when_closed``
    maps door_index -> (i, i+1) edge removed while that door is closed."""
    assert num_velocity_bins(SURFACE_LAUNCH_PAD) == 1
    pts = [
        SurfacePoint(x=_x(i), y=Y, normal_x=0.0, normal_y=-1.0,
                     surface_type=SURFACE_LAUNCH_PAD, segment_id=0, is_edge=False)
        for i in range(N_POINTS)
    ]

    def enumerate_graph(open_set):
        cuts = {gap for d, gap in cuts_when_closed.items() if d not in open_set}
        src, dst, w = [], [], []
        for i in range(N_POINTS - 1):
            if (i, i + 1) in cuts:
                continue
            src += [i, i + 1]
            dst += [i + 1, i]
            w += [1.0, 1.0]
        return SurfaceGraph(
            surface_points=pts, n_points=N_POINTS, n_nodes=N_POINTS,
            velocity_bins_floor=np.zeros(1, np.float32),
            velocity_bins_wall=np.zeros(1, np.float32), max_vel_bins=1,
            edges_src=np.asarray(src, np.int32),
            edges_dst=np.asarray(dst, np.int32),
            edges_weight=np.asarray(w, np.float32),
        )

    return enumerate_graph


def _door(switch_point: int) -> DoorDescriptor:
    return DoorDescriptor(switch_x=_x(switch_point), switch_y=Y)


def test_no_doors_solvable():
    eg = _make_corridor({})
    plan = solve_door_dependencies(eg, [], _x(0), Y, _x(12), Y, _x(14), Y)
    assert plan.solvable
    assert plan.necessary_order == []


def test_nested_doors_resolve_in_order():
    # Door 0 (switch@pt2) cuts 4-5; Door 1 (switch@pt8, behind door 0) cuts 9-10.
    eg = _make_corridor({0: (4, 5), 1: (9, 10)})
    plan = solve_door_dependencies(eg, [_door(2), _door(8)],
                                   _x(0), Y, _x(12), Y, _x(14), Y)
    assert plan.solvable, plan.reason
    assert plan.necessary_order == [0, 1]   # door 0 before door 1
    assert len(plan.door_leg_dstar) == 2
    assert plan.exit_leg_dstar is not None
    # Spawn (node 0) can reach the exit on the all-open graph.
    assert np.isfinite(plan.exit_leg_dstar[0])
    # Leg-0 guidance: cost-to-go to switch 0 (point 2) is finite from spawn.
    assert np.isfinite(plan.door_leg_dstar[0][0])


def test_irrelevant_door_dropped_from_necessary():
    # Door 0 gates the path (cut 4-5). Door 1 cuts 14-15, PAST the exit -> useless.
    eg = _make_corridor({0: (4, 5), 1: (14, 15)})
    plan = solve_door_dependencies(eg, [_door(2), _door(6)],
                                   _x(0), Y, _x(10), Y, _x(12), Y)
    assert plan.solvable
    assert plan.necessary_order == [0]      # only door 0 is necessary
    assert len(plan.door_leg_dstar) == 1


def test_self_gated_door_unsolvable():
    # Door 1's switch (point 11) sits BEHIND its own cut (10-11); nothing opens it.
    eg = _make_corridor({0: (4, 5), 1: (10, 11)})
    plan = solve_door_dependencies(eg, [_door(2), _door(11)],
                                   _x(0), Y, _x(13), Y, _x(14), Y)
    assert not plan.solvable
    assert "unreachable" in plan.reason


def test_enumerations_are_cached_per_config():
    calls = {"n": 0}
    base = _make_corridor({0: (4, 5), 1: (9, 10)})

    def counting(open_set):
        calls["n"] += 1
        return base(open_set)

    solve_door_dependencies(counting, [_door(2), _door(8)],
                            _x(0), Y, _x(12), Y, _x(14), Y)
    # Distinct configs visited: {}, {0}, {0,1} (+ maybe single-drop probes) — but
    # each frozenset is enumerated at most once, so far fewer than the naive count.
    assert calls["n"] <= 5
