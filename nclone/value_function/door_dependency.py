"""Locked-door dependency solver for the lattice planner (faithful per-config).

A locked door is a wall segment that opens permanently when the ninja touches its
switch. Doors can gate each other (a switch behind another door) and gate the exit
switch / exit door, so the planner must decide which doors to open, in what order,
to make the level completable — without augmenting lattice node state with door
bits (2**N).

Earlier this masked a single doors-open graph to approximate each closed-door
topology. That LEAKS: a closed door is a wall whose blocked transitions aren't all
caught by straight-line edge crossing, so necessity was under-counted (e.g. it
falsely reported squiggly's exit reachable with doors closed). This module instead
**enumerates the physics edges per door-configuration** (faithful), via an
``enumerate_graph(open_set)`` callback the caller supplies. Enumerations are cached
by the open-door frozenset, so each distinct config is built once.

Outputs (``DoorPlan``):

* ``necessary_order``: the minimal set of doors that must be opened for the exit to
  be reachable, in a dependency-respecting open order.
* ``door_leg_dstar[j]``: cost-to-go to the j-th necessary switch, computed on the
  graph with the first j necessary doors open (the search/RL guidance for the
  door-opening phases).
* ``exit_leg_dstar``: cost-to-go to the exit door on the all-necessary-open graph.

The agent's phase = number of necessary doors opened; while < k it descends
``door_leg_dstar[phase]`` toward the next switch, then the level's normal
switch->exit potential takes over.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple  # noqa: F401

import numpy as np

from ..constants.physics_constants import NINJA_RADIUS
from .surface_graph import (
    SurfaceGraph,
    dijkstra_backward,
    find_goal_nodes,
    node_id,
    num_velocity_bins,
)


@dataclass
class DoorDescriptor:
    """A locked door: the switch that opens it (``switch_x/y``) and, optionally,
    the barrier segment endpoints (kept for diagnostics; the faithful solver does
    not need them since enumeration models the barrier directly)."""

    switch_x: float
    switch_y: float
    seg_x1: float = 0.0
    seg_y1: float = 0.0
    seg_x2: float = 0.0
    seg_y2: float = 0.0


@dataclass
class DoorPlan:
    """Result of :func:`solve_door_dependencies`."""

    solvable: bool
    necessary_order: List[int] = field(default_factory=list)
    # Per necessary door (aligned with necessary_order): cost-to-go to its switch,
    # on the graph with the earlier necessary doors open.
    door_leg_dstar: List[np.ndarray] = field(default_factory=list)
    # Cost-to-go to the exit door on the all-necessary-doors-open graph.
    exit_leg_dstar: Optional[np.ndarray] = None
    reason: str = ""

    @property
    def num_necessary(self) -> int:
        return len(self.necessary_order)


# ---------------------------------------------------------------------------
# Forward reachability on a (faithfully enumerated) config graph
# ---------------------------------------------------------------------------
def _build_forward_adjacency(graph: SurfaceGraph) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = {}
    src, dst = graph.edges_src, graph.edges_dst
    for i in range(len(src)):
        s = int(src[i])
        adj.setdefault(s, []).append(int(dst[i]))
    return adj


def _start_nodes(graph: SurfaceGraph, x: float, y: float) -> List[int]:
    """Contact nodes near (x, y); falls back to the nearest surface point's bins."""
    nodes = find_goal_nodes(graph, x, y, "exit")
    if nodes:
        return nodes
    best_idx, best_d2 = None, float("inf")
    for pt_idx, sp in enumerate(graph.surface_points):
        d2 = (sp.x - x) ** 2 + (sp.y - y) ** 2
        if d2 < best_d2:
            best_d2, best_idx = d2, pt_idx
    if best_idx is None:
        return []
    return [
        node_id(best_idx, vb, graph.max_vel_bins)
        for vb in range(num_velocity_bins(graph.surface_points[best_idx].surface_type))
    ]


def _reachable_from(graph: SurfaceGraph, x: float, y: float) -> Set[int]:
    """Forward-BFS set of nodes reachable from (x, y) over the graph's edges."""
    fwd = _build_forward_adjacency(graph)
    reached: Set[int] = set(_start_nodes(graph, x, y))
    q: deque[int] = deque(reached)
    while q:
        u = q.popleft()
        for v in fwd.get(u, ()):  # type: ignore[arg-type]
            if v not in reached:
                reached.add(v)
                q.append(v)
    return reached


def _reaches(graph: SurfaceGraph, reach: Set[int], x: float, y: float, kind: str) -> bool:
    return any(n in reach for n in find_goal_nodes(graph, x, y, kind))


# ---------------------------------------------------------------------------
# Faithful solver
# ---------------------------------------------------------------------------
class SurfaceGraphOracle:
    """The original per-configuration surface-graph backend.

    Kept as the default so every existing caller is unchanged.  The solver below
    talks only to this four-method interface, which is what lets the SAME
    dependency algorithm run over the free-space flood instead — see
    ``free_space_door_plan.FreeSpaceDoorOracle``.  The algorithm was never
    graph-specific; only these four primitives were.
    """

    def __init__(self, enumerate_graph: Callable[[frozenset], SurfaceGraph]):
        self._enumerate = enumerate_graph
        self._cache: Dict[frozenset, SurfaceGraph] = {}

    def config(self, open_set):
        key = frozenset(open_set)
        g = self._cache.get(key)
        if g is None:
            g = self._enumerate(key)
            self._cache[key] = g
        return g

    def reachable_from(self, cfg, x: float, y: float):
        return _reachable_from(cfg, x, y)

    def reaches(self, cfg, reach, x: float, y: float, kind: str) -> bool:
        return _reaches(cfg, reach, x, y, kind)

    def leg_field(self, cfg, x: float, y: float, kind: str) -> np.ndarray:
        return dijkstra_backward(cfg, find_goal_nodes(cfg, x, y, kind))


def solve_door_dependencies(
    enumerate_graph: Optional[Callable[[frozenset], SurfaceGraph]],
    doors: Sequence[DoorDescriptor],
    spawn_x: float,
    spawn_y: float,
    switch_x: float,
    switch_y: float,
    exit_x: float,
    exit_y: float,
    oracle=None,
) -> DoorPlan:
    """Resolve locked-door dependencies by faithful per-config re-evaluation.

    ``enumerate_graph(frozenset(open_door_indices))`` must return a SurfaceGraph
    with edges enumerated for that door configuration, over a SHARED surface-point
    set (so node ids are comparable across configs). See the module docstring.

    ``oracle`` supplies a different backend for the same algorithm (see
    :class:`SurfaceGraphOracle` for the interface).  Exactly one of the two is
    required; ``oracle`` wins when both are given.
    """
    n = len(doors)
    sw = [(d.switch_x, d.switch_y) for d in doors]

    if oracle is None:
        if enumerate_graph is None:
            raise ValueError("solve_door_dependencies needs enumerate_graph or oracle")
        oracle = SurfaceGraphOracle(enumerate_graph)

    def G(open_set):
        return oracle.config(open_set)

    def exit_ok(open_set) -> bool:
        g = G(open_set)
        reach = oracle.reachable_from(g, spawn_x, spawn_y)
        return (oracle.reaches(g, reach, exit_x, exit_y, "exit")
                and oracle.reaches(g, reach, switch_x, switch_y, "switch"))

    if n == 0:
        return DoorPlan(solvable=exit_ok(frozenset()), necessary_order=[])

    # --- Closure: which doors are openable from spawn, and a discovery order ----
    opened: Set[int] = set()
    while True:
        g = G(opened)
        reach = oracle.reachable_from(g, spawn_x, spawn_y)
        newly = [di for di in range(n)
                 if di not in opened
                 and oracle.reaches(g, reach, sw[di][0], sw[di][1], "switch")]
        if not newly:
            break
        opened.update(newly)
    closure = set(opened)

    if not exit_ok(closure):
        return DoorPlan(
            solvable=False,
            reason="exit unreachable even after opening every openable door "
            "(dependency cycle or unreachable prerequisite switch)",
        )

    # --- Necessary set: exit-critical doors, CLOSED under prerequisites ---------
    #
    # The previous rule greedily dropped any door whose removal still left the
    # exit reachable. That tests SUFFICIENCY ("with this set open, can I finish")
    # and never tests FEASIBILITY ("can I actually open this set from spawn"),
    # because ``exit_ok`` assumes the candidate set is already open without
    # asking how. On a prerequisite chain -- A's switch is reachable from spawn,
    # B's switch only once A is open, the exit only once B is open -- it drops A
    # (the exit is fine with B open) and the order loop below then finds B
    # unopenable and returns solvable=False on a LEVEL THAT IS SOLVABLE, blaming
    # a "dependency cycle" that does not exist. Prerequisite chains are the
    # normal way multi-door levels are built, so this was not an edge case.
    #
    # The rule now names the two ways a door can matter, which is what a door
    # actually does on these maps:
    #   1. EXIT-CRITICAL -- shutting it (with every other openable door open)
    #      breaks the route to the exit or the exit switch.
    #   2. PREREQUISITE -- shutting it breaks the route to some other necessary
    #      door's SWITCH.
    # Necessity is (1) closed under (2).
    #
    # COST. This is O(n) oracle configurations, the same as the greedy loop it
    # replaces, because both facts are read off the SAME config `closure - {p}`:
    # one pass asks that config about the exit and about every door switch. A
    # feasibility-preserving greedy drop would have been O(n^2) configs, and the
    # corpus will not take that -- official door levels average 7.5 locked doors
    # but the tail runs to 380 on one level, where O(n^2) is ~144k floods.
    #
    # DIRECTION OF ERROR. The blocking test fixes every other door open, so it
    # can miss a pair that only matters jointly; the closure then OVER-includes
    # rather than under-includes. Over-crediting a door costs a spurious
    # milestone reward; under-crediting reports a solvable level unsolvable and
    # deletes its potential. The conservative direction is the correct one here.
    exit_critical: Set[int] = set()
    blocks: Dict[int, Set[int]] = {p: set() for p in closure}
    for p in sorted(closure):
        cand = closure - {p}
        g = G(cand)
        reach = oracle.reachable_from(g, spawn_x, spawn_y)
        if not (oracle.reaches(g, reach, exit_x, exit_y, "exit")
                and oracle.reaches(g, reach, switch_x, switch_y, "switch")):
            exit_critical.add(p)
        # Which OTHER doors' switches does shutting p cut off?
        for d in closure:
            if d == p:
                continue
            if not oracle.reaches(g, reach, sw[d][0], sw[d][1], "switch"):
                blocks[p].add(d)

    # Close the exit-critical set under "is a prerequisite of".
    needed_by: Dict[int, Set[int]] = {d: set() for d in closure}
    for p, ds in blocks.items():
        for d in ds:
            needed_by[d].add(p)
    suff = set(exit_critical)
    stack = list(exit_critical)
    while stack:
        d = stack.pop()
        for p in needed_by.get(d, ()):
            if p not in suff:
                suff.add(p)
                stack.append(p)
    if not suff:
        # Every door is individually droppable and none gates another: the exit
        # is reachable with all doors shut. Genuinely a zero-necessary-door level.
        suff = set()

    # --- Dependency open-order within the minimal set ---------------------------
    order: List[int] = []
    opened = set()
    remaining = set(suff)
    while remaining:
        g = G(opened)
        reach = oracle.reachable_from(g, spawn_x, spawn_y)
        ready = [di for di in remaining
                 if oracle.reaches(g, reach, sw[di][0], sw[di][1], "switch")]
        if not ready:
            return DoorPlan(
                solvable=False,
                reason="minimal door set is not openable in any order "
                "(unexpected dependency cycle)",
            )
        # Nearest-first by Euclidean to keep the visiting order sensible.
        cx, cy = (spawn_x, spawn_y) if not order else sw[order[-1]]
        pick = min(ready, key=lambda di: (sw[di][0] - cx) ** 2 + (sw[di][1] - cy) ** 2)
        order.append(pick)
        opened.add(pick)
        remaining.discard(pick)

    # --- Per-leg cost-to-go (faithful, config-aware) ----------------------------
    door_leg_dstar: List[np.ndarray] = []
    opened = set()
    for di in order:
        g = G(opened)
        door_leg_dstar.append(oracle.leg_field(g, sw[di][0], sw[di][1], "switch"))
        opened.add(di)
    # opened == set(order) here; the all-necessary-open config for the exit leg.
    g_final = G(order)
    exit_leg_dstar = oracle.leg_field(g_final, exit_x, exit_y, "exit")

    return DoorPlan(
        solvable=True,
        necessary_order=order,
        door_leg_dstar=door_leg_dstar,
        exit_leg_dstar=exit_leg_dstar,
    )
