"""Corner Double Jump trick *edges* for the lattice planner.

This promotes the CDJ from a spawn-only executor fallback (``LatticeAudit.
try_cdj_completion``) to a first-class **trick edge** the global ``d*`` can route
through, so tricks compose anywhere in a level (and multiple times) instead of
only at spawn — the prerequisite for open / multi-trick maps.

A trick edge is enumerated by running the adaptive :func:`cdj_primitive.execute_cdj`
(native, position-launched) from a grounded floor launch node near a wall corner.
Because the CDJ needs its dynamic approach to arm both buffers, it cannot be a
static-injection edge like the built-in jump primitives — but it *can* be run
from a grounded launch node, which is exactly what enumeration does. Each edge
records:

  * ``src_node``           — the grounded launch node,
  * ``tape``               — the CDJ per-tick inputs (replayed verbatim by the
                             executor; this IS the edge's "action"),
  * ``dst_node``           — the surface node the ninja lands on after the trick
                             (for *chaining* into normal arcs / further tricks),
  * ``switch/exit_contact``— frames at which the trick trajectory passes within
                             contact range of an air goal (so an air switch the
                             lattice cannot reach becomes reachable *via the
                             trick edge*, like the existing air-contact arcs).

Insertion + ``d*`` recompute + executor replay live in ``tools/lattice_audit``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .surface_extraction import SURFACE_FLOOR
from .surface_graph import (
    SurfaceGraph,
    node_id,
    velocity_to_bin,
)
from .route_extraction import find_nearest_start_node
from .cdj_primitive import execute_cdj

Input = Tuple[int, int]

_GROUNDED_STATES = (0, 1, 2)  # Immobile / Running / GroundSliding


@dataclass
class CDJTrickEdge:
    """A Corner Double Jump as a routable lattice edge."""

    src_node: int
    launch_xy: Tuple[float, float]
    approach_dir: int
    hold_frames: int
    tape: List[Input]                 # CDJ inputs (settle/launch/climb/core)
    climb_px: float
    apex_xy: Tuple[float, float]
    dst_node: Optional[int] = None    # surface node landed on after the trick
    landing_xy: Optional[Tuple[float, float]] = None
    landing_tape: List[Input] = None  # post-core drift inputs to reach dst_node
    switch_contact_frame: Optional[int] = None
    exit_contact_frame: Optional[int] = None

    @property
    def full_tape(self) -> List[Input]:
        """The complete executable tape (CDJ core + the drift that lands it)."""
        return list(self.tape) + list(self.landing_tape or [])

    @property
    def cost_frames(self) -> int:
        return len(self.tape) + len(self.landing_tape or [])


def _inject_grounded(sim, sp) -> None:
    sim.fast_reset()
    sim.inject_ninja_for_value_iteration_grounded(
        float(sp.x), float(sp.y), 0.0, 0.0,
        float(sp.normal_x), float(sp.normal_y),
    )


def _measure_trick(sim, reset_fn, tape: List[Input], switch_xy, exit_xy,
                   switch_range: float, exit_range: float,
                   post_drift: int, post_fall_cap: int):
    """Replay ``tape`` from the launch state (set by ``reset_fn``), then
    drift+fall, recording goal air-contacts and the surface landing. Returns
    (switch_frame, exit_frame, landing_pos, landing_vx, landing_state, drift_tape)."""
    reset_fn()
    sw_f = ex_f = None
    f = 0
    for (h, j) in tape:
        sim.tick(h, j)
        f += 1
        x, y = sim.get_ninja_pos()
        if sw_f is None and math.hypot(x - switch_xy[0], y - switch_xy[1]) <= switch_range:
            sw_f = f
        if ex_f is None and math.hypot(x - exit_xy[0], y - exit_xy[1]) <= exit_range:
            ex_f = f
        if sim.has_died():
            return sw_f, ex_f, None, 0.0, 6, []
    # Drift down to a surface landing for chaining.
    drift: List[Input] = []
    landing = None
    for _ in range(post_fall_cap):
        sim.tick(post_drift, 0)
        drift.append((post_drift, 0))
        f += 1
        x, y = sim.get_ninja_pos()
        if sw_f is None and math.hypot(x - switch_xy[0], y - switch_xy[1]) <= switch_range:
            sw_f = f
        if ex_f is None and math.hypot(x - exit_xy[0], y - exit_xy[1]) <= exit_range:
            ex_f = f
        if sim.has_died():
            return sw_f, ex_f, None, 0.0, 6, drift
        st = sim.get_ninja_state_code()
        if st in _GROUNDED_STATES:
            vx = sim.get_ninja_velocity()[0]
            return sw_f, ex_f, sim.get_ninja_pos(), vx, st, drift
    return sw_f, ex_f, None, 0.0, sim.get_ninja_state_code(), drift


def enumerate_cdj_edges(
    graph: SurfaceGraph,
    sim,
    switch_xy: Tuple[float, float],
    exit_xy: Tuple[float, float],
    *,
    holds: Sequence[int] = (74,),
    dirs: Sequence[int] = (1, -1),
    min_climb: float = 80.0,
    switch_range: float = 16.0,
    exit_range: float = 22.0,
    post_drifts: Sequence[int] = (0, 1, -1),
    post_fall_cap: int = 80,
    spawn_candidate: bool = True,
    approach_sweep: Sequence[int] = (0, 2, 4, 6),
    require_reachable_src: bool = True,
) -> List[CDJTrickEdge]:
    """Find CDJ trick edges across the map (general, not spawn-only).

    For every grounded FLOOR node (plus the spawn, when ``spawn_candidate``),
    runs the adaptive CDJ on the native ``sim`` in each direction; keeps those
    that find a live corner and climb at least ``min_climb`` px. For each,
    records the goal air-contacts and the surface it lands on (trying a few
    post-trick drift directions, preferring the landing that reaches a goal). At
    most one edge per launch node (best — reaches a goal, else highest climb).

    A CDJ only fires where the geometry supports the buffer intersection (a wall
    with a ledge ~one floor-jump apex high).  The exact surface point AT the
    riser wall is an ``is_edge`` "phantom" the run-arc sweep never *lands* on (an
    approaching ninja stops ~a ninja-radius short and snaps to the node before
    it), so launching only from there yields a trick edge whose ``src`` has no
    incoming normal edges — backward ``d*`` then dead-ends at the launch and the
    level reads as unreachable (the chain-composition bug).  ``approach_sweep``
    therefore tries a few short approach-drive lengths so the trick *also* fires
    from a forward-reachable node set back from the wall (the drive-in is baked
    into the recorded tape, keeping ``src`` ↔ ``tape`` self-consistent).
    ``require_reachable_src`` then drops launches with no incoming normal edge
    (the spawn start is exempt — it is a source, reached via ``fast_reset``).

    NOTE / status: reproducing a firing corner on arbitrary built geometry is
    still geometry-sensitive — see the handoff doc.
    """
    import numpy as np

    by_src: dict = {}
    zero_dstar = np.zeros(graph.n_nodes, dtype=np.float32)  # so snapping isn't filtered

    # Forward-reachability of a launch node: does any normal edge LAND on it?
    # (A trick edge whose src is never landed-on cannot be routed *into*.)
    incoming_nodes = (
        set(int(d) for d in graph.edges_dst.tolist())
        if (require_reachable_src and graph.edges_dst is not None
            and len(graph.edges_dst)) else None
    )

    # Launch candidates: (reset_fn, launch_xy, src_node, approaches, is_spawn).
    # FLOOR nodes inject grounded and sweep a short approach so the trick fires
    # from a reachable node, not just the wall-edge phantom; the spawn runs from
    # fast_reset with the adaptive settle (its launch is often NOT a surface
    # node, as on di-required).
    candidates = []
    for pidx, sp in enumerate(graph.surface_points):
        if sp.surface_type != SURFACE_FLOOR:
            continue
        src = node_id(pidx, velocity_to_bin(0.0, SURFACE_FLOOR), graph.max_vel_bins)
        if incoming_nodes is not None and src not in incoming_nodes:
            continue  # unreachable launch — backward d* would dead-end here
        candidates.append(((lambda s=sp: _inject_grounded(sim, s)),
                           (sp.x, sp.y), src, tuple(approach_sweep), False))
    if spawn_candidate:
        sim.fast_reset()
        spawn_xy = sim.get_ninja_pos()
        ssrc = find_nearest_start_node(
            graph, spawn_xy[0], spawn_xy[1], 0.0, 0.0, 1, zero_dstar)
        if ssrc is not None:
            candidates.append((sim.fast_reset, tuple(spawn_xy), int(ssrc),
                               (None,), True))

    for reset_fn, launch_xy, src, approaches, _is_spawn in candidates:
        for d in dirs:
            for hold in holds:
                # Try each approach length until the corner fires; the first
                # firing approach's tape (drive-in + trick) defines this edge.
                ex = None
                for ap in approaches:
                    reset_fn()
                    cand_ex = execute_cdj(sim, approach_dir=int(d),
                                          hold_frames=int(hold), approach_frames=ap,
                                          bail_if_no_corner=True)
                    if cand_ex.corner_found and (not cand_ex.result.died) \
                            and cand_ex.result.climb_px >= min_climb:
                        ex = cand_ex
                        break
                if ex is None:
                    continue
                # Pick the post-trick drift that best lands / reaches a goal.
                best = None
                for drift in post_drifts:
                    sw_f, ex_f, land, lvx, _lst, dtape = _measure_trick(
                        sim, reset_fn, ex.tape, switch_xy, exit_xy,
                        switch_range, exit_range, drift, post_fall_cap)
                    dst = None
                    if land is not None:
                        ld = find_nearest_start_node(
                            graph, land[0], land[1], lvx, 0.0, 1, zero_dstar)
                        dst = int(ld) if ld is not None else None
                    reaches_goal = (sw_f is not None) or (ex_f is not None)
                    score = (reaches_goal, dst is not None, ex.result.climb_px)
                    cand = CDJTrickEdge(
                        src_node=src, launch_xy=launch_xy, approach_dir=int(d),
                        hold_frames=int(hold), tape=list(ex.tape),
                        climb_px=ex.result.climb_px,
                        apex_xy=(launch_xy[0], ex.result.peak_y),
                        dst_node=dst, landing_xy=land, landing_tape=dtape,
                        switch_contact_frame=sw_f, exit_contact_frame=ex_f)
                    if best is None or score > best[0]:
                        best = (score, cand)
                if best is None:
                    continue
                prev = by_src.get(src)
                if prev is None or best[0] > prev[0]:
                    by_src[src] = best
    return [v[1] for v in by_src.values()]
