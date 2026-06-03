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

from .surface_extraction import SURFACE_FLOOR, SURFACE_WALL
from .surface_graph import (
    SurfaceGraph,
    node_id,
    velocity_bin_centers,
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
    # "floor_jump" = settle + floor-jump up the wall (di-required path); "carry" =
    # fire the trick from a node where the ninja already arrives rising against the
    # wall (no settle/launch jump).  Tells the executor whether to settle to rest
    # (floor_jump) or arrive with upward velocity and re-derive the core (carry).
    launch_mode: str = "floor_jump"

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


def _inject_wall_carry(sim, sp, vy: float, wall_sign: float) -> None:
    """Inject a wall-contact state WITH a (typically upward, vy<0) velocity.

    Mirrors the canonical wall/slide-node injection in ``tile_cache/builder.py``:
    ``inject_ninja_for_value_iteration_wall_slide`` takes a wall-normal sign
    (+1 = normal points right).  This is the "carry" launch: the ninja arrives
    rising against the wall, so the CDJ fires without its own floor jump.
    """
    sim.fast_reset()
    sim.inject_ninja_for_value_iteration_wall_slide(
        float(sp.x), float(sp.y), float(vy), float(wall_sign),
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
    enable_floor: bool = True,
    enable_carry: bool = False,
    fire_on_buffer_intersect: bool = False,
    chain_fixpoint: bool = False,
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

    ``enable_carry`` adds a second, generalized launch family: SURFACE_WALL nodes
    carrying UPWARD velocity (the negative wall bins).  A floor jump from rest is
    only ONE way to reach a corner rising — any reachable node where the ninja
    arrives moving up against the wall (off a prior wall jump, a prior CDJ, or
    momentum) can fire the trick.  These candidates inject the wall-rising state
    and run the primitive in ``launch_mode="carry"`` (no settle / no floor jump),
    and the resulting edges are tagged ``launch_mode="carry"`` so the executor
    delivers the ninja with upward velocity instead of settling it to rest.  Off
    by default: with it off the candidate set and returned edges are unchanged.

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
    # Candidate tuple: (reset_fn, launch_xy, src, approaches, is_spawn, launch_mode).
    candidates = []
    if enable_floor:
        for pidx, sp in enumerate(graph.surface_points):
            if sp.surface_type != SURFACE_FLOOR:
                continue
            src = node_id(pidx, velocity_to_bin(0.0, SURFACE_FLOOR), graph.max_vel_bins)
            if incoming_nodes is not None and src not in incoming_nodes:
                continue  # unreachable launch — backward d* would dead-end here
            candidates.append(((lambda s=sp: _inject_grounded(sim, s)),
                               (sp.x, sp.y), src, tuple(approach_sweep), False,
                               "floor_jump"))
        if spawn_candidate:
            sim.fast_reset()
            spawn_xy = sim.get_ninja_pos()
            ssrc = find_nearest_start_node(
                graph, spawn_xy[0], spawn_xy[1], 0.0, 0.0, 1, zero_dstar)
            if ssrc is not None:
                candidates.append((sim.fast_reset, tuple(spawn_xy), int(ssrc),
                                   (None,), True, "floor_jump"))
    if enable_carry:
        # Generalized launch: reachable SURFACE_WALL nodes with UPWARD velocity
        # (the negative wall bins).  Inject the rising wall state and fire the
        # trick in carry mode (no settle / no floor jump).
        wall_centers = velocity_bin_centers(SURFACE_WALL)
        up_bins = [b for b, c in enumerate(wall_centers) if float(c) < 0.0]
        for pidx, sp in enumerate(graph.surface_points):
            if sp.surface_type != SURFACE_WALL:
                continue
            wall_sign = 1.0 if sp.normal_x > 0 else -1.0
            for b in up_bins:
                src = node_id(pidx, b, graph.max_vel_bins)
                # In fixpoint mode we KEEP carry candidates whose src is not yet
                # base-reachable: a higher block's wall becomes reachable only via
                # a CDJ off the block below, so its candidate must be (re)tried as
                # the chain composes.  Outside fixpoint mode the original
                # has-incoming gate stands (byte-identical).
                if (not chain_fixpoint) and incoming_nodes is not None \
                        and src not in incoming_nodes:
                    continue  # no normal edge lands here rising — not reachable
                vy = float(wall_centers[b])
                candidates.append((
                    (lambda s=sp, vyy=vy, ws=wall_sign:
                        _inject_wall_carry(sim, s, vyy, ws)),
                    (sp.x, sp.y), src, (0,), False, "carry"))

    def _process(cand_tuple):
        """Run one launch candidate; return (score, CDJTrickEdge) or None."""
        reset_fn, launch_xy, src, approaches, _is_spawn, launch_mode = cand_tuple
        out = None
        for d in dirs:
            for hold in holds:
                # Try each approach length until the corner fires; the first
                # firing approach's tape (drive-in + trick) defines this edge.
                # Carry candidates ignore ``approach_frames`` (no settle/launch).
                ex = None
                for ap in approaches:
                    reset_fn()
                    cand_ex = execute_cdj(sim, approach_dir=int(d),
                                          hold_frames=int(hold), approach_frames=ap,
                                          bail_if_no_corner=True,
                                          launch_mode=launch_mode,
                                          fire_on_buffer_intersect=fire_on_buffer_intersect)
                    if cand_ex.corner_found and (not cand_ex.result.died) \
                            and cand_ex.result.climb_px >= min_climb:
                        ex = cand_ex
                        break
                if ex is None:
                    continue
                # Pick the post-trick drift that best lands / reaches a goal.
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
                        switch_contact_frame=sw_f, exit_contact_frame=ex_f,
                        launch_mode=launch_mode)
                    if out is None or score > out[0]:
                        out = (score, cand)
        return out

    if not chain_fixpoint:
        for cand_tuple in candidates:
            res = _process(cand_tuple)
            if res is None:
                continue
            src = cand_tuple[2]
            prev = by_src.get(src)
            if prev is None or res[0] > prev[0]:
                by_src[src] = res
        return [v[1] for v in by_src.values()]

    # --- Fixpoint chain enumeration ----------------------------------------
    # A higher floating block's launch node becomes reachable only AFTER a CDJ
    # off the block below lands on it.  So iterate: process the candidates whose
    # src is currently reachable-from-spawn (over base edges + the trick edges
    # found so far), splice the new trick edges into the reachability graph, and
    # repeat until no newly-reachable candidate remains.  This is what lets the
    # carry chain climb rung-by-rung up to the goal.
    import collections
    base_adj = collections.defaultdict(list)
    if graph.edges_src is not None and len(graph.edges_src):
        for s, dnode in zip(graph.edges_src.tolist(), graph.edges_dst.tolist()):
            base_adj[int(s)].append(int(dnode))
    sim.fast_reset()
    spawn_xy = sim.get_ninja_pos()
    spawn_node = find_nearest_start_node(
        graph, spawn_xy[0], spawn_xy[1], 0.0, 0.0, 1, zero_dstar)
    extra_adj = collections.defaultdict(list)  # trick edges found so far

    def _reachable():
        seed = [int(spawn_node)] if spawn_node is not None else []
        # Floor candidates are grounded launches that the base graph already
        # routes into; treat any base-incoming node as a seed too so the chain
        # can start from the existing surface network.
        reached = set(seed)
        stack = list(seed)
        while stack:
            u = stack.pop()
            for w in base_adj.get(u, ()):
                if w not in reached:
                    reached.add(w); stack.append(w)
            for w in extra_adj.get(u, ()):
                if w not in reached:
                    reached.add(w); stack.append(w)
        return reached

    def _reached_positions(reached):
        pos = []
        for n in reached:
            pid = n // graph.max_vel_bins
            if 0 <= pid < len(graph.surface_points):
                sp = graph.surface_points[pid]
                pos.append((sp.x, sp.y))
        return pos

    # A carry launch is on a WALL node, but the chain DELIVERS the ninja to the
    # block's TOP (a grounded landing one node away).  So a carry candidate is
    # eligible once ANY reached node lands on the same block — i.e. within one
    # tile of its launch position — not only when its exact rising-wall node has
    # a base in-edge.  This is what carries the chain from a landing onto the
    # next rung's clincher.
    _SAME_BLOCK_R2 = 28.0 * 28.0

    pending = list(candidates)
    processed_src = set()
    rounds = 0
    while pending and rounds < 40:
        rounds += 1
        reached = _reachable()
        reached_pos = _reached_positions(reached)
        active, rest = [], []
        for ct in pending:
            src = ct[2]
            lx, ly = ct[1]
            on_reached_block = any(
                (px - lx) ** 2 + (py - ly) ** 2 <= _SAME_BLOCK_R2
                for (px, py) in reached_pos
            )
            # Spawn/floor launches and any reachable src/block are eligible.
            if src in reached or on_reached_block \
                    or src in incoming_nodes_or_empty(incoming_nodes) \
                    or ct[5] == "floor_jump":
                active.append(ct)
            else:
                rest.append(ct)
        if not active:
            break
        pending = rest
        progressed = False
        for ct in active:
            src = ct[2]
            if src in processed_src:
                continue
            processed_src.add(src)
            res = _process(ct)
            if res is None:
                continue
            prev = by_src.get(src)
            if prev is None or res[0] > prev[0]:
                by_src[src] = res
            dst = res[1].dst_node
            if dst is not None:
                extra_adj[int(src)].append(int(dst))
                progressed = True
        if not progressed:
            break
    return [v[1] for v in by_src.values()]


def incoming_nodes_or_empty(incoming_nodes):
    return incoming_nodes if incoming_nodes is not None else frozenset()
