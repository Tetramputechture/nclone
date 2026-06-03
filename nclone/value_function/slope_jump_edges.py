"""Slope-jump trick *edges* for the lattice planner.

Promotes the slope-jump chain (``slope_jump_primitive.execute_slope_jump``) from a
maneuver into a first-class **trick edge** the global ``d*`` can route through —
the same promotion ``cdj_edges`` does for the CDJ.  The base enumerator only
emits a single continuous-hold jump per node, so it captures a SINGLE slope jump
but never the **chained re-contact** bounce (jump → land on the slope → jump
again) that converts a slope into vertical height (perp branch) or slope-direction
momentum (downhill branch).  Each edge bundles the whole chain into one verbatim
tape, exactly like a CDJ edge.

Launch candidates are **all slope FLOOR nodes** — any floor node whose surface
normal is non-vertical (``|normal_x| > eps``).  That naturally spans every slope
type the surface extractor produces: 45° slopes (tiles 6-9), mild ~27° slopes
(18-25), steep ~63° slopes (26-33) and the curved convex/concave arcs sampled as
FLOOR (tiles 10-17, tangential model only).

Insertion + ``d*`` recompute + executor replay live in ``tools/lattice_audit``
(``_append_slope_jump_trick_edges``), gated off by default (``NCLONE_SLOPE_JUMP_MODES``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .surface_extraction import SURFACE_FLOOR
from .surface_graph import SurfaceGraph, node_id, velocity_to_bin
from .route_extraction import find_nearest_start_node
from .cdj_edges import _measure_trick  # generic: replay tape, drift, find landing+goal
from .slope_jump_primitive import execute_slope_jump

Input = Tuple[int, int]

# Slope-jump chain families to sweep.  Each entry is the per-hop direction list
# handed to ``execute_slope_jump``'s ``jump_dirs``.  ``+1``/``-1`` select the
# downhill (momentum) vs perp (height) branch depending on slope orientation and
# approach direction; the sweep over ``dirs`` (approach) covers both slope sides.
DEFAULT_CHAINS: Tuple[Tuple[int, ...], ...] = (
    (1,), (-1,),            # single jump (forward / perp)
    (1, 1), (1, -1),        # double: momentum-build / momentum-then-height
    (-1, 1), (1, 1, 1),     # height-then-momentum / triple momentum
)


@dataclass
class SlopeJumpTrickEdge:
    """A slope-jump chain as a routable lattice edge (mirrors CDJTrickEdge)."""

    src_node: int
    launch_xy: Tuple[float, float]
    approach_dir: int
    chain: Tuple[int, ...]
    hold_frames: int
    tape: List[Input]                 # slope-jump chain inputs (approach + hops)
    climb_px: float
    apex_xy: Tuple[float, float]
    dst_node: Optional[int] = None    # surface node landed on after the chain
    landing_xy: Optional[Tuple[float, float]] = None
    landing_tape: List[Input] = None  # post-chain drift inputs to reach dst_node
    switch_contact_frame: Optional[int] = None
    exit_contact_frame: Optional[int] = None
    # Tagged so the executor (lattice_audit) and edge_risk recognise it as a
    # slope-jump trick (re-derive adaptively from the delivery state on replay).
    launch_mode: str = "slope_jump"

    @property
    def full_tape(self) -> List[Input]:
        """The complete executable tape (chain + the drift that lands it)."""
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


def enumerate_slope_jump_edges(
    graph: SurfaceGraph,
    sim,
    switch_xy: Tuple[float, float],
    exit_xy: Tuple[float, float],
    *,
    dirs: Sequence[int] = (1, -1),
    chains: Sequence[Sequence[int]] = DEFAULT_CHAINS,
    holds: Sequence[int] = (10, 35),
    approach_sweep: Sequence[int] = (4, 10, 18),
    min_climb: float = 40.0,
    slope_eps: float = 0.05,
    switch_range: float = 16.0,
    exit_range: float = 22.0,
    post_drifts: Sequence[int] = (0, 1, -1),
    post_fall_cap: int = 90,
    require_reachable_src: bool = True,
) -> List[SlopeJumpTrickEdge]:
    """Find slope-jump trick edges across all slope FLOOR nodes.

    For every reachable slope FLOOR node (``|normal_x| > slope_eps``), runs the
    adaptive slope-jump chain on the native ``sim`` for each (approach dir, chain,
    hold, approach-drive) combo; keeps those that fire and climb at least
    ``min_climb`` px.  For each, records goal air-contacts and the surface it
    lands on (trying a few post-chain drift directions, preferring a landing that
    reaches a goal).  At most one edge per launch node (best — reaches a goal,
    else has a landing, else highest climb).

    ``require_reachable_src`` drops launches with no incoming normal edge (the
    backward ``d*`` would otherwise dead-end at the launch — the chain-composition
    bug guarded against in ``cdj_edges``).  Additive and off by default at the
    audit layer: a map with no firing slope gets no edges and the pipeline is
    unchanged.
    """
    import numpy as np

    by_src: dict = {}
    zero_dstar = np.zeros(graph.n_nodes, dtype=np.float32)  # so snapping isn't filtered

    incoming_nodes = (
        set(int(d) for d in graph.edges_dst.tolist())
        if (require_reachable_src and graph.edges_dst is not None
            and len(graph.edges_dst)) else None
    )

    # Launch candidates: every slope FLOOR node (all slope types) that is
    # forward-reachable (some normal edge lands on it), at zero velocity (the
    # approach drive in the recorded tape builds the along-slope speed).
    candidates = []
    for pidx, sp in enumerate(graph.surface_points):
        if sp.surface_type != SURFACE_FLOOR:
            continue
        if abs(float(sp.normal_x)) <= slope_eps:
            continue  # flat floor — the base jump primitive already covers it
        src = node_id(pidx, velocity_to_bin(0.0, SURFACE_FLOOR), graph.max_vel_bins)
        if incoming_nodes is not None and src not in incoming_nodes:
            continue
        candidates.append(((lambda s=sp: _inject_grounded(sim, s)),
                           (float(sp.x), float(sp.y)), src))

    for reset_fn, launch_xy, src in candidates:
        for adir in dirs:
            for chain in chains:
                for hold in holds:
                    # First (approach, ...) combo that fires with enough climb
                    # defines the edge; the approach drive is baked into the tape.
                    ex = None
                    for ap in approach_sweep:
                        reset_fn()
                        cand = execute_slope_jump(
                            sim, approach_dir=int(adir), jump_dirs=tuple(chain),
                            hold_frames=int(hold), approach_frames=int(ap),
                            bail_if_no_fire=True)
                        if (cand.fired and not cand.result.died
                                and cand.result.climb_px >= min_climb):
                            ex = cand
                            break
                    if ex is None:
                        continue
                    # Pick the post-chain drift that best lands / reaches a goal.
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
                        edge = SlopeJumpTrickEdge(
                            src_node=src, launch_xy=launch_xy,
                            approach_dir=int(adir), chain=tuple(chain),
                            hold_frames=int(hold), tape=list(ex.tape),
                            climb_px=ex.result.climb_px,
                            apex_xy=(launch_xy[0], ex.result.peak_y),
                            dst_node=dst, landing_xy=land, landing_tape=dtape,
                            switch_contact_frame=sw_f, exit_contact_frame=ex_f)
                        if best is None or score > best[0]:
                            best = (score, edge)
                    if best is None:
                        continue
                    prev = by_src.get(src)
                    if prev is None or best[0] > prev[0]:
                        by_src[src] = best
    return [v[1] for v in by_src.values()]
