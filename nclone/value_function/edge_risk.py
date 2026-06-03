"""Soft, bounded edge-cost reweighting for the lattice planner.

The backward-Dijkstra ``d*`` over the surface graph minimises edge weights, which
are *frames* — i.e. it optimises **speed** only.  We want the planner to optimise
**completion > lack-of-loops > speed** (in that priority order) so the ``d*``
gradient guides an imperfect RL policy toward routes it can actually finish: the
benchmark's binding failures were *deaths from mine-drift* and *stalls on hard
jumps* (memory ``planner-benchmark-execution-bound``), not pathing.

We encode this as a **soft, bounded** multiplicative reweight applied as a
graph-level transform *after* enumeration (so it is identical regardless of the
Python/native enumeration backend, and ``w_risk == 0`` reproduces the pure-frames
weights exactly)::

    effective = min(frames * (1 + w_risk * risk + w_loop * loop),
                    frames * risk_cap)

``risk in [0, 1]`` aggregates:
  * **mine proximity** — how close the arc's launch/landing surface points pass to
    a deadly mine (a soft avoidance gradient *on top of* the hard lethal-arc
    pruning + blocked-node masking the enumerator/Dijkstra already apply); and
  * **execution difficulty** — runs/slides are ~free, jumps scale with hold
    duration, trick edges (CDJ) are hardest (these are exactly where the
    hand-coded executor stalls).

``loop in [0, 1]`` (default off, ``w_loop == 0``) penalises arcs that move *away*
from the goal — a per-edge detour proxy.  The *guaranteed* loop removal happens at
route extraction (``route_extraction.dedup_position_loops``); the ``d*`` potential
is loop-neutral under PBRS, so no field-level loop term is required for the reward.

The cap keeps ``d_max`` from blowing up and collapsing the PBRS ``phi``
normalisation range (the same concern that motivated the legacy
``_MINE_INFLATION_MAX_FACTOR`` cap in ``edge_enumeration``).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..constants.physics_constants import (
    MAX_HOR_SPEED,
    MAX_JUMP_DURATION,
    NINJA_RADIUS,
    TOGGLE_MINE_RADIUS_TOGGLED,
)
from .surface_extraction import (
    SURFACE_FLOOR,
    SURFACE_LAUNCH_PAD,
    SURFACE_SLIDE,
    SURFACE_WALL,
)
from .surface_graph import SurfaceGraph, velocity_bin_centers

# Lethal contact distance (ninja-center to mine-center); inside this the node is
# already blocked / the arc is pruned, so risk saturates to 1 there.
_MINE_KILL_RADIUS: float = NINJA_RADIUS + TOGGLE_MINE_RADIUS_TOGGLED  # 14 px
# Support radius (px of clearance beyond the kill radius) over which mine risk
# decays linearly to 0.  Matches ``edge_enumeration._MINE_CHOMP_EPSILON`` so the
# avoidance bubble has the same reach as the legacy CHOMP bowl.
_MINE_RISK_EPSILON: float = 32.0

# Execution-difficulty risk contributions.
_JUMP_RISK_MAX: float = 0.5     # a max-duration jump contributes up to this
_TRICK_RISK: float = 0.8        # CDJ / scripted trick edges are hardest
# A "carry" CDJ fires mid-flight from an arrived rising state (no settle to rest),
# so it is even more frame-sensitive than a floor-jump-launched CDJ: d* should
# prefer a floor-jump CDJ when both reach the same goal.
_TRICK_CARRY_RISK: float = 0.92
# A slope-jump chain depends on the live xspeed/hor_input at each hop AND on
# re-contact timing across multiple bounces — frame-sensitive like a carry, so
# d* prefers a plain arc when one reaches the same goal.
_TRICK_SLOPE_RISK: float = 0.9

# Momentum-preservation risk contributions.  These two terms target the
# "jump the instant you touch a wall" failure mode that crowds out velocity-
# building (curved-tile / running) routes once cheap wall-climb chain edges
# exist in the graph (see memory ``wall-jump-chaining-recontact``):
#   * a jump LANDING on a wall node moving UPWARD (vy < 0) is a climb chain-link
#     — penalise it so a route with fewer such links is preferred; and
#   * a jump LAUNCHED from a slow floor velocity bin threw away the chance to
#     build speed first — penalise it so run-then-jump becomes competitive.
# Both scale with ``w_risk`` (a gentle, bounded nudge) and never on their own
# block a climb that is genuinely the only / cheapest way up.
_WALL_CHAIN_RISK: float = 0.6           # upward wall re-contact chain-link
_LOW_LAUNCH_SPEED_RISK_MAX: float = 0.4  # max extra risk for a standstill floor jump

# Defaults for the planner layer (LatticeAudit / precompute).  ``w_risk`` is
# deliberately gentle so existing routes are preserved except where risk is
# genuinely high; ``w_loop`` is off until validated.
DEFAULT_W_RISK: float = 0.5
DEFAULT_W_LOOP: float = 0.0
DEFAULT_RISK_CAP: float = 8.0

# Additive per-jump-event cost (frames) applied on top of the multiplicative
# risk factor.  Unlike the multiplicative terms it scales with the *count* of
# jumps, not arc length, so a route of many short wall-jumps is dearer than one
# momentum-carrying jump.  This is the primary lever for "fewer gratuitous
# jumps"; tune it per level.  The low-level ``apply_risk_weighting`` defaults to
# 0.0 (preserving the pure-frames invariant for w_risk==0 callers/tests); the
# planner layer (LatticeAudit) opts in to ``DEFAULT_JUMP_TAX``.
#
# The default is sized empirically: a wall-climb's frame advantage is large, so
# a small tax is a no-op.  Measured on official ``250 pipeworks`` (chosen path,
# w_risk=0.5): jump_tax 2 -> 27 jumps / 24 upward-wall chain-links (≈ unchanged
# from the pure-frames 25 / 23); jump_tax 8 -> 14 / 11; 20 -> 12 / 9.  ~8 is the
# knee where the planner trades climb-spam for momentum-carrying arcs without
# over-penalising climbs that are genuinely the only way up (those have no
# cheaper alternative, so their route is unchanged — only the d* cost inflates).
DEFAULT_JUMP_TAX: float = 8.0


def _point_xy(graph: SurfaceGraph) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-surface-point (x, y) arrays indexed by point index."""
    xs = np.fromiter((sp.x for sp in graph.surface_points), dtype=np.float64,
                     count=graph.n_points)
    ys = np.fromiter((sp.y for sp in graph.surface_points), dtype=np.float64,
                     count=graph.n_points)
    return xs, ys


def _nearest_mine_clearance(
    pt_x: np.ndarray, pt_y: np.ndarray,
    deadly_mines: np.ndarray,
) -> np.ndarray:
    """Per-point clearance (px) to the nearest deadly mine, minus kill radius.

    Negative inside the kill radius.  ``+inf`` when there are no deadly mines.
    """
    if deadly_mines.shape[0] == 0:
        return np.full(pt_x.shape[0], np.inf, dtype=np.float64)
    # (n_points, n_mines) distance matrix — points are few thousand, mines tens.
    dx = pt_x[:, None] - deadly_mines[None, :, 0]
    dy = pt_y[:, None] - deadly_mines[None, :, 1]
    min_dist = np.sqrt(dx * dx + dy * dy).min(axis=1)
    return min_dist - _MINE_KILL_RADIUS


def _bin_velocity(stype_per_edge: np.ndarray, bin_per_edge: np.ndarray) -> np.ndarray:
    """Vectorised per-edge velocity-bin *center* lookup (vx for floors, vy for
    walls/slides).  Loops over the handful of surface types and gathers from
    each type's bin-center schedule; out-of-range bins clip to the extents."""
    vel = np.zeros(stype_per_edge.shape[0], dtype=np.float64)
    for st in (SURFACE_FLOOR, SURFACE_WALL, SURFACE_SLIDE, SURFACE_LAUNCH_PAD):
        mask = stype_per_edge == st
        if not mask.any():
            continue
        centers = velocity_bin_centers(st)
        b = np.clip(bin_per_edge[mask], 0, len(centers) - 1)
        vel[mask] = centers[b]
    return vel


def compute_edge_risk(
    graph: SurfaceGraph,
    deadly_mines: Optional[Sequence[Tuple[float, float]]] = None,
) -> np.ndarray:
    """Per-edge completion risk in ``[0, 1]`` (higher = harder to finish safely).

    Combines mine-proximity risk (launch & landing clearance to the nearest
    deadly mine) with execution-difficulty risk (jump hold duration / trick).
    Operates purely on the populated graph arrays, so it is identical regardless
    of which enumeration backend produced the edges.
    """
    n = int(len(graph.edges_src))
    risk = np.zeros(n, dtype=np.float32)
    if n == 0:
        return risk

    max_vb = graph.max_vel_bins
    src_pt = (graph.edges_src.astype(np.int64) // max_vb)
    dst_pt = (graph.edges_dst.astype(np.int64) // max_vb)

    # --- Mine-proximity risk (soft avoidance gradient) ---------------------
    mines = np.asarray(deadly_mines if deadly_mines else [], dtype=np.float64)
    if mines.ndim == 2 and mines.shape[0] > 0:
        pt_x, pt_y = _point_xy(graph)
        clearance_pt = _nearest_mine_clearance(pt_x, pt_y, mines)
        # The arc grazes the closer of its launch / landing point.
        edge_clear = np.minimum(clearance_pt[src_pt], clearance_pt[dst_pt])
        # Linear falloff: 1 at/inside the kill radius -> 0 at >= eps clearance.
        mine_risk = np.clip(
            (_MINE_RISK_EPSILON - edge_clear) / _MINE_RISK_EPSILON, 0.0, 1.0
        ).astype(np.float32)
        risk = np.maximum(risk, mine_risk)

    # --- Execution-difficulty risk -----------------------------------------
    is_jump = None
    if graph.edges_jump_hold is not None:
        jh = graph.edges_jump_hold.astype(np.float32)
        is_jump = jh > 0.0
        jump_risk = np.clip(jh / float(MAX_JUMP_DURATION), 0.0, 1.0) * _JUMP_RISK_MAX
        risk = np.clip(risk + jump_risk, 0.0, 1.0)

    # --- Momentum-preservation risk (wall-chain + low-launch-speed) --------
    # Computed only over jump edges; runs/slides keep their (zero) risk.
    if is_jump is not None and bool(is_jump.any()):
        stype = np.fromiter((sp.surface_type for sp in graph.surface_points),
                            dtype=np.int64, count=graph.n_points)
        src_pt = graph.edges_src.astype(np.int64) // max_vb
        dst_pt = graph.edges_dst.astype(np.int64) // max_vb
        src_bin = graph.edges_src.astype(np.int64) % max_vb
        dst_bin = graph.edges_dst.astype(np.int64) % max_vb
        src_stype = stype[src_pt]
        dst_stype = stype[dst_pt]

        # Wall-jump chain-link: a jump landing on a wall/slide node whose binned
        # velocity is upward (vy < 0) — i.e. "bounce up the wall on contact".
        dst_wall = (dst_stype == SURFACE_WALL) | (dst_stype == SURFACE_SLIDE)
        dst_vel = _bin_velocity(dst_stype, dst_bin)
        chain = is_jump & dst_wall & (dst_vel < 0.0)
        if chain.any():
            risk[chain] = np.clip(risk[chain] + _WALL_CHAIN_RISK, 0.0, 1.0)

        # Low-launch-speed: a floor jump from a slow |vx| bin (didn't run first).
        # Full sprint -> 0 extra risk; standstill -> _LOW_LAUNCH_SPEED_RISK_MAX.
        src_floor = src_stype == SURFACE_FLOOR
        src_vel = _bin_velocity(src_stype, src_bin)
        speed_frac = np.clip(np.abs(src_vel) / float(MAX_HOR_SPEED), 0.0, 1.0)
        low_speed = np.where(
            is_jump & src_floor,
            _LOW_LAUNCH_SPEED_RISK_MAX * (1.0 - speed_frac), 0.0,
        ).astype(np.float32)
        risk = np.clip(risk + low_speed, 0.0, 1.0)

    # Trick edges (CDJ et al.) are the hardest to execute frame-perfectly;
    # "carry" CDJs (no settle, fired from a rising arrival) are harder still.
    trick = getattr(graph, "trick_tapes", None)
    if trick:
        modes = getattr(graph, "trick_modes", None) or {}
        idx = np.fromiter((i for i in trick.keys() if 0 <= i < n),
                          dtype=np.int64)
        if idx.size:
            risk[idx] = np.maximum(risk[idx], _TRICK_RISK)
        carry_idx = np.fromiter(
            (i for i, m in modes.items() if m == "carry" and 0 <= i < n),
            dtype=np.int64)
        if carry_idx.size:
            risk[carry_idx] = np.maximum(risk[carry_idx], _TRICK_CARRY_RISK)
        slope_idx = np.fromiter(
            (i for i, m in modes.items() if m == "slope_jump" and 0 <= i < n),
            dtype=np.int64)
        if slope_idx.size:
            risk[slope_idx] = np.maximum(risk[slope_idx], _TRICK_SLOPE_RISK)

    return np.clip(risk, 0.0, 1.0)


def compute_edge_loop(
    graph: SurfaceGraph,
    goal_xy: Optional[Tuple[float, float]],
) -> np.ndarray:
    """Per-edge detour proxy in ``[0, 1]``: how much the arc moves *away* from
    ``goal_xy`` (landing further from the goal than the launch).  ``0`` when no
    goal is supplied.  Normalised by the mine-risk epsilon scale for a bounded,
    dimensionless term consistent with ``risk``."""
    n = int(len(graph.edges_src))
    loop = np.zeros(n, dtype=np.float32)
    if n == 0 or goal_xy is None:
        return loop
    max_vb = graph.max_vel_bins
    src_pt = (graph.edges_src.astype(np.int64) // max_vb)
    dst_pt = (graph.edges_dst.astype(np.int64) // max_vb)
    pt_x, pt_y = _point_xy(graph)
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    d_launch = np.hypot(pt_x[src_pt] - gx, pt_y[src_pt] - gy)
    d_land = np.hypot(pt_x[dst_pt] - gx, pt_y[dst_pt] - gy)
    # Only penalise net regress (moving away); normalise by ~one jump's reach.
    regress = np.maximum(0.0, d_land - d_launch)
    scale = 8.0 * _MINE_RISK_EPSILON  # ~256 px ≈ a long jump's horizontal reach
    return np.clip(regress / scale, 0.0, 1.0).astype(np.float32)


def apply_risk_weighting(
    graph: SurfaceGraph,
    deadly_mines: Optional[Sequence[Tuple[float, float]]] = None,
    *,
    w_risk: float = DEFAULT_W_RISK,
    w_loop: float = DEFAULT_W_LOOP,
    risk_cap: float = DEFAULT_RISK_CAP,
    jump_tax: float = 0.0,
    goal_xy: Optional[Tuple[float, float]] = None,
) -> dict:
    """Reweight ``graph.edges_weight`` in place: completion > loops > speed.

    ``effective = min(frames * (1 + w_risk*risk + w_loop*loop), frames*risk_cap)
                  + jump_tax * (edge is a jump)``.

    The multiplicative term scales with arc length (penalising long/risky arcs);
    the additive ``jump_tax`` scales with the *count* of jumps (penalising routes
    that chain many short jumps — e.g. wall-climb spam — over momentum-carrying
    ones).  A no-op (returns immediately) when ``w_risk == 0``, ``w_loop == 0``
    AND ``jump_tax == 0`` so the pure-frames behaviour — and byte-for-byte
    cache/parity — is preserved.

    Returns a small stats dict for logging (mean/max risk, fraction reweighted).
    """
    n = int(len(graph.edges_weight))
    if n == 0 or (w_risk == 0.0 and w_loop == 0.0 and jump_tax == 0.0):
        return {"reweighted": False, "n_edges": n}

    risk = (compute_edge_risk(graph, deadly_mines) if w_risk != 0.0
            else np.zeros(n, dtype=np.float32))
    loop = (compute_edge_loop(graph, goal_xy) if w_loop != 0.0
            else np.zeros(n, dtype=np.float32))

    base = graph.edges_weight.astype(np.float32)
    factor = 1.0 + w_risk * risk + w_loop * loop
    factor = np.minimum(factor, float(risk_cap))
    # Never reduce a weight below its frame cost (factor >= 1 by construction).
    eff = base * factor

    n_jump_taxed = 0
    if jump_tax > 0.0 and graph.edges_jump_hold is not None:
        is_jump = graph.edges_jump_hold.astype(np.int64) > 0
        eff = eff + (float(jump_tax) * is_jump.astype(np.float32))
        n_jump_taxed = int(is_jump.sum())
    graph.edges_weight = eff.astype(np.float32)

    nz = int(np.count_nonzero(factor > 1.0 + 1e-6))
    return {
        "reweighted": True,
        "n_edges": n,
        "n_reweighted": nz,
        "frac_reweighted": float(nz) / float(n),
        "mean_risk": float(risk.mean()),
        "max_risk": float(risk.max()),
        "w_risk": float(w_risk),
        "w_loop": float(w_loop),
        "risk_cap": float(risk_cap),
        "jump_tax": float(jump_tax),
        "n_jump_taxed": n_jump_taxed,
    }
