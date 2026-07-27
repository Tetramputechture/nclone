"""Prize-collecting orienteering / TSP selection for score-maximizing routes.

Score = ``(90*60 - frames + 1 + 120*gold)/60`` == maximize ``120*gold - frames``
(frame units): each collected gold is worth a **120-frame detour credit**
(``GOLD_FRAME_VALUE``). Choosing WHICH gold to collect and in WHAT ORDER, subject
to reaching the exit switch then the exit door, is a prize-collecting orienteering
problem on the physics route graph.

**This module is the SELECTION layer only** — pure numpy, no simulator, no map
loading. It runs on the cheap, approximate lattice surface-graph distances
(``surface_graph.dijkstra_backward`` retargeted to arbitrary goals). Approximate
costs are fine here: they only rank subsets/orders. The caller
(``fast_route_solver.plan_score_route``) re-plans every chosen leg with frame-exact
native physics and keeps the sacrifice-backtrack completion guarantee, so a
mis-estimated cost can at worst yield slightly sub-optimal gold — never an invalid
or non-winning route.

Route model (v1): ``spawn -> [chosen gold, in order] -> switch -> exit`` (all chosen
gold before the switch), matching the physics leg-executor's phase structure. Gold
lying on the switch->exit corridor is still grabbed opportunistically by the physics
legs but is not a planned waypoint (a post-switch gold segment is a deferred
enhancement).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .surface_extraction import SURFACE_FLOOR, SURFACE_WALL
from .surface_graph import (
    dijkstra_backward,
    find_goal_nodes,
    find_nearest_point,
    node_id,
    num_velocity_bins,
)

GOLD_FRAME_VALUE = 120.0  # mirror of reward_constants.GOLD_FRAME_VALUE


# --------------------------------------------------------------------------- snap
def _point_nodes(graph, pt_idx: int) -> List[int]:
    """All velocity-bin node ids at a surface point (mirrors _min_dstar_at)."""
    st = graph.surface_points[pt_idx].surface_type
    return [node_id(pt_idx, vb, graph.max_vel_bins)
            for vb in range(num_velocity_bins(st))]


def snap_gold_to_nodes(graph, gold_xy: Sequence[Tuple[float, float]],
                       max_snap_px: float = 48.0
                       ) -> Tuple[List[Optional[int]], List[List[int]]]:
    """Snap each gold (x, y) to the nearest FLOOR/WALL surface point.

    Gold floats off surfaces so it isn't a graph node; we approximate "ninja at the
    gold" by the nearest surface point's velocity-bin nodes (same approximation the
    lattice heuristic uses). Returns ``(snapped_pt, goal_nodes)``; a gold with no
    surface point within ``max_snap_px`` gets ``(None, [])`` and is excluded from the
    waypoint plan (the physics leg may still sweep it opportunistically)."""
    max2 = max_snap_px * max_snap_px
    snapped: List[Optional[int]] = []
    goal_nodes: List[List[int]] = []
    for (gx, gy) in gold_xy:
        best_pt = None
        best_d2 = float("inf")
        for stype in (SURFACE_FLOOR, SURFACE_WALL):
            pt = find_nearest_point(graph, gx, gy, stype)
            if pt is None:
                continue
            sp = graph.surface_points[pt]
            d2 = (sp.x - gx) ** 2 + (sp.y - gy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_pt = pt
        if best_pt is None or best_d2 > max2:
            snapped.append(None)
            goal_nodes.append([])
        else:
            snapped.append(best_pt)
            goal_nodes.append(_point_nodes(graph, best_pt))
    return snapped, goal_nodes


# ------------------------------------------------------------------- cost matrix
def build_cost_matrix(graph, rev_adj, spawn_xy, gold_goal_nodes: List[List[int]],
                      switch_xy, exit_xy) -> Tuple[np.ndarray, np.ndarray]:
    """Directed pairwise frame-cost matrix over ``[gold_0..n-1, spawn, switch, exit]``.

    Indices: gold ``0..n-1``; ``spawn=n``; ``switch=n+1``; ``exit=n+2``.

    One backward Dijkstra per TARGET (each gold, switch, exit; spawn is source-only,
    so no pass for it) — ``rev_adj`` built once by the caller and shared. Column for
    target ``t``: ``d_t = dijkstra_backward(goal_nodes(t))``; then
    ``C[s][t] = min over s's node set of d_t[node]``. Asymmetric (gravity) — the full
    directed matrix is kept. Unreachable pairs / empty node sets -> inf.

    Returns ``(C, reachable_mask)`` where ``reachable_mask[i]`` is True for gold i
    that has a finite spawn->gold->switch chain (a cheap pre-filter; the solver also
    guards inf)."""
    n = len(gold_goal_nodes)
    SPAWN, SWITCH, EXIT = n, n + 1, n + 2
    m = n + 3

    # Source node sets (which nodes represent "ninja at X").
    src_nodes: List[List[int]] = list(gold_goal_nodes)  # 0..n-1
    sp_pt = None
    for stype in (SURFACE_FLOOR, SURFACE_WALL):
        p = find_nearest_point(graph, spawn_xy[0], spawn_xy[1], stype)
        if p is not None:
            sp_pt = p
            break
    src_nodes.append(_point_nodes(graph, sp_pt) if sp_pt is not None else [])  # spawn
    sw_nodes = find_goal_nodes(graph, switch_xy[0], switch_xy[1], "switch")
    src_nodes.append(sw_nodes)  # switch
    ex_nodes = find_goal_nodes(graph, exit_xy[0], exit_xy[1], "exit")
    src_nodes.append(ex_nodes)  # exit (source set unused, kept for symmetry)

    # Target node sets (columns we run Dijkstra for): every gold, switch, exit.
    tgt_nodes: Dict[int, List[int]] = {}
    for i in range(n):
        tgt_nodes[i] = gold_goal_nodes[i]
    tgt_nodes[SWITCH] = sw_nodes
    tgt_nodes[EXIT] = ex_nodes

    C = np.full((m, m), np.inf, dtype=np.float64)
    for t, gnodes in tgt_nodes.items():
        if not gnodes:
            continue
        d = dijkstra_backward(graph, list(gnodes), rev_adj=rev_adj)  # float32[n_nodes]
        for s in range(m):
            if s == t:
                C[s][t] = 0.0
                continue
            sset = src_nodes[s]
            if not sset:
                continue
            best = float("inf")
            for nid in sset:
                v = float(d[nid])
                if v < best:
                    best = v
            C[s][t] = best

    reachable = np.zeros(n, dtype=bool)
    for i in range(n):
        if np.isfinite(C[SPAWN][i]) and np.isfinite(C[i][SWITCH]):
            reachable[i] = True
    return C, reachable


# ----------------------------------------------------------------- orienteering
def _tour_frames(order: Sequence[int], C: np.ndarray, spawn: int, switch: int,
                 exit_: int) -> float:
    """Frames for spawn -> order -> switch -> exit (inf if any hop unreachable)."""
    prev = spawn
    total = 0.0
    for g in order:
        total += C[prev][g]
        prev = g
    total += C[prev][switch] + C[switch][exit_]
    return total


def _held_karp(C, mask_reach, weights, spawn, switch, exit_, n):
    """Exact prize-collecting DP for small n. Returns best ordered gold list.

    dp[S][j] = min frames spawn -> visit exactly gold-set S -> end at gold j.
    Objective at (S, j): 120*sum(weights in S) - (dp[S][j] + C[j][switch] + C[switch][exit]).
    """
    idx = [i for i in range(n) if mask_reach[i]]
    if not idx:
        return []
    remap = {g: b for b, g in enumerate(idx)}      # gold index -> bit
    k = len(idx)
    NEG = float("-inf")
    INF = float("inf")
    # dp over bitmask of the k reachable golds.
    size = 1 << k
    dp = [[INF] * k for _ in range(size)]
    par = [[-1] * k for _ in range(size)]
    wsum = [0.0] * size
    for b in range(k):
        m1 = 1 << b
        dp[m1][b] = C[spawn][idx[b]]
        wsum[m1] = weights[idx[b]]
    for S in range(size):
        # accumulate weight sum for S (from lowest set bit)
        if wsum[S] == 0.0 and S != 0:
            low = S & (-S)
            wsum[S] = wsum[S ^ low] + weights[idx[low.bit_length() - 1]]
        for j in range(k):
            if not (S & (1 << j)) or dp[S][j] == INF:
                continue
            base = dp[S][j]
            for b in range(k):
                if S & (1 << b):
                    continue
                nS = S | (1 << b)
                nc = base + C[idx[j]][idx[b]]
                if nc < dp[nS][b]:
                    dp[nS][b] = nc
                    par[nS][b] = j
    swx = C[switch][exit_]
    best_val = NEG
    best_S = 0
    best_j = -1
    # empty set (no gold)
    empty = -(C[spawn][switch] + swx)
    if np.isfinite(C[spawn][switch]) and np.isfinite(swx):
        best_val = empty
    for S in range(1, size):
        for j in range(k):
            if dp[S][j] == INF:
                continue
            tail = C[idx[j]][switch]
            if not np.isfinite(tail) or not np.isfinite(swx):
                continue
            val = GOLD_FRAME_VALUE * wsum[S] - (dp[S][j] + tail + swx)
            if val > best_val:
                best_val = val
                best_S = S
                best_j = j
    if best_j < 0:
        return []
    # reconstruct order
    order_rev = []
    S, j = best_S, best_j
    while j >= 0:
        order_rev.append(idx[j])
        pj = par[S][j]
        S ^= (1 << j)
        j = pj
    return list(reversed(order_rev))


def _greedy_2opt(C, mask_reach, weights, spawn, switch, exit_, n, time_budget_s):
    """Cheapest-insertion with a marginal-detour gate, then 2-opt / Or-opt(1)."""
    t0 = time.monotonic()
    order: List[int] = []
    remaining = [i for i in range(n) if mask_reach[i]]

    def insertion_cost(g, pos, seq):
        # cost delta of inserting g at index pos in the gold segment seq
        left = spawn if pos == 0 else seq[pos - 1]
        right = switch if pos == len(seq) else seq[pos]
        if not (np.isfinite(C[left][g]) and np.isfinite(C[g][right])):
            return float("inf")
        base = C[left][right] if np.isfinite(C[left][right]) else 0.0
        return C[left][g] + C[g][right] - base

    # greedy insertion while positive marginal score gain
    while remaining:
        best_gain = 0.0
        best = None  # (g, pos)
        for g in remaining:
            for pos in range(len(order) + 1):
                dcost = insertion_cost(g, pos, order)
                gain = GOLD_FRAME_VALUE * weights[g] - dcost
                if gain > best_gain:
                    best_gain = gain
                    best = (g, pos)
        if best is None:
            break
        g, pos = best
        order.insert(pos, g)
        remaining.remove(g)
        if time.monotonic() - t0 > time_budget_s:
            break

    # 2-opt + Or-opt(1) on the gold segment to shave frames (keeps the set fixed)
    def seg_frames(seq):
        return _tour_frames(seq, C, spawn, switch, exit_)

    improved = True
    while improved and (time.monotonic() - t0) <= time_budget_s:
        improved = False
        base = seg_frames(order)
        L = len(order)
        # 2-opt
        for i in range(L - 1):
            for j in range(i + 1, L):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                if seg_frames(cand) < base - 1e-9:
                    order = cand
                    base = seg_frames(order)
                    improved = True
        # Or-opt(1): move a single gold to a better slot
        for i in range(len(order)):
            g = order[i]
            rest = order[:i] + order[i + 1:]
            for pos in range(len(rest) + 1):
                cand = rest[:pos] + [g] + rest[pos:]
                if seg_frames(cand) < base - 1e-9:
                    order = cand
                    base = seg_frames(order)
                    improved = True
                    break

    # final gate: drop any gold whose removal improves score
    changed = True
    while changed:
        changed = False
        cur_f = seg_frames(order)
        for i in range(len(order)):
            g = order[i]
            rest = order[:i] + order[i + 1:]
            # removing g: frames drop by (cur_f - seg_frames(rest)), we lose 120*w
            drop = cur_f - seg_frames(rest)
            if GOLD_FRAME_VALUE * weights[g] < drop - 1e-9:
                order = rest
                changed = True
                break
    return order


def solve_orienteering(C: np.ndarray, reachable_mask: np.ndarray,
                       gold_counts: Sequence[float], *, spawn_idx: int,
                       switch_idx: int, exit_idx: int,
                       gold_frame_value: float = GOLD_FRAME_VALUE,
                       exact_max_n: int = 16, medium_max_n: int = 60,
                       time_budget_s: float = 20.0) -> List[int]:
    """Choose gold subset + visit order maximizing ``120*gold - frames``.

    ``gold_counts[i]`` is the #real-gold represented by node i (1 for a singleton,
    k for a cluster super-node). Returns the ordered list of selected gold-node
    indices (into ``0..n-1``). Regime dispatch on the reachable-gold count n.

    ``gold_frame_value`` is accepted for API symmetry; the score exchange rate is
    the module constant ``GOLD_FRAME_VALUE`` (120), the single source of truth."""
    n = len(gold_counts)
    weights = [float(w) for w in gold_counts]
    n_reach = int(np.count_nonzero(reachable_mask))
    if n_reach == 0:
        return []
    if n_reach <= exact_max_n:
        return _held_karp(C, reachable_mask, weights, spawn_idx, switch_idx,
                          exit_idx, n)
    return _greedy_2opt(C, reachable_mask, weights, spawn_idx, switch_idx,
                        exit_idx, n, time_budget_s)


__all__ = [
    "GOLD_FRAME_VALUE",
    "snap_gold_to_nodes",
    "build_cost_matrix",
    "solve_orienteering",
]
