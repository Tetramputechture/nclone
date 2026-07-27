#!/usr/bin/env python3
"""Incremental (save/restore) d*-guided forward search — a fast drop-in for
``plan_dstar_guided``.

The original :class:`DStarGuidedSpike` restores each node's state by
``fast_reset()`` + **replaying the whole action prefix from spawn** (the
codebase's blessed deterministic-replay pattern).  That is O(N x L): every one of
the ~N expanded nodes replays its ~L-action prefix, so an official level that
needs a ~250-action route burns ~200 s of pure re-simulation on ~60k expansions.

The native ``Simulator`` exposes bit-exact ``save_full_state`` /
``restore_full_state`` (966-float snapshot, ~8 us restore vs ~1 ms full replay).
This planner stores each frontier node's snapshot and expands a child by
``restore_full_state(parent)`` + applying ONE action — O(1) per node, O(N) total.
Determinism is preserved exactly (restore(parent)+one action ==
fast_reset+replay(prefix)), so the search is byte-identical to the original; it is
purely a speed change (measured ~10-20x on official levels).

Everything else — the lattice ``d*`` heuristic, the (x,y,vx,vy,switch,doors)
dedup key, clincher actions, frame/expansion budgets — is inherited unchanged.

``plan_dstar_guided_fast`` mirrors ``plan_dstar_guided``'s signature and return
shape, so callers (``export_demos``) can swap it in behind a flag.
"""

from __future__ import annotations

import heapq
import os
import time
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import math

from .dstar_guided_spike import (
    DStarGuidedSpike,
    GoldChaseSpike,
    _cluster_indices,
    _tape_from_actions,
    shortcut_frame_tape,
)
from .physics_route_spike import ACT
from ..lattice_distance import LatticeDistanceSource


class FastDStarSolver(DStarGuidedSpike):
    """DStarGuidedSpike with O(1)-per-node incremental state via save/restore."""

    def _read_state(self, frames: int) -> dict:
        """Read the CURRENT sim state into the search-node dict (no replay).

        Hot path (called once per child ~O(N) times): fetch entities via the
        lighter ``get_state_dynamic`` ONCE and derive switch + open-doors from it,
        instead of ``_switch_done``+``_open_doors`` each rebuilding the entity list
        (measured ~14 s of the dense-map search was this double scan)."""
        x, y = self.sim.get_ninja_pos()
        vx, vy = self.sim.get_ninja_velocity()
        switch_done = False
        doors: Tuple[int, ...] = ()
        door_idx = 0
        open_doors: List[int] = []
        for e in self.sim.get_state_dynamic().entities:
            et = e.entity_type
            if et == 4:
                switch_done = not bool(getattr(e, "active", True))
            elif et == 6 and self._n_locked_doors:
                if not bool(getattr(e, "door_closed", True)):
                    open_doors.append(door_idx)
                door_idx += 1
        if open_doors:
            doors = tuple(open_doors)
        s = {
            "x": x, "y": y, "vx": vx, "vy": vy,
            "frames": frames,
            "dead": self.sim.has_died(), "won": self.sim.has_won(),
            "switch": switch_done,
            "doors": doors,
        }
        if self.enriched_key:
            s["jd"] = min(self.sim.get_ninja_jump_duration(), 45)
            s["stc"] = self.sim.get_ninja_state_code()
            s["air"] = int(self.sim.get_ninja_airborn())
        return s

    def _apply_one(self, a: int) -> int:
        """Apply ONE action from the CURRENT sim state; return frames advanced.

        Mirrors the per-action semantics of :meth:`_replay` (macro-hold for base
        actions, adaptive clincher otherwise) but does NOT ``fast_reset`` first —
        it stacks on whatever state the sim is currently in (a restored parent)."""
        if a in self._clinch_ids:
            return self._apply_clinch(a)
        h, j = ACT[a]
        frames = 0
        for _ in range(self.macro):
            self.sim.tick(h, j)
            frames += 1
            if self.sim.has_died() or self.sim.has_won():
                return frames
        return frames

    def plan(self) -> dict:  # noqa: C901 - mirrors the parent loop structure
        t0 = time.time()
        self.sim.fast_reset()
        root = self._read_state(0)
        # State snapshots are 966-byte buffers (all values 0-255); store as
        # ``bytes`` (~1 KB/node, compact + immutable) and restore via ``list``.
        root_snap = bytes(self.sim.save_full_state())
        start = (self.w * self._h(root["x"], root["y"], root["switch"], root["doors"]),
                 0.0, 0, [], root, root_snap)
        frontier = [start]
        best_g: Dict[Tuple, float] = {self._key(root): 0.0}
        counter = 1
        expansions = 0
        explored: List[Tuple[float, float, float]] = []
        deaths = 0

        while frontier and expansions < self.max_expansions:
            f, g, _, actions, s, snap = heapq.heappop(frontier)
            if g > best_g.get(self._key(s), float("inf")):
                continue
            expansions += 1
            explored.append((s["x"], s["y"], g))
            if s["won"]:
                return self._finish(actions, g, expansions, deaths, explored, t0, True)
            if g >= self.max_frames or len(actions) >= self.max_actions:
                continue
            for a in self._action_ids:
                # Reset to THIS node's state, then apply exactly one action.
                # ``snap`` is a 966-byte buffer; restore accepts bytes directly
                # (no per-call ``list()`` materialisation — 900k of those on a
                # deep search).
                self.sim.restore_full_state(snap)
                df = self._apply_one(a)
                cg = g + df
                cs = self._read_state(cg)
                if cs["dead"]:
                    deaths += 1
                    continue
                k = self._key(cs)
                if cg < best_g.get(k, float("inf")):
                    best_g[k] = cg
                    child_snap = bytes(self.sim.save_full_state())
                    cf = (cg if cs["won"]
                          else cg + self.w * self._h(cs["x"], cs["y"], cs["switch"],
                                                     cs["doors"]))
                    heapq.heappush(frontier,
                                   (cf, cg, counter, actions + [a], cs, child_snap))
                    counter += 1
        return self._finish([], float("inf"), expansions, deaths, explored, t0, False)


def plan_dstar_guided_fast(map_path: str, *, macro: int = 6, vel_bin: float = 0.25,
                           h_weight: float = 2.0, max_expansions: int = 150000,
                           max_frames: int = 2600, w_risk: Optional[float] = None,
                           jump_tax: Optional[float] = None,
                           momentum_model: str = "legacy",
                           enriched_key: bool = False,
                           clinch_actions: bool = False) -> dict:
    """Fast (incremental save/restore) equivalent of ``plan_dstar_guided``."""
    from .physics_route_spike import PhysicsRouteSpike
    src = LatticeDistanceSource(
        PhysicsRouteSpike(map_path).map_data, w_risk=w_risk, jump_tax=jump_tax,
        momentum_model=momentum_model, use_cache=True,
        name=os.path.basename(map_path.rstrip("/")),
    )
    sp = FastDStarSolver(
        map_path, macro=macro, vel_bin=vel_bin, h_weight=h_weight,
        max_expansions=max_expansions, max_frames=max_frames, lattice_source=src,
        enriched_key=enriched_key, clinch_actions=clinch_actions,
    )
    res = sp.plan()
    res["lattice_available"] = bool(src.available)
    res["macro"] = macro
    if res.get("solved") and clinch_actions:
        res["frame_tape"] = sp.frame_inputs(res["actions"])
    return res


# ===========================================================================
# Score-maximizing (gold orienteering) route planning.
# ===========================================================================
class FastGoldChaseSpike(GoldChaseSpike, FastDStarSolver):
    """GoldChaseSpike with O(1)-per-node save/restore legs.

    MRO: gold state + ``_h_leg``/``_uncollected`` from :class:`GoldChaseSpike`;
    the save/restore mechanics (``_apply_one``, ``_read_state``, byte snapshots)
    from :class:`FastDStarSolver`. Legs use :meth:`_plan_leg_fast` (restore parent
    snapshot + one action) instead of the base ``_plan_leg``'s per-expansion full
    ``fast_reset()``+replay. Byte-identical to the slow leg planner by the same
    determinism argument as ``FastDStarSolver``."""

    def _read_state(self, frames: int) -> dict:
        # FastDStarSolver._read_state (fast, no replay) + the gold count.
        s = super()._read_state(frames)
        s["gold"] = int(self.sim.get_ninja_gold_collected())
        return s

    def _completion_search(self, start_snap: bytes, prefix_len: int,
                           max_expansions: int) -> Optional[List[int]]:
        """Monolithic switch+exit A* from ``start_snap`` (the two-phase lattice
        ``_h``, win == exit reached). Strictly stronger than a leg-decomposed
        switch-then-exit finish: it searches both phases JOINTLY, so it wins
        wherever the completion planner does. Returns actions-to-win or None."""
        self.sim.restore_full_state(start_snap)
        root = self._read_state(0)
        if root["won"]:
            return []
        start = (self.w * self._h(root["x"], root["y"], root["switch"], root["doors"]),
                 0.0, 0, [], root, start_snap)
        frontier = [start]
        best = {self._key(root): 0.0}
        counter = 1
        exp = 0
        while frontier and exp < max_expansions:
            f, g, _, actions, s, snap = heapq.heappop(frontier)
            if g > best.get(self._key(s), float("inf")):
                continue
            exp += 1
            if s["won"]:
                return actions
            if g >= self.max_frames or prefix_len + len(actions) >= self.max_actions:
                continue
            for a in ACT:
                self.sim.restore_full_state(snap)
                df = self._apply_one(a)
                cg = g + df
                cs = self._read_state(cg)
                if cs["dead"]:
                    continue
                k = self._key(cs)
                if cg < best.get(k, float("inf")):
                    best[k] = cg
                    child_snap = self._snap()
                    cf = (cg if cs["won"]
                          else cg + self.w * self._h(cs["x"], cs["y"], cs["switch"],
                                                     cs["doors"]))
                    heapq.heappush(frontier,
                                   (cf, cg, counter, actions + [a], cs, child_snap))
                    counter += 1
        return None

    def _snap(self) -> bytes:
        return bytes(self.sim.save_full_state())

    def _advance(self, snap: bytes, actions: List[int]) -> bytes:
        """Restore ``snap``, apply ``actions`` (one macro-hold each), return new snap."""
        self.sim.restore_full_state(snap)
        for a in actions:
            self._apply_one(a)
        return self._snap()

    def _state_from_snap(self, snap: bytes) -> dict:
        self.sim.restore_full_state(snap)
        return self._read_state(0)

    def _uncollected_from_snap(self, snap: bytes) -> List[int]:
        """Gold indices still active in the state ``snap`` (no replay)."""
        self.sim.restore_full_state(snap)
        active = [(float(e.x), float(e.y))
                  for e in self.sim.get_state_dynamic().entities
                  if e.entity_type == 2 and getattr(e, "active", False)]
        out = []
        for i, (gx, gy) in enumerate(self.gold_xy):
            if any(abs(gx - ax) < 0.5 and abs(gy - ay) < 0.5 for (ax, ay) in active):
                out.append(i)
        return out

    def _plan_leg_fast(self, start_snap: bytes, goal_fn, h_target,
                       max_expansions: int, max_actions: int,
                       prefix_len: int) -> Optional[List[int]]:
        """A* one leg from ``start_snap`` using save/restore. Returns leg macro
        actions (excluding prefix) or None. Mirrors ``GoldChaseSpike._plan_leg``
        (uniform ``macro`` per action, ``_h_leg`` heuristic) but O(1)/node."""
        self._h_target = h_target
        self.sim.restore_full_state(start_snap)
        root = self._read_state(0)
        if goal_fn(root):
            return []
        root_snap = self._snap()
        start = (self.w * self._h_leg(root), 0.0, 0, [], root, root_snap)
        frontier = [start]
        best = {self._key(root): 0.0}
        counter = 1
        exp = 0
        while frontier and exp < max_expansions:
            f, g, _, la, s, snap = heapq.heappop(frontier)
            if g > best.get(self._key(s), float("inf")):
                continue
            exp += 1
            if goal_fn(s):
                return la
            if prefix_len + len(la) >= max_actions:
                continue
            for a in ACT:
                self.sim.restore_full_state(snap)
                self._apply_one(a)
                cs = self._read_state(g + self.macro)
                if cs["dead"]:
                    continue
                cg = g + self.macro
                k = self._key(cs)
                if cg < best.get(k, float("inf")):
                    best[k] = cg
                    child_snap = self._snap()
                    cf = cg if goal_fn(cs) else cg + self.w * self._h_leg(cs)
                    heapq.heappush(frontier,
                                   (cf, cg, counter, la + [a], cs, child_snap))
                    counter += 1
        return None


def _grid_bucket_clusters(gold_xy, max_clusters, cell=96.0):
    """Coarse spatial-grid bucketing to hard-cap cluster count for huge gold sets."""
    from collections import defaultdict
    while True:
        buckets = defaultdict(list)
        for i, (x, y) in enumerate(gold_xy):
            buckets[(int(x // cell), int(y // cell))].append(i)
        if len(buckets) <= max_clusters or cell > 2000:
            return list(buckets.values())
        cell *= 1.5


def _run_gold_route(sp, src, *, macro, leg_expansions, final_expansions,
                    max_frames, cluster_eps, order_mode, gold_frame_value,
                    exact_max_n, medium_max_n, orienteering_time_budget_s,
                    gold_phase_budget_s=90.0, max_cluster_gold=16):
    """Shared executor: collect gold (order-driven or greedy-nearest) then finish
    with the sacrifice-backtrack completion guarantee. ``order_mode`` is
    "orienteering" or "greedy". A wall-clock ``gold_phase_budget_s`` bounds the
    collection phase (dense-gold carpets can't burn minutes); the monolithic finish
    still completes the level. Returns the plan_gold_chase-shaped result dict."""
    t0 = time.time()
    max_actions = max_frames // macro
    sp.sim.fast_reset()
    fresh_snap = sp._snap()

    # Completion floor + FAST uncompletable detection: the monolithic switch+exit
    # search from spawn. If it can't win, the level is uncompletable by the planner
    # -> bail immediately (skip the whole gold phase; this is what made the 1186-gold
    # 'error' map burn ~380s before). Otherwise ``empty_fin`` is the GUARANTEED
    # completion fallback for the sacrifice-backtrack terminal case.
    empty_fin = sp._completion_search(fresh_snap, 0, final_expansions)
    if empty_fin is None:
        return {"solved": False, "won": False, "actions": [], "macro": macro,
                "frames": 0, "frame_tape": [], "frames_shortcut": 0,
                "gold_collected": 0, "gold_total": len(sp.gold_xy),
                "sacrificed_gold": [], "legs": [], "n_clusters": 0,
                "lattice_available": bool(src.available),
                "wall_time_s": time.time() - t0, "estimated_score": 0.0,
                "orienteering_regime": "uncompletable"}
    phase_t0 = time.time()

    def _recoverable(s):
        if not src.available:
            return True
        return math.isfinite(sp.lat.cost_to_go(s["x"], s["y"], bool(s["switch"])))

    def _centroid(idxs):
        return (sum(sp.gold_xy[i][0] for i in idxs) / len(idxs),
                sum(sp.gold_xy[i][1] for i in idxs) / len(idxs))

    legs = []
    gold_legs: List[Tuple[int, List[int]]] = []
    prefix_actions: List[int] = []
    prefix_snap = fresh_snap

    # Cluster gold, capping BOTH cluster count and per-cluster size: a single leg
    # whose goal is "collect all N gold in this cluster" is intractable for large N
    # (a 1186-gold carpet merges to one cluster under single-linkage), so grid-split
    # when either the count or the largest cluster exceeds the caps.
    if sp.gold_xy:
        clusters = _cluster_indices(sp.gold_xy, cluster_eps)
        biggest = max((len(c) for c in clusters), default=0)
        if len(clusters) > medium_max_n or biggest > max_cluster_gold:
            clusters = _grid_bucket_clusters(sp.gold_xy, medium_max_n)
    else:
        clusters = []

    orienteering_regime = "none"
    # Decide cluster visit order.
    if clusters and order_mode == "orienteering" and src.available and sp.lat._graph is not None:
        from ..gold_orienteering import (build_cost_matrix, snap_gold_to_nodes,
                                         solve_orienteering)
        from ..surface_graph import _build_reverse_adjacency
        graph = sp.lat._graph
        reps = [_centroid(cl) for cl in clusters]
        counts = [len(cl) for cl in clusters]
        rev_adj = _build_reverse_adjacency(graph)
        _snapped, goal_nodes = snap_gold_to_nodes(graph, reps)
        C, mask = build_cost_matrix(graph, rev_adj, sp.spawn, goal_nodes,
                                    sp.switch, sp.exit)
        n = len(clusters)
        order = solve_orienteering(C, mask, counts, spawn_idx=n, switch_idx=n + 1,
                                   exit_idx=n + 2, gold_frame_value=gold_frame_value,
                                   exact_max_n=exact_max_n, medium_max_n=medium_max_n,
                                   time_budget_s=orienteering_time_budget_s)
        orienteering_regime = ("exact" if int(mask.sum()) <= exact_max_n
                               else "greedy_2opt")
    else:
        order = None  # greedy-nearest

    # --- Phase A: collect gold, cluster by cluster ---
    # Orienteering needs gold to snap to a surface node for the cost matrix; N++
    # gold often floats in mid-air (unsnappable) -> the matrix is all-inf and
    # orienteering selects NOTHING. But the physics legs (Euclidean-guided) CAN
    # collect mid-air gold, so fall back to greedy collection when orienteering
    # picks nothing yet gold exists. (Snappable maps keep the TSP ordering.)
    if order:
        visit_seq = list(order)
    else:
        visit_seq = None  # greedy-nearest, recomputed each step

    def _plan_cluster(cl_members, base_gold):
        """Plan one cluster leg from prefix_snap; returns (leg, committed_members)."""
        target = base_gold + len(cl_members)
        # Cap the budget scaling so no single leg can eat the whole phase.
        budget = min(leg_expansions * min(max(1, len(cl_members)), 4), final_expansions)
        leg = sp._plan_leg_fast(
            prefix_snap, lambda s, t=target: s["gold"] >= t and _recoverable(s),
            _centroid(cl_members), budget, max_actions, len(prefix_actions))
        if leg is None and len(cl_members) > 1:
            # fall back to nearest single gold in the cluster
            cx, cy = _state_cx_cy
            ti = min(cl_members, key=lambda i: (sp.gold_xy[i][0] - cx) ** 2
                     + (sp.gold_xy[i][1] - cy) ** 2)
            leg = sp._plan_leg_fast(
                prefix_snap, lambda s, b=base_gold: s["gold"] > b and _recoverable(s),
                sp.gold_xy[ti], leg_expansions, max_actions, len(prefix_actions))
            return leg, [ti]
        return leg, list(cl_members)

    remaining = set(sp._uncollected_from_snap(prefix_snap)) if sp.gold_xy else set()
    if visit_seq is not None:
        # ORDERED execution.
        for ci in visit_seq:
            if time.time() - phase_t0 > gold_phase_budget_s:
                break
            live = [i for i in clusters[ci] if i in remaining]
            if not live:
                continue
            cur = sp._state_from_snap(prefix_snap)
            _state_cx_cy = (cur["x"], cur["y"])
            leg, members = _plan_cluster(live, cur["gold"])
            if leg is None:
                ti = live[0]
                legs.append(("gold", ti, 0, False))
                remaining.discard(ti)
                continue
            prefix_actions += leg
            prefix_snap = sp._advance(prefix_snap, leg)
            gold_legs.append((members[0], leg))
            legs.append(("gold-cluster" if len(members) > 1 else "gold",
                         members[0], len(leg), True))
            remaining = set(sp._uncollected_from_snap(prefix_snap))
            if len(prefix_actions) >= max_actions:
                break
    else:
        # GREEDY-nearest execution.
        while remaining:
            if time.time() - phase_t0 > gold_phase_budget_s:
                break
            cur = sp._state_from_snap(prefix_snap)
            cx, cy, base = cur["x"], cur["y"], cur["gold"]
            _state_cx_cy = (cx, cy)
            live = [[i for i in cl if i in remaining] for cl in clusters]
            live = [cl for cl in live if cl]
            if not live:
                break
            cl = min(live, key=lambda c: min((sp.gold_xy[i][0] - cx) ** 2
                                             + (sp.gold_xy[i][1] - cy) ** 2 for i in c))
            leg, members = _plan_cluster(cl, base)
            if leg is None:
                ti = min(cl, key=lambda i: (sp.gold_xy[i][0] - cx) ** 2
                         + (sp.gold_xy[i][1] - cy) ** 2)
                legs.append(("gold", ti, 0, False))
                remaining.discard(ti)
                continue
            prefix_actions += leg
            prefix_snap = sp._advance(prefix_snap, leg)
            gold_legs.append((members[0], leg))
            legs.append(("gold-cluster" if len(members) > 1 else "gold",
                         members[0], len(leg), True))
            remaining = set(sp._uncollected_from_snap(prefix_snap))
            if len(prefix_actions) >= max_actions:
                break

    # --- Phase B/C: MONOLITHIC switch+exit finish + sacrifice-backtrack ---
    # The finish is a single joint switch+exit A* from the post-gold snapshot
    # (``_completion_search``), NOT a leg-decomposed switch-then-exit — the latter
    # can strand the ninja (pick a switch approach with no onward exit), failing
    # where the monolithic completion planner wins. This makes the completion
    # guarantee hold: whenever the gold-free route wins, the empty-gold finish wins.
    def _finish_from(gold_actions):
        snap = sp._advance(fresh_snap, gold_actions)
        fin = sp._completion_search(snap, len(gold_actions), final_expansions)
        return (fin is not None), fin

    def _concat(legs_list):
        out: List[int] = []
        for _, la in legs_list:
            out += la
        return out

    _MAX_SACRIFICE_SEARCHES = 6  # bound backtrack cost; empty_fin is the safety net
    sacrificed: List[int] = []
    full_actions: List[int] = []
    won = False

    ok, fin = _finish_from(_concat(gold_legs))
    if ok:
        full_actions = _concat(gold_legs) + fin
        legs.append(("finish", -1, len(fin), True))
        won = True
    else:
        # Full-gold finish stranded the ninja. Sacrifice most-recent gold legs and
        # retry (bounded); if still failing, drop ALL gold and use the known-good
        # spawn completion (empty_fin) -> completion NEVER regresses.
        sac = 0
        while gold_legs and sac < _MAX_SACRIFICE_SEARCHES:
            gi, _ = gold_legs.pop()
            sacrificed.append(gi)
            legs.append(("sacrifice-gold", gi, 0, True))
            if not gold_legs:
                break
            ok, fin = _finish_from(_concat(gold_legs))
            sac += 1
            if ok:
                full_actions = _concat(gold_legs) + fin
                legs.append(("finish", -1, len(fin), True))
                won = True
                break
        if not won:
            full_actions = list(empty_fin)  # guaranteed win (verified upfront)
            legs.append(("finish-empty", -1, len(empty_fin), True))
            won = True

    # Final state + tape.
    final = sp._state_from_snap(sp._advance(fresh_snap, full_actions))
    won = bool(final["won"])
    gold_collected = int(final["gold"])
    frame_tape = _tape_from_actions(full_actions, macro)
    frames_shortcut = len(frame_tape)
    if won:
        frame_tape = shortcut_frame_tape(sp.map_path, frame_tape)
        frames_shortcut = len(frame_tape)
    gold_total = len(sp.gold_xy)
    est_score = (90 * 60 - frames_shortcut + 1 + gold_collected * 120) / 60.0
    return {
        "solved": won, "won": won, "actions": full_actions, "macro": macro,
        "frames": len(full_actions) * macro, "frame_tape": frame_tape,
        "frames_shortcut": frames_shortcut, "gold_collected": gold_collected,
        "gold_total": gold_total, "sacrificed_gold": sacrificed, "legs": legs,
        "lattice_available": bool(src.available), "wall_time_s": time.time() - t0,
        "estimated_score": est_score, "orienteering_regime": orienteering_regime,
        "n_clusters": len(clusters),
    }


def _build_gold_spike(map_path, *, macro, vel_bin, h_weight, max_frames, w_risk,
                      jump_tax, momentum_model, enriched_key):
    from .physics_route_spike import PhysicsRouteSpike
    src = LatticeDistanceSource(
        PhysicsRouteSpike(map_path).map_data, w_risk=w_risk, jump_tax=jump_tax,
        momentum_model=momentum_model, use_cache=True,
        name=os.path.basename(map_path.rstrip("/")),
    )
    sp = FastGoldChaseSpike(
        map_path, macro=macro, vel_bin=vel_bin, h_weight=h_weight,
        max_expansions=300000, max_frames=max_frames, lattice_source=src,
        enriched_key=enriched_key,
    )
    return sp, src


def plan_score_route(map_path: str, *, macro: int = 6, vel_bin: float = 0.25,
                     h_weight: float = 2.0, leg_expansions: int = 80000,
                     final_expansions: int = 200000, max_frames: int = 6000,
                     w_risk: Optional[float] = None, jump_tax: Optional[float] = None,
                     momentum_model: str = "legacy", enriched_key: bool = False,
                     cluster_eps: float = 48.0, gold_frame_value: float = 120.0,
                     exact_max_n: int = 16, medium_max_n: int = 60,
                     orienteering_time_budget_s: float = 20.0) -> dict:
    """SCORE-maximizing route: orienteering gold selection + physics legs +
    sacrifice-backtrack completion guarantee. Same return shape as
    ``plan_gold_chase`` plus ``estimated_score``/``orienteering_regime``/``n_clusters``."""
    sp, src = _build_gold_spike(map_path, macro=macro, vel_bin=vel_bin,
                                h_weight=h_weight, max_frames=max_frames, w_risk=w_risk,
                                jump_tax=jump_tax, momentum_model=momentum_model,
                                enriched_key=enriched_key)
    return _run_gold_route(sp, src, macro=macro, leg_expansions=leg_expansions,
                           final_expansions=final_expansions, max_frames=max_frames,
                           cluster_eps=cluster_eps, order_mode="orienteering",
                           gold_frame_value=gold_frame_value, exact_max_n=exact_max_n,
                           medium_max_n=medium_max_n,
                           orienteering_time_budget_s=orienteering_time_budget_s)


def plan_gold_chase_fast(map_path: str, *, macro: int = 6, vel_bin: float = 0.25,
                         h_weight: float = 2.0, leg_expansions: int = 80000,
                         final_expansions: int = 200000, max_frames: int = 6000,
                         w_risk: Optional[float] = None, jump_tax: Optional[float] = None,
                         momentum_model: str = "legacy", enriched_key: bool = False,
                         cluster_eps: float = 48.0) -> dict:
    """Fast (save/restore) greedy-nearest all-gold chase — the fast equivalent of
    ``plan_gold_chase``, used for leg-planner parity testing vs the slow version."""
    sp, src = _build_gold_spike(map_path, macro=macro, vel_bin=vel_bin,
                                h_weight=h_weight, max_frames=max_frames, w_risk=w_risk,
                                jump_tax=jump_tax, momentum_model=momentum_model,
                                enriched_key=enriched_key)
    return _run_gold_route(sp, src, macro=macro, leg_expansions=leg_expansions,
                           final_expansions=final_expansions, max_frames=max_frames,
                           cluster_eps=cluster_eps, order_mode="greedy",
                           gold_frame_value=120.0, exact_max_n=16, medium_max_n=60,
                           orienteering_time_budget_s=20.0)
