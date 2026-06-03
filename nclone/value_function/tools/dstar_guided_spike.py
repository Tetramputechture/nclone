#!/usr/bin/env python3
"""Forward physics A* guided by the lattice ``d*`` heuristic (Plan→Search).

``PhysicsRouteSpike`` searches real-physics macro-action sequences but its
heuristic is straight-line distance / max-speed — useless inside a dense mine
maze, where the safe path winds and the straight line points through kill-zones.
This swaps in the **lattice ``d*`` cost-to-go** (phase-aware, already in frames)
as the heuristic: the search is pulled along the planner's known-safe corridor
while still pruning *real* deaths on the native sim.  It combines the macro plan
(``d*``, which knows the route exists) with frame-exact forward search (which the
open-loop executors can't do through tight mines).

Where the lattice has no nearby node (airborne over a gap) it falls back to the
geometric heuristic, so it degrades gracefully.

CLI::

    .venv/bin/python -m nclone.value_function.tools.dstar_guided_spike \
        --map nclone/test_maps/mine-very-difficult --output /tmp/win.mp4 \
        --max-expansions 150000 --max-frames 2600 --h-weight 2.0
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import List, Optional

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

from .physics_route_spike import PhysicsRouteSpike, ACT, _switch_done, MAX_HOR_SPEED
from ..lattice_distance import LatticeDistanceSource


class DStarGuidedSpike(PhysicsRouteSpike):
    """PhysicsRouteSpike whose heuristic is the lattice ``d*`` cost-to-go."""

    def __init__(self, *args, lattice_source: Optional[LatticeDistanceSource] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.lat = lattice_source

    def _h(self, x: float, y: float, switch_done: bool,
           open_doors=None) -> float:
        # MIN of two admissible lower bounds:
        #   - Euclidean / MAX_HOR_SPEED (straight-line lower bound, always
        #     admissible — `super()._h`).
        #   - Lattice d* cost-to-go via the nearest SURFACE point.  This is
        #     admissible WHEN the optimal route really does descend to that
        #     surface, but OVER-estimates for trajectories that stay airborne
        #     and reach the goal more directly (e.g. the canonical "hold the
        #     wall-jump and contact the air-only switch on the way up" pattern
        #     on 035 opportunity costs — the L-wall surface d* says ~150f
        #     assuming descent, but the airborne path reaches goal in ~100f).
        # Taking the min keeps the heuristic admissible (a necessary condition
        # for A* optimality with the en-heap-only termination), so the search
        # actually finds the shorter airborne route instead of bottoming out
        # on the longer descend-then-climb route.  The lattice still provides
        # the strong physics-aware gradient WHERE it is the tighter bound
        # (most of the surface-routed map); the Euclidean dominates only in
        # the airborne pockets where the lattice over-estimates.
        eu = super()._h(x, y, switch_done, open_doors)
        if self.lat is not None and self.lat.available:
            # Pass the per-node open-door set so the heuristic uses the CURRENT
            # door phase's leg (cost-to-go to the next necessary switch, then the
            # exit), advancing as the search opens doors. Without doors this is
            # the plain two-phase cost-to-go.
            c = self.lat.cost_to_go(
                x, y, bool(switch_done), open_door_indices=open_doors)
            if math.isfinite(c) and c < eu:
                return float(c)
        return float(eu)


class GoldChaseSpike(DStarGuidedSpike):
    """d*-guided forward search that collects all gold *then* finishes.

    The base spike's win condition is exit-only, so its optimal route skips gold.
    Gold collection isn't expressible as a single admissible heuristic without a
    2^N remaining-gold state blow-up.  Instead we decompose: search one short
    physics leg per gold (nearest-uncollected first), committing the leg's macro
    actions as a prefix, then a switch leg and an exit leg.  Each leg is a full
    frame-exact A* on the native sim (so it still prunes real mine deaths), but
    its goal is "collect the next gold" / "hit switch" / "win".  Gold legs use a
    Euclidean-to-gold heuristic (admissible, gold isn't in the lattice d* field);
    switch/exit legs keep the lattice d* heuristic for mine-maze navigation.

    Gold tracking uses the native ``get_ninja_gold_collected`` getter + the gold
    entity ``active`` flags (now that native simulates gold)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Gold positions from the native state (type-2 entities).
        st = self.sim.get_state()
        self.gold_xy = [(float(e.x), float(e.y))
                        for e in st.entities if e.entity_type == 2]
        self._h_target = None  # (tx, ty) for gold legs; None => lattice switch/exit

    def _state_after(self, actions):
        s = super()._state_after(actions)
        s["gold"] = int(self.sim.get_ninja_gold_collected())
        return s

    def _h_leg(self, s) -> float:
        if self._h_target is not None:
            d = math.hypot(s["x"] - self._h_target[0], s["y"] - self._h_target[1])
            return d / MAX_HOR_SPEED
        return self._h(s["x"], s["y"], s["switch"])

    def _uncollected(self, prefix) -> List[int]:
        """Indices into ``gold_xy`` still active after replaying ``prefix``."""
        self._replay(prefix)
        st = self.sim.get_state()
        active = [(float(e.x), float(e.y))
                  for e in st.entities if e.entity_type == 2 and e.active]
        out = []
        for i, (gx, gy) in enumerate(self.gold_xy):
            if any(abs(gx - ax) < 0.5 and abs(gy - ay) < 0.5 for (ax, ay) in active):
                out.append(i)
        return out

    def _plan_leg(self, prefix, goal_fn, h_target, max_expansions, max_actions):
        """A* one leg: macro-actions appended to ``prefix`` until ``goal_fn`` holds.

        Returns the leg's macro-action list (excluding prefix) or ``None``."""
        import heapq
        self._h_target = h_target
        root = self._state_after(prefix)
        if goal_fn(root):
            return []
        start = (self.w * self._h_leg(root), 0.0, 0, [], root)
        frontier = [start]
        best = {self._key(root): 0.0}
        counter = 1
        exp = 0
        deaths = 0
        while frontier and exp < max_expansions:
            f, g, _, la, s = heapq.heappop(frontier)
            if g > best.get(self._key(s), float("inf")):
                continue
            exp += 1
            if goal_fn(s):
                return la
            if len(prefix) + len(la) >= max_actions:
                continue
            for a in ACT:
                child = la + [a]
                cs = self._state_after(prefix + child)
                if cs["dead"]:
                    deaths += 1
                    continue
                cg = g + self.macro
                k = self._key(cs)
                if cg < best.get(k, float("inf")):
                    best[k] = cg
                    cf = cg if goal_fn(cs) else cg + self.w * self._h_leg(cs)
                    heapq.heappush(frontier, (cf, cg, counter, child, cs))
                    counter += 1
        return None


def _cluster_indices(points, eps: float):
    """Single-linkage cluster of gold indices (points within ``eps`` px are one
    cluster).  Lets a leg sweep a whole cluster in one arc instead of landing
    between every gold."""
    n = len(points)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    e2 = eps * eps
    for i in range(n):
        for k in range(i + 1, n):
            if (points[i][0] - points[k][0]) ** 2 + (points[i][1] - points[k][1]) ** 2 <= e2:
                ra, rb = find(i), find(k)
                if ra != rb:
                    parent[ra] = rb
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def plan_gold_chase(map_path: str, *, macro: int = 6, vel_bin: float = 0.25,
                    h_weight: float = 2.0, leg_expansions: int = 80000,
                    final_expansions: int = 200000,
                    max_frames: int = 6000, w_risk: Optional[float] = None,
                    jump_tax: Optional[float] = None,
                    momentum_model: str = "legacy",
                    enriched_key: bool = False, shortcut: bool = True,
                    cluster_eps: float = 48.0, cluster_legs: bool = True) -> dict:
    """All-gold route via per-gold d*-guided physics legs, then switch + exit.

    Returns the same shape as ``plan_dstar_guided`` plus ``gold_collected`` /
    ``gold_total`` / ``legs`` (per-leg ``(kind, idx, macro_actions, ok)``).  When
    ``shortcut`` (default), a per-frame loop-removal pass trims redundant
    backtracks for a faster replay; the result carries ``frame_tape`` (the
    shortened per-frame tape) and ``frames_shortcut`` alongside the raw
    macro ``actions`` / ``frames``."""
    from .physics_route_spike import PhysicsRouteSpike as _PRS
    src = LatticeDistanceSource(
        _PRS(map_path).map_data, w_risk=w_risk, jump_tax=jump_tax,
        momentum_model=momentum_model, use_cache=True,
        name=os.path.basename(map_path.rstrip("/")),
    )
    sp = GoldChaseSpike(
        map_path, macro=macro, vel_bin=vel_bin, h_weight=h_weight,
        max_expansions=leg_expansions, max_frames=max_frames,
        lattice_source=src, enriched_key=enriched_key,
    )
    t0 = time.time()
    max_actions = max_frames // macro
    prefix: List[int] = []
    legs = []

    # A gold leg must not only TOUCH the gold but also land back on the safe
    # corridor — a state from which the lattice still has a finite cost-to-go to
    # the current phase goal.  Without this the greedy "touch gold ASAP" search
    # happily strands the ninja in a mine-pocket with no survivable move, and the
    # subsequent switch/exit leg has an empty frontier (observed on 035: last
    # gold left the ninja at (741,145), c2g=inf, exit leg died instantly).
    def _recoverable(s) -> bool:
        if not (src.available):
            return True
        return math.isfinite(sp.lat.cost_to_go(s["x"], s["y"], bool(s["switch"])))

    # Phase A: collect all gold.  MOMENTUM-PRESERVING: collect a whole gold
    # CLUSTER in ONE leg (a single A* whose goal is "all this cluster's gold
    # collected AND recoverable") instead of one gold per leg.  Because the leg
    # only has to land (be recoverable) at the END, the search sweeps adjacent
    # gold in a single arc rather than landing + re-jumping between each — which
    # is exactly the wasteful seam (the backtrack wall-jump) we want to avoid.
    # Falls back to a single-gold leg when a whole-cluster leg can't be found in
    # budget.  Each accepted leg is kept as its own action list so the
    # backtracker can un-commit it.
    gold_legs: List[Tuple[int, List[int]]] = []  # (representative_gold_idx, macro_actions)
    clusters = (_cluster_indices(sp.gold_xy, cluster_eps)
                if cluster_legs else [[i] for i in range(len(sp.gold_xy))])

    def _centroid(idxs):
        return (sum(sp.gold_xy[i][0] for i in idxs) / len(idxs),
                sum(sp.gold_xy[i][1] for i in idxs) / len(idxs))

    remaining = set(sp._uncollected(prefix))
    while remaining:
        cur = sp._state_after(prefix)
        cx, cy, base = cur["x"], cur["y"], cur["gold"]
        # Live clusters = clusters with uncollected gold; pick the nearest by the
        # centroid of its still-uncollected members.
        live = [[i for i in cl if i in remaining] for cl in clusters]
        live = [cl for cl in live if cl]
        if not live:
            break
        cl = min(live, key=lambda c: min((sp.gold_xy[i][0] - cx) ** 2
                                         + (sp.gold_xy[i][1] - cy) ** 2 for i in c))
        target = base + len(cl)  # collect this many more gold (≈ the cluster)
        budget = min(leg_expansions * max(1, len(cl)), final_expansions)
        leg = sp._plan_leg(
            prefix, lambda s, t=target: s["gold"] >= t and _recoverable(s),
            _centroid(cl), budget, max_actions)
        if leg is None and len(cl) > 1:
            # Whole-cluster sweep too hard — fall back to the nearest single gold.
            ti = min(cl, key=lambda i: (sp.gold_xy[i][0] - cx) ** 2
                     + (sp.gold_xy[i][1] - cy) ** 2)
            leg = sp._plan_leg(
                prefix, lambda s, b=base: s["gold"] > b and _recoverable(s),
                sp.gold_xy[ti], leg_expansions, max_actions)
            cl = [ti]
        if leg is None:
            # Unreachable within budget — drop the nearest gold, keep going.
            ti = min(cl, key=lambda i: (sp.gold_xy[i][0] - cx) ** 2
                     + (sp.gold_xy[i][1] - cy) ** 2)
            legs.append(("gold", ti, 0, False))
            remaining.discard(ti)
            continue
        prefix += leg
        gold_legs.append((cl[0], leg))
        legs.append(("gold-cluster" if len(cl) > 1 else "gold", cl[0], len(leg), True))
        remaining = set(sp._uncollected(prefix))
        if len(prefix) >= max_actions:
            break

    # Phases B+C with BACKTRACKING.  A finite lattice cost-to-go at a gold leg's
    # end does NOT guarantee the ninja's *velocity* there leaves a survivable
    # move (the y=132 mine-pocket on 035 strands it — every macro-action dies,
    # exit frontier empties instantly).  So: try switch+exit; if exit can't be
    # reached, sacrifice the most-recently-collected gold and retry.  This keeps
    # the maximum gold for which the level is still completable, and guarantees a
    # winning replay whenever the gold-free route wins.
    def _finish_from(pfx):
        """switch-ensure then exit from prefix ``pfx``; returns (won, full_pfx,
        switch_leg, exit_leg) — legs are [] when not needed / not found."""
        sleg = []
        if not sp._state_after(pfx)["switch"]:
            sleg = sp._plan_leg(pfx, lambda s: s["switch"], None,
                                final_expansions, max_actions)
            if sleg is None:
                return False, pfx, None, None
            pfx = pfx + sleg
        eleg = sp._plan_leg(pfx, lambda s: s["won"], None,
                            final_expansions, max_actions)
        if eleg is None:
            return False, pfx, sleg, None
        return True, pfx + eleg, sleg, eleg

    sacrificed: List[int] = []
    while True:
        prefix = []
        for _, la in gold_legs:
            prefix += la
        won, full, sleg, eleg = _finish_from(prefix)
        if won:
            prefix = full
            if sleg:
                legs.append(("switch", -1, len(sleg), True))
            legs.append(("exit", -1, len(eleg), True))
            break
        if not gold_legs:
            # No gold left to sacrifice and still can't finish.
            if sleg:
                prefix += sleg
                legs.append(("switch", -1, len(sleg), True))
            legs.append(("exit", -1, 0, False))
            break
        gi, _ = gold_legs.pop()
        sacrificed.append(gi)
        legs.append(("sacrifice-gold", gi, 0, True))

    final = sp._state_after(prefix)
    won = bool(final["won"])
    frame_tape = _tape_from_actions(prefix, macro)
    frames_shortcut = len(frame_tape)
    if shortcut and won:
        frame_tape = shortcut_frame_tape(map_path, frame_tape)
        frames_shortcut = len(frame_tape)
    return {
        "solved": won,
        "won": won,
        "actions": prefix,
        "macro": macro,
        "frames": len(prefix) * macro,
        "frame_tape": frame_tape,
        "frames_shortcut": frames_shortcut,
        "gold_collected": int(final["gold"]),
        "gold_total": len(sp.gold_xy),
        "sacrificed_gold": sacrificed,
        "legs": legs,
        "lattice_available": bool(src.available),
        "wall_time_s": time.time() - t0,
    }


class DoorChaseSpike(DStarGuidedSpike):
    """Leg-decomposed search that opens the necessary locked-door switches (each
    guided by a faithful per-config d* field), then reaches the exit.

    A single monolithic forward macro-A* over a long door-gated route is
    intractable, and the all-doors-closed d* to the exit is ``inf`` (so it gives
    no guidance). This visits each necessary switch as its own physics A* leg,
    committing the leg's prefix, with the leg heuristic = min(Euclidean-to-target,
    faithful door-leg d*). Door-open state is already part of the search key
    (``PhysicsRouteSpike``), so opening a door is a real state transition."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._ctg_dstar = None   # current leg's d* array (over the shared node space)
        self._leg_target = None

    def set_leg(self, dstar, target_xy):
        self._ctg_dstar, self._leg_target = dstar, target_xy

    def _ctg(self, x, y) -> float:
        from ..route_extraction import find_nearest_point
        from ..surface_extraction import SURFACE_FLOOR, SURFACE_WALL
        from ..surface_graph import num_velocity_bins, node_id
        ds = self._ctg_dstar
        if ds is None or self.lat is None or self.lat._graph is None:
            return float("inf")
        g = self.lat._graph
        best = float("inf")
        for st in (SURFACE_FLOOR, SURFACE_WALL):
            pt = find_nearest_point(g, x, y, st)
            if pt is None:
                continue
            for vb in range(num_velocity_bins(g.surface_points[pt].surface_type)):
                c = float(ds[node_id(pt, vb, g.max_vel_bins)])
                if math.isfinite(c) and c < best:
                    best = c
        return best

    def _h_leg(self, s) -> float:
        eu = (math.hypot(s["x"] - self._leg_target[0], s["y"] - self._leg_target[1])
              / MAX_HOR_SPEED) if self._leg_target else 0.0
        c = self._ctg(s["x"], s["y"])
        return min(eu, c) if math.isfinite(c) else eu

    def plan_leg(self, prefix, goal_fn, max_expansions, max_actions):
        import heapq
        root = self._state_after(prefix)
        if goal_fn(root):
            return []
        start = (self.w * self._h_leg(root), 0.0, 0, [], root)
        frontier = [start]
        best = {self._key(root): 0.0}
        cnt = 1
        exp = 0
        while frontier and exp < max_expansions:
            f, g, _, la, s = heapq.heappop(frontier)
            if g > best.get(self._key(s), float("inf")):
                continue
            exp += 1
            if goal_fn(s):
                return la
            if len(prefix) + len(la) >= max_actions:
                continue
            for a in ACT:
                child = la + [a]
                cs = self._state_after(prefix + child)
                if cs["dead"]:
                    continue
                cg = g + self.macro
                k = self._key(cs)
                if cg < best.get(k, float("inf")):
                    best[k] = cg
                    cf = cg if goal_fn(cs) else cg + self.w * self._h_leg(cs)
                    heapq.heappush(frontier, (cf, cg, cnt, child, cs))
                    cnt += 1
        return None


def plan_door_chase(map_path: str, *, macro: int = 6, vel_bin: float = 0.25,
                    h_weight: float = 2.5, leg_expansions: int = 200000,
                    final_expansions: int = 400000, max_frames: int = 4000,
                    w_risk: Optional[float] = None, jump_tax: Optional[float] = None,
                    momentum_model: str = "tangential",
                    enriched_key: bool = False) -> dict:
    """Win a locked-door-gated level by visiting the planner's necessary door
    switches (faithful per-config d* guidance), then the exit. Returns the same
    shape as ``plan_dstar_guided`` plus ``necessary_order`` / ``legs``."""
    from .lattice_audit import LatticeAudit
    from ..edge_risk import (
        DEFAULT_W_RISK, DEFAULT_W_LOOP, DEFAULT_RISK_CAP, DEFAULT_JUMP_TAX,
    )
    audit = LatticeAudit(
        map_path=map_path, workers=8, use_native=True,
        w_risk=DEFAULT_W_RISK if w_risk is None else float(w_risk),
        w_loop=DEFAULT_W_LOOP, risk_cap=DEFAULT_RISK_CAP,
        jump_tax=DEFAULT_JUMP_TAX if jump_tax is None else float(jump_tax),
        momentum_model=momentum_model)
    pc = audit.precompute(use_cache=True)
    plan = pc.get("door_plan")
    src = LatticeDistanceSource.__new__(LatticeDistanceSource)
    src.available = True
    src._graph = pc["graph"]

    sp = DoorChaseSpike(map_path, macro=macro, vel_bin=vel_bin, h_weight=h_weight,
                        max_expansions=leg_expansions, max_frames=max_frames,
                        lattice_source=src, enriched_key=enriched_key)
    t0 = time.time()
    max_actions = max_frames // macro
    legs = []
    prefix: List[int] = []

    if plan is None or not plan.solvable:
        return {"solved": False, "won": False, "actions": [], "macro": macro,
                "frames": None, "necessary_order": None, "legs": legs,
                "reason": (plan.reason if plan else "no door plan"),
                "lattice_available": bool(src.available), "wall_time_s": time.time() - t0}

    door_sw = [(d.switch_x, d.switch_y) for d in audit.locked_doors]
    # One leg per necessary door switch (in dependency order).
    for j, di in enumerate(plan.necessary_order):
        sp.set_leg(plan.door_leg_dstar[j], door_sw[di])
        leg = sp.plan_leg(prefix, lambda s, di=di: di in s["doors"],
                          leg_expansions, max_actions)
        legs.append(("door-switch", di, 0 if leg is None else len(leg), leg is not None))
        if leg is None:
            return {"solved": False, "won": False, "actions": prefix, "macro": macro,
                    "frames": None, "necessary_order": plan.necessary_order,
                    "legs": legs, "reason": f"door-switch leg {di} failed",
                    "lattice_available": True, "wall_time_s": time.time() - t0}
        prefix += leg
    # Final leg to the exit door.
    sp.set_leg(plan.exit_leg_dstar, audit.exit)
    leg = sp.plan_leg(prefix, lambda s: s["won"], final_expansions, max_actions)
    legs.append(("exit", -1, 0 if leg is None else len(leg), leg is not None))
    won = leg is not None
    if won:
        prefix += leg
    return {
        "solved": won, "won": won, "actions": prefix, "macro": macro,
        "frames": len(prefix) * macro if won else None,
        "necessary_order": plan.necessary_order, "legs": legs,
        "lattice_available": True, "wall_time_s": time.time() - t0,
    }


def _tape_from_actions(actions: List[int], macro: int):
    """Expand a macro-action list to a per-frame ``(h, j)`` tape."""
    tape = []
    for a in actions:
        h, j = ACT[a]
        tape.extend([(h, j)] * macro)
    return tape


def _replay_frames_native(map_path: str, frame_tape):
    """Replay a per-frame ``(h, j)`` tape on the native sim.

    Returns ``(pre_states, won, gold, gold_total, died_at)`` where ``pre_states[k]``
    is the ninja state ENTERING frame ``k`` (i.e. after applying ``frame_tape[:k]``)
    as ``(x, y, vx, vy, gold_collected, switch_done)`` — exactly the splice-point
    state the shortcutter matches on.  ``pre_states`` has ``len(frame_tape)+1``
    entries (the last is the terminal state)."""
    import nclone_native as Nnative
    from .physics_route_spike import _load_map_data, _switch_done
    sim = Nnative.Simulator.load(_load_map_data(map_path))
    gold_total = len([e for e in sim.get_state().entities if e.entity_type == 2])

    def snap():
        x, y = sim.get_ninja_pos()
        vx, vy = sim.get_ninja_velocity()
        return (x, y, vx, vy, int(sim.get_ninja_gold_collected()),
                bool(_switch_done(sim)))

    pre = [snap()]
    died_at = None
    for k, (h, j) in enumerate(frame_tape):
        sim.tick(h, j)
        pre.append(snap())
        if sim.has_won():
            break
        if sim.has_died():
            died_at = k
            break
    st = sim.get_state().ninja
    return pre, bool(st.has_won), int(st.gold_collected), gold_total, died_at


def shortcut_frame_tape(map_path: str, frame_tape, *, pos_tol: float = 4.0,
                        vel_tol: float = 0.25, max_passes: int = 6):
    """Loop-removal shortcutting on a per-frame tape (faster all-gold replay).

    The greedy per-gold legs end on safe landings, so the route sometimes leaves a
    state, detours (e.g. a wall-jump backtrack), and returns to ~the same state —
    a round-trip that wastes frames.  This finds two frame indices ``i < j`` whose
    ENTERING states match (position within ``pos_tol`` px, velocity within
    ``vel_tol``, *same gold count and switch status* so no progress is discarded),
    splices out ``[i:j]``, and re-validates on the native sim (must still win and
    collect every gold).  Matching states make the open-loop splice physically
    sound; the re-validation makes it safe.  Greedy: each pass removes the single
    largest valid loop, repeating until none remain.  Returns the shortened tape."""
    tape = list(frame_tape)
    pre, won, gold, total, _ = _replay_frames_native(map_path, tape)
    if not won:
        return tape  # never touch a non-winning tape
    # The replay stops at the win frame, so drop any dead post-win frames; ``pre``
    # then has exactly ``len(tape)+1`` entering-states.
    tape = tape[:len(pre) - 1]
    for _ in range(max_passes):
        # Bucket entering-states by quantized key so only plausibly-matching
        # frames are compared (O(n) instead of O(n^2)).
        from collections import defaultdict
        buckets = defaultdict(list)
        qp, qv = 1.0 / pos_tol, 1.0 / vel_tol
        for k in range(len(pre)):
            x, y, vx, vy, g, sw = pre[k]
            buckets[(round(x * qp), round(y * qp),
                     round(vx * qv), round(vy * qv), g, sw)].append(k)
        # Candidate loops: widest (i, j) per bucket, largest removal first.
        cands = []
        for ks in buckets.values():
            if len(ks) >= 2:
                i, j = min(ks), max(ks)
                if j - i >= 1:
                    cands.append((j - i, i, j))
        cands.sort(reverse=True)
        applied = False
        for _gap, i, j in cands:
            cand = tape[:i] + tape[j:]
            p2, w2, g2, t2, _ = _replay_frames_native(map_path, cand)
            if w2 and g2 == total:
                tape, pre, won, gold = cand[:len(p2) - 1], p2, w2, g2
                applied = True
                break
        if not applied:
            break
    return tape


def plan_dstar_guided(map_path: str, *, macro: int = 6, vel_bin: float = 0.25,
                      h_weight: float = 2.0, max_expansions: int = 150000,
                      max_frames: int = 2600, w_risk: Optional[float] = None,
                      jump_tax: Optional[float] = None,
                      momentum_model: str = "legacy",
                      enriched_key: bool = False,
                      clinch_actions: bool = False) -> dict:
    """Build the lattice d* source + run the d*-guided forward search.

    ``enriched_key`` adds the ninja's internal jump-hold timer / movement state /
    airborne flag to the search dedup key so frame-precise variable-height jump
    tricks (air-launch / CDJ) become findable (default off => byte-parity; see
    ``PhysicsRouteSpike.__init__``)."""
    src = LatticeDistanceSource(
        PhysicsRouteSpike(map_path).map_data, w_risk=w_risk, jump_tax=jump_tax,
        momentum_model=momentum_model, use_cache=True,
        name=os.path.basename(map_path.rstrip("/")),
    )
    sp = DStarGuidedSpike(
        map_path, macro=macro, vel_bin=vel_bin, h_weight=h_weight,
        max_expansions=max_expansions, max_frames=max_frames, lattice_source=src,
        enriched_key=enriched_key, clinch_actions=clinch_actions,
    )
    t0 = time.time()
    res = sp.plan()
    res["lattice_available"] = bool(src.available)
    res["wall_time_s"] = time.time() - t0
    res["macro"] = macro
    # Composite clincher actions are adaptive, so the per-frame tape must be
    # captured live (not statically expanded from action ids).
    if res.get("solved") and clinch_actions:
        res["frame_tape"] = sp.frame_inputs(res["actions"])
    return res


def render_win_video(map_path: str, actions: List[int], macro: int,
                     output: str, fps: int = 60, *, overlay: bool = True,
                     animate: bool = False, gold_hud: bool = False,
                     title: Optional[str] = None, frame_tape=None,
                     native_faithful: bool = False) -> None:
    """Replay the winning macro-actions on a render sim and write an mp4.

    ``overlay=True`` (default) composites the route trace + mine kill-zones + a
    debug HUD (mirroring ``route_video``).  ``overlay=False`` writes a CLEAN
    gameplay capture — the raw game render only.  ``animate=True`` enables the
    ninja limb/bone animation (cosmetic; physics unchanged).  For a clean
    gameplay demo pass ``overlay=False, animate=True``.  ``gold_hud=True`` draws
    a live ``GOLD n/total`` banner (for all-gold route demos).  ``frame_tape``, if
    given, is replayed verbatim (per-frame ``(h, j)``) instead of expanding
    ``actions`` — used for the loop-shortcut tape, which isn't macro-aligned.

    ``native_faithful=True``: the search runs on the Rust native sim, but the
    render uses the Python ``nsim``, which drifts ~1 ULP after ~1000 frames and
    loses long-route wins. With this flag we step the NATIVE sim (which actually
    wins) in lockstep and overwrite the Python ninja kinematics + entity (door /
    gold / switch) states each frame, so the rendered frames are the exact native
    trajectory. Required for long door-gated routes (e.g. squiggly)."""
    import imageio
    import nclone_native as Nnative
    from ...nplay_headless import NPlayHeadless
    from .route_video import _to_hwc, _draw_overlay
    from .physics_route_spike import _MINE_STATE_DEADLY, _goal_positions, _load_map_data

    tape = list(frame_tape) if frame_tape is not None else _tape_from_actions(actions, macro)
    # Deadly-mine positions for the overlay come from a native sim (the render
    # NPlayHeadless sim has a different API and no get_state).
    _sw, _ex, mines = _goal_positions(Nnative.Simulator.load(_load_map_data(map_path)))
    deadly = [(x, y) for (x, y, s) in mines if s == _MINE_STATE_DEADLY]

    nh = NPlayHeadless(render_mode="rgb_array", enable_animation=animate,
                       enable_debug_overlay=False)
    nh.load_map(map_path)
    gold_total = len(nh.sim.entity_dic.get(2, [])) if gold_hud else 0

    # Native-faithful puppet: the native sim drives ground truth; nh only renders.
    native = Nnative.Simulator.load(_load_map_data(map_path)) if native_faithful else None
    _py_doors = nh.sim.entity_dic.get(6, [])
    _py_gold = nh.sim.entity_dic.get(2, [])
    _py_switch = nh.sim.entity_dic.get(4, [])

    def _sync_from_native():
        st = native.get_state()
        n = st.ninja
        nj = nh.sim.ninja
        nj.xpos, nj.ypos = n.xpos, n.ypos
        nj.xspeed, nj.yspeed = n.xspeed, n.yspeed
        nj.state = n.state
        nj.gold_collected = int(n.gold_collected)
        nj.doors_opened = int(n.doors_opened)
        for pd, closed in zip(_py_doors, [bool(e.door_closed) for e in st.entities if e.entity_type == 6]):
            if not closed and getattr(pd, "closed", True):
                pd.change_state(closed=False)
        for pg, active in zip(_py_gold, [bool(e.active) for e in st.entities if e.entity_type == 2]):
            pg.active = active
        sw_active = next((bool(e.active) for e in st.entities if e.entity_type == 4), False)
        for ps in _py_switch:
            ps.active = sw_active

    def _clear(x, y):
        return min((math.hypot(x - mx, y - my) for (mx, my) in deadly),
                   default=float("inf"))

    def _gold_banner(arr):
        from PIL import Image, ImageDraw
        im = Image.fromarray(arr).convert("RGB")
        d = ImageDraw.Draw(im)
        coll = int(nh.sim.ninja.gold_collected)
        d.rectangle([0, 0, 300, 28], fill=(0, 0, 0))
        if title:
            d.text((6, 2), f'"{title}"', fill=(235, 235, 235))
        col = (90, 255, 120) if coll == gold_total else (255, 215, 0)
        d.text((6, 15), f"GOLD {coll}/{gold_total}"
                        f"   {'SWITCH ON' if nh.exit_switch_activated() else ''}", fill=col)
        return np.asarray(im)

    # Pass 1: full trajectory for the (faint) planned-route overlay (overlay only).
    traj: list = []
    total = 0
    if overlay:
        nh.reset()
        if native_faithful:
            native.fast_reset()
        traj = [nh.ninja_position()]
        for (h, j) in tape:
            if native_faithful:
                native.tick(h, j)
                traj.append(tuple(native.get_ninja_pos()))
                if native.has_won() or native.has_died():
                    break
            else:
                nh.tick(h, j)
                traj.append(nh.ninja_position())
                if nh.ninja_has_won() or nh.ninja_has_died():
                    break
        total = len(traj) - 1

    def _cap(fn):
        raw = _to_hwc(nh.render())
        if not overlay:
            return _gold_banner(raw) if gold_hud else raw
        x, y = nh.ninja_position()
        vx, vy = nh.ninja_velocity()
        phase = "-> EXIT" if nh.exit_switch_activated() else "-> switch"
        out = _draw_overlay(raw, traj, fn, (x, y), math.hypot(vx, vy),
                            _clear(x, y), fn, total, phase, deadly)
        return _gold_banner(out) if gold_hud else out

    # Pass 2: render frames.
    nh.reset()
    if native_faithful:
        native.fast_reset()
        _sync_from_native()
    frames = [_cap(0)]
    won = False
    for fn, (h, j) in enumerate(tape, start=1):
        if native_faithful:
            native.tick(h, j)
            nh.tick(h, j)            # python physics -> animation pose
            _sync_from_native()      # overwrite kinematics + entities from native
            frames.append(_cap(fn))
            if native.has_won():
                won = True
                break
            if native.has_died():
                break
        else:
            nh.tick(h, j)
            frames.append(_cap(fn))
            if nh.ninja_has_won():
                won = True
                break
            if nh.ninja_has_died():
                break
    if won:
        frames.extend([frames[-1]] * fps)  # hold final frame ~1s
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    imageio.mimsave(output, frames, fps=fps, macro_block_size=1)
    return won


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", required=True)
    p.add_argument("--output", default=None, help="mp4 path (written only on a win)")
    p.add_argument("--macro", type=int, default=6)
    p.add_argument("--vel-bin", type=float, default=0.25)
    p.add_argument("--h-weight", type=float, default=2.0)
    p.add_argument("--max-expansions", type=int, default=150000)
    p.add_argument("--max-frames", type=int, default=2600)
    p.add_argument("--clinch-actions", action="store_true",
                   help="add composite CLINCHER (Corner Double Jump) actions to "
                        "the forward search — frame-precise, momentum-conserving "
                        "CDJs the macro-hold vocabulary cannot express (for CDJ-"
                        "chain levels like clincher; pair with --enriched-key)")
    p.add_argument("--w-risk", type=float, default=None)
    p.add_argument("--jump-tax", type=float, default=None,
                   help="additive per-jump frame cost (discourages wall-climb "
                        "spam; None => planner default DEFAULT_JUMP_TAX)")
    p.add_argument("--momentum-model", choices=["legacy", "tangential"],
                   default="legacy",
                   help="velocity representation: legacy world-vx/vy, or "
                        "tangential along-surface + curved/slope node extraction")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--enriched-key", action="store_true",
                   help="add the ninja's internal jump-hold timer / state / "
                        "airborne flag to the search dedup key so frame-precise "
                        "variable-height jump tricks (air-launch / CDJ) are "
                        "findable (default off => byte-parity; for trick levels)")
    p.add_argument("--clean", action="store_true",
                   help="clean gameplay capture: ninja animation on, no route/"
                        "mine-radius/HUD overlay")
    p.add_argument("--gold", action="store_true",
                   help="all-gold mode: collect every gold (per-gold d*-guided "
                        "physics legs + backtracking) before finishing; draws a "
                        "live GOLD n/total HUD on the replay")
    p.add_argument("--leg-expansions", type=int, default=80000,
                   help="per-leg A* budget (--gold / --door-chase modes)")
    p.add_argument("--final-expansions", type=int, default=200000,
                   help="switch/exit-leg A* budget (--gold / --door-chase modes)")
    p.add_argument("--door-chase", action="store_true",
                   help="locked-door mode: open the planner's NECESSARY door "
                        "switches (faithful per-config d* legs) then reach the "
                        "exit. Required for door-gated levels (e.g. squiggly)")
    p.add_argument("--native-faithful", action="store_true",
                   help="render by puppeting the Python renderer from the native "
                        "sim each frame (eliminates the ~1-ULP nsim drift that "
                        "loses long-route wins). Recommended for long routes")
    p.add_argument("--title", default=None, help="HUD title (e.g. level name)")
    args = p.parse_args()

    if args.door_chase:
        print(f"[dstar_guided_spike --door-chase] {os.path.basename(args.map)}  "
              f"macro={args.macro} w={args.h_weight} "
              f"legs={args.leg_expansions}exp final={args.final_expansions}exp")
        res = plan_door_chase(
            args.map, macro=args.macro, vel_bin=args.vel_bin,
            h_weight=args.h_weight if args.h_weight != 2.0 else 2.5,
            leg_expansions=args.leg_expansions if args.leg_expansions != 80000 else 200000,
            final_expansions=args.final_expansions if args.final_expansions != 200000 else 400000,
            max_frames=args.max_frames if args.max_frames > 2600 else 4000,
            w_risk=args.w_risk, jump_tax=args.jump_tax,
            momentum_model=args.momentum_model, enriched_key=args.enriched_key,
        )
        print(f"  lattice_available={res['lattice_available']}  WON={res['won']}  "
              f"necessary_order={res.get('necessary_order')}  "
              f"frames={res.get('frames')}  wall={res['wall_time_s']:.1f}s")
        for leg in res.get("legs", []):
            print(f"    leg {leg}")
        if res["won"] and args.output:
            won = render_win_video(
                args.map, res["actions"], res["macro"], args.output, fps=args.fps,
                overlay=not args.clean, animate=args.clean,
                title=args.title or os.path.basename(args.map.rstrip("/")),
                native_faithful=args.native_faithful)
            print(f"  render replay won={won}  ->  {args.output}" if won
                  else "  render replay did NOT win — try --native-faithful")
        return 0 if res["won"] else 2

    if args.gold:
        print(f"[dstar_guided_spike --gold] {os.path.basename(args.map)}  "
              f"macro={args.macro} w={args.h_weight} "
              f"legs={args.leg_expansions}exp final={args.final_expansions}exp")
        res = plan_gold_chase(
            args.map, macro=args.macro, vel_bin=args.vel_bin, h_weight=args.h_weight,
            leg_expansions=args.leg_expansions, final_expansions=args.final_expansions,
            max_frames=args.max_frames if args.max_frames > 2600 else 6000,
            w_risk=args.w_risk, jump_tax=args.jump_tax,
            momentum_model=args.momentum_model, enriched_key=args.enriched_key,
        )
        print(f"  lattice_available={res['lattice_available']}  WON={res['won']}  "
              f"gold={res['gold_collected']}/{res['gold_total']}  "
              f"frames={res.get('frames')} -> {res.get('frames_shortcut')} (shortcut)  "
              f"sacrificed={res['sacrificed_gold']}  wall={res['wall_time_s']:.1f}s")
        if res["won"] and args.output:
            won = render_win_video(
                args.map, res["actions"], res["macro"], args.output, fps=args.fps,
                overlay=not args.clean, animate=args.clean, gold_hud=True,
                title=args.title or os.path.basename(args.map.rstrip("/")),
                frame_tape=res.get("frame_tape"), native_faithful=args.native_faithful)
            print(f"  render replay won={won} gold-HUD  ->  {args.output}" if won
                  else "  render replay did NOT win (non-determinism?) — no video")
        return 0 if res["won"] else 2

    print(f"[dstar_guided_spike] {os.path.basename(args.map)}  "
          f"macro={args.macro} w={args.h_weight} "
          f"budget={args.max_expansions}exp/{args.max_frames}f")
    res = plan_dstar_guided(
        args.map, macro=args.macro, vel_bin=args.vel_bin, h_weight=args.h_weight,
        max_expansions=args.max_expansions, max_frames=args.max_frames,
        w_risk=args.w_risk, jump_tax=args.jump_tax,
        momentum_model=args.momentum_model, enriched_key=args.enriched_key,
        clinch_actions=args.clinch_actions,
    )
    print(f"  lattice_available={res['lattice_available']}  "
          f"SOLVED={res['solved']}  frames={res.get('frames')}  "
          f"expansions={res['expansions']}  death_branches={res['death_branches']}  "
          f"wall={res['wall_time_s']:.1f}s")
    if res["solved"] and args.output:
        won = render_win_video(args.map, res["actions"], res["macro"], args.output,
                               fps=args.fps, overlay=not args.clean, animate=args.clean,
                               native_faithful=args.native_faithful,
                               frame_tape=res.get("frame_tape"))
        print(f"  render replay won={won}  ->  {args.output}" if won
              else f"  render replay did NOT win (non-determinism?) — no clean video")
    return 0 if res["solved"] else 2


if __name__ == "__main__":
    sys.exit(main())
