"""All-gold lattice route planner (Phase 4 of the gold-collection objective).

Plans a route that collects ALL gold on the level AND completes it (switch then
exit), in close to the fewest frames. Demonstrates "opportunity cost": the route
spends extra frames detouring to gold rather than beelining to the exit, and it
orders the gold (cluster-aware) so the detour is as cheap as possible.

Approach (see docs/GOLD_COLLECTION_HANDOFF.md, Phase 4):
  1. Cluster the gold (single-linkage, eps=48px) — corpus is highly clustered.
  2. Order clusters with a cheap centroid TSP (nearest-neighbour + 2-opt) from
     spawn; order gold within each cluster by nearest-neighbour. This is the
     cheap-ordering / exact-legs split: ordering uses Euclidean, the LEGS use the
     real lattice d*. (Cluster counts are small — p50=4 — so NN+2-opt is near
     optimal; exact Held-Karp is a drop-in for <=~12 clusters.)
  3. For each gold in the visiting order, run a backward Dijkstra on the lattice
     graph to that gold's contact nodes and extract the leg route from the
     current node (velocity-continuous at the shared node). Then legs to the
     switch and the exit.
  4. Execute via LatticeAudit.replay_micro_search (closed-loop + micro-search on
     the real native sim), which collects gold as a physical side effect and
     returns an executable tape.

The planner reuses the existing lattice machinery (LatticeAudit, surface graph,
route extraction, micro-search executor) unchanged.

Usage:
    .venv/bin/python -m nclone.value_function.tools.gold_route \
        "nclone/maps/official/SL/001 take the easy way out" [--w-risk 0.5]
"""
from __future__ import annotations

import argparse
import math
import os
import time
from typing import List, Optional, Tuple

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from ..edge_enumeration import enumerate_edges
from ..route_extraction import extract_route, find_nearest_start_node
from ..surface_graph import (
    _build_reverse_adjacency,
    dijkstra_backward_with_tree,
    find_goal_nodes,
    node_id,
    num_velocity_bins,
)
from ...constants.physics_constants import GOLD_RADIUS, NINJA_RADIUS, MAX_HOR_SPEED
from .lattice_audit import LatticeAudit

GOLD_TYPE = 2
GOLD_CLUSTER_EPS = 48.0
# Per-gold time-saving exchange rate (frames). Mirror of
# reward_constants.GOLD_FRAME_VALUE — the planner solves the prize-collecting
# (orienteering) route: a gold cluster is visited only if its marginal detour
# costs fewer frames than GOLD_FRAME_VALUE * (gold in the cluster).
GOLD_FRAME_VALUE = 120.0
# Contact margin: route to nodes a bit INSIDE the collection radius so the
# closed-loop executor (which advances within ~12px of a target node) still
# physically overlaps the gold (collected when centres within NINJA+GOLD radius).
GOLD_CONTACT_RANGE = NINJA_RADIUS + GOLD_RADIUS


def gold_positions(map_data) -> List[Tuple[float, float]]:
    """Read gold (x, y) from a python sim. The NATIVE sim excludes gold from
    get_state().entities (it's optional / unused by RL), so we load the python
    Simulator, which keeps gold in entity_dic[2]."""
    from nclone.nplay_headless import NPlayHeadless
    nh = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    nh.load_map_from_map_data(list(map_data))
    nh.reset()
    return [(float(g.xpos), float(g.ypos)) for g in nh.sim.entity_dic.get(GOLD_TYPE, [])]


def cluster_groups(points: List[Tuple[float, float]], eps: float) -> List[List[int]]:
    """Single-linkage clusters as lists of indices into ``points``."""
    n = len(points)
    if n == 0:
        return []
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    grid = {}
    for i, (x, y) in enumerate(points):
        grid.setdefault((int(x // eps), int(y // eps)), []).append(i)
    eps2 = eps * eps
    for i, (x, y) in enumerate(points):
        gx, gy = int(x // eps), int(y // eps)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((gx + dx, gy + dy), ()):
                    if j <= i:
                        continue
                    if (x - points[j][0]) ** 2 + (y - points[j][1]) ** 2 <= eps2:
                        parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _nn_order(start, pts, idxs):
    """Nearest-neighbour visiting order over idxs (indices into pts), from start."""
    remaining = list(idxs)
    order = []
    cur = start
    while remaining:
        j = min(remaining, key=lambda k: _dist(cur, pts[k]))
        order.append(j)
        cur = pts[j]
        remaining.remove(j)
    return order


def _two_opt(order, pts, start, iters=1000):
    """2-opt improve a tour (open path from `start` through pts[order])."""
    def path_len(o):
        d = _dist(start, pts[o[0]]) if o else 0.0
        for a, b in zip(o, o[1:]):
            d += _dist(pts[a], pts[b])
        return d

    best = order[:]
    best_len = path_len(best)
    improved = True
    n = len(best)
    while improved and iters > 0:
        improved = False
        for i in range(n - 1):
            for k in range(i + 1, n):
                cand = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                cl = path_len(cand)
                if cl + 1e-6 < best_len:
                    best, best_len = cand, cl
                    improved = True
            iters -= 1
            if iters <= 0:
                break
    return best


def _gold_goal_nodes(graph, gx, gy, rng=GOLD_CONTACT_RANGE) -> List[int]:
    rng2 = rng * rng
    out: List[int] = []
    for i, sp in enumerate(graph.surface_points):
        if (sp.x - gx) ** 2 + (sp.y - gy) ** 2 <= rng2:
            for vb in range(num_velocity_bins(sp.surface_type)):
                out.append(node_id(i, vb, graph.max_vel_bins))
    return out


def reconstruct_air_arc(graph, map_data, src_node, gx, gy, rng=GOLD_CONTACT_RANGE,
                        momentum_model: str = "legacy"):
    """Recover the fly-through trajectory: the arc launched from ``src_node`` whose
    path passes within ``rng`` of the gold at (gx, gy). Returns a list of (x, y)
    up to the contact frame, or None.

    Mirrors the enumerator's per-surface launch primitives EXACTLY so the
    reconstructed arc matches the edge the planner actually flew:
      - FLOOR: floor jumps (h_during x hold x post-release) + a long run/fall.
      - WALL / SLIDE: WALL JUMPS (h_input x hold, post-release input = h_input —
        the kick-off-the-wall launch, not a floor jump) + a wall-slide fall.
      - LAUNCH_PAD: a fixed-boost kick.
    Velocity init honours the same ``momentum_model`` (tangential along-surface or
    legacy world vx/vy) as the enumerator, and keeps the earliest-contact arc."""
    import nclone_native as Nnative
    from ..surface_graph import node_id_to_parts, velocity_bin_centers
    from ..surface_extraction import (
        SURFACE_FLOOR, SURFACE_WALL, SURFACE_SLIDE, SURFACE_LAUNCH_PAD)
    from ..edge_enumeration import (
        _simulate_arc, _simulate_jump_arc, _LAUNCH_PAD_BOOST_MAG, _DEATH_STATES,
        JUMP_HOLD_DURATIONS, H_INPUTS, POST_RELEASE_INPUTS,
        MAX_JUMP_DURATION, MAX_ARC_FRAMES_DEFAULT,
    )
    pt, vb = node_id_to_parts(src_node, graph.max_vel_bins)
    sp = graph.surface_points[pt]
    st = int(sp.surface_type)
    vel = float(velocity_bin_centers(st)[vb])
    # Match _enumerate_node_edges' velocity init exactly (incl. tangential model).
    if momentum_model == "tangential" and st != SURFACE_LAUNCH_PAD:
        vx0 = vel * sp.tangent_x + 0.0
        vy0 = vel * sp.tangent_y + 0.0
    elif st == SURFACE_FLOOR:
        vx0, vy0 = vel, 0.0
    elif st == SURFACE_LAUNCH_PAD:
        vx0, vy0 = sp.normal_x * _LAUNCH_PAD_BOOST_MAG, sp.normal_y * _LAUNCH_PAD_BOOST_MAG
    else:  # SURFACE_WALL or SURFACE_SLIDE: bin holds vy
        vx0, vy0 = 0.0, vel

    sim = Nnative.Simulator.load(list(map_data))
    snap = sim.save_ninja_state()
    goals = [(float(gx), float(gy), float(rng * rng))]
    best, best_f = None, None

    def _keep(wps, hit_f):
        nonlocal best, best_f
        if not math.isfinite(hit_f) or (best_f is not None and hit_f >= best_f):
            return
        best_f = hit_f
        best = [(sp.x, sp.y)] + [(x, y) for (fr, x, y, _vx, _vy) in wps if fr <= hit_f]

    def _jump(h_during, hold_dur, h_after):
        sim.restore_ninja_state(snap)
        sim.inject_ninja_for_value_iteration(sp.x, sp.y, vx0, vy0, 0)
        sim.tick_tiles_only(h_during, 1)  # frame 1: initiate (floor or wall) jump
        if sim.get_ninja_state_code() in _DEATH_STATES:
            return
        gacc = [float("inf")]
        _l, _f, wps = _simulate_jump_arc(
            sim, h_during, min(hold_dur - 1, MAX_JUMP_DURATION - 1), h_after,
            collect_waypoints=True, waypoint_interval=1,
            max_arc_frames=MAX_ARC_FRAMES_DEFAULT, goals=goals, goal_acc=gacc)
        _keep(wps, gacc[0])

    def _glide(h_input):
        sim.restore_ninja_state(snap)
        sim.inject_ninja_for_value_iteration(sp.x, sp.y, vx0, vy0, 0)
        gacc = [float("inf")]
        _l, _f, wps = _simulate_arc(
            sim, h_input, 0, MAX_ARC_FRAMES_DEFAULT, collect_waypoints=True,
            waypoint_interval=1, goals=goals, goal_acc=gacc)
        _keep(wps, gacc[0])

    if st == SURFACE_FLOOR:
        for h_during in H_INPUTS:
            for hold_dur in JUMP_HOLD_DURATIONS:
                for h_after in POST_RELEASE_INPUTS:
                    _jump(h_during, hold_dur, h_after)
        for h_input in H_INPUTS:
            _glide(h_input)  # run / edge-fall
    elif st == SURFACE_WALL or st == SURFACE_SLIDE:
        # Wall jump: post-release air-control == h_input (enumerator sets
        # h_after = h_input).  "regular"/"slide" templates are identical physics
        # for a fixed node (only the baked-in vy differs), so one pass suffices.
        for h_input in H_INPUTS:
            for hold_dur in JUMP_HOLD_DURATIONS:
                _jump(h_input, hold_dur, h_input)
        for h_input in H_INPUTS:
            _glide(h_input)  # wall-slide fall
    elif st == SURFACE_LAUNCH_PAD:
        for h_input in POST_RELEASE_INPUTS:
            _glide(h_input)
    else:
        for h_input in H_INPUTS:
            _glide(h_input)
    return best


def _terminal_node(pred: np.ndarray, start: int) -> int:
    """Walk the predecessor tree from start to the goal node (pred == -1)."""
    node = start
    guard = 0
    while 0 <= node < len(pred) and pred[node] != -1 and guard < 200000:
        node = int(pred[node])
        guard += 1
    return node


class GoldRoutePlanner:
    def __init__(self, map_path=None, map_data=None, name=None,
                 w_risk: float = 0.5, jump_tax: float = 0.0, workers: int = 8):
        self.audit = LatticeAudit(
            map_path=map_path, map_data=map_data, name=name,
            w_risk=w_risk, jump_tax=jump_tax, workers=workers, use_native=True,
        )
        self.pc = self.audit.precompute(use_cache=True)
        self.graph = self.pc["graph"]
        self.rev_adj = _build_reverse_adjacency(self.graph)
        self.gold = gold_positions(self.audit.map_data)
        self.spawn = self.audit.spawn
        self.switch = self.audit.switch
        self.exit = self.audit.exit
        self._gold_air = None  # lazily computed {gold_index: {src_node: frames}}

    def _dstar(self, goal_nodes, goal_init_costs=None):
        return dijkstra_backward_with_tree(
            self.graph, goal_nodes, rev_adj=self.rev_adj,
            goal_init_costs=goal_init_costs)

    def gold_air_contacts(self):
        """Per-gold air-contacts: {gold_index: {src_node: frames_to_fly_through}}.

        Most corpus gold floats off any walkable surface, so it has no surface
        contact node. The enumerator already detects when an arc's *trajectory*
        passes within collection range of a goal (the same mechanism the planner
        uses for air-reachable switch/exit). We run ONE extra enumeration pass
        with every gold as a goal to harvest those air-contacts; seeding them into
        each gold leg's backward Dijkstra makes airborne gold reachable.

        Isolated from the cached precompute (which only carries switch/exit goals)
        so the corpus/training d* path is untouched. Native + fast.
        """
        if self._gold_air is not None:
            return self._gold_air
        if not self.gold:
            self._gold_air = {}
            return self._gold_air
        goal_positions = [(gx, gy, GOLD_CONTACT_RANGE) for (gx, gy) in self.gold]
        mine_pos = (
            np.asarray(self.audit.all_mines, dtype=np.float32)
            if (self.audit.all_mines and self.audit.mine_inflation) else None
        )
        stats = {}
        enumerate_edges(
            self.graph, self.audit.map_data,
            num_workers=self.audit.workers, collect_waypoints=False,
            mine_positions=mine_pos, stats_out=stats,
            goal_positions=goal_positions, use_native=self.audit.use_native,
            momentum_model=self.audit.momentum_model,
        )
        self._gold_air = stats.get("goal_contacts", {})
        return self._gold_air

    def gold_visiting_order(
        self, orienteer: bool = False, frame_value: float = GOLD_FRAME_VALUE,
    ) -> List[int]:
        """Cluster-aware visiting order of gold indices (cheap Euclidean TSP).

        When ``orienteer`` is True this becomes the prize-collecting route: a
        cluster is KEPT only if its marginal detour costs fewer frames than
        ``frame_value`` * (gold in the cluster) — the same 'is this gold worth the
        120-frame detour?' test the reward and the lattice potential apply, lifted
        to cluster granularity. ``orienteer=False`` is the legacy all-gold tour.

        The marginal detour of a cluster centroid ``c`` from the current tour tip
        ``cur`` is estimated against beelining to the switch:
        ``dist(cur, c) + dist(c, switch) - dist(cur, switch)`` converted to frames
        at max horizontal speed — extra travel over going straight on toward the
        objective. Clusters off the route get a large detour and are dropped;
        clusters on the way get a small one and are kept.
        """
        groups = cluster_groups(self.gold, GOLD_CLUSTER_EPS)
        if not groups:
            return []
        centroids = [
            (sum(self.gold[i][0] for i in g) / len(g),
             sum(self.gold[i][1] for i in g) / len(g))
            for g in groups
        ]
        # Order clusters: NN from spawn, then 2-opt on centroids.
        corder = _nn_order(self.spawn, centroids, list(range(len(groups))))
        corder = _two_opt(corder, centroids, self.spawn)
        # Flatten: within each cluster, NN order the gold from the arrival point.
        speed = max(MAX_HOR_SPEED, 1e-6)
        order: List[int] = []
        self._skipped_clusters: List[int] = []
        cur = self.spawn
        for ci in corder:
            g = groups[ci]
            if orienteer:
                c = centroids[ci]
                detour_px = _dist(cur, c) + _dist(c, self.switch) - _dist(cur, self.switch)
                detour_frames = max(0.0, detour_px) / speed
                if detour_frames >= frame_value * len(g):
                    self._skipped_clusters.append(ci)
                    continue  # not worth it — leave cur where it is
            sub = _nn_order(cur, self.gold, g)
            order.extend(sub)
            cur = self.gold[sub[-1]]
        return order

    def plan(self, orienteer: bool = True,
             frame_value: float = GOLD_FRAME_VALUE) -> dict:
        """Build the gold-augmented ArcTarget route. Returns route + diagnostics.

        ``orienteer=True`` (default) solves the prize-collecting route: only gold
        whose detour is worth its ``frame_value``-frame credit is visited.
        ``orienteer=False`` reverts to the legacy all-gold tour.
        """
        graph = self.graph
        order = self.gold_visiting_order(orienteer=orienteer, frame_value=frame_value)
        start_node = find_nearest_start_node(
            graph, self.spawn[0], self.spawn[1], 0.0, 0.0, 1, self.pc["d_comb"])
        route = []
        reached, missed = [], []
        cur = start_node
        if cur is None:
            return {"route": [], "n_phase0": 0, "reached": [], "missed": order,
                    "error": "no spawn node"}

        # Legs to each gold in visiting order. Goal nodes = surface contact nodes
        # PLUS air-contact src nodes (arcs whose trajectory flies through the gold),
        # the latter seeded with their frames-to-contact so airborne gold is
        # reachable even with no adjacent surface.
        air = self.gold_air_contacts()
        n_air_reached = 0
        air_arcs: List[Tuple[int, int]] = []  # (gold_index, chosen air src node)
        for gi in order:
            gx, gy = self.gold[gi]
            surf = _gold_goal_nodes(graph, gx, gy)
            air_g = air.get(gi, {})
            goal_nodes = list(set(surf) | set(air_g.keys()))
            if not goal_nodes:
                missed.append(gi)
                continue
            # surface nodes cost 0; air nodes cost their frames-to-fly-through.
            init = {n: 0.0 for n in surf}
            for n, fr in air_g.items():
                init[n] = min(init.get(n, float("inf")), float(fr))
            d, pred, best = self._dstar(goal_nodes, goal_init_costs=init)
            if not np.isfinite(d[cur]):
                missed.append(gi)
                continue
            leg = extract_route(graph, d, pred, best, cur)
            route.extend(leg)
            end = _terminal_node(pred, cur)
            if end in air_g and end not in surf:
                n_air_reached += 1
                air_arcs.append((gi, end))
            cur = end
            reached.append(gi)

        # Leg to switch (closes phase 0).
        sw_nodes = find_goal_nodes(graph, self.switch[0], self.switch[1], "switch")
        d, pred, best = self._dstar(sw_nodes)
        if np.isfinite(d[cur]):
            route.extend(extract_route(graph, d, pred, best, cur))
            cur = _terminal_node(pred, cur)
        n_phase0 = len(route)

        # Phase 1: switch -> exit.
        ex_nodes = find_goal_nodes(graph, self.exit[0], self.exit[1], "exit")
        d, pred, best = self._dstar(ex_nodes)
        if np.isfinite(d[cur]):
            route.extend(extract_route(graph, d, pred, best, cur))

        n_clusters = len(cluster_groups(self.gold, GOLD_CLUSTER_EPS))
        skipped = list(getattr(self, "_skipped_clusters", []))
        if skipped:
            log = print  # keep dependency-free; surfaces the orienteering drops
            log(f"[gold_route] orienteering skipped {len(skipped)}/{n_clusters} "
                f"clusters (detour > {frame_value:.0f}f/gold): {skipped}")
        return {
            "route": route, "n_phase0": n_phase0,
            "reached": reached, "missed": missed, "n_air_reached": n_air_reached,
            "air_arcs": air_arcs, "orienteer": orienteer,
            "skipped_clusters": skipped,
            "n_gold": len(self.gold), "n_clusters": n_clusters,
        }

    def execute(self, plan: dict, max_frames: int = 4000) -> dict:
        """Run the route through the closed-loop micro-search executor."""
        route = plan["route"]
        if not route:
            return {"won": False, "tape": [], "error": "empty route"}
        ms = self.audit.replay_micro_search(route, plan["n_phase0"], max_frames=max_frames)
        return ms

    def direct_route(self):
        """Base spawn->switch->exit route (no gold) + its d* frame cost.

        The frame baseline against which the all-gold route's extra frames are
        the 'opportunity cost' of full collection.
        """
        p0, p1, info = self.audit.build_route(self.pc)
        frames = sum(a.arc_cost_frames for a in (p0 + p1))
        return p0, p1, frames

    @staticmethod
    def route_frames(route) -> float:
        return float(sum(a.arc_cost_frames for a in route))

    def execute_gold_chase(self, order: Optional[List[int]] = None,
                           max_frames: int = 5000, capture: bool = False,
                           settle: int = 6) -> dict:
        """Gold-aware closed-loop controller on the PYTHON sim (real gold physics).

        The native micro-search executor can't see gold (native excludes it), and
        open-loop tapes drift. So we control the python sim directly — it collects
        gold and renders — observing actual state each frame. Visits gold in the
        planner's frame-ordered sequence (``order``), steering toward each and
        jumping when it sits above the ninja (mid-row gold over the floor dip;
        wall-jumps emerge from physics when the ninja is against a wall), then the
        switch, then the exit. Returns an executable tape + optional rgb frames.
        """
        from nclone.nplay_headless import NPlayHeadless
        nh = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
        nh.load_map_from_map_data(list(self.audit.map_data))
        nh.reset()
        nj = nh.sim.ninja
        golds = nh.sim.entity_dic.get(GOLD_TYPE, [])
        sw, ex = self.switch, self.exit
        GROUND = {0, 1, 2}
        tape: List[Tuple[int, int]] = []
        imgs = []
        meta = []  # per-frame (collected, switch_on, won) for the video HUD

        def step(h, j):
            nh.tick(h, j)
            tape.append((int(h), int(j)))
            if capture:
                imgs.append(nh.render())
                meta.append((int(nj.gold_collected),
                             bool(nh.exit_switch_activated()),
                             bool(nh.ninja_has_won())))

        for _ in range(settle):
            step(0, 0)

        # Visiting order: planner order if given, else nearest-first greedy.
        order_q = list(order) if order is not None else None
        stuck = 0
        last_pos = (nj.xpos, nj.ypos)
        for _f in range(max_frames):
            if nh.ninja_has_won() or nh.ninja_has_died():
                break
            rem = [g for g in golds if g.active]
            if rem:
                if order_q:
                    # advance past collected; target the next planned gold by index
                    while order_q and not golds[order_q[0]].active:
                        order_q.pop(0)
                    tgt = golds[order_q[0]] if order_q else min(
                        rem, key=lambda g: (g.xpos - nj.xpos) ** 2 + (g.ypos - nj.ypos) ** 2)
                    tx, ty = tgt.xpos, tgt.ypos
                else:
                    tgt = min(rem, key=lambda g: (g.xpos - nj.xpos) ** 2 + (g.ypos - nj.ypos) ** 2)
                    tx, ty = tgt.xpos, tgt.ypos
            elif not nh.exit_switch_activated():
                tx, ty = sw
            else:
                tx, ty = ex
            dx = tx - nj.xpos
            h = 1 if dx > 2 else (-1 if dx < -2 else 0)
            need_up = ty < nj.ypos - 8
            aligned = abs(dx) < 36
            grounded = nj.state in GROUND
            if grounded:
                j = 1 if (need_up and aligned) else 0
                # if stuck against a wall (not moving horizontally), jump to climb
                if not need_up and stuck > 8:
                    j = 1
            else:
                # airborne: hold jump while still rising toward an above-target
                j = 1 if (need_up and nj.yspeed < 0) else 0
            step(h, j)
            # stuck detection (drives wall-jump escape)
            if (nj.xpos - last_pos[0]) ** 2 + (nj.ypos - last_pos[1]) ** 2 < 1.0:
                stuck += 1
            else:
                stuck = 0
            last_pos = (nj.xpos, nj.ypos)

        return {
            "won": nh.ninja_has_won(), "died": nh.ninja_has_died(),
            "frames": len(tape), "collected": int(nj.gold_collected),
            "total": len(golds), "tape": tape, "imgs": imgs, "meta": meta,
        }

    def sweep_order(self) -> List[int]:
        """Frame-fast gold order: sweep from the spawn side toward the exit side,
        cluster-grouped. Avoids the back-and-forth of greedy-nearest on a row."""
        order = self.gold_visiting_order()  # cluster-aware base order
        # If gold is essentially one spatial sweep (few clusters), sort by signed
        # distance along spawn->exit so the ninja never doubles back.
        ax = self.exit[0] - self.spawn[0]
        ay = self.exit[1] - self.spawn[1]
        norm = math.hypot(ax, ay) or 1.0
        ax, ay = ax / norm, ay / norm
        def proj(i):
            gx, gy = self.gold[i]
            return (gx - self.spawn[0]) * ax + (gy - self.spawn[1]) * ay
        return sorted(order, key=proj)

    def render_gold_chase_video(self, out_path: str, fps: int = 60,
                                max_frames: int = 6000, title: str = "opportunity cost") -> dict:
        """Run the gold-chase on the python sim and render the replay to mp4 with a
        HUD (live gold counter + frames). Real-physics gameplay collecting all gold."""
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw
        from .route_video import _to_hwc

        res = self.execute_gold_chase(order=self.sweep_order(), capture=True,
                                      max_frames=max_frames)
        imgs, meta = res["imgs"], res["meta"]
        total = res["total"]
        frames = []
        for img, (coll, sw, won) in zip(imgs, meta):
            im = Image.fromarray(_to_hwc(img)).convert("RGB")
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 360, 30], fill=(0, 0, 0))
            d.text((5, 2), f'"{title}"  —  all-gold route replay', fill=(255, 255, 255))
            col = (90, 255, 120) if coll == total else (255, 215, 0)
            d.text((5, 15), f"gold {coll}/{total}   switch {'ON' if sw else 'off'}"
                            f"   {'WON' if won else ''}", fill=col)
            frames.append(np.asarray(im))
        frames += [frames[-1]] * fps  # hold last frame 1s
        imageio.mimsave(out_path, frames, fps=fps, macro_block_size=1)
        return {**{k: v for k, v in res.items() if k not in ("imgs", "meta", "tape")},
                "output": out_path, "n_video_frames": len(frames)}

    def count_gold_on_tape(self, tape) -> Tuple[int, int]:
        """Replay tape on a python sim and read ninja.gold_collected (ground truth)."""
        from nclone.nplay_headless import NPlayHeadless
        nh = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
        nh.load_map_from_map_data(list(self.audit.map_data))
        nh.reset()
        for (h, j) in tape:
            nh.tick(h, j)
            if nh.ninja_has_won() or nh.ninja_has_died():
                break
        total = len(nh.sim.entity_dic.get(GOLD_TYPE, []))
        collected = int(nh.sim.ninja.gold_collected)
        return collected, total


def render_route(planner: "GoldRoutePlanner", plan: dict, out_path: str,
                 video: bool = False, fps: int = 8) -> str:
    """Render the planned all-gold route over the level (PNG, or arc-by-arc mp4).

    Draws the gold (green = a leg reached it, red = missed), spawn/switch/exit,
    and the route arcs in visiting order (cool→warm gradient = collection order),
    with an opportunity-cost HUD: all-gold frames vs direct-completion frames.
    """
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from ...nplay_headless import NPlayHeadless
    from .route_video import _to_hwc

    nh = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    nh.load_map_from_map_data(list(planner.audit.map_data))
    nh.reset()
    base = _to_hwc(nh.render())

    route = plan["route"]
    _, _, direct_frames = planner.direct_route()
    gold_frames = planner.route_frames(route)
    reached = set(plan["reached"])

    # Reconstruct the fly-through trajectories for air-collected gold so the
    # render visibly shows the route arcing up through each airborne gold.
    air_trajs = []
    for (gi, src) in plan.get("air_arcs", []):
        gx, gy = planner.gold[gi]
        tr = reconstruct_air_arc(planner.graph, planner.audit.map_data, src, gx, gy,
                                 momentum_model=planner.audit.momentum_model)
        if tr:
            air_trajs.append(tr)

    def _gradient(t):  # cool (cyan) -> warm (magenta) along the tour
        return (int(40 + 215 * t), int(220 * (1 - t) + 40 * t), int(255 * (1 - t) + 200 * t))

    def _compose(n_arcs):
        img = Image.fromarray(base).convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        # gold markers
        for gi, (gx, gy) in enumerate(planner.gold):
            col = (90, 255, 120, 255) if gi in reached else (255, 70, 70, 255)
            d.ellipse([gx - 5, gy - 5, gx + 5, gy + 5], outline=col, width=2)
        # route arcs up to n_arcs
        m = max(1, len(route) - 1)
        for i, arc in enumerate(route[:n_arcs]):
            t = i / m
            w = 3 if arc.jump_hold_frames > 0 else 2
            d.line([(arc.launch_x, arc.launch_y), (arc.landing_x, arc.landing_y)],
                   fill=_gradient(t) + (235,), width=w)
        # air-pickup fly-through trajectories (gold-yellow)
        for tr in air_trajs:
            if len(tr) >= 2:
                d.line([(x, y) for (x, y) in tr], fill=(255, 215, 0, 230), width=2)
        # spawn / switch / exit
        sx, sy = planner.spawn
        d.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], outline=(255, 230, 0, 255), width=2)
        wx, wy = planner.switch
        d.ellipse([wx - 6, wy - 6, wx + 6, wy + 6], outline=(255, 150, 0, 255), width=2)
        ex, ey = planner.exit
        d.rectangle([ex - 6, ey - 6, ex + 6, ey + 6], outline=(80, 255, 120, 255), width=2)
        img = Image.alpha_composite(img, ov)
        d2 = ImageDraw.Draw(img)
        d2.rectangle([0, 0, 520, 44], fill=(0, 0, 0, 170))
        d2.text((5, 2), f"{planner.audit.name}  |  all-gold route", fill=(255, 255, 255, 255))
        d2.text((5, 15), f"gold reached {len(reached)}/{plan['n_gold']}  "
                         f"clusters {plan['n_clusters']}", fill=(150, 255, 180, 255))
        oc = gold_frames - direct_frames
        d2.text((5, 28), f"frames: all-gold={gold_frames:.0f}  direct={direct_frames:.0f}  "
                         f"opportunity-cost=+{oc:.0f}f ({oc/max(1,direct_frames)*100:.0f}%)",
                fill=(255, 220, 150, 255))
        return np.asarray(img.convert("RGB"))

    if video:
        frames = [_compose(k) for k in range(0, len(route) + 1)]
        frames += [frames[-1]] * fps  # hold final
        imageio.mimsave(out_path, frames, fps=fps, macro_block_size=1)
    else:
        Image.fromarray(_compose(len(route))).save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map", help="map path")
    ap.add_argument("--w-risk", type=float, default=0.5)
    ap.add_argument("--jump-tax", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=4000)
    ap.add_argument("--render", metavar="OUT.png", help="render planned route to a PNG")
    ap.add_argument("--video", metavar="OUT.mp4", help="animate route arc-by-arc to mp4")
    args = ap.parse_args()

    t = time.time()
    planner = GoldRoutePlanner(
        map_path=args.map, w_risk=args.w_risk, jump_tax=args.jump_tax, workers=args.workers)
    print(f"level: {planner.audit.name}  gold={len(planner.gold)}  "
          f"clusters={len(cluster_groups(planner.gold, GOLD_CLUSTER_EPS))}  "
          f"(precompute {time.time()-t:.1f}s)")
    plan = planner.plan()
    _, _, direct_frames = planner.direct_route()
    gold_frames = planner.route_frames(plan["route"])
    print(f"route: {len(plan['route'])} arcs (phase0={plan['n_phase0']}), "
          f"gold legs reached={len(plan['reached'])}/{plan['n_gold']} "
          f"(of which {plan['n_air_reached']} via air-contact) "
          f"missed={len(plan['missed'])}")
    print(f"opportunity cost: all-gold route {gold_frames:.0f}f vs direct {direct_frames:.0f}f "
          f"= +{gold_frames-direct_frames:.0f}f detour to collect gold")
    if args.render:
        print("rendered:", render_route(planner, plan, args.render, video=False))
    if args.video:
        print("video:", render_route(planner, plan, args.video, video=True))
    ms = planner.execute(plan, max_frames=args.max_frames)
    coll, total = planner.count_gold_on_tape(ms.get("tape", []))
    print(f"execute: {'WON' if ms.get('won') else ('DIED' if ms.get('died') else 'noend')}"
          f"  frames={ms.get('frames')}  switch_frame={ms.get('switch_frame')}")
    print(f"GOLD COLLECTED (executed): {coll}/{total}   "
          f"({'ALL GOLD' if coll == total and total > 0 else 'partial'})")


if __name__ == "__main__":
    main()
