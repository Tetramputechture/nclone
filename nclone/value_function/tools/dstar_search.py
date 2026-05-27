#!/usr/bin/env python3
"""d*-guided forward search: frame-optimal, dense-mine + distant-goal capable.

Unifies the two halves of the planner:

  * The velocity-augmented surface-graph ``d*`` (backward Dijkstra) is a GLOBAL
    shortest path over (position, velocity) — it already trades early speed for
    late speed to minimise TOTAL frames (long-horizon velocity optimisation),
    to bin resolution.  But it can't *thread* dense mine clusters (the discrete
    primitive sweep is incomplete there) and its routes need careful execution.

  * A forward A* on the REAL native sim threads dense mines (real-death pruning),
    has EXACT continuous velocity in its state (so it is velocity-optimal, not
    bin-limited), and the produced tape IS the execution (no arc-follow drift).
    Its weakness was a weak Euclidean heuristic that blows up over distance.

This tool uses ``d*`` as the forward search's heuristic (Euclidean fallback where
``d*`` is ∞, i.e. inside a dense cluster the lattice can't reach).  The tight,
velocity-aware ``d*`` focuses the search so it stays tractable on DISTANT goals,
while the real-sim forward search handles DENSE mines and exact velocity.  Mines
are continuous (real ``overlap_circle_vs_circle`` death — never grid-snapped).

``h_weight=1.0`` treats ``d*`` as guidance (near-optimal, fast); raise for
faster/looser, lower toward an admissible Euclidean bound for stricter optimality
(slower).  True frame-optimality beyond the velocity-dedup resolution is the
distillation/RL target.

Usage::

    python -m nclone.value_function.tools.dstar_search --map nclone/test_maps/minefield
    python -m nclone.value_function.tools.dstar_search --map "nclone/maps/official/SL/008 the machine"
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

from ...constants.physics_constants import MAX_HOR_SPEED
from ..surface_graph import find_nearest_point, num_velocity_bins, node_id
from .physics_route_spike import PhysicsRouteSpike
from .lattice_audit import LatticeAudit


def _make_dstar_lookup(graph, d_comb, d_ex):
    """Return ``h(x, y, switch_done) -> frames`` using the lattice d* field.

    Looks up the nearest surface point and takes the min d* over its velocity
    bins (an optimistic, ~admissible position heuristic).  ``∞`` when the nearest
    surface region is lattice-unreachable (e.g. deep in a dense cluster) so the
    caller can fall back to a Euclidean bound.
    """
    max_bins = graph.max_vel_bins
    cache: dict = {}

    def _best(x, y, d_star, tag):
        key = (int(x / 6.0), int(y / 6.0), tag)
        c = cache.get(key)
        if c is not None:
            return c
        pt = find_nearest_point(graph, x, y, surface_type=None)
        if pt is None:
            cache[key] = float("inf")
            return float("inf")
        sp = graph.surface_points[pt]
        best = float("inf")
        for vb in range(num_velocity_bins(sp.surface_type)):
            v = float(d_star[node_id(pt, vb, max_bins)])
            if v < best:
                best = v
        cache[key] = best
        return best

    def h(x, y, switch_done):
        return _best(x, y, d_ex if switch_done else d_comb,
                     1 if switch_done else 0)

    return h


class DStarGuidedSpike(PhysicsRouteSpike):
    """PhysicsRouteSpike with the lattice d* as its A* heuristic."""

    def attach_dstar(self, dstar_h) -> None:
        self._dstar_h = dstar_h

    def _h(self, x: float, y: float, switch_done: bool) -> float:
        val = self._dstar_h(x, y, switch_done)
        if math.isfinite(val):
            return val
        # Euclidean lower bound (frames) where the lattice can't reach.
        gx, gy = (self.exit if switch_done else self.switch)
        d = math.hypot(x - gx, y - gy)
        if not switch_done:
            d += math.hypot(self.switch[0] - self.exit[0],
                            self.switch[1] - self.exit[1])
        return d / MAX_HOR_SPEED


class SegmentedDStarExecutor:
    """Execute the lattice's velocity-optimal route via short per-segment searches.

    Global forward search blows up on distant goals (route-length x branching),
    but the lattice backward Dijkstra gives the velocity-optimal ROUTE cheaply.
    So we keep the lattice route as a waypoint sequence and run a SHORT bounded
    forward A* to reach each successive waypoint — each segment is local
    (tractable), threads dense mines (real-death pruning), and has exact
    continuous velocity.  Segment search steps with ``tick_physics_mines_only``
    (entity-safe; ``save_ninja_state`` gives O(1) restore between A* nodes); the
    actions found are then applied to the MAIN sim with real ``tick`` (identical
    trajectory, but switch/exit actually activate).
    """

    def __init__(self, map_path: str, workers: int = 4):
        import nclone_native as Nnative
        from .physics_route_spike import _load_map_data, _goal_positions, _MINE_STATE_DEADLY
        self.audit = LatticeAudit(map_path, workers=workers)
        self.pc = self.audit.precompute()
        self.map_data = _load_map_data(map_path)
        self.main = Nnative.Simulator.load(self.map_data)
        self.search = Nnative.Simulator.load(self.map_data)
        self.switch = self.audit.switch
        self.exit = self.audit.exit
        self.deadly = self.audit.deadly

    def _segkey(self, x, y, vx, vy, cell=10.0, vb=0.4):
        return (round(x / cell), round(y / cell), round(vx / vb), round(vy / vb))

    def _search_segment(self, start_blob, tx, ty, tol, max_exp=12000,
                        macro=6, max_macros=80, w=1.0):
        """Bounded forward A* on the search sim from ``start_blob`` to within
        ``tol`` px of (tx, ty).  Returns the macro-action list or None."""
        from .physics_route_spike import ACT
        import heapq
        sim = self.search
        inv_v = 1.0 / MAX_HOR_SPEED

        def heur(x, y):
            return math.hypot(x - tx, y - ty) * inv_v

        sim.restore_ninja_state(start_blob)
        sx, sy = sim.get_ninja_pos()
        svx, svy = sim.get_ninja_velocity()
        if math.hypot(sx - tx, sy - ty) < tol:
            return []
        start = (w * heur(sx, sy), 0.0, 0, start_blob, [])
        frontier = [start]
        best_g = {self._segkey(sx, sy, svx, svy): 0.0}
        ctr = 1
        exp = 0
        while frontier and exp < max_exp:
            f, g, _, blob, acts = heapq.heappop(frontier)
            exp += 1
            if len(acts) >= max_macros:
                continue
            for a in ACT:
                sim.restore_ninja_state(blob)
                h, j = ACT[a]
                dead = False
                for _ in range(macro):
                    sim.tick_physics_mines_only(h, j)
                    if sim.has_died():
                        dead = True
                        break
                if dead:
                    continue
                x, y = sim.get_ninja_pos()
                vx, vy = sim.get_ninja_velocity()
                if math.hypot(x - tx, y - ty) < tol:
                    return acts + [a]
                k = self._segkey(x, y, vx, vy)
                ng = g + macro
                if ng < best_g.get(k, float("inf")):
                    best_g[k] = ng
                    nb = sim.save_ninja_state()
                    heapq.heappush(
                        frontier,
                        (ng + w * heur(x, y), ng, ctr, nb, acts + [a]))
                    ctr += 1
        return None

    def _waypoints(self):
        """Lattice route -> ordered (x, y, tol) waypoints; switch/exit entities
        are the phase-final targets (contact ranges as tol)."""
        from ...constants.physics_constants import (
            NINJA_RADIUS, EXIT_SWITCH_RADIUS, EXIT_DOOR_RADIUS)
        phase0, phase1, _info = self.audit.build_route(self.pc)
        wps = []
        for arc in phase0[:-1]:
            wps.append((arc.landing_x, arc.landing_y, 14.0))
        wps.append((self.switch[0], self.switch[1],
                    NINJA_RADIUS + EXIT_SWITCH_RADIUS))  # hit the switch
        for arc in phase1[:-1]:
            wps.append((arc.landing_x, arc.landing_y, 14.0))
        wps.append((self.exit[0], self.exit[1],
                    NINJA_RADIUS + EXIT_DOOR_RADIUS))     # reach the exit
        return wps

    def plan_tape(self, macro: int = 6) -> dict:
        from .physics_route_spike import ACT, _switch_done
        wps = self._waypoints()
        self.main.fast_reset()
        # settle to grounded
        for _ in range(20):
            if self.main.get_ninja_state_code() in (0, 1, 2):
                vx, vy = self.main.get_ninja_velocity()
                if abs(vy) < 0.01:
                    break
            self.main.tick(0, 0)
        tape = []
        won = died = False
        switch_frame = None
        min_clear = float("inf")
        f = 0
        reached = 0
        for (tx, ty, tol) in wps:
            if won or died:
                break
            blob = self.main.save_ninja_state()
            seg = self._search_segment(blob, tx, ty, tol, macro=macro)
            if seg is None:
                break  # segment unreachable in budget
            for a in seg:
                h, j = ACT[a]
                for _ in range(macro):
                    self.main.tick(h, j)
                    tape.append((h, j))
                    f += 1
                    x, y = self.main.get_ninja_pos()
                    for (mx, my) in self.deadly:
                        c = math.hypot(x - mx, y - my)
                        if c < min_clear:
                            min_clear = c
                    if switch_frame is None and _switch_done(self.main):
                        switch_frame = f
                    if self.main.has_won():
                        won = True
                    if self.main.has_died():
                        died = True
                    if won or died:
                        break
                if won or died:
                    break
            reached += 1
        return {"won": won, "died": died, "frames": f, "tape": tape,
                "switch_frame": switch_frame, "min_clear": min_clear,
                "reached": reached, "n_waypoints": len(wps)}


def render_dstar_video(map_path: str, output: str, workers: int = 4,
                       h_weight: float = 1.0, max_expansions: int = 60000,
                       fps: int = 60, overlay: bool = True) -> dict:
    """Plan with the d*-guided forward search, then render a Python-validated
    completion video.  Handles dense mine fields the macro lattice can't."""
    import os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import numpy as np
    import imageio
    from ...nplay_headless import NPlayHeadless
    from .route_video import _to_hwc, _draw_overlay
    from .physics_route_spike import ACT

    name = os.path.basename(map_path.rstrip("/"))
    print(f"[{name}] planning d*-guided forward search...")
    audit = LatticeAudit(map_path, workers=workers)
    pc = audit.precompute()
    graph, d_comb, d_ex = pc["graph"], pc["d_comb"], pc["d_ex"]
    sp = DStarGuidedSpike(map_path, h_weight=h_weight, max_expansions=max_expansions)
    sp.attach_dstar(_make_dstar_lookup(graph, d_comb, d_ex))
    res = sp.plan()
    if not res["solved"]:
        print(f"  d*-guided search found no route ({res['expansions']} expansions) "
              f"— not rendering.")
        return res
    macro = sp.macro
    deadly = sp.deadly0
    full_traj = res["trajectory"]
    total = len(full_traj) - 1
    print(f"  solved {res['frames']}f, {res['expansions']} expansions; rendering...")

    def _clear(x, y):
        return min((math.hypot(x - mx, y - my) for (mx, my) in deadly),
                   default=float("inf"))

    nh = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    nh.load_map(map_path)
    nh.reset()

    def _cap(fn):
        raw = _to_hwc(nh.render())
        if not overlay:
            return raw
        x, y = nh.ninja_position()
        vx, vy = nh.ninja_velocity()
        phase = "-> EXIT" if nh.exit_switch_activated() else "-> switch"
        return _draw_overlay(raw, full_traj, fn, (x, y), math.hypot(vx, vy),
                             _clear(x, y), fn, total, phase, deadly)

    frames = [_cap(0)]
    won = died = False
    fn = 0
    for a in res["actions"]:
        h, j = ACT[a]
        for _ in range(macro):
            nh.tick(h, j)
            fn += 1
            frames.append(_cap(fn))
            if nh.ninja_has_won():
                won = True
                break
            if nh.ninja_has_died():
                died = True
                break
        if won or died:
            break

    if not won:
        print(f"  render sim did NOT complete (died={died}) — not writing video "
              f"(likely an out-of-scope dynamic entity).")
        return {"won": False, "died": died, "output": None}
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    frames.extend([frames[-1]] * fps)
    imageio.mimsave(output, frames, fps=fps, macro_block_size=1)
    print(f"  WON | {len(frames)} frames @ {fps}fps -> {output}")
    return {"won": True, "n_frames": len(frames), "route_frames": res["frames"],
            "output": output}


def run(map_path: str, workers: int = 4, h_weight: float = 1.0,
        max_expansions: int = 60000, compare_euclidean: bool = False) -> dict:
    name = os.path.basename(map_path.rstrip("/"))
    print(f"[{name}] precomputing lattice d* field...")
    audit = LatticeAudit(map_path, workers=workers)
    pc = audit.precompute()
    graph, d_comb, d_ex = pc["graph"], pc["d_comb"], pc["d_ex"]
    import numpy as np
    print(f"  d* finite at: {int(np.isfinite(d_comb).sum())}/{len(d_comb)} nodes "
          f"(combined), {int(np.isfinite(d_ex).sum())}/{len(d_ex)} (exit)")

    dstar_h = _make_dstar_lookup(graph, d_comb, d_ex)
    sp = DStarGuidedSpike(map_path, h_weight=h_weight,
                          max_expansions=max_expansions)
    sp.attach_dstar(dstar_h)
    print(f"  d*-guided forward A* (w={h_weight})...")
    t0 = time.time()
    res = sp.plan()
    dt = time.time() - t0
    print(f"  solved={res['solved']} frames={res['frames']} "
          f"expansions={res['expansions']} deaths_pruned={res['death_branches']} "
          f"time={dt:.1f}s")

    if compare_euclidean:
        eu = PhysicsRouteSpike(map_path, h_weight=1.25,
                               max_expansions=max_expansions)
        t1 = time.time()
        eures = eu.plan()
        print(f"  [Euclidean spike w=1.25] solved={eures['solved']} "
              f"frames={eures['frames']} expansions={eures['expansions']} "
              f"time={time.time()-t1:.1f}s")
    return res


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--h-weight", type=float, default=1.0)
    p.add_argument("--max-expansions", type=int, default=60000)
    p.add_argument("--compare-euclidean", action="store_true")
    p.add_argument("--segmented", action="store_true",
                   help="run the segmented executor (lattice waypoints + short "
                        "per-segment forward searches) for distant goals")
    p.add_argument("--video", default=None,
                   help="render a completion mp4 to this path (d*-guided plan)")
    args = p.parse_args()
    if args.video:
        render_dstar_video(args.map, args.video, workers=args.workers,
                           h_weight=args.h_weight,
                           max_expansions=args.max_expansions)
        return 0
    if args.segmented:
        import time as _t
        name = os.path.basename(args.map.rstrip("/"))
        print(f"[{name}] segmented d*-guided executor...")
        t0 = _t.time()
        ex = SegmentedDStarExecutor(args.map, workers=args.workers)
        res = ex.plan_tape()
        verdict = "WON" if res["won"] else ("DIED" if res["died"] else "DID-NOT-FINISH")
        print(f"  {verdict} @ {res['frames']}f  waypoints {res['reached']}/{res['n_waypoints']}  "
              f"switch@{res['switch_frame']}  min_mine_clear={res['min_clear']:.1f}px  "
              f"({_t.time()-t0:.1f}s)")
        return 0
    run(args.map, workers=args.workers, h_weight=args.h_weight,
        max_expansions=args.max_expansions,
        compare_euclidean=args.compare_euclidean)
    return 0


if __name__ == "__main__":
    sys.exit(main())
