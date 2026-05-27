#!/usr/bin/env python3
"""#3: Backward-Dijkstra lattice prototype — frame-optimal cost-to-go d*(x,v).

⚠️ EXPERIMENTAL / KNOWN-UNSOUND. This prototype revealed that a velocity-binned
forward search does NOT yield a provably true optimum: distinct exact states
sharing one (cell, vel_bin) key are not interchangeable, so bin dedup breaks
Dijkstra optimality (e.g. long-jump-reachable: Dijkstra=348f but weighted-A*=198f).
The route it extracts can fail replay validation (mine-clearance check). True
frame-optimal route planning needs the velocity-augmented SURFACE GRAPH +
backward Dijkstra (the nclone value_function pipeline, made entity-aware via G1),
where the bins are graph nodes and d* is exact-on-lattice. Kept for reference;
do not rely on its output. Use physics_route_spike (weighted-A*) for fast valid
routes and route_video to render them.

The spike (`physics_route_spike`) runs *weighted*-A* (satisficing, w>1) and emits
a cost-to-*spawn* field. The production planner (`route-planner-decisions`) wants
the frame-optimal cost-to-*go* field `d*` — the reward potential. This prototype
builds it on one mine level:

  1. Enumerate the reachable state lattice from spawn with uniform-cost Dijkstra
     (optimal cost-to-come g*), recording the forward transition edges. Per-node
     O(1) restore via ``save_ninja_state``/``restore_ninja_state`` blobs.
  2. Reverse the edges; multi-source backward Dijkstra from the WON terminals →
     ``d*(state)`` = frame-optimal cost-to-go.
  3. ``d*(spawn)`` is the frame-optimal route length. Compare to the spike's
     weighted-A* (satisficing) route, and cross-check against optimal A* (w=1.0).
  4. Render the d* field (the entity-aware reward potential) + the optimal route.

PRECONDITION: this O(1)-restore enumeration assumes entity state never changes
during search (e.g., all-deadly mines — touching one is death, which is pruned;
no safe mines to toggle, no moving entities). True for ``minefield``. For
toggleable/moving entities, full-state snapshots (replay) are required.

Usage::

    python -m nclone.value_function.tools.lattice_dstar_prototype \
        --map nclone/test_maps/minefield --output docs/stage0/dstar_minefield.png
"""

from __future__ import annotations

import argparse
import heapq
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nclone_native as Nnative
from .physics_route_spike import (
    PhysicsRouteSpike, ACT, _switch_done, _load_map_data, _goal_positions,
    _MINE_STATE_DEADLY, _MINE_KILL_PX,
)

Key = Tuple


class LatticeDStar:
    def __init__(self, map_path: str, macro: int = 6, cell_px: float = 12.0,
                 vel_bin: float = 0.25, max_g_frames: int = 420,
                 state_cap: int = 250000):
        self.map_path = map_path
        self.map_data = _load_map_data(map_path)
        self.sim = Nnative.Simulator.load(self.map_data)
        self.macro = macro
        self.cell_px = cell_px
        self.vel_bin = vel_bin
        self.max_g = max_g_frames
        self.state_cap = state_cap
        self.spawn = self.sim.get_ninja_pos()
        self.switch, self.exit, mines = _goal_positions(self.sim)
        self.deadly0 = [(x, y) for (x, y, s) in mines if s == _MINE_STATE_DEADLY]
        # verify precondition: no safe (toggleable) mines
        self.safe_mines = [(x, y) for (x, y, s) in mines if s != _MINE_STATE_DEADLY]

    def _key(self, x, y, vx, vy, sw) -> Key:
        return (round(x / self.cell_px), round(y / self.cell_px),
                round(vx / self.vel_bin), round(vy / self.vel_bin), bool(sw))

    def _snapshot_state(self) -> dict:
        x, y = self.sim.get_ninja_pos()
        vx, vy = self.sim.get_ninja_velocity()
        return {"x": x, "y": y, "vx": vx, "vy": vy,
                "won": self.sim.has_won(), "sw": _switch_done(self.sim),
                "blob": self.sim.save_ninja_state()}

    def _replay_to(self, path: List[int]) -> None:
        """fast_reset + replay the action path (pure, no snapshot restore)."""
        self.sim.fast_reset()
        for a in path:
            h, j = ACT[a]
            for _ in range(self.macro):
                self.sim.tick(h, j)
                if self.sim.has_died() or self.sim.has_won():
                    return

    def enumerate_lattice(self) -> dict:
        """Forward uniform-cost Dijkstra over the state lattice, expanding via
        **pure action-replay** (no save/restore_ninja_state). Slower than snapshot
        teleport, but every node's stored path replays bit-exactly to that node, so
        the extracted route is guaranteed executable. Returns g*, edges, won paths."""
        t0 = time.time()
        self.sim.fast_reset()
        root_x, root_y = self.sim.get_ninja_pos()
        rvx, rvy = self.sim.get_ninja_velocity()
        spawn_key = self._key(root_x, root_y, rvx, rvy, False)
        g: Dict[Key, float] = {spawn_key: 0.0}
        pos: Dict[Key, Tuple[float, float]] = {spawn_key: (root_x, root_y)}
        fwd: List[Tuple[Key, Key, float]] = []
        won: set = set()
        won_paths: Dict[Key, List[int]] = {}
        settled: set = set()
        pq: List[Tuple[float, int, Key, list]] = [(0.0, 0, spawn_key, [])]
        ctr = 1
        while pq and len(settled) < self.state_cap:
            gc, _, key, path = heapq.heappop(pq)
            if key in settled:
                continue
            settled.add(key)
            if gc + self.macro > self.max_g:
                continue
            for a in ACT:
                self._replay_to(path)            # exact: back to this node
                if self.sim.has_won() or self.sim.has_died():
                    continue                     # node already terminal
                h, j = ACT[a]
                dead = won_here = False
                for _ in range(self.macro):
                    self.sim.tick(h, j)
                    if self.sim.has_died():
                        dead = True; break
                    if self.sim.has_won():
                        won_here = True; break
                if dead:
                    continue  # death-pruned: edge never enters a deadly-mine kill zone
                x, y = self.sim.get_ninja_pos()
                vx, vy = self.sim.get_ninja_velocity()
                ck = self._key(x, y, vx, vy, won_here or _switch_done(self.sim))
                ng = gc + self.macro
                fwd.append((key, ck, float(self.macro)))
                if won_here and ck not in won_paths:
                    won.add(ck); won_paths[ck] = path + [a]
                if ng < g.get(ck, float("inf")):
                    g[ck] = ng
                    pos[ck] = (x, y)
                    heapq.heappush(pq, (ng, ctr, ck, path + [a])); ctr += 1
        return {"g": g, "pos": pos, "fwd": fwd, "won": won, "won_paths": won_paths,
                "spawn_key": spawn_key, "settled": len(settled),
                "time": time.time() - t0}

    def reconstruct_actions(self, pred: Dict[Key, Tuple[Key, int]],
                            spawn_key: Key, target_key: Key) -> List[int]:
        """Walk predecessor links target->spawn, return the forward action list."""
        actions: List[int] = []
        cur = target_key
        guard = 0
        while cur != spawn_key and cur in pred:
            parent, a = pred[cur]
            actions.append(a)
            cur = parent
            guard += 1
            if guard > 100000:
                break
        actions.reverse()
        return actions

    def replay_actions(self, actions: List[int]) -> dict:
        """Replay the action list from spawn, tracing the REAL physics trajectory
        and the closest approach to any deadly mine (mine-clearance validation)."""
        self.sim.fast_reset()
        traj = [self.sim.get_ninja_pos()]
        min_clear = float("inf")
        won = died = False
        switch_frame = None
        frame = 0
        for a in actions:
            h, j = ACT[a]
            for _ in range(self.macro):
                self.sim.tick(h, j)
                frame += 1
                x, y = self.sim.get_ninja_pos()
                traj.append((x, y))
                for (mx, my) in self.deadly0:
                    c = ((x - mx) ** 2 + (y - my) ** 2) ** 0.5
                    if c < min_clear:
                        min_clear = c
                if switch_frame is None and _switch_done(self.sim):
                    switch_frame = frame
                if self.sim.has_won():
                    won = True
                if self.sim.has_died():
                    died = True
                if won or died:
                    break
            if won or died:
                break
        return {"traj": traj, "won": won, "died": died, "frames": frame,
                "switch_frame": switch_frame, "min_mine_clearance": min_clear}

    @staticmethod
    def backward_dstar(fwd: List[Tuple[Key, Key, float]], won: set) -> Dict[Key, float]:
        """Multi-source backward Dijkstra over reversed edges from WON terminals."""
        radj: Dict[Key, List[Tuple[Key, float]]] = defaultdict(list)
        for (u, v, c) in fwd:
            radj[v].append((u, c))
        dstar: Dict[Key, float] = {w: 0.0 for w in won}
        pq = [(0.0, w) for w in won]
        heapq.heapify(pq)
        while pq:
            d, v = heapq.heappop(pq)
            if d > dstar.get(v, float("inf")):
                continue
            for (u, c) in radj.get(v, ()):
                nd = d + c
                if nd < dstar.get(u, float("inf")):
                    dstar[u] = nd
                    heapq.heappush(pq, (nd, u))
        return dstar

    def optimal_route_positions(self, fwd, dstar, spawn_key, pos) -> List[Tuple[float, float]]:
        """Greedy descent of d* from spawn → list of cell positions."""
        out_adj: Dict[Key, List[Tuple[Key, float]]] = defaultdict(list)
        for (u, v, c) in fwd:
            out_adj[u].append((v, c))
        route = [pos[spawn_key]]
        cur = spawn_key
        seen = {cur}
        while dstar.get(cur, float("inf")) > 0 and cur in out_adj:
            best = min(out_adj[cur], key=lambda e: e[1] + dstar.get(e[0], float("inf")))
            nxt = best[0]
            if nxt in seen or dstar.get(nxt, float("inf")) == float("inf"):
                break
            route.append(pos.get(nxt, route[-1]))
            seen.add(nxt); cur = nxt
        return route


def run(map_path: str, output: Optional[str], **kw) -> dict:
    name = os.path.basename(map_path)
    L = LatticeDStar(map_path, **kw)
    if L.safe_mines:
        print(f"WARNING: {name} has {len(L.safe_mines)} toggleable (safe) mines — "
              f"O(1)-restore precondition violated; results may be unsound.")
    print(f"[{name}] enumerating lattice (vel_bin={L.vel_bin}, macro={L.macro})...")
    enum = L.enumerate_lattice()
    dstar = L.backward_dstar(enum["fwd"], enum["won"])
    spawn_key = enum["spawn_key"]
    opt_frames = dstar.get(spawn_key, float("inf"))
    print(f"  settled={enum['settled']} states, edges={len(enum['fwd'])}, "
          f"won_terminals={len(enum['won'])}, time={enum['time']:.1f}s")

    # Reconstruct the optimal route via predecessor chain to the cheapest WON
    # terminal, then REPLAY real physics (every edge was death-pruned ⇒ mine-free).
    opt = {"traj": [], "won": False, "min_mine_clearance": float("inf")}
    if enum["won"]:
        best_won = min(enum["won"], key=lambda k: enum["g"].get(k, float("inf")))
        actions = enum["won_paths"][best_won]  # explicit replayable path (sound)
        opt = L.replay_actions(actions)
        print(f"  FRAME-OPTIMAL d*(spawn)={opt_frames}f (macro-quantized) | "
              f"replayed route: won={opt['won']} frames={opt['frames']} "
              f"min_mine_clearance={opt['min_mine_clearance']:.1f}px "
              f"(kill≈{_MINE_KILL_PX:.0f}px ⇒ {'VALID' if opt['min_mine_clearance']>_MINE_KILL_PX else 'INVALID!'})")
    else:
        print("  no WON terminal in lattice (raise caps?)")

    # Satisficing spike route (w=1.25) on the SAME lattice resolution (apples-to-apples).
    sp = PhysicsRouteSpike(map_path, macro=L.macro, vel_bin=L.vel_bin, h_weight=1.25)
    sat = sp.plan()
    sat_frames = sat["frames"] if sat["solved"] else None
    if sat_frames and opt["won"]:
        print(f"  spike weighted-A* (w=1.25) = {sat_frames} frames  "
              f"→ {100*(sat_frames-opt['frames'])/opt['frames']:+.0f}% vs frame-optimal {opt['frames']}f")

    _render(L, enum, dstar, opt, sat, opt_frames, sat_frames, output)
    return {"opt_frames": opt["frames"] if opt["won"] else None,
            "sat_frames": sat_frames, "settled": enum["settled"],
            "min_mine_clearance": opt["min_mine_clearance"], "time": enum["time"]}


def _render(L, enum, dstar, opt, sat, opt_frames, sat_frames, output):
    from ..value_landscape import _load_tile_dict, _save_or_show
    from ..level_renderer import render_tiles_to_axis
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 1056); ax.set_ylim(600, 0); ax.set_aspect("equal")
    td = _load_tile_dict(L.map_path)
    if td is not None:
        render_tiles_to_axis(ax, td, alpha=0.4)

    # d* cost-to-go field (only finite, phase-0 cells for clarity)
    xs, ys, ds = [], [], []
    for k, d in dstar.items():
        if np.isfinite(d) and k in enum["pos"]:
            xs.append(enum["pos"][k][0]); ys.append(enum["pos"][k][1]); ds.append(d)
    if ds:
        norm = mcolors.Normalize(vmin=min(ds), vmax=max(ds))
        sc = ax.scatter(xs, ys, c=ds, cmap="viridis_r", norm=norm, s=10, zorder=2)
        plt.colorbar(sc, ax=ax, label="d* cost-to-go (frames)", shrink=0.8)

    for (mx, my) in L.deadly0:
        ax.add_patch(plt.Circle((mx, my), _MINE_KILL_PX, color="red", alpha=0.15, zorder=3))
        ax.scatter([mx], [my], marker="X", c="red", s=45, zorder=4)

    if opt["won"] and opt["traj"]:
        r = np.array(opt["traj"])
        valid = opt["min_mine_clearance"] > _MINE_KILL_PX
        ax.plot(r[:, 0], r[:, 1], "-", color="magenta", lw=2.5, zorder=6,
                label=f"frame-optimal route (replayed, "
                      f"{'mine-free' if valid else 'CLIPS MINE'})")
    if sat["solved"] and sat["trajectory"]:
        tr = np.array(sat["trajectory"])
        ax.plot(tr[:, 0], tr[:, 1], "--", color="orange", lw=1.8, zorder=5,
                label="spike weighted-A* (satisficing, replayed)")

    ax.scatter([L.spawn[0]], [L.spawn[1]], marker="o", c="black", s=70, zorder=7, label="spawn")
    ax.scatter([L.switch[0]], [L.switch[1]], marker="*", c="gold", s=220, edgecolors="k", zorder=7)
    ax.scatter([L.exit[0]], [L.exit[1]], marker="s", c="cyan", s=110, edgecolors="k", zorder=7)
    cmp = (f"optimal {opt['frames']}f vs satisficing {sat_frames}f "
           f"({100*(sat_frames-opt['frames'])/opt['frames']:+.0f}%)"
           if (sat_frames and opt["won"]) else "")
    ax.set_title(f"Backward-Dijkstra d* (frame-optimal) — {os.path.basename(L.map_path)}\n"
                 f"{enum['settled']} lattice states, {len(L.deadly0)} deadly mines | {cmp}",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)")
    plt.tight_layout()
    _save_or_show(fig, output)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--macro", type=int, default=6)
    p.add_argument("--vel-bin", type=float, default=0.5)
    p.add_argument("--max-g-frames", type=int, default=540)
    p.add_argument("--state-cap", type=int, default=150000)
    args = p.parse_args()
    run(args.map, args.output, macro=args.macro, vel_bin=args.vel_bin,
        max_g_frames=args.max_g_frames, state_cap=args.state_cap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
