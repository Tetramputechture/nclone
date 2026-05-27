#!/usr/bin/env python3
"""G1 spike: physics-accurate route planner via simulation lookahead.

Proves the core thesis of the route-planner work (`route-planner-decisions`):
*because the simulator is deterministic and ~1µs/frame, we can find a real,
entity-aware route by searching the actual physics* — instead of the
gravity-blind geometric A* the live reward currently uses.

Scope (deliberately): toggle mines + exit switch + exit door only (pure
movement learnability). No thwumps/drones/bounce — see `route-planner-decisions`
for why the native engine only "thinks" toggle mines today.

Method: weighted-A* over macro-action sequences on the **native** simulator.
- Each node stores its macro-action sequence from spawn. State is restored by
  `fast_reset()` + replay (the codebase's blessed deterministic-replay pattern;
  `state_checkpoint.py`). Full `tick()` runs mine toggling + switch/door logic,
  so **deadly-mine deaths prune branches automatically — mines shape the route.**
- Heuristic: straight-line distance to the current objective / MAX_HOR_SPEED
  (admissible lower bound on frames); two-phase (switch -> exit).
- Output: the route (action tape), the physical trajectory, the explored
  cost-to-spawn field, and a rendered PNG overlaying the route on mines+tiles.

Usage::

    python -m nclone.value_function.tools.physics_route_spike \
        --map nclone/test_maps/minefield --output docs/stage0/spike_minefield.png
"""

from __future__ import annotations

import argparse
import heapq
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nclone_native as Nnative
from ...nplay_headless import NPlayHeadless
from ...constants.physics_constants import MAX_HOR_SPEED, NINJA_RADIUS

# action id -> (hor_input, jump_input). Matches the 6-action env space.
ACT = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, 1), 4: (-1, 1), 5: (1, 1)}
_ENTITY_SWITCH = 4
_ENTITY_DOOR = 3
_MINE_TYPES = (1, 21)
_MINE_STATE_DEADLY = 0  # native ToggleMine: state 0 = toggled/deadly, 1 = safe
# Deadly toggle-mine kill radius + ninja radius (see TOGGLE_MINE_RADII / NINJA_RADIUS).
_MINE_KILL_PX = 4.0 + NINJA_RADIUS


def _load_map_data(map_path: str) -> List[int]:
    nh = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    nh.load_map(map_path)
    return list(nh.sim.map_data)


def _goal_positions(sim) -> Tuple[Tuple[float, float], Tuple[float, float], List[Tuple[float, float, int]]]:
    st = sim.get_state()
    switch = exit_door = None
    mines = []
    for e in st.entities:
        if e.entity_type == _ENTITY_SWITCH:
            switch = (float(e.x), float(e.y))
        elif e.entity_type == _ENTITY_DOOR:
            exit_door = (float(e.x), float(e.y))
        elif e.entity_type in _MINE_TYPES:
            mines.append((float(e.x), float(e.y), int(getattr(e, "state", 0))))
    return switch, exit_door, mines


def _switch_done(sim) -> bool:
    """Switch hit ⇒ the type-4 switch entity is no longer active."""
    for e in sim.get_state().entities:
        if e.entity_type == _ENTITY_SWITCH:
            return not bool(getattr(e, "active", True))
    return False


class PhysicsRouteSpike:
    def __init__(self, map_path: str, macro: int = 6, cell_px: float = 12.0,
                 vel_bin: float = 0.25, max_expansions: int = 40000,
                 max_frames: int = 1500, h_weight: float = 1.25):
        self.map_path = map_path
        self.map_data = _load_map_data(map_path)
        self.sim = Nnative.Simulator.load(self.map_data)
        self.macro = macro
        self.cell_px = cell_px
        self.vel_bin = vel_bin
        self.max_expansions = max_expansions
        self.max_actions = max_frames // macro
        self.w = h_weight
        self.spawn = self.sim.get_ninja_pos()
        self.switch, self.exit, self.mines0 = _goal_positions(self.sim)
        self.deadly0 = [(x, y) for (x, y, s) in self.mines0 if s == _MINE_STATE_DEADLY]

    # --- sim helpers ---
    def _replay(self, actions: List[int]) -> None:
        self.sim.fast_reset()
        for a in actions:
            h, j = ACT[a]
            for _ in range(self.macro):
                self.sim.tick(h, j)
                if self.sim.has_died() or self.sim.has_won():
                    return

    def _state_after(self, actions: List[int]) -> dict:
        self._replay(actions)
        x, y = self.sim.get_ninja_pos()
        vx, vy = self.sim.get_ninja_velocity()
        return {
            "x": x, "y": y, "vx": vx, "vy": vy,
            "dead": self.sim.has_died(), "won": self.sim.has_won(),
            "switch": _switch_done(self.sim),
        }

    def _h(self, x: float, y: float, switch_done: bool) -> float:
        gx, gy = (self.exit if switch_done else self.switch)
        d = ((x - gx) ** 2 + (y - gy) ** 2) ** 0.5
        if not switch_done:
            d += ((self.switch[0] - self.exit[0]) ** 2 + (self.switch[1] - self.exit[1]) ** 2) ** 0.5
        return d / MAX_HOR_SPEED  # frames lower bound

    def _key(self, s: dict) -> Tuple:
        return (round(s["x"] / self.cell_px), round(s["y"] / self.cell_px),
                round(s["vx"] / self.vel_bin), round(s["vy"] / self.vel_bin),
                bool(s["switch"]))

    def plan(self) -> dict:
        t0 = time.time()
        root = self._state_after([])
        start_node = (self.w * self._h(root["x"], root["y"], root["switch"]),
                      0.0, 0, [], root)
        frontier = [start_node]
        best_g: Dict[Tuple, float] = {self._key(root): 0.0}
        counter = 1
        expansions = 0
        explored: List[Tuple[float, float, float]] = []  # (x, y, g_frames)
        deaths = 0

        while frontier and expansions < self.max_expansions:
            f, g, _, actions, s = heapq.heappop(frontier)
            if g > best_g.get(self._key(s), float("inf")):
                continue
            expansions += 1
            explored.append((s["x"], s["y"], g))
            if s["won"]:
                return self._finish(actions, g, expansions, deaths, explored, t0, True)
            if len(actions) >= self.max_actions:
                continue
            for a in ACT:
                child_actions = actions + [a]
                cs = self._state_after(child_actions)
                if cs["dead"]:
                    deaths += 1
                    continue
                cg = g + self.macro
                k = self._key(cs)
                if cg < best_g.get(k, float("inf")):
                    best_g[k] = cg
                    if cs["won"]:
                        return self._finish(child_actions, cg, expansions, deaths, explored, t0, True)
                    cf = cg + self.w * self._h(cs["x"], cs["y"], cs["switch"])
                    heapq.heappush(frontier, (cf, cg, counter, child_actions, cs))
                    counter += 1
        return self._finish([], float("inf"), expansions, deaths, explored, t0, False)

    def _finish(self, actions, g, expansions, deaths, explored, t0, solved) -> dict:
        traj, switch_frame = self._trace(actions) if solved else ([], None)
        return {
            "solved": solved, "frames": g if solved else None,
            "actions": actions, "trajectory": traj, "switch_frame": switch_frame,
            "expansions": expansions, "death_branches": deaths,
            "explored": explored, "wall_time_s": time.time() - t0,
        }

    def _trace(self, actions: List[int]) -> Tuple[List[Tuple[float, float]], Optional[int]]:
        """Replay frame-by-frame to capture the full physical trajectory."""
        self.sim.fast_reset()
        traj = [self.sim.get_ninja_pos()]
        switch_frame = None
        frame = 0
        for a in actions:
            h, j = ACT[a]
            for _ in range(self.macro):
                self.sim.tick(h, j)
                frame += 1
                traj.append(self.sim.get_ninja_pos())
                if switch_frame is None and _switch_done(self.sim):
                    switch_frame = frame
                if self.sim.has_died() or self.sim.has_won():
                    return traj, switch_frame
        return traj, switch_frame


# ---------------------------------------------------------------------------
def render(map_path: str, res: dict, output: Optional[str]) -> None:
    from ..value_landscape import _load_tile_dict, _save_or_show
    from ..level_renderer import render_tiles_to_axis

    planner_mines = res["_deadly"]
    spawn, switch, exitp = res["_spawn"], res["_switch"], res["_exit"]

    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 1056); ax.set_ylim(600, 0); ax.set_aspect("equal")
    tile_dic = _load_tile_dict(map_path)
    if tile_dic is not None:
        render_tiles_to_axis(ax, tile_dic, alpha=0.4)

    # explored cost-to-spawn field (faint), shows what the search covered
    if res["explored"]:
        ex = np.array(res["explored"])
        ax.scatter(ex[:, 0], ex[:, 1], c=ex[:, 2], cmap="Greys", s=4, alpha=0.35,
                   zorder=2, label="explored (cost-to-spawn, frames)")

    # deadly mines + kill zones
    for (mx, my) in planner_mines:
        ax.add_patch(plt.Circle((mx, my), _MINE_KILL_PX, color="red", alpha=0.18, zorder=3))
        ax.scatter([mx], [my], marker="X", c="red", s=60, zorder=4)
    if planner_mines:
        ax.scatter([], [], marker="X", c="red", s=60, label="deadly mine (+kill zone)")

    # physics route trajectory, coloured by phase (pre/post switch)
    if res["solved"] and res["trajectory"]:
        tr = np.array(res["trajectory"])
        sf = res["switch_frame"] or len(tr)
        ax.plot(tr[:sf + 1, 0], tr[:sf + 1, 1], "-", color="dodgerblue", lw=2.5,
                zorder=6, label="route → switch")
        ax.plot(tr[sf:, 0], tr[sf:, 1], "-", color="limegreen", lw=2.5,
                zorder=6, label="route → exit")

    ax.scatter([spawn[0]], [spawn[1]], marker="o", c="black", s=80, zorder=7, label="spawn")
    ax.scatter([switch[0]], [switch[1]], marker="*", c="gold", s=240,
               edgecolors="k", zorder=7, label="switch")
    ax.scatter([exitp[0]], [exitp[1]], marker="s", c="cyan", s=120,
               edgecolors="k", zorder=7, label="exit")

    level = os.path.basename(map_path.rstrip("/"))
    status = (f"SOLVED in {res['frames']} frames ({res['frames']/60:.1f}s)"
              if res["solved"] else "NO ROUTE FOUND (raise caps?)")
    ax.set_title(
        f"Physics route spike — {level}   |   {status}\n"
        f"{len(planner_mines)} deadly mines | {res['expansions']} expansions, "
        f"{res['death_branches']} death-branches pruned | {res['wall_time_s']:.1f}s",
        fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)")
    plt.tight_layout()
    _save_or_show(fig, output)


def plan_and_render(map_path: str, output: Optional[str], **kw) -> dict:
    sp = PhysicsRouteSpike(map_path, **kw)
    print(f"[{os.path.basename(map_path)}] spawn={sp.spawn} switch={sp.switch} "
          f"exit={sp.exit} deadly_mines={len(sp.deadly0)}")
    res = sp.plan()
    res.update(_deadly=sp.deadly0, _spawn=sp.spawn, _switch=sp.switch, _exit=sp.exit)
    print(f"  solved={res['solved']} frames={res['frames']} "
          f"expansions={res['expansions']} deaths_pruned={res['death_branches']} "
          f"time={res['wall_time_s']:.1f}s")
    render(map_path, res, output)
    return res


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", required=True, help="test_maps level path")
    p.add_argument("--output", default=None)
    p.add_argument("--macro", type=int, default=6, help="frames per macro-action")
    p.add_argument("--max-expansions", type=int, default=40000)
    p.add_argument("--h-weight", type=float, default=1.25, help="weighted-A* heuristic weight")
    args = p.parse_args()
    plan_and_render(args.map, args.output, macro=args.macro,
                    max_expansions=args.max_expansions, h_weight=args.h_weight)
    return 0


if __name__ == "__main__":
    sys.exit(main())
