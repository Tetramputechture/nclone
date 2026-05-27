#!/usr/bin/env python3
"""Stage-0 diagnostic: geometric-A* potential vs surface-graph d* potential.

Renders a side-by-side comparison that makes the core thesis of
``.cursor/plans/plan_distill_search_merged_*.plan.md`` *visible*:

  * **Left panel** — the potential the *live* reward actually uses:
    6px geometric-A* distance to the goal (``path_distance_calculator``).
    8-connected over free space, Euclidean edge cost, gravity-blind.
  * **Right panel** — the physics-aware alternative: surface-graph ``d*``
    cost-to-go (frames) from the precomputed ``*_surface_pbrs.npz``.
  * **Overlays (both panels)** — the geometric-A* spawn->switch path, coloured
    by *floor depth below each node* (green = on/near a floor, red = flying
    over a void), and a **naive geometric-gradient rollout** (greedy toward the
    goal, no jump) that physically falls where the geometric route claims a
    straight line.
  * **Bottom panel** — floor-depth-below along the geometric route vs path
    arc-length, with the 24px (one-tile) "genuinely airborne" threshold and the
    fly-over stretches shaded. This is the quantitative proof that the geometric
    route the reward rewards is not physically followable.

Coordinate note: reachability-graph nodes live in *tile-data* space
(world - ``NODE_WORLD_COORD_OFFSET`` = world - 24px); npz surface points and
tiles live in *world* space (0..1056 / 0..600). We add +24 to node coords for
rendering so both panels share one frame.

Usage::

    python -m nclone.value_function.tools.potential_compare \
        --map nclone/test_maps/long-jump-reachable \
        --npz value_functions/long-jump-reachable_surface_pbrs.npz \
        --output docs/stage0/long-jump-reachable.png

    # or via the umbrella CLI:
    python -m nclone.value_function.visualize_value_function \
        --mode potential_compare \
        --map-path nclone/test_maps/long-jump-reachable \
        --data value_functions/long-jump-reachable_surface_pbrs.npz \
        --output docs/stage0/long-jump-reachable.png
"""

from __future__ import annotations

import argparse
import heapq
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# Headless-safe backend (WSL / CI). Must precede pyplot import.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ...constants.physics_constants import EXIT_SWITCH_RADIUS, NINJA_RADIUS
from ...graph.reachability.path_distance_calculator import CachedPathDistanceCalculator
from ...graph.reachability.path_distance_cache import NODE_WORLD_COORD_OFFSET as OFF

Node = Tuple[int, int]

# A 6px node whose nearest floor is more than one tile (24px) below is
# "genuinely airborne": the ninja is falling here, not walking.
_AIRBORNE_THRESHOLD_PX = 24.0
_FLOOR_PROBE_MAX_PX = 300.0
_STEP = 6


# ---------------------------------------------------------------------------
# Geometry / physics probes over the reachability graph
# ---------------------------------------------------------------------------
def _nearest_node(nodes: List[Node], pos: Tuple[float, float]) -> Node:
    px, py = pos
    return min(nodes, key=lambda n: (n[0] - px) ** 2 + (n[1] - py) ** 2)


def geometric_distance_field(
    adjacency: Dict[Node, List[Tuple[Node, float]]], goal_node: Node
) -> Dict[Node, float]:
    """Single-source Dijkstra from ``goal_node`` with Euclidean edge costs.

    This reproduces exactly the metric ``_astar_distance`` minimises, so the
    resulting field is the live reward's potential at every reachable node.
    """
    dist: Dict[Node, float] = {goal_node: 0.0}
    pq: List[Tuple[float, Node]] = [(0.0, goal_node)]
    while pq:
        d, node = heapq.heappop(pq)
        if d > dist.get(node, float("inf")):
            continue
        for nbr, _ in adjacency.get(node, []):
            dx = nbr[0] - node[0]
            dy = nbr[1] - node[1]
            nd = d + (dx * dx + dy * dy) ** 0.5
            if nd < dist.get(nbr, float("inf")):
                dist[nbr] = nd
                heapq.heappush(pq, (nd, nbr))
    return dist


def _grounded(node_physics: Dict[Node, dict], node: Node) -> Optional[bool]:
    v = node_physics.get(node)
    return None if v is None else bool(v.get("grounded"))


def floor_depth_below(node_physics: Dict[Node, dict], node: Node) -> float:
    """Pixels down to the first floor; ``inf`` if no floor within probe range."""
    if _grounded(node_physics, node):
        return 0.0
    x, y = node
    for k in range(1, int(_FLOOR_PROBE_MAX_PX // _STEP) + 1):
        below = (x, y + k * _STEP)
        if _grounded(node_physics, below):
            return float(k * _STEP)
        if below not in node_physics:  # solid tile below -> floor is right here
            return float(k * _STEP)
    return float("inf")


def naive_gradient_rollout(
    env, goal_world: Tuple[float, float], max_steps: int = 150
) -> Tuple[List[Tuple[float, float]], bool]:
    """Greedy 'descend the geometric potential' controller: walk toward the
    goal, never jump. Returns (trajectory_world, died). It physically falls
    wherever the geometric route flies over a void."""
    traj: List[Tuple[float, float]] = []
    died = False
    nh = env.nplay_headless
    for _ in range(max_steps):
        x, y = nh.ninja_position()
        traj.append((float(x), float(y)))
        action = 2 if goal_world[0] >= x else 1  # RIGHT / LEFT, no jump
        _, _, term, trunc, info = env.step(action)
        if term or trunc:
            x, y = nh.ninja_position()
            traj.append((float(x), float(y)))
            died = bool(info.get("player_dead", term and not info.get("player_won", False)))
            break
    return traj, died


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _per_point_min_dstar(d_star: np.ndarray, n_points: int, max_bins: int) -> np.ndarray:
    """Best-case (min over velocity bins) cost-to-go per surface point."""
    out = np.full(n_points, np.inf, dtype=np.float64)
    for i in range(n_points):
        block = d_star[i * max_bins : (i + 1) * max_bins]
        if block.size:
            finite = block[np.isfinite(block)]
            if finite.size:
                out[i] = float(finite.min())
    return out


def _draw_path(ax, path_world, depths, label):
    """Scatter the geometric path coloured by floor-depth-below (green->red)."""
    xs = [p[0] for p in path_world]
    ys = [p[1] for p in path_world]
    capped = [min(d, 120.0) for d in depths]  # cap inf/large for colour scale
    ax.plot(xs, ys, "-", color="black", lw=0.8, alpha=0.5, zorder=5)
    sc = ax.scatter(xs, ys, c=capped, cmap="RdYlGn_r", vmin=0, vmax=120,
                    s=18, zorder=6, edgecolors="k", linewidths=0.3, label=label)
    return sc


def _draw_rollout(ax, traj, died):
    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]
    ax.plot(xs, ys, "-", color="deepskyblue", lw=2.0, zorder=7,
            label="naive gradient rollout (no jump)")
    ax.scatter([xs[0]], [ys[0]], marker="o", c="deepskyblue", s=60, zorder=8)
    if died:
        ax.scatter([xs[-1]], [ys[-1]], marker="X", c="red", s=140, zorder=9,
                   label="rollout DEATH (fell into void)")


def render_comparison(map_path: str, npz_path: str, output: Optional[str]) -> int:
    from ..value_landscape import SurfaceGraphVisualizer
    from nclone.gym_environment.config import EnvironmentConfig
    from nclone.gym_environment import NppEnvironment

    if not os.path.exists(npz_path):
        print(f"ERROR: npz not found: {npz_path}")
        return 1

    # --- physics-aware d* side (reuses existing loader + tile/entity render) ---
    viz = SurfaceGraphVisualizer(npz_path, map_path=map_path)
    n_pts = len(viz.graph.surface_points)
    max_bins = viz.graph.max_vel_bins
    dstar_pts = _per_point_min_dstar(viz.d_switch, n_pts, max_bins)

    # --- env: adjacency, node physics, goal positions ---
    cfg = EnvironmentConfig(custom_map_path=map_path)
    env = NppEnvironment(cfg)
    env.reset()
    nh = env.nplay_headless
    spawn = nh.ninja_position()
    switch = nh.exit_switch_position()
    cgd = env.current_graph_data
    adjacency = cgd["adjacency"]
    node_physics = cgd.get("node_physics", {})
    nodes = list(adjacency.keys())

    # geometric field (node space) -> world for plotting (+OFF)
    goal_node = _nearest_node(nodes, (switch[0] - OFF, switch[1] - OFF))
    field = geometric_distance_field(adjacency, goal_node)
    fxs = np.array([n[0] + OFF for n in field])
    fys = np.array([n[1] + OFF for n in field])
    fvals = np.array([field[n] for n in field])

    # geometric path spawn->switch (node space), floor depth per node
    pc = CachedPathDistanceCalculator(max_cache_size=200)
    base = cgd.get("base_adjacency", adjacency)
    _, path = pc.get_path_cost(
        (int(spawn[0]), int(spawn[1])), (int(switch[0]), int(switch[1])),
        adjacency, base, cache_key="switch",
        entity_radius=EXIT_SWITCH_RADIUS, ninja_radius=NINJA_RADIUS,
        return_path=True,
    )
    depths = [floor_depth_below(node_physics, n) for n in path]
    path_world = [(n[0] + OFF, n[1] + OFF) for n in path]

    # naive gradient rollout (must run on a fresh env state)
    env.reset()
    traj, died = naive_gradient_rollout(env, switch)

    # ---- figure: two map panels (top) + floor-depth panel (bottom) ----
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1.0], hspace=0.22, wspace=0.12)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # Panel 1: geometric A* field (what the reward uses)
    viz._setup_axis(ax1, "LIVE REWARD potential: geometric A* distance (px) — gravity-blind")
    if np.isfinite(fvals).any():
        fin = fvals[np.isfinite(fvals)]
        norm = mcolors.Normalize(vmin=float(fin.min()), vmax=float(fin.max()))
        c = np.where(np.isfinite(fvals), fvals, fin.max())
        s1 = ax1.scatter(fxs, fys, c=c, cmap="viridis_r", norm=norm, s=6, zorder=3)
        plt.colorbar(s1, ax=ax1, label="geometric dist-to-switch (px)", shrink=0.8)
    _draw_path(ax1, path_world, depths, "geometric path (color=floor below)")
    _draw_rollout(ax1, traj, died)
    ax1.legend(loc="upper right", fontsize=7)

    # Panel 2: surface-graph d* field (physics cost-to-go)
    viz._setup_axis(ax2, "PHYSICS cost-to-go: surface-graph d* (frames)")
    fin2 = dstar_pts[np.isfinite(dstar_pts)]
    if fin2.size:
        norm2 = mcolors.Normalize(vmin=float(fin2.min()), vmax=float(fin2.max()))
        c2 = np.where(np.isfinite(dstar_pts), dstar_pts, fin2.max())
        s2 = ax2.scatter(viz.xs, viz.ys, c=c2, cmap="viridis_r", norm=norm2,
                         s=8, zorder=3)
        plt.colorbar(s2, ax=ax2, label="d* min over vel-bins (frames)", shrink=0.8)
    _draw_path(ax2, path_world, depths, "geometric path (color=floor below)")
    _draw_rollout(ax2, traj, died)
    ax2.legend(loc="upper right", fontsize=7)

    # Panel 3: floor depth along the geometric route
    arc = [0.0]
    for i in range(1, len(path_world)):
        dx = path_world[i][0] - path_world[i - 1][0]
        dy = path_world[i][1] - path_world[i - 1][1]
        arc.append(arc[-1] + (dx * dx + dy * dy) ** 0.5)
    plot_depths = [min(d, _FLOOR_PROBE_MAX_PX) for d in depths]
    ax3.plot(arc, plot_depths, "-o", color="darkorange", ms=3, label="floor depth below route")
    ax3.axhline(_AIRBORNE_THRESHOLD_PX, color="red", ls="--", lw=1,
                label=f"airborne threshold ({_AIRBORNE_THRESHOLD_PX:.0f}px = 1 tile)")
    ax3.fill_between(arc, plot_depths, _AIRBORNE_THRESHOLD_PX,
                     where=[d > _AIRBORNE_THRESHOLD_PX for d in plot_depths],
                     color="red", alpha=0.25, label="FLY-OVER: route is over a void")
    n_air = sum(1 for d in depths if (not np.isfinite(d)) or d > _AIRBORNE_THRESHOLD_PX)
    ax3.set_xlabel("arc length along geometric route (px)")
    ax3.set_ylabel("floor depth below (px)")
    ax3.set_title(
        f"Floor depth below the geometric route — "
        f"{n_air}/{len(path)} nodes genuinely airborne "
        f"({100 * n_air / max(1, len(path)):.0f}% of the route flies over no floor)"
    )
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(alpha=0.3)

    level = os.path.basename(map_path.rstrip("/"))
    fig.suptitle(
        f"Potential comparison — {level}   "
        f"(rollout outcome: {'DIED in void' if died else 'survived/stalled'})",
        fontsize=13, y=0.99,
    )

    from ..value_landscape import _save_or_show
    _save_or_show(fig, output)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", "--map-path", dest="map_path", required=True,
                   help="Path to the test_maps level directory/file")
    p.add_argument("--npz", "--data", dest="npz_path", required=True,
                   help="Path to {level}_surface_pbrs.npz")
    p.add_argument("--output", default=None, help="Save PNG here (else show)")
    args = p.parse_args()
    return render_comparison(args.map_path, args.npz_path, args.output)


if __name__ == "__main__":
    sys.exit(main())
