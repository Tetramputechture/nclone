#!/usr/bin/env python3
"""Visualize the optimal reward-shaping route on the basics level.

Traces the greedy path through the surface graph (phase 0: spawn→switch,
phase 1: switch→exit) and renders arcs on the level geometry alongside
a d* cost-to-go heatmap.

Usage (from nclone project root)::

    python scripts/visualize_optimal_route.py \
        --map-path test-single-level/basics \
        --pbrs value_functions/basics_surface_pbrs.npz \
        --output basics_optimal_route.png
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nclone.value_function.surface_extraction import SURFACE_FLOOR, SURFACE_WALL
from nclone.value_function.surface_graph import (
    node_id_to_parts,
    velocity_bin_centers,
    velocity_to_bin,
)
from nclone.value_function.level_renderer import render_tiles_to_axis, mark_entities
from nclone.value_function.physics_step_wrapper import PhysicsStepWrapper

_INF = 1e9
_MAX_PATH_STEPS = 400


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_pbrs(path: str):
    """Load surface-graph PBRS arrays from .npz."""
    from nclone.value_function.surface_extraction import SurfacePoint
    data = np.load(path, allow_pickle=True)
    n = len(data["surface_points_x"])
    surface_points = [
        SurfacePoint(
            x=float(data["surface_points_x"][i]),
            y=float(data["surface_points_y"][i]),
            normal_x=float(data["surface_normals_x"][i]),
            normal_y=float(data["surface_normals_y"][i]),
            surface_type=int(data["surface_types"][i]),
            segment_id=int(data["surface_segment_ids"][i]),
            is_edge=bool(data["surface_is_edge"][i]),
        )
        for i in range(n)
    ]
    edges_src = data["edges_src"].astype(np.int64)
    edges_dst = data["edges_dst"].astype(np.int64)
    edges_weight = data["edges_weight"].astype(np.float32)
    d_star_switch = data["d_star_switch"].astype(np.float64)
    d_star_exit = data["d_star_exit"].astype(np.float64)
    max_vel_bins = int(data["metadata"].item()["max_vel_bins"])
    return surface_points, edges_src, edges_dst, edges_weight, d_star_switch, d_star_exit, max_vel_bins


def _build_adj(edges_src, edges_dst, edges_weight) -> Dict[int, List[Tuple[int, float]]]:
    """Build adjacency list {src_node: [(dst_node, weight), ...]}."""
    adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for s, d, w in zip(edges_src, edges_dst, edges_weight):
        adj[int(s)].append((int(d), float(w)))
    return adj


# ---------------------------------------------------------------------------
# Spawn + goal node resolution
# ---------------------------------------------------------------------------

def _get_spawn_landing_pos(map_path: str) -> Tuple[float, float]:
    """Simulate a few frames to find where the ninja first lands after spawn."""
    from nclone.value_function.physics_step_wrapper import PhysicsStepWrapper
    pw = PhysicsStepWrapper(map_path=map_path)
    sim = pw._sim
    for _ in range(30):
        sim.tick_physics_only(0, 0)
        state = sim.get_ninja_state_code()
        if state in (0, 1, 2):  # Immobile, Running, GroundSlide
            pos = sim.get_ninja_pos()
            return (float(pos[0]), float(pos[1]))
    pos = sim.get_ninja_pos()
    return (float(pos[0]), float(pos[1]))


def _nearest_floor_node(
    target_x: float,
    target_y: float,
    target_vx: float,
    surface_points,
    max_vel_bins: int,
) -> int:
    """Return the graph node (floor surface point) nearest to the given position/velocity."""
    best_dist = float("inf")
    best_node = 0
    target_vel_bin = velocity_to_bin(target_vx, SURFACE_FLOOR)
    for i, sp in enumerate(surface_points):
        if sp.surface_type != SURFACE_FLOOR:
            continue
        dist = (sp.x - target_x) ** 2 + (sp.y - target_y) ** 2
        if dist < best_dist:
            best_dist = dist
            best_vel_bin = velocity_to_bin(target_vx, SURFACE_FLOOR)
            best_node = i * max_vel_bins + best_vel_bin
    return best_node


# ---------------------------------------------------------------------------
# Path tracing
# ---------------------------------------------------------------------------

def _trace_greedy_path(
    start_node: int,
    adj: Dict[int, List[Tuple[int, float]]],
    d_star: np.ndarray,
    max_steps: int = _MAX_PATH_STEPS,
) -> List[Tuple[int, int, float]]:
    """Trace the greedy optimal path following descending d* values.

    At each node, follow the outgoing edge whose destination has the
    lowest d* value (i.e. closest to goal).  Stop when d* < 1 frame
    (goal reached) or no improvement is possible.

    Returns:
        List of (src_node, dst_node, weight) for each hop.
    """
    path: List[Tuple[int, int, float]] = []
    visited = set()
    current = start_node

    for _ in range(max_steps):
        cur_d = d_star[current] if current < len(d_star) else _INF
        if cur_d < 1.0:
            break
        if current in visited:
            break
        visited.add(current)

        neighbors = adj.get(current, [])
        if not neighbors:
            break

        # Greedy: pick the neighbor with the minimum d* (closest to goal)
        best_dst, best_w = min(neighbors, key=lambda nw: d_star[nw[0]] if nw[0] < len(d_star) else _INF)
        best_d = d_star[best_dst] if best_dst < len(d_star) else _INF

        if best_d >= cur_d:
            break  # no descent possible

        path.append((current, best_dst, best_w))
        current = best_dst

    return path


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _path_xy(path_edges, surface_points, max_vel_bins):
    """Convert path edge list to (xs, ys) of visited surface-point positions."""
    xs, ys = [], []
    for src_node, dst_node, _ in path_edges:
        src_pt, _ = node_id_to_parts(src_node, max_vel_bins)
        dst_pt, _ = node_id_to_parts(dst_node, max_vel_bins)
        xs.append(surface_points[src_pt].x)
        ys.append(surface_points[src_pt].y)
    if path_edges:
        last_node = path_edges[-1][1]
        last_pt, _ = node_id_to_parts(last_node, max_vel_bins)
        xs.append(surface_points[last_pt].x)
        ys.append(surface_points[last_pt].y)
    return xs, ys


def _draw_path_arrows(ax, path_edges, surface_points, max_vel_bins, color, zorder=6):
    """Draw annotated arrows for each hop in the path."""
    for src_node, dst_node, weight in path_edges:
        src_pt, _ = node_id_to_parts(src_node, max_vel_bins)
        dst_pt, _ = node_id_to_parts(dst_node, max_vel_bins)
        sx, sy = surface_points[src_pt].x, surface_points[src_pt].y
        dx, dy = surface_points[dst_pt].x, surface_points[dst_pt].y
        ax.annotate(
            "",
            xy=(dx, dy), xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.8,
                mutation_scale=12,
                connectionstyle="arc3,rad=0.15",
            ),
            zorder=zorder,
        )


# ---------------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------------

def visualize(map_path: str, pbrs_path: str, output_path: str) -> None:
    print(f"Loading PBRS: {pbrs_path}")
    (surface_points, edges_src, edges_dst, edges_weight,
     d_star_switch, d_star_exit, max_vel_bins) = _load_pbrs(pbrs_path)

    pw = PhysicsStepWrapper(map_path=map_path)
    tile_dic = pw.tile_dic
    switch_pos = pw.get_goal_position("switch")
    exit_pos = pw.get_goal_position("exit")

    print(f"  Surface points: {len(surface_points):,}")
    print(f"  Edges: {len(edges_src):,}")
    print(f"  Switch: {switch_pos}  Exit: {exit_pos}")

    spawn_pos = _get_spawn_landing_pos(map_path)
    print(f"  Spawn landing: {spawn_pos}")

    adj = _build_adj(edges_src, edges_dst, edges_weight)

    spawn_node = _nearest_floor_node(spawn_pos[0], spawn_pos[1], 0.0, surface_points, max_vel_bins)
    switch_node = _nearest_floor_node(switch_pos[0], switch_pos[1], 0.0, surface_points, max_vel_bins)

    print(f"  Spawn node: {spawn_node}  d_switch={d_star_switch[spawn_node]:.0f}  d_exit={d_star_exit[spawn_node]:.0f}")
    print(f"  Switch node: {switch_node}  d_exit={d_star_exit[switch_node]:.0f}")

    path_to_switch = _trace_greedy_path(spawn_node, adj, d_star_switch)
    path_to_exit = _trace_greedy_path(switch_node, adj, d_star_exit)
    print(f"  Phase 0 hops (spawn→switch): {len(path_to_switch)}")
    print(f"  Phase 1 hops (switch→exit):  {len(path_to_exit)}")

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.patch.set_facecolor("#1a1a1a")

    titles = ["Phase 0 — Spawn → Switch (d* switch)", "Phase 1 — Switch → Exit (d* exit)"]
    d_stars = [d_star_switch, d_star_exit]
    paths = [path_to_switch, path_to_exit]
    path_colors = ["#FFD700", "#00FF88"]
    goal_marks = [
        dict(switch_pos=switch_pos, exit_pos=None),
        dict(switch_pos=None, exit_pos=exit_pos),
    ]

    sp_xs = np.array([p.x for p in surface_points])
    sp_ys = np.array([p.y for p in surface_points])
    sp_types = np.array([p.surface_type for p in surface_points])

    for ax, title, d_star, path_edges, path_col, goal_kw in zip(
        axes, titles, d_stars, paths, path_colors, goal_marks
    ):
        ax.set_facecolor("#1a1a1a")
        ax.set_xlim(0, 1056)
        ax.set_ylim(600, 0)
        ax.set_aspect("equal")
        ax.set_xlabel("X (px)", color="#cccccc", fontsize=9)
        ax.set_ylabel("Y (px)", color="#cccccc", fontsize=9)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(colors="#888888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        # Tile geometry
        render_tiles_to_axis(ax, tile_dic, tile_color="#3a3a3a", alpha=0.9)

        # Surface points colored by d* (viridis_r = low cost = bright)
        finite_mask = d_star < _INF / 2
        finite_d = d_star[finite_mask]

        # Per-node d* → per-point d* (take minimum across vel bins)
        pt_d = np.full(len(surface_points), np.inf)
        for nid in range(len(d_star)):
            pt_idx, _ = node_id_to_parts(nid, max_vel_bins)
            if pt_idx < len(surface_points) and d_star[nid] < pt_d[pt_idx]:
                pt_d[pt_idx] = d_star[nid]

        finite_pt = pt_d < _INF / 2
        if finite_pt.any():
            vmin, vmax = float(pt_d[finite_pt].min()), float(pt_d[finite_pt].max())
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            c_vals = np.where(finite_pt, pt_d, vmax)
            sc = ax.scatter(
                sp_xs, sp_ys,
                c=c_vals, cmap="plasma_r",
                norm=norm,
                s=np.where(sp_types == SURFACE_FLOOR, 6, 4),
                alpha=0.7, zorder=3,
            )
            cbar = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
            cbar.set_label("d* (frames)", color="#cccccc", fontsize=8)
            cbar.ax.yaxis.set_tick_params(color="#888888", labelsize=7)
            plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#888888")

        # Draw path edges as curved arrows
        _draw_path_arrows(ax, path_edges, surface_points, max_vel_bins, color=path_col)

        # Draw path waypoints
        px_list, py_list = _path_xy(path_edges, surface_points, max_vel_bins)
        if px_list:
            ax.scatter(px_list[1:-1], py_list[1:-1],
                       c=path_col, s=30, zorder=7, edgecolors="white", linewidths=0.5)
            # Spawn marker (first point in phase 0 path)
            if path_edges and ax is axes[0]:
                ax.scatter([px_list[0]], [py_list[0]],
                           marker="o", c="white", s=120, zorder=9,
                           edgecolors=path_col, linewidths=2.0, label="Spawn")

        # Entity markers
        mark_entities(ax, **goal_kw)

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="lower right", fontsize=8,
                      facecolor="#2a2a2a", edgecolor="#555555",
                      labelcolor="white")

        # Hop count annotation
        n_hops = len(path_edges)
        total_w = sum(w for _, _, w in path_edges)
        ax.text(0.02, 0.02,
                f"Hops: {n_hops}   Est. frames: {total_w:.0f}",
                transform=ax.transAxes, fontsize=8,
                color="#aaaaaa", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#2a2a2a",
                          edgecolor="#555555", alpha=0.8))

    plt.suptitle(
        "Surface-Graph PBRS — Optimal Reward-Shaping Route  (basics level)",
        color="white", fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved: {output_path}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize optimal reward-shaping route through the surface graph"
    )
    parser.add_argument("--map-path", default="test-single-level/basics",
                        help="Path to level .map file")
    parser.add_argument("--pbrs", default="value_functions/basics_surface_pbrs.npz",
                        help="Path to precomputed PBRS .npz")
    parser.add_argument("--output", default="basics_optimal_route.png",
                        help="Output image path")
    args = parser.parse_args()
    visualize(args.map_path, args.pbrs, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
