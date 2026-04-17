#!/usr/bin/env python3
"""Visualize the mine virtual-gap in a precomputed surface-graph PBRS file.

Validates three invariants that must hold after the virtual-gap refactor:

  1. **No surface points inside exclusion zone** — the segment-splitting step
     in ``surface_extraction.py`` must eliminate all sampled nodes within
     ``MINE_SURFACE_EXCLUSION_RADIUS`` (18 px) of every mine centre.

  2. **Gap boundaries are edge-flagged** — the split endpoints C and D should
     be within ``EDGE_PROXIMITY`` (12 px) of their segment endpoints and
     therefore flagged ``is_edge=True``, giving them 3 px dense sampling.

  3. **Jump arcs connect across the gap** — edges that originate on one side
     of the exclusion zone and land on the other represent valid mine-clearing
     trajectories enumerated during edge enumeration.

Output is a multi-panel figure:
  - Row 0 (full width): full-level d* heatmap with mine circles and
    edge-flagged points annotated
  - Rows 1-N: per-mine zoom panels (up to --max-mines, default 4) showing
    surface-point gaps, edges coloured by type, and inline statistics
  - Final panel: text summary with pass/fail per mine

Usage::

    python -m nclone.value_function.visualize_mine_gap \\
        --data value_functions/basics_surface_pbrs.npz \\
        --map-path nclone/test_maps/basics \\
        --output basics_mine_gap.png
"""

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from nclone.constants.physics_constants import NINJA_RADIUS, TOGGLE_MINE_RADIUS_TOGGLED
from nclone.value_function.surface_extraction import (
    MINE_SURFACE_EXCLUSION_RADIUS,
    SurfacePoint,
)
from nclone.value_function.surface_graph import build_graph, node_id_to_parts

# ---------------------------------------------------------------------------
# Radius constants
# ---------------------------------------------------------------------------
KILL_RADIUS = NINJA_RADIUS + TOGGLE_MINE_RADIUS_TOGGLED  # 14.0 px
EXCL_RADIUS = MINE_SURFACE_EXCLUSION_RADIUS              # 18.0 px
ARC_CHECK_RADIUS = KILL_RADIUS + 2.0                     # 16.0 px

# Zoom window half-widths around each mine (pixels)
_ZOOM_X = 110.0
_ZOOM_Y = 85.0

# An edge is "cross-gap" if src and dst are on strictly opposite sides of the
# mine exclusion zone along x (or y for vertical surfaces) with some margin.
_CROSS_GAP_MARGIN = EXCL_RADIUS + 2.0  # = 20 px beyond mine centre

# Level canvas size
_LEVEL_W = 1056
_LEVEL_H = 600


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_npz(path: str):
    """Return (graph, d_star_switch, d_max_switch, mine_positions)."""
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
    graph = build_graph(surface_points)
    if "edges_src" in data:
        graph.edges_src = data["edges_src"]
        graph.edges_dst = data["edges_dst"]
        graph.edges_weight = data["edges_weight"]

    d_star = data["d_star_switch"].astype(np.float32)
    meta = data["metadata"].item() if "metadata" in data else {}
    finite = d_star[np.isfinite(d_star)]
    d_max = float(meta.get("d_max_switch", float(np.max(finite)) if len(finite) else 1.0))

    mine_positions: List[Tuple[float, float]] = []
    if "mine_positions_x" in data and len(data["mine_positions_x"]) > 0:
        mxs = data["mine_positions_x"]
        mys = data["mine_positions_y"]
        mine_positions = [(float(mxs[i]), float(mys[i])) for i in range(len(mxs))]

    return graph, d_star, d_max, mine_positions


def _load_map(map_path: str):
    """Return (tile_dic, switch_pos, exit_pos, mine_positions_from_map)."""
    try:
        from nclone.value_function.physics_step_wrapper import PhysicsStepWrapper
        pw = PhysicsStepWrapper(map_path=map_path)
        mine_positions = []
        for e in pw.entity_dic.get(21, []):
            mine_positions.append((e.xpos, e.ypos))
        for e in pw.entity_dic.get(1, []):
            mine_positions.append((e.xpos, e.ypos))
        return pw.tile_dic, pw.get_goal_position("switch"), pw.get_goal_position("exit"), mine_positions
    except Exception as exc:
        print(f"Warning: could not load map '{map_path}': {exc}")
        return None, None, None, []


# ---------------------------------------------------------------------------
# Per-point cost helper
# ---------------------------------------------------------------------------
def _pt_costs(graph, d_star: np.ndarray) -> np.ndarray:
    """Cost-to-go per surface point using velocity bin 0 (zero velocity)."""
    mb = graph.max_vel_bins
    return np.array([
        float(d_star[i * mb]) if i * mb < len(d_star) else np.inf
        for i in range(len(graph.surface_points))
    ])


# ---------------------------------------------------------------------------
# Edge classification
# ---------------------------------------------------------------------------
def _classify_edges(
    graph,
    mine_x: float,
    mine_y: float,
) -> Tuple[List, List, List, List]:
    """Classify all edges relative to one mine.

    Returns:
        same_side_segs, same_side_wts — edges that stay on one side
        cross_gap_segs, cross_gap_wts — edges that span the exclusion zone
    """
    sp = graph.surface_points
    mb = graph.max_vel_bins
    n_pts = len(sp)

    same_side_segs: List = []
    same_side_wts: List = []
    cross_gap_segs: List = []
    cross_gap_wts: List = []

    for i in range(len(graph.edges_src)):
        si = int(graph.edges_src[i])
        di = int(graph.edges_dst[i])
        s_pt, _ = node_id_to_parts(si, mb)
        d_pt, _ = node_id_to_parts(di, mb)
        if s_pt >= n_pts or d_pt >= n_pts:
            continue

        sx, sy = sp[s_pt].x, sp[s_pt].y
        dx_p, dy_p = sp[d_pt].x, sp[d_pt].y
        w = float(graph.edges_weight[i])

        # Only care about edges near this mine in y
        if abs(sy - mine_y) > _ZOOM_Y * 2.0 and abs(dy_p - mine_y) > _ZOOM_Y * 2.0:
            continue

        s_signed = sx - mine_x
        d_signed = dx_p - mine_x
        seg = [(sx, sy), (dx_p, dy_p)]

        # Cross-gap: strictly opposite sides, both outside the exclusion radius
        if (abs(s_signed) > _CROSS_GAP_MARGIN
                and abs(d_signed) > _CROSS_GAP_MARGIN
                and s_signed * d_signed < 0):
            cross_gap_segs.append(seg)
            cross_gap_wts.append(w)
        else:
            same_side_segs.append(seg)
            same_side_wts.append(w)

    return same_side_segs, same_side_wts, cross_gap_segs, cross_gap_wts


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _draw_tiles(ax, tile_dic) -> None:
    if tile_dic is None:
        return
    try:
        from nclone.value_function.level_renderer import render_tiles_to_axis
        render_tiles_to_axis(ax, tile_dic, alpha=0.35)
    except Exception:
        pass


def _draw_mine_circles(ax, mine_x: float, mine_y: float) -> None:
    """Overlay exclusion zone (orange dashed), arc-check (dotted), kill zone (red)."""
    for radius, color, alpha_fill, lw, ls, label in [
        (EXCL_RADIUS,      "#f39c12", 0.07, 1.8, "--",  f"Exclusion ({EXCL_RADIUS:.0f}px)"),
        (ARC_CHECK_RADIUS, "#e67e22", 0.00, 1.2, ":",   f"Arc check ({ARC_CHECK_RADIUS:.0f}px)"),
        (KILL_RADIUS,      "#e74c3c", 0.10, 1.5, "-",   f"Kill ({KILL_RADIUS:.0f}px)"),
    ]:
        if alpha_fill > 0:
            ax.add_patch(plt.Circle((mine_x, mine_y), radius,
                                    color=color, fill=True, alpha=alpha_fill, zorder=4))
        ax.add_patch(plt.Circle((mine_x, mine_y), radius,
                                color=color, fill=False,
                                linewidth=lw, linestyle=ls, alpha=0.9, zorder=5))
    ax.plot(mine_x, mine_y, "r+", markersize=9, linewidth=1.5, zorder=6)


def _setup_axis(ax, title: str = "") -> None:
    ax.set_aspect("equal")
    ax.set_xlabel("X (px)", fontsize=8)
    ax.set_ylabel("Y (px)", fontsize=8)
    ax.tick_params(labelsize=7)
    if title:
        ax.set_title(title, fontsize=9, pad=4)


# ---------------------------------------------------------------------------
# Panel: full-level overview
# ---------------------------------------------------------------------------
def _draw_global_panel(
    ax,
    graph,
    d_star: np.ndarray,
    d_max: float,
    mine_positions: List[Tuple[float, float]],
    tile_dic=None,
    switch_pos=None,
    exit_pos=None,
) -> None:
    _draw_tiles(ax, tile_dic)
    ax.set_xlim(0, _LEVEL_W)
    ax.set_ylim(_LEVEL_H, 0)
    _setup_axis(ax, "Full Level — d* (switch phase) + mine zones + edge-flagged points")

    sp = graph.surface_points
    xs = np.array([p.x for p in sp])
    ys = np.array([p.y for p in sp])
    is_edge = np.array([p.is_edge for p in sp])
    costs = _pt_costs(graph, d_star)
    finite = np.isfinite(costs)

    if finite.any():
        norm = mcolors.Normalize(
            vmin=float(np.percentile(costs[finite], 2)),
            vmax=float(np.percentile(costs[finite], 98)),
        )
        sc = ax.scatter(xs[finite], ys[finite], c=costs[finite],
                        cmap="viridis_r", norm=norm, s=5, alpha=0.7, zorder=3,
                        label=f"Surface pts ({finite.sum():,})")
        plt.colorbar(sc, ax=ax, label="d* (frames)", fraction=0.02, pad=0.01)

    # Edge-flagged points — these should include C/D gap boundary nodes
    if is_edge.any():
        ax.scatter(xs[is_edge], ys[is_edge],
                   facecolors="none", edgecolors="#e74c3c",
                   s=28, linewidths=0.9, zorder=5,
                   label=f"Edge-flagged ({int(is_edge.sum()):,})")

    for mx, my in mine_positions:
        _draw_mine_circles(ax, mx, my)

    if switch_pos:
        ax.plot(*switch_pos, "g^", markersize=9, zorder=7, label="switch")
    if exit_pos:
        ax.plot(*exit_pos, "b*", markersize=10, zorder=7, label="exit")

    legend_patches = [
        mpatches.Patch(facecolor="#f39c12", alpha=0.4,
                       label=f"Excl zone ({EXCL_RADIUS:.0f} px)"),
        mpatches.Patch(facecolor="#e74c3c", alpha=0.3,
                       label=f"Kill zone ({KILL_RADIUS:.0f} px)"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + legend_patches,
              loc="upper right", fontsize=6, ncol=2)


# ---------------------------------------------------------------------------
# Panel: per-mine zoom
# ---------------------------------------------------------------------------
def _draw_mine_zoom_panel(
    ax,
    graph,
    d_star: np.ndarray,
    d_max: float,
    mine_x: float,
    mine_y: float,
    tile_dic=None,
    mine_idx: int = 0,
) -> dict:
    """Zoomed view of one mine — returns validation statistics dict."""
    sp = graph.surface_points
    mb = graph.max_vel_bins
    n_pts = len(sp)
    xs = np.array([p.x for p in sp])
    ys = np.array([p.y for p in sp])
    is_edge = np.array([p.is_edge for p in sp])
    costs = _pt_costs(graph, d_star)

    x_lo, x_hi = mine_x - _ZOOM_X, mine_x + _ZOOM_X
    y_lo, y_hi = mine_y - _ZOOM_Y, mine_y + _ZOOM_Y

    in_window = (xs >= x_lo) & (xs <= x_hi) & (ys >= y_lo) & (ys <= y_hi)

    dists = np.sqrt((xs - mine_x) ** 2 + (ys - mine_y) ** 2)
    in_excl = dists < EXCL_RADIUS
    near_boundary = (np.abs(dists - EXCL_RADIUS) < 14.0) & in_window

    n_in_excl = int(in_excl.sum())
    n_boundary = int(near_boundary.sum())
    n_boundary_edge = int((near_boundary & is_edge).sum())

    _draw_tiles(ax, tile_dic)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_hi, y_lo)
    _setup_axis(
        ax,
        f"Mine {mine_idx + 1} @ ({mine_x:.0f}, {mine_y:.0f})  "
        f"[±{_ZOOM_X:.0f}×{_ZOOM_Y:.0f} px zoom]",
    )

    # --- Classify and draw edges ---
    same_segs, same_wts, cross_segs, cross_wts = _classify_edges(
        graph, mine_x, mine_y)

    def _in_win(seg):
        (sx, sy), (dx, dy) = seg
        return ((x_lo <= sx <= x_hi or x_lo <= dx <= x_hi)
                and (y_lo <= sy <= y_hi or y_lo <= dy <= y_hi))

    same_w = [(s, w) for s, w in zip(same_segs, same_wts) if _in_win(s)]
    cross_w = [(s, w) for s, w in zip(cross_segs, cross_wts) if _in_win(s)]

    if same_w:
        lc_same = LineCollection(
            [s for s, _ in same_w], colors="#5d6d7e",
            linewidths=0.5, alpha=0.25, zorder=2,
        )
        ax.add_collection(lc_same)

    if cross_w:
        # Colour jump arcs by weight so heavier arcs stand out
        wts_arr = np.array([w for _, w in cross_w])
        norm_arc = mcolors.Normalize(vmin=wts_arr.min(), vmax=wts_arr.max())
        lc_cross = LineCollection(
            [s for s, _ in cross_w],
            cmap="autumn", norm=norm_arc,
            linewidths=1.4, alpha=0.85, zorder=3,
            label=f"Cross-gap arcs ({len(cross_w)})",
        )
        lc_cross.set_array(wts_arr)
        ax.add_collection(lc_cross)
        plt.colorbar(lc_cross, ax=ax, label="Arc weight (frames)",
                     fraction=0.025, pad=0.01)

    # --- Mine circles ---
    _draw_mine_circles(ax, mine_x, mine_y)

    # --- Surface points in window ---
    pts_show = in_window & np.isfinite(costs)
    if pts_show.any():
        c_vals = costs[pts_show]
        norm_pts = mcolors.Normalize(
            vmin=float(np.percentile(c_vals, 5)) if len(c_vals) > 2 else 0,
            vmax=float(np.percentile(c_vals, 95)) if len(c_vals) > 2 else 1,
        )
        ax.scatter(xs[pts_show], ys[pts_show], c=c_vals,
                   cmap="viridis_r", norm=norm_pts,
                   s=18, alpha=0.95, zorder=4,
                   label=f"Surface pts ({pts_show.sum()})")

    # --- Highlight any points inside exclusion zone (should be zero!) ---
    bad_pts = in_excl & in_window
    if bad_pts.any():
        ax.scatter(xs[bad_pts], ys[bad_pts],
                   c="#e74c3c", marker="x", s=80, linewidths=2.0, zorder=8,
                   label=f"INSIDE EXCLUSION ({int(bad_pts.sum())}) [FAIL]")

    # --- Boundary / gap-endpoint nodes (C and D) ---
    if near_boundary.any():
        edge_boundary = near_boundary & is_edge
        non_edge_boundary = near_boundary & ~is_edge
        if edge_boundary.any():
            ax.scatter(xs[edge_boundary], ys[edge_boundary],
                       facecolors="none", edgecolors="#2ecc71",
                       s=60, linewidths=1.5, zorder=6,
                       label=f"Gap boundary edge-flagged ({int(edge_boundary.sum())})")
        if non_edge_boundary.any():
            ax.scatter(xs[non_edge_boundary], ys[non_edge_boundary],
                       facecolors="none", edgecolors="#f1c40f",
                       s=60, linewidths=1.5, zorder=6,
                       label=f"Gap boundary NOT edge-flagged ({int(non_edge_boundary.sum())})")

    # --- Inline statistics box ---
    excl_status = "OK" if n_in_excl == 0 else f"FAIL ({n_in_excl} pts)"
    edge_status = "OK" if n_boundary_edge == n_boundary else (
        f"partial ({n_boundary_edge}/{n_boundary})")
    cross_status = "OK" if len(cross_w) > 0 else "NONE (unreachable?)"
    ax.text(
        0.02, 0.98,
        (f"In excl zone:   {n_in_excl:>3}  [{excl_status}]\n"
         f"Boundary pts:   {n_boundary:>3}  edge-flagged: {n_boundary_edge} [{edge_status}]\n"
         f"Cross-gap arcs: {len(cross_w):>3}  [{cross_status}]\n"
         f"Same-side edges:{len(same_w):>3}"),
        transform=ax.transAxes, fontsize=7.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88),
        family="monospace",
    )

    # --- Legend patches for mine circles ---
    legend_patches = [
        mpatches.Patch(facecolor="#f39c12", alpha=0.3,
                       label=f"Excl ({EXCL_RADIUS:.0f}px dashed)"),
        mpatches.Patch(facecolor="#e74c3c", alpha=0.25,
                       label=f"Kill ({KILL_RADIUS:.0f}px solid)"),
    ]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:5] + legend_patches,
              loc="lower right", fontsize=6)

    return {
        "mine_pos": (mine_x, mine_y),
        "n_in_excl": n_in_excl,
        "n_boundary": n_boundary,
        "n_boundary_edge": n_boundary_edge,
        "n_cross_gap": len(cross_segs),
        "n_cross_gap_window": len(cross_w),
        "n_same_side": len(same_segs),
    }


# ---------------------------------------------------------------------------
# Panel: text statistics
# ---------------------------------------------------------------------------
def _draw_stats_panel(
    ax,
    stats_list: List[dict],
    graph,
    d_star: np.ndarray,
    d_max: float,
) -> None:
    ax.axis("off")

    sp = graph.surface_points
    is_edge = np.array([p.is_edge for p in sp])
    costs = _pt_costs(graph, d_star)
    finite = np.isfinite(costs)

    lines = [
        "MINE VIRTUAL GAP — VALIDATION REPORT",
        "=" * 40,
        "",
        f"Surface points:    {len(sp):>6,}",
        f"  Edge-flagged:    {int(is_edge.sum()):>6,}",
        f"  Floor:           {int(sum(p.surface_type == 0 for p in sp)):>6,}",
        f"  Wall:            {int(sum(p.surface_type == 1 for p in sp)):>6,}",
        "",
        f"Graph nodes:       {graph.n_nodes:>6,}",
        f"Edges:             {len(graph.edges_src):>6,}",
        f"d_max (switch):    {d_max:>8.1f} frames",
        "",
        "Per-mine results:",
        "-" * 40,
    ]

    all_pass = True
    for i, s in enumerate(stats_list):
        mx, my = s["mine_pos"]
        excl_ok = s["n_in_excl"] == 0
        cross_ok = s["n_cross_gap"] > 0
        all_pass = all_pass and excl_ok
        lines += [
            f"Mine {i + 1}  @ ({mx:.0f}, {my:.0f})",
            f"  Pts inside excl zone:    {s['n_in_excl']:>3}  "
            f"{'[OK]' if excl_ok else '[FAIL]'}",
            f"  Boundary pts (gap edge): {s['n_boundary_edge']:>3} / "
            f"{s['n_boundary']:>3}",
            f"  Cross-gap arcs (total):  {s['n_cross_gap']:>3}  "
            f"{'[OK]' if cross_ok else '[none - check ceiling]'}",
            "",
        ]

    lines += [
        "=" * 40,
        f"OVERALL:  {'ALL GAPS CORRECT' if all_pass else 'FAILURES DETECTED'}",
        "",
        "Radii used:",
        f"  MINE_SURFACE_EXCLUSION_RADIUS = {EXCL_RADIUS:.1f} px",
        f"  MINE_ARC_CHECK_RADIUS         = {ARC_CHECK_RADIUS:.1f} px",
        f"  KILL_RADIUS                   = {KILL_RADIUS:.1f} px",
    ]

    ax.text(
        0.05, 0.97,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=8.5, va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f3f4", alpha=0.95),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and visualize mine virtual-gap in surface-graph PBRS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to *_surface_pbrs.npz precomputed file",
    )
    parser.add_argument(
        "--map-path", default=None,
        help="Optional .map file path for tile geometry overlay",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output image path (default: <level>_mine_gap.png)",
    )
    parser.add_argument(
        "--max-mines", type=int, default=4,
        help="Max per-mine zoom panels to render (default: 4)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: data file not found: {args.data}")
        return 1

    print(f"Loading: {args.data}")
    graph, d_star, d_max, mine_positions = _load_npz(args.data)

    tile_dic = None
    switch_pos = None
    exit_pos = None
    if args.map_path and os.path.exists(args.map_path):
        tile_dic, switch_pos, exit_pos, map_mines = _load_map(args.map_path)
        if map_mines:
            mine_positions = map_mines

    if not mine_positions:
        print("No mines found. Nothing to visualize.")
        print("Tip: use --map-path to load entity positions from the level file.")
        return 0

    print(f"Found {len(mine_positions)} mine(s).")

    # --- Console validation ---
    sp = graph.surface_points
    xs = np.array([p.x for p in sp])
    ys = np.array([p.y for p in sp])
    is_edge = np.array([p.is_edge for p in sp])
    print(f"Surface points: {len(sp):,}  |  Edge-flagged: {int(is_edge.sum()):,}")
    for i, (mx, my) in enumerate(mine_positions):
        dists = np.sqrt((xs - mx) ** 2 + (ys - my) ** 2)
        n_excl = int((dists < EXCL_RADIUS).sum())
        n_boundary = int((np.abs(dists - EXCL_RADIUS) < 14.0).sum())
        n_boundary_edge = int(((np.abs(dists - EXCL_RADIUS) < 14.0) & is_edge).sum()) \
            if n_boundary > 0 else 0
        status = "OK" if n_excl == 0 else f"FAIL ({n_excl} pts inside exclusion)"
        print(f"  Mine {i+1} @ ({mx:.1f}, {my:.1f}): excl_zone={status}  "
              f"boundary_pts={n_boundary} (edge-flagged: {n_boundary_edge})")

    # --- Build figure ---
    shown_mines = mine_positions[:args.max_mines]
    n_zoom = len(shown_mines)

    # Layout: row 0 = global panel (full width)
    #         rows 1..ceil(n_zoom/2) = zoom panels (2 per row)
    #         final slot = stats
    n_zoom_rows = math.ceil((n_zoom + 1) / 2)  # +1 to reserve a slot for stats
    n_rows = 1 + n_zoom_rows
    fig_h = 7 + 7 * n_zoom_rows
    fig = plt.figure(figsize=(22, fig_h))
    gs = fig.add_gridspec(n_rows, 2, hspace=0.42, wspace=0.28)

    # Row 0: full-level overview spanning both columns
    ax_global = fig.add_subplot(gs[0, :])
    _draw_global_panel(
        ax_global, graph, d_star, d_max,
        mine_positions, tile_dic, switch_pos, exit_pos,
    )

    # Rows 1+: zoom panels then stats
    stats_list = []
    panel_slots = list(range(n_zoom + 1))  # n_zoom zooms + 1 stats
    for slot, (mx, my) in enumerate(shown_mines):
        row = 1 + slot // 2
        col = slot % 2
        ax_zoom = fig.add_subplot(gs[row, col])
        stats = _draw_mine_zoom_panel(
            ax_zoom, graph, d_star, d_max, mx, my, tile_dic, mine_idx=slot,
        )
        stats_list.append(stats)

    # Stats panel in the next free slot
    stats_slot = n_zoom
    stats_row = 1 + stats_slot // 2
    stats_col = stats_slot % 2
    ax_stats = fig.add_subplot(gs[stats_row, stats_col])
    _draw_stats_panel(ax_stats, stats_list, graph, d_star, d_max)

    level_name = os.path.splitext(os.path.basename(args.data))[0]
    fig.suptitle(
        f"Mine Virtual-Gap Validation — {level_name}\n"
        f"Excl zone: {EXCL_RADIUS:.0f} px  |  Arc check: {ARC_CHECK_RADIUS:.0f} px  "
        f"|  Kill zone: {KILL_RADIUS:.0f} px  |  "
        f"{len(mine_positions)} mine(s)  |  {len(sp):,} surface pts",
        fontsize=13, fontweight="bold",
    )

    output = args.output or f"{level_name}_mine_gap.png"
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
