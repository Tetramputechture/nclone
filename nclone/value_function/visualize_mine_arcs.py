#!/usr/bin/env python3
"""Deep arc visualization for a single surface mine.

Re-simulates the physics engine to draw actual jump trajectory curves across
the mine gap, then presents two complementary views:

  1. **Arc overview** (full-width panel): All valid cross-gap trajectories from
     pre-mine launch nodes, drawn as curves and coloured by arc cost.  Shows
     which arcs physically clear the mine and at what frame cost.

  2. **Per-velocity-bin panels** (grid): One panel per selected floor velocity
     bin showing (a) surface-point d* values at that specific velocity state
     and (b) the jump arcs available from that bin.  Illustrates how the
     approach velocity affects which arcs are accessible and how the potential
     gradient varies across velocity states.

Usage::

    python -m nclone.value_function.visualize_mine_arcs \\
        --data value_functions/basics_surface_pbrs.npz \\
        --map-path test-single-level/basics \\
        --mine-idx 1 \\
        --output basics_mine2_arcs.png
"""

import argparse
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
from nclone.value_function.edge_enumeration import (
    JUMP_HOLD_DURATIONS, H_INPUTS, POST_RELEASE_INPUTS,
    _DEATH_STATES, _SURFACE_STATES,
)
from nclone.value_function.surface_extraction import (
    MINE_SURFACE_EXCLUSION_RADIUS,
    SurfacePoint,
    SURFACE_FLOOR,
)
from nclone.value_function.surface_graph import (
    build_graph,
    num_velocity_bins,
    velocity_bin_centers,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KILL_RADIUS = NINJA_RADIUS + TOGGLE_MINE_RADIUS_TOGGLED  # 14.0 px
EXCL_RADIUS = MINE_SURFACE_EXCLUSION_RADIUS              # 18.0 px
ARC_CHECK_RADIUS = KILL_RADIUS + 2.0                     # 16.0 px

_ZOOM_X = 155.0   # half-width of zoom window
_ZOOM_Y = 90.0    # half-height

# How far left of the exclusion boundary to look for launch points
_LAUNCH_DEPTH = 65.0
# Max distance from mine y to qualify as "near this surface"
_SURFACE_Y_TOL = 22.0
# Max arcs per velocity bin in overview panel (for readability)
_MAX_ARCS_PER_BIN_OVERVIEW = 20
# Velocity bins shown in the per-bin grid (indices into the 15-bin floor array)
_DISPLAY_BINS = [0, 3, 7, 11, 14]
_BIN_LABELS = [
    "Fastest left  (from right →)",
    "Moderate left (from right →)",
    "Near-zero  (both directions ↔)",
    "Moderate right (from left →)",
    "Fastest right  (from left →)",
]
# Which arc directions to display per velocity bin:
#   leftward bins  → ninja approaches from right, so show R→L arcs
#   near-zero      → show both
#   rightward bins → ninja approaches from left, so show L→R arcs
_BIN_DIRECTIONS = ["R→L", "R→L", "both", "L→R", "L→R"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_npz(path: str):
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
    mines: List[Tuple[float, float]] = []
    if "mine_positions_x" in data and len(data["mine_positions_x"]) > 0:
        mxs, mys = data["mine_positions_x"], data["mine_positions_y"]
        mines = [(float(mxs[i]), float(mys[i])) for i in range(len(mxs))]
    return graph, d_star, d_max, mines


# ---------------------------------------------------------------------------
# Physics simulation
# ---------------------------------------------------------------------------
def _simulate_one_arc(
    sim,
    src_x: float, src_y: float, vx0: float,
    h_during: int, hold_dur: int, h_after: int,
    mine_x: float, mine_y: float,
    target_side: str = "right",
) -> Optional[Dict]:
    """Simulate one jump arc and return trajectory data if it crosses the gap.

    Args:
        target_side: ``"right"`` — arc must land post-mine (x > mine_x + EXCL_RADIUS).
                     ``"left"``  — arc must land pre-mine  (x < mine_x − EXCL_RADIUS).

    Returns a dict with ``waypoints``, ``frames``, ``landing_x``, ``landing_y``
    if the arc clears the mine and lands on the correct side.  Returns None otherwise.
    """
    MAX_FRAMES = 400
    snap = sim.save_ninja_state()
    sim.inject_ninja_for_value_iteration(src_x, src_y, vx0, 0.0, 0)

    # Initiate jump (first frame)
    sim.tick_tiles_only(h_during, 1)
    sc = sim.get_ninja_state_code()
    if sc in _DEATH_STATES:
        sim.restore_ninja_state(snap)
        return None

    waypoints: List[Tuple[float, float]] = [(src_x, src_y)]

    # Jump-hold phase
    for _ in range(min(hold_dur - 1, 45)):
        sim.tick_tiles_only(h_during, 1)
        sc = sim.get_ninja_state_code()
        if sc in _DEATH_STATES:
            sim.restore_ninja_state(snap)
            return None
        pos = sim.get_ninja_pos()
        waypoints.append((pos[0], pos[1]))

    # Post-release phase
    for _ in range(MAX_FRAMES - hold_dur):
        sim.tick_tiles_only(h_after, 0)
        sc = sim.get_ninja_state_code()
        if sc in _DEATH_STATES:
            sim.restore_ninja_state(snap)
            return None
        pos = sim.get_ninja_pos()
        vel = sim.get_ninja_velocity()
        waypoints.append((pos[0], pos[1]))

        # Reject arcs that enter the mine kill zone
        dist_to_mine = ((pos[0] - mine_x) ** 2 + (pos[1] - mine_y) ** 2) ** 0.5
        if dist_to_mine < ARC_CHECK_RADIUS:
            sim.restore_ninja_state(snap)
            return None

        if sc in _SURFACE_STATES:
            if target_side == "right":
                landed_ok = pos[0] > mine_x + EXCL_RADIUS - 2.0
            else:
                landed_ok = pos[0] < mine_x - EXCL_RADIUS + 2.0
            sim.restore_ninja_state(snap)
            if landed_ok:
                return {
                    "waypoints": waypoints,
                    "frames": len(waypoints),
                    "landing_x": pos[0],
                    "landing_y": pos[1],
                    "vx_land": vel[0],
                }
            return None

    sim.restore_ninja_state(snap)
    return None


def simulate_all_cross_gap_arcs(
    sim,
    graph,
    mine_x: float,
    mine_y: float,
) -> List[Dict]:
    """Simulate cross-gap arcs from BOTH sides of the mine.

    Left-side launch (→ right landing): covers rightward velocity states.
    Right-side launch (→ left landing): covers leftward velocity states.

    Returns one dict per valid cross-gap trajectory::

        {pt_idx, vel_bin, vx0, src_x, src_y, direction,
         waypoints, frames, landing_x, landing_y, vx_land}
    """
    sp_list = graph.surface_points
    centers = velocity_bin_centers(SURFACE_FLOOR)
    n_bins = num_velocity_bins(SURFACE_FLOOR)

    # Left-side launch zone (pre-mine): approach from left
    launch_pts_left = [
        i for i, sp in enumerate(sp_list)
        if sp.surface_type == SURFACE_FLOOR
        and (mine_x - EXCL_RADIUS - _LAUNCH_DEPTH) <= sp.x <= (mine_x - EXCL_RADIUS)
        and abs(sp.y - mine_y) < _SURFACE_Y_TOL
    ]
    # Right-side launch zone (post-mine): approach from right
    launch_pts_right = [
        i for i, sp in enumerate(sp_list)
        if sp.surface_type == SURFACE_FLOOR
        and (mine_x + EXCL_RADIUS) <= sp.x <= (mine_x + EXCL_RADIUS + _LAUNCH_DEPTH)
        and abs(sp.y - mine_y) < _SURFACE_Y_TOL
    ]
    print(f"  Left-side launch points  (→ right): {len(launch_pts_left)}")
    print(f"  Right-side launch points (→ left):  {len(launch_pts_right)}")

    arcs: List[Dict] = []
    snap = sim.save_ninja_state()

    def _run_side(launch_pts, target_side: str, direction_tag: str) -> None:
        for pt_idx in launch_pts:
            sp = sp_list[pt_idx]
            for vel_bin in range(n_bins):
                vx0 = float(centers[vel_bin])
                for h_during in H_INPUTS:
                    for hold_dur in JUMP_HOLD_DURATIONS:
                        for h_after in POST_RELEASE_INPUTS:
                            result = _simulate_one_arc(
                                sim, sp.x, sp.y, vx0,
                                h_during, hold_dur, h_after,
                                mine_x, mine_y,
                                target_side=target_side,
                            )
                            if result is not None:
                                arcs.append({
                                    "pt_idx": pt_idx,
                                    "vel_bin": vel_bin,
                                    "vx0": vx0,
                                    "src_x": sp.x,
                                    "src_y": sp.y,
                                    "direction": direction_tag,
                                    **result,
                                })

    _run_side(launch_pts_left,  target_side="right", direction_tag="L→R")
    _run_side(launch_pts_right, target_side="left",  direction_tag="R→L")

    sim.restore_ninja_state(snap)
    n_lr = sum(1 for a in arcs if a["direction"] == "L→R")
    n_rl = sum(1 for a in arcs if a["direction"] == "R→L")
    print(f"  Left→Right arcs: {n_lr}   Right→Left arcs: {n_rl}   Total: {len(arcs)}")
    return arcs


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _draw_mine_circles(ax, mine_x: float, mine_y: float) -> None:
    for radius, color, alpha_fill, lw, ls in [
        (EXCL_RADIUS,      "#f39c12", 0.07, 1.8, "--"),
        (ARC_CHECK_RADIUS, "#e67e22", 0.00, 1.2, ":"),
        (KILL_RADIUS,      "#e74c3c", 0.12, 1.5, "-"),
    ]:
        if alpha_fill > 0:
            ax.add_patch(plt.Circle((mine_x, mine_y), radius,
                                    color=color, fill=True, alpha=alpha_fill, zorder=4))
        ax.add_patch(plt.Circle((mine_x, mine_y), radius,
                                color=color, fill=False,
                                linewidth=lw, linestyle=ls, alpha=0.9, zorder=5))
    ax.plot(mine_x, mine_y, "r+", markersize=9, linewidth=1.5, zorder=6)


def _draw_tiles(ax, tile_dic) -> None:
    if tile_dic is None:
        return
    try:
        from nclone.value_function.level_renderer import render_tiles_to_axis
        render_tiles_to_axis(ax, tile_dic, alpha=0.35)
    except Exception:
        pass


def _setup_zoom_axis(ax, mine_x, mine_y, title="") -> None:
    ax.set_xlim(mine_x - _ZOOM_X, mine_x + _ZOOM_X)
    ax.set_ylim(mine_y + _ZOOM_Y, mine_y - _ZOOM_Y)
    ax.set_aspect("equal")
    ax.set_xlabel("X (px)", fontsize=8)
    ax.set_ylabel("Y (px)", fontsize=8)
    ax.tick_params(labelsize=7)
    if title:
        ax.set_title(title, fontsize=9, pad=4)


def _pt_cost_for_bin(d_star: np.ndarray, pt_idx: int, vel_bin: int, max_vel_bins: int) -> float:
    node = pt_idx * max_vel_bins + vel_bin
    return float(d_star[node]) if node < len(d_star) else np.inf


def _draw_surface_pts_by_velbin(
    ax,
    graph,
    d_star: np.ndarray,
    vel_bin: int,
    mine_x: float, mine_y: float,
    norm: mcolors.Normalize,
) -> None:
    """Scatter surface points near mine coloured by d*[vel_bin]."""
    sp_list = graph.surface_points
    mb = graph.max_vel_bins
    x_lo, x_hi = mine_x - _ZOOM_X, mine_x + _ZOOM_X
    y_lo, y_hi = mine_y - _ZOOM_Y, mine_y + _ZOOM_Y

    pts_x, pts_y, costs, edge_flags = [], [], [], []
    for i, sp in enumerate(sp_list):
        if not (x_lo <= sp.x <= x_hi and y_lo <= sp.y <= y_hi):
            continue
        if vel_bin >= num_velocity_bins(sp.surface_type):
            continue
        c = _pt_cost_for_bin(d_star, i, vel_bin, mb)
        pts_x.append(sp.x)
        pts_y.append(sp.y)
        costs.append(c)
        edge_flags.append(sp.is_edge)

    if not pts_x:
        return

    xs_a = np.array(pts_x)
    ys_a = np.array(pts_y)
    cs_a = np.array(costs)
    finite = np.isfinite(cs_a)

    ax.scatter(xs_a[finite], ys_a[finite], c=cs_a[finite],
               cmap="viridis_r", norm=norm, s=22, alpha=0.92, zorder=3,
               label=f"Surface pts ({finite.sum()})")
    if not finite.all():
        ax.scatter(xs_a[~finite], ys_a[~finite],
                   c="#aaaaaa", s=14, alpha=0.5, zorder=2, label="Unreachable")


def _arcs_to_line_segments(arcs: List[Dict]) -> Tuple[List, List[float]]:
    """Convert arc waypoints to line-segment pairs for LineCollection."""
    segments, costs = [], []
    for arc in arcs:
        wps = arc["waypoints"]
        if len(wps) < 2:
            continue
        for k in range(len(wps) - 1):
            segments.append([wps[k], wps[k + 1]])
            costs.append(arc["frames"])
    return segments, costs


# ---------------------------------------------------------------------------
# Panel: full arc overview
# ---------------------------------------------------------------------------
def _draw_arc_overview(
    ax,
    graph,
    d_star: np.ndarray,
    d_max: float,
    mine_x: float, mine_y: float,
    arcs: List[Dict],
    tile_dic=None,
) -> None:
    """All cross-gap arcs overlaid on d*[vel_bin_7≈0] surface heatmap."""
    _draw_tiles(ax, tile_dic)
    _setup_zoom_axis(ax, mine_x, mine_y,
                     f"Cross-gap arc trajectories — Mine @ ({mine_x:.0f}, {mine_y:.0f})  "
                     f"|  {len(arcs):,} valid arcs found")

    # Surface points at zero-velocity bin (bin 7)
    zero_bin = 7
    mb = graph.max_vel_bins
    x_lo, x_hi = mine_x - _ZOOM_X, mine_x + _ZOOM_X
    y_lo, y_hi = mine_y - _ZOOM_Y, mine_y + _ZOOM_Y
    pts_x, pts_y, costs = [], [], []
    for i, sp in enumerate(graph.surface_points):
        if not (x_lo <= sp.x <= x_hi and y_lo <= sp.y <= y_hi):
            continue
        c = _pt_cost_for_bin(d_star, i, zero_bin, mb)
        pts_x.append(sp.x)
        pts_y.append(sp.y)
        costs.append(c)

    if pts_x:
        finite_c = [c for c in costs if np.isfinite(c)]
        norm_pts = mcolors.Normalize(
            vmin=float(np.percentile(finite_c, 2)) if finite_c else 0,
            vmax=float(np.percentile(finite_c, 98)) if finite_c else 1,
        )
        cs_a = np.array(costs)
        finite = np.isfinite(cs_a)
        sc = ax.scatter(
            np.array(pts_x)[finite], np.array(pts_y)[finite],
            c=cs_a[finite], cmap="viridis_r", norm=norm_pts,
            s=20, alpha=0.9, zorder=3, label=f"Surface d* (v≈0, {finite.sum()} pts)",
        )
        plt.colorbar(sc, ax=ax, label="d* (frames) at v≈0", fraction=0.025, pad=0.01)

    # Sub-sample arcs per (vel_bin, direction) bucket for readability
    rng = np.random.default_rng(42)
    selected_lr: List[Dict] = []
    selected_rl: List[Dict] = []
    by_bin_lr: Dict[int, List[Dict]] = {}
    by_bin_rl: Dict[int, List[Dict]] = {}
    for arc in arcs:
        if arc["direction"] == "L→R":
            by_bin_lr.setdefault(arc["vel_bin"], []).append(arc)
        else:
            by_bin_rl.setdefault(arc["vel_bin"], []).append(arc)
    for vb, bin_arcs in sorted(by_bin_lr.items()):
        n = min(_MAX_ARCS_PER_BIN_OVERVIEW, len(bin_arcs))
        chosen = rng.choice(len(bin_arcs), n, replace=False)
        selected_lr.extend(bin_arcs[i] for i in chosen)
    for vb, bin_arcs in sorted(by_bin_rl.items()):
        n = min(_MAX_ARCS_PER_BIN_OVERVIEW, len(bin_arcs))
        chosen = rng.choice(len(bin_arcs), n, replace=False)
        selected_rl.extend(bin_arcs[i] for i in chosen)

    all_frames = [a["frames"] for a in arcs] if arcs else [1, 100]
    norm_arc = mcolors.Normalize(
        vmin=float(np.percentile(all_frames, 2)),
        vmax=float(np.percentile(all_frames, 98)),
    )

    # L→R arcs — plasma (warm)
    if selected_lr:
        segs, seg_costs = _arcs_to_line_segments(selected_lr)
        lc_lr = LineCollection(segs, cmap="plasma", norm=norm_arc,
                               linewidths=0.6, alpha=0.55, zorder=5)
        lc_lr.set_array(np.array(seg_costs))
        ax.add_collection(lc_lr)
        plt.colorbar(lc_lr, ax=ax, label="Arc cost — L→R (frames)", fraction=0.022, pad=0.01)
        launch_xs = [a["src_x"] for a in selected_lr]
        launch_ys = [graph.surface_points[a["pt_idx"]].y for a in selected_lr]
        ax.scatter(launch_xs, launch_ys, c="#f39c12", s=40, marker="^",
                   zorder=7, label=f"Launch L→R ({len(set(launch_xs))} pts)")
        ax.scatter([a["landing_x"] for a in selected_lr],
                   [a["landing_y"] for a in selected_lr],
                   c="#f39c12", s=10, marker=".", alpha=0.35, zorder=6)

    # R→L arcs — winter (cool teal)
    if selected_rl:
        segs, seg_costs = _arcs_to_line_segments(selected_rl)
        lc_rl = LineCollection(segs, cmap="winter", norm=norm_arc,
                               linewidths=0.6, alpha=0.55, zorder=5)
        lc_rl.set_array(np.array(seg_costs))
        ax.add_collection(lc_rl)
        plt.colorbar(lc_rl, ax=ax, label="Arc cost — R→L (frames)", fraction=0.022, pad=0.04)
        launch_xs = [a["src_x"] for a in selected_rl]
        launch_ys = [graph.surface_points[a["pt_idx"]].y for a in selected_rl]
        ax.scatter(launch_xs, launch_ys, c="#1abc9c", s=40, marker="v",
                   zorder=7, label=f"Launch R→L ({len(set(launch_xs))} pts)")
        ax.scatter([a["landing_x"] for a in selected_rl],
                   [a["landing_y"] for a in selected_rl],
                   c="#1abc9c", s=10, marker=".", alpha=0.35, zorder=6)

    _draw_mine_circles(ax, mine_x, mine_y)

    n_lr_total = sum(1 for a in arcs if a["direction"] == "L→R")
    n_rl_total = sum(1 for a in arcs if a["direction"] == "R→L")
    legend_patches = [
        mpatches.Patch(facecolor="#f39c12", alpha=0.6, label=f"L→R: {n_lr_total} arcs"),
        mpatches.Patch(facecolor="#1abc9c", alpha=0.6, label=f"R→L: {n_rl_total} arcs"),
        mpatches.Patch(facecolor="#f39c12", alpha=0.3, label=f"Excl zone ({EXCL_RADIUS:.0f}px)"),
        mpatches.Patch(facecolor="#e74c3c", alpha=0.3, label=f"Kill zone ({KILL_RADIUS:.0f}px)"),
    ]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + legend_patches, loc="upper left", fontsize=6.5, ncol=2)


# ---------------------------------------------------------------------------
# Panel: one velocity bin
# ---------------------------------------------------------------------------
def _draw_velbin_panel(
    ax,
    graph,
    d_star: np.ndarray,
    d_max: float,
    mine_x: float, mine_y: float,
    vel_bin: int,
    bin_label: str,
    all_arcs_for_bin: List[Dict],
    direction_filter: str,
    global_arc_norm: mcolors.Normalize,
    tile_dic=None,
) -> None:
    """One panel: surface d*[vel_bin] and arcs for a specific direction.

    Args:
        direction_filter: ``"L→R"``, ``"R→L"``, or ``"both"``.
    """
    _draw_tiles(ax, tile_dic)
    vx_val = float(velocity_bin_centers(SURFACE_FLOOR)[vel_bin])

    # Filter arcs by direction
    if direction_filter == "both":
        arcs_lr = [a for a in all_arcs_for_bin if a["direction"] == "L→R"]
        arcs_rl = [a for a in all_arcs_for_bin if a["direction"] == "R→L"]
        display_arcs = all_arcs_for_bin
    elif direction_filter == "L→R":
        arcs_lr = [a for a in all_arcs_for_bin if a["direction"] == "L→R"]
        arcs_rl = []
        display_arcs = arcs_lr
    else:  # R→L
        arcs_lr = []
        arcs_rl = [a for a in all_arcs_for_bin if a["direction"] == "R→L"]
        display_arcs = arcs_rl

    _setup_zoom_axis(
        ax, mine_x, mine_y,
        f"Bin {vel_bin}: {bin_label}\n"
        f"vx = {vx_val:+.2f} px/f  |  "
        f"L→R: {len(arcs_lr)}  R→L: {len(arcs_rl)}",
    )

    # Surface points coloured by d* at this velocity bin
    mb = graph.max_vel_bins
    x_lo, x_hi = mine_x - _ZOOM_X, mine_x + _ZOOM_X
    y_lo, y_hi = mine_y - _ZOOM_Y, mine_y + _ZOOM_Y
    pts_x, pts_y, costs = [], [], []
    for i, sp in enumerate(graph.surface_points):
        if not (x_lo <= sp.x <= x_hi and y_lo <= sp.y <= y_hi):
            continue
        if vel_bin >= num_velocity_bins(sp.surface_type):
            continue
        c = _pt_cost_for_bin(d_star, i, vel_bin, mb)
        pts_x.append(sp.x)
        pts_y.append(sp.y)
        costs.append(c)

    if pts_x:
        cs_a = np.array(costs)
        finite = np.isfinite(cs_a)
        finite_c = cs_a[finite]
        norm_pts = mcolors.Normalize(
            vmin=float(np.percentile(finite_c, 2)) if len(finite_c) > 1 else 0,
            vmax=float(np.percentile(finite_c, 98)) if len(finite_c) > 1 else d_max,
        )
        sc = ax.scatter(
            np.array(pts_x)[finite], np.array(pts_y)[finite],
            c=cs_a[finite], cmap="viridis_r", norm=norm_pts,
            s=25, alpha=0.95, zorder=3,
        )
        cbar = plt.colorbar(sc, ax=ax, label="d* (frames)", fraction=0.03, pad=0.01)
        cbar.ax.tick_params(labelsize=7)

    # L→R arcs in plasma colormap (warm = expensive)
    if arcs_lr:
        segs, seg_costs = _arcs_to_line_segments(arcs_lr)
        lc = LineCollection(segs, cmap="plasma", norm=global_arc_norm,
                            linewidths=0.9, alpha=0.75, zorder=4,
                            label=f"L→R ({len(arcs_lr)})")
        lc.set_array(np.array(seg_costs))
        ax.add_collection(lc)
        for arc in arcs_lr:
            src_y = graph.surface_points[arc["pt_idx"]].y
            ax.plot(arc["src_x"], src_y, "^", color="#f39c12",
                    markersize=5, zorder=7, alpha=0.85)
            ax.plot(arc["landing_x"], arc["landing_y"], ".",
                    color="#f39c12", markersize=4, zorder=6, alpha=0.5)

    # R→L arcs in a teal/cool colormap so they contrast clearly with plasma
    if arcs_rl:
        segs, seg_costs = _arcs_to_line_segments(arcs_rl)
        lc2 = LineCollection(segs, cmap="winter", norm=global_arc_norm,
                             linewidths=0.9, alpha=0.75, zorder=4,
                             label=f"R→L ({len(arcs_rl)})")
        lc2.set_array(np.array(seg_costs))
        ax.add_collection(lc2)
        for arc in arcs_rl:
            src_y = graph.surface_points[arc["pt_idx"]].y
            ax.plot(arc["src_x"], src_y, "v", color="#1abc9c",
                    markersize=5, zorder=7, alpha=0.85)
            ax.plot(arc["landing_x"], arc["landing_y"], ".",
                    color="#1abc9c", markersize=4, zorder=6, alpha=0.5)

    _draw_mine_circles(ax, mine_x, mine_y)

    # Legend for arc directions (only when both or either present)
    if arcs_lr or arcs_rl:
        dir_patches = []
        if arcs_lr:
            dir_patches.append(mpatches.Patch(color="#f39c12", label=f"L→R arcs ({len(arcs_lr)})"))
        if arcs_rl:
            dir_patches.append(mpatches.Patch(color="#1abc9c", label=f"R→L arcs ({len(arcs_rl)})"))
        ax.legend(handles=dir_patches, loc="lower right", fontsize=6.5)

    # Stats box
    all_f = [a["frames"] for a in display_arcs]
    if all_f:
        ax.text(0.02, 0.02,
                f"Total arcs: {len(display_arcs)}\nCost: {min(all_f)}–{max(all_f)} f",
                transform=ax.transAxes, fontsize=7, va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
                family="monospace")
    else:
        ax.text(0.5, 0.5, "No cross-gap arcs\nfrom this velocity",
                transform=ax.transAxes, fontsize=9, ha="center", va="center",
                color="#888888")


# ---------------------------------------------------------------------------
# Panel: cost distribution histogram
# ---------------------------------------------------------------------------
def _draw_cost_histogram(
    ax, arcs: List[Dict], mine_x: float, mine_y: float,
) -> None:
    """Arc-cost distribution split by direction (L→R vs R→L)."""
    ax.set_xlabel("Arc cost (frames)", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title("Arc cost distribution — L→R (plasma) vs R→L (teal)", fontsize=9, pad=4)
    ax.tick_params(labelsize=7)

    centers = velocity_bin_centers(SURFACE_FLOOR)
    lr_by_bin: Dict[int, List[int]] = {}
    rl_by_bin: Dict[int, List[int]] = {}
    for arc in arcs:
        if arc["direction"] == "L→R":
            lr_by_bin.setdefault(arc["vel_bin"], []).append(arc["frames"])
        else:
            rl_by_bin.setdefault(arc["vel_bin"], []).append(arc["frames"])

    all_bins = sorted(set(list(lr_by_bin.keys()) + list(rl_by_bin.keys())))
    plasma_colors = plt.cm.plasma(np.linspace(0.15, 0.85, max(len(all_bins), 1)))
    winter_colors = plt.cm.winter(np.linspace(0.15, 0.85, max(len(all_bins), 1)))

    for idx, vb in enumerate(all_bins):
        if vb in lr_by_bin:
            ax.hist(lr_by_bin[vb], bins=25, alpha=0.55, color=plasma_colors[idx],
                    label=f"L→R bin{vb} vx={centers[vb]:+.2f}", density=False)
        if vb in rl_by_bin:
            ax.hist(rl_by_bin[vb], bins=25, alpha=0.55, color=winter_colors[idx],
                    label=f"R→L bin{vb} vx={centers[vb]:+.2f}", density=False,
                    linestyle="dashed", histtype="step", linewidth=1.5)

    ax.legend(loc="upper right", fontsize=6.5, ncol=3)
    ax.grid(axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Arc trajectory visualization for a single surface mine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data", required=True,
                        help="Path to *_surface_pbrs.npz")
    parser.add_argument("--map-path", required=True,
                        help=".map file (required for physics simulation)")
    parser.add_argument("--mine-idx", type=int, default=1,
                        help="0-based mine index to visualize (default: 1 = Mine 2)")
    parser.add_argument("--output", default=None,
                        help="Output image path")
    args = parser.parse_args()

    for p in [args.data, args.map_path]:
        if not os.path.exists(p):
            print(f"ERROR: not found: {p}")
            return 1

    print(f"Loading: {args.data}")
    graph, d_star, d_max, mines = _load_npz(args.data)

    if not mines:
        print("No mines in data.")
        return 1
    if args.mine_idx >= len(mines):
        print(f"Mine index {args.mine_idx} out of range (found {len(mines)} mines).")
        return 1

    mine_x, mine_y = mines[args.mine_idx]
    print(f"Visualizing Mine {args.mine_idx + 1} @ ({mine_x:.1f}, {mine_y:.1f})")

    # Load tile geometry
    tile_dic = None
    try:
        from nclone.value_function.physics_step_wrapper import PhysicsStepWrapper
        pw = PhysicsStepWrapper(map_path=args.map_path)
        tile_dic = pw.tile_dic
    except Exception as e:
        print(f"Warning: tile dict unavailable: {e}")

    # Load simulator and run arc simulations
    print("Simulating cross-gap arcs (this takes ~15–30 seconds)...")
    from nclone_native import Simulator as NativeSimulator
    with open(args.map_path, "rb") as f:
        map_data = [int(b) for b in f.read()]
    sim = NativeSimulator.load(map_data)

    arcs = simulate_all_cross_gap_arcs(sim, graph, mine_x, mine_y)

    if not arcs:
        print("No cross-gap arcs found. Check mine index and map path.")
        return 1

    # Group arcs by velocity bin
    by_bin: Dict[int, List[Dict]] = {}
    for arc in arcs:
        by_bin.setdefault(arc["vel_bin"], []).append(arc)

    all_frames = [a["frames"] for a in arcs]
    global_arc_norm = mcolors.Normalize(
        vmin=float(np.percentile(all_frames, 2)),
        vmax=float(np.percentile(all_frames, 98)),
    )

    print(f"Arc cost range: {min(all_frames)}–{max(all_frames)} frames  "
          f"|  Velocity bins with arcs: {sorted(by_bin.keys())}")

    # ---------------------------------------------------------------------------
    # Build figure
    # ---------------------------------------------------------------------------
    # Layout:
    #   Row 0: overview panel (tall, full width)
    #   Row 1: first three velocity-bin panels
    #   Row 2: last two velocity-bin panels + histogram
    n_cols = 3
    # height_ratios: overview is 2× taller than a vel-bin row; histogram is 0.9×
    height_ratios = [2.2, 1.5, 1.5, 0.9]
    n_rows = len(height_ratios)
    fig_height = sum(height_ratios) * 4.2
    fig = plt.figure(figsize=(22, fig_height))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.45, wspace=0.30,
                          height_ratios=height_ratios)

    # Row 0: overview (full width)
    ax_overview = fig.add_subplot(gs[0, :])
    _draw_arc_overview(ax_overview, graph, d_star, d_max,
                       mine_x, mine_y, arcs, tile_dic)

    # Rows 1–2: per-velocity-bin panels (3 cols each row)
    for panel_i, (vb, bin_label, bin_dir) in enumerate(
        zip(_DISPLAY_BINS, _BIN_LABELS, _BIN_DIRECTIONS)
    ):
        row = 1 + panel_i // n_cols
        col = panel_i % n_cols
        ax_bin = fig.add_subplot(gs[row, col])
        _draw_velbin_panel(
            ax_bin, graph, d_star, d_max,
            mine_x, mine_y,
            vel_bin=vb,
            bin_label=bin_label,
            all_arcs_for_bin=by_bin.get(vb, []),
            direction_filter=bin_dir,
            global_arc_norm=global_arc_norm,
            tile_dic=tile_dic,
        )

    # Row 3: histogram (full width)
    ax_hist = fig.add_subplot(gs[3, :])
    _draw_cost_histogram(ax_hist, arcs, mine_x, mine_y)

    level_name = os.path.splitext(os.path.basename(args.data))[0]
    centers = velocity_bin_centers(SURFACE_FLOOR)
    fig.suptitle(
        f"Mine Arc Analysis — {level_name}  |  Mine {args.mine_idx + 1} @ "
        f"({mine_x:.0f}, {mine_y:.0f})\n"
        f"{len(arcs):,} valid cross-gap trajectories across "
        f"{len(by_bin)} velocity bins  |  "
        f"excl zone: {EXCL_RADIUS:.0f}px  |  arc check: {ARC_CHECK_RADIUS:.0f}px",
        fontsize=12, fontweight="bold",
    )

    output = args.output or f"{level_name}_mine{args.mine_idx + 1}_arcs.png"
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
