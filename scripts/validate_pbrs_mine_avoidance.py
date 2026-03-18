#!/usr/bin/env python3
"""Validate precomputed PBRS mine avoidance for the basics level.

Performs a holistic analysis of the surface-graph PBRS value function,
focusing on whether the potential landscape correctly incentivises the agent
to route around toggle mines rather than walk into them.

Five checks are performed for each mine:
  A. Exclusion zone integrity  -- no surface nodes inside 18 px of mine centre
  B. Potential landscape probe -- d* / phi profile along the floor near the mine
  C. Gravity-well detection    -- secondary local minima in d* that attract the
                                  agent toward the mine
  D. Reward trajectory analysis -- per-frame F_t rewards for "walk into mine"
                                   vs "jump over mine" trajectories
  E. Cross-gap arc inventory   -- count and cost of edges that bridge the mine gap

A text report and a multi-panel PNG visualisation are produced.

Usage::

    python scripts/validate_pbrs_mine_avoidance.py \\
        --data value_functions/basics_surface_pbrs.npz \\
        [--map-path test-single-level/basics] \\
        [--output validation_output/]

All outputs go to --output (default: validation_output/).
"""

import argparse
import math
import os
import sys
import textwrap
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nclone.constants.physics_constants import (
    NINJA_RADIUS,
    TOGGLE_MINE_RADIUS_TOGGLED,
    MAX_HOR_SPEED,
)
from nclone.value_function.surface_extraction import (
    MINE_SURFACE_EXCLUSION_RADIUS,
    SurfacePoint,
    SURFACE_FLOOR,
    SURFACE_WALL,
)
from nclone.value_function.surface_graph import (
    build_graph,
    node_id_to_parts,
    velocity_bin_centers,
    velocity_to_bin,
    NUM_VELOCITY_BINS_FLOOR,
)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
KILL_RADIUS = NINJA_RADIUS + TOGGLE_MINE_RADIUS_TOGGLED   # 14.0 px
EXCL_RADIUS = MINE_SURFACE_EXCLUSION_RADIUS               # 18.0 px
C_PHASE = 0.5                                             # switch-collected bonus

# ---------------------------------------------------------------------------
# Training hyperparameters (from base_environment.py + reward_constants.py)
# ---------------------------------------------------------------------------
PBRS_GAMMA = 1.0        # PBRSConfig default (config.py line 114)
PBRS_ALPHA = 2.0        # base_environment.py line 348
R_DEATH = -0.5          # reward_constants.py line 45
R_DEATH_MINE = -2.0     # reward_constants.py — extra penalty when death_cause=='mine'
R_SWITCH = 1.0          # reward_constants.py line 43
R_EXIT = 5.0            # reward_constants.py line 44
# FiGAR mine interrupt threshold (figar_mine_interrupt_threshold in training config)
FIGAR_MINE_INTERRUPT_PX = 40.0

# Typical agent walking speed (pixels/frame) — approximated from MAX_HOR_SPEED
WALK_SPEED_PX = MAX_HOR_SPEED * 0.6   # typical run speed, not sprint

# ---------------------------------------------------------------------------
# Level dimensions
# ---------------------------------------------------------------------------
LEVEL_W, LEVEL_H = 1056, 600

# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_npz(path: str):
    """Load surface graph and all relevant arrays from a .npz file."""
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

    meta = data["metadata"].item() if "metadata" in data else {}
    d_star_switch = data["d_star_switch"].astype(np.float32)
    d_star_exit = data["d_star_exit"].astype(np.float32)

    # Robust (99th-pct) d_max
    def _p99(arr):
        finite = arr[np.isfinite(arr)]
        return float(np.percentile(finite, 99)) if len(finite) else 1.0

    d_max_switch = float(meta.get("d_max_switch", _p99(d_star_switch)))
    d_max_exit = float(meta.get("d_max_exit", _p99(d_star_exit)))
    d_max_switch_raw = float(meta.get("d_max_switch_raw", float(np.max(d_star_switch[np.isfinite(d_star_switch)]))))
    d_max_exit_raw = float(meta.get("d_max_exit_raw", float(np.max(d_star_exit[np.isfinite(d_star_exit)]))))

    mines_x = data["mine_positions_x"].astype(np.float32) if "mine_positions_x" in data else np.empty(0)
    mines_y = data["mine_positions_y"].astype(np.float32) if "mine_positions_y" in data else np.empty(0)
    mine_types = data["mine_types"].astype(np.uint8) if "mine_types" in data else np.empty(0, dtype=np.uint8)
    mine_initial_states = data["mine_initial_states"].astype(np.uint8) if "mine_initial_states" in data else np.empty(0, dtype=np.uint8)
    mine_boundary_nodes = data["mine_boundary_nodes"].astype(np.int32) if "mine_boundary_nodes" in data else None

    return {
        "graph": graph,
        "surface_points": surface_points,
        "d_star_switch": d_star_switch,
        "d_star_exit": d_star_exit,
        "d_max_switch": d_max_switch,
        "d_max_exit": d_max_exit,
        "d_max_switch_raw": d_max_switch_raw,
        "d_max_exit_raw": d_max_exit_raw,
        "meta": meta,
        "mines_x": mines_x,
        "mines_y": mines_y,
        "mine_types": mine_types,
        "mine_initial_states": mine_initial_states,
        "mine_boundary_nodes": mine_boundary_nodes,
        "n_surface_points": n,
    }


def _load_map_info(map_path: str):
    """Load tile dict and goal positions from a .map file (optional)."""
    try:
        from nclone.value_function.physics_step_wrapper import PhysicsStepWrapper
        pw = PhysicsStepWrapper(map_path=map_path)
        return {
            "tile_dic": pw.tile_dic,
            "switch_pos": pw.get_goal_position("switch"),
            "exit_pos": pw.get_goal_position("exit"),
        }
    except Exception as exc:
        print(f"  [warn] Could not load map '{map_path}': {exc}")
        return {"tile_dic": None, "switch_pos": None, "exit_pos": None}


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _pt_cost(d_star: np.ndarray, pt_idx: int, vel_bin: int, max_vel_bins: int) -> float:
    """Cost-to-go for a given (surface_point, velocity_bin) node."""
    nid = pt_idx * max_vel_bins + vel_bin
    return float(d_star[nid]) if nid < len(d_star) else math.inf


def _phi(cost: float, d_max: float) -> float:
    """Normalised PBRS potential: Phi(s) = -d*(s) / d_max, clamped to [-1, 0]."""
    if not math.isfinite(cost):
        return -1.0
    return max(-cost / d_max, -1.0)


def _floor_profile_near_mine(
    surface_points: List[SurfacePoint],
    d_star: np.ndarray,
    d_max: float,
    max_vel_bins: int,
    mine_x: float,
    mine_y: float,
    window_px: float = 180.0,
    y_tol: float = 22.0,
) -> List[Dict]:
    """Return floor surface points near the mine, sorted by x, with d*/phi at v=0/vL/vR."""
    floor_bin_centers = velocity_bin_centers(SURFACE_FLOOR)
    bin_v0 = velocity_to_bin(0.0, SURFACE_FLOOR)
    bin_vL = 0                               # fastest left
    bin_vR = NUM_VELOCITY_BINS_FLOOR - 1     # fastest right

    pts = []
    for i, sp in enumerate(surface_points):
        if sp.surface_type != SURFACE_FLOOR:
            continue
        if abs(sp.x - mine_x) > window_px:
            continue
        if abs(sp.y - mine_y) > y_tol:
            continue
        d_v0 = _pt_cost(d_star, i, bin_v0, max_vel_bins)
        d_vL = _pt_cost(d_star, i, bin_vL, max_vel_bins)
        d_vR = _pt_cost(d_star, i, bin_vR, max_vel_bins)
        pts.append({
            "idx": i,
            "x": sp.x,
            "y": sp.y,
            "is_edge": sp.is_edge,
            "d_v0": d_v0,
            "d_vL": d_vL,
            "d_vR": d_vR,
            "phi_v0": _phi(d_v0, d_max),
            "phi_vL": _phi(d_vL, d_max),
            "phi_vR": _phi(d_vR, d_max),
            "d_to_mine": abs(sp.x - mine_x),
        })
    pts.sort(key=lambda p: p["x"])
    return pts


def _detect_local_extrema(
    profile: List[Dict],
    field: str = "d_v0",
) -> Dict:
    """Find local minima and maxima in a 1D profile sequence.

    Profile items must have an ``"x_rel"`` key (position relative to mine centre,
    negative = left of mine) and the requested ``field`` for cost values.

    Returns a dict with keys:
      local_minima, local_maxima: list of dicts with ``x_rel``, ``value``, ``idx``
      monotonic:   bool — True if approach is strictly monotonic (ideal)
      gravity_well: None or the local-min dict closest to the mine
    """
    if len(profile) < 3:
        return {"local_minima": [], "local_maxima": [], "monotonic": True, "gravity_well": None}

    values = [p[field] for p in profile]
    x_rels = [p["x_rel"] for p in profile]

    local_minima = []
    local_maxima = []
    for i in range(1, len(values) - 1):
        v_prev, v_curr, v_next = values[i - 1], values[i], values[i + 1]
        if math.isfinite(v_curr) and math.isfinite(v_prev) and math.isfinite(v_next):
            if v_curr < v_prev and v_curr < v_next:
                local_minima.append({
                    "x_rel": x_rels[i], "value": v_curr, "idx": profile[i]["idx"]
                })
            elif v_curr > v_prev and v_curr > v_next:
                local_maxima.append({
                    "x_rel": x_rels[i], "value": v_curr, "idx": profile[i]["idx"]
                })

    # Gravity well: the local minimum on the approach side (x_rel < 0)
    # closest to the mine (smallest |x_rel|).
    gravity_well = None
    if local_minima:
        approach_mins = [m for m in local_minima if m["x_rel"] < 0]
        if approach_mins:
            gravity_well = min(approach_mins, key=lambda m: abs(m["x_rel"]))

    monotonic = len(local_minima) == 0

    return {
        "local_minima": local_minima,
        "local_maxima": local_maxima,
        "monotonic": monotonic,
        "gravity_well": gravity_well,
    }


def _cross_gap_edges(
    graph,
    mine_x: float,
    mine_y: float,
    y_tol: float = 40.0,
) -> Tuple[List[Dict], List[Dict]]:
    """Partition graph edges near a mine into cross-gap and same-side groups.

    Cross-gap: src on one side, dst on the other side of the exclusion zone.
    """
    sp = graph.surface_points
    mb = graph.max_vel_bins
    n_pts = len(sp)

    cross_gap = []
    same_side = []

    for i in range(len(graph.edges_src)):
        si = int(graph.edges_src[i])
        di = int(graph.edges_dst[i])
        s_pt, s_vb = node_id_to_parts(si, mb)
        d_pt, d_vb = node_id_to_parts(di, mb)
        if s_pt >= n_pts or d_pt >= n_pts:
            continue

        sx, sy = sp[s_pt].x, sp[s_pt].y
        dx, dy = sp[d_pt].x, sp[d_pt].y

        # Only consider edges near this mine in y
        if abs(sy - mine_y) > y_tol and abs(dy - mine_y) > y_tol:
            continue

        s_side = sx - mine_x   # negative = left of mine, positive = right
        d_side = dx - mine_x

        w = float(graph.edges_weight[i])

        if (abs(s_side) > EXCL_RADIUS - 2.0
                and abs(d_side) > EXCL_RADIUS - 2.0
                and s_side * d_side < 0):
            cross_gap.append({
                "src_x": sx, "src_y": sy,
                "dst_x": dx, "dst_y": dy,
                "weight": w,
                "src_pt": s_pt, "dst_pt": d_pt,
                "src_vb": s_vb, "dst_vb": d_vb,
            })
        else:
            if abs(sx - mine_x) <= EXCL_RADIUS * 3 or abs(dx - mine_x) <= EXCL_RADIUS * 3:
                same_side.append({
                    "src_x": sx, "dst_x": dx,
                    "weight": w,
                })

    return cross_gap, same_side


def _euclidean_dist_to_mine(
    pt_x: float, pt_y: float, mine_x: float, mine_y: float,
) -> float:
    """Euclidean distance from a surface point to the mine centre."""
    return math.sqrt((pt_x - mine_x) ** 2 + (pt_y - mine_y) ** 2)


def _simulate_walk_trajectory(
    profile: List[Dict],
    mine_x: float,
    mine_y: float,
    d_max: float,
    gamma: float = PBRS_GAMMA,
    alpha: float = PBRS_ALPHA,
    r_death: float = R_DEATH,
    r_death_mine: float = R_DEATH_MINE,
    stop_at_mine: bool = True,
) -> List[Dict]:
    """Simulate F_t = gamma*phi(s') - phi(s) for walking along the profile.

    Uses Euclidean distance to determine when the ninja enters the mine kill zone.
    Returns list of per-step dicts with x, phi, F_t, cumulative_reward.
    r_death_mine: extra penalty for mine deaths (R_DEATH + R_DEATH_MINE when death at mine).
    stop_at_mine: if True, terminates with death when Euclidean distance <= kill radius.
    """
    if not profile:
        return []

    # Skip if no profile point is actually within Euclidean kill range —
    # the mine may be below the floor (Mine 1 case).
    min_euc_dist = min(
        _euclidean_dist_to_mine(p["x"], p["y"], mine_x, mine_y) for p in profile
    )
    can_die = min_euc_dist <= KILL_RADIUS + 4.0   # 4px margin to account for sampling

    steps = []
    cumulative = 0.0
    prev_phi = _phi(profile[0]["d_v0"], d_max)

    for i, pt in enumerate(profile):
        euc_dist = _euclidean_dist_to_mine(pt["x"], pt["y"], mine_x, mine_y)

        if stop_at_mine and can_die and euc_dist <= KILL_RADIUS + 2.0:
            # Death at mine — terminal potential = 0; apply R_DEATH + R_DEATH_MINE
            f_t = alpha * (gamma * 0.0 - prev_phi) + r_death + r_death_mine
            cumulative += f_t
            steps.append({
                "x": pt["x"],
                "phi": 0.0,
                "F_t": f_t,
                "cumulative": cumulative,
                "event": "DEATH",
            })
            break

        # Detect the exclusion zone gap: consecutive profile pts with a large x gap that
        # straddles the mine.  When a floor-walking agent reaches the platform edge at
        # pt[i] and the next pt[i+1] is on the OTHER side of the mine, the agent would
        # fall into the mine gap — model this as a death event.  Use gap_contains_mine
        # (not can_die) because falling into the gap changes geometry; the agent can
        # hit the mine when descending even if no floor point is within kill range.
        if (stop_at_mine and i < len(profile) - 1
                and abs(profile[i + 1]["x"] - pt["x"]) > EXCL_RADIUS * 1.5):
            # Gap detected: check if mine_x falls in the gap
            x_gap_start = min(pt["x"], profile[i + 1]["x"])
            x_gap_end = max(pt["x"], profile[i + 1]["x"])
            gap_contains_mine = x_gap_start < mine_x < x_gap_end
            if gap_contains_mine:
                curr_phi = _phi(pt["d_v0"], d_max)
                if i > 0:
                    f_t = alpha * (gamma * curr_phi - prev_phi)
                    cumulative += f_t
                    steps.append({
                        "x": pt["x"],
                        "phi": curr_phi,
                        "F_t": f_t,
                        "cumulative": cumulative,
                        "event": "",
                    })
                # Falling off the edge → death inside the mine gap (R_DEATH + R_DEATH_MINE)
                f_t = alpha * (gamma * 0.0 - curr_phi) + r_death + r_death_mine
                cumulative += f_t
                steps.append({
                    "x": mine_x,
                    "phi": 0.0,
                    "F_t": f_t,
                    "cumulative": cumulative,
                    "event": "FALL→DEATH",
                })
                return steps

        curr_phi = _phi(pt["d_v0"], d_max)
        if i == 0:
            f_t = 0.0
        else:
            f_t = alpha * (gamma * curr_phi - prev_phi)

        cumulative += f_t
        steps.append({
            "x": pt["x"],
            "phi": curr_phi,
            "F_t": f_t,
            "cumulative": cumulative,
            "event": "",
        })
        prev_phi = curr_phi

    return steps


def _simulate_jump_trajectory(
    profile: List[Dict],
    jump_launch_x: float,
    jump_land_phi: float,
    jump_land_x: float,
    d_max: float,
    gamma: float = PBRS_GAMMA,
    alpha: float = PBRS_ALPHA,
) -> List[Dict]:
    """Simulate walking to jump_launch_x, then jumping to land at jump_land_x.

    jump_land_phi is the phi at landing (from cross-gap arc endpoint).
    Returns per-step dicts up to and including landing.
    """
    steps = []
    cumulative = 0.0
    prev_phi = None

    for pt in profile:
        if pt["x"] > jump_launch_x + 1.0:
            break
        curr_phi = _phi(pt["d_v0"], d_max)
        if prev_phi is None:
            f_t = 0.0
        else:
            f_t = alpha * (gamma * curr_phi - prev_phi)
        cumulative += f_t
        steps.append({
            "x": pt["x"],
            "phi": curr_phi,
            "F_t": f_t,
            "cumulative": cumulative,
            "event": "",
        })
        prev_phi = curr_phi

    # Jump landing
    if prev_phi is not None:
        f_t = alpha * (gamma * jump_land_phi - prev_phi)
        cumulative += f_t
        steps.append({
            "x": jump_land_x,
            "phi": jump_land_phi,
            "F_t": f_t,
            "cumulative": cumulative,
            "event": "JUMP_LAND",
        })

    return steps


# ---------------------------------------------------------------------------
# Per-mine analysis
# ---------------------------------------------------------------------------

def analyse_mine(
    mine_idx: int,
    mine_x: float,
    mine_y: float,
    mine_initial_state: int,
    mine_boundary_row: Optional[np.ndarray],
    data: Dict,
) -> Dict:
    """Run all five checks for one mine, return a results dict."""
    graph = data["graph"]
    sp = data["surface_points"]
    d_star = data["d_star_switch"]
    d_max = data["d_max_switch"]
    mb = graph.max_vel_bins

    # ---------- A. Exclusion zone integrity ----------
    xs = np.array([p.x for p in sp], dtype=np.float32)
    ys = np.array([p.y for p in sp], dtype=np.float32)
    dists = np.sqrt((xs - mine_x) ** 2 + (ys - mine_y) ** 2)
    inside_excl = int(np.sum(dists < EXCL_RADIUS))
    inside_kill = int(np.sum(dists < KILL_RADIUS))
    near_boundary = (np.abs(dists - EXCL_RADIUS) < 14.0)
    n_boundary = int(near_boundary.sum())
    is_edge = np.array([p.is_edge for p in sp])
    n_boundary_edge = int((near_boundary & is_edge).sum())

    excl_ok = inside_excl == 0
    kill_ok = inside_kill == 0

    # ---------- B. Potential landscape profile ----------
    profile = _floor_profile_near_mine(sp, d_star, d_max, mb, mine_x, mine_y)

    # ---------- C. Gravity-well detection ----------
    # Re-express x relative to mine centre (negative = approach from left)
    profile_relative = [{**p, "x_rel": p["x"] - mine_x} for p in profile]
    # Approach from the left (x < mine_x - exclusion boundary)
    approach_profile = [p for p in profile_relative if p["x_rel"] < -EXCL_RADIUS]
    extrema = _detect_local_extrema(approach_profile, field="d_v0")
    # Gravity well: local min on approach side closest to mine — reported in absolute coords
    gravity_well = None
    if extrema["gravity_well"] is not None:
        gw = extrema["gravity_well"]
        # x_rel is negative (left of mine), absolute x = mine_x + x_rel
        abs_x = mine_x + gw["x_rel"]
        gravity_well = {
            "x": abs_x,
            "x_rel": gw["x_rel"],
            "d_v0": gw["value"],
            "phi_v0": _phi(gw["value"], d_max),
            "dist_from_mine": abs(gw["x_rel"]),
        }

    # Determine repulsive gradient strength (phi change per px) in last 60px before excl boundary
    repulsive_strength = None
    repulsive_pts = [p for p in approach_profile if -60.0 <= p["x_rel"] < -EXCL_RADIUS]
    if len(repulsive_pts) >= 2:
        # phi should decrease as x increases toward mine (d* should increase)
        phi_start = repulsive_pts[0]["phi_v0"]
        phi_end = repulsive_pts[-1]["phi_v0"]
        span = repulsive_pts[-1]["x"] - repulsive_pts[0]["x"]
        if span != 0:
            repulsive_strength = (phi_end - phi_start) / span  # should be negative

    # ---------- D. Reward trajectory analysis ----------
    # Find optimal jump launch point: pre-mine edge node closest to mine
    pre_mine_edge_pts = [p for p in profile if p["is_edge"] and p["x"] < mine_x - EXCL_RADIUS + 2]
    jump_launch_pt = None
    if pre_mine_edge_pts:
        jump_launch_pt = max(pre_mine_edge_pts, key=lambda p: p["x"])

    # Find cheapest cross-gap arc landing point
    cross_gap_edges, same_side_edges = _cross_gap_edges(graph, mine_x, mine_y)
    best_cross = None
    if cross_gap_edges:
        valid = [e for e in cross_gap_edges if e["src_x"] < mine_x]
        if valid:
            best_cross = min(valid, key=lambda e: _pt_cost(d_star, e["dst_pt"], e["dst_vb"], mb))

    walk_traj = _simulate_walk_trajectory(
        profile, mine_x, mine_y, d_max,
        r_death=data.get("r_death", R_DEATH),
        r_death_mine=data.get("r_death_mine", R_DEATH_MINE),
    )
    jump_traj = []
    if jump_launch_pt and best_cross:
        land_cost = _pt_cost(d_star, best_cross["dst_pt"], best_cross["dst_vb"], mb)
        land_phi = _phi(land_cost, d_max)
        jump_traj = _simulate_jump_trajectory(
            profile, jump_launch_pt["x"], land_phi, best_cross["dst_x"], d_max)

    # Total cumulative reward comparison
    walk_total = walk_traj[-1]["cumulative"] if walk_traj else None
    jump_total = jump_traj[-1]["cumulative"] if jump_traj else None
    walk_ends_in_death = (
        bool(walk_traj)
        and walk_traj[-1].get("event") in ("DEATH", "FALL→DEATH")
    )

    # ---------- E. Cross-gap arc inventory ----------
    cross_gap_weights = [e["weight"] for e in cross_gap_edges]
    cross_gap_left_to_right = [e for e in cross_gap_edges if e["src_x"] < mine_x]
    cross_gap_right_to_left = [e for e in cross_gap_edges if e["src_x"] > mine_x]

    # ---------- Boundary node d* values ----------
    boundary_nodes_info = []
    if mine_boundary_row is not None:
        for node_idx in mine_boundary_row:
            if node_idx >= 0:
                pt_i, vb_i = node_id_to_parts(int(node_idx), mb)
                if pt_i < len(sp):
                    nd = _pt_cost(d_star, pt_i, vb_i, mb)
                    boundary_nodes_info.append({
                        "node_idx": int(node_idx),
                        "x": sp[pt_i].x,
                        "y": sp[pt_i].y,
                        "d": nd,
                        "phi": _phi(nd, d_max),
                        "is_edge": sp[pt_i].is_edge,
                    })

    # FiGAR interrupt effectiveness: does the interrupt trigger BEFORE the agent reaches the well?
    # figar_ok = True means FiGAR interrupts at FIGAR_MINE_INTERRUPT_PX, which is FURTHER from the
    # mine than the gravity well (dist_gw), so the macro is interrupted before the well is reached.
    figar_ok = True
    if gravity_well is not None:
        dist_gw = gravity_well["dist_from_mine"]
        # FiGAR triggers at dist=FIGAR_MINE_INTERRUPT_PX from mine.
        # Well is at dist=dist_gw. Agent reaches dist=FIGAR first (larger), so FiGAR fires first.
        figar_ok = FIGAR_MINE_INTERRUPT_PX >= dist_gw

    return {
        "mine_idx": mine_idx,
        "mine_x": mine_x,
        "mine_y": mine_y,
        "initial_state": mine_initial_state,
        # A
        "excl_zone_ok": excl_ok,
        "kill_zone_ok": kill_ok,
        "n_inside_excl": inside_excl,
        "n_inside_kill": inside_kill,
        "n_boundary": n_boundary,
        "n_boundary_edge": n_boundary_edge,
        # B
        "profile": profile,
        "approach_profile": approach_profile,
        # C
        "extrema": extrema,
        "gravity_well": gravity_well,
        "repulsive_strength": repulsive_strength,
        # D
        "walk_traj": walk_traj,
        "jump_traj": jump_traj,
        "walk_total_reward": walk_total,
        "walk_ends_in_death": walk_ends_in_death,
        "jump_total_reward": jump_total,
        "jump_launch_pt": jump_launch_pt,
        # E
        "cross_gap_edges": cross_gap_edges,
        "cross_gap_left_to_right": cross_gap_left_to_right,
        "cross_gap_right_to_left": cross_gap_right_to_left,
        "cross_gap_weights": cross_gap_weights,
        "same_side_edges": same_side_edges,
        # boundary nodes
        "boundary_nodes_info": boundary_nodes_info,
        # FiGAR
        "figar_ok": figar_ok,
    }


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def _print_separator(char: str = "-", width: int = 72) -> None:
    print(char * width)


def print_report(data: Dict, mine_results: List[Dict]) -> None:
    """Print the full validation report to stdout."""
    _print_separator("=")
    print("  PBRS MINE AVOIDANCE VALIDATION REPORT")
    print(f"  basics_surface_pbrs.npz")
    _print_separator("=")
    print()

    meta = data["meta"]
    print(f"Surface points:  {data['n_surface_points']:,}")
    print(f"Graph nodes:     {data['meta'].get('n_graph_nodes', '?'):,}")
    print(f"Edges:           {data['meta'].get('n_edges', '?'):,}")
    print(f"d_max_switch:    {data['d_max_switch']:.1f} frames (p99)")
    print(f"d_max_switch_raw:{data['d_max_switch_raw']:.1f} frames (raw max)")
    print(f"d_max_exit:      {data['d_max_exit']:.1f} frames (p99)")
    print(f"Inflation ratio: {data['d_max_switch_raw'] / max(data['d_max_switch'], 1):.1f}x")
    print()
    r_death_mine = data.get("r_death_mine", R_DEATH_MINE)
    print(f"Training hyperparameters used in analysis:")
    print(f"  gamma = {PBRS_GAMMA}  alpha = {PBRS_ALPHA}  R_DEATH = {R_DEATH}")
    print(f"  R_DEATH_MINE = {r_death_mine}  R_SWITCH = {R_SWITCH}  R_EXIT = {R_EXIT}")
    print(f"  FiGAR mine interrupt threshold = {FIGAR_MINE_INTERRUPT_PX:.0f} px")
    print()

    mines = data["mines_x"]
    print(f"Mines detected: {len(mines)}")
    for i in range(len(mines)):
        state_label = "initially DEADLY (state=0)" if data["mine_initial_states"][i] == 0 else "initially SAFE (state=1)"
        print(f"  Mine {i+1}:  x={data['mines_x'][i]:.1f}  y={data['mines_y'][i]:.1f}  "
              f"type={data['mine_types'][i]}  {state_label}")
    print()

    overall_pass = True

    for r in mine_results:
        mi = r["mine_idx"] + 1
        _print_separator()
        print(f"MINE {mi}  @  ({r['mine_x']:.0f}, {r['mine_y']:.0f})")
        _print_separator()
        print()

        # --- A ---
        excl_str = "[PASS]" if r["excl_zone_ok"] else f"[FAIL] {r['n_inside_excl']} pts inside"
        kill_str = "[PASS]" if r["kill_zone_ok"] else f"[FAIL] {r['n_inside_kill']} pts inside"
        print(f"  A. Exclusion zone integrity")
        print(f"       Inside excl zone ({EXCL_RADIUS:.0f}px): {r['n_inside_excl']}  {excl_str}")
        print(f"       Inside kill zone ({KILL_RADIUS:.0f}px): {r['n_inside_kill']}  {kill_str}")
        print(f"       Boundary pts: {r['n_boundary']}  edge-flagged: {r['n_boundary_edge']}")
        if r["boundary_nodes_info"]:
            for bn in r["boundary_nodes_info"]:
                side = "left" if bn["x"] < r["mine_x"] else "right"
                print(f"         -> boundary node at x={bn['x']:.1f}  d*={bn['d']:.1f}  "
                      f"phi={bn['phi']:+.4f}  is_edge={bn['is_edge']}  [{side}]")
        print()
        if not r["excl_zone_ok"]:
            overall_pass = False

        # --- B ---
        print(f"  B. Potential landscape (d* at v=0 along floor near mine)")
        profile = r["profile"]
        if profile:
            print(f"       {'x':>8}  {'d_v0':>9}  {'phi_v0':>9}  {'phi_vL':>9}  {'phi_vR':>9}  {'edge':>5}  {'d_mine':>7}")
            for pt in profile:
                edge_mark = "*" if pt["is_edge"] else " "
                is_near_boundary = pt["is_edge"] and pt["d_to_mine"] < EXCL_RADIUS + 15
                row = (f"       {pt['x']:8.1f}  {pt['d_v0']:9.1f}  {pt['phi_v0']:+9.4f}  "
                       f"{pt['phi_vL']:+9.4f}  {pt['phi_vR']:+9.4f}  {edge_mark:>5}  "
                       f"{pt['d_to_mine']:7.1f}")
                if is_near_boundary:
                    row += "  <-- boundary"
                if r["gravity_well"] and abs(pt["x"] - r["gravity_well"]["x"]) < 4:
                    row += "  <-- GRAVITY WELL"
                print(row)
        print()

        # --- C ---
        gw = r["gravity_well"]
        extrema = r["extrema"]
        print(f"  C. Gravity-well analysis (approach from left)")
        if extrema["monotonic"]:
            print(f"       [PASS] d* is monotonically increasing on approach — no gravity well")
        else:
            print(f"       [WARN] Non-monotonic d* on approach side!")
            print(f"       Local minima: {len(extrema['local_minima'])}  maxima: {len(extrema['local_maxima'])}")
            for lm in extrema["local_minima"]:
                abs_x = r["mine_x"] + lm["x_rel"]
                dist = abs(lm["x_rel"])
                print(f"         Local min at x_rel={lm['x_rel']:+.1f} (x={abs_x:.1f})"
                      f"  d*={lm['value']:.1f}  phi={_phi(lm['value'], data['d_max_switch']):+.4f}"
                      f"  dist_from_mine={dist:.1f}px")
            for lm in extrema["local_maxima"]:
                abs_x = r["mine_x"] + lm["x_rel"]
                print(f"         Local max at x_rel={lm['x_rel']:+.1f} (x={abs_x:.1f})"
                      f"  d*={lm['value']:.1f}")
        if gw is not None:
            print(f"       Gravity well: x={gw['x']:.1f}  d*={gw['d_v0']:.1f}"
                  f"  phi={gw['phi_v0']:+.4f}  dist_from_mine={gw['dist_from_mine']:.1f}px")
            print(f"       KILL zone at {KILL_RADIUS:.0f}px -> gravity well is"
                  f" {gw['dist_from_mine'] - KILL_RADIUS:.1f}px outside kill zone"
                  f" ({gw['dist_from_mine']:.1f}px total)")
        if r["repulsive_strength"] is not None:
            rs = r["repulsive_strength"]
            direction = "repulsive (correct)" if rs < 0 else "ATTRACTIVE (bad!) gradient toward mine"
            print(f"       Gradient in last 60px before exclusion boundary: "
                  f"{rs:+.6f} phi/px  [{direction}]")
        print()

        # --- D ---
        print(f"  D. Reward trajectory comparison (gamma={PBRS_GAMMA}, alpha={PBRS_ALPHA})")
        # Check if the mine is reachable from the floor profile (Euclidean kill-zone check).
        # A ninja on the floor at y=pt_y can be killed when moving along x if the mine
        # is within kill_radius in 2D.  The min 2D distance is the vertical offset
        # |pt_y - mine_y| when directly above.  If that vertical gap alone exceeds
        # kill_radius, the mine can never kill the ninja on that floor surface.
        mine_reachable_from_floor = True
        if r["profile"]:
            y_gap = abs(r["profile"][0]["y"] - r["mine_y"])  # vertical gap (floor to mine)
            if y_gap >= KILL_RADIUS:
                mine_reachable_from_floor = False
                print(f"     [NOTE] Floor surface y={r['profile'][0]['y']:.0f} is {y_gap:.1f}px "
                      f"above mine centre y={r['mine_y']:.0f} (kill radius={KILL_RADIUS:.0f}px).")
                print(f"     The mine is NOT reachable from this floor surface — "
                      f"the mine sits below the floor. Trajectory analysis skipped.")
                print()

        if mine_reachable_from_floor and r["walk_traj"]:
            wt = r["walk_traj"]
            print(f"     Walk-into-mine trajectory ({len(wt)} steps):")
            for step in wt:
                event = f"  [{step['event']}]" if step["event"] else ""
                print(f"       x={step['x']:7.1f}  phi={step['phi']:+.4f}"
                      f"  F_t={step['F_t']:+.4f}  cumulative={step['cumulative']:+.5f}{event}")
            print(f"     Walk total reward: {r['walk_total_reward']:+.4f}")
        elif mine_reachable_from_floor:
            print(f"     Walk trajectory: no floor profile available near mine")

        if mine_reachable_from_floor:
            print()
            if r["jump_traj"] and r["jump_launch_pt"]:
                jt = r["jump_traj"]
                lp = r["jump_launch_pt"]
                print(f"     Jump-over-mine trajectory (launch at x={lp['x']:.1f}):")
                for step in jt:
                    event = f"  [{step['event']}]" if step["event"] else ""
                    print(f"       x={step['x']:7.1f}  phi={step['phi']:+.4f}"
                          f"  F_t={step['F_t']:+.4f}  cumulative={step['cumulative']:+.5f}{event}")
                print(f"     Jump total reward: {r['jump_total_reward']:+.4f}")
            else:
                print(f"     Jump trajectory: could not compute (no cross-gap arc found)")

        if mine_reachable_from_floor and r["walk_total_reward"] is not None and r["jump_total_reward"] is not None:
            diff = r["jump_total_reward"] - r["walk_total_reward"]
            verdict = "[PASS] Jump is preferred" if diff > 0 else "[FAIL] Walk is NOT penalised enough vs jump"
            print()
            print(f"     Jump advantage over walk: {diff:+.4f}  {verdict}")
            if r["walk_total_reward"] > 0:
                print(f"     [WARN] Walk-into-mine yields NET POSITIVE reward: {r['walk_total_reward']:+.4f}")
                print(f"     (PBRS terminal gain dominates R_DEATH; mine avoidance relies on exploration)")
        print()

        # --- E ---
        n_cross = len(r["cross_gap_edges"])
        n_lr = len(r["cross_gap_left_to_right"])
        n_rl = len(r["cross_gap_right_to_left"])
        print(f"  E. Cross-gap arc inventory")
        print(f"       Total cross-gap arcs: {n_cross}  (L→R: {n_lr}  R→L: {n_rl})")
        if r["cross_gap_weights"]:
            wts = r["cross_gap_weights"]
            finite_wts = [w for w in wts if math.isfinite(w)]
            if finite_wts:
                print(f"       Arc cost range: {min(finite_wts):.1f} – {max(finite_wts):.1f} frames")
                print(f"       Median: {float(np.median(finite_wts)):.1f}  p25: {float(np.percentile(finite_wts,25)):.1f}"
                      f"  p75: {float(np.percentile(finite_wts,75)):.1f}")
        if n_cross == 0:
            print(f"       [WARN] No cross-gap arcs found — agent cannot jump over this mine!")
        else:
            print(f"       [PASS] Cross-gap arcs exist")

        # FiGAR
        if r["gravity_well"]:
            dist_gw = r["gravity_well"]["dist_from_mine"]
            if r["figar_ok"]:
                figar_str = f"[PASS] FiGAR ({FIGAR_MINE_INTERRUPT_PX:.0f}px) interrupts BEFORE gravity well ({dist_gw:.1f}px)"
            else:
                figar_str = f"[WARN] Gravity well ({dist_gw:.1f}px) is BEYOND FiGAR interrupt ({FIGAR_MINE_INTERRUPT_PX:.0f}px)"
            print(f"       FiGAR analysis: {figar_str}")
        print()

    # --- Recommendations ---
    _print_separator("=")
    print("  RECOMMENDATIONS")
    _print_separator("=")
    print()

    any_gravity_well = any(r["gravity_well"] is not None for r in mine_results)
    any_positive_walk = any(
        r.get("walk_ends_in_death") and r["walk_total_reward"] is not None
        and r["walk_total_reward"] > 0
        for r in mine_results
    )
    any_no_cross_gap = any(len(r["cross_gap_edges"]) == 0 for r in mine_results)

    recs = []

    if any_gravity_well:
        gw_depths = [r["gravity_well"]["dist_from_mine"] for r in mine_results if r["gravity_well"]]
        min_dist = min(gw_depths)
        recs.append((
            "1. Recompute value function with higher mine_inflation_beta",
            f"   Current: beta=50  →  Recommended: beta=200 (or higher).\n"
            f"   The gravity well at {min_dist:.1f}px from the mine centre is caused by cross-gap\n"
            f"   arcs with cheaper costs than surface walk arcs. Increasing beta raises the\n"
            f"   cost of those arcs until the d* gradient is strictly repulsive all the way\n"
            f"   to the exclusion boundary.\n"
            f"   Command: python -m nclone.value_function.precompute_cli \\\n"
            f"       --map-path test-single-level/basics \\\n"
            f"       --output-dir value_functions \\\n"
            f"       --mine-inflation-beta 200",
        ))

    if any_positive_walk:
        walk_rewards = [
            r["walk_total_reward"] for r in mine_results
            if r.get("walk_ends_in_death") and r["walk_total_reward"] is not None
        ]
        max_walk = max(walk_rewards)
        recs.append((
            "2. Increase R_DEATH magnitude or reduce PBRS alpha for mine-terminal episodes",
            f"   Max walk-into-mine cumulative reward: {max_walk:+.4f}\n"
            f"   With gamma=1.0, PBRS terminal gain = alpha * |phi(spawn)| ≈ {PBRS_ALPHA*0.42:+.2f}\n"
            f"   which exceeds R_DEATH ({R_DEATH}) when the agent starts far from goals.\n"
            f"   Options (in priority order):\n"
            f"   a) Increase R_DEATH to -2.0 or -3.0 to ensure net-negative mine deaths.\n"
            f"      Risk: passive risk-aversion (run-221 showed 37% NOOP with R_DEATH=-1.5).\n"
            f"      Mitigate by also increasing HWM_ALPHA to maintain forward-progress SNR.\n"
            f"   b) Use gamma<1.0 (e.g. 0.995) so the PBRS terminal gain telescopes to\n"
            f"      (1-gamma)*sum(phi) instead of phi(start), shrinking the mine-death gain.\n"
            f"   c) Add a per-mine-death penalty (detected via death_cause=='mine' in info).",
        ))

    if any_no_cross_gap:
        recs.append((
            "3. Verify cross-gap arc enumeration for mines with zero arcs",
            f"   Some mines have no cross-gap edges. Run visualize_mine_arcs.py to check\n"
            f"   whether jump trajectories physically exist for those mines:\n"
            f"   python -m nclone.value_function.visualize_mine_arcs \\\n"
            f"       --data value_functions/basics_surface_pbrs.npz \\\n"
            f"       --map-path test-single-level/basics",
        ))

    for mines_r in mine_results:
        if not mines_r["figar_ok"] and mines_r["gravity_well"]:
            dist_gw = mines_r["gravity_well"]["dist_from_mine"]
            recs.append((
                "4. Increase FiGAR mine interrupt threshold",
                f"   Gravity well is at {dist_gw:.1f}px from mine centre.\n"
                f"   FiGAR currently interrupts at {FIGAR_MINE_INTERRUPT_PX:.0f}px, which is LESS than\n"
                f"   the gravity well distance — the agent reaches the well before FiGAR fires.\n"
                f"   Set figar_mine_interrupt_threshold to at least {dist_gw + 10:.0f}px in the\n"
                f"   training config (graph_free_stability_late_stage_figar.json).",
            ))
            break

    if not recs:
        print("  All checks passed — no corrective actions required.")
    else:
        for title, body in recs:
            print(f"  {title}")
            for line in body.split("\n"):
                print(f"  {line}")
            print()

    _print_separator("=")
    overall_str = "PASS" if overall_pass else "FAIL (see mine reports above)"
    print(f"  OVERALL: {overall_str}")
    _print_separator("=")
    print()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _draw_tiles(ax, tile_dic) -> None:
    if tile_dic is None:
        return
    try:
        from nclone.value_function.level_renderer import render_tiles_to_axis
        render_tiles_to_axis(ax, tile_dic, alpha=0.35)
    except Exception:
        pass


def _draw_mine_rings(ax, mx: float, my: float) -> None:
    for radius, color, ls, lw in [
        (KILL_RADIUS, "#e74c3c", "-", 1.5),
        (EXCL_RADIUS, "#f39c12", "--", 1.5),
        (FIGAR_MINE_INTERRUPT_PX, "#3498db", ":", 1.2),
    ]:
        ax.add_patch(plt.Circle(
            (mx, my), radius,
            color=color, fill=False, linewidth=lw, linestyle=ls, alpha=0.85, zorder=5,
        ))
    ax.plot(mx, my, "r+", markersize=8, linewidth=1.5, zorder=6)


def _build_figure(data: Dict, mine_results: List[Dict], tile_dic=None,
                  switch_pos=None, exit_pos=None) -> plt.Figure:
    """Build a multi-panel figure showing global and per-mine analysis."""
    n_mines = len(mine_results)
    # Layout: row 0 = global d* heatmap (full width)
    #         rows 1..n_mines: per-mine 3-column panel (profile, walk reward, jump reward comparison)
    n_rows = 1 + n_mines
    fig_h = 7 + 5 * n_mines
    fig = plt.figure(figsize=(22, fig_h))
    gs = fig.add_gridspec(n_rows, 3, hspace=0.45, wspace=0.32)

    # ------------------------------------------------------------------ Row 0: global overview
    ax_global = fig.add_subplot(gs[0, :])
    _draw_tiles(ax_global, tile_dic)
    ax_global.set_xlim(0, LEVEL_W)
    ax_global.set_ylim(LEVEL_H, 0)
    ax_global.set_aspect("equal")
    ax_global.set_title(
        "Full Level — d* (switch) heatmap + mine zones + FiGAR interrupt radius",
        fontsize=10, pad=4,
    )
    ax_global.set_xlabel("X (px)", fontsize=8)
    ax_global.set_ylabel("Y (px)", fontsize=8)

    sp = data["surface_points"]
    d_star = data["d_star_switch"]
    d_max = data["d_max_switch"]
    mb = data["graph"].max_vel_bins

    xs_sp = np.array([p.x for p in sp])
    ys_sp = np.array([p.y for p in sp])
    # Cost per surface point (minimum over velocity bins)
    pt_d = np.full(len(sp), np.inf)
    for nid in range(len(d_star)):
        pt_i, _ = node_id_to_parts(nid, mb)
        if pt_i < len(sp) and d_star[nid] < pt_d[pt_i]:
            pt_d[pt_i] = d_star[nid]

    finite_mask = np.isfinite(pt_d)
    if finite_mask.any():
        norm = mcolors.Normalize(
            vmin=float(np.percentile(pt_d[finite_mask], 2)),
            vmax=float(np.percentile(pt_d[finite_mask], 98)),
        )
        sc = ax_global.scatter(
            xs_sp[finite_mask], ys_sp[finite_mask],
            c=pt_d[finite_mask], cmap="viridis_r", norm=norm,
            s=6, alpha=0.75, zorder=3,
        )
        plt.colorbar(sc, ax=ax_global, label="d* (frames)", fraction=0.018, pad=0.01)

    for mx, my in zip(data["mines_x"], data["mines_y"]):
        _draw_mine_rings(ax_global, mx, my)

    if switch_pos:
        ax_global.plot(*switch_pos, "g^", markersize=9, zorder=7, label="switch")
    if exit_pos:
        ax_global.plot(*exit_pos, "b*", markersize=10, zorder=7, label="exit")

    legend_handles = [
        mpatches.Patch(facecolor="#e74c3c", alpha=0.5, label=f"Kill ({KILL_RADIUS:.0f}px)"),
        mpatches.Patch(facecolor="#f39c12", alpha=0.4, label=f"Excl ({EXCL_RADIUS:.0f}px)"),
        mpatches.Patch(facecolor="#3498db", alpha=0.3, label=f"FiGAR ({FIGAR_MINE_INTERRUPT_PX:.0f}px)"),
    ]
    ax_global.legend(handles=legend_handles, loc="upper right", fontsize=7, ncol=3)

    # ------------------------------------------------------------------ Rows 1+: per-mine
    for row_i, r in enumerate(mine_results):
        row = row_i + 1
        mine_x = r["mine_x"]
        mine_y = r["mine_y"]
        profile = r["profile"]

        # --- Col 0: d*/phi profile along floor ---
        ax_prof = fig.add_subplot(gs[row, 0])
        _draw_tiles(ax_prof, tile_dic)
        _draw_mine_rings(ax_prof, mine_x, mine_y)
        ZOOM_X, ZOOM_Y = 160.0, 70.0
        ax_prof.set_xlim(mine_x - ZOOM_X, mine_x + ZOOM_X)
        ax_prof.set_ylim(mine_y + ZOOM_Y, mine_y - ZOOM_Y)
        ax_prof.set_aspect("equal")
        ax_prof.set_title(
            f"Mine {r['mine_idx']+1} @ ({mine_x:.0f},{mine_y:.0f}) — d*/phi profile",
            fontsize=9, pad=3,
        )
        ax_prof.set_xlabel("X (px)", fontsize=7)
        ax_prof.set_ylabel("Y (px)", fontsize=7)
        ax_prof.tick_params(labelsize=6)

        if profile:
            prof_xs = np.array([p["x"] for p in profile])
            prof_costs = np.array([p["d_v0"] for p in profile])
            finite = np.isfinite(prof_costs)
            if finite.any():
                finite_costs = prof_costs[finite]
                norm_p = mcolors.Normalize(
                    vmin=float(np.percentile(finite_costs, 5)),
                    vmax=float(np.percentile(finite_costs, 95)),
                )
                prof_ys = np.array([p["y"] for p in profile])
                sc2 = ax_prof.scatter(
                    prof_xs[finite], prof_ys[finite],
                    c=finite_costs, cmap="RdYlGn_r", norm=norm_p,
                    s=20, zorder=4, alpha=0.95,
                )
                plt.colorbar(sc2, ax=ax_prof, label="d* v=0 (frames)", fraction=0.03, pad=0.01)

            # Mark gravity well
            if r["gravity_well"]:
                gw = r["gravity_well"]
                gw_pts = [p for p in profile if abs(p["x"] - gw["x"]) < 4]
                if gw_pts:
                    ax_prof.scatter(
                        [gw_pts[0]["x"]], [gw_pts[0]["y"]],
                        c="red", s=100, marker="v", zorder=8,
                        label=f"Gravity well\n(d*={gw['d_v0']:.0f})",
                    )

            # Mark jump launch point
            if r["jump_launch_pt"]:
                lp = r["jump_launch_pt"]
                ax_prof.scatter(
                    [lp["x"]], [lp["y"]], c="#2ecc71",
                    s=80, marker="^", zorder=8,
                    label=f"Jump launch\n(x={lp['x']:.0f})",
                )

            ax_prof.legend(loc="upper left", fontsize=6)

        # --- Col 1: per-frame F_t reward along walk trajectory ---
        ax_walk = fig.add_subplot(gs[row, 1])
        ax_walk.set_title(f"Mine {r['mine_idx']+1} — Per-frame F_t (walk trajectory)", fontsize=9, pad=3)
        ax_walk.set_xlabel("X position (px)", fontsize=7)
        ax_walk.set_ylabel("Shaping reward F_t", fontsize=7)
        ax_walk.tick_params(labelsize=6)
        ax_walk.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax_walk.axvline(mine_x - EXCL_RADIUS, color="#f39c12", linewidth=1.0,
                        linestyle="--", label="Excl boundary")
        ax_walk.axvline(mine_x - KILL_RADIUS, color="#e74c3c", linewidth=1.0,
                        linestyle="--", label="Kill boundary")
        ax_walk.axvline(mine_x - FIGAR_MINE_INTERRUPT_PX, color="#3498db", linewidth=1.0,
                        linestyle=":", label=f"FiGAR ({FIGAR_MINE_INTERRUPT_PX:.0f}px)")

        if r["walk_traj"]:
            wx = [s["x"] for s in r["walk_traj"]]
            wft = [s["F_t"] for s in r["walk_traj"]]
            wcum = [s["cumulative"] for s in r["walk_traj"]]
            colors = ["#e74c3c" if f < 0 else "#27ae60" for f in wft]
            ax_walk.bar(wx, wft, color=colors, width=4, alpha=0.7, label="F_t (per step)")
            ax_walk_twin = ax_walk.twinx()
            ax_walk_twin.plot(wx, wcum, color="#2c3e50", linewidth=1.5, label="Cumulative")
            ax_walk_twin.set_ylabel("Cumulative reward", fontsize=7)
            ax_walk_twin.tick_params(labelsize=6)
            # Shade region with positive cumulative reward
            ax_walk_twin.fill_between(wx, 0, wcum,
                                       where=[c > 0 for c in wcum],
                                       color="#e74c3c", alpha=0.12, label="+cumulative zone")

        ax_walk.legend(loc="upper right", fontsize=6)

        if r["gravity_well"]:
            ax_walk.axvline(r["gravity_well"]["x"], color="#8e44ad", linewidth=1.0,
                            linestyle="-.", label=f"Gravity well (x={r['gravity_well']['x']:.0f})")

        # --- Col 2: walk vs jump cumulative reward comparison ---
        ax_cmp = fig.add_subplot(gs[row, 2])
        ax_cmp.set_title(f"Mine {r['mine_idx']+1} — Walk vs Jump cumulative reward", fontsize=9, pad=3)
        ax_cmp.set_xlabel("X position (px)", fontsize=7)
        ax_cmp.set_ylabel("Cumulative reward", fontsize=7)
        ax_cmp.tick_params(labelsize=6)
        ax_cmp.axhline(0, color="#888888", linewidth=0.8, linestyle="--")

        if r["walk_traj"]:
            wx = [s["x"] for s in r["walk_traj"]]
            wcum = [s["cumulative"] for s in r["walk_traj"]]
            ax_cmp.plot(wx, wcum, color="#e74c3c", linewidth=2.0,
                        label=f"Walk→death ({r['walk_total_reward']:+.3f})")

        if r["jump_traj"]:
            jx = [s["x"] for s in r["jump_traj"]]
            jcum = [s["cumulative"] for s in r["jump_traj"]]
            ax_cmp.plot(jx, jcum, color="#27ae60", linewidth=2.0,
                        label=f"Walk→jump ({r['jump_total_reward']:+.3f})")

        ax_cmp.axvline(mine_x - EXCL_RADIUS, color="#f39c12", linewidth=0.8,
                       linestyle="--", alpha=0.7)
        ax_cmp.axvline(mine_x, color="#e74c3c", linewidth=0.8,
                       linestyle=":", alpha=0.7, label="Mine centre")

        # Gravity well marker
        if r["gravity_well"]:
            ax_cmp.axvline(r["gravity_well"]["x"], color="#8e44ad", linewidth=1.0,
                           linestyle="-.", alpha=0.8, label=f"Gravity well")

        ax_cmp.legend(loc="upper left", fontsize=6.5)

        # Stats text box
        n_cross = len(r["cross_gap_edges"])
        gw_str = f"GW @ x={r['gravity_well']['x']:.0f}" if r["gravity_well"] else "No GW"
        excl_str = "Excl OK" if r["excl_zone_ok"] else "EXCL FAIL"
        jump_adv = (
            f"Jump adv: {r['jump_total_reward'] - r['walk_total_reward']:+.3f}"
            if r["walk_total_reward"] is not None and r["jump_total_reward"] is not None
            else "Jump: N/A"
        )
        ax_cmp.text(
            0.02, 0.02,
            f"{excl_str} | {gw_str}\nCross-gap arcs: {n_cross}\n{jump_adv}",
            transform=ax_cmp.transAxes, fontsize=7, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88),
            family="monospace",
        )

    r_death_mine = data.get("r_death_mine", R_DEATH_MINE)
    fig.suptitle(
        "PBRS Mine Avoidance Validation — basics_surface_pbrs.npz\n"
        f"Kill zone: {KILL_RADIUS:.0f}px | Excl zone: {EXCL_RADIUS:.0f}px | "
        f"FiGAR: {FIGAR_MINE_INTERRUPT_PX:.0f}px | "
        f"gamma={PBRS_GAMMA} alpha={PBRS_ALPHA} R_DEATH={R_DEATH} R_DEATH_MINE={r_death_mine}",
        fontsize=12, fontweight="bold",
    )

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Holistic PBRS mine-avoidance validation for the basics level",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data", default="value_functions/basics_surface_pbrs.npz",
        help="Path to *_surface_pbrs.npz file (default: value_functions/basics_surface_pbrs.npz)",
    )
    parser.add_argument(
        "--map-path", default=None,
        help="Optional .map file for tile geometry overlay (default: None)",
    )
    parser.add_argument(
        "--output", default="validation_output",
        help="Output directory for report and PNG (default: validation_output/)",
    )
    parser.add_argument(
        "--r-death-mine", type=float, default=R_DEATH_MINE, dest="r_death_mine",
        help="Extra penalty for mine deaths (default: -2.0). Total = R_DEATH + r_death_mine.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: data file not found: {args.data}")
        return 1

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading: {args.data}")
    data = _load_npz(args.data)
    data["r_death_mine"] = args.r_death_mine

    map_info = {"tile_dic": None, "switch_pos": None, "exit_pos": None}
    if args.map_path:
        map_info = _load_map_info(args.map_path)

    mines_x = data["mines_x"]
    mines_y = data["mines_y"]
    mine_initial_states = data["mine_initial_states"]
    mine_boundary_nodes = data["mine_boundary_nodes"]

    if len(mines_x) == 0:
        print("No mines found in data file.")
        return 0

    print(f"Found {len(mines_x)} mine(s). Analysing...")
    print()

    mine_results = []
    for i in range(len(mines_x)):
        boundary_row = None
        if mine_boundary_nodes is not None and i < len(mine_boundary_nodes):
            boundary_row = mine_boundary_nodes[i]
        r = analyse_mine(
            mine_idx=i,
            mine_x=float(mines_x[i]),
            mine_y=float(mines_y[i]),
            mine_initial_state=int(mine_initial_states[i]) if len(mine_initial_states) > i else 0,
            mine_boundary_row=boundary_row,
            data=data,
        )
        mine_results.append(r)

    # Text report to stdout and file
    report_path = os.path.join(args.output, "pbrs_mine_validation_report.txt")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report(data, mine_results)
    report_text = buf.getvalue()
    print(report_text)

    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Report saved: {report_path}")

    # Visualisation
    png_path = os.path.join(args.output, "pbrs_mine_validation.png")
    print(f"Generating visualisation...")
    fig = _build_figure(
        data, mine_results,
        tile_dic=map_info["tile_dic"],
        switch_pos=map_info["switch_pos"],
        exit_pos=map_info["exit_pos"],
    )
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualisation saved: {png_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
