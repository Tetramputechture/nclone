#!/usr/bin/env python3
"""Command-line tool for precomputing surface-graph PBRS.

This script builds the exact cost-to-go potential for a level by:
  1. Extracting walkable surfaces from tile geometry.
  2. Building a velocity-augmented surface graph (~45K nodes).
  3. Enumerating edges via physics simulation (~5M sims, parallelisable).
  4. Running backward Dijkstra to switch and exit goals (<1 s each).
  5. Building an arc cache for O(1) airborne lookups.
  6. Saving everything to a single .npz file.

Usage:
    # Basic (8 workers)
    python -m nclone.value_function.precompute_cli \\
        --map-path levels/rampjump.map --num-workers 8

    # Verbose, single-process (for debugging)
    python -m nclone.value_function.precompute_cli \\
        --map-path levels/rampjump.map --num-workers 1 -v
"""

import argparse
import logging
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from nclone.value_function.surface_extraction import (
    extract_surfaces,
    compute_mine_boundary_nodes,
)
from nclone.value_function.surface_graph import (
    SurfaceGraph,
    build_graph,
    compute_cost_to_go,
)
from nclone.value_function.edge_enumeration import (
    enumerate_edges,
    populate_graph_edges,
)
from nclone.value_function.arc_cache import ArcCache
from nclone.value_function.demo_ingestion import (
    find_replay_files,
    ingest_demos_into_caches,
)
from nclone.value_function.physics_step_wrapper import PhysicsStepWrapper
from nclone.value_function.surface_extraction import MINE_SURFACE_EXCLUSION_RADIUS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def _banner(text: str, char: str = "=", width: int = 70) -> None:
    print(f"\n{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}\n")


def _section(text: str, char: str = "-", width: int = 70) -> None:
    print(f"\n{text}")
    print(char * width)


def _safe_max(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)]
    return float(np.max(finite)) if len(finite) > 0 else 1.0


def _robust_d_max(arr: np.ndarray, percentile: float = 99.0) -> float:
    """99th-percentile of finite d* values, for mine-inflation-resistant normalization.

    Mine-proximity penalties can inflate a small number of graph nodes to
    costs in the millions of frames.  Using the raw max as the normalization
    constant collapses all normal phi values to [-0.0001, 0], destroying the
    PBRS gradient.  The 99th percentile removes those outliers while correctly
    assigning phi = -1 (clamped) to paths that are truly very costly.
    """
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return 1.0
    return float(np.percentile(finite, percentile))


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def _save_surface_pbrs(
    output_path: str,
    graph: SurfaceGraph,
    d_star_switch: np.ndarray,
    d_star_exit: np.ndarray,
    arc_cache_switch: ArcCache,
    arc_cache_exit: ArcCache,
    mine_xs: np.ndarray,
    mine_ys: np.ndarray,
    mine_types: np.ndarray,
    mine_initial_states: np.ndarray,
    mine_boundary_nodes: np.ndarray,
) -> None:
    """Save all precomputed data to a single .npz file."""
    sp = graph.surface_points
    n = len(sp)

    ac_keys_sw, ac_vals_sw = arc_cache_switch.to_arrays()
    ac_keys_ex, ac_vals_ex = arc_cache_exit.to_arrays()

    d_max_switch_raw = _safe_max(d_star_switch)
    d_max_exit_raw = _safe_max(d_star_exit)
    d_max_switch = _robust_d_max(d_star_switch)
    d_max_exit = _robust_d_max(d_star_exit)

    metadata = {
        "n_surface_points": n,
        "n_graph_nodes": graph.n_nodes,
        "n_edges": len(graph.edges_src),
        "max_vel_bins": graph.max_vel_bins,
        # Robust (99th-percentile) normalization constant used at runtime.
        "d_max_switch": d_max_switch,
        "d_max_exit": d_max_exit,
        # Raw maximum kept for audit; typically >>d_max due to mine inflation.
        "d_max_switch_raw": d_max_switch_raw,
        "d_max_exit_raw": d_max_exit_raw,
    }

    np.savez_compressed(
        output_path,
        # Surface points
        surface_points_x=np.array([p.x for p in sp], dtype=np.float32),
        surface_points_y=np.array([p.y for p in sp], dtype=np.float32),
        surface_normals_x=np.array([p.normal_x for p in sp], dtype=np.float32),
        surface_normals_y=np.array([p.normal_y for p in sp], dtype=np.float32),
        surface_types=np.array([p.surface_type for p in sp], dtype=np.uint8),
        surface_segment_ids=np.array([p.segment_id for p in sp], dtype=np.int32),
        surface_is_edge=np.array([p.is_edge for p in sp], dtype=np.bool_),
        # Velocity bins
        velocity_bins_floor=graph.velocity_bins_floor,
        velocity_bins_wall=graph.velocity_bins_wall,
        # Edges
        edges_src=graph.edges_src,
        edges_dst=graph.edges_dst,
        edges_weight=graph.edges_weight,
        # Dijkstra distance arrays
        d_star_switch=d_star_switch.astype(np.float32),
        d_star_exit=d_star_exit.astype(np.float32),
        # Arc caches
        arc_cache_keys_switch=ac_keys_sw,
        arc_cache_values_switch=ac_vals_sw,
        arc_cache_keys_exit=ac_keys_ex,
        arc_cache_values_exit=ac_vals_ex,
        # Toggle mine data for runtime masking (no blocking at precompute time)
        mine_positions_x=mine_xs,
        mine_positions_y=mine_ys,
        mine_types=mine_types,
        mine_initial_states=mine_initial_states,
        # Per-mine boundary node pairs for exclusion-zone-aware PBRS lookup.
        # Shape (num_always_deadly_mines, 2): [mine_idx, 0] is the closest
        # is_edge boundary node on the negative-axis side (left/top),
        # [mine_idx, 1] is the positive-axis side (right/bottom).  -1 = none.
        mine_boundary_nodes=mine_boundary_nodes,
        # Metadata
        metadata=np.array(metadata),
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def precompute_surface_pbrs(
    map_path: str,
    output_dir: str = "value_functions",
    num_workers: int = 4,
    verbose: bool = False,
    demo_dir: str | None = None,
    mine_inflation_beta: float = 50.0,
) -> str:
    """Run the full surface-graph PBRS precomputation pipeline.

    Args:
        map_path: Path to ``.map`` file.
        output_dir: Output directory.
        num_workers: Worker processes for edge enumeration.
        verbose: Enable progress logging.
        demo_dir: Optional directory containing ``.replay`` expert demo files
            for this level.  If provided, airborne states from the demos are
            ingested into the arc caches after the edge-enumeration build,
            extending coverage to trajectories not captured by graph arcs.
        mine_inflation_beta: Mine proximity inflation strength β (default 50.0).
            Higher values produce stronger avoidance of mine danger zones.
            Recommended sweep: 10 (mild), 50 (strong), 200 (near-hard).

    Returns:
        Path to the saved ``.npz`` file.
    """
    start_time = time.time()
    level_name = os.path.splitext(os.path.basename(map_path))[0]

    _banner("Surface-Graph PBRS Precomputation")
    print(f"Level:   {level_name}")
    print(f"Workers: {num_workers}")
    output_path = os.path.join(output_dir, f"{level_name}_surface_pbrs.npz")
    print(f"Output:  {output_path}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Load level
    # ------------------------------------------------------------------
    _section("Step 1: Loading Level")
    with open(map_path, "rb") as f:
        raw = f.read()
    map_data = [int(b) for b in raw]
    physics = PhysicsStepWrapper(map_path=map_path)
    tile_dict = physics.tile_dic
    print(f"  Level loaded ({len(raw) / 1024:.1f} KB)")
    print(f"  Tiles: {len(tile_dict)} cells")

    # Goal positions
    switch_pos = physics.get_goal_position("switch")
    exit_pos = physics.get_goal_position("exit")
    print(f"  Switch: ({switch_pos[0]:.1f}, {switch_pos[1]:.1f})")
    print(f"  Exit:   ({exit_pos[0]:.1f}, {exit_pos[1]:.1f})")

    # Collect toggle mine positions for storage in the .npz file.
    # These are NOT used to block Dijkstra — mine hazard masking is applied at
    # runtime so that the actual mine state (which can only change via ninja
    # contact, never via switch) is respected correctly.
    #
    # initial_state is read directly from e.state (the simulator's loaded state
    # at t=0, before any simulation):
    #   0 = toggled / deadly
    #   1 = untoggled / safe
    # Entity type 21 maps to state=0 by default; entity type 1 maps to state=1
    # by default — but the level data can override this, so we always use e.state.
    mine_entries = []
    for e in physics.entity_dic.get(21, []):
        mine_entries.append((e.xpos, e.ypos, 21, int(e.state)))
    for e in physics.entity_dic.get(1, []):
        mine_entries.append((e.xpos, e.ypos, 1, int(e.state)))

    if mine_entries:
        mine_xs = np.array([m[0] for m in mine_entries], dtype=np.float32)
        mine_ys = np.array([m[1] for m in mine_entries], dtype=np.float32)
        mine_types_arr = np.array([m[2] for m in mine_entries], dtype=np.uint8)
        mine_initial_states_arr = np.array([m[3] for m in mine_entries], dtype=np.uint8)
    else:
        mine_xs = np.empty(0, dtype=np.float32)
        mine_ys = np.empty(0, dtype=np.float32)
        mine_types_arr = np.empty(0, dtype=np.uint8)
        mine_initial_states_arr = np.empty(0, dtype=np.uint8)

    n_deadly = int(np.sum(mine_initial_states_arr == 0)) if len(mine_initial_states_arr) else 0
    n_safe = int(np.sum(mine_initial_states_arr == 1)) if len(mine_initial_states_arr) else 0
    print(f"  Toggle mines (initially deadly, state=0): {n_deadly}")
    print(f"  Toggle mines (initially safe,   state=1): {n_safe}")

    # Build mine position arrays for surface splitting and arc clearance.
    # Both mine states are included — even "safe" mines create virtual gaps
    # because touching them toggles their state and endangers the return path.
    # mine_positions_list: passed to extract_surfaces() for segment splitting
    # mine_positions:      numpy array passed to enumerate_edges() for arc rejection
    mine_positions_list = [(float(m[0]), float(m[1])) for m in mine_entries]
    if len(mine_xs) > 0:
        mine_positions = np.column_stack([mine_xs, mine_ys]).astype(np.float32)
    else:
        mine_positions = np.empty((0, 2), dtype=np.float32)

    if len(mine_positions_list) > 0:
        from nclone.value_function.edge_enumeration import _MINE_KILL_RADIUS
        print(f"  Mine handling: continuous cost inflation "
              f"(beta={mine_inflation_beta:.1f}, kill_radius={_MINE_KILL_RADIUS:.1f} px, "
              f"surface exclusion radius={MINE_SURFACE_EXCLUSION_RADIUS:.1f} px)")
    else:
        print("  Mine handling: no mines in level")
    print(f"  Elapsed: {_format_time(time.time() - start_time)}")

    # ------------------------------------------------------------------
    # Step 2: Extract surfaces
    # ------------------------------------------------------------------
    _section("Step 2: Extracting Surfaces")
    t0 = time.time()
    surface_points = extract_surfaces(tile_dict, mine_positions=mine_positions_list)
    n_floor = sum(1 for p in surface_points if p.surface_type == 0)
    n_wall = sum(1 for p in surface_points if p.surface_type == 1)
    n_edge = sum(1 for p in surface_points if p.is_edge)
    print(f"  Surface points: {len(surface_points):,}")
    print(f"    Floor: {n_floor:,}  Wall: {n_wall:,}  Edge-flagged: {n_edge:,}")
    print(f"  Extraction time: {_format_time(time.time() - t0)}")

    # Compute boundary node pairs for always-deadly mines only.  These pairs
    # are the is_edge=True nodes on each side of each mine exclusion zone, used
    # at runtime to pin the PBRS nearest-node lookup so the agent is never
    # rewarded for running into a mine gap.
    always_deadly_positions = [
        (float(mine_xs[i]), float(mine_ys[i]))
        for i in range(len(mine_xs))
        if mine_initial_states_arr[i] == 0
    ]
    mine_boundary_nodes = compute_mine_boundary_nodes(
        surface_points, always_deadly_positions
    )
    n_pinned = int(np.sum(mine_boundary_nodes >= 0))
    print(f"  Mine boundary nodes: {mine_boundary_nodes.shape[0]} mines, "
          f"{n_pinned} boundary node slots populated")

    # ------------------------------------------------------------------
    # Step 3: Build graph nodes
    # ------------------------------------------------------------------
    _section("Step 3: Building Graph")
    graph = build_graph(surface_points)
    print(f"  Nodes: {graph.n_nodes:,}")
    print(f"  Max velocity bins: {graph.max_vel_bins}")
    print(f"  Spatial index cells: {len(graph.spatial_index):,}")

    # ------------------------------------------------------------------
    # Step 4: Enumerate edges
    # ------------------------------------------------------------------
    _section("Step 4: Enumerating Edges (this may take several minutes)")
    t0 = time.time()
    edges, waypoints = enumerate_edges(
        graph,
        map_data=map_data,
        num_workers=num_workers,
        collect_waypoints=True,
        verbose=verbose,
        mine_positions=mine_positions,
        mine_inflation_beta=mine_inflation_beta,
    )
    populate_graph_edges(graph, edges)
    print(f"  Edges discovered: {len(edges):,}")
    print(f"  Arc waypoints: {len(waypoints):,}")
    print(f"  Enumeration time: {_format_time(time.time() - t0)}")

    # ------------------------------------------------------------------
    # Step 5: Dijkstra to switch and exit
    # ------------------------------------------------------------------
    _section("Step 5: Computing Cost-to-Go (Dijkstra)")
    t0 = time.time()

    # Run Dijkstra — mine hazard is encoded structurally (virtual gap: no
    # surface nodes in mine zones, jump arcs through mines discarded).
    # Runtime mine masking in SurfaceGraphPBRS.get_potential() provides an
    # additional safety net for the agent's live position.
    d_star_switch = compute_cost_to_go(graph, switch_pos[0], switch_pos[1], "switch")
    d_max_sw_raw = _safe_max(d_star_switch)
    d_max_sw = _robust_d_max(d_star_switch)
    n_reachable_sw = int(np.sum(np.isfinite(d_star_switch)))
    print(f"  Switch: d_max(p99)={d_max_sw:.1f} frames, "
          f"d_max(raw)={d_max_sw_raw:.1f} frames, "
          f"inflation={d_max_sw_raw / max(d_max_sw, 1):.1f}x, "
          f"reachable={n_reachable_sw:,}/{graph.n_nodes:,}")

    d_star_exit = compute_cost_to_go(graph, exit_pos[0], exit_pos[1], "exit")
    d_max_ex_raw = _safe_max(d_star_exit)
    d_max_ex = _robust_d_max(d_star_exit)
    n_reachable_ex = int(np.sum(np.isfinite(d_star_exit)))
    print(f"  Exit:   d_max(p99)={d_max_ex:.1f} frames, "
          f"d_max(raw)={d_max_ex_raw:.1f} frames, "
          f"inflation={d_max_ex_raw / max(d_max_ex, 1):.1f}x, "
          f"reachable={n_reachable_ex:,}/{graph.n_nodes:,}")
    print(f"  Dijkstra time: {_format_time(time.time() - t0)}")

    # ------------------------------------------------------------------
    # Step 6: Build arc caches
    # ------------------------------------------------------------------
    _section("Step 6: Building Arc Caches")
    t0 = time.time()

    arc_cache_switch = ArcCache()
    arc_cache_switch.build(waypoints, d_star_switch)

    arc_cache_exit = ArcCache()
    arc_cache_exit.build(waypoints, d_star_exit)

    print(f"  Switch cache: {arc_cache_switch.n_entries:,} cells")
    print(f"  Exit cache:   {arc_cache_exit.n_entries:,} cells")
    print(f"  Build time: {_format_time(time.time() - t0)}")

    # ------------------------------------------------------------------
    # Step 6b: Ingest expert demonstrations (optional)
    # ------------------------------------------------------------------
    if demo_dir:
        _section("Step 6b: Ingesting Expert Demonstrations")
        t0 = time.time()
        replay_paths = find_replay_files(demo_dir)
        print(f"  Demo directory: {demo_dir}")
        print(f"  Replay files found: {len(replay_paths)}")
        if replay_paths:
            sw_before = arc_cache_switch.n_entries
            ex_before = arc_cache_exit.n_entries
            n_sw, n_ex = ingest_demos_into_caches(
                replay_paths,
                map_data,
                arc_cache_switch,
                arc_cache_exit,
                d_star_switch,
                d_star_exit,
                graph,
            )
            print(
                f"  Switch cache: {arc_cache_switch.n_entries:,} cells "
                f"(+{arc_cache_switch.n_entries - sw_before:,} from demos)"
            )
            print(
                f"  Exit cache:   {arc_cache_exit.n_entries:,} cells "
                f"(+{arc_cache_exit.n_entries - ex_before:,} from demos)"
            )
        else:
            print("  No .replay files found — skipping demo ingestion")
        print(f"  Demo ingestion time: {_format_time(time.time() - t0)}")

    # ------------------------------------------------------------------
    # Step 7: Save
    # ------------------------------------------------------------------
    _section("Step 7: Saving")
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    _save_surface_pbrs(
        output_path, graph,
        d_star_switch, d_star_exit,
        arc_cache_switch, arc_cache_exit,
        mine_xs=mine_xs,
        mine_ys=mine_ys,
        mine_types=mine_types_arr,
        mine_initial_states=mine_initial_states_arr,
        mine_boundary_nodes=mine_boundary_nodes,
    )
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved to: {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Save time: {_format_time(time.time() - t0)}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_time = time.time() - start_time
    _banner("Precomputation Complete")
    print(f"  Level:          {level_name}")
    print(f"  Surface points: {len(surface_points):,}")
    print(f"  Graph nodes:    {graph.n_nodes:,}")
    print(f"  Edges:          {len(edges):,}")
    print(f"  File size:      {file_size_mb:.1f} MB")
    print(f"  Total time:     {_format_time(total_time)}")
    print()

    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Precompute surface-graph PBRS for a level",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--map-path", required=True,
        help="Path to .map file",
    )
    parser.add_argument(
        "--output-dir", default="value_functions",
        help="Output directory (default: value_functions/)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Worker processes for edge enumeration (default: 4)",
    )
    parser.add_argument(
        "--demo-dir", default=None,
        help=(
            "Optional directory containing .replay expert demo files for this "
            "level.  Airborne states from the demos are ingested into the arc "
            "caches to extend coverage beyond graph-optimal arcs."
        ),
    )
    parser.add_argument(
        "--mine-inflation-beta", type=float, default=50.0,
        help=(
            "Mine proximity inflation strength β (default: 50.0). "
            "Higher values create stronger avoidance of mine danger zones. "
            "Recommended sweep: 10 (mild), 50 (strong), 200 (near-hard). "
            "Set to 0 to disable inflation (falls back to kill-zone-only pruning)."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        precompute_surface_pbrs(
            map_path=args.map_path,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            verbose=args.verbose,
            demo_dir=args.demo_dir,
            mine_inflation_beta=args.mine_inflation_beta,
        )
    except Exception as e:
        logger.error(f"Precomputation failed: {e}", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
