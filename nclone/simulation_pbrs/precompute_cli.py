#!/usr/bin/env python3
"""Command-line tool for precomputing simulation-based phi grids.

Precomputes phi grids for one or more map files using the physics simulation.
Results are cached as .npz files for O(1) runtime loading.

Usage:
    # Single map
    python -m nclone.simulation_pbrs.precompute_cli \
        --map-path maps/training/level_001.map \
        --output-dir simulation_phi_cache/

    # All maps in a directory
    python -m nclone.simulation_pbrs.precompute_cli \
        --map-dir maps/training/ \
        --output-dir simulation_phi_cache/

    # With custom frame limit
    python -m nclone.simulation_pbrs.precompute_cli \
        --map-path maps/training/level_001.map \
        --output-dir simulation_phi_cache/ \
        --max-frames 50
"""

import argparse
import logging
import os
import sys
import time
import glob

from .phi_grid_builder import SimulationPhiGridBuilder

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _create_sim_and_goals(map_path: str, use_native: bool = True):
    """Create simulator and extract switch/exit positions.

    When use_native=True, tries nclone_native first; falls back to Python if
    not installed. When use_native=False, uses NPlayHeadless.

    Args:
        map_path: Path to .map file.
        use_native: If True, prefer Rust simulator.

    Returns:
        Tuple of (sim, switch_pos, exit_pos) where sim has .ninja, .tile_dic,
        .tick_physics_only() for SimulationPhiGridBuilder.
    """
    if use_native:
        try:
            from .native_sim_adapter import create_native_sim_from_map_path

            sim, switch_pos, exit_pos = create_native_sim_from_map_path(map_path)
            logger.debug(f"Using native Rust simulator for {map_path}")
            return sim, switch_pos, exit_pos
        except ImportError as e:
            logger.warning(
                f"nclone_native not available ({e}), falling back to Python simulator"
            )

    # Python fallback
    from ..nplay_headless import NPlayHeadless

    nplay = NPlayHeadless(enable_rendering=False)
    nplay.load_map(map_path)
    switch_pos = nplay.exit_switch_position()
    exit_pos = nplay.exit_door_position()

    if switch_pos == (0, 0):
        logger.warning(f"Switch position is (0,0) for {map_path} - entities may not be loaded")
    if exit_pos == (0, 0):
        logger.warning(f"Exit position is (0,0) for {map_path} - entities may not be loaded")

    return nplay.sim, switch_pos, exit_pos


def precompute_single_map(
    map_path: str,
    output_dir: str,
    max_frames: int = 50,
    force_recompute: bool = False,
    use_native: bool = True,
) -> bool:
    """Precompute phi grids for a single map.

    Args:
        map_path: Path to .map file.
        output_dir: Directory to store cached results.
        max_frames: Maximum frames per transition.
        force_recompute: If True, recompute even if cached.

    Returns:
        True if successful, False otherwise.
    """
    map_name = os.path.splitext(os.path.basename(map_path))[0]
    logger.info(f"Processing: {map_name} ({map_path})")

    try:
        sim, switch_pos, exit_pos = _create_sim_and_goals(map_path, use_native=use_native)

        builder = SimulationPhiGridBuilder(
            cache_dir=output_dir,
            max_frames=max_frames,
        )

        start_time = time.perf_counter()
        phi_switch, phi_exit = builder.build(
            sim,
            switch_pos,
            exit_pos,
            force_recompute=force_recompute,
        )
        elapsed = time.perf_counter() - start_time

        logger.info(
            f"  Completed {map_name} in {elapsed:.1f}s "
            f"(phi_switch range: [{phi_switch.min():.3f}, {phi_switch.max():.3f}], "
            f"phi_exit range: [{phi_exit.min():.3f}, {phi_exit.max():.3f}])"
        )
        return True

    except Exception as e:
        logger.error(f"  Failed {map_name}: {e}", exc_info=True)
        return False


def main():
    """Main entry point for precomputation CLI."""
    parser = argparse.ArgumentParser(
        description="Precompute simulation-based PBRS phi grids"
    )
    parser.add_argument(
        "--map-path",
        type=str,
        help="Path to a single .map file",
    )
    parser.add_argument(
        "--map-dir",
        type=str,
        help="Path to directory containing .map files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="simulation_phi_cache",
        help="Output directory for cached phi grids (default: simulation_phi_cache/)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="Maximum simulation frames per transition (default: 50)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if cached",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        default=True,
        help="Use native Rust physics simulator (default: True)",
    )
    parser.add_argument(
        "--no-native",
        action="store_true",
        help="Use Python simulator instead of native",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    use_native = args.native and not args.no_native
    _setup_logging(args.verbose)

    # Collect map files
    map_files = []
    if args.map_path:
        map_files.append(args.map_path)
    if args.map_dir:
        patterns = [
            os.path.join(args.map_dir, "*.map"),
            os.path.join(args.map_dir, "**", "*.map"),
        ]
        for pattern in patterns:
            map_files.extend(glob.glob(pattern, recursive=True))

    if not map_files:
        logger.error("No map files specified. Use --map-path or --map-dir.")
        sys.exit(1)

    # Deduplicate and sort
    map_files = sorted(set(os.path.abspath(f) for f in map_files))
    logger.info(f"Processing {len(map_files)} map file(s)")

    # Process each map
    total_start = time.perf_counter()
    success_count = 0
    fail_count = 0

    for i, map_path in enumerate(map_files, 1):
        logger.info(f"\n=== Map {i}/{len(map_files)} ===")
        if precompute_single_map(
            map_path, args.output_dir, args.max_frames, args.force, use_native
        ):
            success_count += 1
        else:
            fail_count += 1

    total_elapsed = time.perf_counter() - total_start
    logger.info(
        f"\nDone: {success_count} succeeded, {fail_count} failed, "
        f"total time: {total_elapsed:.1f}s"
    )

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
