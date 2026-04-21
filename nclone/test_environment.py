"""
N++ Environment Test with Graph Visualization

This script provides a comprehensive testing environment for the N++ simulator
with integrated graph visualization capabilities.

Modes (command-line):
  --step-on-input - Simulation only ticks when input pressed (frame-perfect testing)

Interactive Controls (during runtime):
-------------------------------------

E - Toggle exploration debug overlay
C - Toggle grid debug overlay
I - Toggle tile type overlay
L - Toggle tile rendering
M - Toggle mine predictor debug overlay
D - Toggle mine death probability debug overlay
W - Toggle action mask debug overlay
U - Toggle reachable walls debug overlay
Q - Toggle demo checkpoint heatmap (requires --demo-replays)
F - Toggle value function (PBRS cost-to-go) overlay
Y - Toggle analytical reachability overlay (floor + wall jump hulls)
P - Toggle PBRS path overlay (shortest path to goal + cost-to-go heatmap)
O - Toggle route PBRS overlay (surface heatmap + all arcs + optimal route + arc features)

Examples:
--------
python test_environment.py --headless --export-frame test_level.png
python test_environment.py --headless --profile-frames 1000  # Benchmark with profiling
python test_environment.py --benchmark-dataset --test-dataset-path datasets/mechanic_benchmark
"""

import os
import sys

import pygame
from nclone.gym_environment.environment_factory import create_visual_testing_env
import argparse
import time
import cProfile
import pstats
import sys
import numpy as np
from PIL import Image
import tracemalloc
import gc
import psutil
import os

# Cache management is now handled internally by env.reset()

from nclone.replay.gameplay_recorder import GameplayRecorder
from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.evaluation.mechanic_benchmark_loader import MechanicBenchmarkLoader


class MemoryProfiler:
    """Memory profiling utility for environment memory analysis."""

    def __init__(self, snapshot_interval=100):
        self.snapshot_interval = snapshot_interval
        self.process = psutil.Process(os.getpid())
        self.snapshots = []
        self.baseline_snapshot = None
        self.frame_count = 0

    def start(self):
        """Start memory profiling."""
        tracemalloc.start()
        gc.collect()
        self.baseline_snapshot = tracemalloc.take_snapshot()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        print("\n" + "=" * 60)
        print("MEMORY PROFILING ACTIVE")
        print("=" * 60)
        print(f"Baseline memory usage: {self.baseline_memory:.2f} MB")
        print(f"Snapshot interval: {self.snapshot_interval} frames")
        print("=" * 60 + "\n")

    def take_snapshot(self, frame_number, obs=None):
        """Take a memory snapshot and analyze."""
        gc.collect()  # Force garbage collection for accurate measurement
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        snapshot = tracemalloc.take_snapshot()

        # Calculate observation sizes if provided
        obs_sizes = {}
        if obs is not None:
            try:
                if hasattr(obs, "items"):
                    for key, value in obs.items():
                        if isinstance(value, np.ndarray):
                            obs_sizes[key] = {
                                "shape": value.shape,
                                "dtype": str(value.dtype),
                                "size_mb": value.nbytes / 1024 / 1024,
                            }
            except:
                pass

        snapshot_data = {
            "frame": frame_number,
            "memory_mb": current_memory,
            "memory_delta_mb": current_memory - self.baseline_memory,
            "snapshot": snapshot,
            "observation_sizes": obs_sizes,
        }

        self.snapshots.append(snapshot_data)

        # Print snapshot info
        print(f"\n[Frame {frame_number}] Memory Snapshot:")
        print(f"  Total Memory: {current_memory:.2f} MB")
        print(f"  Delta from baseline: {snapshot_data['memory_delta_mb']:+.2f} MB")

        if obs_sizes:
            print("  Observation sizes:")
            total_obs_size = 0
            for key, info in obs_sizes.items():
                print(
                    f"    {key}: {info['shape']} ({info['dtype']}) = {info['size_mb']:.3f} MB"
                )
                total_obs_size += info["size_mb"]
            print(f"  Total observation size: {total_obs_size:.3f} MB")

        # Show top memory allocations
        top_stats = snapshot.statistics("lineno")[:5]
        print("  Top 5 memory allocations:")
        for stat in top_stats:
            print(f"    {stat}")

        return snapshot_data

    def compare_snapshots(self, snapshot_idx1=0, snapshot_idx2=-1):
        """Compare two snapshots to identify memory growth."""
        if len(self.snapshots) < 2:
            print("Not enough snapshots for comparison")
            return

        snap1 = self.snapshots[snapshot_idx1]
        snap2 = self.snapshots[snapshot_idx2]

        print("\n" + "=" * 60)
        print(f"MEMORY COMPARISON: Frame {snap1['frame']} vs Frame {snap2['frame']}")
        print("=" * 60)
        print(f"Memory growth: {snap2['memory_mb'] - snap1['memory_mb']:.2f} MB")

        # Compare tracemalloc snapshots
        top_stats = snap2["snapshot"].compare_to(snap1["snapshot"], "lineno")[:10]
        print("\nTop 10 memory growth sources:")
        for stat in top_stats:
            print(f"  {stat}")
        print("=" * 60)

    def finalize(self):
        """Generate final memory report."""
        print("\n" + "=" * 60)
        print("MEMORY PROFILING REPORT")
        print("=" * 60)

        if self.snapshots:
            # Memory growth analysis
            first_snap = self.snapshots[0]
            last_snap = self.snapshots[-1]
            total_growth = last_snap["memory_mb"] - first_snap["memory_mb"]
            frames_elapsed = last_snap["frame"] - first_snap["frame"]

            print(f"Total frames profiled: {frames_elapsed}")
            print(f"Starting memory: {first_snap['memory_mb']:.2f} MB")
            print(f"Ending memory: {last_snap['memory_mb']:.2f} MB")
            print(f"Total growth: {total_growth:.2f} MB")
            if frames_elapsed > 0:
                print(f"Growth per frame: {total_growth / frames_elapsed:.4f} MB")

            # Peak memory
            peak_snap = max(self.snapshots, key=lambda s: s["memory_mb"])
            print(
                f"\nPeak memory: {peak_snap['memory_mb']:.2f} MB at frame {peak_snap['frame']}"
            )

            # Compare first and last snapshot
            if len(self.snapshots) >= 2:
                print("\nMemory leak analysis (comparing first and last snapshots):")
                top_stats = last_snap["snapshot"].compare_to(
                    first_snap["snapshot"], "lineno"
                )[:15]
                for i, stat in enumerate(top_stats, 1):
                    print(f"  {i}. {stat}")

        print("=" * 60)

        # Save detailed report
        with open("memory_profiling_report.txt", "w") as f:
            f.write("DETAILED MEMORY PROFILING REPORT\n")
            f.write("=" * 80 + "\n\n")

            if self.snapshots:
                f.write(f"Total snapshots: {len(self.snapshots)}\n")
                f.write(f"Baseline memory: {self.baseline_memory:.2f} MB\n\n")

                for snap_data in self.snapshots:
                    f.write(f"\nFrame {snap_data['frame']}:\n")
                    f.write(f"  Memory: {snap_data['memory_mb']:.2f} MB\n")
                    f.write(f"  Delta: {snap_data['memory_delta_mb']:+.2f} MB\n")

                    if snap_data["observation_sizes"]:
                        f.write("  Observations:\n")
                        for key, info in snap_data["observation_sizes"].items():
                            f.write(
                                f"    {key}: {info['shape']} ({info['dtype']}) = {info['size_mb']:.3f} MB\n"
                            )

                    f.write("  Top allocations:\n")
                    top_stats = snap_data["snapshot"].statistics("lineno")[:10]
                    for stat in top_stats:
                        f.write(f"    {stat}\n")

            # Memory leak analysis
            if len(self.snapshots) >= 2:
                f.write("\n\nMEMORY LEAK ANALYSIS\n")
                f.write("=" * 80 + "\n")
                first_snap = self.snapshots[0]["snapshot"]
                last_snap = self.snapshots[-1]["snapshot"]
                top_stats = last_snap.compare_to(first_snap, "lineno")[:30]
                for stat in top_stats:
                    f.write(f"{stat}\n")

        print("Detailed report saved to memory_profiling_report.txt")
        tracemalloc.stop()


# Initialize pygame
pygame.init()
pygame.display.set_caption("N++ Environment Test")

# Argument parser
parser = argparse.ArgumentParser(description="Test N++ environment.")
parser.add_argument(
    "--log-frametimes", action="store_true", help="Enable frametime logging to stdout."
)
parser.add_argument(
    "--headless", action="store_true", help="Run in headless mode (no GUI)."
)
parser.add_argument(
    "--profile-frames",
    type=int,
    default=None,
    help="Run for a specific number of frames and then exit (for profiling).",
)
parser.add_argument(
    "--profile-memory",
    action="store_true",
    help="Enable detailed memory profiling using tracemalloc and memory_profiler.",
)
parser.add_argument(
    "--memory-snapshot-interval",
    type=int,
    default=100,
    help="Number of frames between memory snapshots when memory profiling is enabled (default: 100).",
)

parser.add_argument(
    "--map",
    type=str,
    default=None,
    help="Custom map file path to load (overrides default map behavior).",
)
# Graph visualization arguments
parser.add_argument(
    "--visualize-graph",
    action="store_true",
    help="Enable graph visualization overlay on simulator.",
)
parser.add_argument(
    "--standalone-graph",
    action="store_true",
    help="Show standalone graph visualization window.",
)
parser.add_argument(
    "--interactive-graph",
    action="store_true",
    help="Launch interactive graph visualization.",
)

parser.add_argument(
    "--save-graph",
    type=str,
    default=None,
    help="Save graph visualization to image file.",
)

parser.add_argument(
    "--export-frame",
    type=str,
    default=None,
    help="Export first frame of simulation to specified image file and quit (for AI testing).",
)

# Path-aware reward shaping debug/visualization arguments
parser.add_argument(
    "--test-path-aware",
    action="store_true",
    help="Test path-aware reward shaping system (precomputed tile connectivity + pathfinding).",
)
parser.add_argument(
    "--show-path-distances",
    action="store_true",
    help="Display path distances to switch/exit overlaid on visualization.",
)
parser.add_argument(
    "--visualize-adjacency-graph",
    action="store_true",
    help="Visualize the adjacency graph built from tile connectivity.",
)
parser.add_argument(
    "--show-blocked-entities",
    action="store_true",
    help="Highlight blocked positions from doors/mines on visualization.",
)
parser.add_argument(
    "--benchmark-pathfinding",
    action="store_true",
    help="Run pathfinding performance benchmarks and print detailed timing stats.",
)
parser.add_argument(
    "--export-path-analysis",
    type=str,
    default=None,
    help="Export frame with path-aware analysis overlay to specified image file and quit.",
)

# Replay recording arguments
parser.add_argument(
    "--record",
    action="store_true",
    help="Enable compact replay recording for behavioral cloning",
)
parser.add_argument(
    "--recording-output",
    type=str,
    default="datasets/human_replays",
    help="Output directory for recorded replays",
)

# Demo checkpoint visualization arguments
parser.add_argument(
    "--demo-replays",
    type=str,
    default=None,
    help="Directory containing demo replay files for checkpoint visualization",
)

# Test suite loading arguments
parser.add_argument(
    "--test-suite",
    action="store_true",
    help="Load and validate test suite levels sequentially",
)
parser.add_argument(
    "--test-dataset-path",
    type=str,
    default="datasets/test",
    help=(
        "Path to legacy test suite root (test_metadata.json) or, with "
        "--benchmark-dataset, the mechanic benchmark root containing manifest.json "
        "(default: datasets/test; if --benchmark-dataset and this is still the "
        "default, datasets/mechanic_benchmark is used)"
    ),
)
parser.add_argument(
    "--start-level",
    type=int,
    default=0,
    help="Start from this level index in test suite (default: 0)",
)
parser.add_argument(
    "--auto-advance",
    action="store_true",
    help="Auto-advance to next level on completion (for validation)",
)
parser.add_argument(
    "--benchmark-dataset",
    action="store_true",
    help=(
        "Load levels from a mechanic-primary benchmark dataset (manifest.json + "
        "train/<BUCKET>/T<tier>/*.pkl from scripts/generate_benchmark_dataset.py). "
        "Uses --test-dataset-path as the dataset root; if that is still the default "
        "datasets/test, uses datasets/mechanic_benchmark instead. Incompatible with "
        "--test-suite."
    ),
)
parser.add_argument(
    "--benchmark-split",
    type=str,
    choices=("train", "test", "both"),
    default="train",
    help="Which split(s) to include when --benchmark-dataset is set (default: train).",
)

# Generator testing arguments
parser.add_argument(
    "--test-generators",
    action="store_true",
    help="Enable generator testing mode - dynamically generate levels from categories",
)
parser.add_argument(
    "--generator-category",
    type=str,
    default=None,
    help="Test specific category (e.g., 'simple', 'medium', 'complex'). If not specified, cycles through all categories.",
)
parser.add_argument(
    "--generator-seed-start",
    type=int,
    default=10000,
    help="Starting seed for generator testing (default: 10000)",
)

# Official maps testing arguments
parser.add_argument(
    "--test-official-maps",
    action="store_true",
    help="Enable official maps testing mode - cycle through maps in nclone/maps/official/SI",
)

# Input mode arguments
parser.add_argument(
    "--discrete-actions",
    action="store_true",
    help="Enable discrete action mode - single action per keypress instead of continuous actions while held",
)
parser.add_argument(
    "--step-on-input",
    action="store_true",
    help="Enable step-on-input mode - simulation only ticks when an input is pressed (frame-perfect input sequences)",
)

args = parser.parse_args()

# Dataset root for mechanic benchmark mode (see docs/BENCHMARK_DATASET.md)
benchmark_dataset_root = args.test_dataset_path
if args.benchmark_dataset and args.test_dataset_path == "datasets/test":
    benchmark_dataset_root = "datasets/mechanic_benchmark"

if args.benchmark_dataset and args.test_suite:
    print("Error: --benchmark-dataset cannot be used together with --test-suite.")
    sys.exit(1)
if args.benchmark_dataset and args.test_generators:
    print("Error: --benchmark-dataset cannot be used together with --test-generators.")
    sys.exit(1)
if args.benchmark_dataset and args.test_official_maps:
    print("Error: --benchmark-dataset cannot be used together with --test-official-maps.")
    sys.exit(1)

print(f"Headless: {args.headless}")

# Display discrete action mode info
if args.discrete_actions:
    print("\n" + "=" * 60)
    print("DISCRETE ACTION MODE ENABLED")
    print("=" * 60)
    print("Each keypress will trigger a single action")
    print("Keys must be released and pressed again for each action")
    print("Holding keys will NOT repeat actions")
    print("=" * 60 + "\n")

# Display step-on-input mode info (implies discrete actions)
if args.step_on_input:
    if args.headless:
        print(
            "Warning: --step-on-input requires visual display; ignoring in headless mode."
        )
        args.step_on_input = False
    else:
        args.discrete_actions = True  # Step-on-input implies discrete actions
        print("\n" + "=" * 60)
        print("STEP-ON-INPUT MODE ENABLED (Frame-Perfect)")
        print("=" * 60)
        print("Simulation advances ONLY when an input is pressed")
        print("Tap (<200ms) = single step; hold (200ms+) = repeated steps")
        print("Arrows: Left/Right/Jump | Z=Jump+Left | X=Jump+Right | .=NOOP")
        print("No tick when no input - test frame-perfect input sequences")
        print("Input sequence overlay shown when active")
        print("=" * 60 + "\n")

# Display help information for generator testing
if args.test_generators:
    print("\n" + "=" * 60)
    print("GENERATOR TESTING MODE")
    print("=" * 60)
    if args.generator_category:
        print(f"• Testing category: {args.generator_category}")
    else:
        print("• Testing all categories (cycling)")
    print(f"• Starting seed: {args.generator_seed_start}")
    print("\nControls:")
    print("  R - Reset (generate new map from current generator)")
    print("  G - Next generator in category")
    print("  Shift+G - Previous generator in category")
    print("  K - Next category")
    print("  Shift+K - Previous category")
    print("  L - List all generators in current category")
    print("  1-9 - Jump to generator number in category")
    print("  V - Toggle ASCII visualization output")
    print("=" * 60 + "\n")

# Display help information for official maps testing
if args.test_official_maps:
    print("\n" + "=" * 60)
    print("OFFICIAL MAPS TESTING MODE")
    print("=" * 60)
    print("• Testing maps from: nclone/maps/official/SI")
    print("\nControls:")
    print("  R - Reset (reload current map)")
    print("  G - Next map")
    print("  Shift+G - Previous map")
    print("  1-9 - Jump to map number (1-9)")
    print("=" * 60 + "\n")

# Display help information for test suite loading
if args.test_suite:
    print("\n" + "=" * 60)
    print("TEST SUITE VALIDATION MODE")
    print("=" * 60)
    print(f"• Dataset path: {args.test_dataset_path}")
    print(f"• Starting from level index: {args.start_level}")
    if args.auto_advance:
        print("• Auto-advance enabled: next level loads on completion")
    else:
        print("• Manual advance: press 'N' to load next level")
    print("\nControls:")
    print("  R - Reset current level")
    print("  N - Load next level")
    print("  P - Load previous level")
    print("=" * 60 + "\n")

# Display help for mechanic benchmark dataset mode
if args.benchmark_dataset:
    print("\n" + "=" * 60)
    print("MECHANIC BENCHMARK DATASET MODE")
    print("=" * 60)
    print(f"• Dataset root: {benchmark_dataset_root}")
    print(f"• Split(s): {args.benchmark_split}")
    print(f"• Starting level index: {args.start_level}")
    if args.auto_advance:
        print("• Auto-advance enabled: next level loads on completion")
    else:
        print("• Manual advance: press 'N' / 'P' to change levels")
    print("\nOn-screen HUD shows bucket, tier, generator, preset, seed, and level id.")
    print("\nControls:")
    print("  R - Reset current level")
    print("  N - Load next level")
    print("  P - Load previous level")
    print("=" * 60 + "\n")

# Display help information for graph visualization
if args.visualize_graph or args.standalone_graph or args.interactive_graph:
    print("\n" + "=" * 60)
    print("GRAPH VISUALIZATION ACTIVE")
    print("=" * 60)
    if args.visualize_graph:
        print("• Graph overlay enabled on simulator")
    if args.standalone_graph:
        print("• Standalone graph window enabled")
    if args.interactive_graph:
        print("• Interactive graph mode enabled")

    print("\nRuntime Controls:")
    print("  V - Toggle graph overlay")
    print("  S - Save graph visualization")
    print("  G/E/C - Toggle debug overlays")
    print("  R - Reset environment")

    if args.show_edges:
        print(f"\nEdge types shown: {', '.join(args.show_edges)}")

    print("=" * 60 + "\n")

# Display help information for path-aware reward shaping testing
if (
    args.test_path_aware
    or args.visualize_adjacency_graph
    or args.show_blocked_entities
    or args.benchmark_pathfinding
):
    print("\n" + "=" * 60)
    print("PATH-AWARE REWARD SHAPING TESTING")
    print("=" * 60)
    print("Testing precomputed tile connectivity + pathfinding system")
    if args.visualize_adjacency_graph:
        print("• Adjacency graph visualization enabled")
    if args.show_blocked_entities:
        print("• Blocked entity highlighting enabled")
    if args.benchmark_pathfinding:
        print("• Performance benchmarking enabled")

    print("\nRuntime Controls:")
    print("  A - Toggle adjacency graph visualization")
    print("  B - Toggle blocked entity highlighting")
    print("  P - Toggle PBRS path overlay (shortest path to goal + cost-to-go heatmap)")
    print("  O - Toggle route PBRS overlay (surface heatmap + arcs + optimal route + arc features)")
    print("  M - Toggle Mine SDF (Signed Distance Field) heatmap")
    print("      • Red=danger zone | Yellow=boundary | Green=safe")
    print("  T - Run pathfinding benchmark at current position")
    print("  X - Export path analysis screenshot")
    print("  R - Reset environment")
    print("  Arrow Keys - Move ninja")

    print("\nPerformance Targets:")
    print("  • Graph build (first call): < 5ms")
    print("  • Graph build (cached): < 0.05ms")
    print("  • Path distance (BFS): 2-3ms")
    print("  • Path distance (A*): 1-2ms")
    print("  • Path distance (cached): < 0.1ms")

    print("=" * 60 + "\n")

# Display help information for demo checkpoint visualization
if args.demo_replays:
    print("\n" + "=" * 60)
    print("DEMO CHECKPOINT VISUALIZATION")
    print("=" * 60)
    print(f"Demo replay directory: {args.demo_replays}")
    print("Visualizes expert demonstration trajectories as heatmap overlay")
    print("\nRuntime Controls:")
    print("  Q - Toggle demo checkpoint heatmap")
    print("      • Blue = Low reward (start of trajectory)")
    print("      • Green/Yellow = Mid reward (progress)")
    print("      • Red = High reward (near goal)")
    print("      • Red circle marker = Highest reward position")
    print("\nNote:")
    print("  • Checkpoints are loaded when 'Q' is first pressed")
    print("  • Level name matching is case-insensitive")
    print("  • Uses same 12px grid as Go-Explore checkpoints")
    print("=" * 60 + "\n")

if args.interactive_graph and args.headless:
    print("Error: Interactive graph visualization cannot be used in headless mode.")
    sys.exit(1)

# Create environment
render_mode = "grayscale_array" if args.headless else "human"
debug_overlay_enabled = True  # Disable overlay in headless mode

# Create environment configuration with custom map path if provided
if args.map:
    from nclone.gym_environment.config import (
        EnvironmentConfig,
        RenderConfig,
        GraphConfig,
        ReachabilityConfig,
    )

    # Create config with proper render mode for headless
    render_config = RenderConfig(
        render_mode=render_mode,
        enable_animation=not args.headless,
        enable_debug_overlay=debug_overlay_enabled,
    )
    config = EnvironmentConfig(
        custom_map_path=args.map,
        enable_logging=False,
        render=render_config,
        graph=GraphConfig(
            debug=False,
        ),
        reachability=ReachabilityConfig(debug=False),
        test_dataset_path=args.test_dataset_path,
        frame_skip=1,
        enable_profiling=args.profile_frames is not None,
    )
    env = create_visual_testing_env(config=config)
    print(f"Loading custom map from: {args.map}")
else:
    from nclone.gym_environment.config import (
        EnvironmentConfig,
        RenderConfig,
        GraphConfig,
        ReachabilityConfig,
    )

    # Create config with proper render mode for headless
    render_config = RenderConfig(
        render_mode=render_mode,
        enable_animation=not args.headless,
        enable_debug_overlay=debug_overlay_enabled,
    )
    config = EnvironmentConfig(
        enable_logging=False,
        render=render_config,
        graph=GraphConfig(
            debug=False,
        ),
        reachability=ReachabilityConfig(debug=False),
        test_dataset_path=args.test_dataset_path,
        frame_skip=1,
        enable_profiling=args.profile_frames is not None,
    )
    env = create_visual_testing_env(config=config)


def _draw_benchmark_overlay(lines) -> None:
    """Draw mechanic-benchmark metadata in the top-left (after env.render())."""
    if not lines:
        return
    screen = pygame.display.get_surface()
    if screen is None:
        return
    font = pygame.font.SysFont("monospace", 14)
    x0, y = 8, 6
    pad = 4
    for line in lines:
        text = font.render(line, True, (255, 255, 255))
        bg = pygame.Surface((text.get_width() + pad * 2, text.get_height() + 4))
        bg.set_alpha(210)
        bg.fill((20, 22, 32))
        screen.blit(bg, (x0, y))
        screen.blit(text, (x0 + pad, y + 2))
        y += bg.get_height() + 4
    pygame.display.flip()


# Define manual_reset helper function before initialization blocks that use it
def manual_reset(env: NppEnvironment, map_name: str = None):
    """
    Reset environment after externally loading a new map.

    This is used in test_environment.py for generator testing and level switching.
    Instead of reimplementing reset logic, we call the environment's built-in reset()
    with skip_map_load=True to ensure all caches are properly cleared and rebuilt.

    Args:
        env: The environment to reset
        map_name: Optional name of the map that was just loaded (for cache keying)
    """
    print("Manual reset called - delegating to env.reset() with skip_map_load=True")

    # Use the environment's standard reset flow with skip_map_load
    # This ensures all caches are properly cleared and rebuilt
    reset_options = {"skip_map_load": True}
    if map_name is not None:
        reset_options["map_name"] = map_name
        print(f"  Map name: {map_name}")

    env.reset(options=reset_options)

    print("Manual reset complete")


# Initialize generator testing if enabled
generator_tester = None
show_ascii_on_reset = False

if args.test_generators:
    try:
        from nclone.map_generation.generator_configs import CATEGORIES
        from nclone.map_generation.generator_factory import GeneratorFactory

        class GeneratorTester:
            """Manages cycling through categories and generators for testing."""

            def __init__(self, start_seed=10000, specific_category=None):
                self.categories = CATEGORIES
                self.category_names = list(self.categories.keys())
                self.current_seed = start_seed

                if specific_category:
                    if specific_category not in self.category_names:
                        print(
                            f"Warning: Category '{specific_category}' not found. Available: {self.category_names}"
                        )
                        self.current_category_idx = 0
                    else:
                        self.current_category_idx = self.category_names.index(
                            specific_category
                        )
                else:
                    self.current_category_idx = 0

                self.current_generator_idx = 0
                self.specific_category = specific_category

            def get_current_category(self):
                """Get current category config."""
                return self.categories[self.category_names[self.current_category_idx]]

            def get_current_generator_info(self):
                """Get current generator type and preset."""
                category = self.get_current_category()
                generators = category.generators
                return generators[self.current_generator_idx]

            def next_generator(self):
                """Move to next generator in current category."""
                category = self.get_current_category()
                self.current_generator_idx = (self.current_generator_idx + 1) % len(
                    category.generators
                )
                return self.get_current_generator_info()

            def prev_generator(self):
                """Move to previous generator in current category."""
                category = self.get_current_category()
                self.current_generator_idx = (self.current_generator_idx - 1) % len(
                    category.generators
                )
                return self.get_current_generator_info()

            def next_category(self):
                """Move to next category."""
                if not self.specific_category:
                    self.current_category_idx = (self.current_category_idx + 1) % len(
                        self.category_names
                    )
                    self.current_generator_idx = 0
                return self.category_names[self.current_category_idx]

            def prev_category(self):
                """Move to previous category."""
                if not self.specific_category:
                    self.current_category_idx = (self.current_category_idx - 1) % len(
                        self.category_names
                    )
                    self.current_generator_idx = 0
                return self.category_names[self.current_category_idx]

            def generate_map(self):
                """Generate a new map using current generator."""
                gen_type, preset = self.get_current_generator_info()

                # Use factory for all generators
                map_obj = GeneratorFactory.create_from_preset(
                    gen_type, preset, seed=self.current_seed
                )
                map_obj.generate(seed=self.current_seed)

                # Increment seed for next generation
                self.current_seed += 1

                return map_obj

            def get_info_string(self):
                """Get current state info string."""
                category_name = self.category_names[self.current_category_idx]
                category = self.get_current_category()
                gen_type, preset = self.get_current_generator_info()

                return (
                    f"Category: {category_name} "
                    f"({self.current_generator_idx + 1}/{len(category.generators)}) | "
                    f"Generator: {gen_type}:{preset} | "
                    f"Seed: {self.current_seed}"
                )

            def list_generators(self):
                """Print all generators in current category."""
                category_name = self.category_names[self.current_category_idx]
                category = self.get_current_category()

                print("\n" + "=" * 60)
                print(f"GENERATORS IN CATEGORY: {category_name}")
                print("=" * 60)
                print(f"Description: {category.description}")
                print(f"Total generators: {len(category.generators)}\n")

                for idx, (gen_type, preset) in enumerate(category.generators, 1):
                    marker = "→" if idx - 1 == self.current_generator_idx else " "
                    print(f"{marker} {idx}. {gen_type}:{preset}")

                print("=" * 60 + "\n")

            def jump_to_generator(self, index):
                """Jump to a specific generator by index (0-based)."""
                category = self.get_current_category()
                if 0 <= index < len(category.generators):
                    self.current_generator_idx = index
                    return True
                return False

        generator_tester = GeneratorTester(
            start_seed=args.generator_seed_start,
            specific_category=args.generator_category,
        )

        # Generate initial map
        initial_map = generator_tester.generate_map()
        env.nplay_headless.load_map_from_map_data(initial_map.map_data())
        env._reset_graph_state()
        env._update_graph_from_env_state()
        env._build_door_feature_cache()

        print("Generator Testing Initialized")
        print(f"  {generator_tester.get_info_string()}")

    except Exception as e:
        print(f"Error initializing generator testing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

# Initialize official maps testing if enabled
official_map_tester = None

if args.test_official_maps:
    try:
        import os

        class OfficialMapTester:
            """Manages cycling through official maps in the SI folder."""

            def __init__(self):
                # Get the path to the SI maps folder
                # Maps are in nclone/nclone/maps/official/SI
                # Use same pattern as nplay_headless.py for consistency
                current_file_dir = os.path.dirname(__file__)
                self.maps_dir = os.path.join(current_file_dir, "maps", "official", "SI")

                if not os.path.exists(self.maps_dir):
                    raise FileNotFoundError(
                        f"Official maps directory not found: {self.maps_dir}"
                    )

                # Get all map files and sort by filename (which includes numeric prefix)
                map_files = [
                    f
                    for f in os.listdir(self.maps_dir)
                    if os.path.isfile(os.path.join(self.maps_dir, f))
                ]
                # Sort naturally by filename (000, 001, 002, etc.)
                map_files.sort()
                self.map_files = map_files
                self.current_map_idx = 0

                if not self.map_files:
                    raise ValueError(f"No map files found in {self.maps_dir}")

                print(f"Found {len(self.map_files)} official maps")

            def get_current_map_path(self):
                """Get the full path to the current map file."""
                return os.path.join(self.maps_dir, self.map_files[self.current_map_idx])

            def get_current_map_name(self):
                """Get the name of the current map file."""
                return self.map_files[self.current_map_idx]

            def next_map(self):
                """Move to next map."""
                self.current_map_idx = (self.current_map_idx + 1) % len(self.map_files)
                return self.get_current_map_path()

            def prev_map(self):
                """Move to previous map."""
                self.current_map_idx = (self.current_map_idx - 1) % len(self.map_files)
                return self.get_current_map_path()

            def jump_to_map(self, index):
                """Jump to a specific map by index (0-based)."""
                if 0 <= index < len(self.map_files):
                    self.current_map_idx = index
                    return True
                return False

            def get_info_string(self):
                """Get current state info string."""
                return (
                    f"Map {self.current_map_idx + 1}/{len(self.map_files)}: "
                    f"{self.get_current_map_name()}"
                )

        official_map_tester = OfficialMapTester()

        # Load initial map and reset environment
        initial_map_path = official_map_tester.get_current_map_path()
        env.nplay_headless.load_map(initial_map_path)

        # Use manual_reset with map name for proper cache building
        map_name = official_map_tester.get_current_map_name()
        manual_reset(env, map_name=map_name)

        print("Official Maps Testing Initialized")
        print(f"  {official_map_tester.get_info_string()}")

    except Exception as e:
        print(f"Error initializing official maps testing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

# Initialize test suite loader if test suite mode is enabled
test_suite_loader = None
test_suite_level_ids = []
current_level_idx = 0

# Mechanic-primary benchmark dataset (see docs/BENCHMARK_DATASET.md)
benchmark_loader = None
benchmark_index = 0
benchmark_payload = None

if args.test_suite:
    try:
        from nclone.evaluation.test_suite_loader import TestSuiteLoader

        test_suite_loader = TestSuiteLoader(args.test_dataset_path)
        test_suite_level_ids = test_suite_loader.get_all_levels()
        current_level_idx = args.start_level

        if current_level_idx >= len(test_suite_level_ids):
            print(
                f"Error: Start level {current_level_idx} exceeds available levels ({len(test_suite_level_ids)})"
            )
            sys.exit(1)

        # Load first level from test suite
        level_id = test_suite_level_ids[current_level_idx]
        level = test_suite_loader.get_level(level_id)
        env.nplay_headless.load_map_from_map_data(level["map_data"])

        print(
            f"Loaded level {current_level_idx + 1}/{len(test_suite_level_ids)}: {level_id}"
        )
        print(f"Category: {level.get('category', 'unknown')}")
        print(f"Description: {level.get('metadata', {}).get('description', 'N/A')}\n")

        # Reset environment after loading test suite map
        env.reset()
    except Exception as e:
        print(f"Error loading test suite: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

if args.benchmark_dataset:
    try:
        splits = (
            ("train", "test")
            if args.benchmark_split == "both"
            else (args.benchmark_split,)
        )
        benchmark_loader = MechanicBenchmarkLoader(
            benchmark_dataset_root, splits=splits
        )
        if len(benchmark_loader) == 0:
            print(f"Error: no levels found under {benchmark_dataset_root!r}")
            sys.exit(1)

        benchmark_index = args.start_level
        if benchmark_index >= len(benchmark_loader):
            print(
                f"Error: Start level {benchmark_index} exceeds available levels "
                f"({len(benchmark_loader)})"
            )
            sys.exit(1)

        benchmark_payload = benchmark_loader.load_payload(benchmark_index)
        env.nplay_headless.load_map_from_map_data(benchmark_payload["map_data"])
        map_name = f"benchmark_{benchmark_payload.get('level_id', benchmark_index)}"
        manual_reset(env, map_name=map_name)

        print(
            f"Loaded benchmark level {benchmark_index + 1}/{len(benchmark_loader)} "
            f"(level_id={benchmark_payload.get('level_id')})"
        )
        print(
            f"  {benchmark_payload.get('bucket')} T{benchmark_payload.get('tier')} "
            f"{benchmark_payload.get('generator')}:{benchmark_payload.get('preset')}\n"
        )
    except Exception as e:
        print(f"Error loading benchmark dataset: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

# Normal environment initialization - call reset only if not in special modes
if (
    not args.test_generators
    and not args.test_suite
    and not args.test_official_maps
    and not args.benchmark_dataset
):
    env.reset()

# Initialize replay recorder if enabled
recorder = None
if args.record:
    recorder = GameplayRecorder(output_dir=args.recording_output)
    print("\n" + "=" * 60)
    print("COMPACT REPLAY RECORDING ENABLED")
    print("=" * 60)
    print(f"Output directory: {args.recording_output}")
    print("Storage format: Binary (map + inputs only)")
    print("\nControls:")
    print("  B - Start recording (resets to spawn on SAME map)")
    print("  N - Stop recording (without saving)")
    print("  R - Reset (auto-saves if successful)")
    print("\nIMPORTANT:")
    print("  • Pressing 'B' resets to spawn position on current map")
    print("  • The map does NOT change when starting a recording")
    print("  • Play through the level to completion (win/death)")
    print("  • Only complete episodes are saved for training")
    print("=" * 60 + "\n")

graph_debug_enabled = False
grid_debug_enabled = False
tile_rendering_enabled = True
tile_types_debug_enabled = False
action_mask_debug_enabled = False
reachable_walls_debug_enabled = False
mine_sdf_debug_enabled = False
path_aware_system = None
adjacency_graph_debug_enabled = False
blocked_entities_debug_enabled = False
show_paths_to_goals = False
value_function_debug_enabled = False
value_function_pbrs_cache = {}  # level_name -> pbrs instance
reachability_overlay_enabled = False
pbrs_path_overlay_enabled = False
route_overlay_enabled = False
route_pbrs_instance = None   # RoutePBRS loaded for the 'R' key overlay

if (
    args.test_path_aware
    or args.visualize_adjacency_graph
    or args.show_blocked_entities
    or args.benchmark_pathfinding
    or args.export_path_analysis
):
    print("Initializing path-aware reward shaping system...")

    try:
        from nclone.graph.reachability.graph_builder import GraphBuilder
        from nclone.graph.reachability.path_distance_calculator import (
            CachedPathDistanceCalculator,
        )
        from nclone.graph.reachability.tile_connectivity_loader import (
            TileConnectivityLoader,
        )

        # Initialize components
        graph_builder = GraphBuilder()
        path_calculator = CachedPathDistanceCalculator()
        connectivity_loader = TileConnectivityLoader()

        # Store in dict for easy access
        path_aware_system = {
            "graph_builder": graph_builder,
            "path_calculator": path_calculator,
            "connectivity_loader": connectivity_loader,
            "current_graph": None,
            "current_entity_mask": None,
            "cached_level_data": None,
            "cached_switch_states": None,  # Track switch states to detect changes
        }

        # Set initial debug states
        adjacency_graph_debug_enabled = args.visualize_adjacency_graph
        blocked_entities_debug_enabled = args.show_blocked_entities

        print("✅ Path-aware reward shaping system initialized successfully")
        print(f"   - Connectivity table shape: {connectivity_loader.table_shape}")
        print(
            f"   - Connectivity table size: {connectivity_loader.table_size_kb:.2f} KB"
        )
        print(f"   - Adjacency graph display: {adjacency_graph_debug_enabled}")
        print(f"   - Blocked entities display: {blocked_entities_debug_enabled}")

        # Set initial flags on environment if any are enabled
        if adjacency_graph_debug_enabled or blocked_entities_debug_enabled:
            env.set_adjacency_graph_debug_enabled(adjacency_graph_debug_enabled)
            env.set_blocked_entities_debug_enabled(blocked_entities_debug_enabled)

    except Exception as e:
        print(f"Warning: Could not initialize path-aware system: {e}")
        import traceback

        traceback.print_exc()

# Initialize demo checkpoint visualization system if demo directory provided
demo_checkpoint_system = None
demo_checkpoints_debug_enabled = False

if args.demo_replays:
    print("Initializing demo checkpoint visualization system...")

    try:
        # Import DemoCheckpointSeeder from nclone.replay
        from nclone.replay.demo_checkpoint_seeder import DemoCheckpointSeeder

        demo_checkpoint_system = {
            "seeder": DemoCheckpointSeeder(
                replay_dir=args.demo_replays,
                max_demos_per_level=10,
                min_cumulative_reward=0.0,
            ),
            "checkpoint_cache": {},  # Maps level_name -> list of checkpoint dicts
            "cached_level_name": None,  # Track current level for cache invalidation
        }

        print("✅ Demo checkpoint visualization system initialized successfully")
        print(f"   - Demo replay directory: {args.demo_replays}")
        print("   - Toggle with 'Q' key during runtime")

    except Exception as e:
        print(f"Warning: Could not initialize demo checkpoint system: {e}")
        import traceback

        traceback.print_exc()
        demo_checkpoint_system = None

# Handle frame export if requested
if args.export_frame:
    print(f"Exporting initial frame to {args.export_frame}...")

    # Step the environment once to get a proper frame
    observation, reward, terminated, truncated, info = env.step(0)  # NOOP action

    # Get the rendered frame
    if args.headless or env.render_mode == "grayscale_array":
        # In headless mode, we need to call render to get the RGB array
        frame = env.render()
        if frame is not None:
            # Convert to PIL Image and save
            if isinstance(frame, np.ndarray):
                # Handle different frame formats
                if len(frame.shape) == 3:
                    if frame.shape[2] == 1:
                        # Single channel (grayscale) - squeeze to 2D
                        frame_2d = np.squeeze(frame, axis=2)
                        image = Image.fromarray(frame_2d.astype(np.uint8), mode="L")
                        print(f"Exporting grayscale frame with shape {frame.shape}")
                    elif frame.shape[2] == 3:
                        # RGB format
                        image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
                        print(f"Exporting RGB frame with shape {frame.shape}")
                    elif frame.shape[2] == 4:
                        # RGBA format
                        image = Image.fromarray(frame.astype(np.uint8), mode="RGBA")
                        print(f"Exporting RGBA frame with shape {frame.shape}")
                    else:
                        print(
                            f"Warning: Unsupported frame format with {frame.shape[2]} channels"
                        )
                        image = None
                elif len(frame.shape) == 2:
                    # Already 2D grayscale
                    image = Image.fromarray(frame.astype(np.uint8), mode="L")
                    print(f"Exporting 2D grayscale frame with shape {frame.shape}")
                else:
                    print(f"Warning: Unsupported frame shape {frame.shape}")
                    image = None

                if image is not None:
                    image.save(args.export_frame)
                    print(f"Frame successfully exported to {args.export_frame}")
                else:
                    print("Could not export frame due to unsupported format")
            else:
                print(f"Warning: Frame is not a numpy array: {type(frame)}")
        else:
            print("Warning: Could not get frame from environment render()")
    else:
        print(
            "Warning: Frame export requires headless mode or grayscale_array render mode"
        )
        print("Hint: Use --headless flag with --export-frame")

    # Clean up and exit
    pygame.quit()
    env.close()
    sys.exit(0)

# Initialize clock for 60 FPS
clock = None
if not args.headless:
    clock = pygame.time.Clock()
running = True
last_time = time.perf_counter()

# Create a profiler object
profiler = cProfile.Profile()

# Initialize memory profiler if requested
memory_profiler = None
if args.profile_memory:
    memory_profiler = MemoryProfiler(snapshot_interval=args.memory_snapshot_interval)
    memory_profiler.start()

# Discrete action mode tracking
keys_just_pressed = (
    set()
)  # Track which keys were pressed this frame (for discrete mode)
keys_held = set()  # Track which keys are currently held (to prevent repeat on hold)

# Step-on-input mode: track input sequence for overlay
input_sequence = []
HOLD_REPEAT_DELAY_MS = 200  # ms before held keys repeat (enables single-frame taps)
hold_start_time = None  # when current key hold started
hold_action = 0  # action being held (for repeat delay)

manual_reset(env)
# Main game loop
# Wrap the game loop with profiler.enable() and profiler.disable()
profiler.enable()
frame_counter = 0  # Initialize frame counter
while running:
    # Reset keys pressed this frame (for discrete action mode)
    keys_just_pressed.clear()
    # Handle pygame events
    if not args.headless:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYUP:
                # Track key releases for discrete action mode
                if args.discrete_actions and event.key in keys_held:
                    keys_held.discard(event.key)
            if event.type == pygame.KEYDOWN:
                # Track arrow key presses for discrete action mode
                if args.discrete_actions and event.key in (
                    pygame.K_LEFT,
                    pygame.K_RIGHT,
                    pygame.K_UP,
                ):
                    # Only register if key was not already held (prevents repeat on hold)
                    if event.key not in keys_held:
                        keys_just_pressed.add(event.key)
                        keys_held.add(event.key)

                if event.key == pygame.K_r:
                    if args.step_on_input:
                        input_sequence.clear()
                    # Check if episode was successful before reset (for recorder)
                    if recorder is not None and recorder.is_recording:
                        # Determine if episode was successful
                        player_won = env.nplay_headless.ninja_has_won()
                        recorder.stop_recording(success=player_won, save=player_won)

                    # Benchmark dataset: reload current level from pickle
                    if benchmark_loader is not None and benchmark_payload is not None:
                        try:
                            env.nplay_headless.load_map_from_map_data(
                                benchmark_payload["map_data"]
                            )
                            map_name = (
                                f"benchmark_{benchmark_payload.get('level_id', benchmark_index)}"
                            )
                            manual_reset(env, map_name=map_name)
                            initial_obs = env._get_observation()
                            observation = env._process_observation(initial_obs)
                            print(
                                f"Reloaded benchmark level {benchmark_index + 1}/"
                                f"{len(benchmark_loader)} (level_id="
                                f"{benchmark_payload.get('level_id')})"
                            )
                        except Exception as e:
                            print(f"Error reloading benchmark level: {e}")
                            import traceback

                            traceback.print_exc()
                    # Generator testing mode: generate new map from current generator
                    elif generator_tester is not None:
                        try:
                            new_map = generator_tester.generate_map()
                            env.nplay_headless.load_map_from_map_data(
                                new_map.map_data()
                            )

                            manual_reset(env)

                            # Get initial observation
                            initial_obs = env._get_observation()
                            observation = env._process_observation(initial_obs)

                            print(
                                f"Generated new map: {generator_tester.get_info_string()}"
                            )
                        except Exception as e:
                            print(f"Error generating map: {e}")
                            import traceback

                            traceback.print_exc()
                    elif official_map_tester is not None:
                        # Official maps testing mode: reload current map
                        try:
                            current_map_path = (
                                official_map_tester.get_current_map_path()
                            )
                            env.nplay_headless.load_map(current_map_path)

                            # Pass map name for proper cache keying
                            map_name = official_map_tester.get_current_map_name()
                            manual_reset(env, map_name=map_name)

                            # Get initial observation
                            initial_obs = env._get_observation()
                            observation = env._process_observation(initial_obs)

                            print(
                                f"Reloaded map: {official_map_tester.get_info_string()}"
                            )
                        except Exception as e:
                            print(f"Error reloading map: {e}")
                            import traceback

                            traceback.print_exc()
                    elif generator_tester is None and official_map_tester is None:
                        # Normal reset for non-generator-testing mode
                        observation, info = env.reset()

                if event.key == pygame.K_b and recorder is not None:
                    # Start recording
                    if not recorder.is_recording:
                        if args.step_on_input:
                            input_sequence.clear()
                        # CRITICAL: Reset environment to spawn position on SAME map
                        # Save current map data before resetting
                        current_map_data = list(env.nplay_headless.current_map_data)

                        print("Resetting to spawn position on current map...")

                        # Reset physics and observation processors
                        manual_reset(env)

                        # Reset ninja physics but keep same map
                        env.nplay_headless.reset()
                        env.nplay_headless.load_map_from_map_data(current_map_data)

                        # Get initial observation
                        initial_obs = env._get_observation()
                        observation = env._process_observation(initial_obs)

                        # Prepare map data for recording
                        map_data = bytes(
                            max(0, min(255, int(b))) for b in current_map_data
                        )
                        map_name = (
                            getattr(env.map_loader, "current_map_name", "unknown")
                            if hasattr(env, "map_loader")
                            else "unknown"
                        )
                        level_id = None  # TODO: Extract from env if using test suite
                        recorder.start_recording(map_data, map_name, level_id)
                    else:
                        print("Already recording!")

                if event.key == pygame.K_n:
                    # Test suite / benchmark navigation or recording control
                    if test_suite_loader is not None and test_suite_level_ids:
                        if args.step_on_input:
                            input_sequence.clear()
                        # Load next level in test suite
                        current_level_idx = (current_level_idx + 1) % len(
                            test_suite_level_ids
                        )
                        level_id = test_suite_level_ids[current_level_idx]
                        level = test_suite_loader.get_level(level_id)
                        env.nplay_headless.load_map_from_map_data(level["map_data"])
                        env.reset()
                        print(
                            f"Loaded level {current_level_idx + 1}/{len(test_suite_level_ids)}: {level_id}"
                        )
                        print(f"Category: {level.get('category', 'unknown')}")
                    elif benchmark_loader is not None:
                        if args.step_on_input:
                            input_sequence.clear()
                        benchmark_index = (benchmark_index + 1) % len(benchmark_loader)
                        benchmark_payload = benchmark_loader.load_payload(benchmark_index)
                        env.nplay_headless.load_map_from_map_data(
                            benchmark_payload["map_data"]
                        )
                        map_name = (
                            f"benchmark_{benchmark_payload.get('level_id', benchmark_index)}"
                        )
                        manual_reset(env, map_name=map_name)
                        print(
                            f"Loaded benchmark {benchmark_index + 1}/{len(benchmark_loader)}: "
                            f"level_id={benchmark_payload.get('level_id')} "
                            f"{benchmark_payload.get('bucket')} T{benchmark_payload.get('tier')} "
                            f"{benchmark_payload.get('generator')}:{benchmark_payload.get('preset')}"
                        )
                    elif recorder is not None and recorder.is_recording:
                        # Stop recording without saving
                        recorder.stop_recording(success=False, save=False)
                        print("Recording stopped (not saved)")

                if event.key == pygame.K_p:
                    if (
                        test_suite_loader is not None
                        and test_suite_level_ids
                    ):
                        # Load previous level in test suite
                        if args.step_on_input:
                            input_sequence.clear()
                        current_level_idx = (current_level_idx - 1) % len(
                            test_suite_level_ids
                        )
                        level_id = test_suite_level_ids[current_level_idx]
                        level = test_suite_loader.get_level(level_id)
                        env.nplay_headless.load_map_from_map_data(level["map_data"])
                        env.reset()
                        print(
                            f"Loaded level {current_level_idx + 1}/{len(test_suite_level_ids)}: {level_id}"
                        )
                        print(f"Category: {level.get('category', 'unknown')}")
                    elif benchmark_loader is not None:
                        if args.step_on_input:
                            input_sequence.clear()
                        benchmark_index = (benchmark_index - 1) % len(benchmark_loader)
                        benchmark_payload = benchmark_loader.load_payload(benchmark_index)
                        env.nplay_headless.load_map_from_map_data(
                            benchmark_payload["map_data"]
                        )
                        map_name = (
                            f"benchmark_{benchmark_payload.get('level_id', benchmark_index)}"
                        )
                        manual_reset(env, map_name=map_name)
                        print(
                            f"Loaded benchmark {benchmark_index + 1}/{len(benchmark_loader)}: "
                            f"level_id={benchmark_payload.get('level_id')}"
                        )
                if event.key == pygame.K_c:
                    # Toggle grid outline debug overlay
                    grid_debug_enabled = not grid_debug_enabled
                    try:
                        env.set_grid_debug_enabled(grid_debug_enabled)
                    except Exception:
                        pass
                if event.key == pygame.K_w:
                    # Toggle action mask debug overlay
                    action_mask_debug_enabled = not action_mask_debug_enabled
                    try:
                        env.set_action_mask_debug_enabled(action_mask_debug_enabled)
                        print(
                            f"Action mask debug: {'ON' if action_mask_debug_enabled else 'OFF'}"
                        )
                    except Exception as e:
                        print(f"Failed to toggle action mask debug: {e}")
                if event.key == pygame.K_l:
                    # Toggle tile rendering
                    tile_rendering_enabled = not tile_rendering_enabled
                    try:
                        env.set_tile_rendering_enabled(tile_rendering_enabled)
                        print(
                            f"Tile rendering: {'ON' if tile_rendering_enabled else 'OFF'}"
                        )
                    except Exception as e:
                        print(f"Could not toggle tile rendering: {e}")

                if event.key == pygame.K_i:
                    # Toggle tile type overlay
                    tile_types_debug_enabled = not tile_types_debug_enabled
                    try:
                        env.set_tile_types_debug_enabled(tile_types_debug_enabled)
                        print(
                            f"Tile type overlay: {'ON' if tile_types_debug_enabled else 'OFF'}"
                        )
                    except Exception as e:
                        print(f"Could not toggle tile type overlay: {e}")

                if event.key == pygame.K_m:
                    # Toggle mine SDF (Signed Distance Field) overlay
                    mine_sdf_debug_enabled = not mine_sdf_debug_enabled
                    try:
                        env.set_mine_sdf_debug_enabled(mine_sdf_debug_enabled)
                        print(
                            f"Mine SDF overlay: {'ON' if mine_sdf_debug_enabled else 'OFF'}"
                        )
                    except Exception as e:
                        print(f"Could not toggle mine SDF overlay: {e}")

                # Generator testing controls
                if event.key == pygame.K_g and generator_tester is not None:
                    # Next/previous generator
                    mods = pygame.key.get_mods()
                    try:
                        if mods & pygame.KMOD_SHIFT:
                            # Shift+G: Previous generator
                            generator_tester.prev_generator()
                        else:
                            # G: Next generator
                            generator_tester.next_generator()

                        # Generate new map with new generator
                        new_map = generator_tester.generate_map()
                        env.nplay_headless.load_map_from_map_data(new_map.map_data())

                        manual_reset(env)

                        # Get initial observation
                        initial_obs = env._get_observation()
                        observation = env._process_observation(initial_obs)

                        print(
                            f"Switched generator: {generator_tester.get_info_string()}"
                        )

                        if show_ascii_on_reset:
                            print("\nASCII Visualization:")
                            print(new_map.to_ascii(show_coords=False))
                            print()
                    except Exception as e:
                        print(f"Error switching generator: {e}")

                # Official maps testing controls
                if event.key == pygame.K_g and official_map_tester is not None:
                    # Next/previous map
                    mods = pygame.key.get_mods()
                    try:
                        if mods & pygame.KMOD_SHIFT:
                            # Shift+G: Previous map
                            map_path = official_map_tester.prev_map()
                        else:
                            # G: Next map
                            map_path = official_map_tester.next_map()

                        env.nplay_headless.load_map(map_path)

                        # Pass map name for proper cache keying
                        map_name = official_map_tester.get_current_map_name()
                        manual_reset(env, map_name=map_name)

                        # Get initial observation
                        initial_obs = env._get_observation()
                        observation = env._process_observation(initial_obs)

                        print(f"Switched map: {official_map_tester.get_info_string()}")
                    except Exception as e:
                        print(f"Error switching map: {e}")
                        import traceback

                        traceback.print_exc()

                if event.key == pygame.K_k and generator_tester is not None:
                    # Next/previous category
                    mods = pygame.key.get_mods()
                    try:
                        if mods & pygame.KMOD_SHIFT:
                            # Shift+K: Previous category
                            category_name = generator_tester.prev_category()
                        else:
                            # K: Next category
                            category_name = generator_tester.next_category()

                        # Generate new map from first generator in new category
                        new_map = generator_tester.generate_map()
                        env.nplay_headless.load_map_from_map_data(new_map.map_data())

                        # Manually reset physics without reloading map
                        manual_reset(env)
                        initial_obs = env._get_observation()
                        observation = env._process_observation(initial_obs)

                        print(
                            f"Switched category: {generator_tester.get_info_string()}"
                        )

                        if show_ascii_on_reset:
                            print("\nASCII Visualization:")
                            print(new_map.to_ascii(show_coords=False))
                            print()
                    except Exception as e:
                        print(f"Error switching category: {e}")

                if event.key == pygame.K_v and generator_tester is not None:
                    # Toggle ASCII visualization on reset
                    show_ascii_on_reset = not show_ascii_on_reset
                    print(
                        f"ASCII visualization on reset: {'ON' if show_ascii_on_reset else 'OFF'}"
                    )

                if event.key == pygame.K_l and generator_tester is not None:
                    # List all generators in current category
                    generator_tester.list_generators()

                # Number keys 1-9 to jump to specific generator
                if generator_tester is not None:
                    number_key_map = {
                        pygame.K_1: 0,
                        pygame.K_2: 1,
                        pygame.K_3: 2,
                        pygame.K_4: 3,
                        pygame.K_5: 4,
                        pygame.K_6: 5,
                        pygame.K_7: 6,
                        pygame.K_8: 7,
                        pygame.K_9: 8,
                    }
                    if event.key in number_key_map:
                        target_idx = number_key_map[event.key]
                        if generator_tester.jump_to_generator(target_idx):
                            try:
                                # Generate new map with selected generator
                                new_map = generator_tester.generate_map()
                                env.nplay_headless.load_map_from_map_data(
                                    new_map.map_data()
                                )

                                # Manually reset physics without reloading map
                                manual_reset(env)

                                initial_obs = env._get_observation()
                                observation = env._process_observation(initial_obs)

                                print(
                                    f"Jumped to generator: {generator_tester.get_info_string()}"
                                )

                                if show_ascii_on_reset:
                                    print("\nASCII Visualization:")
                                    print(new_map.to_ascii(show_coords=False))
                                    print()
                            except Exception as e:
                                print(f"Error jumping to generator: {e}")
                        else:
                            print(
                                f"Generator #{target_idx + 1} does not exist in current category"
                            )

                # Number keys 1-9 to jump to specific map (for official maps testing)
                if official_map_tester is not None:
                    number_key_map = {
                        pygame.K_1: 0,
                        pygame.K_2: 1,
                        pygame.K_3: 2,
                        pygame.K_4: 3,
                        pygame.K_5: 4,
                        pygame.K_6: 5,
                        pygame.K_7: 6,
                        pygame.K_8: 7,
                        pygame.K_9: 8,
                    }
                    if event.key in number_key_map:
                        target_idx = number_key_map[event.key]
                        if official_map_tester.jump_to_map(target_idx):
                            try:
                                map_path = official_map_tester.get_current_map_path()
                                env.nplay_headless.load_map(map_path)

                                # Pass map name for proper cache keying
                                map_name = official_map_tester.get_current_map_name()
                                manual_reset(env, map_name=map_name)

                                initial_obs = env._get_observation()
                                observation = env._process_observation(initial_obs)

                                print(
                                    f"Jumped to map: {official_map_tester.get_info_string()}"
                                )
                            except Exception as e:
                                print(f"Error jumping to map: {e}")
                                import traceback

                                traceback.print_exc()
                        else:
                            print(
                                f"Map #{target_idx + 1} does not exist (only {len(official_map_tester.map_files)} maps available)"
                            )

                # Path-aware reward shaping controls
                if path_aware_system is not None:
                    if event.key == pygame.K_a:
                        # Toggle adjacency graph visualization
                        adjacency_graph_debug_enabled = (
                            not adjacency_graph_debug_enabled
                        )
                        print(
                            f"Adjacency graph visualization: {'ON' if adjacency_graph_debug_enabled else 'OFF'}"
                        )

                    if event.key == pygame.K_b and recorder is None:
                        # Toggle blocked entities display (only if not in recording mode)
                        blocked_entities_debug_enabled = (
                            not blocked_entities_debug_enabled
                        )
                        print(
                            f"Blocked entities display: {'ON' if blocked_entities_debug_enabled else 'OFF'}"
                        )

                # Value function (PBRS cost-to-go) overlay
                if event.key == pygame.K_f:
                    value_function_debug_enabled = not value_function_debug_enabled
                    if value_function_debug_enabled:
                        try:
                            level_name = getattr(
                                env.map_loader, "current_map_name", None
                            )
                            if level_name is None and args.map:
                                from pathlib import Path

                                level_name = Path(args.map).stem
                            if level_name:
                                if level_name not in value_function_pbrs_cache:
                                    from nclone.value_function.two_phase_pbrs import (
                                        create_two_phase_pbrs,
                                    )

                                    pbrs = create_two_phase_pbrs(level_name)
                                    value_function_pbrs_cache[level_name] = pbrs
                                env.set_value_function_pbrs(
                                    value_function_pbrs_cache[level_name]
                                )
                                print(
                                    f"Value function overlay: ON (level={level_name})"
                                )
                            else:
                                print("Cannot determine level name for value function")
                                value_function_debug_enabled = False
                        except FileNotFoundError as e:
                            print(
                                f"Value function data not found: {e}. "
                                "Run: python -m nclone.value_function.precompute_cli "
                                "--map-path <level.map>"
                            )
                            value_function_debug_enabled = False
                        except Exception as e:
                            print(f"Failed to load value function: {e}")
                            import traceback

                            traceback.print_exc()
                            value_function_debug_enabled = False
                    else:
                        env.set_value_function_pbrs(None)
                        print("Value function overlay: OFF")
                    env.set_value_function_debug_enabled(value_function_debug_enabled)

                # Analytical reachability overlay
                if event.key == pygame.K_y:
                    reachability_overlay_enabled = not reachability_overlay_enabled
                    try:
                        env.set_reachability_overlay_enabled(
                            reachability_overlay_enabled
                        )
                        print(
                            f"Reachability overlay: {'ON' if reachability_overlay_enabled else 'OFF'}"
                        )
                    except Exception as e:
                        print(f"Failed to toggle reachability overlay: {e}")
                        import traceback

                        traceback.print_exc()
                        reachability_overlay_enabled = False

                # PBRS path overlay (analytical reachability shortest path)
                if event.key == pygame.K_p:
                    pbrs_path_overlay_enabled = not pbrs_path_overlay_enabled
                    try:
                        # Build reachability PBRS if not yet built
                        if pbrs_path_overlay_enabled:
                            pbrs = getattr(env, "_current_reachability_pbrs", None)
                            if pbrs is None:
                                # Ensure graph is built (e.g. after reset)
                                if hasattr(env, "_update_graph_from_env_state"):
                                    env._update_graph_from_env_state()
                                graph_data = getattr(env, "current_graph_data", None)
                                if graph_data:
                                    print(
                                        "Building reachability PBRS for path overlay..."
                                    )
                                    from nclone.analytical_physics.reachability_builder import (
                                        build_reachability_pbrs,
                                    )

                                    level_data = env.level_data
                                    switch_pos = (
                                        env.nplay_headless.exit_switch_position()
                                    )
                                    exit_pos = env.nplay_headless.exit_door_position()
                                    pbrs = build_reachability_pbrs(
                                        graph_data=graph_data,
                                        level_data=level_data,
                                        switch_pos=switch_pos,
                                        exit_pos=exit_pos,
                                    )
                                    env._current_reachability_pbrs = pbrs
                                    print(
                                        f"Built reachability PBRS: "
                                        f"{pbrs.graph.n_points} pts, "
                                        f"{len(pbrs.graph.edges_src)} edges"
                                    )
                                else:
                                    print(
                                        "No graph data available. Reset the env or press R first."
                                    )
                                    pbrs_path_overlay_enabled = False

                        env.set_pbrs_path_overlay_enabled(pbrs_path_overlay_enabled)
                        print(
                            f"PBRS path overlay: {'ON' if pbrs_path_overlay_enabled else 'OFF'}"
                        )
                    except Exception as e:
                        print(f"Failed to toggle PBRS path overlay: {e}")
                        import traceback

                        traceback.print_exc()
                        pbrs_path_overlay_enabled = False

                # Route PBRS overlay (surface heatmap + all arcs + planned route)
                if event.key == pygame.K_o:
                    route_overlay_enabled = not route_overlay_enabled
                    if route_overlay_enabled:
                        try:
                            level_name = getattr(
                                env.map_loader, "current_map_name", None
                            )
                            if level_name is None and args.map:
                                from pathlib import Path
                                level_name = Path(args.map).stem
                            if level_name:
                                from nclone.value_function.route_pbrs import (
                                    create_route_pbrs,
                                )
                                route_pbrs_instance = create_route_pbrs(
                                    level_name, "value_functions"
                                )
                                if route_pbrs_instance is None:
                                    print(
                                        f"Route PBRS not available for '{level_name}'. "
                                        "Run precompute_cli.py with --map-path to generate it."
                                    )
                                    route_overlay_enabled = False
                                else:
                                    ninja_pos = env.nplay_headless.ninja_position()
                                    ninja_vel = env.nplay_headless.ninja_velocity()
                                    route_pbrs_instance.reset(
                                        ninja_pos[0], ninja_pos[1],
                                        ninja_vel[0], ninja_vel[1],
                                        contact_state=None,
                                    )
                                    env.set_route_pbrs(route_pbrs_instance)
                                    env.set_route_overlay_enabled(True)
                                    print(f"Route overlay: ON (level={level_name})")
                            else:
                                print("Cannot determine level name for route overlay.")
                                route_overlay_enabled = False
                        except Exception as e:
                            print(f"Failed to load route overlay: {e}")
                            import traceback
                            traceback.print_exc()
                            route_overlay_enabled = False
                    else:
                        env.set_route_overlay_enabled(False)
                        print("Route overlay: OFF")

                # Demo checkpoint visualization controls
                if demo_checkpoint_system is not None:
                    if event.key == pygame.K_q:
                        # Toggle demo checkpoint visualization
                        demo_checkpoints_debug_enabled = (
                            not demo_checkpoints_debug_enabled
                        )

                        if demo_checkpoints_debug_enabled:
                            # Load checkpoints for current level when toggled ON
                            try:
                                # Get current level name from environment
                                level_name = getattr(
                                    env.map_loader, "current_map_name", None
                                )

                                # Fallback to custom map filename if available
                                if level_name is None and args.map:
                                    from pathlib import Path

                                    level_name = Path(args.map).stem

                                if level_name is None:
                                    print(
                                        "Cannot determine level name for demo checkpoint loading"
                                    )
                                    demo_checkpoints_debug_enabled = False
                                else:
                                    # Check if already cached for this level
                                    if (
                                        demo_checkpoint_system["cached_level_name"]
                                        != level_name
                                    ):
                                        # Find matching demo files
                                        seeder = demo_checkpoint_system["seeder"]
                                        demo_files = seeder.find_matching_demos(
                                            level_name
                                        )

                                        if not demo_files:
                                            print(
                                                f"No demo replays found matching level: {level_name}"
                                            )
                                            demo_checkpoints_debug_enabled = False
                                        else:
                                            print(
                                                f"Processing {len(demo_files)} demo file(s) for level: {level_name}..."
                                            )

                                            # Process all demos to extract checkpoints
                                            all_checkpoints = []
                                            for demo_file in demo_files:
                                                checkpoints = seeder._extract_positions_from_demo_simple(
                                                    demo_file
                                                )
                                                all_checkpoints.extend(checkpoints)

                                            # Cache the checkpoints for this level
                                            demo_checkpoint_system["checkpoint_cache"][
                                                level_name
                                            ] = all_checkpoints
                                            demo_checkpoint_system[
                                                "cached_level_name"
                                            ] = level_name

                                            # Debug: Print sample checkpoint data
                                            if all_checkpoints:
                                                sample = all_checkpoints[0]
                                                print(
                                                    f"✅ Demo checkpoint visualization: ON ({len(all_checkpoints)} checkpoints loaded)"
                                                )
                                                print(
                                                    f"   Sample checkpoint: position={sample.get('position')}, "
                                                    f"cell={sample.get('cell')}, reward={sample.get('cumulative_reward', 0):.3f}"
                                                )
                                            else:
                                                print(
                                                    "✅ Demo checkpoint visualization: ON (0 checkpoints loaded)"
                                                )
                                    else:
                                        # Use cached checkpoints
                                        cached_count = len(
                                            demo_checkpoint_system[
                                                "checkpoint_cache"
                                            ].get(level_name, [])
                                        )
                                        print(
                                            f"✅ Demo checkpoint visualization: ON ({cached_count} checkpoints from cache)"
                                        )

                            except Exception as e:
                                print(f"Error loading demo checkpoints: {e}")
                                import traceback

                                traceback.print_exc()
                                demo_checkpoints_debug_enabled = False
                        else:
                            print("Demo checkpoint visualization: OFF")

    else:  # Minimal event handling for headless
        for event in pygame.event.get(pygame.QUIT):  # only process QUIT events
            if event.type == pygame.QUIT:
                running = False

    # Map keyboard inputs to environment actions (arrow keys only)
    action = 0  # Default to NOOP
    if not args.headless:  # Only process keyboard inputs if not in headless mode
        if args.step_on_input:
            # Step-on-input: use keys held (holding = repeated steps, release = stop)
            # Z/X = dedicated Jump+Left / Jump+Right | Period = dedicated NOOP
            keys = pygame.key.get_pressed()
            if keys[pygame.K_z]:
                action = 4  # Jump + Left (Z)
            elif keys[pygame.K_x]:
                action = 5  # Jump + Right (X)
            elif keys[pygame.K_PERIOD]:
                action = 0  # NOOP (.) - explicit advance with no input
            elif keys[pygame.K_UP]:
                if keys[pygame.K_LEFT]:
                    action = 4  # Jump + Left
                elif keys[pygame.K_RIGHT]:
                    action = 5  # Jump + Right
                else:
                    action = 3  # Jump only
            else:
                if keys[pygame.K_LEFT]:
                    action = 1  # Left
                elif keys[pygame.K_RIGHT]:
                    action = 2  # Right
        elif args.discrete_actions:
            # Discrete action mode: only register actions on key press (not hold)
            if pygame.K_UP in keys_just_pressed:
                if pygame.K_LEFT in keys_just_pressed:
                    action = 4  # Jump + Left
                elif pygame.K_RIGHT in keys_just_pressed:
                    action = 5  # Jump + Right
                else:
                    action = 3  # Jump only
            else:
                if pygame.K_LEFT in keys_just_pressed:
                    action = 1  # Left
                elif pygame.K_RIGHT in keys_just_pressed:
                    action = 2  # Right
        else:
            # Continuous action mode: register actions while keys are held
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                if keys[pygame.K_LEFT]:
                    action = 4  # Jump + Left
                elif keys[pygame.K_RIGHT]:
                    action = 5  # Jump + Right
                else:
                    action = 3  # Jump only
            else:
                if keys[pygame.K_LEFT]:
                    action = 1  # Left
                elif keys[pygame.K_RIGHT]:
                    action = 2  # Right
    else:
        # In headless mode, we can choose to send a default action or no action
        # For now, let's send NOOP. This part could be modified if automated
        # actions are desired in headless testing.
        action = 0

    explicit_noop = (
        args.step_on_input
        and not args.headless
        and pygame.key.get_pressed()[pygame.K_PERIOD]
    )

    # Record action if recording is active (only when we actually step)
    if recorder is not None and recorder.is_recording:
        if not args.step_on_input or action != 0 or explicit_noop:
            recorder.record_action(action)

    # Pass input sequence overlay data to env (step-on-input mode)
    if args.step_on_input and not args.headless:
        env.set_input_sequence_overlay(input_sequence, enabled=True)
    elif hasattr(env, "set_input_sequence_overlay"):
        env.set_input_sequence_overlay([], enabled=False)

    # Update path-aware visualization if enabled
    if path_aware_system is not None:
        # Get current level data
        level_data = env.level_data

        # Extract current switch states directly from simulator (authoritative source)
        # LevelData.__eq__ doesn't check switch states, and level_data.entities may be stale,
        # so we read directly from the simulator to get real-time switch states
        current_switch_states = {}
        if hasattr(env, "nplay_headless") and hasattr(env.nplay_headless, "sim"):
            sim = env.nplay_headless.sim
            if hasattr(sim, "entity_dic"):
                # Exit switches (entity_dic key 3 contains both EntityExitSwitch and EntityExit)
                if 3 in sim.entity_dic:
                    exit_entities = sim.entity_dic[3]
                    for i, entity in enumerate(exit_entities):
                        # Check if this is an EntityExitSwitch
                        if type(entity).__name__ == "EntityExitSwitch":
                            # For exit switches, active=False means collected/activated
                            switch_id = f"exit_switch_{i}"
                            current_switch_states[switch_id] = not getattr(
                                entity, "active", True
                            )

                # Locked door switches (entity_dic key 7)
                if 7 in sim.entity_dic:
                    locked_door_switches = sim.entity_dic[7]
                    for i, switch in enumerate(locked_door_switches):
                        switch_id = f"locked_door_switch_{i}"
                        # active=False means collected/activated
                        current_switch_states[switch_id] = not getattr(
                            switch, "active", True
                        )

        # Check if switch states have changed (critical for cache invalidation)
        switch_states_changed = (
            path_aware_system.get("cached_switch_states") is None
            or path_aware_system["cached_switch_states"] != current_switch_states
        )

        # Build/update graph if level has changed (uses LevelData.__eq__ comparison)
        # OR if switch states have changed (LevelData.__eq__ doesn't check switch states)
        if (
            path_aware_system["current_graph"] is None
            or path_aware_system.get("cached_level_data") is None
            or path_aware_system["cached_level_data"] != level_data
            or switch_states_changed
        ):
            # If switch states changed, clear pathfinding caches before rebuilding
            if switch_states_changed:
                # Clear path calculator cache (forces rebuild with new goals)
                path_aware_system["path_calculator"].clear_cache()

                # Clear debug overlay renderer pathfinding cache if accessible
                if (
                    hasattr(env, "nplay_headless")
                    and hasattr(env.nplay_headless, "sim_renderer")
                    and hasattr(
                        env.nplay_headless.sim_renderer, "debug_overlay_renderer"
                    )
                ):
                    debug_renderer = (
                        env.nplay_headless.sim_renderer.debug_overlay_renderer
                    )
                    if hasattr(debug_renderer, "clear_pathfinding_cache"):
                        debug_renderer.clear_pathfinding_cache()

            # CRITICAL: Update level_data entities with fresh simulator data before building cache/graph
            # level_data.entities may be stale, so we need to refresh them to get current switch states
            if hasattr(env, "entity_extractor"):
                fresh_entities = env.entity_extractor.extract_graph_entities()
                # Create a copy of level_data with updated entities
                from nclone.graph.level_data import LevelData

                updated_level_data = LevelData(
                    start_position=level_data.start_position,
                    tiles=level_data.tiles,
                    entities=fresh_entities,
                    level_id=level_data.level_id,
                    metadata=level_data.metadata,
                    entity_start_positions=level_data.entity_start_positions,
                )
                level_data = updated_level_data

            # Rebuild graph for new level
            # Convert LevelData to dict format for graph_builder
            level_data_dict = (
                level_data.to_dict() if hasattr(level_data, "to_dict") else level_data
            )
            # Build static adjacency graph (reachability computed dynamically during visualization)
            path_aware_system["current_graph"] = path_aware_system[
                "graph_builder"
            ].build_graph(level_data_dict)
            # Build level cache for path distances
            if (
                path_aware_system["current_graph"]
                and "adjacency" in path_aware_system["current_graph"]
            ):
                adjacency = path_aware_system["current_graph"]["adjacency"]
                base_adjacency = path_aware_system["current_graph"].get(
                    "base_adjacency", adjacency
                )
                # PERFORMANCE OPTIMIZATION: Pass graph_data containing physics cache
                path_aware_system["path_calculator"].build_level_cache(
                    level_data,
                    adjacency,
                    base_adjacency,
                    path_aware_system["current_graph"],
                )
            # Cache the level data and switch states for comparison on next frame
            path_aware_system["cached_level_data"] = level_data
            path_aware_system["cached_switch_states"] = current_switch_states
        elif path_aware_system.get("cached_switch_states") is None:
            # First frame: cache switch states even if level_data comparison passed
            path_aware_system["cached_switch_states"] = current_switch_states

        # Set debug flags on environment
        env.set_adjacency_graph_debug_enabled(adjacency_graph_debug_enabled)
        env.set_blocked_entities_debug_enabled(blocked_entities_debug_enabled)
        env.set_show_paths_to_goals(show_paths_to_goals)

        # Get cached paths from reward calculator (ensures visualization matches what's being rewarded)
        cached_paths = None
        if hasattr(env, "reward_calculator") and env.reward_calculator:
            cached_paths = env.reward_calculator.get_cached_paths()

        # Pass graph data and cached paths to environment for visualization
        env.set_path_aware_data(
            graph_data=path_aware_system["current_graph"],
            entity_mask=path_aware_system.get("current_entity_mask"),
            level_data=level_data,
            cached_paths=cached_paths,
        )
    elif path_aware_system is not None:
        # Turn off all path-aware debug flags when none are enabled
        env.set_adjacency_graph_debug_enabled(False)
        env.set_blocked_entities_debug_enabled(False)
        env.set_show_paths_to_goals(False)

    # Update value function overlay if enabled
    env.set_value_function_debug_enabled(value_function_debug_enabled)
    if value_function_debug_enabled:
        level_name = getattr(env.map_loader, "current_map_name", None)
        if level_name is None and args.map:
            from pathlib import Path

            level_name = Path(args.map).stem
        if level_name and level_name in value_function_pbrs_cache:
            env.set_value_function_pbrs(value_function_pbrs_cache[level_name])
        elif level_name and level_name not in value_function_pbrs_cache:
            try:
                from nclone.value_function.two_phase_pbrs import (
                    create_two_phase_pbrs,
                )

                pbrs = create_two_phase_pbrs(level_name)
                value_function_pbrs_cache[level_name] = pbrs
                env.set_value_function_pbrs(pbrs)
            except FileNotFoundError:
                env.set_value_function_pbrs(None)
            except Exception:
                env.set_value_function_pbrs(None)
        else:
            env.set_value_function_pbrs(None)
    else:
        env.set_value_function_pbrs(None)

    # Rebuild PBRS path overlay if enabled (graph changes on level change)
    if pbrs_path_overlay_enabled:
        try:
            from nclone.analytical_physics.reachability_builder import (
                build_reachability_pbrs,
            )

            graph_data = getattr(env, "current_graph_data", None)
            if graph_data:
                level_data = env.level_data
                switch_pos = env.nplay_headless.exit_switch_position()
                exit_pos = env.nplay_headless.exit_door_position()
                pbrs = build_reachability_pbrs(
                    graph_data=graph_data,
                    level_data=level_data,
                    switch_pos=switch_pos,
                    exit_pos=exit_pos,
                )
                env._current_reachability_pbrs = pbrs
        except Exception:
            pass
    env.set_pbrs_path_overlay_enabled(pbrs_path_overlay_enabled)

    # Per-frame route PBRS update (arc advancement / replanning)
    if route_overlay_enabled and route_pbrs_instance is not None:
        try:
            ninja_pos = env.nplay_headless.ninja_position()
            ninja_vel = env.nplay_headless.ninja_velocity()
            switch_activated = env.nplay_headless.exit_switch_activated()
            state_dict = {
                "x": ninja_pos[0],
                "y": ninja_pos[1],
                "vx": ninja_vel[0],
                "vy": ninja_vel[1],
                "switch_activated": switch_activated,
                "on_surface": True,
            }
            route_pbrs_instance.update(state_dict)
        except Exception:
            pass
    env.set_route_overlay_enabled(route_overlay_enabled)

    # Update demo checkpoint visualization if enabled
    if demo_checkpoint_system is not None:
        env.set_demo_checkpoints_debug_enabled(demo_checkpoints_debug_enabled)

        if demo_checkpoints_debug_enabled:
            # Get current level name
            level_name = getattr(env.map_loader, "current_map_name", None)
            if level_name is None and args.map:
                from pathlib import Path

                level_name = Path(args.map).stem

            # Get cached checkpoints for current level
            if level_name:
                checkpoints = demo_checkpoint_system["checkpoint_cache"].get(
                    level_name, []
                )
                env.set_demo_checkpoints_data(checkpoints)

    # Step the environment
    if args.step_on_input:
        if action == 0 and not explicit_noop:
            # No input: render current state without stepping
            hold_start_time = None
            hold_action = 0
            env.render()
            did_step = False
        else:
            # Keys held: 200ms delay before repeat (enables single-frame taps)
            now_ms = time.perf_counter() * 1000
            if hold_start_time is None or hold_action != action:
                # New press or key combo change: step immediately
                hold_start_time = now_ms
                hold_action = action
                observation, reward, terminated, truncated, info = env.step(action)
                input_sequence.append(action)
                did_step = True
            elif (now_ms - hold_start_time) >= HOLD_REPEAT_DELAY_MS:
                # Past delay: repeat every frame
                observation, reward, terminated, truncated, info = env.step(action)
                input_sequence.append(action)
                did_step = True
            else:
                # Within 200ms of initial press: no repeat (single-frame tap)
                env.render()
                did_step = False
    else:
        observation, reward, terminated, truncated, info = env.step(action)
        if args.step_on_input:
            input_sequence.append(action)
        did_step = True

    # Memory profiling snapshot (only when we stepped)
    if (
        did_step
        and memory_profiler is not None
        and frame_counter % memory_profiler.snapshot_interval == 0
    ):
        memory_profiler.take_snapshot(frame_counter, obs=observation)

    # Reset if episode is done (only when we stepped)
    if did_step and (terminated or truncated):
        if args.step_on_input:
            input_sequence.clear()
        # Stop recording if active (episode naturally ended)
        if recorder is not None and recorder.is_recording:
            # Determine if episode was successful
            player_won = env.nplay_headless.ninja_has_won()
            recorder.stop_recording(success=player_won, save=player_won)

        # Auto-advance to next level if in test suite mode with auto-advance enabled
        if args.test_suite and args.auto_advance and test_suite_loader is not None:
            player_won = env.nplay_headless.ninja_has_won()
            result_str = "✓ Success" if player_won else "✗ Failed"
            print(
                f"{result_str} - Level {current_level_idx + 1}: {test_suite_level_ids[current_level_idx]}"
            )

            # Load next level
            current_level_idx = (current_level_idx + 1) % len(test_suite_level_ids)
            level_id = test_suite_level_ids[current_level_idx]
            level = test_suite_loader.get_level(level_id)
            env.nplay_headless.load_map_from_map_data(level["map_data"])
            print(
                f"Loaded level {current_level_idx + 1}/{len(test_suite_level_ids)}: {level_id}"
            )
        elif (
            args.benchmark_dataset
            and args.auto_advance
            and benchmark_loader is not None
        ):
            player_won = env.nplay_headless.ninja_has_won()
            result_str = "✓ Success" if player_won else "✗ Failed"
            print(
                f"{result_str} - benchmark {benchmark_index + 1}/{len(benchmark_loader)}: "
                f"level_id={benchmark_payload.get('level_id') if benchmark_payload else '?'}"
            )
            benchmark_index = (benchmark_index + 1) % len(benchmark_loader)
            benchmark_payload = benchmark_loader.load_payload(benchmark_index)
            env.nplay_headless.load_map_from_map_data(benchmark_payload["map_data"])
            map_name = f"benchmark_{benchmark_payload.get('level_id', benchmark_index)}"
            manual_reset(env, map_name=map_name)
            print(
                f"Auto-advanced to benchmark {benchmark_index + 1}/{len(benchmark_loader)}: "
                f"level_id={benchmark_payload.get('level_id')}"
            )

        # Generate new map in generator testing mode, reload map in official maps mode, otherwise normal reset
        if generator_tester is not None:
            try:
                new_map = generator_tester.generate_map()
                env.nplay_headless.load_map_from_map_data(new_map.map_data())

                # Manually reset physics without reloading map
                manual_reset(env)

                initial_obs = env._get_observation()
                observation = env._process_observation(initial_obs)

                print(
                    f"Episode complete! Generated new map: {generator_tester.get_info_string()}"
                )

                if show_ascii_on_reset:
                    print("\nASCII Visualization:")
                    print(new_map.to_ascii(show_coords=False))
                    print()
            except Exception as e:
                print(f"Error generating new map on episode end: {e}")
                import traceback

                traceback.print_exc()
        elif official_map_tester is not None:
            # Reload current map on episode completion
            try:
                current_map_path = official_map_tester.get_current_map_path()
                env.nplay_headless.load_map(current_map_path)

                # Pass map name for proper cache keying
                map_name = official_map_tester.get_current_map_name()
                manual_reset(env, map_name=map_name)

                initial_obs = env._get_observation()
                observation = env._process_observation(initial_obs)

                print(
                    f"Episode complete! Reloaded map: {official_map_tester.get_info_string()}"
                )
            except Exception as e:
                print(f"Error reloading map on episode end: {e}")
                import traceback

                traceback.print_exc()
        else:
            # Normal episode reset (test suite, benchmark without auto-advance, default map)
            observation, info = env.reset()

    current_time = time.perf_counter()
    if args.log_frametimes:
        frame_time_ms = (current_time - last_time) * 1000
        print("Frametime: {0:.2f} ms".format(frame_time_ms))
    last_time = current_time

    # In headless mode, we don't call clock.tick() as it relies on pygame.display
    if not args.headless and clock:
        clock.tick(60)

    # Human mode: ensure a frame is presented; overlay mechanic-benchmark metadata
    if not args.headless:
        env.render()
        if benchmark_loader is not None and benchmark_payload is not None:
            _draw_benchmark_overlay(
                MechanicBenchmarkLoader.overlay_lines(
                    benchmark_payload,
                    benchmark_index,
                    len(benchmark_loader),
                )
            )

    frame_counter += 1
    if args.profile_frames is not None and frame_counter >= args.profile_frames:
        running = False

profiler.disable()

# Print granular step timing report when profiling was enabled
if (
    args.profile_frames is not None
    and hasattr(env, "profile_enabled")
    and env.profile_enabled
):
    env._print_profiling_report()

# Print cache statistics
print("\n" + "=" * 60)
print("CACHE STATISTICS")
print("=" * 60)

# Door feature cache statistics
if hasattr(env, "door_feature_cache"):
    door_stats = env.door_feature_cache.get_statistics()
    print("\nDoor Feature Cache:")
    print(f"  - Cache built: {door_stats['cache_built']}")
    print(f"  - Grid cells cached: {door_stats['grid_cells']}")
    print(f"  - Doors tracked: {door_stats['n_doors']}")
    print(f"  - Cache hits: {door_stats['cache_hits']}")
    print(f"  - Cache misses: {door_stats['cache_misses']}")
    print(f"  - Hit rate: {door_stats['hit_rate']:.1%}")
    print(f"  - Memory: {door_stats['memory_mb']:.2f} MB")

# Finalize memory profiling if active
if memory_profiler is not None:
    memory_profiler.finalize()
    if len(memory_profiler.snapshots) >= 2:
        memory_profiler.compare_snapshots(0, -1)

# Print recording statistics if recorder was enabled
if recorder is not None:
    recorder.print_statistics()

# Cleanup
pygame.quit()
env.close()

# Process and print profiling stats
with open("profiling_stats.txt", "w") as f:
    ps = pstats.Stats(profiler, stream=f).sort_stats("cumulative")
    ps.print_stats()

print("Profiling stats saved to profiling_stats.txt")
