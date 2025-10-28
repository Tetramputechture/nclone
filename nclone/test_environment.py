"""
N++ Environment Test with Graph Visualization

This script provides a comprehensive testing environment for the N++ simulator
with integrated graph visualization capabilities.

Interactive Controls (during runtime):
-------------------------------------

E - Toggle exploration debug overlay
C - Toggle grid debug overlay
I - Toggle tile type overlay
L - Toggle tile rendering
R - Reset environment

Examples:
--------
python test_environment.py --headless --export-frame test_level.png
"""

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

from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.reachability.reachability_system import ReachabilitySystem
from nclone.replay.gameplay_recorder import GameplayRecorder

from nclone.planning import LevelCompletionPlanner
from nclone.graph.reachability.subgoal_integration import ReachabilitySubgoalIntegration
from nclone.graph.reachability.frontier_detector import FrontierDetector


def _get_ninja_position(env):
    """Get current ninja position from environment."""
    if hasattr(env, "nplay_headless") and hasattr(env.nplay_headless, "ninja_position"):
        pos = env.nplay_headless.ninja_position()
        # Handle different return formats
        if isinstance(pos, tuple) and len(pos) == 2:
            return pos
        elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
            return (pos[0], pos[1])
        else:
            # If it's a single value or unexpected format, use fallback
            return (100, 100)
    elif hasattr(env, "sim") and hasattr(env.sim, "ninja"):
        return (env.sim.ninja.xpos, env.sim.ninja.ypos)
    else:
        return (100, 100)  # Fallback


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
    "--show-edges",
    nargs="*",
    choices=["walk", "jump", "fall", "wall_slide", "one_way", "functional"],
    default=["walk", "jump"],
    help="Edge types to visualize.",
)

parser.add_argument(
    "--export-frame",
    type=str,
    default=None,
    help="Export first frame of simulation to specified image file and quit (for AI testing).",
)

# Reachability visualization arguments
parser.add_argument(
    "--visualize-reachability",
    action="store_true",
    help="Enable reachability analysis visualization overlay.",
)
parser.add_argument(
    "--reachability-from-ninja",
    action="store_true",
    help="Show reachability analysis from current ninja position.",
)
parser.add_argument(
    "--show-subgoals",
    action="store_true",
    help="Visualize identified subgoals and strategic waypoints.",
)
parser.add_argument(
    "--show-frontiers",
    action="store_true",
    help="Visualize exploration frontiers for curiosity-driven RL.",
)
parser.add_argument(
    "--export-reachability",
    type=str,
    default=None,
    help="Export frame with reachability analysis to specified image file and quit.",
)

# Subgoal visualization arguments
parser.add_argument(
    "--visualize-subgoals",
    action="store_true",
    help="Enable subgoal visualization overlay.",
)
parser.add_argument(
    "--subgoal-mode",
    type=str,
    choices=["basic", "detailed", "reachability"],
    default="detailed",
    help="Subgoal visualization mode.",
)
parser.add_argument(
    "--export-subgoals",
    type=str,
    default=None,
    help="Export frame with subgoal visualization to specified image file and quit.",
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
    help="Path to test dataset directory (default: datasets/test)",
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

args = parser.parse_args()

print(f"Headless: {args.headless}")

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

# Display help information for reachability visualization
if (
    args.visualize_reachability
    or args.reachability_from_ninja
    or args.show_subgoals
    or args.show_frontiers
):
    print("\n" + "=" * 60)
    print("REACHABILITY ANALYSIS ACTIVE")
    print("=" * 60)
    if args.visualize_reachability:
        print("• Reachability analysis overlay enabled")
    if args.reachability_from_ninja:
        print("• Dynamic reachability from ninja position")
    if args.show_subgoals:
        print("• Subgoal and waypoint visualization enabled")
    if args.show_frontiers:
        print("• Exploration frontier visualization enabled")

    print("\nRuntime Controls:")
    print("  T - Toggle reachability overlay")
    print("  N - Update reachability from ninja position")
    print("  U - Toggle subgoal visualization")
    print("  F - Toggle frontier visualization")
    print("  X - Export reachability screenshot")
    print("  R - Reset environment")

    print("=" * 60 + "\n")

# Display help information for subgoal visualization
if args.visualize_subgoals or args.export_subgoals:
    print("\n" + "=" * 60)
    print("SUBGOAL VISUALIZATION ACTIVE")
    print("=" * 60)
    if args.visualize_subgoals:
        print("• Subgoal visualization overlay enabled")
        print(f"• Visualization mode: {args.subgoal_mode}")
    if args.export_subgoals:
        print(f"• Export subgoal visualization to: {args.export_subgoals}")

    print("\nRuntime Controls:")
    print("  S - Toggle subgoal visualization overlay")
    print("  M - Cycle through visualization modes")
    print("  P - Update subgoal plan from current position")
    print("  O - Export subgoal visualization screenshot")
    print("  R - Reset environment")

    print("=" * 60 + "\n")

if args.interactive_graph and args.headless:
    print("Error: Interactive graph visualization cannot be used in headless mode.")
    sys.exit(1)

# Create environment
render_mode = "grayscale_array" if args.headless else "human"
debug_overlay_enabled = False  # Disable overlay in headless mode

# Create environment configuration with custom map path if provided
if args.map:
    from nclone.gym_environment.config import (
        EnvironmentConfig,
        RenderConfig,
        PBRSConfig,
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
        pbrs=PBRSConfig(enable_pbrs=False),
        graph=GraphConfig(enable_graph_updates=False, debug=False),
        reachability=ReachabilityConfig(enable_reachability=False, debug=False),
    )
    env = create_visual_testing_env(config=config)
    print(f"Loading custom map from: {args.map}")
else:
    from nclone.gym_environment.config import (
        EnvironmentConfig,
        RenderConfig,
        PBRSConfig,
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
        pbrs=PBRSConfig(enable_pbrs=False),
        graph=GraphConfig(enable_graph_updates=False, debug=False),
        reachability=ReachabilityConfig(enable_reachability=False, debug=False),
    )
    env = create_visual_testing_env(config=config)

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

                # Special handling for horizontal generator
                if gen_type == "horizontal":
                    from nclone.map_generation.generate_test_suite_maps import (
                        TestSuiteGenerator,
                    )

                    test_gen = TestSuiteGenerator()
                    map_obj = test_gen._create_minimal_simple_level_horizontal(
                        self.current_seed,
                        8 if preset == "minimal" else self.current_generator_idx,
                        height=1 if preset == "minimal" else None,
                    )
                else:
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

        print("Generator Testing Initialized")
        print(f"  {generator_tester.get_info_string()}")

    except Exception as e:
        print(f"Error initializing generator testing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

# Initialize test suite loader if test suite mode is enabled
test_suite_loader = None
test_suite_level_ids = []
current_level_idx = 0

if args.test_suite:
    try:
        from nclone.evaluation.test_suite_loader import TestSuiteLoader

        test_suite_loader = TestSuiteLoader(args.test_dataset_path)
        test_suite_level_ids = test_suite_loader.get_all_level_ids()
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

# Normal environment initialization - call reset only if not in special modes
if not args.test_generators and not args.test_suite:
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

if (
    args.visualize_graph
    or args.standalone_graph
    or args.interactive_graph
    or args.save_graph
):
    print("Initializing graph visualization system...")

    # Build graph data
    try:
        print("Building graph data...")
        import time

        start_time = time.time()

        graph_builder = HierarchicalGraphBuilder()
        level_data = getattr(env, "level_data", None)
        ninja_pos = (
            env.nplay_headless.ninja_position()
            if hasattr(env, "nplay_headless")
            else (0.0, 0.0)
        )

        if level_data:
            hierarchical_data = graph_builder.build_graph(level_data, ninja_pos)
            graph_data = hierarchical_data.sub_cell_graph

            build_time = time.time() - start_time
            print(
                f"Graph built successfully in {build_time:.2f}s ({graph_data.num_nodes} nodes, {graph_data.num_edges} edges)"
            )
        else:
            print("Warning: No level data available for graph construction")
    except Exception as e:
        print(f"Warning: Could not build graph: {e}")
        import traceback

        traceback.print_exc()

    # Create standalone window if requested
    if args.standalone_graph and not args.headless:
        standalone_window = pygame.display.set_mode((1200, 800))
        pygame.display.set_caption("N++ Graph Visualization")

graph_debug_enabled = False
exploration_debug_enabled = False
grid_debug_enabled = False
tile_rendering_enabled = True  # Tiles are rendered by default
tile_types_debug_enabled = False  # Tile type overlay disabled by default

# Initialize reachability system if requested
reachability_analyzer = None
subgoal_planner = None
frontier_detector = None
reachability_debug_enabled = False
subgoals_debug_enabled = False
frontiers_debug_enabled = False

if (
    args.visualize_reachability
    or args.reachability_from_ninja
    or args.show_subgoals
    or args.show_frontiers
    or args.export_reachability
    or args.visualize_subgoals
    or args.export_subgoals
):
    print("Initializing reachability analysis system...")

    try:
        reachability_analyzer = ReachabilitySystem()

        level_completion_planner = LevelCompletionPlanner()
        subgoal_planner = ReachabilitySubgoalIntegration(level_completion_planner)

        frontier_detector = FrontierDetector()

        # Set initial debug states
        reachability_debug_enabled = args.visualize_reachability
        subgoals_debug_enabled = args.show_subgoals
        frontiers_debug_enabled = args.show_frontiers

        print("✅ Reachability analysis system initialized successfully")

    except Exception as e:
        print(f"Warning: Could not initialize reachability system: {e}")
        import traceback

        traceback.print_exc()

# Initialize subgoal visualization system if requested
subgoal_debug_enabled = False
current_subgoals = []
current_subgoal_plan = None

if args.visualize_subgoals or args.export_subgoals:
    print("Initializing subgoal visualization system...")

    try:
        # Enable subgoal visualization
        subgoal_debug_enabled = args.visualize_subgoals

        # Set visualization mode
        env.set_subgoal_visualization_mode(args.subgoal_mode)
        env.set_subgoal_debug_enabled(subgoal_debug_enabled)

        # If visualization is enabled, initialize some basic subgoal data
        if subgoal_debug_enabled and subgoal_planner:
            ninja_pos = _get_ninja_position(env)
            if ninja_pos:
                print(
                    f"   - Initializing subgoal data from ninja position: ({ninja_pos[0]:.1f}, {ninja_pos[1]:.1f})"
                )

                # Create initial completion plan
                completion_plan = (
                    subgoal_planner.subgoal_planner.create_hierarchical_completion_plan(
                        ninja_pos,
                        env.level_data,
                        env.entities if hasattr(env, "entities") else [],
                    )
                )

                if completion_plan:
                    current_subgoals = completion_plan.subgoals
                    current_subgoal_plan = completion_plan

                    # Get reachable positions if available
                    reachable_positions = None
                    if reachability_analyzer:
                        ninja_pos_int = (int(ninja_pos[0]), int(ninja_pos[1]))
                        switch_states = {}
                        reachability_state = reachability_analyzer.analyze_reachability(
                            env.level_data,
                            ninja_pos_int,
                            switch_states,
                        )
                        reachable_positions = reachability_state.reachable_positions

                    # Set the data in the debug overlay renderer
                    env.set_subgoal_data(
                        current_subgoals, current_subgoal_plan, reachable_positions
                    )

                    print(f"   - Initial subgoals identified: {len(current_subgoals)}")
                    if reachable_positions:
                        print(f"   - Reachable positions: {len(reachable_positions)}")
                else:
                    print("   - No initial completion plan could be created")

        print("✅ Subgoal visualization system initialized successfully")
        print(f"   - Mode: {args.subgoal_mode}")
        print(f"   - Overlay enabled: {subgoal_debug_enabled}")

    except Exception as e:
        print(f"Warning: Could not initialize subgoal visualization system: {e}")
        import traceback

        traceback.print_exc()

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

# Handle reachability export if requested
if args.export_reachability and reachability_analyzer:
    print(f"Exporting reachability analysis to {args.export_reachability}...")

    try:
        # Get current ninja position
        ninja_pos = _get_ninja_position(env)
        if ninja_pos:
            ninja_row, ninja_col = ninja_pos

            # Perform reachability analysis from ninja position
            # Convert ninja position to integer coordinates and add switch states
            ninja_pos_int = (int(ninja_pos[0]), int(ninja_pos[1]))
            switch_states = {}  # Empty switch states for export
            reachability_state = reachability_analyzer.analyze_reachability(
                env.level_data, ninja_pos_int, switch_states
            )

            # Get subgoals if requested
            subgoals = []
            if subgoal_planner and args.show_subgoals:
                subgoals = subgoal_planner.enhance_subgoals_with_reachability(
                    env.level_data, reachability_state
                )

            # Get frontiers if requested
            frontiers = []
            if frontier_detector and args.show_frontiers:
                frontiers = frontier_detector.detect_frontiers(
                    env.level_data, reachability_state
                )

            # Enable reachability visualization temporarily
            env.set_reachability_debug_enabled(True)
            env.set_reachability_data(reachability_state, subgoals, frontiers)

            # Step the environment once to get a proper frame with reachability overlay
            observation, reward, terminated, truncated, info = env.step(
                0
            )  # NOOP action

            # Get the rendered frame with reachability overlay
            frame = env.render()
            if frame is not None and isinstance(frame, np.ndarray):
                # Convert to PIL Image and save
                if len(frame.shape) == 3 and frame.shape[2] == 1:
                    # Single channel (grayscale) - squeeze to 2D
                    frame_2d = np.squeeze(frame, axis=2)
                    image = Image.fromarray(frame_2d.astype(np.uint8), mode="L")
                elif len(frame.shape) == 3 and frame.shape[2] == 3:
                    image = Image.fromarray(frame, "RGB")
                elif len(frame.shape) == 3 and frame.shape[2] == 4:
                    image = Image.fromarray(frame, "RGBA")
                elif len(frame.shape) == 2:
                    image = Image.fromarray(frame, "L")
                else:
                    print(f"Warning: Unsupported frame shape {frame.shape}")
                    image = None

                if image:
                    image.save(args.export_reachability)
                    print(
                        f"✅ Reachability analysis exported to {args.export_reachability}"
                    )
                    print(
                        f"   - Reachable positions: {len(reachability_state.reachable_positions)}"
                    )
                    if subgoals:
                        print(f"   - Subgoals identified: {len(subgoals)}")
                    if frontiers:
                        print(f"   - Frontiers detected: {len(frontiers)}")
            else:
                print("Warning: Could not get frame with reachability overlay")
        else:
            print(
                "Warning: Could not determine ninja position for reachability analysis"
            )

    except Exception as e:
        print(f"Error during reachability export: {e}")
        import traceback

        traceback.print_exc()

    # Clean up and exit
    pygame.quit()
    env.close()
    sys.exit(0)

# Handle subgoal export if requested
if args.export_subgoals:
    print(f"Exporting subgoal visualization to {args.export_subgoals}...")

    try:
        # Get current ninja position
        ninja_pos = _get_ninja_position(env)
        if ninja_pos:
            # Generate subgoals using the subgoal planner
            if subgoal_planner:
                # Create hierarchical completion plan
                completion_plan = (
                    subgoal_planner.subgoal_planner.create_hierarchical_completion_plan(
                        ninja_pos,
                        env.level_data,
                        env.entities if hasattr(env, "entities") else [],
                    )
                )

                # Initialize default values
                current_subgoals = []
                current_subgoal_plan = None
                reachable_positions = None

                if completion_plan:
                    current_subgoals = completion_plan.subgoals
                    current_subgoal_plan = completion_plan
                    print(
                        f"✅ Subgoal completion plan created with {len(current_subgoals)} subgoals"
                    )
                else:
                    print(
                        "⚠️  No subgoal completion plan created, exporting basic visualization"
                    )

                # Get reachable positions if reachability analyzer is available
                if reachability_analyzer:
                    ninja_pos_int = (int(ninja_pos[0]), int(ninja_pos[1]))
                    switch_states = {}
                    reachability_state = reachability_analyzer.analyze_reachability(
                        env.level_data,
                        ninja_pos_int,
                        switch_states,
                    )
                    reachable_positions = reachability_state.reachable_positions

                # Set subgoal data in debug overlay renderer
                env.set_subgoal_data(
                    current_subgoals, current_subgoal_plan, reachable_positions
                )
                env.set_subgoal_debug_enabled(True)

                # Step the environment once to get a proper frame with subgoal overlay
                observation, reward, terminated, truncated, info = env.step(
                    0
                )  # NOOP action

                # Export using debug overlay renderer
                try:
                    success = env.export_subgoal_visualization(args.export_subgoals)
                    if success:
                        print(
                            f"✅ Subgoal visualization exported to {args.export_subgoals}"
                        )
                        print(f"   - Subgoals identified: {len(current_subgoals)}")
                        if current_subgoal_plan:
                            print(
                                f"   - Execution order: {len(current_subgoal_plan.execution_order)} steps"
                            )
                        if reachable_positions:
                            print(
                                f"   - Reachable positions: {len(reachable_positions)}"
                            )
                    else:
                        print("❌ Failed to export subgoal visualization")
                except Exception as e:
                    print(f"❌ Error during subgoal visualization export: {e}")
                    import traceback

                    traceback.print_exc()
            else:
                print("Warning: Subgoal planner not initialized")
        else:
            print("Warning: Could not determine ninja position for subgoal analysis")

    except Exception as e:
        print(f"Error during subgoal export: {e}")
        import traceback

        traceback.print_exc()

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

# Main game loop
# Wrap the game loop with profiler.enable() and profiler.disable()
profiler.enable()
frame_counter = 0  # Initialize frame counter
while running:
    # Handle pygame events
    if not args.headless:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Check if episode was successful before reset (for recorder)
                    if recorder is not None and recorder.is_recording:
                        # Determine if episode was successful
                        player_won = env.nplay_headless.ninja_has_won()
                        recorder.stop_recording(success=player_won, save=player_won)

                    # Generator testing mode: generate new map from current generator
                    if generator_tester is not None:
                        try:
                            new_map = generator_tester.generate_map()
                            env.nplay_headless.load_map_from_map_data(
                                new_map.map_data()
                            )

                            # Manually reset physics without reloading map
                            env.observation_processor.reset()
                            env.reward_calculator.reset()
                            env.truncation_checker.reset()
                            env.current_ep_reward = 0
                            env.nplay_headless.reset()

                            # Get initial observation
                            initial_obs = env._get_observation()
                            observation = env._process_observation(initial_obs)

                            print(
                                f"Generated new map: {generator_tester.get_info_string()}"
                            )

                            if show_ascii_on_reset:
                                print("\nASCII Visualization:")
                                print(new_map.to_ascii(show_coords=False))
                                print()
                        except Exception as e:
                            print(f"Error generating map: {e}")
                            import traceback

                            traceback.print_exc()
                    else:
                        # Normal reset for non-generator-testing mode
                        observation, info = env.reset()

                if event.key == pygame.K_b and recorder is not None:
                    # Start recording
                    if not recorder.is_recording:
                        # CRITICAL: Reset environment to spawn position on SAME map
                        # Save current map data before resetting
                        current_map_data = list(env.nplay_headless.current_map_data)

                        print("Resetting to spawn position on current map...")

                        # Reset physics and observation processors
                        env.observation_processor.reset()
                        env.reward_calculator.reset()
                        env.truncation_checker.reset()
                        env.current_ep_reward = 0

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
                    # Test suite navigation or recording control
                    if test_suite_loader is not None and test_suite_level_ids:
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
                    elif recorder is not None and recorder.is_recording:
                        # Stop recording without saving
                        recorder.stop_recording(success=False, save=False)
                        print("Recording stopped (not saved)")

                if (
                    event.key == pygame.K_p
                    and test_suite_loader is not None
                    and test_suite_level_ids
                ):
                    # Load previous level in test suite
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

                if event.key == pygame.K_e:
                    # Toggle exploration debug overlay
                    exploration_debug_enabled = not exploration_debug_enabled
                    try:
                        env.set_exploration_debug_enabled(exploration_debug_enabled)
                    except Exception:
                        pass
                if event.key == pygame.K_c:
                    # Toggle grid outline debug overlay
                    grid_debug_enabled = not grid_debug_enabled
                    try:
                        env.set_grid_debug_enabled(grid_debug_enabled)
                    except Exception:
                        pass
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

                        # Manually reset physics without reloading map
                        env.observation_processor.reset()
                        env.reward_calculator.reset()
                        env.truncation_checker.reset()
                        env.current_ep_reward = 0
                        env.nplay_headless.reset()

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
                        env.observation_processor.reset()
                        env.reward_calculator.reset()
                        env.truncation_checker.reset()
                        env.current_ep_reward = 0
                        env.nplay_headless.reset()

                        # Get initial observation
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
                                env.observation_processor.reset()
                                env.reward_calculator.reset()
                                env.truncation_checker.reset()
                                env.current_ep_reward = 0
                                env.nplay_headless.reset()

                                # Get initial observation
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

                # Reachability visualization controls
                if event.key == pygame.K_t:
                    # Toggle reachability overlay
                    if reachability_analyzer:
                        reachability_debug_enabled = not reachability_debug_enabled
                        try:
                            env.set_reachability_debug_enabled(
                                reachability_debug_enabled
                            )
                            print(
                                f"Reachability overlay: {'ON' if reachability_debug_enabled else 'OFF'}"
                            )
                        except Exception as e:
                            print(f"Could not toggle reachability overlay: {e}")

                if event.key == pygame.K_n:
                    # Update reachability from ninja position
                    if reachability_analyzer and reachability_debug_enabled:
                        try:
                            ninja_pos = _get_ninja_position(env)
                            if ninja_pos:
                                ninja_row, ninja_col = ninja_pos
                                ninja_pos_int = (int(ninja_pos[0]), int(ninja_pos[1]))
                                switch_states = {}
                                reachability_state = (
                                    reachability_analyzer.analyze_reachability(
                                        env.level_data,
                                        ninja_pos_int,
                                        switch_states,
                                    )
                                )

                                # Get subgoals and frontiers if enabled
                                subgoals = []
                                frontiers = []
                                if subgoal_planner and subgoals_debug_enabled:
                                    subgoals = subgoal_planner.enhance_subgoals_with_reachability(
                                        env.level_data, reachability_state
                                    )
                                if frontier_detector and frontiers_debug_enabled:
                                    frontiers = frontier_detector.detect_frontiers(
                                        env.level_data, reachability_state
                                    )

                                env.set_reachability_data(
                                    reachability_state, subgoals, frontiers
                                )
                                print(
                                    f"Updated reachability from ninja position ({ninja_row}, {ninja_col})"
                                )
                                print(
                                    f"  - Reachable positions: {len(reachability_state.reachable_positions)}"
                                )
                        except Exception as e:
                            print(f"Could not update reachability: {e}")

                if event.key == pygame.K_u:
                    # Toggle subgoal visualization
                    if subgoal_planner:
                        subgoals_debug_enabled = not subgoals_debug_enabled
                        print(
                            f"Subgoal visualization: {'ON' if subgoals_debug_enabled else 'OFF'}"
                        )

                        # Update visualization if reachability is active
                        if reachability_debug_enabled:
                            try:
                                ninja_pos = _get_ninja_position(env)
                                if ninja_pos:
                                    ninja_row, ninja_col = ninja_pos
                                    ninja_pos_int = (
                                        int(ninja_pos[0]),
                                        int(ninja_pos[1]),
                                    )
                                    switch_states = {}
                                    reachability_state = (
                                        reachability_analyzer.analyze_reachability(
                                            env.level_data,
                                            ninja_pos_int,
                                            switch_states,
                                        )
                                    )
                                    subgoals = (
                                        subgoal_planner.enhance_subgoals_with_reachability(
                                            env.level_data, reachability_state
                                        )
                                        if subgoals_debug_enabled
                                        else []
                                    )
                                    frontiers = (
                                        frontier_detector.detect_frontiers(
                                            env.level_data, reachability_state
                                        )
                                        if frontiers_debug_enabled
                                        else []
                                    )
                                    env.set_reachability_data(
                                        reachability_state, subgoals, frontiers
                                    )
                            except Exception as e:
                                print(f"Could not update subgoal visualization: {e}")

                if event.key == pygame.K_f:
                    # Toggle frontier visualization
                    if frontier_detector:
                        frontiers_debug_enabled = not frontiers_debug_enabled
                        print(
                            f"Frontier visualization: {'ON' if frontiers_debug_enabled else 'OFF'}"
                        )

                        # Update visualization if reachability is active
                        if reachability_debug_enabled:
                            try:
                                ninja_pos = _get_ninja_position(env)
                                if ninja_pos:
                                    ninja_row, ninja_col = ninja_pos
                                    ninja_pos_int = (
                                        int(ninja_pos[0]),
                                        int(ninja_pos[1]),
                                    )
                                    switch_states = {}
                                    reachability_state = (
                                        reachability_analyzer.analyze_reachability(
                                            env.level_data,
                                            ninja_pos_int,
                                            switch_states,
                                        )
                                    )
                                    subgoals = (
                                        subgoal_planner.enhance_subgoals_with_reachability(
                                            env.level_data, reachability_state
                                        )
                                        if subgoals_debug_enabled
                                        else []
                                    )
                                    frontiers = (
                                        frontier_detector.detect_frontiers(
                                            env.level_data, reachability_state
                                        )
                                        if frontiers_debug_enabled
                                        else []
                                    )
                                    env.set_reachability_data(
                                        reachability_state, subgoals, frontiers
                                    )
                            except Exception as e:
                                print(f"Could not update frontier visualization: {e}")

                if event.key == pygame.K_x:
                    # Export reachability screenshot
                    if reachability_analyzer and reachability_debug_enabled:
                        try:
                            timestamp = int(time.time())
                            filename = f"reachability_export_{timestamp}.png"
                            frame = env.render()
                            if frame is not None and isinstance(frame, np.ndarray):
                                if len(frame.shape) == 3 and frame.shape[2] == 3:
                                    image = Image.fromarray(frame, "RGB")
                                    image.save(filename)
                                    print(
                                        f"✅ Reachability screenshot saved to {filename}"
                                    )
                                else:
                                    print(
                                        "Warning: Unsupported frame format for export"
                                    )
                            else:
                                print("Warning: Could not get frame for export")
                        except Exception as e:
                            print(f"Could not export reachability screenshot: {e}")

                # Subgoal visualization controls
                if event.key == pygame.K_s:
                    # Toggle subgoal visualization overlay
                    subgoal_debug_enabled = not subgoal_debug_enabled
                    env.set_subgoal_debug_enabled(subgoal_debug_enabled)
                    print(
                        f"Subgoal visualization: {'ON' if subgoal_debug_enabled else 'OFF'}"
                    )

                if event.key == pygame.K_m:
                    # Cycle through visualization modes
                    modes = ["basic", "detailed", "reachability"]
                    current_mode = getattr(args, "subgoal_mode", "detailed")
                    try:
                        current_index = modes.index(current_mode)
                        next_index = (current_index + 1) % len(modes)
                        args.subgoal_mode = modes[next_index]

                        env.set_subgoal_visualization_mode(args.subgoal_mode)
                        print(f"Subgoal visualization mode: {args.subgoal_mode}")
                    except ValueError:
                        args.subgoal_mode = "detailed"
                        print("Reset subgoal visualization mode to: detailed")

                if event.key == pygame.K_p:
                    # Update subgoal plan from current position
                    if subgoal_planner and subgoal_debug_enabled:
                        try:
                            ninja_pos = _get_ninja_position(env)
                            if ninja_pos:
                                # Create new completion plan
                                completion_plan = subgoal_planner.subgoal_planner.create_hierarchical_completion_plan(
                                    ninja_pos,
                                    env.level_data,
                                    env.entities if hasattr(env, "entities") else [],
                                )

                                if completion_plan:
                                    current_subgoals = completion_plan.subgoals
                                    current_subgoal_plan = completion_plan

                                    # Get reachable positions if available
                                    reachable_positions = None
                                    if reachability_analyzer:
                                        ninja_pos_int = (
                                            int(ninja_pos[0]),
                                            int(ninja_pos[1]),
                                        )
                                        switch_states = {}
                                        reachability_state = (
                                            reachability_analyzer.analyze_reachability(
                                                env.level_data,
                                                ninja_pos_int,
                                                switch_states,
                                            )
                                        )
                                        reachable_positions = (
                                            reachability_state.reachable_positions
                                        )

                                    # Update debug overlay renderer
                                    env.set_subgoal_data(
                                        current_subgoals,
                                        current_subgoal_plan,
                                        reachable_positions,
                                    )

                                    print(
                                        f"Updated subgoal plan from ninja position ({ninja_pos[0]:.1f}, {ninja_pos[1]:.1f})"
                                    )
                                    print(
                                        f"  - Subgoals identified: {len(current_subgoals)}"
                                    )
                                    print(
                                        f"  - Execution steps: {len(current_subgoal_plan.execution_order)}"
                                    )
                                else:
                                    print("Could not create subgoal completion plan")
                        except Exception as e:
                            print(f"Could not update subgoal plan: {e}")

                if event.key == pygame.K_o:
                    # Export subgoal visualization screenshot
                    if subgoal_debug_enabled:
                        try:
                            timestamp = int(time.time())
                            filename = f"subgoal_export_{timestamp}.png"
                            success = env.export_subgoal_visualization(filename)
                            if success:
                                print(
                                    f"✅ Subgoal visualization screenshot saved to {filename}"
                                )
                            else:
                                print(
                                    "❌ Failed to export subgoal visualization screenshot"
                                )
                        except Exception as e:
                            print(f"Could not export subgoal screenshot: {e}")

    else:  # Minimal event handling for headless
        for event in pygame.event.get(pygame.QUIT):  # only process QUIT events
            if event.type == pygame.QUIT:
                running = False

    # Map keyboard inputs to environment actions
    action = 0  # Default to NOOP
    if not args.headless:  # Only process keyboard inputs if not in headless mode
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                action = 4  # Jump + Left
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                action = 5  # Jump + Right
            else:
                action = 3  # Jump only
        else:
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                action = 1  # Left
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                action = 2  # Right
    else:
        # In headless mode, we can choose to send a default action or no action
        # For now, let's send NOOP. This part could be modified if automated
        # actions are desired in headless testing.
        action = 0

    # Record action if recording is active
    if recorder is not None and recorder.is_recording:
        recorder.record_action(action)

    # Step the environment
    observation, reward, terminated, truncated, info = env.step(action)

    # Memory profiling snapshot
    if (
        memory_profiler is not None
        and frame_counter % memory_profiler.snapshot_interval == 0
    ):
        memory_profiler.take_snapshot(frame_counter, obs=observation)

    # Reset if episode is done
    if terminated or truncated:
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

        # Generate new map in generator testing mode, otherwise normal reset
        if generator_tester is not None:
            try:
                new_map = generator_tester.generate_map()
                env.nplay_headless.load_map_from_map_data(new_map.map_data())

                # Manually reset physics without reloading map
                env.observation_processor.reset()
                env.reward_calculator.reset()
                env.truncation_checker.reset()
                env.current_ep_reward = 0
                env.nplay_headless.reset()

                # Get initial observation
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
        else:
            observation, info = env.reset()

    current_time = time.perf_counter()
    if args.log_frametimes:
        frame_time_ms = (current_time - last_time) * 1000
        print("Frametime: {0:.2f} ms".format(frame_time_ms))
    last_time = current_time

    # In headless mode, we don't call clock.tick() as it relies on pygame.display
    if not args.headless and clock:
        clock.tick(60)

    frame_counter += 1
    if args.profile_frames is not None and frame_counter >= args.profile_frames:
        running = False

profiler.disable()

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
