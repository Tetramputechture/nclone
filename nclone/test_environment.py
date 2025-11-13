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
M - Toggle mine predictor debug overlay
D - Toggle mine death probability debug overlay
T - Toggle terminal velocity death probability debug overlay
W - Toggle action mask debug overlay
U - Toggle reachable walls debug overlay
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

# Clear all caches for new level
from nclone.cache_management import (
    clear_all_caches_for_new_level,
)

from nclone.replay.gameplay_recorder import GameplayRecorder
from nclone.gym_environment.npp_environment import NppEnvironment


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

# Input mode arguments
parser.add_argument(
    "--discrete-actions",
    action="store_true",
    help="Enable discrete action mode - single action per keypress instead of continuous actions while held",
)

args = parser.parse_args()

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
    print("  P - Toggle path to goals visualization (with colored segments)")
    print("      • Cyan: Right | Magenta: Left | Gold: Down | Green: Up")
    print("      • Thicker line = current segment")
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
        env._build_mine_death_lookup_table()
        env._build_door_feature_cache()

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

graph_debug_enabled = False
exploration_debug_enabled = False
grid_debug_enabled = False
tile_rendering_enabled = True
tile_types_debug_enabled = False
mine_predictor_debug_enabled = False
death_probability_debug_enabled = False
terminal_velocity_probability_debug_enabled = False
action_mask_debug_enabled = False
reachable_walls_debug_enabled = False
path_aware_system = None
adjacency_graph_debug_enabled = False
blocked_entities_debug_enabled = False
show_paths_to_goals = False

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


def manual_reset(env: NppEnvironment):
    print("Manual reset called")
    print("Resetting observation processor")
    env.observation_processor.reset()
    print("Resetting reward calculator")
    env.reward_calculator.reset()
    print("Resetting truncation checker")
    env.truncation_checker.reset()
    print("Resetting episode reward")
    env.current_ep_reward = 0
    clear_all_caches_for_new_level(env)
    print("Resetting graph state")
    env._reset_graph_state()
    print("Updating graph from env state")
    env._update_graph_from_env_state()
    print("Clearing reachability cache")
    env._clear_reachability_cache()
    print("Building door feature cache")
    env._build_door_feature_cache()
    print("Building mine death lookup table")
    env._build_mine_death_lookup_table()
    print("Building terminal velocity lookup table")
    env._build_terminal_velocity_lookup_table()
    print("Initializing path guidance predictor")
    env._initialize_path_guidance_predictor()


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
                if event.key == pygame.K_m:
                    # Toggle mine predictor debug overlay
                    mine_predictor_debug_enabled = not mine_predictor_debug_enabled
                    try:
                        env.set_mine_predictor_debug_enabled(
                            mine_predictor_debug_enabled
                        )
                        print(
                            f"Mine predictor debug: {'ON' if mine_predictor_debug_enabled else 'OFF'}"
                        )
                    except Exception as e:
                        print(f"Failed to toggle mine predictor debug: {e}")
                if event.key == pygame.K_d:
                    # Toggle mine death probability debug overlay
                    death_probability_debug_enabled = (
                        not death_probability_debug_enabled
                    )
                    try:
                        env.set_death_probability_debug_enabled(
                            death_probability_debug_enabled
                        )
                        print(
                            f"Mine death probability debug: {'ON' if death_probability_debug_enabled else 'OFF'}"
                        )
                    except Exception as e:
                        print(f"Failed to toggle mine death probability debug: {e}")
                if event.key == pygame.K_t:
                    # Toggle terminal velocity death probability debug overlay
                    terminal_velocity_probability_debug_enabled = (
                        not terminal_velocity_probability_debug_enabled
                    )
                    try:
                        env.set_terminal_velocity_probability_debug_enabled(
                            terminal_velocity_probability_debug_enabled
                        )
                        print(
                            f"Terminal velocity death probability debug: {'ON' if terminal_velocity_probability_debug_enabled else 'OFF'}"
                        )
                    except Exception as e:
                        print(
                            f"Failed to toggle terminal velocity death probability debug: {e}"
                        )
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
                if event.key == pygame.K_u:
                    # Toggle reachable walls debug overlay
                    reachable_walls_debug_enabled = not reachable_walls_debug_enabled
                    try:
                        env.set_reachable_walls_debug_enabled(
                            reachable_walls_debug_enabled
                        )
                        print(
                            f"Reachable walls debug: {'ON' if reachable_walls_debug_enabled else 'OFF'}"
                        )
                    except Exception as e:
                        print(f"Failed to toggle reachable walls debug: {e}")
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

                    if event.key == pygame.K_p:
                        # Toggle path to goals visualization
                        show_paths_to_goals = not show_paths_to_goals
                        print(
                            f"Path to goals visualization: {'ON' if show_paths_to_goals else 'OFF'}"
                        )

                    if event.key == pygame.K_x and not (event.mod & pygame.KMOD_CTRL):
                        # Export path analysis screenshot
                        try:
                            timestamp = int(time.time())
                            filename = f"path_analysis_export_{timestamp}.png"
                            # Get current frame
                            frame = env.render()
                            # handle pygame surface
                            if isinstance(frame, pygame.Surface):
                                # save pygame surface to file and quit
                                pygame.image.save(frame, filename)
                                print(
                                    f"✅ Path analysis screenshot saved to {filename}"
                                )
                            elif frame is not None and isinstance(frame, np.ndarray):
                                image = Image.fromarray(
                                    frame.astype(np.uint8), mode="RGB"
                                )
                                image.save(filename)
                                print(
                                    f"✅ Path analysis screenshot saved to {filename}"
                                )
                            else:
                                print("❌ Failed to export path analysis screenshot")
                        except Exception as e:
                            print(f"Could not export path analysis screenshot: {e}")

    else:  # Minimal event handling for headless
        for event in pygame.event.get(pygame.QUIT):  # only process QUIT events
            if event.type == pygame.QUIT:
                running = False

    # Map keyboard inputs to environment actions (arrow keys only)
    action = 0  # Default to NOOP
    if not args.headless:  # Only process keyboard inputs if not in headless mode
        if args.discrete_actions:
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

    # Record action if recording is active
    if recorder is not None and recorder.is_recording:
        recorder.record_action(action)

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
                path_aware_system["path_calculator"].build_level_cache(
                    level_data, path_aware_system["current_graph"]["adjacency"]
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

        # Pass graph data to environment for visualization
        env.set_path_aware_data(
            graph_data=path_aware_system["current_graph"],
            entity_mask=path_aware_system.get("current_entity_mask"),
            level_data=level_data,
        )
    elif path_aware_system is not None:
        # Turn off all path-aware debug flags when none are enabled
        env.set_adjacency_graph_debug_enabled(False)
        env.set_blocked_entities_debug_enabled(False)
        env.set_show_paths_to_goals(False)

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

# Entity cache statistics
if hasattr(env, "nplay_headless") and hasattr(env.nplay_headless, "entity_cache"):
    entity_cache = env.nplay_headless.entity_cache
    if entity_cache.is_cache_built():
        print("\nEntity Cache:")
        print("  - Cache built: True")
        print(f"  - Total entities: {entity_cache.cache.n_entities}")
        print(f"  - Toggle mines: {entity_cache.cache.n_toggle_mines}")
        memory_kb = (
            entity_cache.cache.positions.nbytes
            + entity_cache.cache.types.nbytes
            + entity_cache.cache.active_states.nbytes
            + entity_cache.cache.mine_states.nbytes
        ) / 1024
        print(f"  - Memory: {memory_kb:.2f} KB")

# Mine death predictor statistics
if (
    hasattr(env, "nplay_headless")
    and hasattr(env.nplay_headless, "sim")
    and hasattr(env.nplay_headless.sim, "ninja")
    and hasattr(env.nplay_headless.sim.ninja, "mine_death_predictor")
    and env.nplay_headless.sim.ninja.mine_death_predictor is not None
):
    predictor = env.nplay_headless.sim.ninja.mine_death_predictor
    stats = predictor.get_stats()
    print("\nMine Death Predictor:")
    print(f"  - Reachable mines: {stats.reachable_mines}")
    print(f"  - Danger zone cells: {stats.danger_zone_cells}")
    print(f"  - Tier 1 queries: {stats.tier1_queries}")
    print(f"  - Tier 2 queries: {stats.tier2_queries}")
    print(f"  - Tier 3 queries: {stats.tier3_queries}")
    total_queries = stats.tier1_queries + stats.tier2_queries + stats.tier3_queries
    if total_queries > 0:
        tier1_rate = stats.tier1_queries / total_queries
        tier2_rate = stats.tier2_queries / total_queries
        tier3_rate = stats.tier3_queries / total_queries
        print(f"  - Tier 1 rate: {tier1_rate:.1%} (fast path)")
        print(f"  - Tier 2 rate: {tier2_rate:.1%} (medium path)")
        print(f"  - Tier 3 rate: {tier3_rate:.1%} (slow path)")

print("=" * 60 + "\n")

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
