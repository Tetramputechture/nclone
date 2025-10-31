"""Configuration and argument parsing for test environment."""

import argparse
import sys
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class TestConfig:
    """Configuration for test environment session."""
    
    # Basic settings
    headless: bool = False
    log_frametimes: bool = False
    profile_frames: Optional[int] = None
    profile_memory: bool = False
    memory_snapshot_interval: int = 100
    custom_map: Optional[str] = None
    
    # Graph visualization
    visualize_graph: bool = False
    standalone_graph: bool = False
    interactive_graph: bool = False
    save_graph: Optional[str] = None
    show_edges: List[str] = None
    export_frame: Optional[str] = None
    
    # Reachability visualization
    visualize_reachability: bool = False
    reachability_from_ninja: bool = False
    show_subgoals: bool = False
    show_frontiers: bool = False
    export_reachability: Optional[str] = None
    
    # Path-aware reward shaping
    test_path_aware: bool = False
    show_path_distances: bool = False
    visualize_adjacency_graph: bool = False
    show_blocked_entities: bool = False
    benchmark_pathfinding: bool = False
    export_path_analysis: Optional[str] = None
    
    # Subgoal visualization
    visualize_subgoals: bool = False
    subgoal_mode: str = "detailed"
    export_subgoals: Optional[str] = None
    
    # Recording
    record: bool = False
    recording_output: str = "datasets/human_replays"
    
    # Test suite
    test_suite: bool = False
    test_dataset_path: str = "datasets/test"
    start_level: int = 0
    auto_advance: bool = False
    
    # Generator testing
    test_generators: bool = False
    generator_category: Optional[str] = None
    generator_seed_start: int = 10000
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.interactive_graph and self.headless:
            raise ValueError(
                "Interactive graph visualization cannot be used in headless mode."
            )
        
        if self.show_edges is None:
            self.show_edges = ["walk", "jump"]


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser for test environment.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description="Test N++ environment.")
    
    # Basic settings
    parser.add_argument(
        "--log-frametimes",
        action="store_true",
        help="Enable frametime logging to stdout."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI)."
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
    
    return parser


def print_config_help(config: TestConfig):
    """Print help information based on configuration settings.
    
    Args:
        config: TestConfig instance to display help for
    """
    print(f"Headless: {config.headless}")
    
    # Generator testing help
    if config.test_generators:
        print("\n" + "=" * 60)
        print("GENERATOR TESTING MODE")
        print("=" * 60)
        if config.generator_category:
            print(f"• Testing category: {config.generator_category}")
        else:
            print("• Testing all categories (cycling)")
        print(f"• Starting seed: {config.generator_seed_start}")
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
    
    # Test suite help
    if config.test_suite:
        print("\n" + "=" * 60)
        print("TEST SUITE VALIDATION MODE")
        print("=" * 60)
        print(f"• Dataset path: {config.test_dataset_path}")
        print(f"• Starting from level index: {config.start_level}")
        if config.auto_advance:
            print("• Auto-advance enabled: next level loads on completion")
        else:
            print("• Manual advance: press 'N' to load next level")
        print("\nControls:")
        print("  R - Reset current level")
        print("  N - Load next level")
        print("  P - Load previous level")
        print("=" * 60 + "\n")
    
    # Graph visualization help
    if config.visualize_graph or config.standalone_graph or config.interactive_graph:
        print("\n" + "=" * 60)
        print("GRAPH VISUALIZATION ACTIVE")
        print("=" * 60)
        if config.visualize_graph:
            print("• Graph overlay enabled on simulator")
        if config.standalone_graph:
            print("• Standalone graph window enabled")
        if config.interactive_graph:
            print("• Interactive graph mode enabled")
        
        print("\nRuntime Controls:")
        print("  V - Toggle graph overlay")
        print("  S - Save graph visualization")
        print("  G/E/C - Toggle debug overlays")
        print("  R - Reset environment")
        
        if config.show_edges:
            print(f"\nEdge types shown: {', '.join(config.show_edges)}")
        
        print("=" * 60 + "\n")
    
    # Reachability visualization help
    if (config.visualize_reachability or config.reachability_from_ninja or 
        config.show_subgoals or config.show_frontiers):
        print("\n" + "=" * 60)
        print("REACHABILITY ANALYSIS ACTIVE")
        print("=" * 60)
        if config.visualize_reachability:
            print("• Reachability analysis overlay enabled")
        if config.reachability_from_ninja:
            print("• Dynamic reachability from ninja position")
        if config.show_subgoals:
            print("• Subgoal and waypoint visualization enabled")
        if config.show_frontiers:
            print("• Exploration frontier visualization enabled")
        
        print("\nRuntime Controls:")
        print("  T - Toggle reachability overlay")
        print("  N - Update reachability from ninja position")
        print("  U - Toggle subgoal visualization")
        print("  F - Toggle frontier visualization")
        print("  X - Export reachability screenshot")
        print("  R - Reset environment")
        
        print("=" * 60 + "\n")
    
    # Subgoal visualization help
    if config.visualize_subgoals or config.export_subgoals:
        print("\n" + "=" * 60)
        print("SUBGOAL VISUALIZATION ACTIVE")
        print("=" * 60)
        if config.visualize_subgoals:
            print("• Subgoal visualization overlay enabled")
            print(f"• Visualization mode: {config.subgoal_mode}")
        if config.export_subgoals:
            print(f"• Export subgoal visualization to: {config.export_subgoals}")
        
        print("\nRuntime Controls:")
        print("  S - Toggle subgoal visualization overlay")
        print("  M - Cycle through visualization modes")
        print("  P - Update subgoal plan from current position")
        print("  O - Export subgoal visualization screenshot")
        print("  R - Reset environment")
        
        print("=" * 60 + "\n")
    
    # Path-aware reward shaping help
    if (config.test_path_aware or config.show_path_distances or 
        config.visualize_adjacency_graph or config.show_blocked_entities or 
        config.benchmark_pathfinding):
        print("\n" + "=" * 60)
        print("PATH-AWARE REWARD SHAPING TESTING")
        print("=" * 60)
        print("Testing precomputed tile connectivity + pathfinding system")
        if config.show_path_distances:
            print("• Path distance display enabled")
        if config.visualize_adjacency_graph:
            print("• Adjacency graph visualization enabled")
        if config.show_blocked_entities:
            print("• Blocked entity highlighting enabled")
        if config.benchmark_pathfinding:
            print("• Performance benchmarking enabled")
        
        print("\nRuntime Controls:")
        print("  P - Toggle path distance display")
        print("  A - Toggle adjacency graph visualization")
        print("  B - Toggle blocked entity highlighting")
        print("  T - Run pathfinding benchmark at current position")
        print("  X - Export path analysis screenshot")
        print("  R - Reset environment")
        
        print("\nPerformance Targets:")
        print("  • Graph build (first call): < 5ms")
        print("  • Graph build (cached): < 0.05ms")
        print("  • Path distance (BFS): 2-3ms")
        print("  • Path distance (A*): 1-2ms")
        print("  • Path distance (cached): < 0.1ms")
        
        print("=" * 60 + "\n")


def parse_arguments() -> TestConfig:
    """Parse command-line arguments and create TestConfig.
    
    Returns:
        TestConfig instance with parsed arguments
        
    Raises:
        SystemExit: If argument validation fails
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create config from arguments
    config = TestConfig(
        headless=args.headless,
        log_frametimes=args.log_frametimes,
        profile_frames=args.profile_frames,
        profile_memory=args.profile_memory,
        memory_snapshot_interval=args.memory_snapshot_interval,
        custom_map=args.map,
        visualize_graph=args.visualize_graph,
        standalone_graph=args.standalone_graph,
        interactive_graph=args.interactive_graph,
        save_graph=args.save_graph,
        show_edges=args.show_edges,
        export_frame=args.export_frame,
        visualize_reachability=args.visualize_reachability,
        reachability_from_ninja=args.reachability_from_ninja,
        show_subgoals=args.show_subgoals,
        show_frontiers=args.show_frontiers,
        export_reachability=args.export_reachability,
        test_path_aware=args.test_path_aware,
        show_path_distances=args.show_path_distances,
        visualize_adjacency_graph=args.visualize_adjacency_graph,
        show_blocked_entities=args.show_blocked_entities,
        benchmark_pathfinding=args.benchmark_pathfinding,
        export_path_analysis=args.export_path_analysis,
        visualize_subgoals=args.visualize_subgoals,
        subgoal_mode=args.subgoal_mode,
        export_subgoals=args.export_subgoals,
        record=args.record,
        recording_output=args.recording_output,
        test_suite=args.test_suite,
        test_dataset_path=args.test_dataset_path,
        start_level=args.start_level,
        auto_advance=args.auto_advance,
        test_generators=args.test_generators,
        generator_category=args.generator_category,
        generator_seed_start=args.generator_seed_start,
    )
    
    # Print help information
    print_config_help(config)
    
    return config
