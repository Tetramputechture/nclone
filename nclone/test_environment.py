"""
N++ Environment Test with Graph Visualization

This script provides a comprehensive testing environment for the N++ simulator
with integrated graph visualization capabilities.

Interactive Controls (during runtime):
-------------------------------------

E - Toggle exploration debug overlay
C - Toggle grid debug overlay
R - Reset environment

Examples:
--------
python test_environment.py --headless --export-frame test_level.png
"""

import pygame
from nclone.gym_environment.npp_environment import NppEnvironment
import argparse
import time
import cProfile
import pstats
import sys
import numpy as np
from PIL import Image

from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
from nclone.graph.reachability.tiered_system import TieredReachabilitySystem
from nclone.graph.reachability.reachability_types import PerformanceTarget

# Removed legacy trajectory calculator import
from nclone.graph.subgoal_planner import SubgoalPlanner
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

args = parser.parse_args()

print(f"Headless: {args.headless}")

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
render_mode = "rgb_array" if args.headless else "human"
debug_overlay_enabled = not args.headless  # Disable overlay in headless mode
env = NppEnvironment(
    render_mode=render_mode,
    enable_debug_overlay=debug_overlay_enabled,
    eval_mode=False,
    seed=42,
    custom_map_path=args.map,
)
env.reset()

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
        # Initialize simplified reachability analyzer
        reachability_analyzer = TieredReachabilitySystem()

        # Initialize subgoal planner and integration
        base_subgoal_planner = SubgoalPlanner()
        subgoal_planner = ReachabilitySubgoalIntegration(base_subgoal_planner)

        # Initialize frontier detector
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
        if hasattr(env, 'debug_overlay_renderer'):
            env.debug_overlay_renderer.set_subgoal_visualization_mode(args.subgoal_mode)
            env.debug_overlay_renderer.set_subgoal_debug_enabled(subgoal_debug_enabled)
        
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
    if args.headless or env.render_mode == "rgb_array":
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
        print("Warning: Frame export requires headless mode or rgb_array render mode")
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
                env.level_data, ninja_pos_int, switch_states, PerformanceTarget.BALANCED
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
                completion_plan = subgoal_planner.subgoal_planner.create_hierarchical_completion_plan(
                    ninja_pos, env.level_data, env.entities if hasattr(env, 'entities') else []
                )
                
                # Initialize default values
                current_subgoals = []
                current_subgoal_plan = None
                reachable_positions = None
                
                if completion_plan:
                    current_subgoals = completion_plan.subgoals
                    current_subgoal_plan = completion_plan
                    print(f"✅ Subgoal completion plan created with {len(current_subgoals)} subgoals")
                else:
                    print("⚠️  No subgoal completion plan created, exporting basic visualization")
                
                # Get reachable positions if reachability analyzer is available
                if reachability_analyzer:
                    ninja_pos_int = (int(ninja_pos[0]), int(ninja_pos[1]))
                    switch_states = {}
                    reachability_state = reachability_analyzer.analyze_reachability(
                        env.level_data, ninja_pos_int, switch_states, PerformanceTarget.BALANCED
                    )
                    reachable_positions = reachability_state.reachable_positions
                
                # Set subgoal data in debug overlay renderer
                debug_renderer = None
                if hasattr(env, 'debug_overlay_renderer'):
                    debug_renderer = env.debug_overlay_renderer
                elif hasattr(env, 'nplay_headless') and hasattr(env.nplay_headless, 'sim_renderer') and hasattr(env.nplay_headless.sim_renderer, 'debug_overlay_renderer'):
                    debug_renderer = env.nplay_headless.sim_renderer.debug_overlay_renderer
                
                if debug_renderer:
                    debug_renderer.set_subgoal_data(
                        current_subgoals, current_subgoal_plan, reachable_positions
                    )
                    debug_renderer.set_subgoal_debug_enabled(True)
                
                # Step the environment once to get a proper frame with subgoal overlay
                observation, reward, terminated, truncated, info = env.step(0)  # NOOP action
                
                # Export using debug overlay renderer
                debug_renderer = None
                if hasattr(env, 'debug_overlay_renderer'):
                    debug_renderer = env.debug_overlay_renderer
                elif hasattr(env, 'nplay_headless') and hasattr(env.nplay_headless, 'sim_renderer') and hasattr(env.nplay_headless.sim_renderer, 'debug_overlay_renderer'):
                    debug_renderer = env.nplay_headless.sim_renderer.debug_overlay_renderer
                
                if debug_renderer:
                    try:
                        success = debug_renderer.export_subgoal_visualization(args.export_subgoals)
                        if success:
                            print(f"✅ Subgoal visualization exported to {args.export_subgoals}")
                            print(f"   - Subgoals identified: {len(current_subgoals)}")
                            if current_subgoal_plan:
                                print(f"   - Execution order: {len(current_subgoal_plan.execution_order)} steps")
                            if reachable_positions:
                                print(f"   - Reachable positions: {len(reachable_positions)}")
                        else:
                            print("❌ Failed to export subgoal visualization")
                    except Exception as e:
                        print(f"❌ Error during subgoal visualization export: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("Warning: Debug overlay renderer not available for export")
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
                    # Reset environment
                    observation, info = env.reset()
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
                                        PerformanceTarget.FAST,
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
                                            PerformanceTarget.FAST,
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
                                            PerformanceTarget.FAST,
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
                    if hasattr(env, 'debug_overlay_renderer'):
                        env.debug_overlay_renderer.set_subgoal_debug_enabled(subgoal_debug_enabled)
                    print(f"Subgoal visualization: {'ON' if subgoal_debug_enabled else 'OFF'}")

                if event.key == pygame.K_m:
                    # Cycle through visualization modes
                    modes = ["basic", "detailed", "reachability"]
                    current_mode = getattr(args, 'subgoal_mode', 'detailed')
                    try:
                        current_index = modes.index(current_mode)
                        next_index = (current_index + 1) % len(modes)
                        args.subgoal_mode = modes[next_index]
                        
                        if hasattr(env, 'debug_overlay_renderer'):
                            env.debug_overlay_renderer.set_subgoal_visualization_mode(args.subgoal_mode)
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
                                    ninja_pos, env.level_data, env.entities if hasattr(env, 'entities') else []
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
                                            env.level_data, ninja_pos_int, switch_states, PerformanceTarget.FAST
                                        )
                                        reachable_positions = reachability_state.reachable_positions
                                    
                                    # Update debug overlay renderer
                                    if hasattr(env, 'debug_overlay_renderer'):
                                        env.debug_overlay_renderer.set_subgoal_data(
                                            current_subgoals, current_subgoal_plan, reachable_positions
                                        )
                                    
                                    print(f"Updated subgoal plan from ninja position ({ninja_pos[0]:.1f}, {ninja_pos[1]:.1f})")
                                    print(f"  - Subgoals identified: {len(current_subgoals)}")
                                    print(f"  - Execution steps: {len(current_subgoal_plan.execution_order)}")
                                else:
                                    print("Could not create subgoal completion plan")
                        except Exception as e:
                            print(f"Could not update subgoal plan: {e}")

                if event.key == pygame.K_o:
                    # Export subgoal visualization screenshot
                    if subgoal_debug_enabled and hasattr(env, 'debug_overlay_renderer'):
                        try:
                            timestamp = int(time.time())
                            filename = f"subgoal_export_{timestamp}.png"
                            success = env.debug_overlay_renderer.export_subgoal_visualization(filename)
                            if success:
                                print(f"✅ Subgoal visualization screenshot saved to {filename}")
                            else:
                                print("❌ Failed to export subgoal visualization screenshot")
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

    # Step the environment
    observation, reward, terminated, truncated, info = env.step(action)

    # Reset if episode is done
    if terminated or truncated:
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

# Cleanup
pygame.quit()
env.close()

# Process and print profiling stats
with open("profiling_stats.txt", "w") as f:
    ps = pstats.Stats(profiler, stream=f).sort_stats("cumulative")
    ps.print_stats()

print("Profiling stats saved to profiling_stats.txt")
