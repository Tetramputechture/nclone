"""
N++ Environment Test with Graph Visualization

This script provides a comprehensive testing environment for the N++ simulator
with integrated graph visualization and pathfinding capabilities.

GRAPH VISUALIZATION USAGE:
==========================

Basic Usage:
-----------
# Run simulator with graph overlay
python test_environment.py --visualize-graph

# Show standalone graph window alongside simulator
python test_environment.py --standalone-graph

# Launch interactive graph exploration tool
python test_environment.py --interactive-graph

# Enable pathfinding with A* algorithm (fast)
python test_environment.py --pathfind --algorithm astar

# Enable pathfinding with Dijkstra algorithm (optimal)
python test_environment.py --pathfind --algorithm dijkstra

# Save graph visualization to image
python test_environment.py --save-graph graph.png

Advanced Usage:
--------------
# Combine multiple features
python test_environment.py --visualize-graph --pathfind --standalone-graph

# Customize edge visualization
python test_environment.py --visualize-graph --show-edges walk jump fall

# Use custom map with graph visualization
python test_environment.py --map custom_level.txt --visualize-graph --pathfind

# Headless mode with graph export
python test_environment.py --headless --save-graph analysis.png

Interactive Controls (during runtime):
-------------------------------------
V - Toggle graph overlay on/off
P - Trigger pathfinding demonstration
S - Save current graph visualization
G - Toggle built-in graph debug overlay
E - Toggle exploration debug overlay
C - Toggle grid debug overlay
R - Reset environment

Algorithm Selection:
-------------------
--algorithm astar    : Use A* for fast pathfinding (default)
--algorithm dijkstra : Use Dijkstra for guaranteed optimal paths

Edge Type Visualization:
-----------------------
--show-edges walk jump fall wall_slide one_way functional
Default: walk jump

Examples:
--------
# Real-time RL training visualization
python test_environment.py --visualize-graph --pathfind --algorithm astar

# Level analysis and validation
python test_environment.py --standalone-graph --pathfind --algorithm dijkstra --show-edges walk jump fall

# Interactive graph exploration
python test_environment.py --interactive-graph

# Export graph for documentation
python test_environment.py --headless --save-graph level_graph.png --show-edges walk jump fall functional
"""

import pygame
from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import (
    BasicLevelNoGold,
)
import argparse
import time
import cProfile
import pstats
import sys

from nclone.graph.visualization import (
    GraphVisualizer,
    VisualizationConfig,
    InteractiveGraphVisualizer,
)
from nclone.graph.pathfinding import PathfindingEngine, PathfindingAlgorithm
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder


def _get_ninja_position(env):
    """Get current ninja position from environment."""
    if hasattr(env, "nplay_headless") and hasattr(env.nplay_headless, "ninja_position"):
        return env.nplay_headless.ninja_position()
    elif hasattr(env, "sim") and hasattr(env.sim, "ninja"):
        return (env.sim.ninja.x, env.sim.ninja.y)
    else:
        return (100, 100)  # Fallback


def _run_hierarchical_pathfinding_demo(env, graph_builder, pathfinding_engine, graph_data, level_data):
    """Run hierarchical pathfinding demonstration."""
    try:
        print("=== Hierarchical Pathfinding Demo ===")
        
        ninja_pos = _get_ninja_position(env)
        print(f"Ninja position: {ninja_pos}")

        # Find ninja node
        from nclone.graph.visualization_api import GraphVisualizationAPI
        api = GraphVisualizationAPI()
        start_node = api._find_closest_node(graph_data, ninja_pos)
        
        if start_node is None:
            print("❌ Could not find ninja node")
            return None
            
        print(f"Ninja node: {start_node}")
        
        # Try hierarchical pathfinding
        if not (hasattr(graph_builder, 'graph_constructor') and 
                hasattr(graph_builder.graph_constructor, 'reachability_analyzer')):
            print("❌ Reachability analyzer not available")
            return None
            
        if not level_data:
            print("❌ No level data available")
            return None
            
        from nclone.graph.subgoal_planner import SubgoalPlanner
        
        # Create subgoal planner
        subgoal_planner = SubgoalPlanner(pathfinding_engine, debug=True)
        
        # Get reachability state
        reachability_state = graph_builder.graph_constructor.reachability_analyzer.analyze_reachability(
            level_data, ninja_pos
        )
        
        # Create and execute plan
        subgoal_plan = subgoal_planner.create_subgoal_plan(
            reachability_state, graph_data, start_node, target_goal_type='exit'
        )

        if subgoal_plan:
            print(f"✅ Created hierarchical plan with {len(subgoal_plan.execution_order)} subgoals")
            paths = subgoal_planner.execute_subgoal_plan(subgoal_plan, start_node)

            if paths:
                print(f"✅ Found {len(paths)} path segments")
                # Use first path for visualization
                return type('PathResult', (), {
                    'path': paths[0],
                    'cost': len(paths[0]) * 10.0,
                    'nodes_explored': len(paths[0])
                })()
            else:
                print("❌ No paths found in hierarchical plan")
        else:
            print("❌ Could not create hierarchical plan")
            
    except Exception as e:
        print(f"❌ Hierarchical pathfinding failed: {e}")
        import traceback
        traceback.print_exc()
        
    return None


# Initialize pygame
pygame.init()
pygame.display.set_caption("N++ Environment Test")

# Argument parser
parser = argparse.ArgumentParser(
    description="Test N++ environment with graph visualization and pathfinding."
)
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
    "--pathfind", action="store_true", help="Enable pathfinding test and visualization."
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
    "--algorithm",
    choices=["astar", "dijkstra"],
    default="astar",
    help="Pathfinding algorithm to use (astar=fast, dijkstra=optimal).",
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

args = parser.parse_args()

print(f"Headless: {args.headless}")

# Display help information for graph visualization
if (
    args.visualize_graph
    or args.standalone_graph
    or args.interactive_graph
    or args.pathfind
):
    print("\n" + "=" * 60)
    print("GRAPH VISUALIZATION ACTIVE")
    print("=" * 60)
    if args.visualize_graph:
        print("• Graph overlay enabled on simulator")
    if args.standalone_graph:
        print("• Standalone graph window enabled")
    if args.interactive_graph:
        print("• Interactive graph mode enabled")
    if args.pathfind:
        print(f"• Pathfinding enabled with {args.algorithm.upper()} algorithm")

    print("\nRuntime Controls:")
    print("  V - Toggle graph overlay")
    print("  P - Trigger pathfinding demo")
    print("  S - Save graph visualization")
    print("  G/E/C - Toggle debug overlays")
    print("  R - Reset environment")

    if args.show_edges:
        print(f"\nEdge types shown: {', '.join(args.show_edges)}")

    print("=" * 60 + "\n")

if args.interactive_graph and args.headless:
    print("Error: Interactive graph visualization cannot be used in headless mode.")
    sys.exit(1)

# Create environment
render_mode = "rgb_array" if args.headless else "human"
debug_overlay_enabled = not args.headless  # Disable overlay in headless mode
env = BasicLevelNoGold(
    render_mode=render_mode,
    enable_frame_stack=False,
    enable_debug_overlay=debug_overlay_enabled,
    eval_mode=False,
    seed=42,
    custom_map_path=args.map,
)
env.reset()

# Initialize graph visualization components
graph_visualizer = None
pathfinding_engine = None
graph_data = None
standalone_window = None
graph_overlay_surface = None

if (
    args.visualize_graph
    or args.standalone_graph
    or args.interactive_graph
    or args.save_graph
):
    print("Initializing graph visualization system...")

    # Create visualization configuration
    config = VisualizationConfig(
        show_walk_edges="walk" in args.show_edges,
        show_jump_edges="jump" in args.show_edges,
        show_fall_edges="fall" in args.show_edges,
        show_wall_slide_edges="wall_slide" in args.show_edges,
        show_one_way_edges="one_way" in args.show_edges,
        show_functional_edges="functional" in args.show_edges,
        show_shortest_path=args.pathfind,
        alpha=0.7,  # Semi-transparent overlay
    )

    graph_visualizer = GraphVisualizer(config)

    # Initialize pathfinding engine if needed
    if args.pathfind or args.visualize_graph or args.standalone_graph:
        try:
            level_data = getattr(env, "level_data", None)
            # Get entities for the pathfinding engine
            entities = []
            if hasattr(env, "_extract_graph_entities"):
                entities = env._extract_graph_entities()

            if level_data:
                pathfinding_engine = PathfindingEngine(level_data, entities)
                print(f"Pathfinding engine initialized with {args.algorithm} algorithm")
            else:
                print("Warning: No level data available for pathfinding engine")
        except Exception as e:
            print(f"Warning: Could not initialize pathfinding engine: {e}")

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

    # Handle interactive graph mode
    if args.interactive_graph and graph_data:
        print("Launching interactive graph visualization...")
        interactive_viz = InteractiveGraphVisualizer(width=1200, height=800)
        interactive_viz.run(graph_data)
        # Interactive mode exits after use
        sys.exit(0)

    # Create standalone window if requested
    if args.standalone_graph and not args.headless:
        standalone_window = pygame.display.set_mode((1200, 800))
        pygame.display.set_caption("N++ Graph Visualization")

graph_debug_enabled = False
exploration_debug_enabled = False
grid_debug_enabled = False
path_updated = False  # Track when pathfinding results change

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
                if event.key == pygame.K_g:
                    # Toggle graph debug overlay
                    graph_debug_enabled = not graph_debug_enabled
                    try:
                        env.set_graph_debug_enabled(graph_debug_enabled)
                    except Exception:
                        pass
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

                # Graph visualization controls
                if event.key == pygame.K_v and graph_visualizer:
                    # Toggle graph overlay visualization
                    args.visualize_graph = not args.visualize_graph
                    print(f"Graph overlay: {'ON' if args.visualize_graph else 'OFF'}")

                if event.key == pygame.K_p and pathfinding_engine and graph_data:
                    # Trigger hierarchical pathfinding demonstration
                    try:
                        print("=== Hierarchical Pathfinding Demo ===")
                        
                        # Get current ninja position
                        if hasattr(env, "nplay_headless") and hasattr(
                            env.nplay_headless, "ninja_position"
                        ):
                            ninja_pos = env.nplay_headless.ninja_position()
                        elif hasattr(env, "sim") and hasattr(env.sim, "ninja"):
                            ninja_pos = (env.sim.ninja.x, env.sim.ninja.y)
                        else:
                            ninja_pos = (100, 100)  # Fallback

                        print(f"Ninja position: {ninja_pos}")

                        # Find ninja node
                        from nclone.graph.visualization_api import GraphVisualizationAPI
                        api = GraphVisualizationAPI()
                        start_node = api._find_closest_node(graph_data, ninja_pos)
                        
                        if start_node is not None:
                            print(f"Ninja node: {start_node}")
                            
                            # Try hierarchical pathfinding to exit
                            try:
                                from nclone.graph.subgoal_planner import SubgoalPlanner
                                
                                # Create subgoal planner
                                subgoal_planner = SubgoalPlanner(pathfinding_engine)
                                
                                # Get reachability state from graph builder
                                if hasattr(graph_builder, 'graph_constructor') and hasattr(graph_builder.graph_constructor, 'reachability_analyzer'):
                                    # Re-run reachability analysis to get subgoals
                                    level_data = getattr(env, "level_data", None)
                                    if level_data:
                                        reachability_state = graph_builder.graph_constructor.reachability_analyzer.analyze_reachability(
                                            level_data, ninja_pos
                                        )
                                        
                                        # Create subgoal plan to reach exit
                                        subgoal_plan = subgoal_planner.create_subgoal_plan(
                                            reachability_state, graph_data, start_node, target_goal_type='exit'
                                        )
                                        
                                        if subgoal_plan:
                                            print(f"✅ Created hierarchical plan with {len(subgoal_plan.execution_order)} subgoals")
                                            
                                            # Execute the plan
                                            paths = subgoal_planner.execute_subgoal_plan(subgoal_plan, start_node)
                                            
                                            if paths:
                                                print(f"✅ Found {len(paths)} path segments")
                                                for i, path in enumerate(paths):
                                                    subgoal = subgoal_plan.subgoals[subgoal_plan.execution_order[i]]
                                                    print(f"  Path {i+1} to {subgoal.goal_type}: {len(path)} nodes")
                                                    
                                                # Store first path for visualization
                                                if paths:
                                                    path_result = type('PathResult', (), {
                                                        'success': True,
                                                        'path': paths[0],
                                                        'cost': len(paths[0]) * 10.0,
                                                        'nodes_explored': len(paths[0])
                                                    })()
                                            else:
                                                print("❌ No paths found in hierarchical plan")
                                                path_result = None
                                        else:
                                            print("❌ Could not create hierarchical plan")
                                            path_result = None
                                    else:
                                        print("❌ No level data available for reachability analysis")
                                        path_result = None
                                else:
                                    print("❌ Reachability analyzer not available")
                                    path_result = None
                                    
                            except Exception as e:
                                print(f"❌ Hierarchical pathfinding failed: {e}")
                                import traceback
                                traceback.print_exc()
                                path_result = None
                                
                            # Fallback to simple pathfinding if hierarchical fails
                            if path_result is None:
                                print("Falling back to simple pathfinding...")
                                
                                # Use mouse position or default goal
                                mouse_pos = pygame.mouse.get_pos()
                                if mouse_pos[0] > 0 and mouse_pos[1] > 0:
                                    screen = pygame.display.get_surface()
                                    if screen:
                                        screen_width = screen.get_width()
                                        screen_height = screen.get_height()
                                        world_x = (mouse_pos[0] / screen_width) * 1056.0
                                        world_y = (mouse_pos[1] / screen_height) * 600.0
                                        goal_pos = (world_x, world_y)
                                    else:
                                        goal_pos = (400, 200)
                                else:
                                    goal_pos = (400, 200)  # Default goal
                                    
                                goal_node = api._find_closest_node(graph_data, goal_pos)
                                
                                if goal_node is not None:
                                    algorithm = (
                                        PathfindingAlgorithm.A_STAR
                                        if args.algorithm == "astar"
                                        else PathfindingAlgorithm.DIJKSTRA
                                    )
                                    path_result = pathfinding_engine.find_shortest_path(
                                        graph_data, start_node, goal_node, algorithm=algorithm
                                    )

                            if path_result.success:
                                print(
                                    f"Path found! Length: {len(path_result.path)} nodes, Cost: {path_result.total_cost:.2f}, Nodes explored: {path_result.nodes_explored}"
                                )
                                # Force recreation of graph overlay to show the path
                                graph_overlay_surface = None
                                path_updated = True
                            else:
                                print("No path found between the selected positions")
                        else:
                            print(
                                f"Could not find valid nodes near positions (start: {start_node}, goal: {goal_node})"
                            )
                    except Exception as e:
                        print(f"Pathfinding error: {e}")
                        import traceback

                        traceback.print_exc()

                if (
                    event.key == pygame.K_s
                    and args.save_graph
                    and graph_visualizer
                    and graph_data
                ):
                    # Save graph visualization
                    try:
                        surface = graph_visualizer.create_standalone_visualization(
                            graph_data, width=1200, height=800
                        )
                        pygame.image.save(surface, args.save_graph)
                        print(f"Graph visualization saved to {args.save_graph}")
                    except Exception as e:
                        print(f"Error saving graph: {e}")
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
        # Pathfinding logic already handled above for the reset case
        if not (
            args.pathfind and not args.headless
        ):  # if pathfinding didn't already reset
            observation, info = env.reset()

    # Graph visualization rendering with performance optimizations
    if graph_visualizer and graph_data and not args.headless:
        try:
            # Render graph overlay on main simulator window
            if args.visualize_graph:
                # Only recreate overlay if it doesn't exist, on first frame, or after pathfinding
                if not graph_overlay_surface or frame_counter == 1 or path_updated:
                    screen = pygame.display.get_surface()
                    if screen:
                        # Get current ninja position for overlay
                        ninja_pos = None
                        if hasattr(env, "nplay_headless") and hasattr(
                            env.nplay_headless, "ninja_position"
                        ):
                            ninja_pos = env.nplay_headless.ninja_position()
                        elif hasattr(env, "sim") and hasattr(env.sim, "ninja"):
                            ninja_pos = (env.sim.ninja.x, env.sim.ninja.y)

                        # Get level data and entities for caching
                        level_data = getattr(env, "level_data", None)
                        entities = (
                            getattr(env, "entities", [])
                            if hasattr(env, "entities")
                            else []
                        )

                        graph_overlay_surface = (
                            graph_visualizer.create_overlay_visualization(
                                graph_data,
                                screen,
                                ninja_position=ninja_pos,
                                level_data=level_data,
                                entities=entities,
                            )
                        )
                        path_updated = False  # Reset flag after recreation

                if graph_overlay_surface:
                    screen = pygame.display.get_surface()
                    if screen:
                        screen.blit(graph_overlay_surface, (0, 0))

            # Render standalone graph window (cache this too for performance)
            if args.standalone_graph and standalone_window:
                # Only update standalone window every few frames for performance
                if frame_counter % 5 == 0:  # Update every 5 frames
                    standalone_surface = (
                        graph_visualizer.create_standalone_visualization(
                            graph_data,
                            standalone_window.get_width(),
                            standalone_window.get_height(),
                        )
                    )
                    standalone_window.blit(standalone_surface, (0, 0))
                    pygame.display.flip()  # Update standalone window

        except Exception as e:
            print(f"Graph visualization error: {e}")
            import traceback

            traceback.print_exc()

    # Save graph image if requested (one-time save)
    if args.save_graph and graph_visualizer and graph_data and frame_counter == 1:
        try:
            surface = graph_visualizer.create_standalone_visualization(
                graph_data, width=1200, height=800
            )
            pygame.image.save(surface, args.save_graph)
            print(f"Graph visualization saved to {args.save_graph}")
        except Exception as e:
            print(f"Error saving graph: {e}")

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
