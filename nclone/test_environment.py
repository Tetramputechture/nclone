"""
N++ Environment Test with Graph Visualization

This script provides a comprehensive testing environment for the N++ simulator
with integrated graph visualization capabilities.

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



# Save graph visualization to image
python test_environment.py --save-graph graph.png

Advanced Usage:
--------------
# Combine multiple features
python test_environment.py --visualize-graph --standalone-graph

# Customize edge visualization
python test_environment.py --visualize-graph --show-edges walk jump fall

# Use custom map with graph visualization
python test_environment.py --map custom_level.txt --visualize-graph

# Headless mode with graph export
python test_environment.py --headless --save-graph analysis.png

Interactive Controls (during runtime):
-------------------------------------
V - Toggle graph overlay on/off

S - Save current graph visualization
G - Toggle built-in graph debug overlay
E - Toggle exploration debug overlay
C - Toggle grid debug overlay
R - Reset environment



Edge Type Visualization:
-----------------------
--show-edges walk jump fall wall_slide one_way functional
Default: walk jump

Examples:
--------
# Real-time RL training visualization
python test_environment.py --visualize-graph

# Level analysis and validation
python test_environment.py --standalone-graph --show-edges walk jump fall

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

from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder


def _get_ninja_position(env):
    """Get current ninja position from environment."""
    if hasattr(env, "nplay_headless") and hasattr(env.nplay_headless, "ninja_position"):
        return env.nplay_headless.ninja_position()
    elif hasattr(env, "sim") and hasattr(env.sim, "ninja"):
        return (env.sim.ninja.x, env.sim.ninja.y)
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
