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

parser.add_argument(
    "--export-frame",
    type=str,
    default=None,
    help="Export first frame of simulation to specified image file and quit (for AI testing).",
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
env = NppEnvironment(
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
                elif len(frame.shape) == 2:
                    # Already 2D grayscale
                    image = Image.fromarray(frame.astype(np.uint8), mode="L")
                    print(f"Exporting 2D grayscale frame with shape {frame.shape}")
                else:
                    print(f"Warning: Unsupported frame shape {frame.shape}")

                image.save(args.export_frame)
                print(f"Frame successfully exported to {args.export_frame}")
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
