import pygame
from .nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
import argparse
import time
import cProfile
import pstats
import numpy as np # Added for tile_map conversion

# Pathfinding imports
from .pathfinding.surface_parser import SurfaceParser
from .pathfinding.navigation_graph import NavigationGraphBuilder, JumpCalculator
from .pathfinding.astar_pathfinder import PlatformerAStar
from .pathfinding.utils import CollisionChecker # Added CollisionChecker import

# Initialize pygame
pygame.init()
pygame.display.set_caption("N++ Environment Test")

# Argument parser
parser = argparse.ArgumentParser(description="Test N++ environment with frametime logging.")
parser.add_argument('--log-frametimes', action='store_true', help='Enable frametime logging to stdout.')
parser.add_argument('--headless', action='store_true', help='Run in headless mode (no GUI).')
parser.add_argument('--profile-frames', type=int, default=None, help='Run for a specific number of frames and then exit (for profiling).')
parser.add_argument('--pathfind', action='store_true', help='Enable pathfinding test and visualization.') # Added pathfind argument
args = parser.parse_args()

print(f'Headless: {args.headless}')
# Create environment
render_mode = 'rgb_array' if args.headless else 'human'
debug_overlay_enabled = not args.headless  # Disable overlay in headless mode
env = BasicLevelNoGold(render_mode=render_mode,
                       enable_frame_stack=False, enable_debug_overlay=debug_overlay_enabled, eval_mode=False, seed=42)
env.reset()

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
    else: # Minimal event handling for headless
        for event in pygame.event.get(pygame.QUIT): # only process QUIT events
            if event.type == pygame.QUIT:
                running = False

    # Map keyboard inputs to environment actions
    action = 0  # Default to NOOP
    if not args.headless: # Only process keyboard inputs if not in headless mode
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
        if not (args.pathfind and not args.headless): # if pathfinding didn't already reset
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
with open('profiling_stats.txt', 'w') as f:
    ps = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
    ps.print_stats()

print("Profiling stats saved to profiling_stats.txt")
