import pygame
from .nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
import argparse
import time
import cProfile
import pstats
import io
import numpy as np # Added for tile_map conversion

# Pathfinding imports
from .pathfinding.surface_parser import SurfaceParser
from .pathfinding.navigation_graph import NavigationGraphBuilder, JumpCalculator
from .pathfinding.astar_pathfinder import AStarPathfinder
from .pathfinding.utils import CollisionChecker # Added CollisionChecker import
from .tile_definitions import get_tile_definitions # For SurfaceParser

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

# --- Pathfinding Setup ---
pathfinding_results = None
if args.pathfind and not args.headless:
    print("Pathfinding test enabled.")
    # Initial observation to get map data, etc.
    obs, _ = env.reset()
    
    tile_map_raw = env.nplay_headless.sim.tile_dic
    # Convert tile_map from dict {(x,y): id} to numpy array
    # Assuming map dimensions are fixed or accessible. For now, let's use typical N++ dimensions.
    # Max X = 43, Max Y = 24 (0-indexed for array means 44x25 size)
    # The tile_dic seems to be 1-indexed and includes border tiles.
    # SurfaceParser expects 0-indexed map_data without border, so we might need to adjust.
    # From pathfinding_strategy.md: "Tile Map Input: The system utilizes a 2D NumPy array representing the level's tile map."
    # Let's assume the pathfinding components expect a map that corresponds to the playable area.
    # From nsim.py: WORLD_WIDTH = 1056, WORLD_HEIGHT = 600, TILE_PIXEL_SIZE = 24
    # MAP_WIDTH_TILES = 1056/24 = 44, MAP_HEIGHT_TILES = 600/24 = 25
    # These seem to be the dimensions including borders.
    # Playable area is often (1,1) to (42,23) if 0-indexed is (0,0) to (43,24)
    
    # Let's try to create the map based on typical N++ internal dimensions
    # map_width_sim = 44 # Max X coord in tile_dic is 43 (0 to 43)
    # map_height_sim = 25 # Max Y coord in tile_dic is 24 (0 to 24)
    # However, surface_parser might expect the inner, playable area.
    # Let's assume tile_dic keys are (col, row)
    max_x = 0
    max_y = 0
    # Determine map dimensions from tile_dic
    if tile_map_raw: # Ensure tile_map_raw is not empty
        max_x = max(key[0] for key in tile_map_raw.keys())
        max_y = max(key[1] for key in tile_map_raw.keys())
    else: # Fallback or raise error if map is empty
        print("Warning: tile_map_raw is empty. Pathfinding may fail.")
        # Default to some size if you want to proceed, or handle error
        max_x = 43 
        max_y = 24

    # Create a numpy array. tile_dic is 1-indexed for keys. Let's make numpy array 0-indexed.
    # If max_x is 43, array size is 44. If max_y is 24, array size is 25.
    game_map = np.zeros((max_y + 1, max_x + 1), dtype=int)
    for (x, y), tile_id in tile_map_raw.items():
        if 0 <= y < game_map.shape[0] and 0 <= x < game_map.shape[1]:
            game_map[y, x] = tile_id
        else:
            print(f"Warning: Tile coordinate ({x},{y}) out of bounds for game_map shape {game_map.shape}")

    tile_definitions = get_tile_definitions()
    collision_checker = CollisionChecker(game_map) # Instantiate CollisionChecker
    surface_parser = SurfaceParser(game_map, tile_definitions)
    surfaces = surface_parser.parse_surfaces()
    
    nav_graph_builder = NavigationGraphBuilder(surfaces, collision_checker)
    nav_graph = nav_graph_builder.build_graph()

    # Add jump edges
    jump_calculator = JumpCalculator(collision_checker) # Pass CollisionChecker
    for u_node_id, u_data in nav_graph.nodes(data=True):
        for v_node_id, v_data in nav_graph.nodes(data=True):
            if u_node_id == v_node_id: continue
            
            # Call calculate_jump with positions and surface types
            jump_result = jump_calculator.calculate_jump(
                start_pos=u_data['position'], 
                end_pos=v_data['position'],
                start_surface_type=u_data['surface_type'],
                start_surface_normal=u_data.get('nav_node_object').surface.normal if u_data.get('nav_node_object') else None, # Get normal from NavNode's surface
                # initial_run_velocities_x can be None for defaults
            )
            if jump_result: # jump_result is a JumpTrajectory object or None
                trajectory_frames = jump_result.frames # For visualization if needed
                nav_graph.add_edge(u_node_id, v_node_id, 
                                   move_type=jump_result.jump_type, 
                                   frames=jump_result.total_frames, 
                                   trajectory=trajectory_frames)

    # Define start and goal for pathfinding (example)
    # For this test, let's try to find a path from the first node to the last node if graph is not empty
    start_node_id = None
    goal_node_id = None
    if nav_graph.number_of_nodes() > 1:
        # Attempt to use player start if available, otherwise first node
        player_start_pos = env.nplay_headless.ninja_position()
        min_dist_start = float('inf')
        for node_id, data in nav_graph.nodes(data=True):
            dist = np.linalg.norm(np.array(data['position']) - np.array(player_start_pos))
            if dist < min_dist_start:
                min_dist_start = dist
                start_node_id = node_id
        
        # Goal: Use exit door position if available
        exit_pos = None
        if env.nplay_headless._sim_exit_door(): # Check if exit door exists
            exit_pos = env.nplay_headless.exit_door_position()
        
        if exit_pos:
            min_dist_goal = float('inf')
            for node_id, data in nav_graph.nodes(data=True):
                dist = np.linalg.norm(np.array(data['position']) - np.array(exit_pos))
                if dist < min_dist_goal:
                    min_dist_goal = dist
                    goal_node_id = node_id
        else: # Fallback to last node
            if nav_graph.number_of_nodes() > 0:
                 # Sort nodes by some criteria, e.g., x then y, to make it somewhat deterministic
                sorted_nodes = sorted(list(nav_graph.nodes(data=True)), key=lambda item: (item[1]['position'][0], item[1]['position'][1]))
                if start_node_id == sorted_nodes[-1][0] and len(sorted_nodes) > 1:
                     goal_node_id = sorted_nodes[-2][0] # Avoid start == goal if possible
                elif len(sorted_nodes) > 0:
                     goal_node_id = sorted_nodes[-1][0]


    path = []
    if start_node_id is not None and goal_node_id is not None and start_node_id != goal_node_id:
        print(f"Attempting pathfinding from node {start_node_id} to {goal_node_id}")
        # Heuristic: Euclidean distance for A*
        def heuristic(node1_id, node2_id):
            pos1 = nav_graph.nodes[node1_id]['position']
            pos2 = nav_graph.nodes[node2_id]['position']
            return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

        astar = AStarPathfinder(nav_graph, heuristic_func=heuristic, weight_key='frames')
        path = astar.find_path(start_node_id, goal_node_id)
        if path:
            print(f"Path found: {path}")
        else:
            print("No path found.")
    else:
        print("Could not determine valid start/goal for pathfinding or graph too small.")

    pathfinding_results = {
        'surfaces': surfaces,
        'nav_graph': nav_graph,
        'path': path,
        # 'jump_trajectory': None, # Example: Add specific trajectory if needed
        # 'los_path': None,      # Example: Add LOS path if optimizer is used
    }
    env.set_pathfinding_data(pathfinding_results)
# --- End Pathfinding Setup ---


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

    # Update pathfinding data on reset if enabled (e.g. if map changes)
    if (terminated or truncated) and args.pathfind and not args.headless:
        print("Resetting and re-running pathfinding...")
        obs, _ = env.reset() # Env reset reloads map
        tile_map_raw = env.nplay_headless.sim.tile_dic
        max_x = 0
        max_y = 0
        if tile_map_raw:
            max_x = max(key[0] for key in tile_map_raw.keys())
            max_y = max(key[1] for key in tile_map_raw.keys())
        else:
            print("Warning: tile_map_raw is empty on reset. Pathfinding may fail.")
            max_x = 43
            max_y = 24
            
        game_map = np.zeros((max_y + 1, max_x + 1), dtype=int)
        for (x, y), tile_id in tile_map_raw.items():
            if 0 <= y < game_map.shape[0] and 0 <= x < game_map.shape[1]:
                game_map[y, x] = tile_id
            else:
                print(f"Warning: Tile coordinate ({x},{y}) out of bounds for game_map shape {game_map.shape} on reset")

        tile_definitions = get_tile_definitions()
        collision_checker = CollisionChecker(game_map) # Instantiate CollisionChecker
        surface_parser = SurfaceParser(game_map, tile_definitions)
        surfaces = surface_parser.parse_surfaces()
        
        nav_graph_builder = NavigationGraphBuilder(surfaces, collision_checker)
        nav_graph = nav_graph_builder.build_graph()

        jump_calculator = JumpCalculator(collision_checker) # Pass CollisionChecker
        for u_node_id, u_data in nav_graph.nodes(data=True):
            for v_node_id, v_data in nav_graph.nodes(data=True):
                if u_node_id == v_node_id: continue
                jump_result = jump_calculator.calculate_jump(
                    start_pos=u_data['position'],
                    end_pos=v_data['position'],
                    start_surface_type=u_data['surface_type'],
                    start_surface_normal=u_data.get('nav_node_object').surface.normal if u_data.get('nav_node_object') else None,
                )
                if jump_result:
                    trajectory_frames = jump_result.frames
                    nav_graph.add_edge(u_node_id, v_node_id, 
                                       move_type=jump_result.jump_type, 
                                       frames=jump_result.total_frames, 
                                       trajectory=trajectory_frames)

        start_node_id = None
        goal_node_id = None
        if nav_graph.number_of_nodes() > 1:
            player_start_pos = env.nplay_headless.ninja_position()
            min_dist_start = float('inf')
            for node_id, data in nav_graph.nodes(data=True):
                dist = np.linalg.norm(np.array(data['position']) - np.array(player_start_pos))
                if dist < min_dist_start:
                    min_dist_start = dist
                    start_node_id = node_id
            
            exit_pos = None
            if env.nplay_headless._sim_exit_door():
                exit_pos = env.nplay_headless.exit_door_position()
            if exit_pos:
                min_dist_goal = float('inf')
                for node_id, data in nav_graph.nodes(data=True):
                    dist = np.linalg.norm(np.array(data['position']) - np.array(exit_pos))
                    if dist < min_dist_goal:
                        min_dist_goal = dist
                        goal_node_id = node_id
            else:
                if nav_graph.number_of_nodes() > 0:
                    sorted_nodes = sorted(list(nav_graph.nodes(data=True)), key=lambda item: (item[1]['position'][0], item[1]['position'][1]))
                    if start_node_id == sorted_nodes[-1][0] and len(sorted_nodes) > 1:
                        goal_node_id = sorted_nodes[-2][0]
                    elif len(sorted_nodes) > 0:
                        goal_node_id = sorted_nodes[-1][0]

        path = []
        if start_node_id is not None and goal_node_id is not None and start_node_id != goal_node_id:
            print(f"Attempting pathfinding from node {start_node_id} to {goal_node_id} (on reset)")
            def heuristic(node1_id, node2_id):
                pos1 = nav_graph.nodes[node1_id]['position']
                pos2 = nav_graph.nodes[node2_id]['position']
                return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            astar = AStarPathfinder(nav_graph, heuristic_func=heuristic, weight_key='frames')
            path = astar.find_path(start_node_id, goal_node_id)
            if path:
                print(f"Path found (on reset): {path}")
            else:
                print("No path found (on reset).")
        else:
            print("Could not determine valid start/goal for pathfinding on reset or graph too small.")

        pathfinding_results = {
            'surfaces': surfaces,
            'nav_graph': nav_graph,
            'path': path,
        }
        env.set_pathfinding_data(pathfinding_results)
        # Need to render once after setting data for it to show up immediately after reset
        if not args.headless:
            env.render() 
            pygame.display.flip() # Ensure display update

    # Reset if episode is done
    if terminated or truncated:
        # Pathfinding logic already handled above for the reset case
        if not (args.pathfind and not args.headless): # if pathfinding didn't already reset
            observation, info = env.reset()

    # Print observation shape
    # print(observation['game_state'].shape)

    # print(f'Gold collected: {env.get_gold_collected()}')

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
