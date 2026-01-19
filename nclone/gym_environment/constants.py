"""Constants for the gym environment."""

import numpy as np

# Max time in frames per level before episode truncation (fallback)
# Note: Actual truncation limits are now calculated dynamically per level
# based on PBRS surface area and reachable mine count
MAX_TIME_IN_FRAMES = (
    10000  # Fallback when dynamic calculation unavailable (~2.8 minutes)
)

# Player frame size
PLAYER_FRAME_WIDTH = 84
PLAYER_FRAME_HEIGHT = 84

# Constants for rendered global view
# We use flipped dimensions because our screen is flipped and we want to preserve the aspect ratio
RENDERED_VIEW_WIDTH = 100  # 100 / 6
RENDERED_VIEW_HEIGHT = 176  # 1056 / 6
RENDERED_VIEW_CHANNELS = 1  # Grayscale

# Game state feature dimensions (enhanced physics + time_remaining)
NINJA_STATE_DIM = 41  # 40 enhanced physics features + 1 time_remaining
GAME_STATE_CHANNELS = NINJA_STATE_DIM

# Switch states array (legacy format, still used by some components)
MAX_LOCKED_DOORS = 5  # Maximum doors for switch_states array
FEATURES_PER_DOOR = 5  # [switch_x, switch_y, door_x, door_y, collected]
SWITCH_STATES_DIM = MAX_LOCKED_DOORS * FEATURES_PER_DOOR  # 25 total

# Locked door array for objective attention (supports 1-16 doors)
MAX_LOCKED_DOORS_ATTENTION = 16
LOCKED_DOOR_FEATURES_DIM = 8  # [switch_x, switch_y, switch_collected, switch_path_dist, door_x, door_y, door_open, door_path_dist]
LOCKED_DOOR_ARRAY_SIZE = MAX_LOCKED_DOORS_ATTENTION * LOCKED_DOOR_FEATURES_DIM  # 128

# Reachability features dimension
# 38 features (expanded for blind jump verification - Phases 1-5):
#   Base (4): [reachable_area_ratio, dist_to_switch_inv, dist_to_exit_inv, exit_reachable]
#   Path distances (2): [path_dist_to_switch_norm, path_dist_to_exit_norm]
#   Direction vectors (4): [dir_to_switch_x, dir_to_switch_y, dir_to_exit_x, dir_to_exit_y]
#   Mine context (2): [total_mines_norm, deadly_mine_ratio]
#   Phase indicator (1): [switch_activated] - critical for Markov property
#   Path direction (8): [next_hop_dir_x/y, waypoint_dir_x/y, waypoint_dist, path_detour_flag, mine_clearance_dir_x/y]
#   Path difficulty (1): [path_difficulty_ratio] - physics_cost/geometric_distance
#   Path curvature (3): [multi_hop_dir_x/y, path_curvature] - 8-hop lookahead and turn indicator
#   Exit lookahead (5): [exit_next_hop_x/y, exit_multi_hop_x/y, near_switch_indicator] - continuity at switch transition
#   Directional connectivity (8): [platform_dist_E, NE, N, NW, W, SW, S, SE] - blind jump verification
REACHABILITY_FEATURES_DIM = 38

# Mine SDF features dimension (for actor safety awareness)
# 3 features from global Mine Signed Distance Field:
#   SDF value (1): distance to nearest mine (negative=inside danger zone)
#   SDF gradient (2): escape direction from mines (unit vector)
# These are computed from ALL mines, not just 8 nearest in spatial_context
MINE_SDF_FEATURES_DIM = 3

# Spatial context features dimension (graph-free alternative)
# 112 features (graph-free local geometry with velocity-hazard alignment):
#   Local tile grid (64): 8×8 grid of simplified tile categories (0-4) normalized
#   Mine overlay (48): 8 nearest mines with 6 features each:
#     - relative_pos(2) + state(1) + radius(1)
#     - velocity_dot_direction(1) + distance_rate(1) [Markovian: current state only]
SPATIAL_CONTEXT_DIM = 112

# Minimal observation dimension (simplified for testing time-independence)
# 40 features (core physics + path guidance + nearest mines + buffers):
#   Core physics (12): velocity(2) + state_one_hot(5) + airborne + walled + wall_normal + floor_normal(2)
#   Path guidance (8): next_hop_dir(2) + waypoint_dir(2) + exit_dir(2) + phase + curvature
#   Mine context (16): 4 nearest mines × 4 features (x, y, state, distance)
#   Buffers (4): jump_buffer + floor_buffer + wall_buffer + launch_pad_buffer
MINIMAL_OBSERVATION_DIM = 40

LEVEL_WIDTH = 1056.0
LEVEL_HEIGHT = 600.0
LEVEL_DIAGONAL = np.sqrt(LEVEL_WIDTH**2 + LEVEL_HEIGHT**2)
