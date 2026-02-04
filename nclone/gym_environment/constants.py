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

# Game state feature dimensions (streamlined, no redundancies)
# Version 2.0 (Feb 2026): Reduced from 41 to 27 dims by removing redundant features
# Features: velocity(2) + movement_state(4) + airborne(1) + inputs(2) + buffers(3) +
#           jump_duration_remaining(1) + contacts(3) + normals(3) + walled(1) +
#           physics(3) + position(2) + time_remaining(1) = 26
NINJA_STATE_DIM = 26  # Was 41: removed 14 redundant features
GAME_STATE_CHANNELS = NINJA_STATE_DIM

# Switch states array (legacy format, still used by some components)
MAX_LOCKED_DOORS = 5  # Maximum doors for switch_states array
FEATURES_PER_DOOR = 5  # [switch_x, switch_y, door_x, door_y, collected]
SWITCH_STATES_DIM = MAX_LOCKED_DOORS * FEATURES_PER_DOOR  # 25 total

# Locked door array for objective attention (supports 1-16 doors)
MAX_LOCKED_DOORS_ATTENTION = 16
LOCKED_DOOR_FEATURES_DIM = 8  # [switch_x, switch_y, switch_collected, switch_path_dist, door_x, door_y, door_open, door_path_dist]
LOCKED_DOOR_ARRAY_SIZE = MAX_LOCKED_DOORS_ATTENTION * LOCKED_DOOR_FEATURES_DIM  # 128

# Reachability features dimension (streamlined, no misleading features)
# Version 2.0 (Feb 2026): Reduced from 22 to 11 dims by removing redundant/misleading features
# 11 features:
#   Path distances (2): [path_dist_to_switch_norm, path_dist_to_exit_norm]
#   Phase (1): [switch_activated] - critical for Markov property
#   A* guidance (2): [next_hop_dir_x, next_hop_dir_y] - optimal path direction
#   Detour detection (1): [path_requires_detour] - flag when A* opposes Euclidean
#   Mine clearance (2): [mine_clearance_dir_x, mine_clearance_dir_y] - SDF gradient
#   Reachability (1): [exit_reachable]
#   Platform distances (2): [platform_ahead, platform_below] - critical for blind jumps
# REMOVED: Global stats (reachable_area_ratio, total_mines, deadly_mine_ratio),
#          Euclidean directions (misleading, can point into walls),
#          redundant inverses (dist_to_switch_inv, dist_to_exit_inv),
#          6 of 8 directional connectivity features (kept only 2 most relevant)
REACHABILITY_FEATURES_DIM = 11  # Was 22

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

# Minimal observation dimension (waypoints + LSTM features removed 2026-02-03)
# 30 features (core physics + path guidance + nearest mines + buffers):
#   Core physics (12): velocity(2) + state_one_hot(5) + airborne + walled + wall_normal + floor_normal(2)
#   Path guidance (4): next_hop_dir(2) + exit_dir(2)
#   Mine context (12): 3 nearest mines × 4 features (x, y, state, distance)
#   Buffers (4): jump_buffer + floor_buffer + wall_buffer + launch_pad_buffer
# REMOVED: waypoints (2 + distance), phase indicator (frame stacking handles), curvature (LSTM-specific)
MINIMAL_OBSERVATION_DIM = 30

# Total observation space (for graph_free architecture, Version 2.0)
# 27 (game_state) + 11 (reachability) + 112 (spatial_context) + 3 (mine_sdf) = 153 dims
# Version 2.0 (Feb 2026): 49% reduction from 301 dims (v1 with frame stacking)

LEVEL_WIDTH = 1056.0
LEVEL_HEIGHT = 600.0
LEVEL_DIAGONAL = np.sqrt(LEVEL_WIDTH**2 + LEVEL_HEIGHT**2)
