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
# Version 2.1 (Feb 2026): Reduced from 26 to 24 dims by removing hor_input, jump_input
# Features: velocity(2) + movement_state(4) + airborne(1) + buffers(3) +
#           jump_duration_remaining(1) + contacts(3) + normals(3) + walled(1) +
#           physics(3) + position(2) + time_remaining(1) = 24
# Physics-derived (3): horizontal_range_norm, max_jump_height_norm, time_to_ground_norm
NINJA_STATE_DIM = 24  # Was 26: removed own-action features (hor_input, jump_input)
PHYSICS_DERIVED_DIM = 3
GAME_STATE_CHANNELS = NINJA_STATE_DIM + PHYSICS_DERIVED_DIM

# Switch states array (legacy format, still used by some components)
MAX_LOCKED_DOORS = 5  # Maximum doors for switch_states array
FEATURES_PER_DOOR = 5  # [switch_x, switch_y, door_x, door_y, collected]
SWITCH_STATES_DIM = MAX_LOCKED_DOORS * FEATURES_PER_DOOR  # 25 total

# Locked door array for objective attention (supports 1-16 doors)
MAX_LOCKED_DOORS_ATTENTION = 16
LOCKED_DOOR_FEATURES_DIM = 8  # [switch_x, switch_y, switch_collected, switch_path_dist, door_x, door_y, door_open, door_path_dist]
LOCKED_DOOR_ARRAY_SIZE = MAX_LOCKED_DOORS_ATTENTION * LOCKED_DOOR_FEATURES_DIM  # 128

# Reachability features dimension (streamlined, no misleading features)
# Version 2.1 (Feb 2026): Reduced from 10 to 7 dims; added path_dist_to_current_goal (8 dims)
# 10 features (added phi_gradient_x/y Mar 2026):
#   Path distances (2): [path_dist_to_switch_norm, path_dist_to_exit_norm]
#   Phase (1): [switch_activated] - critical for Markov property
#   A* guidance (2): [next_hop_dir_x, next_hop_dir_y] - optimal path direction
#   Platform distances (2): [platform_ahead, platform_below] - critical for blind jumps
#   Path distance to current goal (1): [path_dist_to_current_goal] - phase-aware distance
#   Phi gradient (2): [phi_gradient_x, phi_gradient_y] - physics-aware cost-to-go direction
# REMOVED: mine_clearance_dir (duplicate of escape_gradient in mine_sdf_features)
# REMOVED: exit_reachable (near-constant, derivable from path distance)
REACHABILITY_FEATURES_DIM = 10

# Mine SDF features dimension (for actor safety awareness)
# 3 features from global Mine Signed Distance Field:
#   SDF value (1): distance to nearest mine (negative=inside danger zone)
#   SDF gradient (2): escape direction from mines (unit vector)
# These are computed from ALL mines, not just 8 nearest in spatial_context
MINE_SDF_FEATURES_DIM = 3

# Spatial context features dimension (graph-free alternative)
# 84 features (graph-free local geometry with velocity-hazard alignment):
#   Local tile grid (64): 8×8 grid of simplified tile categories (0-4) normalized
#   Mine overlay (20): 5 nearest mines with 4 features each:
#     - relative_pos(2) + state(1) + velocity_dot_direction(1)
SPATIAL_CONTEXT_DIM = 84

# Minimal observation dimension (waypoints + LSTM features removed 2026-02-03)
# 30 features (core physics + path guidance + nearest mines + buffers):
#   Core physics (12): velocity(2) + state_one_hot(5) + airborne + walled + wall_normal + floor_normal(2)
#   Path guidance (4): next_hop_dir(2) + exit_dir(2)
#   Mine context (12): 3 nearest mines × 4 features (x, y, state, distance)
#   Buffers (4): jump_buffer + floor_buffer + wall_buffer + launch_pad_buffer
# REMOVED: waypoints (2 + distance), phase indicator (frame stacking handles), curvature (LSTM-specific)
MINIMAL_OBSERVATION_DIM = 30

# Total observation space (for graph_free architecture, Version 2.1)
# 27 (game_state) + 10 (reachability) + 84 (spatial_context) + 3 (mine_sdf) = 124 dims
# Version 2.1 (Feb 2026): Streamlined from 151 dims (removed redundancies)
# Mar 2026: +3 physics-derived, +2 phi-gradient

LEVEL_WIDTH = 1056.0
LEVEL_HEIGHT = 600.0

# Index of the agent position features within the game_state vector.
# Layout: velocity(2)+movement_state(4)+airborne(1)+buffers(3)+
#         jump_duration_remaining(1)+contacts(3)+normals(3)+walled(1)+physics(3)
#         = 21 features before position.
# game_state[:, GAME_STATE_POSITION_START : GAME_STATE_POSITION_START + 2]
# gives [pos_x, pos_y], both normalized to [0, 1] by LEVEL_WIDTH / LEVEL_HEIGHT.
GAME_STATE_POSITION_START = 21
LEVEL_DIAGONAL = np.sqrt(LEVEL_WIDTH**2 + LEVEL_HEIGHT**2)

# Level context encoding constants
# Tile type IDs run 0–37 (38 distinct types) in the N++ tile system
NUM_TILE_TYPES = 38
# CNN output dimension for the level context encoder (tile grid -> 64D vector)
LEVEL_CONTEXT_DIM = 64
