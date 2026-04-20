"""Constants for the gym environment."""

import numpy as np

# Max time in frames per level before episode truncation (fallback)
# Note: Actual truncation limits are now calculated dynamically per level
# based on PBRS surface area and reachable mine count
MAX_TIME_IN_FRAMES = (
    10000  # Fallback when dynamic calculation unavailable (~2.8 minutes)
)

# Game state feature dimensions (streamlined, no redundancies)
# Version 2.1 (Feb 2026): Reduced from 26 to 24 dims by removing hor_input, jump_input
# Features: velocity(2) + movement_state(4) + airborne(1) + buffers(3) +
#           jump_duration_remaining(1) + contacts(3) + normals(3) + walled(1) +
#           physics(3) + position(2) + time_remaining(1) = 24
# Physics-derived (3): horizontal_range_norm, max_jump_height_norm, time_to_ground_norm
NINJA_STATE_DIM = 24  # Was 26: removed own-action features (hor_input, jump_input)
PHYSICS_DERIVED_DIM = 3
# L1.4 (NPP-12): append switch_activated (0/1) for phase indicator in the policy MLP stream.
GAME_STATE_SWITCH_ACTIVATED_IDX = NINJA_STATE_DIM + PHYSICS_DERIVED_DIM
GAME_STATE_CHANNELS = NINJA_STATE_DIM + PHYSICS_DERIVED_DIM + 1

# Switch states array (legacy format, still used internally by reward/PBRS code)
MAX_LOCKED_DOORS = 5  # Maximum doors for switch_states array
FEATURES_PER_DOOR = 5  # [switch_x, switch_y, door_x, door_y, collected]
SWITCH_STATES_DIM = MAX_LOCKED_DOORS * FEATURES_PER_DOOR  # 25 total

# Locked door array for objective attention (supports 1-16 doors) — still used by
# debug/overlay code even though it is no longer exposed via the policy obs dict.
MAX_LOCKED_DOORS_ATTENTION = 16
LOCKED_DOOR_FEATURES_DIM = 8  # [switch_x, switch_y, switch_collected, switch_path_dist, door_x, door_y, door_open, door_path_dist]
LOCKED_DOOR_ARRAY_SIZE = MAX_LOCKED_DOORS_ATTENTION * LOCKED_DOOR_FEATURES_DIM  # 128

# L1 toggle-mines hazard map (NPP-46): 1-channel deadly-mine SDF on a 6px grid over
# the 1008×552 px level. This is the Layer 1 policy-facing key. Shape is (C, H, W)
# to match channel-first conventions used throughout the policy CNN path.
HAZARD_MAP_CHANNELS = 1
HAZARD_MAP_WIDTH = 168  # 1008 / 6
HAZARD_MAP_HEIGHT = 92  # 552 / 6

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

# Tile type IDs run 0–37 (38 distinct types) in the N++ tile system
NUM_TILE_TYPES = 38
