"""Constants for the gym environment."""

import numpy as np

# Max time in frames per level before episode truncation
MAX_TIME_IN_FRAMES = 5000

# Player frame size
PLAYER_FRAME_WIDTH = 84
PLAYER_FRAME_HEIGHT = 84

# Constants for rendered global view
# We use flipped dimensions because our screen is flipped and we want to preserve the aspect ratio
RENDERED_VIEW_WIDTH = 100  # 100 / 6
RENDERED_VIEW_HEIGHT = 176  # 1056 / 6
RENDERED_VIEW_CHANNELS = 1  # Grayscale

# Game state feature dimensions
NINJA_STATE_DIM = 29
PATH_AWARE_OBJECTIVES_DIM = 15
MINE_FEATURES_DIM = 8  # CHANGED: 5→8 (+3 features)
PROGRESS_FEATURES_DIM = 3
SEQUENTIAL_GOAL_DIM = 3  # NEW: sequential task features
ACTION_DEATH_PROBABILITIES_DIM = 6  # Mine death probability per action [0.0-1.0]
TERMINAL_VELOCITY_DEATH_PROBABILITIES_DIM = 6  # Terminal velocity death probability per action [0.0-1.0]
GAME_STATE_CHANNELS = (
    NINJA_STATE_DIM
    + PATH_AWARE_OBJECTIVES_DIM
    + MINE_FEATURES_DIM
    + PROGRESS_FEATURES_DIM
    + SEQUENTIAL_GOAL_DIM
    + ACTION_DEATH_PROBABILITIES_DIM
    + TERMINAL_VELOCITY_DEATH_PROBABILITIES_DIM
)  # 70 total (was 64)

# Switch states dimensions
MAX_LOCKED_DOORS = 5
FEATURES_PER_DOOR = 5  # [switch_x, switch_y, door_x, door_y, collected]
SWITCH_STATES_DIM = MAX_LOCKED_DOORS * FEATURES_PER_DOOR  # 25 total

# Locked door array for objective attention (supports 1-16 doors)
MAX_LOCKED_DOORS_ATTENTION = 16
LOCKED_DOOR_FEATURES_DIM = 8  # [switch_x, switch_y, switch_collected, switch_path_dist, door_x, door_y, door_open, door_path_dist]
LOCKED_DOOR_ARRAY_SIZE = MAX_LOCKED_DOORS_ATTENTION * LOCKED_DOOR_FEATURES_DIM  # 128

# Entity positions dimension
ENTITY_POSITIONS_DIM = 6  # [ninja_x, ninja_y, switch_x, switch_y, exit_x, exit_y]

# Reachability features dimension
REACHABILITY_FEATURES_DIM = 8

# Hierarchical subtask features dimension
SUBTASK_FEATURES_DIM = 4  # [subtask_type, progress, priority, completion_bonus]

LEVEL_WIDTH = 1056.0
LEVEL_HEIGHT = 600.0
LEVEL_DIAGONAL = np.sqrt(LEVEL_WIDTH**2 + LEVEL_HEIGHT**2)
