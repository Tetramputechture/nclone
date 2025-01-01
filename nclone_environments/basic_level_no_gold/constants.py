"""Constants for the basic level no gold environment."""

# Max time in frames per level before episode truncation
MAX_TIME_IN_FRAMES_SMALL_LEVEL = 2000

# Player frame size
PLAYER_FRAME_WIDTH = 84
PLAYER_FRAME_HEIGHT = 84

# Temporal frames
TEMPORAL_FRAMES = 3

# Ninja state size
NINJA_STATE_SIZE = 10

# Only ninja and 1 exit and 1 switch, no tile data (to be used in conjunction with frame stacking) =
# 10 ninja states
# 2 entity states (exit and switch active)
# 1 time remaining
# 4 for vectors to objectives (exit and switch)
GAME_STATE_FEATURES_ONLY_NINJA_AND_EXIT_AND_SWITCH = 17

# LEVEL_WIDTH and LEVEL_HEIGHT
LEVEL_WIDTH = 1056.0
LEVEL_HEIGHT = 600.0
