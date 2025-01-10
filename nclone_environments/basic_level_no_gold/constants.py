"""Constants for the basic level no gold environment."""

# Max time in frames per level before episode truncation
MAX_TIME_IN_FRAMES = 20000

# Player frame size
PLAYER_FRAME_WIDTH = 84
PLAYER_FRAME_HEIGHT = 84

# Constants for rendered global view
# We use flipped dimensions because our screen is flipped and we want to preserve the aspect ratio
RENDERED_VIEW_WIDTH = 100  # 100 / 6
RENDERED_VIEW_HEIGHT = 176  # 1056 / 6
RENDERED_VIEW_CHANNELS = 1  # Grayscale


# Temporal frames
TEMPORAL_FRAMES = 4

# Total game state features including all entities
GAME_STATE_FEATURES = 159019

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
