"""Constants for the gym environment."""

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
# Increased to 12 based on research suggesting benefits of larger frame stacks for temporal modeling
# in procedural environments (e.g., findings from ProcGen benchmark research).
TEMPORAL_FRAMES = 12

# Total game state features including all entities
GAME_STATE_FEATURES = 159019

# Total game state features with limited entity count
GAME_STATE_FEATURES_LIMITED_ENTITY_COUNT = 2594

# Ninja state size
NINJA_STATE_SIZE = 10
NINJA_STATE_SIZE_RICH = (
    24  # Enhanced with buffers, contacts, normals, impact risk, state flags
)

GAME_STATE_CHANNELS = 31

# LEVEL_WIDTH and LEVEL_HEIGHT
LEVEL_WIDTH = 1056.0
LEVEL_HEIGHT = 600.0
