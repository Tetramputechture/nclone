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

GAME_STATE_CHANNELS = 26

LEVEL_WIDTH = 1056.0
LEVEL_HEIGHT = 600.0
LEVEL_DIAGONAL = np.sqrt(LEVEL_WIDTH**2 + LEVEL_HEIGHT**2)
