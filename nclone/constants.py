"""
Shared constants for physics, ninja, and collision systems.

This module contains all the constants used across the nclone physics simulation
to avoid circular imports between modules.
"""

# ============================================================================
# NINJA PHYSICS CONSTANTS
# ============================================================================

# Basic ninja properties
NINJA_RADIUS = 10

# Gravity constants
GRAVITY_FALL = 0.06666666666666665
GRAVITY_JUMP = 0.01111111111111111

# Movement acceleration constants
GROUND_ACCEL = 0.06666666666666665
AIR_ACCEL = 0.04444444444444444

# Drag constants (applied to velocity each frame)
DRAG_REGULAR = 0.9933221725495059  # 0.99^(2/3)
DRAG_SLOW = 0.8617738760127536  # 0.80^(2/3)

# Friction constants
FRICTION_GROUND = 0.9459290248857720  # 0.92^(2/3)
FRICTION_GROUND_SLOW = 0.8617738760127536  # 0.80^(2/3)
FRICTION_WALL = 0.9113380468927672  # 0.87^(2/3)

# Speed limits
MAX_HOR_SPEED = 3.333333333333333

# Jump mechanics
MAX_JUMP_DURATION = 45

# Collision and damage constants
MAX_SURVIVABLE_IMPACT = 6
MIN_SURVIVABLE_CRUSHING = 0.05

# ============================================================================
# JUMP CONSTANTS
# ============================================================================

# Basic jump velocities
JUMP_FLAT_GROUND_Y = -2

# Slope jump constants
JUMP_SLOPE_DOWNHILL_X = 2/3
JUMP_SLOPE_DOWNHILL_Y = 2
JUMP_SLOPE_DOWNHILL_OPPOSITE_Y = -1.4
JUMP_SLOPE_UPHILL_FORWARD_Y = -1.4
JUMP_SLOPE_UPHILL_PERP_X = 2/3
JUMP_SLOPE_UPHILL_PERP_Y = 2

# Wall jump constants
JUMP_WALL_SLIDE_X = 2/3
JUMP_WALL_SLIDE_Y = -1
JUMP_WALL_REGULAR_X = 1
JUMP_WALL_REGULAR_Y = -1.4

# Launch pad constants
JUMP_LAUNCH_PAD_BOOST_SCALAR = 1.7
JUMP_LAUNCH_PAD_BOOST_FACTOR = 2/3

# ============================================================================
# ANIMATION CONSTANTS
# ============================================================================

# Animation data file
ANIM_DATA = "anim_data_line_new.txt.bin"

# Victory dance settings
DANCE_RANDOM = True  # Choose whether the victory dance is picked randomly.
DANCE_ID_DEFAULT = 0  # Choose the id of the dance that will always play if DANCE_RANDOM is false.

# Dance IDs (names of dances courtesy of Eddy, 0-12 are new, 13-20 are classics):
# 0:Default pose, 1:Tired and sweaty, 2:Hands in the air, 3:Crab walk, 4:Shuffle, 5:Turk dance,
# 6:Russian squat dance, 7:Arm wave, 8:The carlton, 9:Hanging, 10:The worm, 11:Thriller,
# 12:Side to side, 13:Clutch to the sky, 14:Frontflip, 15:Waving, 16:On one leg, 17:Backflip,
# 18:Kneeling, 19:Fall to the floor, 20:Russian squat dance (classic version), 21:Kick
DANCE_DIC = {
    0: (104, 104), 1: (106, 225), 2: (226, 345), 3: (346, 465), 4: (466, 585), 5: (586, 705),
    6: (706, 825), 7: (826, 945), 8: (946, 1065), 9: (1066, 1185), 10: (1186, 1305),
    11: (1306, 1485), 12: (1486, 1605), 13: (1606, 1664), 14: (1665, 1731), 15: (1732, 1810),
    16: (1811, 1852), 17: (1853, 1946), 18: (1947, 2004), 19: (2005, 2156), 20: (2157, 2241),
    21: (2242, 2295)
}

# ============================================================================
# RAGDOLL CONSTANTS
# ============================================================================

# Ragdoll physics constants
RAGDOLL_GRAVITY = 0.06666666666666665
RAGDOLL_DRAG = 0.99999 

# ============================================================================
# MAP CONSTANTS
# ============================================================================

MAP_TILE_WIDTH = 42
MAP_TILE_HEIGHT = 23