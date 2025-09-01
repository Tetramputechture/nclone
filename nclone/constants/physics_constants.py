"""
Centralized physics constants for N++ simulation and RL integration.
All physics-related constants should be defined here to avoid duplication.
"""

# === NINJA PHYSICS CONSTANTS ===
# Basic ninja properties
NINJA_RADIUS = 10  # Ninja collision radius in pixels

# Movement speeds
MAX_HOR_SPEED = 9.0  # Maximum horizontal speed
MAX_VER_SPEED = 12.0  # Maximum vertical speed (upward)
MIN_HORIZONTAL_VELOCITY = 0.1  # Minimum horizontal velocity for calculations

# Gravity and physics (using original N++ values)
GRAVITY_FALL = 0.06666666666666665  # Original N++ gravity when falling
GRAVITY_JUMP = 0.01111111111111111  # Original N++ gravity when jumping
TERMINAL_VELOCITY = 12.0  # Maximum falling speed

# Jump mechanics
JUMP_INITIAL_VELOCITY = -6.0  # Initial upward velocity when jumping
WALL_JUMP_HORIZONTAL_BOOST = 4.5  # Horizontal boost from wall jumps

# === BOUNCE BLOCK CONSTANTS ===
# Interaction parameters
BOUNCE_BLOCK_INTERACTION_RADIUS = 18.0  # Pixels - interaction detection radius
BOUNCE_BLOCK_CHAIN_DISTANCE = 36.0  # Pixels - maximum chaining distance
BOUNCE_BLOCK_SIZE = 9.0  # Pixels - bounce block is 9x9

# Boost mechanics
BOUNCE_BLOCK_BOOST_MIN = 1.2  # Minimum boost multiplier
BOUNCE_BLOCK_BOOST_MAX = 3.0  # Maximum boost multiplier
BOUNCE_BLOCK_BOOST_NEUTRAL = 1.0  # Boost when in neutral state

# Energy and efficiency
BOUNCE_BLOCK_ENERGY_EFFICIENCY = 0.7  # Energy cost multiplier for bounce block movement
BOUNCE_BLOCK_SUCCESS_BONUS = 0.15  # Success probability bonus

# Spring-mass system (for data tracking only)
BOUNCE_BLOCK_SPRING_CONSTANT = 0.8  # Spring constant for compression calculation
BOUNCE_BLOCK_MASS = 1.0  # Mass for energy calculations
BOUNCE_BLOCK_DAMPING = 0.95  # Damping factor for oscillations

# Compression thresholds
BOUNCE_BLOCK_MIN_COMPRESSION = 0.1  # Minimum compression to register
BOUNCE_BLOCK_MAX_COMPRESSION = 1.0  # Maximum compression amount
BOUNCE_BLOCK_COMPRESSION_THRESHOLD = 0.5  # Threshold for state transitions

# === TRAJECTORY CALCULATION CONSTANTS ===
# Time and distance
DEFAULT_MINIMUM_TIME = 0.1  # Minimum time for trajectory calculations
MAX_TRAJECTORY_TIME = 10.0  # Maximum time for trajectory calculations
TRAJECTORY_POINT_INTERVAL = 0.1  # Time interval between trajectory points

# Success probability
SUCCESS_PROBABILITY_MIN = 0.1  # Minimum success probability
SUCCESS_PROBABILITY_MAX = 1.0  # Maximum success probability
BASE_SUCCESS_PROBABILITY = 0.8  # Base success probability for normal movement

# Distance penalties
DISTANCE_PENALTY_FACTOR = 0.001  # Penalty per pixel of distance
HEIGHT_PENALTY_FACTOR = 0.002  # Penalty per pixel of height difference
VELOCITY_PENALTY_FACTOR = 0.05  # Penalty per unit of velocity
TIME_PENALTY_FACTOR = 0.02  # Penalty per unit of time

# === GRAPH BUILDING CONSTANTS ===
# Edge weights
BASE_EDGE_WEIGHT = 1.0  # Base weight for graph edges
BOUNCE_BLOCK_EDGE_WEIGHT_MULTIPLIER = 0.8  # Multiplier for bounce block edges
PLATFORM_EDGE_WEIGHT = 0.9  # Weight for platform edges

# Traversability
MIN_TRAVERSABLE_GAP = 18.0  # Minimum gap size for traversability
MAX_JUMP_DISTANCE = 150.0  # Maximum jump distance for edge creation
MAX_FALL_DISTANCE = 300.0  # Maximum fall distance for edge creation

# === ENTITY TYPE CONSTANTS ===
ENTITY_TYPE_BOUNCE_BLOCK = 17  # Entity type ID for bounce blocks

# === ANIMATION CONSTANTS ===
# Animation data file
ANIM_DATA = "anim_data_line_new.txt.bin"

# Victory dance settings
DANCE_RANDOM = True  # Choose whether the victory dance is picked randomly.
DANCE_ID_DEFAULT = 0  # Choose the id of the dance that will always play if DANCE_RANDOM is false.

# Dance IDs (names of dances courtesy of Eddy, 0-12 are new, 13-20 are classics):
DANCE_DIC = {
    0: (104, 104), 1: (106, 225), 2: (226, 345), 3: (346, 465), 4: (466, 585), 5: (586, 705),
    6: (706, 825), 7: (826, 945), 8: (946, 1065), 9: (1066, 1185), 10: (1186, 1305),
    11: (1306, 1485), 12: (1486, 1605), 13: (1606, 1664), 14: (1665, 1731), 15: (1732, 1810),
    16: (1811, 1852), 17: (1853, 1946), 18: (1947, 2004), 19: (2005, 2156), 20: (2157, 2241),
    21: (2242, 2295)
}

# === PHYSICS SIMULATION CONSTANTS ===
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

# Ragdoll physics constants
RAGDOLL_GRAVITY = 0.06666666666666665
RAGDOLL_DRAG = 0.99999 

# === MAP CONSTANTS ===
# All N++ levels are exactly the same size: 44x25 tiles (1056x600 pixels)
MAP_TILE_WIDTH = 42  # Playable area width
MAP_TILE_HEIGHT = 23  # Playable area height
TILE_PIXEL_SIZE = 24

# Padding around playable area
MAP_PADDING = 1

# FIXED TOTAL MAP DIMENSIONS (use these constants throughout codebase)
FULL_MAP_WIDTH = 44  # MAP_TILE_WIDTH + 2 * MAP_PADDING
FULL_MAP_HEIGHT = 25  # MAP_TILE_HEIGHT + 2 * MAP_PADDING
FULL_MAP_WIDTH_PX = 1056  # FULL_MAP_WIDTH * TILE_PIXEL_SIZE
FULL_MAP_HEIGHT_PX = 600  # FULL_MAP_HEIGHT * TILE_PIXEL_SIZE
ENTITY_TYPE_NINJA = 0  # Entity type ID for ninja

# === COLLISION DETECTION CONSTANTS ===
COLLISION_EPSILON = 0.1  # Small value for collision detection precision
OVERLAP_THRESHOLD = 1.0  # Threshold for entity overlap detection

# === PERFORMANCE CONSTANTS ===
MAX_ENTITIES_PER_CALCULATION = 100  # Maximum entities to process in one calculation
CACHE_TIMEOUT_FRAMES = 60  # Number of frames to cache calculations