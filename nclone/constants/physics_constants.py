"""
Centralized physics constants for N++ simulation and RL integration.
All physics-related constants should be defined here to avoid duplication.
"""

# === ENTITY PHYSICS CONSTANTS ===
# Basic entity properties
NINJA_RADIUS = 10  # Ninja collision radius in pixels
DRONE_RADIUS = 7.5  # Drone collision radius in pixels
MINI_DRONE_RADIUS = 4.0  # Mini drone collision radius in pixels

# Grid and movement constants
DRONE_GRID_SIZE = 24  # Regular drone grid cell size in pixels
MINI_DRONE_GRID_SIZE = 12  # Mini drone grid cell size in pixels
DRONE_LAUNCH_SPEED = 4.0  # Drone launch speed in pixels/frame

# Thwump constants
THWUMP_SEMI_SIDE = 9  # Thwump semi-side size in pixels
THWUMP_FORWARD_SPEED = 20/7  # Thwump forward movement speed
THWUMP_BACKWARD_SPEED = 8/7  # Thwump backward movement speed

# Shove Thwump constants  
SHOVE_THWUMP_SEMI_SIDE = 12  # Shove thwump semi-side size in pixels
SHOVE_THWUMP_PROJECTILE_RADIUS = 8  # Shove thwump projectile radius in pixels

# One-way platform constants
ONE_WAY_PLATFORM_SEMI_SIDE = 12  # One-way platform semi-side size in pixels

# Toggle mine radii by state (from entity_toggle_mine.py)
TOGGLE_MINE_RADII = {0: 4.0, 1: 3.5, 2: 4.5}  # 0:toggled, 1:untoggled, 2:toggling

# Bounce block constants
BOUNCE_BLOCK_SEMI_SIDE = 4.5  # Bounce block semi-side size in pixels (9x9 pixel square)

# Movement speeds (exact values from sim_mechanics_doc.md)
MAX_HOR_SPEED = 3.333  # Maximum horizontal speed: 3.333 pixels/frame
MAX_VER_SPEED = 12.0  # Maximum vertical speed (upward) of the ninja (player)
MIN_HORIZONTAL_VELOCITY = 0.1  # Minimum horizontal velocity for calculations

# Gravity and physics (exact values from sim_mechanics_doc.md)
GRAVITY_FALL = 0.0667  # Fall gravity: 0.0667 pixels/frame²
GRAVITY_JUMP = 0.0111  # Jump gravity: 0.0111 pixels/frame²
TERMINAL_VELOCITY = 12.0  # Maximum falling speed

# Jump mechanics (exact values from sim_mechanics_doc.md)
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

# === MOVEMENT CLASSIFICATION CONSTANTS ===
# Thresholds for movement detection
MOVEMENT_THRESHOLD = 1e-6  # General movement detection threshold
VERTICAL_MOVEMENT_THRESHOLD = 1e-6  # Vertical movement threshold
WALK_SPEED_THRESHOLD = 0.5  # Speed threshold for walk classification
JUMP_VELOCITY_THRESHOLD = 0.3  # Velocity threshold for jump detection
WALL_CONTACT_DISTANCE = 15.0  # Distance for wall contact detection
MIN_BOUNCE_BLOCK_MOVEMENT_DISTANCE = 10.0  # Minimum distance for bounce block movement detection
MIN_UPWARD_MOVEMENT_FOR_BOUNCE = -10.0  # Minimum upward movement to consider bounce block

# Movement direction thresholds
HORIZONTAL_MOVEMENT_THRESHOLD = 2.0  # Threshold for horizontal movement
UPWARD_MOVEMENT_THRESHOLD = -5.0  # Threshold for upward movement
DOWNWARD_MOVEMENT_THRESHOLD = 5.0  # Threshold for downward movement

# Jump and trajectory thresholds
JUMP_THRESHOLD_Y = -1.0  # Y threshold for jump requirement
JUMP_THRESHOLD_VELOCITY = 0.5  # Velocity threshold for jump requirement
VELOCITY_MARGIN_MULTIPLIER = 2  # Multiplier for velocity margin calculations

# === ENERGY AND COST CONSTANTS ===
# Base energy costs
ENERGY_COST_BASE = 10.0  # Base energy cost for movements
ENERGY_COST_JUMP_MULTIPLIER = 2.0  # Multiplier for jump energy costs
JUMP_ENERGY_BASE = 1.5  # Base energy for jump movements
FALL_ENERGY_BASE = 0.5  # Base energy for fall movements
WALL_SLIDE_ENERGY_COST = 1.2  # Energy cost for wall sliding
WALL_JUMP_ENERGY_BASE = 2.0  # Base energy for wall jumps
LAUNCH_PAD_ENERGY_COST = 0.3  # Energy cost for launch pad usage

# Energy calculation factors
HEIGHT_FACTOR_DIVISOR = 50.0  # Divisor for height factor calculations
HEIGHT_FACTOR_MAX = 2.0  # Maximum height factor
DISTANCE_FACTOR_DIVISOR = 100.0  # Divisor for distance factor calculations
DISTANCE_FACTOR_MAX = 1.5  # Maximum distance factor
FALL_ENERGY_DISTANCE_DIVISOR = 100.0  # Divisor for fall energy distance
FALL_ENERGY_DISTANCE_MAX = 0.5  # Maximum fall energy distance factor

# === TRAJECTORY CALCULATION CONSTANTS ===
# Time and distance
DEFAULT_MINIMUM_TIME = 0.1  # Minimum time for trajectory calculations
DEFAULT_TIME_ESTIMATE = 1.0  # Default time estimate for movements
MAX_TRAJECTORY_TIME = 10.0  # Maximum time for trajectory calculations
TRAJECTORY_POINT_INTERVAL = 0.1  # Time interval between trajectory points
DEFAULT_TRAJECTORY_POINTS = 10  # Default number of trajectory points
JUMP_TIME_FALLBACK = 5.0  # Fallback time for jump calculations

# Success probability
SUCCESS_PROBABILITY_MIN = 0.1  # Minimum success probability
SUCCESS_PROBABILITY_MAX = 1.0  # Maximum success probability
BASE_SUCCESS_PROBABILITY = 0.8  # Base success probability for normal movement
SUCCESS_PROBABILITY_BASE = 0.8  # Base success probability (alias)
SUCCESS_PROBABILITY_HIGH_BASE = 0.95  # High base success probability
SUCCESS_PROBABILITY_DISTANCE_FACTOR = 0.001  # Distance factor for success probability

# Penalty calculations
DISTANCE_PENALTY_FACTOR = 0.001  # Penalty per pixel of distance
DISTANCE_PENALTY_DIVISOR = 100.0  # Divisor for distance penalties
DISTANCE_PENALTY_MAX = 0.3  # Maximum distance penalty
HEIGHT_PENALTY_FACTOR = 0.002  # Penalty per pixel of height difference
HEIGHT_PENALTY_DIVISOR = 50.0  # Divisor for height penalties
HEIGHT_PENALTY_MAX = 0.2  # Maximum height penalty
VELOCITY_PENALTY_FACTOR = 0.05  # Penalty per unit of velocity
VELOCITY_PENALTY_MAX = 0.2  # Maximum velocity penalty
TIME_PENALTY_FACTOR = 0.02  # Penalty per unit of time
TIME_PENALTY_DIVISOR = 30.0  # Divisor for time penalties
TIME_PENALTY_MAX = 0.1  # Maximum time penalty

# === MOVEMENT DIFFICULTY CONSTANTS ===
# Difficulty calculations
DEFAULT_DIFFICULTY = 1.0  # Default movement difficulty
JUMP_DIFFICULTY_DIVISOR = 3.0  # Divisor for jump difficulty
WALL_SLIDE_DIFFICULTY = 0.7  # Difficulty for wall sliding
WALL_JUMP_DIFFICULTY = 0.8  # Difficulty for wall jumping
LAUNCH_PAD_DIFFICULTY = 0.4  # Difficulty for launch pad usage

# === WALL MOVEMENT CONSTANTS ===
# Wall sliding
WALL_SLIDE_SPEED_DIVISOR = 20.0  # Divisor for wall slide speed
WALL_SLIDE_MIN_TIME = 0.5  # Minimum time for wall sliding

# === LAUNCH PAD CONSTANTS ===
# Launch pad mechanics
LAUNCH_PAD_VELOCITY_MULTIPLIER = 1.5  # Velocity multiplier for launch pads
LAUNCH_PAD_BOOST_FACTOR = 1.7  # Boost factor for launch pads
LAUNCH_PAD_GRAVITY_DIVISOR = 0.1  # Gravity divisor for launch pad calculations
LAUNCH_PAD_MIN_TIME = 1.0  # Minimum time for launch pad movement
LAUNCH_PAD_DISTANCE_THRESHOLD = 100.0  # Distance threshold for launch pad detection

# === WIN CONDITION CONSTANTS ===
# Switch and door mechanics
SWITCH_DOOR_MAX_DISTANCE = 500.0  # Maximum distance for switch-door pairing
WIN_CONDITION_SWITCH_BONUS = 0.3  # Bonus for approaching switches
WIN_CONDITION_EXIT_BONUS = 0.5  # Bonus for approaching exits
WIN_CONDITION_DOOR_BONUS = 0.4  # Bonus for utilizing opened doors
WIN_CONDITION_DOOR_PROXIMITY = 100.0  # Distance for door utilization bonus

# === NINJA STATE CONSTANTS ===
# Movement states from sim_mechanics_doc.md
GROUND_STATES = {0, 1, 2}  # Immobile, Running, Ground Sliding
AIR_STATES = {3, 4}  # Jumping, Falling
WALL_STATES = {5}  # Wall Sliding
INACTIVE_STATES = {6, 7, 8, 9}  # Inactive movement states

# === PROXIMITY AND DETECTION CONSTANTS ===
# Entity proximity
PROXIMITY_THRESHOLD = 100.0  # General proximity threshold for entities
HAZARD_PROXIMITY_THRESHOLD = 50.0  # Proximity threshold for hazard detection

# === HAZARD SYSTEM CONSTANTS ===
# Hazard detection and analysis
HAZARD_UPDATE_RADIUS = 150.0  # Pixels from ninja position for hazard updates
THWUMP_ACTIVATION_RANGE = 38.0  # Line-of-sight activation range for thwumps
SHOVE_THWUMP_CORE_RADIUS = 8.0  # Deadly core radius for shove thwumps
ONE_WAY_PLATFORM_THICKNESS = 12.0  # Platform collision thickness
DRONE_PREDICTION_TIME = 60.0  # Frames to predict drone movement

# Toggle mine radii by state
TOGGLE_MINE_RADIUS_TOGGLED = 4.0  # Toggled state radius
TOGGLE_MINE_RADIUS_UNTOGGLED = 3.5  # Untoggled state radius  
TOGGLE_MINE_RADIUS_TOGGLING = 4.5  # Toggling state radius

# Entity danger radii
THWUMP_DANGER_RADIUS = 18.0  # 1.5 tiles danger radius
SHOVE_THWUMP_DANGER_RADIUS = 24.0  # Outer size danger radius
ONE_WAY_PLATFORM_SIZE = 24  # Platform size in pixels (12*2)

# === LEVEL GEOMETRY CONSTANTS ===
# Level dimensions (duplicated from map constants for convenience)
LEVEL_WIDTH_PX = 1056.0  # Level width in pixels
LEVEL_HEIGHT_PX = 600.0  # Level height in pixels
NORMALIZED_HEIGHT_DIVISOR = 600.0  # Height divisor for normalization

# === GRAPH BUILDING CONSTANTS ===
# Edge weights
BASE_EDGE_WEIGHT = 1.0  # Base weight for graph edges
BOUNCE_BLOCK_EDGE_WEIGHT_MULTIPLIER = 0.8  # Multiplier for bounce block edges
PLATFORM_EDGE_WEIGHT = 0.9  # Weight for platform edges

# Traversability
MIN_TRAVERSABLE_GAP = 18.0  # Minimum gap size for traversability
MAX_JUMP_DISTANCE = 200.0  # Increased maximum jump distance for better connectivity
MAX_FALL_DISTANCE = 400.0  # Increased maximum fall distance for better connectivity

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

# Speed limits (MAX_HOR_SPEED defined above as 9.0)

# Jump mechanics
MAX_JUMP_DURATION = 45

# Collision and damage constants
MAX_SURVIVABLE_IMPACT = 6
MIN_SURVIVABLE_CRUSHING = 0.05

# Jump velocities (exact values from sim_mechanics_doc.md)
# Floor Jump: From flat ground, applies velocity (0, -2)
JUMP_FLOOR_X = 0
JUMP_FLOOR_Y = -2

# Wall Jump (Regular): Applies velocity (1 * wall_normal, -1.4)
JUMP_WALL_REGULAR_X_MULTIPLIER = 1
JUMP_WALL_REGULAR_Y = -1.4

# Wall Jump (Slide): From wall slide state, applies velocity (2/3 * wall_normal, -1)
JUMP_WALL_SLIDE_X_MULTIPLIER = 2/3
JUMP_WALL_SLIDE_Y = -1

# Slope jump constants
JUMP_SLOPE_DOWNHILL_X_MULTIPLIER = 2/3
JUMP_SLOPE_DOWNHILL_Y_MULTIPLIER = 2
JUMP_SLOPE_UPHILL_DEFAULT_Y = -1.4
JUMP_SLOPE_UPHILL_PERP_X_MULTIPLIER = 2/3
JUMP_SLOPE_UPHILL_PERP_Y_MULTIPLIER = 2

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

# === COLLISION DETECTION CONSTANTS ===
COLLISION_EPSILON = 0.1  # Small value for collision detection precision
OVERLAP_THRESHOLD = 1.0  # Threshold for entity overlap detection

# === PERFORMANCE CONSTANTS ===
MAX_ENTITIES_PER_CALCULATION = 100  # Maximum entities to process in one calculation
CACHE_TIMEOUT_FRAMES = 60  # Number of frames to cache calculations