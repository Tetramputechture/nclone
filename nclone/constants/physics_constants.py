"""
Centralized physics constants for N++ simulation and RL integration.
All physics-related constants should be defined here to avoid duplication.
"""

import os
from pathlib import Path

# === ENTITY PHYSICS CONSTANTS ===
# Basic entity properties
NINJA_RADIUS = 10  # Ninja collision radius in pixels
NINJA_RADIUS_SQ = (
    NINJA_RADIUS * NINJA_RADIUS
)  # Precomputed squared radius for performance
DRONE_RADIUS = 7.5  # Drone collision radius in pixels
MINI_DRONE_RADIUS = 4.0  # Mini drone collision radius in pixels

# Entity collision radii (for pathfinding and reachability)
EXIT_SWITCH_RADIUS = 6  # Exit switch collision radius in pixels
EXIT_DOOR_RADIUS = 12  # Exit door collision radius in pixels
LOCKED_DOOR_SWITCH_RADIUS = 5  # Locked door switch collision radius in pixels
REGULAR_DOOR_RADIUS = 10  # Regular door collision radius in pixels
TRAP_DOOR_RADIUS = 5  # Trap door collision radius in pixels
GOLD_RADIUS = 6  # Gold collectible collision radius in pixels
LAUNCH_PAD_RADIUS = 6  # Launch pad collision radius in pixels
BOOST_PAD_RADIUS = 6  # Boost pad collision radius in pixels
LASER_RADIUS = 5.9  # Laser collision radius in pixels
DEATH_BALL_RADIUS = 5  # Death ball collision radius in pixels

# Grid and movement constants
DRONE_GRID_SIZE = 24  # Regular drone grid cell size in pixels
MINI_DRONE_GRID_SIZE = 12  # Mini drone grid cell size in pixels
DRONE_LAUNCH_SPEED = 4.0  # Drone launch speed in pixels/frame

# Thwump constants
THWUMP_SEMI_SIDE = 9  # Thwump semi-side size in pixels
THWUMP_FORWARD_SPEED = 20 / 7  # Thwump forward movement speed
THWUMP_BACKWARD_SPEED = 8 / 7  # Thwump backward movement speed

# Shove Thwump constants
SHOVE_THWUMP_SEMI_SIDE = 12  # Shove thwump semi-side size in pixels
SHOVE_THWUMP_PROJECTILE_RADIUS = 8  # Shove thwump projectile radius in pixels

# One-way platform constants
ONE_WAY_PLATFORM_SEMI_SIDE = 12  # One-way platform semi-side size in pixels

# Toggle mine radii by state (from entity_toggle_mine.py)
# 0:toggled, 1:untoggled, 2:toggling.
#
# The mine's `+0x28` reads **4.0 in every captured frame of every state** -- 8867
# samples in state 0, 2514 in state 1, 88 in state 2, and no other value. That
# looks like proof the 3.5/4.5 pair (which was `[I]`) is wrong, and it was
# briefly changed to a flat 4.0 on the strength of it.
#
# **It is not the collision radius.** A flat 4.0 costs 13 demos, all of them
# mine deaths the state-dependent radii avoid; `1006151` and `10131429` are the
# clearest. So `+0x28` is a stored or drawn radius that the contact test does not
# read, and the three values stay.
#
# Two process notes, because this one got shipped wrong for several rounds:
# the flat-4.0 change was measured on the **native** engine while it only ever
# affected **python** -- the Rust twin has its own `radius_for_state` in
# `entities/toggle_mine.rs`, so patching `constants::toggle_mine_radius` changed
# nothing there. The measurement said "exactly neutral" because it was measuring
# an engine the change had not reached, and the resulting python/native
# divergence was invisible to `ab_cmp` because that only replays `losers.txt`,
# which by construction contains no demo either engine wins.
#
# **Diff the two engines' win sets over `smoke.txt`, not `losers.txt`,** after
# any change that could touch only one of them.
# Toggle mine radii by state: 0=toggled(deadly), 1=untoggled(safe), 2=toggling.
#
# Only the state-0 entry is the KILL radius, and that one is confirmed: the
# mine's `+0x28` is what `FUN_10417F30` hands to `FUN_1050C650`, and
# `tools/npptrace/minefields.py` reads it as 4.0 over 36 captures and 326,539
# mine-frames. It is 4.0 in *every* state there -- but propagating that to all
# three entries costs 44 demos, because 3.5 and 4.5 do not govern the kill. They
# govern the two contact tests in `think` (untoggled -> toggling and toggling ->
# toggled), which live in a different function and do not read `+0x28`. The
# state machine they drive agrees with the game on 100% of 207,895 captured
# mine-frames, so leave them alone; `+0x28` simply is not the field behind them.
TOGGLE_MINE_RADII = {0: 4.0, 1: 3.5, 2: 4.5}

# Bounce block constants
BOUNCE_BLOCK_SEMI_SIDE = 4.5  # Bounce block semi-side size in pixels (9x9 pixel square)

# Movement speeds (exact values from sim_mechanics_doc.md)
MAX_HOR_SPEED = 3.333333333333333  # Maximum horizontal speed (10/3)
MAX_VER_SPEED = 12.0  # Maximum vertical speed (upward) of the ninja (player)
MIN_HORIZONTAL_VELOCITY = 0.1  # Minimum horizontal velocity for calculations

# Gravity and physics (exact values from sim_mechanics_doc.md)
GRAVITY_FALL = 0.06666666666666665  # Fall gravity
GRAVITY_JUMP = 0.01111111111111111  # Jump gravity
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
MIN_BOUNCE_BLOCK_MOVEMENT_DISTANCE = (
    10.0  # Minimum distance for bounce block movement detection
)
MIN_UPWARD_MOVEMENT_FOR_BOUNCE = (
    -10.0
)  # Minimum upward movement to consider bounce block

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
NINJA_STATE_CELEBRATING = 8  # Celebrating (level completed, indicates win)

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
MAX_JUMP_DISTANCE = 200.0  # Standard maximum jump distance in pixels
EXTENDED_JUMP_DISTANCE = 300.0  # Extended jump distance for gap crossing
MAX_FALL_DISTANCE = 400.0  # Increased maximum fall distance for better connectivity

# === ANIMATION CONSTANTS ===
# Project root path (parent directory of nclone package)
_PROJECT_ROOT = Path(__file__).parent.parent.parent

# Animation data file (always use absolute path from project root)
ANIM_DATA = os.path.join(_PROJECT_ROOT, "anim_data_line_new.txt.bin")

# Victory dance settings
DANCE_RANDOM = True  # Choose whether the victory dance is picked randomly.
DANCE_ID_DEFAULT = (
    0  # Choose the id of the dance that will always play if DANCE_RANDOM is false.
)

# Dance IDs (names of dances courtesy of Eddy, 0-12 are new, 13-20 are classics):
DANCE_DIC = {
    0: (104, 104),
    1: (106, 225),
    2: (226, 345),
    3: (346, 465),
    4: (466, 585),
    5: (586, 705),
    6: (706, 825),
    7: (826, 945),
    8: (946, 1065),
    9: (1066, 1185),
    10: (1186, 1305),
    11: (1306, 1485),
    12: (1486, 1605),
    13: (1606, 1664),
    14: (1665, 1731),
    15: (1732, 1810),
    16: (1811, 1852),
    17: (1853, 1946),
    18: (1947, 2004),
    19: (2005, 2156),
    20: (2157, 2241),
    21: (2242, 2295),
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
JUMP_FLOOR_Y = -2  # legacy alias: equals floor.y * JUMP_IMPULSE_Y_SCALE on flat ground

#: `Ninja::jump` (`docs/decomp/R5_07_jump_apply_105110C0.c:24-26`) scales its two
#: raw arguments by three ninja fields, evaluated **left to right**:
#:
#:     x_impulse = (field(+0xE8) * jx) * field(+0x1C8)
#:     y_impulse = ((field(+0xE8) + field(+0xF0)) * jy) * field(+0x1C8)
#:
#: The field values below were read live out of the running game at
#: `ninja + 0xE8 / 0xF0 / 0x1C8` (`docs/game_traces/`, frame 29 of 10131738).
#:
#: DO NOT collapse these to `jx * 2/3` and `jy * 2`. That is algebraically
#: identical and numerically is not: `(3 * fy) * (2/3)` and `2 * fy` differ by
#: 1 ULP for most `fy`, and this port used to do the latter. On frame 29 of
#: 10131738 that single last bit was the entire difference between winning and
#: dying -- see `docs/ENTITY_RE_ROUND8_FINDINGS.md` §3.
JUMP_SCALE_A = 1.0  # ninja + 0xE8
JUMP_SCALE_B = 2.0  # ninja + 0xF0
JUMP_SCALE_C = 2 / 3  # ninja + 0x1C8

#: Kept for callers that only need the effective magnitude. Never use these to
#: compute an impulse -- go through `Ninja._apply_jump` instead.
JUMP_IMPULSE_X_SCALE = 2 / 3
JUMP_IMPULSE_Y_SCALE = 2
#: `DAT_10586710 = -0.7`, the adverse-slope row, as passed to `Ninja::jump`
#: (`NINJA_update_10515300.c:352`). This is the RAW argument, not the impulse.
JUMP_UPSLOPE_Y_RAW = -0.7
JUMP_UPSLOPE_Y = JUMP_UPSLOPE_Y_RAW * JUMP_IMPULSE_Y_SCALE

# Wall jump rows, as passed to `Ninja::jump` at
# `docs/decomp/NINJA_update_10515300.c:387-396`:
#
#     jump(wall_normal * mult, field(+0x118) - sub, 1)
#
# with `(mult, sub) = (1.5, 0.7)` regular and `(1.0, 0.5)` for the slide-away
# case. These are RAW arguments -- the A/B/C scaling in `Ninja::jump` turns them
# into the effective impulses. `field(+0x118)` reads 0.0 in every captured frame,
# so the raw Y is just `-sub`.
#
# The effective impulses are -1.3999999999999997 and -0.9999999999999999, NOT the
# exact -1.4 and -1 this port used to apply. Frame 318 of 10131738.
JUMP_WALL_REGULAR_X_RAW_MULT = 1.5  # DAT_105729e8
JUMP_WALL_REGULAR_Y_RAW = -0.7  # 0.0 - DAT_105866b8
JUMP_WALL_SLIDE_X_RAW_MULT = 1.0  # DAT_1056c350
JUMP_WALL_SLIDE_Y_RAW = -0.5  # 0.0 - DAT_1056c348

#: Effective magnitudes, for callers that only need to reason about the size of
#: the impulse. Never use these to compute one.
JUMP_WALL_REGULAR_X_MULTIPLIER = 1
JUMP_WALL_REGULAR_Y = -1.4
JUMP_WALL_SLIDE_X_MULTIPLIER = 2 / 3
JUMP_WALL_SLIDE_Y = -1

# Slope jump constants
JUMP_SLOPE_DOWNHILL_X_MULTIPLIER = 2 / 3
JUMP_SLOPE_DOWNHILL_Y_MULTIPLIER = 2
JUMP_SLOPE_UPHILL_DEFAULT_Y = -1.4
JUMP_SLOPE_UPHILL_PERP_X_MULTIPLIER = 2 / 3
JUMP_SLOPE_UPHILL_PERP_Y_MULTIPLIER = 2

# Launch pad constants
JUMP_LAUNCH_PAD_BOOST_SCALAR = 1.7
JUMP_LAUNCH_PAD_BOOST_FACTOR = 2 / 3

# === RAGDOLL CONSTANTS ===
# Every value below is read out of npp.dll's .data / .rdata, not fitted:
# `tools/npptrace/dump_ragdoll_tables.py` prints the tables and
# `tools/npptrace/dump_consts.py <va>` prints any scalar. The decompiled
# ragdoll lives in `docs/decomp/full/ninja_105184A0.c` (integrate),
# `ninja_105193A0.c` (constraints), `ninja_10517310.c` (tiles),
# `ninja_10517100.c` (entities), `ninja_105188A0.c` (post) and
# `ninja_10516A80.c` (activate).

RAGDOLL_GRAVITY = 0.06666666666666665  # 0x10586518, == GRAVITY_FALL
RAGDOLL_DRAG = 0.99999  # 0x10586AE0

#: 13 bones. `0x10586AD0`.
RAGDOLL_NUM_BONES = 13

#: The ragdoll numbers its bones differently from the animation. `activate`
#: copies `ninja.bones[RAGDOLL_BONE_SOURCE[i]]` into ragdoll bone `i`
#: (`ninja_10516A80.c` lines 22-60). Everything in this module -- the stick
#: table, the angle table, the collision loops -- is in *ragdoll* order.
RAGDOLL_BONE_SOURCE = (12, 1, 0, 11, 9, 10, 8, 5, 3, 4, 2, 6, 7)

#: Inverse of the above: ragdoll bone index for a given animation bone index.
#: The renderer needs this because `render_utils.LIMBS` is in animation order.
RAGDOLL_BONE_OF_ANIM = (2, 1, 10, 8, 9, 7, 11, 12, 6, 4, 5, 3, 0)

#: The 11 sticks and their rest lengths, `0x10586730`, `{i32 a; i32 b; f64 len}`.
#: Expressed in animation-bone indices these are exactly `render_utils.LIMBS`.
RAGDOLL_SEGS = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (1, 5),
    (2, 6),
    (3, 7),
    (4, 8),
    (5, 9),
    (6, 10),
    (11, 12),
)
RAGDOLL_SEG_LENGTHS = (3.0, 3.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0)

#: The 9 angle constraints, `0x10586890`,
#: `{i32 centre; i32 arm_a; i32 arm_b; i32 pad; f64 target; f64 tolerance}`.
#: `target` is multiplied by the ninja's facing before use.
RAGDOLL_ANGLES = (
    (0, 1, 2, 3.141592653589793, 0.31415926535897953),
    (3, 1, 7, 1.7278759594743862, 1.1309733552923256),
    (4, 8, 2, 1.7278759594743862, 1.1309733552923256),
    (5, 1, 9, 1.7278759594743862, 1.1309733552923256),
    (6, 10, 2, 1.7278759594743862, 1.1309733552923256),
    (1, 0, 3, 5.497787143782138, 2.670353755551324),
    (2, 0, 4, 2.827433388230814, 1.8849555921538759),
    (1, 0, 5, 5.497787143782138, 2.670353755551324),
    (2, 0, 6, 2.827433388230814, 1.8849555921538759),
)

#: `0x105723A8`. The game's "two pi" is a *truncated* literal, not 2*math.pi;
#: the angle solver wraps against this exact value.
RAGDOLL_TWO_PI = 6.283185

# --- the exploded skeleton -------------------------------------------------
# `FUN_10517F80` (`docs/decomp/full/ninja_10517F80.c`) rebuilds the corpse as
# the *second* ragdoll kind: 17 bones, where 13..16 are fresh copies of the
# shoulder (1) and hip (2). The stick table then hangs each limb off its own
# copy instead of off the torso, so the arms and legs come away. Same two
# tables, second half: `0x105867E0` and `0x10586950`.

RAGDOLL_EXPLODED_NUM_BONES = 17

#: Which intact bone each of the four new bones is cloned from.
RAGDOLL_EXPLODED_BONE_SOURCE = (1, 2, 1, 2)

#: And the kick each one gets, in pixels per frame.
RAGDOLL_EXPLODED_KICKS = ((2.0, 0.0), (0.0, 4.0), (0.0, -6.0), (-8.0, 0.0))

RAGDOLL_EXPLODED_SEGS = (
    (0, 1),
    (0, 2),
    (13, 3),
    (14, 4),
    (15, 5),
    (16, 6),
    (3, 7),
    (4, 8),
    (5, 9),
    (6, 10),
    (11, 12),
)
RAGDOLL_EXPLODED_SEG_LENGTHS = RAGDOLL_SEG_LENGTHS

RAGDOLL_EXPLODED_ANGLES = (
    (0, 1, 2, 1.7278759594743862, 1.1309733552923256),
    (3, 13, 7, 1.7278759594743862, 1.1309733552923256),
    (4, 8, 14, 1.7278759594743862, 1.1309733552923256),
    (5, 15, 9, 1.7278759594743862, 1.1309733552923256),
    (6, 10, 16, 1.7278759594743862, 1.1309733552923256),
)

#: A death only bursts the corpse when its kill type maps to category 0 or 4
#: (`0x10586520`, applied in `ninja_10515F90.c` line 62). Resolving that table
#: through the entity->kill-type table at `0x10586CE0` leaves exactly these:
#: both mine types (1, 21), the rocket (18), the evil ninja (22) and the
#: player-controlled rocket (29). Drones, lasers, thwumps, the death ball,
#: falling impacts and crushes all leave the corpse intact.
RAGDOLL_EXPLODING_DEATH_CAUSES = frozenset(
    {"mine", "rocket", "evil_ninja", "pc_rocket"}
)

# --- blast, an nclone extension -------------------------------------------
# The game's own `explode` only nudges the four cloned bones (see
# RAGDOLL_EXPLODED_KICKS); the outward burst you see on screen is particles it
# spawns, which nclone does not draw. These add a real radial impulse instead,
# so the corpse actually comes apart. Set RAGDOLL_EXPLOSION_FORCE to 0.0 to get
# the game's literal behaviour back.

#: Peak outward speed, in pixels per frame, added to a bone at the blast centre.
RAGDOLL_EXPLOSION_FORCE = 8.0

#: Distance over which the blast decays. Beyond it a bone still gets the floor
#: below, so a limb across the room is thrown too, just less hard.
RAGDOLL_EXPLOSION_RADIUS = 24.0

#: Fraction of the peak that survives at maximum distance.
RAGDOLL_EXPLOSION_FLOOR = 0.25

#: A flying limb arms and sets off any live mine it touches, which bursts the
#: corpse again and throws the pieces on. This is the bone-vs-entity collision
#: the game runs in `FUN_105188A0` (the branch that checks for entity types
#: 0x01 and 0x15), narrowed to the case nclone can draw. Slack added to the
#: mine's own radius before a bone counts as touching it.
RAGDOLL_MINE_TOUCH_SLACK = 1.0

#: Stop after this many chained detonations in one frame, so a dense minefield
#: cannot spin forever on a single tick.
RAGDOLL_MAX_CHAIN_PER_FRAME = 8

RAGDOLL_BONE_RADIUS = 1.0  # 0x1056C350, depenetration radius of one bone
RAGDOLL_SWEEP_RADIUS = 0.5  # 0x1056C348, radius used for the per-bone sweep
RAGDOLL_STICK_RADIUS = 0.9  # 0x10571CD8, half-thickness of a limb
RAGDOLL_GATHER_RADIUS = 10.0  # 0x10571D08, segment-gather box for a bone
RAGDOLL_SWEEP_MIN_DIST_SQ = 0.1  # 0x10571D98, below this the sweep is skipped
RAGDOLL_MAX_DEPEN_ITERS = 32  # 0x20, per bone per pass
RAGDOLL_SLIDE_SCALE = 1.5  # 0x105729E8, tangential-correction clamp
RAGDOLL_DEGENERATE_SQ = 0.01  # 0x10571CB8, shared-corner tolerance
RAGDOLL_EPSILON = 2.220446049250313e-16  # 0x105705C0

#: Bounce off an accumulated contact normal: `v -= 1.8 * (v.n) * n` (so a
#: restitution of 0.8) with `0.1` of the tangential component removed.
#: `0x10586CA0` / `0x10571D98`, `ninja_105188A0.c` lines 100-105.
RAGDOLL_BOUNCE = 1.8
RAGDOLL_FRICTION = 0.1

#: `activate` scales the death impulse by `(1 - min(dist/12, 1)) * 1.5 + 0.5`.
#: `0x1056C358` / `0x105729E8` / `0x1056C348`, `ninja_10516A80.c` lines 119-123.
RAGDOLL_IMPULSE_FALLOFF = 12.0
RAGDOLL_IMPULSE_NEAR = 1.5
RAGDOLL_IMPULSE_FAR = 0.5

#: `activate` finishes by kicking the two head bones with a uniform random in
#: [0, 2). The game draws from its global RNG; we use a private, deterministic
#: one so replays stay reproducible (see `Ragdoll.reseed`).
RAGDOLL_HEAD_KICK = 2.0
RAGDOLL_HEAD_KICK_SEED = 0x4E505044  # b"NPPD"

#: Sentinels of the ragdoll's time-of-impact helpers, which do *not* share the
#: ninja's 0/1 convention: `-1` means "already overlapping at t=0" and `2`
#: means "no impact this step". `0x10572548` / `0x10571CE8`,
#: `docs/decomp/R5_04_toi_seg_1050C3A0.c`.
RAGDOLL_TOI_OVERLAP = -1.0
RAGDOLL_TOI_MISS = 2.0

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

# === MINE DEATH PREDICTION CONSTANTS (HYBRID APPROACH) ===
# Tier 1: Spatial danger zone grid
MINE_DANGER_ZONE_RADIUS = (
    80.0  # Pixels - cells within this radius marked as danger zone
)
MINE_DANGER_ZONE_CELL_SIZE = 24  # Pixels - grid cell size (matches game tiles)

# Tier 2: Distance-based threshold for simulation
MINE_DANGER_THRESHOLD = 30.0  # Pixels - distance threshold to trigger full simulation

# Tier 3: Physics simulation parameters
MINE_DEATH_LOOKAHEAD_FRAMES = 6  # Number of frames to simulate for collision detection

# Collision detection
MINE_COLLISION_THRESHOLD = 12.5  # Tuned value for accurate collision detection (prevents false positive masking)

# === TERMINAL VELOCITY MASKING CONSTANTS ===
# Prevents agent from selecting actions leading to terminal impact death
# (collision with floor/ceiling at speeds exceeding MAX_SURVIVABLE_IMPACT)

# Tier 1: Quick state filter
TERMINAL_IMPACT_SAFE_VELOCITY = (
    3  # Below this velocity, skip expensive checks (px/frame)
)

# Tier 2: Lookup table quantization
TERMINAL_VELOCITY_QUANTIZATION = (
    0.5  # Velocity quantization for lookup table (px/frame)
)
TERMINAL_DISTANCE_QUANTIZATION = (
    12  # Distance quantization for lookup table (pixels, half-tile)
)

# Tier 3: Physics simulation
TERMINAL_IMPACT_SIMULATION_FRAMES = (
    30  # Max frames to simulate forward for impact prediction
)

# === DETERMINISTIC PATH MASKING CONSTANTS ===
# Masks actions that are provably suboptimal on trivial straight paths
# This ensures 100% success on simple corridor scenarios

# Path straightness detection thresholds
STRAIGHT_PATH_HORIZONTAL_THRESHOLD = 0.8  # |dx| must exceed this for horizontal path
STRAIGHT_PATH_VERTICAL_THRESHOLD = 0.1  # |dy| must be below this for horizontal path
STRAIGHT_PATH_DOWNWARD_THRESHOLD = (
    0.3  # dy must exceed this for downward path (positive = down)
)
STRAIGHT_PATH_MIN_DISTANCE = 12.0  # Only apply masking for goals >12px away (half-tile)

# Straightness masking: extend path with switch-to-exit when switch is this close
MIN_SWITCH_HOPS_FOR_PATH_EXTENSION = (
    4  # If path to switch < 4 hops, extend with switch-to-exit path
)

# --- animation constants that became physics inputs when the limb test landed ---
#: DAT_105866e0, added to the atan2 result in ull/ninja_10516280.c:35. A
#: STORED 64-bit constant, not M_PI_2 -- it differs from pi/2 by ~4.9e-12.
#: Round 7 §3.3 asked for it verbatim; round 9 §2.3 confirmed it from the binary.
TILT_QUARTER_TURN = 1.57079632679
#: DAT_10571d98. The airborne tilt decay is  - a*0.1 (:71-72), NOT
#:  *= 0.9; there is no 0.9 constant in that function. Round 9 §2.1.
TILT_DECAY = 0.1
