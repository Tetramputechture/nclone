"""Constants for the gym environment."""

import numpy as np

# Max time in frames per level before episode truncation (fallback)
# Note: Actual truncation limits are now calculated dynamically per level
# based on PBRS surface area and reachable mine count
MAX_TIME_IN_FRAMES = (
    4500  # Cap on the dynamic truncation limit (was 10000). Tightened so failed
    # episodes (mostly timeouts) end ~2x sooner for throughput; still ~75s, far
    # above a typical solve. Also the fallback when dynamic calc is unavailable.
)

# Game state feature dimensions (streamlined, no redundancies)
# Version 2.1 (Feb 2026): Reduced from 26 to 24 dims by removing hor_input, jump_input
# Features: velocity(2) + movement_state(4) + airborne(1) + buffers(4) +
#           contacts(3) + normals(3) + walled(1) + physics(3) + position(2) +
#           time_remaining(1) = 24 (get_ninja_state returns 23; time_remaining
#           is appended in base_environment to make NINJA_STATE_DIM).
# Physics-derived (3): horizontal_range_norm, max_jump_height_norm, time_to_ground_norm
NINJA_STATE_DIM = 24  # 23 ninja physics + time_remaining (appended in base env)
PHYSICS_DERIVED_DIM = 3
# L1.4 (NPP-12): append switch_activated (0/1) for phase indicator in the policy MLP stream.
GAME_STATE_SWITCH_ACTIVATED_IDX = NINJA_STATE_DIM + PHYSICS_DERIVED_DIM  # 27
# Goal geometry: ninja-relative vectors to the exit switch and exit door, each
# (dx, dy) normalized by LEVEL_WIDTH / LEVEL_HEIGHT and clipped to [-1, 1]. Static
# entities sit on a 6px grid, so these are exact. The switch_activated phase bit
# selects which one is the live objective; exposing both keeps the policy's goal
# geometry continuous across the switch->exit transition.
GAME_STATE_SWITCH_VEC_IDX = GAME_STATE_SWITCH_ACTIVATED_IDX + 1   # 28: (dx, dy) to switch
GAME_STATE_EXIT_VEC_IDX = GAME_STATE_SWITCH_VEC_IDX + 2           # 30: (dx, dy) to exit
# Surface ray-casts: normalized distance [0, 1] to the nearest grounded platform
# in the 8 compass directions (E, NE, N, NW, W, SW, S, SE); 0 = adjacent, 1 = no
# platform within 500px. Lets the policy verify landing surfaces / gaps beyond the
# 24px tile-grid view (blind-jump verification) without reading them out of tiles.
GAME_STATE_RAYCAST_IDX = GAME_STATE_EXIT_VEC_IDX + 2             # 32: 8 compass distances
GAME_STATE_RAYCAST_DIM = 8
# Free-space navigation hint: physics-free steepest-descent over the mine-aware
# free-space cost-to-go toward the current objective. 4 dims [dx_hop, dy_hop,
# cost_to_go_norm, phase] — the next-step direction that ROUNDS obstacles (unlike
# the straight-line goal vectors above), plus normalized remaining cost.
GAME_STATE_FREE_SPACE_NAV_IDX = GAME_STATE_RAYCAST_IDX + GAME_STATE_RAYCAST_DIM  # 40
GAME_STATE_FREE_SPACE_NAV_DIM = 4
# Nearest deadly-mine distance, normalized [0, 1] (0 = touching, 1 = none within
# the local normalizer). Scalar danger gauge for the MLP stream — complements the
# spatial deadly-mine hazard SDF (CNN). Computed from the already-gathered mines.
GAME_STATE_NEAREST_MINE_IDX = GAME_STATE_FREE_SPACE_NAV_IDX + GAME_STATE_FREE_SPACE_NAV_DIM  # 44
# Closing speed toward the current objective: dot(velocity, unit goal direction) /
# MAX_HOR_SPEED, clipped [-1, 1]. +1 = approaching the live goal at full speed,
# -1 = receding. Derived from values already in the vector (velocity + goal geom).
GAME_STATE_CLOSING_SPEED_IDX = GAME_STATE_NEAREST_MINE_IDX + 1  # 45
# Jump-execution features (help precise jump / wall-jump timing for the FEED-FORWARD
# policy, which cannot track its own action history). Appended at the END so no
# existing index ever shifts:
#   jump_held    : was jump held last frame — jumps fire only on the RISING edge
#                  (ninja.py: new_jump_check = jump_input_old == 0), so a held button
#                  does NOT re-trigger; needed to chain jumps / time a wall-jump.
#   jump_useless : would pressing jump NOW do nothing (airborne, not jumping, holding
#                  jump, no active buffer) — a graded "jump-ready" signal robust to
#                  action-mask annealing.
#   launchpad_buffer : the launch-pad timing window (the 4th buffer; the other three
#                  are already in the ninja-state block).
GAME_STATE_JUMP_HELD_IDX = GAME_STATE_CLOSING_SPEED_IDX + 1        # 46
GAME_STATE_JUMP_USELESS_IDX = GAME_STATE_JUMP_HELD_IDX + 1         # 47
GAME_STATE_LAUNCHPAD_BUFFER_IDX = GAME_STATE_JUMP_USELESS_IDX + 1  # 48
GAME_STATE_JUMP_EXEC_DIM = 3
GAME_STATE_CHANNELS = (
    NINJA_STATE_DIM + PHYSICS_DERIVED_DIM + 1 + 2 + 2
    + GAME_STATE_RAYCAST_DIM + GAME_STATE_FREE_SPACE_NAV_DIM + 1 + 1
    + GAME_STATE_JUMP_EXEC_DIM
)  # 49

# Switch states array (legacy format, still used internally by reward/PBRS code)
MAX_LOCKED_DOORS = 5  # Maximum doors for switch_states array
FEATURES_PER_DOOR = 5  # [switch_x, switch_y, door_x, door_y, collected]
SWITCH_STATES_DIM = MAX_LOCKED_DOORS * FEATURES_PER_DOOR  # 25 total

# Locked door array for objective attention (supports 1-16 doors) — still used by
# debug/overlay code even though it is no longer exposed via the policy obs dict.
MAX_LOCKED_DOORS_ATTENTION = 16
LOCKED_DOOR_FEATURES_DIM = 8  # [switch_x, switch_y, switch_collected, switch_path_dist, door_x, door_y, door_open, door_path_dist]
LOCKED_DOOR_ARRAY_SIZE = MAX_LOCKED_DOORS_ATTENTION * LOCKED_DOOR_FEATURES_DIM  # 128

# L1 toggle-mines hazard map (NPP-46): 2-channel mine SDF on a 6px grid over the
# 1008×552 px level. This is the Layer 1 policy-facing key.
#   ch0 = deadly (toggled) mine SDF
#   ch1 = safe (untoggled) mine SDF
# Both are `clip((d - 14) / 24, -1, 1)` where d is the px distance to the nearest
# mine of that class. Shape is (C, W, H) — the SAME axis order as `tile_grid` and
# `entity_overlays` (W=168, H=92), so all spatial tensors co-register without a
# transpose. The Rust engine emits a row-major (H, W) buffer that the env
# transposes to (W, H) before stacking.
HAZARD_MAP_CHANNELS = 2
HAZARD_MAP_WIDTH = 168  # 1008 / 6
HAZARD_MAP_HEIGHT = 92  # 552 / 6

# Optional lattice-d* navigation hint (policy obs key ``lattice_nav``), enabled
# via ReachabilityConfig.expose_lattice_features.  Layout (4 dims):
#   [dx_hop, dy_hop, cost_to_go_norm, phase] — see value_function/lattice_distance.
LATTICE_NAV_DIM = 4

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
