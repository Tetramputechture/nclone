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
# Slope/curve EXECUTION features (help precise slope landing/traversal — the
# policy is blind to the angle of a surface until it is standing on it):
#   landing_normal : the outward surface normal (nx, ny) of the slope/curve the
#                  ninja is moving toward (look-ahead along velocity / fall),
#                  from the EXACT tile geometry (tile_surface_normals); flat
#                  ground -> (0, -1). Lets the airborne policy pre-orient for a
#                  sloped landing — most deaths are impact/fall, not navigation.
#   landing_dist : normalized distance to that surface (0 = adjacent, 1 = none
#                  within range).
#   vertical_terminal : signed vertical speed normalized by the vertical terminal
#                  cap MAX_VER_SPEED (+1 = at terminal UPWARD, -1 = terminal
#                  downward). The base velocity dims saturate (normalized by the
#                  much smaller horizontal cap), so this exposes the unsaturated
#                  vertical-speed / terminal-velocity signal.
GAME_STATE_LANDING_NORMAL_IDX = GAME_STATE_LAUNCHPAD_BUFFER_IDX + 1    # 49: (nx, ny)
GAME_STATE_LANDING_DIST_IDX = GAME_STATE_LANDING_NORMAL_IDX + 2        # 51
GAME_STATE_VERTICAL_TERMINAL_IDX = GAME_STATE_LANDING_DIST_IDX + 1     # 52
GAME_STATE_SLOPE_EXEC_DIM = 4  # landing_normal(2) + landing_dist(1) + vertical_terminal(1)
# Execution-timing scalars (cheap ninja-state derived; help precise jump/landing):
#   time_to_contact : tau = px distance to the look-ahead landing surface / closing
#                  speed into it, normalized (1 = far / not approaching, ~0 =
#                  imminent contact). Tau-theory timing signal for the dominant
#                  impact-death failure mode.
#   frames_airborne : consecutive airborne frames, normalized — a fall-height /
#                  impact-energy proxy (pairs with MAX_SURVIVABLE_IMPACT).
#   acceleration : per-frame velocity delta (ax, ay) from xspeed_old/yspeed_old —
#                  distinguishes being pushed / decelerated (explicit accel beats
#                  relying on frame-stacking; Flare/Hart).
GAME_STATE_TIME_TO_CONTACT_IDX = GAME_STATE_VERTICAL_TERMINAL_IDX + 1  # 53
GAME_STATE_FRAMES_AIRBORNE_IDX = GAME_STATE_TIME_TO_CONTACT_IDX + 1    # 54
GAME_STATE_ACCEL_IDX = GAME_STATE_FRAMES_AIRBORNE_IDX + 1              # 55: (ax, ay)
GAME_STATE_EXEC_EXTRA_DIM = 4  # tau(1) + frames_airborne(1) + accel(2)
# Forward-biased terrain "scandots" (the strongest locomotion feature for rough
# terrain in legged robotics: Rudin'21 / Agarwal-Kumar'22 — anticipatory traversal
# of slopes/curves/gaps). For a fan of tile columns ahead of the ninja (biased by
# travel direction), the tile-distance DOWN to the next floor and UP to the next
# ceiling, RELATIVE to the ninja (ego-centric, so it generalizes across absolute
# position). Lets the policy pre-position for an upcoming slope/curve/gap/landing.
GAME_STATE_SCANDOTS_IDX = GAME_STATE_ACCEL_IDX + 2                     # 57
GAME_STATE_SCANDOTS_COLUMNS = 6  # offsets [-1,0,1,2,3,4] * travel direction
GAME_STATE_SCANDOTS_DIM = GAME_STATE_SCANDOTS_COLUMNS * 2  # (floor_gap, ceil_gap)/col
# Ghost-arc trajectory-projection bundle (Audit II P0-1): for the three
# constant-input hypotheses u in {-1, 0, +1}, the native TrajectoryProjector
# integrates the REAL airborne recurrence to first contact (solid tile /
# deadly-mine radius / frame cap) and emits 13 dims per arc (v15; was 9):
#   [dx/500, dy/500, nx, ny, mine_flag, none_flag, clear_width/9,
#    impact_ratio/2, tau/60, vx_contact/MAX_HOR_SPEED, vy_contact/10,
#    surface_class, mine_proximity]
# The u=+-1 arcs are the bang-bang controllability-envelope endpoints (the
# reachable-landing interval); clear_width is the contiguous mine-free
# standable extent at the contact — previously unrepresented at ANY range.
# v15 APPEND-ONLY additions (indices 9-12 per arc; 0-8 byte-identical to v14):
#   [9]  vx_contact / MAX_HOR_SPEED  — full signed horizontal landing velocity
#   [10] vy_contact / 10.0           — full signed vertical landing velocity
#        (10.0 ≈ drag-terminal fall speed). Together with [9] these expose the
#        TANGENTIAL/sliding component that impact_ratio[7] (normal-only) drops —
#        the F5 landing-velocity-control feature.
#   [11] surface_class: +1 floor / 0 wall / -1 ceiling (0 for mine/none) —
#        makes "will land on a standable surface" vs "will smack a wall/ceiling"
#        explicit instead of a threshold on ny. (No raw normal-angle scalar:
#        (nx,ny) at [2]/[3] already is (cosθ,sinθ) — an angle would be redundant
#        and add a wraparound discontinuity.)
#   [12] mine_proximity: 0 = the arc's flight never comes near a deadly mine,
#        ->1 as its CLOSEST approach nears the kill radius. Graded threading
#        margin complementing the binary mine_flag[4] — for the SURVIVING arcs
#        it says how much mine clearance each steering choice keeps. Targets the
#        high-tier mine-death failure mass (held-out eval: VERTICAL_SHAFT/T3
#        mine 0.53, MULTI_JUMP/T4 0.40, PRECISION/T3 0.17).
#   [13] goal_approach (v16 C): 0 = the ACTIVE objective (switch pre / exit post)
#        is beyond this arc's flight-proximity range, ->1 as the arc's closest
#        approach nears the goal-collection radius. The action-conditioned
#        "am I getting warmer" gradient — the symmetric reward-side companion to
#        mine_proximity[12] (danger-side). 0 when no goal is set.
#   [14] goal_hit (v16 C, 0/1): this arc's flight passes within the goal
#        collection radius of the active objective — the decisive "this
#        commitment REACHES the objective" bit. Mostly-zero on long multi-jump
#        levels; decisive on the final approach + compact precision/chamber
#        levels (v15 held-out: CHAMBER timeouts + high-tier precision).
# Grounded frames use a fast-path fill (self-contact: zeros + ny=-1).
GAME_STATE_GHOST_ARC_IDX = GAME_STATE_SCANDOTS_IDX + GAME_STATE_SCANDOTS_DIM  # 69
GAME_STATE_GHOST_ARC_ARCS = 3
# v16 (feature C): 15 dims/arc (was 13; +goal_approach[13], +goal_hit[14]).
GAME_STATE_GHOST_ARC_PER_ARC = 15
GAME_STATE_GHOST_ARC_DIM = GAME_STATE_GHOST_ARC_ARCS * GAME_STATE_GHOST_ARC_PER_ARC  # 45
# Frame horizon for ghost arcs. A full-height fall from rest takes ~130-140
# frames under drag (slow start), so 180 covers any single arc in the 42x23
# field with margin; longer means "no contact within 3 seconds" = none-flag.
GHOST_ARC_MAX_FRAMES = 180

# Grounded JUMP-NOW arc bundle (v13): the counterfactual "if I jumped this
# frame" arcs — flat-floor jump impulse (yspeed -> min(yspeed,0) + JUMP_FLOOR_Y)
# applied to the current velocity, full-hold reduced gravity
# (jf = MAX_JUMP_DURATION), then the same three u in {-1, 0, +1} projections
# and the same 9 dims/arc as the committed bundle. This is the takeoff-frame
# affordance the committed bundle cannot provide (it is zeros on the ground):
# landing point / mine / survivability / ledge width BEFORE committing a jump.
# Slope takeoffs use the flat-floor impulse as an approximation (slope jump
# impulses are direction/input-dependent; the flat model is exact on the
# precision-jump geometry this exists for). Airborne frames are zero-filled
# (the committed bundle covers the air).
GAME_STATE_JUMP_ARC_IDX = GAME_STATE_GHOST_ARC_IDX + GAME_STATE_GHOST_ARC_DIM  # 114
GAME_STATE_JUMP_ARC_DIM = GAME_STATE_GHOST_ARC_DIM  # 45

# Per-arc route-progress delta (v13): for each of the 6 arcs (3 committed +
# 3 jump-now, in bundle order), tanh((Phi(ninja) - Phi(arc landing)) / 150) —
# positive = the arc's landing IMPROVES the committed free-space cost-to-go
# (route progress), negative = it loses ground, 0 = no arc / no source /
# non-finite landing cost. Collapses "is this jump useful for the route"
# (previously only inferable by correlating arc dx/dy against the cost-to-go
# map channel) into one scalar per arc.
GAME_STATE_ARC_PHI_IDX = GAME_STATE_JUMP_ARC_IDX + GAME_STATE_JUMP_ARC_DIM  # 159
GAME_STATE_ARC_PHI_DIM = 6
ARC_PHI_NORM_COST = 150.0

# TAP-JUMP arc row (v14, V13_NEXT_STEPS B1): the v13 jump-now arcs are
# FULL-HOLD only (jf = MAX_JUMP_DURATION). Under a low ceiling — LOOP_CORRIDOR's
# mined walkway allows only a 48px-headroom hop over each armed mine — all
# three full-hold arcs slam the ceiling, so the move that actually works
# there (a tap jump) is invisible to the policy. Same flat-floor takeoff
# impulse, jf = 0 (jump released immediately -> full gravity after the
# impulse), the same three u in {-1, 0, +1} projections and the same 9
# dims/arc. Airborne frames zero-filled (the committed bundle covers the
# air), mirroring the full-hold row.
GAME_STATE_TAP_ARC_IDX = GAME_STATE_ARC_PHI_IDX + GAME_STATE_ARC_PHI_DIM  # 165
GAME_STATE_TAP_ARC_DIM = GAME_STATE_GHOST_ARC_DIM  # 45

# WALL-JUMP-NOW arc row (v14, follows B1): the counterfactual "if I wall-jump
# this frame" arcs while AIRBORNE + WALLED (wall-slide / wall contact). The
# committed bundle on those frames shows only the slide/fall continuation —
# the kick affordance was invisible, exactly the B1 blindness for the third
# contact regime. Regular wall-jump impulse as ``Ninja.wall_jump`` applies it
# (xspeed zeroed if into the wall, downward yspeed zeroed, then
# +JUMP_WALL_REGULAR_X_MULTIPLIER*wall_normal / JUMP_WALL_REGULAR_Y, position
# displaced by the impulse), full hold (jf = MAX_JUMP_DURATION), same three
# u in {-1, 0, +1} projections x 9 dims. Zeros when grounded or un-walled.
# Targets the wall-jump chains in LOOP_CORRIDOR shafts / VERTICAL_SHAFT.
GAME_STATE_WALL_ARC_IDX = GAME_STATE_TAP_ARC_IDX + GAME_STATE_TAP_ARC_DIM  # 210
GAME_STATE_WALL_ARC_DIM = GAME_STATE_GHOST_ARC_DIM  # 45

# PASSIVE no-input arc (v16): the "if I hold NO inputs from this frame" fall.
# The committed bundle's u=0 arc is the no-HORIZONTAL-input arc, but it inherits
# the CURRENT jump-hold gravity (jf>0 while the jump is held) — so while the
# ninja is holding jump, the committed u=0 shows the continued-hold path, NOT the
# release path. This block always projects jf=0, u=0 (jump released + no steer =
# the true do-nothing outcome), every airborne frame, so the agent always knows
# the baseline trajectory it defaults to and whether letting go is safe. Six
# policy-ready dims:
#   [0] landing dx / 500 px, clipped [-1, 1]
#   [1] landing dy / 500 px, clipped [-1, 1]
#   [2] landing vx / MAX_HOR_SPEED, clipped [-1, 1]
#   [3] landing vy / 10.0 (drag-terminal), clipped [-1, 1]
#   [4] fatal flag (0/1): 1 = the no-input landing kills (deadly-mine contact OR
#       impact speed >= the surface's survivable threshold). The EXPLICIT
#       "doing nothing is lethal" signal (previously only inferable by
#       thresholding the committed u=0 impact ratio, and unavailable at all while
#       holding jump).
#   [5] no-contact-within-horizon flag (0/1): the fall did not reach a surface
#       within GHOST_ARC_MAX_FRAMES (still airborne at the horizon).
# Zeros when grounded (no passive fall to project). Reuses the native
# TrajectoryProjector.project(u=0, jf=0) — no native change; filled in python.
GAME_STATE_PASSIVE_ARC_IDX = GAME_STATE_WALL_ARC_IDX + GAME_STATE_WALL_ARC_DIM  # 255
GAME_STATE_PASSIVE_ARC_DIM = 6

# Coarse far ego crop (Audit II P1-6, ``ego_crop_coarse``): 21x21 window of
# 2x2-tile pooled cells centered on the ninja — +-21 tiles Chebyshev reach.
# Channels: [solid fraction, deadly-mine density, safe-mine density,
# objective marker (1.0 = active objective, 0.5 = the other)].
EGO_CROP_COARSE_SIZE = 21
EGO_CROP_COARSE_POOL_TILES = 2
EGO_CROP_COARSE_CHANNELS = 4
# v17 P1: optional +2 route channels appended to the coarse crop (when
# ego_crop_coarse_route is on) — the 2-tile-pooled steepest-descent UNIT vector
# (ux, uy) of the ACTIVE-phase cost-to-go, i.e. the SAME grad(d*) route signal as
# the fine ego_crop_route but at the coarse crop's +-21-tile maze scale. The base
# coarse crop only carries a target BEARING (objective marker) + solidity; for a
# switch beyond the fine crop's +-10-tile horizon this gives the DIRECTED route
# (which corridor leads there), the biggest maze-scale obs gap. Total coarse
# channels = 4 (+2 when route on) = 6.
EGO_CROP_COARSE_ROUTE_CHANNELS = 2

# Ego route-gradient crop (v14, V13_NEXT_STEPS B2): 2 channels co-registered
# with the tile ego_crop — the local steepest-descent UNIT vector (ux, uy) of
# the ACTIVE-phase cost-to-go field (the committed free-space/jump nav source)
# at tile resolution, in the ninja-centered window. Removes the
# self-localization burden of the global 12px-downsampled cost map (the policy
# had to find itself on a 168px-per-readout-cell global grid to follow the
# route); this is the Markov-consistent alternative to memory — make the map
# LEGIBLE, don't remember it. Zeros where solid / unreachable / at the goal.
EGO_CROP_ROUTE_CHANNELS = 2

# v22 §3-C fine ego route crop (``ego_crop_route_fine``): the steepest-descent unit
# vector (ux, uy) of the ACTIVE-phase 6px UN-POOLED free-space cost-to-go field in a
# small ninja-centered window. 2 channels; the window is EGO_CROP_ROUTE_FINE_SIZE
# cells on a side at 6px/cell (~half a tile per cell), so the go-around vector stays
# crisp one 6px cell behind a thin wall (where the tile-min-pooled ego_crop_route
# goes degenerate). Odd size so the ninja sits on the exact center cell.
EGO_CROP_ROUTE_FINE_CHANNELS = 2
EGO_CROP_ROUTE_FINE_SIZE = 21  # 21 * 6px = 126px window (~+-2.6 tiles)

GAME_STATE_CHANNELS = (
    NINJA_STATE_DIM + PHYSICS_DERIVED_DIM + 1 + 2 + 2
    + GAME_STATE_RAYCAST_DIM + GAME_STATE_FREE_SPACE_NAV_DIM + 1 + 1
    + GAME_STATE_JUMP_EXEC_DIM + GAME_STATE_SLOPE_EXEC_DIM
    + GAME_STATE_EXEC_EXTRA_DIM + GAME_STATE_SCANDOTS_DIM
    + GAME_STATE_GHOST_ARC_DIM
    + GAME_STATE_JUMP_ARC_DIM + GAME_STATE_ARC_PHI_DIM
    + GAME_STATE_TAP_ARC_DIM + GAME_STATE_WALL_ARC_DIM
    + GAME_STATE_PASSIVE_ARC_DIM
)  # 237

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

# Optional cost-to-go map obs (``cost_to_go_map``) is projected at a COARSER grid
# than the 6px hazard map. Its encoder (_CostToGoEncoder) pools to a 3x3 readout
# regardless of input resolution, so a 2x (12px) downsample is ~lossless for the
# routing signal it carries while cutting the obs payload ~4x — and it is the
# single largest observation (dominates the per-step deepcopy/slice/buffer/pipe
# cost). MUST divide HAZARD_MAP_WIDTH/HEIGHT evenly.
COST_TO_GO_MAP_DOWNSAMPLE = 2
COST_TO_GO_MAP_WIDTH = HAZARD_MAP_WIDTH // COST_TO_GO_MAP_DOWNSAMPLE  # 84
COST_TO_GO_MAP_HEIGHT = HAZARD_MAP_HEIGHT // COST_TO_GO_MAP_DOWNSAMPLE  # 46

# v17 episodic visitation map (``visitation_map``): a per-episode "where have I
# been" grid co-registered EXACTLY with cost_to_go_map (same 12px cells, same
# border-stripped playable frame, same (W, H) axis order). One channel carrying a
# recency-decayed visit signal per cell (1.0 = just here, ->0 = long ago / never).
# Externalizes the episodic memory a recurrent policy would otherwise need so a
# Markov policy can tell visited junctions/dead-ends from unexplored ones in
# tree-structured mazes (CHAMBER_MAZE). Shares the same underlying per-episode
# visit-count grid that drives the count-based visitation reward bonus.
VISITATION_MAP_WIDTH = COST_TO_GO_MAP_WIDTH  # 84 (12px cells)
VISITATION_MAP_HEIGHT = COST_TO_GO_MAP_HEIGHT  # 46
VISITATION_MAP_CHANNELS = 1
# Cell size in raw sim px and the playable-frame origin offset (border strip),
# derived to co-register with cost_to_go_map's downsampled grid. The hazard grid
# is 6px/cell; cost_to_go downsamples by 2 -> 12px cells, offset past the sim's
# 24px (1-tile) solid border.
VISITATION_CELL_PX = 6 * COST_TO_GO_MAP_DOWNSAMPLE  # 12px
VISITATION_ORIGIN_PX = 24  # the sim's 1-tile solid border
# v17 P2 directed-exploration floor: when the visitation bonus is d*-directed
# (visitation_directed), a cell's novelty is scaled by max(FLOOR, 1 - d*_norm)
# where d*_norm is the per-level-normalized cost-to-go toward the active goal.
# The floor keeps a baseline count-based bonus everywhere (so far-from-goal
# frontier is still explored) while amplifying novelty along the route to the
# switch/exit and damping it in dead-ends (which increase d*). Normalization is
# per-level (by the field's finite max), NOT the map's fixed K=1000 — maze d*
# costs are ~15-50 cell-steps, so K=1000 would flatten the weight to ~1.0.
VISIT_DIRECTED_FLOOR = 0.25

# v18 C1 spatial frontier map (``frontier_map``): the reachable-but-unvisited
# set as a spatial channel, co-registered EXACTLY with cost_to_go_map /
# visitation_map (same 12px cells, same border-stripped playable frame, same
# (W, H) axis order). frontier = reachable AND (episodic visit count == 0) —
# the one directed, non-myopic coverage signal a memoryless feedforward policy
# lacks: "where is the reachable space I have NOT searched yet". Computed
# vectorized from the free-space source's pooled reachable mask + the v17
# visit-count grid (the dead FrontierDetector's per-step Python flood-fill is
# deliberately NOT used — it is a throughput cliff at 3k fps). Optionally
# weighted by local unexplored-region density (frontier_map_weighted).
FRONTIER_MAP_WIDTH = VISITATION_MAP_WIDTH  # 84
FRONTIER_MAP_HEIGHT = VISITATION_MAP_HEIGHT  # 46
FRONTIER_MAP_CHANNELS = 1

# Optional lattice-d* navigation hint (policy obs key ``lattice_nav``), enabled
# via ReachabilityConfig.expose_lattice_features.  Layout (4 dims):
#   [dx_hop, dy_hop, cost_to_go_norm, phase] — see value_function/lattice_distance.
LATTICE_NAV_DIM = 4
# Audit II P2-11: the waypoint_nav hint carries EXECUTABLE annotations the
# planner computed but previously discarded — the 4 legacy dims plus leg
# length, landing tolerance, airborne-leg flag, and the landing surface normal:
# [ux, uy, ctg_norm, phase, leg_len/192, tol/24, airborne_leg, nx, ny].
WAYPOINT_NAV_DIM = 9

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

# Optional ego-centric high-res crop obs (``ego_crop``), enabled via
# ReachabilityConfig.enable_ego_crop. A square window of raw tile IDs sliced from
# the full-resolution (42, 23) tile_grid, re-centered on the ninja's tile every
# step (the ninja is ALWAYS at the crop center). Precision-platforming fix: the
# global tile_grid is downsampled 3x by the spatial CNN and the cost_to_go_map is
# 3x3-pooled, so neither localizes a single-tile (24px) landing platform relative
# to the ninja. A ninja-centered crop keeps per-tile geometry at native resolution
# and is translation-invariant (landing target is always read in ego coordinates).
#
# Default 21 => a ±10-tile (±240px) window, ~2-3 single-tile platform gaps and a
# full jump arc's landing zone in each axis. Odd so the ninja sits on the exact
# center cell. Same (W, H)=[x, y] axis order as tile_grid.
EGO_CROP_SIZE_DEFAULT = 21
# Fill value for cells outside the playfield (crop overhangs a level edge). The
# level border is SOLID, so pad with the full-solid tile ID (1) rather than empty
# (0) — the policy must not read out-of-bounds as traversable free space.
EGO_CROP_PAD_TILE = 1
