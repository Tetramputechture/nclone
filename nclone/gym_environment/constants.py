"""Constants for the gym environment."""

import numpy as np

# FALLBACK ONLY -- this is NOT a cap on anything.
#
# It was documented as "Cap on the dynamic truncation limit", and it is not.
# ``truncation_calculator`` used to fold it in as
# ``min(HARD_TRUNCATION_CEILING, max(MAX_TIME_IN_FRAMES, limit))``, which is
# algebraically identical to clipping at HARD_TRUNCATION_CEILING alone --
# verified over every integer input in [0, 40000], zero differences. Whenever the
# formula exceeded 4500 the max() returned the formula; whenever it was below,
# the value was already under the cap. It therefore never bounded a single
# episode, while its comment implied episodes were capped at 4500 when the real
# ceiling was 24000, 5.3x higher. That dead arithmetic has been removed.
#
# The ONE thing this value still does: it is ``TruncationChecker``'s initial
# ``current_truncation_limit``, used on the rare path where a level's surface
# area cannot be computed and no dynamic limit is ever set.
MAX_TIME_IN_FRAMES = 4500

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

# v24 fine ego MINE crop (``ego_crop_mines_fine``): the 6px signed-distance mine
# field (hazard_map: ch0 deadly, ch1 safe; sdf=clip((d-14)/24,-1,1)) sliced in a
# small ninja-centered window at 6px/cell — sub-tile mine geometry exactly where
# deaths happen (around the ninja). The policy's only other fine (6px) mine input,
# the global hazard_map, is average-pooled to a ~7-tile-per-bin embedding before it
# reaches the policy, so the lethal 14px disc is unresolvable; every other mine view
# is >=24px. This crop restores sub-tile mine placement locally. Same window/frame
# as ego_crop_route_fine so the two co-register. 2 channels (deadly, safe).
EGO_CROP_MINES_FINE_CHANNELS = 2
EGO_CROP_MINES_FINE_SIZE = 21  # 21 * 6px = 126px window (~+-2.6 tiles)

# v26 W6a fine ego FALL-LETHALITY crop (``ego_crop_fall_fine``): the static
# survivable-impact field (see gym_environment/fall_lethality.py) sliced in a
# ninja-centered 6px window. Impact deaths are ~40% of v25's losses and are
# committed at TAKEOFF — by the time the ninja is falling, no action recovers —
# so a local view of "which cells around me are above a fatal drop" is the one
# impact signal that arrives while the decision is still open.
#
# Deliberately WIDER than the mine crops: the flat-floor lethal drop is ~6.8
# tiles, so a +-2.6-tile window cannot contain the hazard it is describing. 31
# cells * 6px = 186px (~+-3.9 tiles) covers the near half of a lethal fall, and
# the co-registered global fall_lethality_map carries the rest.
EGO_CROP_FALL_FINE_CHANNELS = 2
EGO_CROP_FALL_FINE_SIZE = 31  # 31 * 6px = 186px window (~+-3.9 tiles)

# v25 directional mine bearing + CLOSING SPEED (``mine_proximity``). The v24 fine
# mine crop above fixed *where* the mines are, but it is a still frame: it carries
# no velocity, so the policy cannot tell "drifting past a mine" from "about to hit
# it". After v24 mine-death fell to ~2% on the easy procgen buckets yet stayed at
# ~0.51 on OFFICIAL layouts, and switch_reached sat at 0.52 — i.e. half of official
# episodes die before the switch, mostly on mines. This key adds the missing
# motion-relative terms for the K nearest deadly mines, ninja-centered:
#   [0] dx_hat, [1] dy_hat  — unit bearing to the mine
#   [2] proximity           — 1.0 touching the 14px kill disc, 0.0 at/past range
#   [3] closing             — signed velocity along the bearing / speed norm,
#                             clipped [-1, 1]; > 0 = closing on the mine
#   [4] ttc                 — 1.0 when contact is imminent, decaying to 0.0 when
#                             far off or receding (the term the still frame lacks)
# Exposed as its OWN obs key rather than appended to game_state: game_state is a
# truncation-sensitive vector with several Box variants, and a dedicated key gets
# its own encoder branch. Empty slots (fewer than K deadly mines) read all-zero,
# which is also the "no threat" reading, so levels with no mines are consistent.
MINE_PROXIMITY_K = 4
MINE_PROXIMITY_FEATURES = 5
MINE_PROXIMITY_DIM = MINE_PROXIMITY_K * MINE_PROXIMITY_FEATURES

# --- Threat proximity: the shooter TELEGRAPH (plan §3.3) --------------------
#
# WHY THIS EXISTS. The threat FIELD carries where a hazard is lethal *now*, and
# for patrol drones that is enough because the field also forecasts them. It is
# not enough for the shooters, and the audit made the gap concrete: on the maps
# that isolate them, TURRET and DRONE_CHAINGUN read 0.0% field coverage in every
# channel over 150 frames of active play.
#
# That is not a scatter bug. Their BODIES are deliberately harmless
# (`body_is_lethal` is false for shooters; a turret is a wall fixture), so the
# only lethal thing they own is a beam or a shot — and a turret's beam is
# HITSCAN. It exists on the frame it kills. A field can only ever show it one
# frame before it is already too late, which makes turret death unlearnable from
# the field no matter how many horizons it carries.
#
# What IS learnable is the wind-up. Every shooter telegraphs:
#   turret   SEARCHING -> TRACKING (charge to 90) -> FIRING (15f) -> RECOVERING
#   rocket   IDLE -> CHARGING (15f) -> FLYING
#   shooters IDLE -> AIMING (charge to 53) -> FIRING -> COOLDOWN (90f)
# `imminence` is that countdown, normalised to [0, 1]. It converts "a turret
# exists over there" into "a turret fires in N frames", which is the difference
# between an unlearnable frame and a learnable one.
#
# Grouped by SUPER-CLASS, not per entity, so the width is fixed at 4x5 whatever
# the corpus contains — the per-entity alternative would be 21 bespoke slots for
# a flat coverage curve. Own obs key rather than appended to game_state, for the
# same reasons recorded above for mine_proximity.
#
#   class 0  contact-patrol  zap(14), mini(26)          — bodies kill on contact
#   class 1  homing          chaser(15), death ball(25), eye bat(27), evil ninja(22)
#   class 2  beam            turret(19), dual laser(23), laser drone(13)
#   class 3  projectile      rocket(18)
#
# per class, ninja-centered and bounded to [-1, 1]:
#   [0] dx_hat, [1] dy_hat  — unit bearing to the selected member
#   [2] proximity           — 1.0 at contact, 0.0 at/past range
#   [3] closing             — RELATIVE closing rate (v_ninja - v_threat)·bearing
#   [4] imminence           — telegraph countdown; 0.0 when nothing is winding up
#   [5] vx, [6] vy          — the THREAT's own velocity / speed norm
#   [7] spin_up             — speed it will gain next frame / its own max speed
#
# Dims 5–7 are the motion block, and they are why this is 8 wide rather than the
# 5 it shipped as. Without them the first five describe a still frame plus a
# countdown, and two opposite situations collapse onto one vector: a rocket
# projectile 200px away accelerating straight at the ninja, and one drifting past
# on a tangent, read IDENTICALLY — because `closing` used the ninja's velocity
# alone and the threat's own motion appeared nowhere. `spin_up` is the
# acceleration term and is non-zero only for the rocket projectile, the one
# hazard in the game that accelerates: it decays to exactly 0.0 at terminal speed
# (~24 frames after launch), so "still winding up" and "committed at full speed"
# are distinguishable.
#
# An absent class reads all-zero, which is also the "no threat" reading — so a
# level with no turrets is indistinguishable from one whose turret is idle and
# far away, which is the correct semantics for a threat signal.
#
# COMPUTED IN RUST (`nclone-native/src/threat_proximity.rs`). It has to be: the
# native `_build_entity_dic` whitelists {1,2,3,4,6,7,11,21} and not one threat
# type is in that set, so the Python path returned an all-zero vector on the
# training backend while passing every unit test. `threat_proximity_reference.py`
# is the pinned Python twin used by the Python backend and the dev tool.
THREAT_PROX_CLASSES = 4
THREAT_PROX_FEATURES = 8
THREAT_PROXIMITY_DIM = THREAT_PROX_CLASSES * THREAT_PROX_FEATURES
#: Index of each feature within a class block, so consumers never hardcode 0..7.
THREAT_PROX_IDX_DX = 0
THREAT_PROX_IDX_DY = 1
THREAT_PROX_IDX_PROXIMITY = 2
THREAT_PROX_IDX_CLOSING = 3
THREAT_PROX_IDX_IMMINENCE = 4
THREAT_PROX_IDX_VX = 5
THREAT_PROX_IDX_VY = 6
THREAT_PROX_IDX_SPIN_UP = 7
#: Reaction window. Wider than the mine window: a turret's shot crosses the room,
#: so "in range" is about line-of-fire, not contact.
THREAT_PROXIMITY_RANGE_PX = 240.0
#: Same normaliser as the mine block so the two keys share units.
THREAT_PROXIMITY_SPEED_NORM_PX = 4.0

#: Super-class membership. Entities appear in exactly one class; where an entity
#: is both a body and a beam (the laser drone) it is classed by what KILLS.
THREAT_CLASS_CONTACT = 0
THREAT_CLASS_HOMING = 1
THREAT_CLASS_BEAM = 2
THREAT_CLASS_PROJECTILE = 3
THREAT_CLASS_OF_TYPE = {
    14: THREAT_CLASS_CONTACT,
    26: THREAT_CLASS_CONTACT,
    15: THREAT_CLASS_HOMING,
    25: THREAT_CLASS_HOMING,
    27: THREAT_CLASS_HOMING,
    22: THREAT_CLASS_HOMING,
    19: THREAT_CLASS_BEAM,
    23: THREAT_CLASS_BEAM,
    13: THREAT_CLASS_BEAM,
    12: THREAT_CLASS_BEAM,
    18: THREAT_CLASS_PROJECTILE,
}
# NINJA_RADIUS + toggled-mine radius: contact at or inside this distance kills.
# Must agree with mine_threading_distance._KILL_RADIUS and the hazard SDF's
# `clip((d - 14) / 24, -1, 1)`.
MINE_PROXIMITY_KILL_RADIUS_PX = 14.0
# Beyond this the mine is not an immediate threat and `proximity` reads 0.
# ~4 tiles: far enough to react, near enough to stay a local signal.
MINE_PROXIMITY_RANGE_PX = 96.0
# Speed normalizer for the closing term. Vertical fall speed exceeds
# MAX_HOR_SPEED (3.33), so use a norm that keeps a terminal-velocity dive inside
# the clip rather than saturating every airborne frame.
MINE_PROXIMITY_SPEED_NORM_PX = 6.0
# Frames-to-contact scale for `ttc`: ttc = 1 / (1 + t / TAU), so t = 0 -> 1.0 and
# t = TAU (~0.5 s) -> 0.5. Matches the FiGAR macro horizon the policy commits over.
MINE_PROXIMITY_TTC_TAU_FRAMES = 30.0

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

# Always-on gold-fraction-remaining scalar appended right after the base
# game_state (docs/GOLD_AND_SCORE_OPTIMIZATION.md part b). It is appended
# UNCONDITIONALLY — independent of enable_gold_rewards — so the emitted
# game_state is ALWAYS one wider than GAME_STATE_CHANNELS. Every optional block
# (blocked_goal, wall_tau, precision_state, time_phase) is appended after it.
GAME_STATE_GOLD_TAIL_DIM = 1
# The width the env actually emits with no optional blocks enabled. Assert
# against THIS, not GAME_STATE_CHANNELS: the latter is the base the appends are
# measured from, and comparing an emitted vector to it is off by the gold tail.
GAME_STATE_EMITTED_CHANNELS = GAME_STATE_CHANNELS + GAME_STATE_GOLD_TAIL_DIM

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

# Optional drone hazard map (``drone_hazard_map``), gated by
# EnvConfig.enable_drone_hazard. Same 6px grid, sign convention and (W, H) axis
# order as `hazard_map`, so it co-registers cell-for-cell and its encoder reads
# the same units. Kept as a SEPARATE key rather than a third `hazard_map`
# channel so the strict Layer 1 contract and every existing checkpoint are
# unaffected while the drone work is in progress.
#   ch0        = deadly drone SDF NOW
#   ch1..chN   = the same field at each horizon in DRONE_HAZARD_HORIZONS
# Zap and mini drones are merged within a channel because they are behaviourally
# identical to the policy — deterministic ninja-independent patrols, lethal on
# contact. They differ only in radius and speed, and the radius difference is
# already carried by the per-class kill radius below.
#
# The forecast channels are what make timing learnable. A single present-frame
# field says only "a drone is there now"; it cannot distinguish a corridor whose
# drone is walking away from one whose drone is walking in, and those demand
# opposite actions. Because patrol drones are ninja-independent, their future is
# not a prediction problem at all — it is a lookup, exact to the last bit (see
# `Simulator::forecast_drone_positions`). Mines have no analogue of this.
#
# Horizons are in SIM FRAMES, not env steps: at the training frame_skip of 4 they
# are 3, 6 and 12 actions ahead. A zap drone covers a 24px cell in 21 frames, so
# 12/24/48 span roughly half a cell, one cell and two cells — while the ninja at
# full run speed covers ~6.7 cells in 48 frames, which is the range over which a
# route commitment is still revocable.
DRONE_HAZARD_HORIZONS = (12, 24, 48)
DRONE_HAZARD_MAP_CHANNELS = 1 + len(DRONE_HAZARD_HORIZONS)

# Contact-death boundary per drone class, in world px: NINJA_RADIUS + drone
# radius. This is the true kill distance, mirroring how the mine SDF's 14 px
# comes from NINJA_RADIUS (10) + mine radius (4).
DRONE_KILL_RADIUS_PX = 17.5  # zap drone: 10 + 7.5
MINI_DRONE_KILL_RADIUS_PX = 14.0  # mini drone: 10 + 4

#: Entity type ids of the deterministic patrol drones the drone hazard map
#: covers. Deliberately EXCLUDES the chaser (15), whose movement depends on the
#: ninja: a chaser is not predictable from the frame counter, so it cannot be
#: forward-rolled and would need different treatment entirely.
DRONE_HAZARD_ENTITY_TYPES = (14, 26)

# Optional cost-to-go map obs (``cost_to_go_map``) is projected at a COARSER grid
# than the 6px hazard map. Its encoder (_CostToGoEncoder) pools to a 3x3 readout
# regardless of input resolution, so a 2x (12px) downsample is ~lossless for the
# routing signal it carries while cutting the obs payload ~4x — and it is the
# single largest observation (dominates the per-step deepcopy/slice/buffer/pipe
# cost). MUST divide HAZARD_MAP_WIDTH/HEIGHT evenly.
COST_TO_GO_MAP_DOWNSAMPLE = 2
COST_TO_GO_MAP_WIDTH = HAZARD_MAP_WIDTH // COST_TO_GO_MAP_DOWNSAMPLE  # 84
COST_TO_GO_MAP_HEIGHT = HAZARD_MAP_HEIGHT // COST_TO_GO_MAP_DOWNSAMPLE  # 46

# The POLICY-FACING `hazard_map` / `fall_lethality_map` observations are emitted
# at the same coarser grid, for the same reason. Both fields are still COMPUTED
# at 6px — the ego crops (`ego_crop_mines_fine`, `ego_crop_fall_fine`) and the
# hazard/impact PBRS terms slice them at full fidelity, and those consumers are
# untouched. Only the global obs keys are pooled.
#
# The two maps are 2 channels of 168x92 float32 each = 121 KB per env per map,
# together ~64% of the whole observation payload, while their encoders pool to a
# small fixed readout (AdaptiveAvgPool2d) that cannot see the difference. Every
# per-step operation that scales with observation BYTES — GPU transfer, rollout
# buffer write, VecNormalize copy, SIL slice — pays for that resolution. Pooling
# 2x takes each map 121 KB -> 30 KB.
#
# Pooling is conservative (worst sub-cell wins), see obs_map_pooling. MUST divide
# HAZARD_MAP_WIDTH/HEIGHT evenly.
HAZARD_MAP_OBS_DOWNSAMPLE = 2
HAZARD_MAP_OBS_WIDTH = HAZARD_MAP_WIDTH // HAZARD_MAP_OBS_DOWNSAMPLE  # 84
HAZARD_MAP_OBS_HEIGHT = HAZARD_MAP_HEIGHT // HAZARD_MAP_OBS_DOWNSAMPLE  # 46

# Phase 1 threat/surface keys. Both are emitted by the native simulator directly
# on the 12px observation grid (HAZARD_MAP_OBS_*), so they co-register with the
# pooled hazard map cell-for-cell and need no pooling pass of their own.
#
# `threat_ttl` -- [C_sdf, C_ttl, R_sdf, R_ttl]. The SDF channels give spatial
# margin, the TTL channels give time-to-lethal (0 = now, 1 = not within the
# window). Polarity matches the hazard map throughout: 1.0 = safe, so `min`
# stays the conservative reduction on every channel.
THREAT_TTL_CHANNELS = 4
THREAT_TTL_CH_CERTAIN_SDF = 0
THREAT_TTL_CH_CERTAIN_TTL = 1
THREAT_TTL_CH_REACTIVE_SDF = 2
THREAT_TTL_CH_REACTIVE_TTL = 3

# `solid_map` -- entity geometry that gates, carries, propels or blocks. None of
# it is in `tile_grid`, which carries static tiles only.
#
#   0  static         thwump, shove thwump, CLOSED locked/trap/regular door
#   1  bounce         bounce block (springy, so it changes the jump that leaves it)
#   2  oneway         one-way platform occupancy
#   3  oneway_nx      \ the blocking normal -- passable from one side, solid from
#   4  oneway_ny      / the other, which is the entire point of a one-way
#   5  door_open      the same doors as ch0, while OPEN
#   6  door_switch    switch positions of doors that are still CLOSED
#   7  launchpad      launch pad occupancy
#   8  launchpad_bx   \ boost direction; two pads in a cell pointing opposite
#   9  launchpad_by   / ways are opposite affordances
#  10  boost_pad      boost pad occupancy (scales existing velocity, no vector)
#
# Channels 0-4 are SURFACES (what blocks or carries), 5-10 are TRAVERSAL (what
# gates or propels). Doors are unified by STATE rather than type -- what the
# policy needs is "is this barrier passable right now", and locked / trap /
# regular differ only in what flips that bit. Type identity that DOES matter (a
# trap door closes behind you) belongs in the planner, not the pixel grid.
#
# ENTITY_INTEGRATION_PLAN.md section 3.4 proposes twelve traversal channels on
# `entity_overlays` instead. Six of those already exist here as channels 0-4 --
# this field shipped after that plan was written -- so only the six genuinely
# new ones were built, and they were built HERE so they co-register cell for cell
# with the solid geometry they qualify (an open door is the same barrier as a
# closed one at a different moment) and so the boost vectors sit alongside the
# one-way normals they are structurally identical to.
SOLID_MAP_CHANNELS = 11
SOLID_MAP_CH_STATIC = 0
SOLID_MAP_CH_BOUNCE = 1
SOLID_MAP_CH_ONEWAY = 2
SOLID_MAP_CH_ONEWAY_NX = 3
SOLID_MAP_CH_ONEWAY_NY = 4
SOLID_MAP_CH_DOOR_OPEN = 5
SOLID_MAP_CH_DOOR_SWITCH = 6
SOLID_MAP_CH_LAUNCHPAD = 7
SOLID_MAP_CH_LAUNCHPAD_BX = 8
SOLID_MAP_CH_LAUNCHPAD_BY = 9
SOLID_MAP_CH_BOOST_PAD = 10

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
