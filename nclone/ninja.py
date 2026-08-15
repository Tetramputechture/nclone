import struct
import math
import struct as _struct
import random
import os

# Local imports
from .physics import (
    gather_entities_from_neighbourhood,
    gather_segments_from_region,
    sweep_circle_vs_tiles,
    get_single_closest_point,
)
from .constants import (
    # Animation constants
    ANIM_DATA,
    DANCE_RANDOM,
    DANCE_ID_DEFAULT,
    DANCE_DIC,
    # Physics constants
    GRAVITY_FALL,
    GRAVITY_JUMP,
    GROUND_ACCEL,
    AIR_ACCEL,
    DRAG_REGULAR,
    DRAG_SLOW,
    FRICTION_GROUND,
    FRICTION_GROUND_SLOW,
    FRICTION_WALL,
    MAX_HOR_SPEED,
    MAX_JUMP_DURATION,
    MAX_SURVIVABLE_IMPACT,
    MIN_SURVIVABLE_CRUSHING,
    NINJA_RADIUS,
    # Jump constants
    JUMP_UPSLOPE_Y,
    TILT_QUARTER_TURN,
    TILT_DECAY,
    JUMP_UPSLOPE_Y_RAW,
    JUMP_SCALE_A,
    JUMP_SCALE_B,
    JUMP_SCALE_C,
    JUMP_IMPULSE_X_SCALE,
    JUMP_IMPULSE_Y_SCALE,
    JUMP_SLOPE_DOWNHILL_X_MULTIPLIER,
    JUMP_SLOPE_DOWNHILL_Y_MULTIPLIER,
    JUMP_SLOPE_UPHILL_DEFAULT_Y,
    JUMP_SLOPE_UPHILL_PERP_X_MULTIPLIER,
    JUMP_SLOPE_UPHILL_PERP_Y_MULTIPLIER,
    JUMP_WALL_SLIDE_X_MULTIPLIER,
    JUMP_WALL_SLIDE_Y,
    JUMP_WALL_REGULAR_X_RAW_MULT,
    JUMP_WALL_REGULAR_Y_RAW,
    JUMP_WALL_SLIDE_X_RAW_MULT,
    JUMP_WALL_SLIDE_Y_RAW,
    JUMP_WALL_REGULAR_X_MULTIPLIER,
    JUMP_WALL_REGULAR_Y,
    JUMP_LAUNCH_PAD_BOOST_SCALAR,
    JUMP_LAUNCH_PAD_BOOST_FACTOR,
    # Ragdoll constants
    RAGDOLL_GRAVITY,
    RAGDOLL_DRAG,
    RAGDOLL_NUM_BONES,
    RAGDOLL_BONE_SOURCE,
    RAGDOLL_SEGS,
    RAGDOLL_SEG_LENGTHS,
    RAGDOLL_ANGLES,
    RAGDOLL_TWO_PI,
    RAGDOLL_BONE_RADIUS,
    RAGDOLL_SWEEP_RADIUS,
    RAGDOLL_STICK_RADIUS,
    RAGDOLL_GATHER_RADIUS,
    RAGDOLL_SWEEP_MIN_DIST_SQ,
    RAGDOLL_MAX_DEPEN_ITERS,
    RAGDOLL_SLIDE_SCALE,
    RAGDOLL_DEGENERATE_SQ,
    RAGDOLL_EPSILON,
    RAGDOLL_BOUNCE,
    RAGDOLL_FRICTION,
    RAGDOLL_IMPULSE_FALLOFF,
    RAGDOLL_IMPULSE_NEAR,
    RAGDOLL_IMPULSE_FAR,
    RAGDOLL_HEAD_KICK,
    RAGDOLL_HEAD_KICK_SEED,
    RAGDOLL_TOI_OVERLAP,
    RAGDOLL_TOI_MISS,
    RAGDOLL_EXPLODED_NUM_BONES,
    RAGDOLL_EXPLODED_BONE_SOURCE,
    RAGDOLL_EXPLODED_KICKS,
    RAGDOLL_EXPLODED_SEGS,
    RAGDOLL_EXPLODED_SEG_LENGTHS,
    RAGDOLL_EXPLODED_ANGLES,
    RAGDOLL_EXPLODING_DEATH_CAUSES,
    RAGDOLL_EXPLOSION_FORCE,
    RAGDOLL_EXPLOSION_RADIUS,
    RAGDOLL_EXPLOSION_FLOOR,
    RAGDOLL_MINE_TOUCH_SLACK,
    RAGDOLL_MAX_CHAIN_PER_FRAME,
)

import logging

logger = logging.getLogger(__name__)

#: `_DAT_10571CA8`, the depenetration loop's exit test. There are **two**
#: epsilons in the game's collision code and round 3 conflated them: this one,
#: and `1e-5` (`DAT_105866A0`), which belongs to the separate wall-normal scan in
#: `post_collision` that only feeds wall-jumping. Round 3 told us to change this
#: to 1e-5; round 4 retracted that (`docs/ENTITY_RE_ROUND4_FINDINGS.md` §11).
DEPENETRATION_EPSILON = 1e-7

#: Private RNG for the cosmetic victory dance, so picking one never perturbs the
#: `random` module's shared stream. See `update_graphics`.
_DANCE_RNG = random.Random()

# `corner_tie_break` lived here and is GONE. It was a lookup table of six
# hardcoded `xpos` float literals that forced `dx` to +-2^-47, standing in for
# the denormal `floor_normalized_x` the port could not produce on flat ground.
#
# `FUN_10511AB0` (`docs/decomp/R5_01_collide_vs_tiles_10511AB0.c`) takes
# `o = pos - cp` with no special case, so it never should have existed; it
# survived three rounds because deleting it alone always measured worse.
#
# What it was really compensating for was found in the end: door segments wound
# high-to-low made `get_closest_point` inexact by exactly 2^-47, which is why its
# first table entry keyed on `xpos == 50.51197510492316` -- frame 11 of
# `MET-S-E-11-04`, the very position that bug produces. Winding doors low-to-high
# fixed the cause (see `entity_door_base.py`), and the stand-in is now dead:
#
#   * `10131255`, the one demo the table was fitted for, is bit-exact WITHOUT it;
#   * it had already been off by default in both engines;
#   * switching it back on costs 29 demos of the 8000-demo gate (6504 -> 6475),
#     19 of them to mines.


cached_ninja_animation = []


def get_ninja_animation():
    """Get the ninja animation data from the file"""
    global cached_ninja_animation
    if len(cached_ninja_animation) > 0:
        return cached_ninja_animation
    else:
        print("Loading ninja animation data")
        with open(ANIM_DATA, mode="rb") as f:
            frames = struct.unpack("<L", f.read(4))[0]
            cached_ninja_animation = [
                [list(struct.unpack("<2d", f.read(16))) for _ in range(13)]
                for _ in range(frames)
            ]
        return cached_ninja_animation


def has_active_jump_buffer(n) -> bool:
    """True when any input buffer would let a jump fire this frame.

    Takes any ninja-LIKE object (the real Ninja or native _NinjaProxy) so the
    python and native backends cannot diverge on this predicate. Windows differ:
    jump/floor/wall are 5 frames, launch_pad is 4 (``-1 < b < 4``).
    """
    return (
        -1 < n.jump_buffer < 5
        or -1 < n.floor_buffer < 5
        or -1 < n.wall_buffer < 5
        or -1 < n.launch_pad_buffer < 4
    )


def jump_is_useless(n) -> bool:
    """SINGLE SOURCE OF TRUTH for "pressing jump right now does nothing".

    Consumed by the action mask, Ninja.is_jump_useless, and the game_state
    observation, so none of them can contradict the others. Jumps fire on the
    RISING edge: holding jump from the previous frame creates no new edge, so it
    is useless unless a buffer is already armed. Grounded or already jumping, a
    jump always has an immediate effect.
    """
    if not (n.airborn and n.state != 3):
        return False
    if n.jump_input != 0:
        return not has_active_jump_buffer(n)
    return False



def _f32(x: float) -> float:
    """Round to float32 and widen, the way the binary stores `+0xC38`."""
    return _struct.unpack("<f", _struct.pack("<f", x))[0]

class Ninja:
    """This class is responsible for updating and storing the positions and velocities of each ninja.
    self.xposlog and self.yposlog contain all the coordinates used to generate the traces of the replays.
    """

    def __init__(self, sim, ninja_anim_mode: bool):
        """Initiate ninja position at spawn point, and initiate other values to their initial state"""
        self.ninja_anim_mode = ninja_anim_mode and os.path.isfile(ANIM_DATA)
        self.ninja_animation = []
        if self.ninja_anim_mode:
            self.ninja_animation = get_ninja_animation()

        self.sim = sim
        self.xpos = sim.map_data[1231] * 6
        self.ypos = sim.map_data[1232] * 6

        # Store initial spawn position for fast reset
        self._initial_xpos = self.xpos
        self._initial_ypos = self.ypos
        self.xspeed = 0
        self.yspeed = 0
        self.applied_gravity = GRAVITY_FALL
        self.applied_drag = DRAG_REGULAR
        self.applied_friction = FRICTION_GROUND
        self.state = 0  # 0:Immobile, 1:Running, 2:Ground sliding, 3:Jumping, 4:Falling, 5:Wall sliding
        self.airborn = False
        self.airborn_old = False
        self.walled = False
        self.wall_normal = 0
        self.jump_input_old = 0
        self.jump_duration = 0
        self.jump_buffer = -1
        self.floor_buffer = -1
        self.wall_buffer = -1
        self.launch_pad_buffer = -1
        self.is_jump_useless = False
        self.last_jump_was_buffered = False
        self.floor_normalized_x = 0
        self.floor_normalized_y = -1
        self.ceiling_normalized_x = 0
        self.ceiling_normalized_y = 1
        self.anim_state = 0
        self.facing = 1
        self.tilt = 0
        self.anim_rate = 0
        self.anim_frame = 11
        self.frame_residual = 0
        #: `+0xC34` (clipB) and `+0xC38` (blend). The run cycle cross-fades out of
        #: the PREVIOUS clip; see `update_graphics`. Round 9 §2.2.
        self.anim_frame_prev = 11
        self.anim_blend = 0.0
        if ninja_anim_mode:
            self.bones = [[0, 0] for _ in range(13)]
            self.update_graphics()
            self.ragdoll = Ragdoll(sim)
        else:
            self.bones = None
            self.ragdoll = None

        if sim.sim_config.debug:
            self.poslog = []  # Used for debug
            self.speedlog = []
            self.xposlog = []  # Used to produce trace
            self.yposlog = []
        else:
            self.poslog = None
            self.speedlog = None
            self.xposlog = None
            self.yposlog = None

        # inputs
        self.hor_input = 0
        self.jump_input = 0

        # positions
        self.xpos_old = 0
        self.ypos_old = 0
        self.death_xpos = 0
        self.death_ypos = 0
        self.terminal_impact = False
        self.death_cause = (
            None  # Track cause of death: "mine", "impact", "hazard", etc.
        )
        self.xpos_at_action = self.xpos  # Position when action was applied
        self.ypos_at_action = self.ypos

        # speeds
        self.xspeed_old = 0
        self.yspeed_old = 0
        self.death_xspeed = 0
        self.death_yspeed = 0

        # collisions
        self.floor_count = 0
        self.wall_count = 0
        self.ceiling_count = 0
        self.floor_normal_x = 0
        self.floor_normal_y = 0
        self.ceiling_normal_x = 0
        self.ceiling_normal_y = 0
        self.is_crushable = False
        self.x_crush = 0
        self.y_crush = 0
        self.crush_len = 0
        # `Ninja::addContact`'s two contact flags. They live on the ninja rather
        # than on the loop because the shove thwump only clears `isSolid` and
        # nothing re-zeroes them between entities or frames -- see
        # `collide_vs_objects`.
        self.contact_is_solid = False
        self.contact_is_moving = False

        # gold collected
        self.gold_collected = 0

        # doors opened
        self.doors_opened = 0

        # death cause
        self.death_cause = None

        # Track consecutive frames airborne for death context learning
        self.frames_airborne = 0

        # Track state transitions for temporal features
        self.state_change_frame = 0  # Frames since last state change
        self.previous_state = 0  # Track previous state

        # Track consecutive contact frames for stability features
        self.consecutive_floor_frames = 0
        self.consecutive_wall_frames = 0

        self.log()

    def integrate(self):
        """Update position and speed by applying drag and gravity before collision phase."""
        self.xspeed *= self.applied_drag
        self.yspeed *= self.applied_drag
        self.yspeed += self.applied_gravity
        self.xpos_old = self.xpos
        self.ypos_old = self.ypos
        self.xpos += self.xspeed
        self.ypos += self.yspeed

    def pre_collision(self):
        """Reset some values used for collision phase."""
        self.xspeed_old = self.xspeed
        self.yspeed_old = self.yspeed
        self.floor_count = 0
        self.wall_count = 0
        self.ceiling_count = 0
        self.floor_normal_x, self.floor_normal_y = 0, 0
        self.ceiling_normal_x, self.ceiling_normal_y = 0, 0
        self.is_crushable = False
        self.x_crush, self.y_crush = 0, 0
        self.crush_len = 0
        self._cached_entities = gather_entities_from_neighbourhood(
            self.sim, self.xpos, self.ypos
        )

    def collide_vs_objects(self):
        """Gather all entities in neighbourhood and apply physical collisions if possible."""
        entities = self._cached_entities
        for entity in entities:
            if entity.is_physical_collidable:
                depen = entity.physical_collision()
                if depen:
                    depen_x, depen_y = depen[0]
                    depen_len = depen[1][0]
                    pop_x, pop_y = depen_x * depen_len, depen_y * depen_len
                    self.xpos += pop_x
                    self.ypos += pop_y
                    # `Ninja::addContact` (`docs/decomp/R5_01_add_contact_10514E30.c`)
                    # accumulates the crush vector only when the contact is
                    # `isSolid || isMoving`, and each entity states those two
                    # flags in its physical collision -- as a store into the same
                    # two adjacent bytes of the contact struct:
                    #
                    #   one-way (11)       word  1      -> solid 1, moving 0
                    #   bounce block (17)  word  0      -> solid 0, moving 0
                    #   thwump (20)        word  0x100  -> solid 0, moving 1
                    #   shove thwump (28)  BYTE  0      -> solid 0, moving UNCHANGED
                    #
                    # The shove thwump's store is one byte narrower than its three
                    # siblings' (round 7 §2, verified against the instruction width
                    # and against the bounce block writing the identical value as a
                    # word). So it clears `isSolid` and inherits whatever `isMoving`
                    # the last physical contact left -- nothing re-zeroes the struct
                    # between entities or between frames. Observable only when a
                    # thwump has touched the ninja and a shove thwump touches it
                    # afterwards, which is rare and deterministic.
                    if entity.type == 11:
                        self.contact_is_solid, self.contact_is_moving = True, False
                    elif entity.type == 17:
                        self.contact_is_solid, self.contact_is_moving = False, False
                    elif entity.type == 20:
                        self.contact_is_solid, self.contact_is_moving = False, True
                    elif entity.type == 28:
                        self.contact_is_solid = False  # `isMoving` survives
                    if self.contact_is_solid or self.contact_is_moving:
                        self.x_crush += pop_x
                        self.y_crush += pop_y
                        self.crush_len += depen_len
                    if (
                        entity.type == 20
                    ):  # Ninja can only get crushed if collision with thwump.
                        self.is_crushable = True
                    # Depenetration for bounce blocks, thwumps and shwumps.
                    if entity.type in (17, 20, 28):
                        self.xspeed += pop_x
                        self.yspeed += pop_y
                    # A solid contact PROJECTS the velocity, and only when the
                    # ninja is moving into the surface (`dot(v, n) < 0`); a
                    # non-solid one adds `n * depth` as an impulse, which is the
                    # branch above. The gate is a no-op today because the one-way
                    # only depenetrates when `normal_proj <= 0` anyway, but it is
                    # what the binary tests and it costs nothing to state.
                    #
                    # `v -= dot * n`, the same form round 8 §2b.1 established for
                    # the tile path -- the one-way was left on the old
                    # cross-product-with-the-tangent spelling. The two are
                    # algebraically identical for a unit normal and numerically
                    # are not: landing flat on a one-way, `nx` is exactly zero, so
                    # `yspeed -= dot * ny` cancels to `+0.0` whichever way the
                    # ninja is running, while `(vx*ny - vy*nx) * -nx` carries the
                    # sign of `vx` into the zero. Ground truth (frame 8 of
                    # `10138412`, `10138730` and `10139437`, running in opposite
                    # directions) has `+0.0` in every case.
                    if entity.type == 11:  # Depenetration for one ways
                        dot_product = self.xspeed * depen_x + self.yspeed * depen_y
                        if dot_product < 0:
                            self.xspeed = self.xspeed - dot_product * depen_x
                            self.yspeed = self.yspeed - dot_product * depen_y
                    # Adjust ceiling variables if ninja collides with ceiling (or wall!)
                    if depen_y >= -0.0001:
                        self.ceiling_count += 1
                        self.ceiling_normal_x += depen_x
                        self.ceiling_normal_y += depen_y
                    else:  # Adjust floor variables if ninja collides with floor
                        self.floor_count += 1
                        self.floor_normal_x += depen_x
                        self.floor_normal_y += depen_y

    def collide_vs_tiles(self):
        """Gather all tile segments in neighbourhood and handle collisions with those."""
        # Interpolation routine mainly to prevent from going through walls.
        dx = self.xpos - self.xpos_old
        dy = self.ypos - self.ypos_old
        time = sweep_circle_vs_tiles(
            self.sim, self.xpos_old, self.ypos_old, dx, dy, NINJA_RADIUS * 0.5
        )
        self.xpos = self.xpos_old + time * dx
        self.ypos = self.ypos_old + time * dy

        # Cache frequently accessed instance variables for performance
        xpos = self.xpos
        ypos = self.ypos
        xspeed = self.xspeed
        yspeed = self.yspeed
        x_crush = self.x_crush
        y_crush = self.y_crush
        crush_len = self.crush_len
        ceiling_count = self.ceiling_count
        ceiling_normal_x = self.ceiling_normal_x
        ceiling_normal_y = self.ceiling_normal_y
        floor_count = self.floor_count
        floor_normal_x = self.floor_normal_x
        floor_normal_y = self.floor_normal_y

        # Find the closest point from the ninja, apply depenetration and update speed. Loop 32 times.
        for _ in range(32):
            # RE-GATHERED every iteration, from the CURRENT position. The game's
            # `get_single_closest_point` (`FUN_1050B910`,
            # `docs/decomp/caller_1050B910.c`) calls the region gather itself, so
            # each depenetration step sees the cells the ninja is in *now*. This
            # port used to gather once, before the loop, from the pre-depenetration
            # position -- which loses any segment the ninja is pushed into and
            # keeps ones it has been pushed out of.
            result, closest_point = get_single_closest_point(
                self.sim, xpos, ypos, NINJA_RADIUS
            )
            if result == 0:
                break
            a, b = closest_point
            # `o = pos - cp`, verbatim -- `FUN_10511AB0` has no special case here
            # and neither do we now. See the note where `corner_tie_break` used
            # to live, above.
            dx = xpos - a
            dy = ypos - b

            dist = math.sqrt(dx**2 + dy**2)
            depen_len = NINJA_RADIUS - dist * result
            if dist == 0 or depen_len < DEPENETRATION_EPSILON:
                # Write back before returning, matching original's direct self.* updates
                self.xpos = xpos
                self.ypos = ypos
                self.xspeed = xspeed
                self.yspeed = yspeed
                self.x_crush = x_crush
                self.y_crush = y_crush
                self.crush_len = crush_len
                self.ceiling_count = ceiling_count
                self.ceiling_normal_x = ceiling_normal_x
                self.ceiling_normal_y = ceiling_normal_y
                self.floor_count = floor_count
                self.floor_normal_x = floor_normal_x
                self.floor_normal_y = floor_normal_y
                return
            # NORMALIZE FIRST, then work in the normalized frame -- this is not a
            # refactor, it changes results. `Ninja::collideVsTiles`
            # (`docs/decomp/NINJA_collideVsTiles_10511AB0.c:82-101`) computes
            #     nx = dx / dist ; ny = dy / dist
            # once and uses those for the depenetration, the velocity projection
            # AND the ceiling/floor test.
            norm_x = dx / dist
            norm_y = dy / dist

            depen_x = norm_x * depen_len
            depen_y = norm_y * depen_len
            xpos += depen_x
            ypos += depen_y
            x_crush += depen_x
            y_crush += depen_y
            crush_len += depen_len

            # Project velocity onto surface only if moving towards surface.
            #
            # The binary does `v -= dot * n` against the NORMALIZED normal
            # (`:93-96`), not the cross-product-over-dist**2 form this port used
            # to. They are algebraically identical and numerically are not: on
            # flat ground `nx` is exactly `0.0`, so `xspeed -= dot * nx` is an
            # exact no-op, whereas `(xspeed*dy - yspeed*dx) / dist**2 * dy`
            # rounds three times and lands 1 ULP off. That single last-bit error
            # on frame 10 of `10131738` amplified to 167 px by frame 587 and
            # killed the demo -- see `docs/ENTITY_RE_ROUND8_FINDINGS.md` §2.
            dot_product = xspeed * norm_x + yspeed * norm_y
            if dot_product < 0:
                xspeed = xspeed - dot_product * norm_x
                yspeed = yspeed - dot_product * norm_y

            # Adjust ceiling/floor variables. The binary compares the NORMALIZED
            # `ny` against -0.0001 (`:98`), not the raw `dy`.
            if norm_y >= -0.0001:
                ceiling_count += 1
                ceiling_normal_x += norm_x
                ceiling_normal_y += norm_y
            else:
                floor_count += 1
                floor_normal_x += norm_x
                floor_normal_y += norm_y

        # Write back cached variables to instance
        self.xpos = xpos
        self.ypos = ypos
        self.xspeed = xspeed
        self.yspeed = yspeed
        self.x_crush = x_crush
        self.y_crush = y_crush
        self.crush_len = crush_len
        self.ceiling_count = ceiling_count
        self.ceiling_normal_x = ceiling_normal_x
        self.ceiling_normal_y = ceiling_normal_y
        self.floor_count = floor_count
        self.floor_normal_x = floor_normal_x
        self.floor_normal_y = floor_normal_y

    def post_collision(self, skip_entities=False):
        """Perform logical collisions with entities, check for airborn state,
        check for walled state, calculate floor normals, check for impact or crush death.

        Args:
            skip_entities: If True, skip entity collision checks (optimization for terminal velocity)
        """
        # Perform LOGICAL collisions between the ninja and nearby entities.
        # Also check if the ninja can interact with the walls of entities when applicable.
        #
        # Contacts accumulate into a LIST, in the game's order -- entities first,
        # then tile segments -- because `Ninja::postCollision`
        # (`docs/decomp/ninja_collide_10513C60.c:354-359`) takes `list[0]` and
        # discards the rest. Summing and taking the sign lets two opposing
        # contacts cancel, which reports `walled == False` inside a one-tile
        # shaft where the game reports True.
        wall_normals = []
        if not skip_entities:
            entities = gather_entities_from_neighbourhood(
                self.sim, self.xpos, self.ypos
            )
        else:
            entities = []
        for entity in entities:
            if entity.is_logical_collidable:
                collision_result = entity.logical_collision()
                if collision_result:
                    if (
                        entity.type == 10
                    ):  # If collision with launch pad, update speed and position.
                        xboost = collision_result[0] * 2 / 3
                        yboost = collision_result[1] * 2 / 3
                        self.xpos += xboost
                        self.ypos += yboost
                        self.xspeed = xboost
                        self.yspeed = yboost
                        self.floor_count = 0
                        self.floor_buffer = -1
                        boost_scalar = math.sqrt(xboost**2 + yboost**2)
                        self.xlp_boost_normalized = xboost / boost_scalar
                        self.ylp_boost_normalized = yboost / boost_scalar
                        self.launch_pad_buffer = 0
                        if self.state == 3:
                            self.applied_gravity = GRAVITY_FALL
                        self.state = 4
                    else:  # If touched wall of bounce block, oneway, thwump or shwump, retrieve wall normal.
                        # The `if collision_result:` above already drops the 0.0
                        # a top/bottom bounce-block contact returns, which is the
                        # game's `out[1] == 0.0` condition -- only a horizontal
                        # normal is a wall. Keep that: a zero at the head of the
                        # list would otherwise be taken as the wall normal.
                        wall_normals.append(collision_result)

        # Check if the ninja can interact with walls from nearby tile segments.
        if not skip_entities:
            rad = NINJA_RADIUS + 0.1
            segments = gather_segments_from_region(
                self.sim,
                self.xpos - rad,
                self.ypos - rad,
                self.xpos + rad,
                self.ypos + rad,
            )
        else:
            segments = []
        for segment in segments:
            result = segment.get_closest_point(self.xpos, self.ypos)
            a, b = result[1], result[2]
            dx = self.xpos - a
            dy = self.ypos - b
            dist = math.sqrt(dx**2 + dy**2)
            if abs(dy) < 0.00001 and 0 < dist <= rad:
                wall_normals.append(dx / dist)

        # Check if airborn or walled.
        self.airborn_old = self.airborn
        self.airborn = True
        self.walled = False
        if wall_normals:
            self.walled = True
            # `list[0]` verbatim, NOT normalised to +-1: the game stores the
            # contact's own x component, which for a tile is `dx/dist` and is
            # exactly +-1 only when `dy` is exactly zero. `+0x110` is read as a
            # full double on the physics path.
            self.wall_normal = wall_normals[0]

        # Calculate the combined floor normalized normal vector if the ninja has touched any floor.
        if self.floor_count > 0:
            self.airborn = False
            floor_scalar = math.sqrt(self.floor_normal_x**2 + self.floor_normal_y**2)
            if floor_scalar == 0:
                self.floor_normalized_x = 0
                self.floor_normalized_y = -1
            else:
                self.floor_normalized_x = self.floor_normal_x / floor_scalar
                self.floor_normalized_y = self.floor_normal_y / floor_scalar
            if (
                self.state != 8 and self.airborn_old
            ):  # Check if died from terminal impact on the floor
                impact_vel = -(
                    self.floor_normalized_x * self.xspeed_old
                    + self.floor_normalized_y * self.yspeed_old
                )
                if impact_vel > MAX_SURVIVABLE_IMPACT - 4 / 3 * abs(
                    self.floor_normalized_y
                ):
                    self.xspeed = self.xspeed_old
                    self.yspeed = self.yspeed_old
                    self.kill(
                        1,
                        self.xpos,
                        self.ypos,
                        self.xspeed * 0.5,
                        self.yspeed * 0.5,
                        cause="terminal_impact",
                    )
                    self.terminal_impact = True

        # Track frames_airborne: increment when airborne, reset when on floor
        if self.airborn:
            self.frames_airborne += 1
        else:
            self.frames_airborne = 0

        # Track consecutive contact frames for stability features
        if not self.airborn:
            self.consecutive_floor_frames += 1
        else:
            self.consecutive_floor_frames = 0

        if self.walled:
            self.consecutive_wall_frames += 1
        else:
            self.consecutive_wall_frames = 0

        # Calculate the combined ceiling normalized normal vector if the ninja has touched any ceiling.
        if self.ceiling_count > 0:
            ceiling_scalar = math.sqrt(
                self.ceiling_normal_x**2 + self.ceiling_normal_y**2
            )
            if ceiling_scalar == 0:
                self.ceiling_normalized_x = 0
                self.ceiling_normalized_y = 1
            else:
                self.ceiling_normalized_x = self.ceiling_normal_x / ceiling_scalar
                self.ceiling_normalized_y = self.ceiling_normal_y / ceiling_scalar
            if self.state != 8:  # Check if died from ceiling impact
                impact_vel = -(
                    self.ceiling_normalized_x * self.xspeed_old
                    + self.ceiling_normalized_y * self.yspeed_old
                )
                if impact_vel > MAX_SURVIVABLE_IMPACT - 4 / 3 * abs(
                    self.ceiling_normalized_y
                ):
                    self.xspeed = self.xspeed_old
                    self.yspeed = self.yspeed_old
                    self.kill(
                        1,
                        self.xpos,
                        self.ypos,
                        self.xspeed * 0.5,
                        self.yspeed * 0.5,
                        cause="terminal_impact",
                    )
                    self.terminal_impact = True

        # Check if ninja died from crushing.
        if self.is_crushable and self.crush_len > 0:
            if (
                math.sqrt(self.x_crush**2 + self.y_crush**2) / self.crush_len
                < MIN_SURVIVABLE_CRUSHING
            ):
                self.kill(2, self.xpos, self.ypos, 0, 0, cause="crush")

    def floor_jump(self, was_buffered: bool = False):
        """Perform floor jump depending on slope angle and direction.

        Args:
            was_buffered: True if jump was executed via floor buffer
        """
        self.jump_buffer = -1
        self.floor_buffer = -1
        self.launch_pad_buffer = -1
        if was_buffered:
            self.last_jump_was_buffered = True
        self.state = 3
        self.applied_gravity = GRAVITY_JUMP
        # `Ninja::think` has exactly TWO grounded rows
        # (`docs/decomp/NINJA_update_10515300.c:346-357`, and round-4 §8.3):
        #
        #     if (floor.x * hor_input < 0)  jump(0.0, -0.7, 0)
        #     else                          jump(floor.x, floor.y, 0)
        #
        # and `Ninja::jump` turns `(jx, jy)` into `(jx*2/3, jy*2)`. There is no
        # flat-ground special case -- a flat floor is `(0, -1)`, which falls into
        # the second row and yields `(0, -2)`, the value this used to hardcode as
        # `JUMP_FLOOR_Y`.
        #
        # This replaces a four-way tree that discriminated on **xspeed** rather
        # than the horizontal input, and that had a "perp jump" branch which also
        # zeroed `xspeed`. The binary has no such branch and never zeroes xspeed
        # here -- `jump()` zeroes only a component that OPPOSES the impulse.
        fx = self.floor_normalized_x
        fy = self.floor_normalized_y
        if fx * self.hor_input < 0:
            self._apply_jump(0.0, JUMP_UPSLOPE_Y_RAW)
        else:
            self._apply_jump(fx, fy)

    def _apply_jump(self, raw_x: float, raw_y: float):
        """`Ninja::jump` -- `docs/decomp/R5_07_jump_apply_105110C0.c`.

        Takes the RAW row arguments (a floor normal, or a constant pair) and
        applies the scaled impulse. Every jump in the game funnels through here.

        The two multiplications are written exactly as the binary evaluates
        them, left to right, because the association is observable: see
        `JUMP_SCALE_A/B/C` in `physics_constants`.
        """
        jx = (JUMP_SCALE_A * raw_x) * JUMP_SCALE_C
        jy = ((JUMP_SCALE_A + JUMP_SCALE_B) * raw_y) * JUMP_SCALE_C
        # Zero any velocity component OPPOSING the impulse, then add (`:16-21`,
        # `:27-30`) -- an add, not an assignment.
        if self.xspeed * jx < 0:
            self.xspeed = 0
        if self.yspeed * jy < 0:
            self.yspeed = 0
        self.xspeed += jx
        self.yspeed += jy
        self.xpos += jx
        self.ypos += jy
        self.jump_duration = 0

    def wall_jump(self, was_buffered: bool = False):
        """Perform wall jump depending on wall normal and if sliding or not.

        Args:
            was_buffered: True if jump was executed via wall buffer
        """
        # `NINJA_update_10515300.c:387-396`. The binary picks the row on
        # `state != 5 || wall_normal * hor_input >= 0` -> regular, which is the
        # negation of the slide test below.
        if self.hor_input * self.wall_normal < 0 and self.state == 5:  # Slide wall jump
            raw_x = self.wall_normal * JUMP_WALL_SLIDE_X_RAW_MULT
            raw_y = JUMP_WALL_SLIDE_Y_RAW
        else:  # Regular wall jump
            raw_x = self.wall_normal * JUMP_WALL_REGULAR_X_RAW_MULT
            raw_y = JUMP_WALL_REGULAR_Y_RAW
        if was_buffered:
            self.last_jump_was_buffered = True
        self.state = 3
        self.applied_gravity = GRAVITY_JUMP
        # `_apply_jump` does the zeroing itself, on the sign of the impulse. That
        # is the same test this used to spell out by hand: the x impulse always
        # carries the sign of `wall_normal`, and the y impulse is always negative.
        self._apply_jump(raw_x, raw_y)
        self.jump_buffer = -1
        self.wall_buffer = -1
        self.launch_pad_buffer = -1

    def lp_jump(self):
        """Perform launch pad jump."""
        self.floor_buffer = -1
        self.wall_buffer = -1
        self.jump_buffer = -1
        self.launch_pad_buffer = -1
        boost_scalar = 2 * abs(self.xlp_boost_normalized) + 2
        if boost_scalar == 2:
            boost_scalar = (
                JUMP_LAUNCH_PAD_BOOST_SCALAR  # This was really needed. Thanks Metanet
            )
        self.xspeed += (
            self.xlp_boost_normalized * boost_scalar * JUMP_LAUNCH_PAD_BOOST_FACTOR
        )
        self.yspeed += (
            self.ylp_boost_normalized * boost_scalar * JUMP_LAUNCH_PAD_BOOST_FACTOR
        )

    def has_active_jump_buffer(self) -> bool:
        """True when any input buffer would let a jump fire this frame.

        Windows differ per buffer: jump/floor/wall are 5 frames, launch_pad is 4
        (``-1 < b < 4``; see think()).
        """
        return has_active_jump_buffer(self)

    def jump_is_useless_now(self) -> bool:
        """SINGLE SOURCE OF TRUTH for "pressing jump right now does nothing".

        Consumed by BOTH the action mask and the game_state observation so the
        two can never contradict each other.

        Jumps fire on the RISING edge, so holding jump from the previous frame
        means a fresh press creates no new edge -- useless unless a buffer is
        already armed. Grounded or already-jumping, a jump always has an
        immediate effect.
        """
        return jump_is_useless(self)

    def get_valid_action_mask(self) -> list:
        """Compute mask of valid actions based on current state.

        Masks several types of useless actions:
        1. Useless jumps (airborne without buffers)
        2. Directional inputs into walls (when walled)
        3. Actions leading directly into toggled mines (certain death)

        Timing: Called BEFORE action is applied. self.jump_input reflects PREVIOUS frame's action.

        Buffer activation: In think(), pressing jump while airborne activates jump_buffer=0.
        Buffer counts up each frame (0->1->2->3->4->5->-1). Active when: -1 < buffer < 5.

        Returns:
            List of 6 bools [NOOP, LEFT, RIGHT, JUMP, JUMP+LEFT, JUMP+RIGHT] (True = valid).
        """
        # Store ninja state snapshot for logging if masked action is selected
        if not hasattr(self, "_mask_debug_state"):
            self._mask_debug_state = {}

        self._mask_debug_state = {
            "position": (self.xpos, self.ypos),
            "velocity": (self.xspeed, self.yspeed),
            "airborn": self.airborn,
            "walled": self.walled,
            "wall_normal": self.wall_normal if self.walled else None,
            "state": self.state,
            "jump_input": self.jump_input,
            "jump_buffer": self.jump_buffer,
            "floor_buffer": self.floor_buffer,
            "wall_buffer": self.wall_buffer,
            "launch_pad_buffer": self.launch_pad_buffer,
        }

        mask = [True] * 6

        # Check for active buffers (allow jumps without holding jump button)
        has_active_buffer = (
            -1
            < self.jump_buffer
            < 5  # Jump buffer: activated by pressing jump while airborne
            or -1 < self.floor_buffer < 5  # Floor buffer: activated when touching floor
            or -1 < self.wall_buffer < 5  # Wall buffer: activated when touching wall
            or -1
            < self.launch_pad_buffer
            < 4  # Launch pad buffer: activated by launch pad
        )

        # Determine if jump is useless. Delegated to jump_is_useless_now() so the
        # ACTION MASK and the OBSERVATION cannot drift apart: the policy must not
        # perceive "a jump would work here" while the mask forbids it, or vice
        # versa. (Three disagreeing copies of this predicate previously existed:
        # this mask, Ninja.is_jump_useless, and the game_state feature.)
        jump_useless = self.jump_is_useless_now()

        # Mask jump actions if useless (actions 0-2 always valid)
        if jump_useless:
            mask[3] = False  # JUMP
            mask[4] = (
                False  # JUMP+LEFT (mask entire action to prevent bad learning patterns)
            )
            mask[5] = False  # JUMP+RIGHT (same reasoning)

        # Mask directional inputs into walls based on state
        # wall_normal points FROM wall TO ninja:
        #   wall_normal > 0: wall is on left side
        #   wall_normal < 0: wall is on right side
        # Only mask pure directional actions (LEFT/RIGHT), not jump combos
        # because JUMP+direction triggers wall jumps which are useful
        if self.walled:
            if not self.airborn:
                # Case 1: Grounded and walled - mask direct movement toward wall
                # Moving into wall while grounded provides no benefit
                if self.wall_normal > 0:
                    mask[1] = False  # Wall on left, mask LEFT
                elif self.wall_normal < 0:
                    mask[2] = False  # Wall on right, mask RIGHT
            else:
                # Airborne and walled
                if self.state == 5:
                    # Case 3: Already wall sliding - mask direct input toward wall
                    # Input is already being used to maintain wall slide
                    if self.wall_normal > 0:
                        mask[1] = False  # Mask LEFT
                    elif self.wall_normal < 0:
                        mask[2] = False  # Mask RIGHT
                else:
                    # Case 2: Airborne but not wall sliding yet
                    # Wall slide triggers when: yspeed > 0 and input toward wall
                    # If input would trigger wall slide, DON'T mask (it's useful)
                    # If input wouldn't trigger wall slide, mask it (it's useless)
                    would_trigger_wall_slide = self.yspeed >= 0

                    if self.wall_normal > 0:
                        # Wall on left
                        if not would_trigger_wall_slide:
                            mask[1] = False  # Mask LEFT if NOT triggering wall slide
                        # Keep JUMP+LEFT (action 4) valid - triggers wall jump
                    elif self.wall_normal < 0:
                        # Wall on right
                        if not would_trigger_wall_slide:
                            mask[2] = False  # Mask RIGHT if NOT triggering wall slide
                        # Keep JUMP+RIGHT (action 5) valid - triggers wall jump

        # === ENHANCED PATH-BASED ACTION MASKING ===
        # Uses full direction vector (dx, dy) from physics-optimal path to mask
        # provably suboptimal actions. More comprehensive than legacy "straight path"
        # masking - handles diagonal paths, respects ninja state (grounded vs airborne).
        #
        # This masking is applied AFTER physics-based masking (wall/jump masking)
        # and uses multi-hop path direction (3-4 hops) from physics-optimal path.
        # Zero performance cost (O(1) lookup from pre-computed path).
        #
        # Key principles:
        # 1. Mask horizontal actions moving away from goal (LEFT when goal is right, etc.)
        # 2. Mask jumps when goal is DOWN and ninja is grounded (jumping goes up)
        # 3. Mask NOOP when grounded with clear direction (no timing objectives)
        # 4. Keep NOOP when airborne (needed for precise air control)
        #
        # ANNEALED MASKING (2026-01-05): Only apply hard masking in early training
        # Mode transitions: hard (early) -> soft (mid) -> none (late)
        # The masking mode is controlled by ActionMaskingConfig based on success rate
        # and curriculum stage progression.
        #
        # UPDATED 2026-01-06: Made conditional for hyperparameter optimization
        # Can be disabled via _enable_straightness_masking attribute (defaults to True)
        # enable_path_masking = getattr(self.sim, "_enable_straightness_masking", True)
        # masking_mode = getattr(self.sim, "_action_masking_mode", "hard")
        # if enable_path_masking and masking_mode == "hard":
        #     path_direction = getattr(self.sim, "_path_straightness_direction", None)
        #     if path_direction == "left":
        #         # Goal is directly to the left on straight horizontal path
        #         # Mask: RIGHT (2), JUMP (3), JUMP+LEFT (4), JUMP+RIGHT (5)
        #         # Keep: NOOP (0), LEFT (1)
        #         #
        #         # Rationale:
        #         # - Only LEFT is needed on a straight horizontal path to the left
        #         # - All other actions are suboptimal (wrong direction or unnecessary jumps)
        #         mask[2] = False  # Mask RIGHT (wrong direction)
        #         mask[3] = False  # Mask JUMP (unnecessary on straight path)
        #         mask[4] = False  # Mask JUMP+LEFT (jumping is unnecessary)
        #         mask[5] = False  # Mask JUMP+RIGHT (wrong direction + unnecessary jump)
        #     elif path_direction == "right":
        #         # Goal is directly to the right on straight horizontal path
        #         # Mask: LEFT (1), JUMP (3), JUMP+LEFT (4), JUMP+RIGHT (5)
        #         # Keep: NOOP (0), RIGHT (2)
        #         mask[1] = False  # Mask LEFT (wrong direction)
        #         mask[3] = False  # Mask JUMP (unnecessary on straight path)
        #         mask[4] = False  # Mask JUMP+LEFT (wrong direction + unnecessary jump)
        #         mask[5] = False  # Mask JUMP+RIGHT (jumping is unnecessary)
        #     elif path_direction == "down":
        #         # Goal is downward (vertically or diagonally down)
        #         # Mask: JUMP (3), JUMP+LEFT (4), JUMP+RIGHT (5)
        #         # Keep: NOOP (0), LEFT (1), RIGHT (2)
        #         #
        #         # Rationale:
        #         # - Path goes DOWN, but JUMP moves UP (opposite of optimal direction)
        #         # - Horizontal movements are still allowed (may need to navigate while falling)
        #         # - This handles downward slopes, drops, and descending paths
        #         mask[3] = False  # Mask JUMP (moves up when we need to go down)
        #         mask[4] = False  # Mask JUMP+LEFT (moves up when we need to go down)
        #         mask[5] = False  # Mask JUMP+RIGHT (moves up when we need to go down)

        # Ensure at least one action is always valid
        # If all actions are masked (inevitable death scenario), unmask NOOP
        # This prevents NaN in the policy distribution
        if not any(mask):
            # Ninja is in inevitable death scenario (surrounded by mines, etc.)
            # Allow NOOP as a last resort to prevent policy NaN
            mask[0] = True

        # Validate mask before returning
        # # These assertions catch bugs early before they cause NaN in the policy
        # assert isinstance(mask, list), f"Mask must be a list, got {type(mask)}"
        # assert len(mask) == 6, f"Mask must have 6 elements, got {len(mask)}"
        # assert all(isinstance(m, bool) for m in mask), (
        #     f"All mask elements must be boolean, got types: {[type(m) for m in mask]}"
        # )
        # assert any(mask), (
        #     "At least one action must be valid. This should be enforced by "
        #     "the fallback above. If you see this, there's a logic bug."
        # )

        # Store computed mask for debugging
        self._mask_debug_state["computed_mask"] = mask.copy()
        self._mask_debug_state["masked_actions"] = [i for i in range(6) if not mask[i]]
        self._mask_debug_state["valid_actions"] = [i for i in range(6) if mask[i]]

        # Add fingerprint for tracking (only in debug mode, check via sim config)
        if self.sim.sim_config.debug:
            self._mask_debug_state["mask_fingerprint"] = (
                round(self.xpos, 2),
                round(self.ypos, 2),
                self.airborn,
                self.walled,
                self.jump_input,
                self.state,
            )
            self._mask_debug_state["mask_timestamp"] = self.sim.frame

        return mask

    def record_action_start_position(self):
        """Record position at start of action for delta detection.

        Called by environment AFTER action input but BEFORE physics tick.
        """
        self.xpos_at_action = self.xpos
        self.ypos_at_action = self.ypos

    def think(self):
        """This function handles all the ninja's actions depending of the inputs and its environment."""
        # Reset buffer tracking flag at start of each frame
        self.last_jump_was_buffered = False

        # Track state transitions for temporal features
        if self.state != self.previous_state:
            self.state_change_frame = 0
            self.previous_state = self.state
        else:
            self.state_change_frame += 1

        # Logic to determine if you're starting a new jump.
        if not self.jump_input:
            new_jump_check = False
        else:
            new_jump_check = self.jump_input_old == 0
        self.jump_input_old = self.jump_input

        # Determine if within buffer ranges. If so, increment buffers.
        if -1 < self.launch_pad_buffer < 3:
            self.launch_pad_buffer += 1
        else:
            self.launch_pad_buffer = -1
        in_lp_buffer = -1 < self.launch_pad_buffer < 4
        if -1 < self.jump_buffer < 5:
            self.jump_buffer += 1
        else:
            self.jump_buffer = -1
        in_jump_buffer = -1 < self.jump_buffer < 5
        if -1 < self.wall_buffer < 5:
            self.wall_buffer += 1
        else:
            self.wall_buffer = -1
        in_wall_buffer = -1 < self.wall_buffer < 5
        if -1 < self.floor_buffer < 5:
            self.floor_buffer += 1
        else:
            self.floor_buffer = -1
        in_floor_buffer = -1 < self.floor_buffer < 5

        # Initiate jump buffer if beginning a new jump and airborn.
        jump_buffer_just_activated = False
        if new_jump_check and self.airborn:
            self.jump_buffer = 0
            jump_buffer_just_activated = True

        # Initiate wall buffer if touched a wall this frame.
        if self.walled:
            self.wall_buffer = 0

        # Initiate floor buffer if touched a floor this frame.
        if not self.airborn:
            self.floor_buffer = 0

        # Track if current jump input is useless
        # A jump is useless when: airborn, not in jump state (3), no active buffers
        # Note: If we just activated the jump buffer (new_jump_check and airborn),
        # then the jump is NOT useless, even if buffers weren't active before
        if self.jump_input:
            # Recompute has_active_buffer after buffer activation
            # If jump buffer was just activated, it's now active (value is 0)
            if jump_buffer_just_activated:
                has_active_buffer = True
            else:
                # Check if jump_buffer is now active (could have been activated above)
                current_in_jump_buffer = -1 < self.jump_buffer < 5
                has_active_buffer = (
                    current_in_jump_buffer
                    or in_floor_buffer
                    or in_wall_buffer
                    or in_lp_buffer
                )
            # Same predicate the action mask and the observation use.
            self.is_jump_useless = self.jump_is_useless_now()
        else:
            self.is_jump_useless = False

        # This part deals with the special states: dead, awaiting death, celebrating, disabled.
        if self.state in (6, 9):
            return
        if self.state == 7:
            self.think_awaiting_death()
            return
        if self.state == 8:
            self.applied_drag = DRAG_REGULAR if self.airborn else DRAG_SLOW
            return

        # This block deals with the case where the ninja is touching a floor.
        if not self.airborn:
            xspeed_new = self.xspeed + GROUND_ACCEL * self.hor_input
            if abs(xspeed_new) < MAX_HOR_SPEED:
                self.xspeed = xspeed_new
            if self.state > 2:
                if self.xspeed * self.hor_input <= 0:
                    if self.state == 3:
                        self.applied_gravity = GRAVITY_FALL
                    self.state = 2
                else:
                    if self.state == 3:
                        self.applied_gravity = GRAVITY_FALL
                    self.state = 1
            if not in_jump_buffer and not new_jump_check:  # if not jumping
                if self.state == 2:
                    projection = abs(
                        self.yspeed * self.floor_normalized_x
                        - self.xspeed * self.floor_normalized_y
                    )
                    if self.hor_input * projection * self.xspeed > 0:
                        self.state = 1
                        return
                    if projection < 0.1 and self.floor_normalized_x == 0:
                        self.state = 0
                        return
                    if self.yspeed < 0 and self.floor_normalized_x != 0:
                        # Up slope friction formula, very dumb but that's how it is
                        speed_scalar = math.sqrt(self.xspeed**2 + self.yspeed**2)
                        fric_force = abs(
                            self.xspeed
                            * (1 - FRICTION_GROUND)
                            * self.floor_normalized_y
                        )
                        fric_force2 = (
                            speed_scalar - fric_force * self.floor_normalized_y**2
                        )
                        self.xspeed = self.xspeed / speed_scalar * fric_force2
                        self.yspeed = self.yspeed / speed_scalar * fric_force2
                        return
                    self.xspeed *= FRICTION_GROUND
                    return
                if self.state == 1:
                    projection = abs(
                        self.yspeed * self.floor_normalized_x
                        - self.xspeed * self.floor_normalized_y
                    )
                    if self.hor_input * projection * self.xspeed > 0:
                        # if holding inputs in downhill direction or flat ground
                        if self.hor_input * self.floor_normalized_x >= 0:
                            return
                        if abs(xspeed_new) < MAX_HOR_SPEED:
                            boost = GROUND_ACCEL / 2 * self.hor_input
                            xboost = (
                                boost
                                * self.floor_normalized_y
                                * self.floor_normalized_y
                            )
                            yboost = (
                                boost
                                * self.floor_normalized_y
                                * -self.floor_normalized_x
                            )
                            self.xspeed += xboost
                            self.yspeed += yboost
                        return
                    self.state = 2
                else:  # if you were in state 0 I guess
                    if self.hor_input:
                        self.state = 1
                        return
                    projection = abs(
                        self.yspeed * self.floor_normalized_x
                        - self.xspeed * self.floor_normalized_y
                    )
                    if projection < 0.1:
                        self.xspeed *= FRICTION_GROUND_SLOW
                        return
                    self.state = 2
                return
            self.floor_jump()  # if you're jumping
            return

        # This block deals with the case where the ninja didn't touch a floor
        else:
            xspeed_new = self.xspeed + AIR_ACCEL * self.hor_input
            if abs(xspeed_new) < MAX_HOR_SPEED:
                self.xspeed = xspeed_new
            if self.state < 3:
                self.state = 4
                return
            if self.state == 3:
                self.jump_duration += 1
                if not self.jump_input or self.jump_duration > MAX_JUMP_DURATION:
                    self.applied_gravity = GRAVITY_FALL
                    self.state = 4
                    return
            if in_jump_buffer or new_jump_check:  # if able to perfrom jump
                if self.walled or in_wall_buffer:
                    # Only count as buffered if wall_buffer was used (not just walled)
                    self.wall_jump(was_buffered=in_wall_buffer)
                    return
                if in_floor_buffer:
                    self.floor_jump(was_buffered=True)
                    return
                if in_lp_buffer and new_jump_check:
                    self.lp_jump()
                    return
            if not self.walled:
                if self.state == 5:
                    self.state = 4
            else:
                if self.state == 5:
                    if self.hor_input * self.wall_normal <= 0:
                        self.yspeed *= FRICTION_WALL
                    else:
                        self.state = 4
                else:
                    if self.yspeed > 0 and self.hor_input * self.wall_normal < 0:
                        if self.state == 3:
                            self.applied_gravity = GRAVITY_FALL
                        self.state = 5

    def think_awaiting_death(self):
        """Set state to dead and activate ragdoll."""
        self.state = 6
        if self.ninja_anim_mode:
            bones_speed = [
                [
                    self.bones[i][0] - self.bones_old[i][0],
                    self.bones[i][1] - self.bones_old[i][1],
                ]
                for i in range(13)
            ]
            self.ragdoll.activate(
                self.xpos,
                self.ypos,
                self.xspeed,
                self.yspeed,
                self.death_xpos,
                self.death_ypos,
                self.death_xspeed,
                self.death_yspeed,
                self.bones,
                bones_speed,
                # `+0xC18`, the ninja's facing. The angle constraints are
                # mirrored by it, so a corpse falls the way it was running.
                self.facing,
            )
            # `ninja_10515F90.c` line 62: the kill's category decides whether
            # the corpse bursts, and it does so immediately, before the first
            # ragdoll frame runs.
            if self.death_cause in RAGDOLL_EXPLODING_DEATH_CAUSES:
                self.ragdoll.explode()

    def update_graphics(self):
        """Update parameters necessary to draw the limbs of the ninja."""
        if not self.ninja_anim_mode:
            return

        anim_state_old = self.anim_state
        if self.state == 5:
            self.anim_state = 4
            self.tilt = 0
            self.facing = -self.wall_normal
            self.anim_rate = self.yspeed
        elif not self.airborn and self.state != 3:
            # `DAT_105866e0`, a STORED 64-bit constant added to the atan2 result
            # (`full/ninja_10516280.c:35`) -- not `M_PI_2`, not computed. It
            # differs from pi/2 by ~4.9e-12, and with the limb test live that is a
            # physics input. Round 9 §2.3.
            self.tilt = (
                math.atan2(self.floor_normalized_y, self.floor_normalized_x)
                + TILT_QUARTER_TURN
            )
            if self.state == 0:
                self.anim_state = 0
            if self.state == 1:
                self.anim_state = 1
                self.anim_rate = abs(
                    self.yspeed * self.floor_normalized_x
                    - self.xspeed * self.floor_normalized_y
                )
                if self.hor_input:
                    self.facing = self.hor_input
            if self.state == 2:
                self.anim_state = 2
                self.anim_rate = abs(
                    self.yspeed * self.floor_normalized_x
                    - self.xspeed * self.floor_normalized_y
                )
            if self.state == 8:
                self.anim_state = 6
        else:
            self.anim_state = 3
            self.anim_rate = self.yspeed
            if self.state == 3:
                self.tilt = 0
            else:
                # `a - a*0.1`, NOT `a *= 0.9`. `full/ninja_10516280.c:71-72` with
                # `DAT_10571d98 = 0.1`; there is no 0.9 constant anywhere in that
                # function. The two round differently (0.3 -> 0.27 vs
                # 0.26999999999999996). Round 9 §2.1.
                self.tilt = self.tilt - self.tilt * TILT_DECAY
        if self.state != 5:
            if abs(self.xspeed) > 0.01:
                self.facing = 1 if self.xspeed > 0 else -1

        if self.anim_state != anim_state_old:
            if self.anim_state == 0:
                # `full/ninja_10516280.c:88-97` sets all four animation fields to
                # `min(c28, 0)` -- so 0 for any non-negative frame, and the guard
                # only preserves a NEGATIVE counter. The port set 1, which the
                # tick below then advanced to 2, leaving the landing clip one
                # frame ahead of the game's for as long as the ninja stood still.
                # `10137755`: the bones are inert at ~2e-12 through frame 7 and
                # jump to 2.3e-02 on frame 8, the landing.
                self.anim_frame = min(self.anim_frame, 0)
                self.run_cycle = 0
            if self.anim_state == 1:
                # `full/ninja_10516280.c:98-121`. The incoming state picks where
                # in the 432-step run cycle the clip starts, and the port had the
                # table wrong: it sent **0 (immobile) and 4 (wall slide)** down the
                # 12/0 default, where the binary sends them to the same 39/162 as
                # 2. `LAB_10516582` is reached from `iVar2 == 0 || iVar2 == 4` and
                # jumped to from `iVar2 == 2`; only 3 and the genuine default are
                # separate. `10137755` frame 17 is a 0 -> 1 transition, and it is
                # where the bones diverge by 0.48 once the landing frame is right.
                if anim_state_old in (0, 2, 4):
                    self.anim_frame = 39
                    self.run_cycle = 162
                elif anim_state_old == 3:
                    self.anim_frame = 18
                    self.run_cycle = 36
                else:
                    self.anim_frame = 12
                    self.run_cycle = 0
                self.frame_residual = 0
            if self.anim_state == 2:
                self.anim_frame = 0
            if self.anim_state == 3:
                self.anim_frame = 84
            if self.anim_state == 4:
                self.anim_frame = 103
            if self.anim_state == 6:
                # The victory dance, in state 8 -- the level is already won, so
                # nothing observable depends on which one plays. Drawn from a
                # PRIVATE Random rather than the `random` module's shared stream:
                # the skeleton is now maintained on every map with a hitscan
                # hazard, and one extra global draw per won episode would shift
                # the RNG stream of anything training on top of the sim.
                # `nclone-native` cannot reproduce a random choice at all and
                # always uses `DANCE_DEFAULT`, so the two engines differ here and
                # only here, cosmetically, after the win frame.
                self.dance_id = (
                    _DANCE_RNG.choice(list(DANCE_DIC))
                    if DANCE_RANDOM
                    else DANCE_ID_DEFAULT
                )
                self.anim_frame = DANCE_DIC[self.dance_id][0]
            # `full/ninja_10516280.c:149` zeroes the blend on every state change,
            # so a clip entered from another state starts un-crossfaded.
            self.anim_blend = 0.0
            self.anim_frame_prev = self.anim_frame

        if self.anim_state == 0:
            if self.anim_frame < 11:
                self.anim_frame += 1
        if self.anim_state == 1:
            new_cycle = self.anim_rate / 0.15 + self.frame_residual
            self.frame_residual = new_cycle - math.floor(new_cycle)
            self.run_cycle = (self.run_cycle + math.floor(new_cycle)) % 432
            # `clipB` is the PREVIOUS `clipA`, latched only when `clipA` actually
            # changes, and the blend weights IT -- so the run cycle cross-fades OUT
            # of the outgoing clip into the incoming one, lagging by one clip.
            # `full/ninja_10516280.c:158-178`, round 9 §2.2. The port used to
            # target `(anim_frame - 12) % 72 + 12`, which is `anim_frame` itself for
            # every frame the running state can produce, making the lerp a no-op.
            prev_clip = self.anim_frame
            self.anim_frame = self.run_cycle // 6 + 12
            if self.anim_frame != prev_clip:
                self.anim_frame_prev = prev_clip
            # Stored as a FLOAT by the binary (`+0xC38`), then widened.
            self.anim_blend = _f32(1.0 - (self.run_cycle % 6) / 6.0)
        if self.anim_state == 3:
            if self.anim_rate >= 0:
                rate = math.sqrt(min(self.anim_rate * 0.6, 1))
            else:
                rate = max(self.anim_rate * 1.5, -1)
            self.anim_frame = 93 + math.floor(9 * rate)
        if self.anim_state == 6:
            if self.anim_frame < DANCE_DIC[self.dance_id][1]:
                self.anim_frame += 1

        self.bones_old = self.bones
        self.calc_ninja_position()

    def calc_ninja_position(self):
        """Calculate the positions of ninja's joints. The positions are fetched from the animation data,
        after applying mirroring, rotation or interpolation if necessary."""
        # Reuse lists instead of deep copying
        if not hasattr(self, "new_bones"):
            self.new_bones = [[0, 0] for _ in range(13)]

        anim_frame_bones = self.ninja_animation[self.anim_frame]
        for i in range(13):
            self.new_bones[i][0] = anim_frame_bones[i][0]
            self.new_bones[i][1] = anim_frame_bones[i][1]

        if self.anim_state == 1:
            # `FUN_105114B0:86,127` -- gated on `blend > 0.0f`, then
            # `a += (b - a) * blend` with `b` from clipB. Round 9 §2.2.
            interpolation = self.anim_blend
            if interpolation > 0:
                next_bones = self.ninja_animation[self.anim_frame_prev]
                for i in range(13):
                    self.new_bones[i][0] += interpolation * (
                        next_bones[i][0] - self.new_bones[i][0]
                    )
                    self.new_bones[i][1] += interpolation * (
                        next_bones[i][1] - self.new_bones[i][1]
                    )

        for i in range(13):
            self.new_bones[i][0] *= self.facing
            x, y = self.new_bones[i]
            tcos, tsin = math.cos(self.tilt), math.sin(self.tilt)
            self.new_bones[i][0] = x * tcos - y * tsin
            self.new_bones[i][1] = x * tsin + y * tcos

        # Swap references instead of copying
        self.bones_old = self.bones
        self.bones = self.new_bones
        self.new_bones = self.bones_old

    def get_temporal_features(self):
        """Extract 6 temporal features for RL observation space.

        Returns:
            List of 6 floats normalized to [-1, 1] range:
            - velocity_change_x: Frame-to-frame acceleration in x
            - velocity_change_y: Frame-to-frame acceleration in y
            - momentum_magnitude: Current velocity magnitude
            - frames_airborne_normalized: Consecutive frames airborne
            - jump_duration_normalized: Current jump duration
            - state_transition_timing: Frames since last state change
        """
        return [
            # velocity_change_x: Frame-to-frame acceleration (already normalized)
            max(-1.0, min(1.0, (self.xspeed - self.xspeed_old) / MAX_HOR_SPEED)),
            # velocity_change_y: Frame-to-frame acceleration (already normalized)
            max(-1.0, min(1.0, (self.yspeed - self.yspeed_old) / MAX_HOR_SPEED)),
            # momentum_magnitude: Current velocity magnitude for momentum tracking
            min(math.sqrt(self.xspeed**2 + self.yspeed**2) / (MAX_HOR_SPEED * 1.5), 1.0)
            * 2
            - 1,
            # frames_airborne_normalized: Consecutive frames airborne (for landing prediction)
            min(self.frames_airborne / 60.0, 1.0) * 2 - 1,
            # jump_duration_normalized: Current jump duration (for jump timing)
            min(self.jump_duration / MAX_JUMP_DURATION, 1.0) * 2 - 1,
            # state_transition_timing: Frames since last state change (for physics awareness)
            min(self.state_change_frame / 30.0, 1.0) * 2 - 1,
        ]

    def win(self):
        """Set ninja's state to celebrating."""
        if self.state < 6:
            if self.state == 3:
                self.applied_gravity = GRAVITY_FALL
            self.state = 8

    def kill(self, type, xpos, ypos, xspeed, yspeed, cause=None):
        """Set ninja's state to just killed.

        Args:
            type: Death type (unused, kept for compatibility)
            xpos, ypos: Death position
            xspeed, yspeed: Death velocity
            cause: String indicating death cause ("mine", "impact", "hazard", etc.)
        """
        if self.state < 6:
            self.death_xpos = xpos
            self.death_ypos = ypos
            self.death_xspeed = xspeed
            self.death_yspeed = yspeed
            self.death_cause = cause  # Store death cause
            if self.state == 3:
                self.applied_gravity = GRAVITY_FALL
            self.state = 7

    def is_valid_target(self):
        """Return whether the ninja is a valid target for various interactions."""
        return self.state not in (6, 8, 9)

    def log(self):
        """Log position and velocity vectors of the ninja for the current frame"""
        if self.sim.sim_config.debug:
            self.poslog.append(
                (self.sim.frame, round(self.xpos, 6), round(self.ypos, 6))
            )
            self.speedlog.append(
                (self.sim.frame, round(self.xspeed, 6), round(self.yspeed, 6))
            )
            self.xposlog.append(self.xpos)
            self.yposlog.append(self.ypos)

    def reset_state(self):
        """Reset ninja to initial spawn state for fast environment reset.

        This method resets the ninja to its spawn position and initial state
        without creating a new object. Used for optimized same-level resets.
        """
        # Reset position to spawn
        self.xpos = self._initial_xpos
        self.ypos = self._initial_ypos

        # Reset velocities
        self.xspeed = 0
        self.yspeed = 0
        self.xspeed_old = 0
        self.yspeed_old = 0

        # Reset physics parameters
        self.applied_gravity = GRAVITY_FALL
        self.applied_drag = DRAG_REGULAR
        self.applied_friction = FRICTION_GROUND

        # Reset state flags
        self.state = 0  # Immobile
        self.airborn = False
        self.airborn_old = False
        self.walled = False
        self.wall_normal = 0

        # Reset jump-related state
        self.jump_input_old = 0
        self.jump_duration = 0
        self.jump_buffer = -1
        self.floor_buffer = -1
        self.wall_buffer = -1
        self.launch_pad_buffer = -1
        self.is_jump_useless = False
        self.last_jump_was_buffered = False

        # Reset collision normals
        self.floor_normalized_x = 0
        self.floor_normalized_y = -1
        self.ceiling_normalized_x = 0
        self.ceiling_normalized_y = 1

        # Reset collision counters
        self.floor_count = 0
        self.wall_count = 0
        self.ceiling_count = 0
        self.floor_normal_x = 0
        self.floor_normal_y = 0
        self.ceiling_normal_x = 0
        self.ceiling_normal_y = 0
        self.is_crushable = False
        self.x_crush = 0
        self.y_crush = 0
        self.crush_len = 0

        # Reset animation state
        self.anim_state = 0
        self.facing = 1
        self.tilt = 0
        self.anim_rate = 0
        self.anim_frame = 11
        self.frame_residual = 0
        #: `+0xC34` (clipB) and `+0xC38` (blend). The run cycle cross-fades out of
        #: the PREVIOUS clip; see `update_graphics`. Round 9 §2.2.
        self.anim_frame_prev = 11
        self.anim_blend = 0.0

        # Reset inputs
        self.hor_input = 0
        self.jump_input = 0

        # Reset position tracking
        self.xpos_old = 0
        self.ypos_old = 0
        self.death_xpos = 0
        self.death_ypos = 0
        self.terminal_impact = False
        self.death_cause = None
        self.xpos_at_action = self.xpos
        self.ypos_at_action = self.ypos

        # Reset death speeds
        self.death_xspeed = 0
        self.death_yspeed = 0

        # Reset collectibles
        self.gold_collected = 0
        self.doors_opened = 0

        # Reset temporal tracking
        self.frames_airborne = 0
        self.state_change_frame = 0
        self.previous_state = 0
        self.consecutive_floor_frames = 0
        self.consecutive_wall_frames = 0

        # Clear debug logs if enabled
        if self.sim.sim_config.debug:
            if self.poslog is not None:
                self.poslog.clear()
            if self.speedlog is not None:
                self.speedlog.clear()
            if self.xposlog is not None:
                self.xposlog.clear()
            if self.yposlog is not None:
                self.yposlog.clear()

        # Re-log initial position
        self.log()

    def save_state(self) -> dict:
        """Snapshot all physics-relevant state for restoration.

        This method captures the complete physics state of the ninja,
        allowing it to be restored later for value iteration purposes.
        Used by PhysicsStepWrapper to enable stateless physics queries.

        Returns:
            Dictionary containing all physics-critical state fields
        """
        return {
            # Core position and velocity
            "xpos": self.xpos,
            "ypos": self.ypos,
            "xspeed": self.xspeed,
            "yspeed": self.yspeed,
            "xpos_old": self.xpos_old,
            "ypos_old": self.ypos_old,
            "xspeed_old": self.xspeed_old,
            "yspeed_old": self.yspeed_old,
            # Physics parameters
            "applied_gravity": self.applied_gravity,
            "applied_drag": self.applied_drag,
            "applied_friction": self.applied_friction,
            # State and flags
            "state": self.state,
            "airborn": self.airborn,
            "airborn_old": self.airborn_old,
            "walled": self.walled,
            "wall_normal": self.wall_normal,
            # Jump state
            "jump_duration": self.jump_duration,
            "jump_input_old": self.jump_input_old,
            "jump_buffer": self.jump_buffer,
            "floor_buffer": self.floor_buffer,
            "wall_buffer": self.wall_buffer,
            "launch_pad_buffer": self.launch_pad_buffer,
            "is_jump_useless": self.is_jump_useless,
            "last_jump_was_buffered": self.last_jump_was_buffered,
            # Collision normals (derived but needed for death checks)
            "floor_normalized_x": self.floor_normalized_x,
            "floor_normalized_y": self.floor_normalized_y,
            "ceiling_normalized_x": self.ceiling_normalized_x,
            "ceiling_normalized_y": self.ceiling_normalized_y,
            # Collision counters
            "floor_count": self.floor_count,
            "wall_count": self.wall_count,
            "ceiling_count": self.ceiling_count,
            "floor_normal_x": self.floor_normal_x,
            "floor_normal_y": self.floor_normal_y,
            "ceiling_normal_x": self.ceiling_normal_x,
            "ceiling_normal_y": self.ceiling_normal_y,
            "is_crushable": self.is_crushable,
            "x_crush": self.x_crush,
            "y_crush": self.y_crush,
            "crush_len": self.crush_len,
            # Input state
            "hor_input": self.hor_input,
            "jump_input": self.jump_input,
            # Temporal tracking
            "frames_airborne": self.frames_airborne,
            "state_change_frame": self.state_change_frame,
            "previous_state": self.previous_state,
            "consecutive_floor_frames": self.consecutive_floor_frames,
            "consecutive_wall_frames": self.consecutive_wall_frames,
            # Death tracking
            "terminal_impact": self.terminal_impact,
            "death_cause": self.death_cause,
        }

    def restore_state(self, snapshot: dict):
        """Restore ninja to a previous state snapshot.

        This method restores all physics-relevant state from a snapshot
        created by save_state(). Used by PhysicsStepWrapper to enable
        stateless physics queries.

        Args:
            snapshot: Dictionary containing physics state (from save_state())
        """
        for key, value in snapshot.items():
            setattr(self, key, value)

    def has_won(self):
        """Return True if ninja has won (celebrating state, level completed)."""
        return self.state == 8  # Celebrating

    def has_died(self):
        return self.state == 6 or self.state == 7

    def set_death_cause(self, death_cause):
        self.death_cause = death_cause


def _ragdoll_toi_point(xpos, ypos, dx, dy, cx, cy, radius):
    """Time at which a circle of `radius` swept along `(dx, dy)` first touches
    the point `(cx, cy)`.

    `FUN_1050C3A0` (`docs/decomp/R5_04_toi_seg_1050C3A0.c`). This is *not*
    `get_time_of_intersection_circle_vs_circle`: the ninja's version answers 0
    for "already overlapping" and 1 for "no impact", the ragdoll's answers
    `RAGDOLL_TOI_OVERLAP` (-1) and `RAGDOLL_TOI_MISS` (2), and the caller then
    keeps only results in `[0, 1]`. Sharing the ninja's helper here would make
    every already-overlapping stick read as a fresh collision.
    """
    px = xpos - cx
    py = ypos - cy
    a = dx * dx + dy * dy
    b = 2.0 * (px * dx + py * dy)
    c = px * px + py * py - radius * radius
    if c <= 0.0:
        return RAGDOLL_TOI_OVERLAP
    if abs(a) > 0.0001 and b < 0.0:
        radicand = b * b - 4.0 * a * c
        if radicand >= 0.0:
            # The stable form of the quadratic, exactly as the game writes it.
            q = (b - math.sqrt(radicand)) * -0.5
            return min(q / a, c / q)
    return RAGDOLL_TOI_MISS


def _ragdoll_toi_lineseg(xpos, ypos, dx, dy, ax, ay, bx, by, radius):
    """Same, against the body of the line segment `(ax, ay)-(bx, by)`.

    `FUN_1050C4D0` (`docs/decomp/R5_04_toi_1050C4D0.c`).
    """
    wx = bx - ax
    wy = by - ay
    seg_len = math.sqrt(wx * wx + wy * wy)
    inv = 1.0 / seg_len
    nx = wx * inv
    ny = wy * inv
    normal_proj = (xpos - ax) * -ny + (ypos - ay) * nx
    hor_proj = (ypos - ay) * ny + (xpos - ax) * nx
    closing = -ny * dx + nx * dy
    gap = abs(normal_proj) - radius
    if gap < 0.0:
        if 0.0 <= hor_proj <= seg_len:
            return RAGDOLL_TOI_OVERLAP
        return RAGDOLL_TOI_MISS
    if closing * normal_proj >= 0.0:
        return RAGDOLL_TOI_MISS
    t = gap / abs(closing)
    hor_proj2 = (ny * dy + nx * dx) * t + hor_proj
    if 0.0 <= hor_proj2 <= seg_len:
        return t
    return RAGDOLL_TOI_MISS


def _ragdoll_toi_circle(xpos, ypos, dx, dy, cx, cy, ax, ay, bx, by, radius):
    """Same, against a full circle of `radius` centred on `(cx, cy)`, keeping
    only impacts inside the wedge spanned by the arc ends `(ax, ay)`/`(bx, by)`.

    `FUN_1050C0A0` (`docs/decomp/full/physics_1050C0A0.c`).
    """
    px = xpos - cx
    py = ypos - cy
    a = dx * dx + dy * dy
    b = 2.0 * (px * dx + py * dy)
    c = px * px + py * py - radius * radius
    if abs(a) <= 0.0001:
        return RAGDOLL_TOI_MISS
    radicand = b * b - 4.0 * a * c
    if radicand < 0.0:
        return RAGDOLL_TOI_MISS
    q = (b - math.sqrt(radicand)) * -0.5
    guard = abs(q)
    if guard <= 1.0:
        guard = 1.0
    if abs(q) <= guard * RAGDOLL_EPSILON:
        return RAGDOLL_TOI_MISS
    t0 = q / a
    t1 = c / q
    # The two arc ends, as vectors from the centre, and the chord between them.
    arm_ax = ax - cx
    arm_ay = -(ay - cy)
    arm_bx = bx - cx
    arm_by = -(by - cy)
    chord_x = bx - ax
    chord_y = by - ay
    side_a = arm_ax * chord_y + chord_x * arm_ay
    side_b = arm_bx * chord_y + chord_x * arm_by
    if t0 < 0.0:
        t0 = RAGDOLL_TOI_MISS
    if t1 < 0.0:
        t1 = RAGDOLL_TOI_MISS
    if t0 <= 1.0:
        hy = (dy * t0 + ypos) - cy
        hx = (dx * t0 + xpos) - cx
        if (arm_ax * hy + hx * arm_ay) * side_a <= 0.0 or (
            hx * arm_by + arm_bx * hy
        ) * side_b >= 0.0:
            t0 = RAGDOLL_TOI_MISS
    if t1 <= 1.0:
        hx = (dx * t1 + xpos) - cx
        hy = (dy * t1 + ypos) - cy
        if (arm_ax * hy + hx * arm_ay) * side_a <= 0.0 or (
            arm_bx * hy + hx * arm_by
        ) * side_b >= 0.0:
            t1 = RAGDOLL_TOI_MISS
    return min(t0, t1)


def _ragdoll_sweep_segment(segment, xpos, ypos, dx, dy, radius):
    """Time at which a circle of `radius` swept from `(xpos, ypos)` along
    `(dx, dy)` first touches `segment`, in the ragdoll's -1/2 convention.

    `FUN_10518520` (circular, `docs/decomp/full/ninja_10518520.c`) and
    `FUN_10518720` (linear, `ninja_10518720.c`).
    """
    if segment.type == "circular":
        p1x, p1y = segment.p_hor
        p2x, p2y = segment.p_ver
        t1 = _ragdoll_toi_point(xpos, ypos, dx, dy, p1x, p1y, radius)
        t2 = _ragdoll_toi_point(xpos, ypos, dx, dy, p2x, p2y, radius)
        # The arc body is only consulted when we are not already on it; the
        # game leaves the result at -1 otherwise, which rejects the segment.
        _, ca, cb = segment.get_closest_point(xpos, ypos)
        dist = math.sqrt((xpos - ca) ** 2 + (ypos - cb) ** 2)
        t3 = RAGDOLL_TOI_OVERLAP
        if radius < dist:
            arc_r = math.sqrt((p1x - segment.xpos) ** 2 + (p1y - segment.ypos) ** 2)
            t3 = min(
                _ragdoll_toi_circle(
                    xpos, ypos, dx, dy, segment.xpos, segment.ypos,
                    p1x, p1y, p2x, p2y, arc_r + radius,
                ),
                _ragdoll_toi_circle(
                    xpos, ypos, dx, dy, segment.xpos, segment.ypos,
                    p1x, p1y, p2x, p2y, arc_r - radius,
                ),
            )
        best = min(t1, t2)
        return best if best <= t3 else t3
    t1 = _ragdoll_toi_point(xpos, ypos, dx, dy, segment.x1, segment.y1, radius)
    t2 = _ragdoll_toi_point(xpos, ypos, dx, dy, segment.x2, segment.y2, radius)
    t3 = _ragdoll_toi_lineseg(
        xpos, ypos, dx, dy,
        segment.x1, segment.y1, segment.x2, segment.y2, radius,
    )
    best = min(t1, t2)
    return best if best <= t3 else t3


class Ragdoll:
    """The corpse the game hands the ninja when it dies.

    Thirteen Verlet particles held together by eleven fixed-length sticks and
    nine angle constraints, bounced off the tile geometry. Purely cosmetic --
    it only runs once `Ninja.state == 6`, which is after the demo's outcome is
    already decided -- so nothing here can move a gate number.

    Ported from the decompiled game, one function per frame stage:

    ==========================  ==============================================
    stage                       game function
    ==========================  ==============================================
    `activate`                  `FUN_10516A80`  `ninja_10516A80.c`
    `integrate`                 `FUN_105184A0`  `ninja_105184A0.c`
    `pre_collision`             `FUN_10514910`, a no-op in state 6
    `solve_constraints`         `FUN_105193A0`  `ninja_105193A0.c`
    `collide_vs_tiles`          `FUN_10517310`  `ninja_10517310.c`
    `post_collision`            `FUN_105188A0`  `ninja_105188A0.c`
    ==========================  ==============================================

    `nsim.tick` drives these as `integrate`, `pre_collision`,
    4x(`collide_vs_objects`, `collide_vs_tiles`), `post_collision`, which is
    the *same* shape the game's tick uses (`R5_01_simulator_tick_1051D800.c`
    lines 273-325): eight-player loops around one `FUN_10513B50`/`FUN_10514910`
    pair, then four rounds of `FUN_105152E0`/`FUN_105119C0`/`FUN_10511AB0`,
    then a single `FUN_10513C60`. `FUN_105152E0` -- the constraint solve -- has
    no separate hook in nclone's loop, so `collide_vs_objects` runs it first,
    which puts it in the game's position: constraints, entities, tiles.

    Bone indices here are the *ragdoll's*, not the animation's; see
    `RAGDOLL_BONE_SOURCE`.

    Not ported: generic bone-vs-entity depenetration (`FUN_10517100`), which
    would need a per-bone collision entry point on every entity class, and the
    gore particles of `FUN_105188A0`; nclone draws neither. The one visible
    piece of that function -- a limb setting off a live mine -- is ported, in
    `collide_vs_mines`.
    """

    GRAVITY = RAGDOLL_GRAVITY
    DRAG = RAGDOLL_DRAG

    def __init__(self, sim=None):
        self.sim = sim
        self.state = 0
        self.num = RAGDOLL_NUM_BONES
        self.bones_pos_old = [[0.0, 0.0] for _ in range(self.num)]
        self.bones_pos = [[0.0, 0.0] for _ in range(self.num)]
        self.bones_speed = [[0.0, 0.0] for _ in range(self.num)]
        #: Contact normals gathered during `collide_vs_tiles`, cleared by
        #: `integrate` and consumed by `post_collision`. `bone + 0x20`.
        self.bones_normal = [[0.0, 0.0] for _ in range(self.num)]
        #: Frames until a bone may register another impact. `bone + 0x40`.
        self.bones_impact_timer = [0] * self.num
        self.segs = RAGDOLL_SEGS
        self.seg_lengths = RAGDOLL_SEG_LENGTHS
        self.angles = RAGDOLL_ANGLES
        self.dir = 1
        self.active = False
        #: Centroid of the eleven sticks, `FUN_10517080`. The game uses it to
        #: point the camera at the corpse.
        self.xpos = 0.0
        self.ypos = 0.0
        # The game kicks the head with its global RNG. A private stream keeps
        # the sim reproducible without changing the distribution.
        self._rng = random.Random(RAGDOLL_HEAD_KICK_SEED)

    def reseed(self, seed=RAGDOLL_HEAD_KICK_SEED):
        """Reset the head-kick RNG. Called on `activate` so that replaying the
        same demo twice produces the same corpse."""
        self._rng.seed(seed)

    # ------------------------------------------------------------------ setup

    def activate(
        self,
        xpos,
        ypos,
        xspeed,
        yspeed,
        death_xpos,
        death_ypos,
        death_xspeed,
        death_yspeed,
        bones_pos,
        bones_speed,
        dir=1,
    ):
        """`FUN_10516A80`. Snapshot the ninja's skeleton into world space and
        add the death impulse.

        `bones_pos` / `bones_speed` are the ninja's 13 animation bones and their
        per-frame deltas, in ninja diameters relative to the ninja's centre;
        everything the ragdoll stores is absolute pixels.
        """
        self.reseed()
        self.active = True
        self.state = 0
        for i in range(self.num):
            src = RAGDOLL_BONE_SOURCE[i]
            bx, by = bones_pos[src]
            self.bones_pos_old[i][0] = xpos + 24.0 * bx
            self.bones_pos_old[i][1] = ypos + 24.0 * by
            # `integrate` writes bones_pos before anything reads it, but the
            # renderer and `solve_constraints` can both run first on the frame
            # the ninja dies, so seed it too. The stub never did, which is why
            # frame 2 integrated from a stale base for ever.
            self.bones_pos[i][0] = self.bones_pos_old[i][0]
            self.bones_pos[i][1] = self.bones_pos_old[i][1]
            sx, sy = bones_speed[src]
            self.bones_speed[i][0] = xspeed + 24.0 * sx
            self.bones_speed[i][1] = yspeed + 24.0 * sy
            self.bones_normal[i][0] = 0.0
            self.bones_normal[i][1] = 0.0
            self.bones_impact_timer[i] = 0
        self.dir = dir

        # Two head bones get a random kick so no two deaths look identical.
        self.bones_speed[11][0] += RAGDOLL_HEAD_KICK * self._rng.random()
        self.bones_speed[12][1] -= RAGDOLL_HEAD_KICK * self._rng.random()

        # The death impulse, strongest on the bones nearest the thing that
        # killed us. `dist` is measured from the *absolute* bone position; the
        # stub measured animation units against pixels, and then applied the
        # impulse to the caller's list instead of its own, so it never landed.
        for i in range(self.num):
            dx = self.bones_pos_old[i][0] - death_xpos
            dy = self.bones_pos_old[i][1] - death_ypos
            dist = math.sqrt(dx * dx + dy * dy)
            falloff = dist / RAGDOLL_IMPULSE_FALLOFF
            if falloff >= 1.0:
                falloff = 1.0
            scale = (1.0 - falloff) * RAGDOLL_IMPULSE_NEAR + RAGDOLL_IMPULSE_FAR
            self.bones_speed[i][0] += scale * death_xspeed
            self.bones_speed[i][1] += scale * death_yspeed

        self._update_centre()

    def _update_centre(self):
        """`FUN_10517080`. Mean of both endpoints of all eleven sticks."""
        sx = 0.0
        sy = 0.0
        for a, b in self.segs:
            sx += self.bones_pos_old[a][0] + self.bones_pos_old[b][0]
            sy += self.bones_pos_old[a][1] + self.bones_pos_old[b][1]
        self.xpos = sx / 22.0
        self.ypos = sy / 22.0

    def explode(self, origin=None, mid_frame=False):
        """`FUN_10517E20` / `FUN_10517F80`, plus a blast. Burst the corpse.

        Four new bones are cloned from the shoulder and the hip, the stick
        table is swapped for the one that hangs each limb off its clone rather
        than off the torso, and each clone is kicked outwards -- so the arms
        and legs come away while the torso and head stay together. Then
        everything is thrown radially away from `origin`, defaulting to the
        corpse's own centre; see `RAGDOLL_EXPLOSION_FORCE`.

        The skeleton only changes the first time. The blast does not: a second
        mine going off under an already-scattered corpse throws the pieces
        again, which is the point of the chain reaction.
        """
        if self.state != 1:
            self.state = 1
            for slot, src in enumerate(RAGDOLL_EXPLODED_BONE_SOURCE):
                self.bones_pos_old.append(list(self.bones_pos_old[src]))
                self.bones_pos.append(list(self.bones_pos[src]))
                kick_x, kick_y = RAGDOLL_EXPLODED_KICKS[slot]
                self.bones_speed.append(
                    [
                        self.bones_speed[src][0] + kick_x,
                        self.bones_speed[src][1] + kick_y,
                    ]
                )
                self.bones_normal.append(list(self.bones_normal[src]))
                self.bones_impact_timer.append(self.bones_impact_timer[src])
            self.num = RAGDOLL_EXPLODED_NUM_BONES
            self.segs = RAGDOLL_EXPLODED_SEGS
            self.seg_lengths = RAGDOLL_EXPLODED_SEG_LENGTHS
            self.angles = RAGDOLL_EXPLODED_ANGLES
        self.blast(origin, mid_frame=mid_frame)

    def blast(self, origin=None, mid_frame=False):
        """Throw every bone away from `origin` (default: the corpse's centre).

        Falls off with distance but never to nothing, so a limb already across
        the room still gets moved rather than sitting there while the rest of
        the corpse leaves.

        `mid_frame` picks where the impulse lands, and getting it wrong makes
        the blast vanish. In a Verlet integrator the velocity is not stored --
        `post_collision` recomputes `bones_speed` as `pos - pos_old` at the end
        of every frame, so anything written to `bones_speed` *during* a frame is
        overwritten before it is ever integrated. A mid-frame impulse therefore
        has to go into the position: displacing `pos` by `v` while `pos_old`
        stays put makes the next `pos - pos_old` come out `v` larger, which is
        the same impulse and survives. `activate` runs outside the frame, before
        any integrate, where the opposite holds -- `pos` is about to be rebuilt
        from `pos_old + speed`, so there the impulse goes into the speed.
        """
        if RAGDOLL_EXPLOSION_FORCE == 0.0:
            return
        if origin is None:
            origin = (self.xpos, self.ypos)
        ox, oy = origin
        span = 1.0 - RAGDOLL_EXPLOSION_FLOOR
        target = self.bones_pos if mid_frame else self.bones_speed
        for i in range(self.num):
            dx = self.bones_pos[i][0] - ox
            dy = self.bones_pos[i][1] - oy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-9:
                # Sitting on the blast centre: no direction to push along, so
                # pick one. Deterministic, from the ragdoll's own stream.
                angle = self._rng.random() * math.tau
                dx, dy = math.cos(angle), math.sin(angle)
            else:
                dx /= dist
                dy /= dist
            falloff = dist / RAGDOLL_EXPLOSION_RADIUS
            if falloff > 1.0:
                falloff = 1.0
            mag = RAGDOLL_EXPLOSION_FORCE * (
                (1.0 - falloff) * span + RAGDOLL_EXPLOSION_FLOOR
            )
            target[i][0] += dx * mag
            target[i][1] += dy * mag

    # ------------------------------------------------------------- per frame

    def integrate(self):
        """`FUN_105184A0`. Drag, gravity, and the Verlet step."""
        for i in range(self.num):
            speed = self.bones_speed[i]
            sx = self.DRAG * speed[0]
            sy = self.DRAG * speed[1] + self.GRAVITY
            speed[0] = sx
            speed[1] = sy
            self.bones_pos[i][0] = self.bones_pos_old[i][0] + sx
            self.bones_pos[i][1] = self.bones_pos_old[i][1] + sy
            self.bones_normal[i][0] = 0.0
            self.bones_normal[i][1] = 0.0

    def pre_collision(self):
        """`FUN_10514910` returns immediately in state 6."""
        return

    def solve_constraints(self):
        """`FUN_105193A0`. One Gauss-Seidel pass over the eleven sticks, then
        one over the nine angle constraints.

        The game runs this four times per frame (once per collision substep),
        which `collide_vs_objects` reproduces.
        """
        pos = self.bones_pos
        for i, (a, b) in enumerate(self.segs):
            ax, ay = pos[a]
            bx, by = pos[b]
            dx = bx - ax
            dy = by - ay
            dist = math.sqrt(dx * dx + dy * dy)
            guard = abs(dist)
            if guard <= 1.0:
                guard = 1.0
            correction = (dist - self.seg_lengths[i]) * 0.5
            if abs(dist) <= guard * RAGDOLL_EPSILON:
                nx = 0.0
                ny = 0.0
            else:
                nx = dx / dist * correction
                ny = dy / dist * correction
            pos[a][0] = ax + nx
            pos[a][1] = ay + ny
            pos[b][0] = bx - nx
            pos[b][1] = by - ny
        self._solve_angles()

    def _solve_angles(self):
        """The second half of `FUN_105193A0`: hold each joint near a target
        angle, within a tolerance, by rotating the two arms about the centre
        bone (the game calls it `Solve_Angle_Analytical`)."""
        pos = self.bones_pos
        facing = float(self.dir)
        for centre, arm_a, arm_b, target, tolerance in self.angles:
            # The game's own "two pi" is a truncated literal; wrapping against
            # math.tau here would move every joint by ~5e-8 rad per pass.
            want = facing * target
            while want < 0.0 or want > RAGDOLL_TWO_PI:
                want = want + RAGDOLL_TWO_PI if want < 0.0 else want - RAGDOLL_TWO_PI

            cx, cy = pos[centre]
            aax = pos[arm_a][0] - cx
            aay = pos[arm_a][1] - cy
            abx = pos[arm_b][0] - cx
            aby = pos[arm_b][1] - cy

            # Ghidra loses which of `FUN_105193A0`'s two `_CIatan2` calls got
            # which arm (both arguments arrive on the x87 stack), so the sign
            # of this subtraction is not readable from the decompilation. It is
            # forced by the gradients below: they are -2*d(angle_a)/da and
            # +2*d(angle_b)/db, and the position update is a subtraction, so
            # only `arm_b - arm_a` moves the joint *toward* its target. Taken
            # the other way the solver adds (excess - tolerance) to the error
            # every pass and the corpse tears itself apart.
            delta = math.atan2(aby, abx) - math.atan2(aay, aax)
            while delta < 0.0 or delta > RAGDOLL_TWO_PI:
                delta = (
                    delta + RAGDOLL_TWO_PI if delta < 0.0 else delta - RAGDOLL_TWO_PI
                )
            delta -= want
            while abs(delta) > math.pi:
                delta = delta - RAGDOLL_TWO_PI if delta >= 0.0 else delta + RAGDOLL_TWO_PI

            excess = abs(delta)
            if excess <= tolerance:
                continue
            scale = (excess - tolerance) * (delta / excess + delta / excess)
            if scale == 0.0:
                continue

            len_a = aay * aay + aax * aax
            len_b = abx * abx + aby * aby
            if len_a == 0.0 or len_b == 0.0:
                logger.debug(
                    "ragdoll angle constraint has a null edge: %f, %f", len_a, len_b
                )
                continue
            # d(angle)/d(position) for each of the three bones. Written in the
            # game's exact order of operations -- divide by the *negated*
            # squared length, then double -- because these feed a sum of
            # squares that the epsilon guard below compares against.
            neg_len_a = -len_a
            ga_y = aax / neg_len_a
            ga_x = -aay / neg_len_a
            ga_y += ga_y
            gb_x = -aby / len_b
            ga_x += ga_x
            gb_y = abx / len_b + abx / len_b
            gb_x += gb_x
            gc_y = -ga_y - gb_y
            gc_x = -ga_x - gb_x
            denom = (
                gc_y * gc_y
                + gc_x * gc_x
                + ga_y * ga_y
                + ga_x * ga_x
                + gb_y * gb_y
                + gb_x * gb_x
            )
            guard = abs(denom)
            if guard <= 1.0:
                guard = 1.0
            if abs(denom) <= guard * RAGDOLL_EPSILON:
                continue
            step = scale / denom
            mean_x = (-gc_x - ga_x - gb_x) / 3.0
            mean_y = (-gc_y - ga_y - gb_y) / 3.0
            pos[centre][0] -= (gc_x - mean_x) * step
            pos[centre][1] -= (gc_y - mean_y) * step
            pos[arm_a][0] -= (ga_x - mean_x) * step
            pos[arm_a][1] -= (ga_y - mean_y) * step
            pos[arm_b][0] -= (gb_x - mean_x) * step
            pos[arm_b][1] -= (gb_y - mean_y) * step

    def collide_vs_objects(self):
        """The game's `FUN_105152E0` (constraints) plus `FUN_105119C0`
        (entities) for this substep."""
        self.solve_constraints()
        self.collide_vs_mines()

    def collide_vs_mines(self):
        """A flying limb sets off any live mine it lands on.

        The game does this in `FUN_105188A0`: every bone gathers nearby
        entities and the branch for types 0x01 / 0x15 (the two toggle-mine
        types) tears the corpse up. nclone draws neither the gore particles nor
        the generic bone-vs-entity depenetration, so this ports the one part
        that is visible -- the detonation, and the blast it throws.

        Only reachable from `Ragdoll.collide_vs_objects`, which `nsim.tick`
        only calls once the ninja is in state 6, so this cannot touch a live
        ninja's path.
        """
        sim = self.sim
        if sim is None:
            return
        fired = 0
        for i in range(self.num):
            bx, by = self.bones_pos[i]
            if not (math.isfinite(bx) and math.isfinite(by)):
                continue
            for entity in gather_entities_from_neighbourhood(sim, bx, by):
                if entity.type not in (1, 21) or not entity.active:
                    continue
                # State 0 is armed; 1 is spent and 2 is mid-toggle.
                if getattr(entity, "state", 1) != 0 or getattr(
                    entity, "detonated", False
                ):
                    continue
                reach = entity.RADIUS + RAGDOLL_BONE_RADIUS + RAGDOLL_MINE_TOUCH_SLACK
                dx = bx - entity.xpos
                dy = by - entity.ypos
                if dx * dx + dy * dy > reach * reach:
                    continue
                # Deliberately NOT entity.set_state(1). Changing a mine's state
                # moves `get_mine_state_signature`, which throws away the
                # reachability path cache and forces a full recompute
                # (`graph/reachability/path_visualization_cache.py`) -- fifteen
                # of them in a good chain reaction, on a level whose episode is
                # already over. `detonated` alone does everything that is still
                # needed: the renderer stops drawing the mine, this loop will
                # not fire it twice, and `logical_collision` is already inert
                # because a dead ninja is not a valid target.
                entity.detonated = True
                self.explode((entity.xpos, entity.ypos), mid_frame=True)
                fired += 1
                if fired >= RAGDOLL_MAX_CHAIN_PER_FRAME:
                    return

    def collide_vs_tiles(self):
        """`FUN_10517310`. Sweep each bone against the geometry, push it back
        out, then keep whole limbs from lying across a tile corner."""
        if self.sim is None:
            return
        self._collide_bones_vs_tiles()
        self._collide_sticks_vs_tiles()

    def _collide_bones_vs_tiles(self):
        """The first half of `FUN_10517310`: per bone, a swept move followed by
        up to 32 rounds of closest-point depenetration.

        Same shape as the ninja's own `collide_vs_tiles`, and as the death
        ball's loop, with three differences the game insists on: the sweep uses
        radius 0.5 while the depenetration uses radius 1.0, the sweep is skipped
        entirely for moves shorter than sqrt(0.1), and each correction also
        cancels the part of the frame's motion that runs *along* the surface.
        """
        sim = self.sim
        for i in range(self.num):
            x_old, y_old = self.bones_pos_old[i]
            dx = self.bones_pos[i][0] - x_old
            dy = self.bones_pos[i][1] - y_old
            if dx * dx + dy * dy > RAGDOLL_SWEEP_MIN_DIST_SQ:
                time = sweep_circle_vs_tiles(
                    sim, x_old, y_old, dx, dy, RAGDOLL_SWEEP_RADIUS
                )
                self.bones_pos[i][0] = x_old + time * dx
                self.bones_pos[i][1] = y_old + time * dy

            for _ in range(RAGDOLL_MAX_DEPEN_ITERS):
                xpos, ypos = self.bones_pos[i]
                result, closest_point = get_single_closest_point(
                    sim, xpos, ypos, RAGDOLL_GATHER_RADIUS
                )
                if result == 0:
                    break
                a, b = closest_point
                dx = xpos - a
                dy = ypos - b
                dist = math.sqrt(dx * dx + dy * dy)
                depen_len = RAGDOLL_BONE_RADIUS - dist * result
                if depen_len <= RAGDOLL_EPSILON:
                    break
                guard = abs(dist)
                if guard <= RAGDOLL_BONE_RADIUS:
                    guard = RAGDOLL_BONE_RADIUS
                if guard < 0.0:
                    guard = 0.0
                if abs(dist) <= guard * RAGDOLL_EPSILON:
                    break
                nx = dx / dist
                ny = dy / dist
                depen = result * depen_len
                new_x = xpos + depen * nx
                new_y = ypos + depen * ny
                # How far this frame's motion has carried the bone *along* the
                # surface since pos_old, signed by the surface tangent.
                slide = (new_y - y_old) * nx + (new_x - x_old) * -ny
                mag = abs(slide)
                if depen + depen <= mag:
                    if mag <= 0.0:
                        # Depenetrated but sliding immeasurably: keep the push
                        # and leave the tangent alone.
                        self.bones_pos[i][0] = new_x
                        self.bones_pos[i][1] = new_y
                        self.bones_normal[i][0] += nx
                        self.bones_normal[i][1] += ny
                        continue
                    slide *= min(depen * RAGDOLL_SLIDE_SCALE / mag, 1.0)
                new_y -= slide * nx
                new_x -= slide * -ny
                self.bones_pos[i][0] = new_x
                self.bones_pos[i][1] = new_y
                self.bones_normal[i][0] += nx
                self.bones_normal[i][1] += ny

    def _collide_sticks_vs_tiles(self):
        """The second half of `FUN_10517310`: a limb lying across a tile corner
        gets pushed off it.

        A stick only registers when it crosses *two* segments -- the two faces
        meeting at the corner it is draped over. One segment means the bone
        depenetration above already has it; more than two, or two that share no
        endpoint, and the game gives up with a log line.
        """
        sim = self.sim
        pos = self.bones_pos
        for index, (bone_a, bone_b) in enumerate(self.segs):
            ax, ay = pos[bone_a]
            bx, by = pos[bone_b]
            dx = bx - ax
            dy = by - ay
            if dx * dx + dy * dy == 0.0:
                logger.debug("ragdoll stick is degenerate: %d", index)
                continue

            r = RAGDOLL_BONE_RADIUS
            segments = gather_segments_from_region(
                sim,
                min(ax, bx) - r,
                min(ay, by) - r,
                max(ax, bx) + r,
                max(ay, by) + r,
            )
            hits = []
            for segment in segments:
                time = _ragdoll_sweep_segment(
                    segment, ax, ay, dx, dy, RAGDOLL_STICK_RADIUS
                )
                if 0.0 <= time <= 1.0:
                    if len(hits) < 2:
                        hits.append(segment)
                    else:
                        logger.debug(
                            "ragdoll stick collision found way too many colliding "
                            "segs: %d", len(hits)
                        )
            if len(hits) != 2:
                continue

            corner = self._shared_corner(hits[0], hits[1])
            if corner is None:
                logger.debug("ragdoll stick collision completely penetrated a tile")
                continue
            px, py = corner

            # Unit normal of the stick, and the corner's signed distance from
            # the stick's line.
            nx = -dy
            ny = dx
            length = math.sqrt(nx * nx + ny * ny)
            if length != 0.0:
                nx /= length
                ny /= length
            to_corner_x = px - ax
            to_corner_y = py - ay
            signed = to_corner_y * ny + to_corner_x * nx

            if not self._corner_is_behind(hits[0], ax, ay, dx, dy, nx, ny, signed):
                nx = -nx
                ny = -ny
                signed = -signed

            push = signed + RAGDOLL_STICK_RADIUS
            # Where along the stick the corner sits; the near end takes most of
            # the correction.
            along = (to_corner_x * dx + to_corner_y * dy) / (dx * dx + dy * dy)
            pos[bone_a][0] += push * nx * (1.0 - along)
            pos[bone_a][1] += push * ny * (1.0 - along)
            pos[bone_b][0] += push * nx * along
            pos[bone_b][1] += push * ny * along

    @staticmethod
    def _shared_corner(seg_a, seg_b):
        """The endpoint the two colliding segments have in common, or None if
        they have none (the stick went clean through a tile)."""
        a1, a2 = Ragdoll._segment_endpoints(seg_a)
        b1, b2 = Ragdoll._segment_endpoints(seg_b)
        dx = b2[0] - a1[0]
        dy = b2[1] - a1[1]
        if dx * dx + dy * dy < RAGDOLL_DEGENERATE_SQ:
            return a1
        dx = b1[0] - a2[0]
        dy = b1[1] - a2[1]
        if dx * dx + dy * dy < RAGDOLL_DEGENERATE_SQ:
            return a2
        return None

    @staticmethod
    def _segment_endpoints(segment):
        """`+0x08` and `+0x18` of a segment: the two ends of a linear segment,
        the two ends of the arc for a circular one."""
        if segment.type == "circular":
            return segment.p_hor, segment.p_ver
        return (segment.x1, segment.y1), (segment.x2, segment.y2)

    @staticmethod
    def _corner_is_behind(segment, ax, ay, dx, dy, nx, ny, signed):
        """Whether `(nx, ny)` already points away from the tile.

        The game probes the segment twice with a zero-radius sweep: once from
        the stick's own start, once from a start nudged one unit along the
        candidate normal.
        """
        if signed != 0.0:
            time = _ragdoll_sweep_segment(segment, ax, ay, dx, dy, 0.0)
            if time < 0.0 or time > 1.0:
                return signed < 0.0
            return signed > 0.0
        time = _ragdoll_sweep_segment(segment, ax + nx, ay + ny, dx, dy, 0.0)
        return not 0.0 <= time <= 1.0

    def post_collision(self):
        """`FUN_105188A0`. Re-derive the Verlet velocity from where collisions
        actually left each bone, then bounce off whatever it touched.

        This is the step the stub was missing entirely: without
        `bones_pos_old = bones_pos` here, `integrate` rebuilds every position
        from the frame the ninja died and the corpse never moves.
        """
        for i in range(self.num):
            x, y = self.bones_pos[i]
            self.bones_speed[i][0] = x - self.bones_pos_old[i][0]
            self.bones_speed[i][1] = y - self.bones_pos_old[i][1]
            self.bones_pos_old[i][0] = x
            self.bones_pos_old[i][1] = y

        impacted = [False] * self.num
        for i in range(self.num):
            nx, ny = self.bones_normal[i]
            length = math.sqrt(nx * nx + ny * ny)
            guard = abs(length)
            if guard <= 1.0:
                guard = 1.0
            if abs(length) <= guard * RAGDOLL_EPSILON:
                continue
            nx /= length
            ny /= length
            self.bones_normal[i][0] = nx
            self.bones_normal[i][1] = ny
            sx, sy = self.bones_speed[i]
            approach = nx * sx + ny * sy
            if approach >= 0.0:
                continue
            normal_part = approach * RAGDOLL_BOUNCE
            tangent_part = (nx * sy - ny * sx) * RAGDOLL_FRICTION
            self.bones_speed[i][0] = sx - (nx * normal_part - ny * tangent_part)
            self.bones_speed[i][1] = sy - (ny * normal_part + nx * tangent_part)
            if self.bones_impact_timer[i] == 0:
                # Where the game plays the thud and throws dust; nclone has
                # neither, so only the cooldown survives the port.
                impacted[i] = True

        # 20 frames of quiet before a bone may register another impact, so a
        # corpse settling into a corner does not machine-gun the effect.
        for i in range(self.num):
            if impacted[i]:
                self.bones_impact_timer[i] = 20
            elif self.bones_impact_timer[i] >= 1:
                self.bones_impact_timer[i] -= 1
            else:
                self.bones_impact_timer[i] = 0

        self._update_centre()
