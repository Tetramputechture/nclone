import struct
import math
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
    JUMP_FLOOR_Y,
    JUMP_SLOPE_DOWNHILL_X_MULTIPLIER,
    JUMP_SLOPE_DOWNHILL_Y_MULTIPLIER,
    JUMP_SLOPE_UPHILL_DEFAULT_Y,
    JUMP_SLOPE_UPHILL_PERP_X_MULTIPLIER,
    JUMP_SLOPE_UPHILL_PERP_Y_MULTIPLIER,
    JUMP_WALL_SLIDE_X_MULTIPLIER,
    JUMP_WALL_SLIDE_Y,
    JUMP_WALL_REGULAR_X_MULTIPLIER,
    JUMP_WALL_REGULAR_Y,
    JUMP_LAUNCH_PAD_BOOST_SCALAR,
    JUMP_LAUNCH_PAD_BOOST_FACTOR,
    # Ragdoll constants
    RAGDOLL_GRAVITY,
    RAGDOLL_DRAG,
)

import logging

logger = logging.getLogger(__name__)

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
        if ninja_anim_mode:
            self.bones = [[0, 0] for _ in range(13)]
            self.update_graphics()
            self.ragdoll = Ragdoll()
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
                    if (
                        entity.type != 17
                    ):  # Update crushing parameters unless collision with bounce block.
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
                    if entity.type == 11:  # Depenetration for one ways
                        xspeed_new = (
                            self.xspeed * depen_y - self.yspeed * depen_x
                        ) * depen_y
                        yspeed_new = (self.xspeed * depen_y - self.yspeed * depen_x) * (
                            -depen_x
                        )
                        self.xspeed = xspeed_new
                        self.yspeed = yspeed_new
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

        # Gather segments once for the entire depenetration loop
        # The ninja moves very little during depenetration, so segments rarely change
        rad = NINJA_RADIUS
        segments = gather_segments_from_region(
            self.sim, self.xpos - rad, self.ypos - rad, self.xpos + rad, self.ypos + rad
        )

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
            result, closest_point = get_single_closest_point(
                self.sim, xpos, ypos, NINJA_RADIUS, segments=segments
            )
            if result == 0:
                break
            a, b = closest_point
            dx = xpos - a
            dy = ypos - b
            # This part tries to reproduce corner cases in some positions. Band-aid solution, and not in the game's code.
            if abs(dx) <= 0.0000001:
                dx = 0
                if xpos in (50.51197510492316, 49.23232124849253):
                    dx = -(2**-47)
                if xpos == 49.153536108584795:
                    dx = 2**-47

            # Compute distance squared first for early exit optimization
            dx_sq = dx * dx
            dy_sq = dy * dy
            dist_sq = dx_sq + dy_sq

            # Early exit before expensive sqrt if distance is too small
            if dist_sq < 1e-16:  # Equivalent to dist < 1e-8
                break

            dist = math.sqrt(dist_sq)
            depen_len = NINJA_RADIUS - dist * result
            if depen_len < 0.0000001:
                break

            # Compute normalized direction once and reuse
            inv_dist = 1.0 / dist
            norm_dx = dx * inv_dist
            norm_dy = dy * inv_dist

            # Apply depenetration using precomputed normalized direction
            depen_x = norm_dx * depen_len
            depen_y = norm_dy * depen_len
            xpos += depen_x
            ypos += depen_y
            x_crush += depen_x
            y_crush += depen_y
            crush_len += depen_len

            # Project velocity onto surface only if moving towards surface
            dot_product = xspeed * dx + yspeed * dy
            if dot_product < 0:
                cross_product = xspeed * dy - yspeed * dx
                inv_dist_sq = inv_dist * inv_dist
                xspeed = cross_product * inv_dist_sq * dy
                yspeed = cross_product * inv_dist_sq * (-dx)

            # Adjust ceiling/floor variables using precomputed normalized direction
            if dy >= -0.0001:
                ceiling_count += 1
                ceiling_normal_x += norm_dx
                ceiling_normal_y += norm_dy
            else:
                floor_count += 1
                floor_normal_x += norm_dx
                floor_normal_y += norm_dy

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
        wall_normal = 0
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
                        wall_normal += collision_result

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
                wall_normal += dx / dist

        # Check if airborn or walled.
        self.airborn_old = self.airborn
        self.airborn = True
        self.walled = False
        if wall_normal:
            self.walled = True
            self.wall_normal = wall_normal / abs(wall_normal)

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
                self.kill(2, self.xpos, self.ypos, 0, 0)

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
        if self.floor_normalized_x == 0:  # Jump from flat ground
            jx = 0
            jy = JUMP_FLOOR_Y
        else:  # Slope jump
            dx = self.floor_normalized_x
            dy = self.floor_normalized_y
            if self.xspeed * dx >= 0:  # Moving downhill
                if self.xspeed * self.hor_input >= 0:
                    jx = JUMP_SLOPE_DOWNHILL_X_MULTIPLIER * dx
                    jy = JUMP_SLOPE_DOWNHILL_Y_MULTIPLIER * dy
                else:
                    jx = 0
                    jy = JUMP_SLOPE_UPHILL_DEFAULT_Y
            else:  # Moving uphill
                if self.xspeed * self.hor_input > 0:  # Forward jump
                    jx = 0
                    jy = JUMP_SLOPE_UPHILL_DEFAULT_Y
                else:
                    self.xspeed = 0  # Perp jump
                    jx = JUMP_SLOPE_UPHILL_PERP_X_MULTIPLIER * dx
                    jy = JUMP_SLOPE_UPHILL_PERP_Y_MULTIPLIER * dy
        if self.yspeed > 0:
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
        if self.hor_input * self.wall_normal < 0 and self.state == 5:  # Slide wall jump
            jx = JUMP_WALL_SLIDE_X_MULTIPLIER
            jy = JUMP_WALL_SLIDE_Y
        else:  # Regular wall jump
            jx = JUMP_WALL_REGULAR_X_MULTIPLIER
            jy = JUMP_WALL_REGULAR_Y
        if was_buffered:
            self.last_jump_was_buffered = True
        self.state = 3
        self.applied_gravity = GRAVITY_JUMP
        if self.xspeed * self.wall_normal < 0:
            self.xspeed = 0
        if self.yspeed > 0:
            self.yspeed = 0
        self.xspeed += jx * self.wall_normal
        self.yspeed += jy
        self.xpos += jx * self.wall_normal
        self.ypos += jy
        self.jump_buffer = -1
        self.wall_buffer = -1
        self.launch_pad_buffer = -1
        self.jump_duration = 0

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

        # Determine if jump is useless
        # Jump is useless when: airborne AND not jumping AND no buffers AND won't create buffer
        #
        # Key insight: Mask is computed BEFORE action, so we predict buffer activation.
        # - jump_input == 0: Selecting JUMP will be new press → activates buffer → NOT useless
        # - jump_input != 0: Selecting JUMP again won't activate buffer → useless unless buffer exists
        # - Not airborne or jumping: Jump has immediate effect → NOT useless
        if self.airborn and self.state != 3:
            if self.jump_input != 0:
                # Already holding jump → selecting again won't activate buffer
                jump_useless = not has_active_buffer
            else:
                # Not holding jump → selecting JUMP will activate buffer → NOT useless
                jump_useless = False
        else:
            # Not airborne or already jumping → jump has immediate effect
            jump_useless = False

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
        enable_path_masking = getattr(self.sim, "_enable_straightness_masking", True)
        masking_mode = getattr(self.sim, "_action_masking_mode", "hard")
        if enable_path_masking and masking_mode == "hard":
            path_direction_data = getattr(self.sim, "_path_direction_data", None)
            if path_direction_data is not None:
                from .constants import (
                    PATH_DIRECTION_HORIZONTAL_THRESHOLD,
                    PATH_DIRECTION_VERTICAL_THRESHOLD,
                    STRAIGHT_PATH_HORIZONTAL_THRESHOLD,
                    STRAIGHT_PATH_VERTICAL_THRESHOLD,
                    STRAIGHT_PATH_DOWNWARD_THRESHOLD,
                )

                dx = path_direction_data.get("dx", 0.0)
                dy = path_direction_data.get("dy", 0.0)
                # distance = path_direction_data.get("distance", 0.0)  # Available if needed

                # === UNIVERSAL HORIZONTAL MASKING (Applied First) ===
                # ALWAYS mask actions moving opposite to goal's horizontal direction
                # This applies regardless of airborne state or path straightness
                # Rationale: If goal is LEFT, moving RIGHT is ALWAYS suboptimal
                if abs(dx) > PATH_DIRECTION_HORIZONTAL_THRESHOLD:
                    if dx < 0:  # Goal is to the LEFT
                        mask[2] = False  # Mask RIGHT (always wrong direction)
                        mask[5] = False  # Mask JUMP+RIGHT (always wrong direction)
                    else:  # Goal is to the RIGHT
                        mask[1] = False  # Mask LEFT (always wrong direction)
                        mask[4] = False  # Mask JUMP+LEFT (always wrong direction)

                # === UNIVERSAL VERTICAL MASKING (Applied Second) ===
                # ALWAYS mask jumps when goal is BELOW and ninja is moving UPWARD
                # Rationale: If goal is below and we're going up, continuing to jump is counterproductive
                # Let gravity bring us back down instead
                if dy > PATH_DIRECTION_VERTICAL_THRESHOLD and self.yspeed < 0:
                    # Goal is BELOW (dy > 0) and ninja is moving UP (yspeed < 0)
                    mask[3] = False  # Mask JUMP (going up when we need to go down)
                    mask[4] = False  # Mask JUMP+LEFT (going up when we need to go down)
                    mask[5] = False  # Mask JUMP+RIGHT (going up when we need to go down)

                # === STRICT STRAIGHT PATH MASKING (Additional Restrictions) ===
                # For VERY straight paths, apply even more aggressive masking
                # Uses stricter thresholds than general direction masking

                # Check for very straight horizontal path (flat corridor)
                is_very_straight_horizontal = (
                    abs(dy) < STRAIGHT_PATH_VERTICAL_THRESHOLD
                    and abs(dx) > STRAIGHT_PATH_HORIZONTAL_THRESHOLD
                )

                # Check for very straight downward path (vertical drop)
                is_very_straight_downward = (
                    dy > STRAIGHT_PATH_DOWNWARD_THRESHOLD and not self.airborn
                )

                if is_very_straight_horizontal:
                    # Path is completely straight horizontal - mask everything except one direction
                    if dx < 0:  # Straight path to the LEFT
                        mask[0] = False  # Mask NOOP
                        mask[2] = False  # Mask RIGHT (wrong direction)
                        mask[3] = False  # Mask JUMP (unnecessary on flat path)
                        mask[4] = False  # Mask JUMP+LEFT (jumping unnecessary)
                        mask[5] = (
                            False  # Mask JUMP+RIGHT (wrong direction + unnecessary)
                        )
                        # Keep only: LEFT (1)
                    else:  # Straight path to the RIGHT
                        mask[0] = False  # Mask NOOP
                        mask[1] = False  # Mask LEFT (wrong direction)
                        mask[3] = False  # Mask JUMP (unnecessary on flat path)
                        mask[4] = (
                            False  # Mask JUMP+LEFT (wrong direction + unnecessary)
                        )
                        mask[5] = False  # Mask JUMP+RIGHT (jumping unnecessary)
                        # Keep only: RIGHT (2)

                elif is_very_straight_downward:
                    # Path goes straight down - mask all jumps (jumping goes UP)
                    # Keep horizontal movement and NOOP for navigation while falling
                    mask[3] = False  # Mask JUMP (moves up when we need down)
                    mask[4] = False  # Mask JUMP+LEFT (moves up when we need down)
                    mask[5] = False  # Mask JUMP+RIGHT (moves up when we need down)
                    # Keep: NOOP (0), LEFT (1), RIGHT (2)

                else:
                    # Path is NOT very straight - apply additional conditional masking
                    # Horizontal masking already applied universally above

                    # === VERTICAL MASKING ===
                    # Mask jumps when goal is DOWN and ninja is grounded
                    # (jumping moves UP, opposite of goal direction)
                    if abs(dy) > PATH_DIRECTION_VERTICAL_THRESHOLD:
                        if dy > 0 and not self.airborn:  # Goal is DOWN, ninja grounded
                            mask[3] = False  # Mask JUMP (moves up when we need down)
                            mask[4] = (
                                False  # Mask JUMP+LEFT (moves up when we need down)
                            )
                            mask[5] = (
                                False  # Mask JUMP+RIGHT (moves up when we need down)
                            )
                        # Note: if dy < 0 (goal UP), we DON'T mask jumps - ninja needs to jump up

                    # === NOOP MASKING ===
                    # Mask NOOP when grounded with clear directional path
                    # Keep NOOP when airborne (needed for precise air control)
                    if not self.airborn:
                        if abs(dx) > PATH_DIRECTION_HORIZONTAL_THRESHOLD:
                            mask[0] = False  # Mask NOOP - should be moving toward goal

        # Ensure at least one action is always valid
        # If all actions are masked (inevitable death scenario), unmask NOOP
        # This prevents NaN in the policy distribution
        if not any(mask):
            # Ninja is in inevitable death scenario (surrounded by mines, etc.)
            # Allow NOOP as a last resort to prevent policy NaN
            mask[0] = True

        # Validate mask before returning
        # These assertions catch bugs early before they cause NaN in the policy
        assert isinstance(mask, list), f"Mask must be a list, got {type(mask)}"
        assert len(mask) == 6, f"Mask must have 6 elements, got {len(mask)}"
        assert all(isinstance(m, bool) for m in mask), (
            f"All mask elements must be boolean, got types: {[type(m) for m in mask]}"
        )
        assert any(mask), (
            "At least one action must be valid. This should be enforced by "
            "the fallback above. If you see this, there's a logic bug."
        )

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
            self.is_jump_useless = (
                self.airborn and self.state != 3 and not has_active_buffer
            )
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
            )

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
            self.tilt = (
                math.atan2(self.floor_normalized_y, self.floor_normalized_x)
                + math.pi / 2
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
                self.tilt *= 0.9
        if self.state != 5:
            if abs(self.xspeed) > 0.01:
                self.facing = 1 if self.xspeed > 0 else -1

        if self.anim_state != anim_state_old:
            if self.anim_state == 0:
                if self.anim_frame > 0:
                    self.anim_frame = 1
            if self.anim_state == 1:
                if anim_state_old != 3:
                    if anim_state_old == 2:
                        self.anim_frame = 39
                        self.run_cycle = 162
                        self.frame_residual = 0
                    else:
                        self.anim_frame = 12
                        self.run_cycle = 0
                        self.frame_residual = 0
                else:
                    self.anim_frame = 18
                    self.run_cycle = 36
                    self.frame_residual = 0
            if self.anim_state == 2:
                self.anim_frame = 0
            if self.anim_state == 3:
                self.anim_frame = 84
            if self.anim_state == 4:
                self.anim_frame = 103
            if self.anim_state == 6:
                self.dance_id = (
                    random.choice(list(DANCE_DIC)) if DANCE_RANDOM else DANCE_ID_DEFAULT
                )
                self.anim_frame = DANCE_DIC[self.dance_id][0]

        if self.anim_state == 0:
            if self.anim_frame < 11:
                self.anim_frame += 1
        if self.anim_state == 1:
            new_cycle = self.anim_rate / 0.15 + self.frame_residual
            self.frame_residual = new_cycle - math.floor(new_cycle)
            self.run_cycle = (self.run_cycle + math.floor(new_cycle)) % 432
            self.anim_frame = self.run_cycle // 6 + 12
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
            interpolation = (self.run_cycle % 6) / 6
            if interpolation > 0:
                next_bones = self.ninja_animation[(self.anim_frame - 12) % 72 + 12]
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

    def has_won(self):
        return self.state == 8

    def has_died(self):
        return self.state == 6 or self.state == 7

    def set_death_cause(self, death_cause):
        self.death_cause = death_cause


class Ragdoll:
    """None of this is working yet. Might never will."""

    GRAVITY = RAGDOLL_GRAVITY
    DRAG = RAGDOLL_DRAG

    def __init__(self):
        self.state = 0
        self.num = 13
        self.bones_pos_old = [[0, 0] for _ in range(self.num)]
        self.bones_speed_old = [[0, 0] for _ in range(self.num)]
        self.bones_pos = [[0, 0] for _ in range(self.num)]
        self.bones_speed = [[0, 0] for _ in range(self.num)]
        self.segs = (
            (0, 12),
            (1, 12),
            (2, 8),
            (3, 9),
            (4, 10),
            (5, 11),
            (6, 7),
            (8, 0),
            (9, 0),
            (10, 1),
            (11, 1),
        )

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
    ):
        self.bones_pos_old = [
            [xpos + 24 * bone[0], ypos + 24 * bone[1]] for bone in bones_pos
        ]
        self.bones_speed = [
            [xspeed + 24 * bone[0], yspeed + 24 * bone[1]] for bone in bones_speed
        ]
        for i in range(self.num):
            dist = math.sqrt(
                (bones_pos[i][0] - death_xpos) ** 2
                + (bones_pos[i][1] - death_ypos) ** 2
            )
            scale = max(1 - dist / 12, 0) * 1.5 + 0.5
            bones_speed[i][0] += scale * death_xspeed
            bones_speed[i][1] += scale * death_yspeed

    def explode(self):
        pass

    def integrate(self):
        for i in range(self.num):
            self.bones_speed[i][0] *= self.DRAG
            self.bones_speed[i][1] *= self.DRAG
            self.bones_speed[i][1] += self.GRAVITY
            self.bones_pos[i][0] = self.bones_pos_old[i][0] + self.bones_speed[i][0]
            self.bones_pos[i][1] = self.bones_pos_old[i][1] + self.bones_speed[i][1]

    def pre_collision(self):
        return

    def solve_constraints(self):
        for seg in self.segs:
            dx = self.bones_pos_old[seg[0]][0] - self.bones_pos_old[seg[1]][0]
            dy = self.bones_pos_old[seg[0]][1] - self.bones_pos_old[seg[1]][1]
            math.sqrt(dx**2 + dy**2)

    def collide_vs_objects(self):
        pass

    def collide_vs_tiles(self):
        pass

    def post_collision(self):
        pass
