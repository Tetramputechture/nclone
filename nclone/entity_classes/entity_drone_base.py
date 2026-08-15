import math
import os

from ..entities import Entity
from ..physics import is_empty_column, is_empty_row
from ..constants.physics_constants import DRONE_RADIUS, DRONE_GRID_SIZE


_CELL = float(os.environ.get("NCLONE_AB_EXTENT_CELL", 12))
_EXTENT_GATE = os.environ.get("NCLONE_AB_CHASER_EXTENT") == "1"
_TARGET_AIM = os.environ.get("NCLONE_AB_CHASER_TARGET_AIM") == "1"
_STALE_TARGET = os.environ.get("NCLONE_AB_CHASER_STALE_TARGET") == "1"


class EntityDroneBase(Entity):
    """Drone Base Entity - Parent Class for All Drone Types

    Abstract base class that provides core drone functionality for regular drones, chasers,
    and mini drones. Drones are grid-based patrolling enemies that follow deterministic
    movement patterns and are lethal to touch.

    Physical Properties:
        - Default Radius: 7.5 pixels (overridable by subclasses)
        - Grid Movement: 24-pixel cell size (standard)
        - Movement Modes: 4 distinct patrol patterns
            * 0: Follow wall clockwise
            * 1: Follow wall counter-clockwise
            * 2: Wander clockwise
            * 3: Wander counter-clockwise
        - Direction System:
            * 0: Right (1,0)
            * 1: Down (0,1)
            * 2: Left (-1,0)
            * 3: Up (0,-1)

    Core Behavior:
        - Grid-Based Movement:
            * Moves between cell centers
            * Checks for wall collisions
            * Maintains target position
        - Patrolling Logic:
            * Tests available directions in mode-specific order
            * Follows walls or wanders based on mode
            * Handles orientation and direction changes
        - Movement Precision:
            * Interpolates between cell centers
            * Handles partial movements
            * Maintains consistent speed

    Technical Implementation:
        - State Management:
            * Tracks current and target positions
            * Maintains direction and orientation
            * Handles mode-specific behavior
        - Movement System:
            * Grid-aligned movement
            * Collision checking
            * Direction selection
        - Position Logging:
            * Optional position tracking
            * Direction change logging
            * State normalization for AI

    Derived Classes:
        - Zap Drone (Type 14): Standard patrol drone
        - Mini Drone (Type 26): Smaller, faster variant
        - Chaser Drone: Advanced drone with pursuit behavior

    Note: This is an abstract base class and should not be instantiated directly.
    Use one of the derived drone types instead.
    """

    RADIUS = DRONE_RADIUS
    GRID_WIDTH = DRONE_GRID_SIZE
    DIR_TO_VEC = {0: [1, 0], 1: [0, 1], 2: [-1, 0], 3: [0, -1]}
    # Dictionary to choose the next direction from the patrolling mode of the drone.
    # Patrolling modes : {0:follow wall CW, 1:follow wall CCW, 2:wander CW, 3:wander CCW}
    # Directions : {0:keep forward, 1:turn right, 2:go backward, 3:turn left}
    DIR_LIST = {0: [1, 0, 3, 2], 1: [3, 0, 1, 2], 2: [0, 1, 3, 2], 3: [0, 3, 1, 2]}

    def __init__(self, type, sim, xcoord, ycoord, orientation, mode, speed):
        super().__init__(type, sim, xcoord, ycoord)
        self.log_positions = self.sim.sim_config.full_export
        self.is_movable = True
        self.speed = speed
        self.dir = None
        self.turn(orientation // 2)
        self.orientation = orientation
        # Only patrol modes 0-3 are defined (DIR_LIST). A malformed level (e.g. a
        # procgen drone with mode >= 4) would otherwise KeyError in move() and crash
        # the whole sim/training run. Wrap into the valid range: identity for valid
        # 0-3 (no parity change on good levels), deterministic for out-of-spec.
        self.mode = mode if mode in self.DIR_LIST else (mode % len(self.DIR_LIST))
        self.xtarget, self.ytarget = self.xpos, self.ypos
        self.xpos2, self.ypos2 = self.xpos, self.ypos
        self._initial_xpos, self._initial_ypos = self.xpos, self.ypos
        self._initial_cell = self.cell
        self._initial_dir = self.dir

    def turn(self, dir):
        """Change the drone's direction and log it."""
        self.dir_old = self.dir or dir
        self.dir = dir
        if self.sim.sim_config.full_export:
            self.log_collision(dir)

    def move(self):
        """Make the drone move along the grid. The drone will try to move towards the center of an
        adjacent cell. When at the center of that cell, it will then choose the next cell to move
        towards.
        """
        xspeed = self.speed * self.DIR_TO_VEC[self.dir][0]
        yspeed = self.speed * self.DIR_TO_VEC[self.dir][1]
        dx = self.xtarget - self.xpos
        dy = self.ytarget - self.ypos
        dist = math.sqrt(dx**2 + dy**2)
        # If the drone has reached or passed the center of the cell, choose the next cell to go to.
        if (
            dist < 0.000001
            or (
                dx * (self.xtarget - (self.xpos + xspeed))
                + dy * (self.ytarget - (self.ypos + yspeed))
            )
            < 0
        ):
            self.xpos, self.ypos = self.xtarget, self.ytarget
            can_move = self.choose_next_direction_and_goal()
            if can_move:
                # Clamped at zero, as `FUN_103FADA0`
                # (`docs/decomp/HELPER_dronebase_move_103FADA0.c:56`) does. The
                # remainder is normally positive -- the overshoot test only fires
                # once `dist < speed` -- but not when the drone's heading is not
                # the direction of its target, which happens on the frame a chaser
                # relatches and at spawn. Unclamped, the drone then steps
                # *backwards* by `dist - speed`, from which it never recovers.
                disp = max(self.speed - dist, 0.0)
                self.xpos += disp * self.DIR_TO_VEC[self.dir][0]
                self.ypos += disp * self.DIR_TO_VEC[self.dir][1]
        # Otherwise, make the drone keep moving along its current direction.
        else:
            xspeed = self.speed * self.DIR_TO_VEC[self.dir][0]
            yspeed = self.speed * self.DIR_TO_VEC[self.dir][1]
            self.xpos += xspeed
            self.ypos += yspeed
        # Unconditional, on both branches: the binary calls the cell-index helper
        # once after the if/else and stores it (`HELPER_dronebase_move_103FADA0.c`
        # lines 68-73). Doing it only on the straight-line branch leaves the cell
        # stale for a frame every time the drone turns a corner, which changes
        # what `gather_entities_from_neighbourhood` finds.
        self.grid_move()

    def predict_positions(self, sample_at):
        """Where this drone will be at each frame offset in ``sample_at``.

        ``sample_at`` must be sorted ascending; the roll-out is a single pass
        that samples as it goes, so the cost is ``max(sample_at)`` moves, not
        their sum.

        EXACT, not an estimate, for the patrol drones (zap 14, mini 26): their
        motion is a pure function of the tile grid and the frame counter. It is
        NOT valid for the chaser (15) or the shooters (12/13), which read the
        ninja -- rolling those forward would be predicting the agent's own
        future. Callers select by type; this method does not, because the
        subclasses share this body.

        Implemented by driving the real ``move()`` on a saved-and-restored copy
        of the mutable state rather than by reimplementing the patrol rules. A
        second copy of those rules would drift from the simulator silently, and
        a forecast that disagrees with what actually happens is worse than none.
        The two side effects ``move()`` has beyond this entity -- the occupancy
        grid and the collision log -- are shadowed out for the duration; the
        twin in ``nclone-native`` (``Drone::forecast``) splits the same seam.
        """
        saved = (
            self.xpos,
            self.ypos,
            self.xtarget,
            self.ytarget,
            self.dir,
            self.dir_old,
        )
        # Instance attributes shadow the bound methods; `del` restores the class
        # ones. grid_move would mutate sim.grid_entity (and self.cell);
        # log_collision would append phantom entries to the export log.
        self.grid_move = lambda: None
        self.log_collision = lambda state=1: None
        try:
            out = []
            frame = 0
            for target in sample_at:
                while frame < target:
                    self.move()
                    frame += 1
                out.append((float(self.xpos), float(self.ypos)))
        finally:
            del self.grid_move
            del self.log_collision
            (
                self.xpos,
                self.ypos,
                self.xtarget,
                self.ytarget,
                self.dir,
                self.dir_old,
            ) = saved
        return out

    def choose_next_direction_and_goal(self):
        """Return true if the drone can move in at least one of four directions.
        The directions are tested in the order according to the drone's preference depending of its mode.
        """
        for i in range(4):
            new_dir = (self.dir + self.DIR_LIST[self.mode][i]) % 4
            valid_dir = self.test_next_direction_and_goal(new_dir)
            if valid_dir:
                self.turn(new_dir)
                return True
        return False

    def test_next_direction_and_goal(self, dir):
        """Return true if the drone can move to the adjacent cell along the given direction.
        This is true if there are no walls impeding the drone's movement.
        If true, set the center of the adjacent cell as the drone's next target."""
        xdir, ydir = self.DIR_TO_VEC[dir]
        xtarget = self.xpos + self.GRID_WIDTH * xdir
        ytarget = self.ypos + self.GRID_WIDTH * ydir
        if not ydir:
            cell_x = math.floor((self.xpos + xdir * self.RADIUS) / 12)
            cell_xtarget = math.floor((xtarget + xdir * self.RADIUS) / 12)
            cell_y1 = math.floor((self.ypos - self.RADIUS) / 12)
            cell_y2 = math.floor((self.ypos + self.RADIUS) / 12)
            while cell_x != cell_xtarget:
                if not is_empty_column(self.sim, cell_x, cell_y1, cell_y2, xdir):
                    return False
                cell_x += xdir
        else:
            cell_y = math.floor((self.ypos + ydir * self.RADIUS) / 12)
            cell_ytarget = math.floor((ytarget + ydir * self.RADIUS) / 12)
            cell_x1 = math.floor((self.xpos - self.RADIUS) / 12)
            cell_x2 = math.floor((self.xpos + self.RADIUS) / 12)
            while cell_y != cell_ytarget:
                if not is_empty_row(self.sim, cell_x1, cell_x2, cell_y, ydir):
                    return False
                cell_y += ydir
        self.xtarget, self.ytarget = xtarget, ytarget
        return True

    #: Half-width of the corridor used by every drone that needs to know whether
    #: something is lined up along one of its four axes -- the chaser's detection
    #: (`0x1056C358`, spec §8.4) and, by the same machinery, the shooter drones'
    #: firing lane.
    #: Half-width of the detection corridor. 12.0 is a HALF cell; 24.0 is a
    #: full one. On  the game latches a chase at frame 106 with the
    #: ninja 21.8 px off the drone's axis, which a 12.0 corridor rejects and a
    #: 24.0 one accepts -- this port does not latch until frame 111, when the
    #: offset has fallen to 10.1. .
    #: Perpendicular tolerance of the sighting corridor. 
    #: overrides it so the value can be swept: 1001030 drone 3 brackets it in
    #: (18.0, 19.1429] -- the game latches at dy=18.0 and not at dy=19.142857.
    CORRIDOR_HALF_WIDTH = float(
        os.environ.get(
            "NCLONE_AB_CORRIDOR_W",
            24.0 if os.environ.get("NCLONE_AB_CORRIDOR_24") == "1" else 12.0,
        )
    )
    #: Loop bound on a corridor cell walk, mirroring the floor guard's `100.0`.
    MAX_CELL_WALK = 100

    def ninja_in_corridor(self, direction, ninja):
        """Is the ninja lined up along `direction`, within the corridor, in sight?

        This is the drone family's one notion of "I can see something that way",
        and it is deliberately shared rather than reimplemented per subclass:

        * the chaser (15) runs it over three directions to decide whether to latch
          a chase (spec §8.4, [V] -- a **corridor, not a cone**, and never
          backward);
        * the shooter drones (12/13) run it over their single facing direction to
          decide whether to fire, because a drone's gun points along the axis it
          faces. Free-aiming a hitscan at the ninja instead makes those drones
          unmissable -- measured against the attract corpus, it killed the ninja in
          43 of 43 demos that the real game's drones let through.
        """
        xdir, ydir = self.DIR_TO_VEC[direction]
        # `t15_Chaser_think.c` reads +0x30/+0x38 here, which the captures show
        # are xtarget/ytarget -- NOT +0x10/+0x18 (xpos/ypos), which is what
        # `FUN_103fe0d0` uses for its step probe. Up to a full cell apart.
        if _TARGET_AIM and getattr(self, "USES_TARGET_ORIGIN", False):
            prev = getattr(self, "_prev_target", None)
            if _STALE_TARGET and prev is not None:
                ox, oy = prev
            else:
                ox, oy = self.xtarget, self.ytarget
        else:
            ox, oy = self.xpos, self.ypos
        dx = ninja.xpos - ox
        dy = ninja.ypos - oy
        if dy * ydir + dx * xdir <= 0.0:
            return False  # behind, or exactly abeam
        if abs(dy * xdir - dx * ydir) > self.CORRIDOR_HALF_WIDTH:
            return False  # outside the corridor
        if _EXTENT_GATE:
            extent = self.corridor_extent(direction)
            if ydir:
                here = math.floor(self.ypos / _CELL)
                there = math.floor(ninja.ypos / _CELL)
            else:
                here = math.floor(self.xpos / _CELL)
                there = math.floor(ninja.xpos / _CELL)
            return min(extent, here) <= there <= max(extent, here)
        return self.corridor_clear(direction, ninja.xpos, ninja.ypos)

    def corridor_extent(self, direction):
        """How far the open corridor runs from this drone along `direction`.

        Literal port of `FUN_103ff070` (horizontal) and `FUN_103ff100`
        (vertical), from `docs/decomp/A10_chaser_extentA.c` / `...extentB.c`:
        step one cell at a time from the drone's OWN cell, testing a single-cell
        span, and return the last cell reached. On overrun the game returns the
        unmoved start, so a runaway walk cannot widen the corridor.
        """
        xdir, ydir = self.DIR_TO_VEC[direction]
        sign = 1 if direction < 2 else -1
        if not ydir:
            span = math.floor(self.ypos / _CELL)
            start = cur = math.floor(self.xpos / _CELL)
            for _ in range(self.MAX_CELL_WALK):
                if not is_empty_column(self.sim, cur, span, span, sign):
                    return cur
                cur += sign
            return start
        span = math.floor(self.xpos / _CELL)
        start = cur = math.floor(self.ypos / _CELL)
        for _ in range(self.MAX_CELL_WALK):
            if not is_empty_row(self.sim, span, span, cur, sign):
                return cur
            cur += sign
        return start

    def corridor_clear(self, direction, target_x, target_y):
        """Is the straight run of cells from here to the target unobstructed?

        Stands in for `FUN_103FE0D0` plus the two axis-dependent extent checks
        (`FUN_103FF070` horizontal, `FUN_103FF100` vertical), all three unread.
        Their stated job is "confirm the ninja's cell lies within the clear run",
        which is `test_next_direction_and_goal`'s grid-edge probe generalised from
        one cell to the whole distance -- so it reuses the same half-cell
        granularity and the same radius-offset footprint rather than introducing a
        second notion of "blocked".
        """
        xdir, ydir = self.DIR_TO_VEC[direction]
        steps = 0
        if not ydir:
            cell_x = math.floor((self.xpos + xdir * self.RADIUS) / 12)
            cell_xtarget = math.floor((target_x + xdir * self.RADIUS) / 12)
            cell_y1 = math.floor((self.ypos - self.RADIUS) / 12)
            cell_y2 = math.floor((self.ypos + self.RADIUS) / 12)
            while cell_x != cell_xtarget:
                if not is_empty_column(self.sim, cell_x, cell_y1, cell_y2, xdir):
                    return False
                cell_x += xdir
                steps += 1
                if steps > self.MAX_CELL_WALK:
                    return False
        else:
            cell_y = math.floor((self.ypos + ydir * self.RADIUS) / 12)
            cell_ytarget = math.floor((target_y + ydir * self.RADIUS) / 12)
            cell_x1 = math.floor((self.xpos - self.RADIUS) / 12)
            cell_x2 = math.floor((self.xpos + self.RADIUS) / 12)
            while cell_y != cell_ytarget:
                if not is_empty_row(self.sim, cell_x1, cell_x2, cell_y, ydir):
                    return False
                cell_y += ydir
                steps += 1
                if steps > self.MAX_CELL_WALK:
                    return False
        return True

    def reset_state(self):
        """Restore the spawn position, target and facing for a fast reset.

        Without this, `Simulator.fast_reset` fell through to its "just set active"
        fallback and every drone resumed the next episode from wherever the last
        one left it -- while `nclone-native`'s `Drone::reset_state` did restore
        spawn state, so the two engines disagreed after the first reset.
        """
        self.xpos, self.ypos = self._initial_xpos, self._initial_ypos
        self.cell = self._initial_cell
        self.xtarget, self.ytarget = self.xpos, self.ypos
        self.xpos2, self.ypos2 = self.xpos, self.ypos
        self.dir = self._initial_dir
        self.dir_old = self._initial_dir
        self.active = True

    def get_state(self):
        state = super().get_state()
        # Normalize mode (0-3)
        # state.append(max(0.0, min(1.0, float(self.mode) / 3)))
        # state[6] = max(0.0, min(1.0, (float(self.dir) + 1) /
        #                2 if self.dir is not None else 0.5))  # Normalize direction
        # Normalize orientation
        # state[7] = max(0.0, min(1.0, float(self.orientation) / 7))
        return state
