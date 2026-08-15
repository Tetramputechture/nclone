"""Chaser Drone Entity (Type 15).

Ported from `FUN_103FF1B0` (`t15_Chaser_think.c`), `FUN_103FEC60`
(`full/entity_103FEC60.c`) and the ctor `FUN_103FEC00` (`t15_Chaser_ctor.c`).
The chaser derives from `EntityDroneZap`, **not** from `EntityDroneBase` (spec
§3), and its constructor adds no geometry of its own -- so it is a zap drone plus
a `think` and a `chooseNextDirectionAndGoal`.

### The two halves, and why the old reading of them was wrong

Everything the chase *does* used to be invented, because §8.4 marked it [U]. It
is not: the ctor names the fields and `FUN_103FEC60` is the whole behaviour.

```c
// FUN_103FEC00, the ctor
speed = this[0x20];
this[0x78] = speed;          // the patrol speed, saved
this[0x80] = speed + speed;  // the CHASE speed
this[0x70] = 0;              // the latch
this[0x88] = -1;             // the committed direction
```

`+0x20` is the **speed**. This port read `+0x80` as a goal position and built a
"chase toward the cell the ninja was seen in, stop on arrival" model around it;
`this[0x20] = this[0x80]` is `speed = 2 * speed` and nothing more.

So a chase is not bounded by a goal. `FUN_103FEC60` runs at every cell centre:

1. **while latched** -- force the heading onto the committed direction and take
   another cell if the step is legal. The chase runs **straight until a wall**.
   When one blocks it: turn by `-DIR_LIST[mode][0]`, restore `+0x78`, clear the
   latch.
2. then re-run the **same four-gate acquisition** as `think`, from the drone's
   **position**, starting at the committed direction if one is still set.
   On success it latches, doubles the speed and turns immediately.
3. otherwise fall through to the ordinary patrol search.

`think` is the same acquisition again, every frame, from `+0x30`/`+0x38`
(xtarget/ytarget) instead of the position, and it does **not** turn the drone --
it only sets the speed, the commitment and the latch. The turn waits for the next
cell centre.

### Detection, unchanged from the earlier reading and still right

* it is a straight **corridor, not a cone**: a +-12 px perpendicular offset from
  one of three axis-aligned rays, so a ninja 200 px away down a clear corridor is
  seen while one 30 px away diagonally is not;
* **only three of four directions are tested** -- `dir-1`, `dir`, `dir+1`, in
  that order. A chaser is blind directly behind itself.

`FUN_103FE0D0` is a **pure** test at both call sites in `think` (it writes its
goal to a caller-supplied stack buffer), and a real one inside `FUN_103FEC60`,
where the buffer it is handed is `this + 0x30` -- the drone's own target. That is
why the game calls it twice on the success path, and why calling ours
destructively during acquisition was worth 23 points when it was found (§64).
"""

import math
import os

from ..physics import is_empty_column, is_empty_row
from .entity_drone_zap import EntityDroneZap

#: Restore the invented goal-chase model. See the module docstring.
_LEGACY = os.environ.get("NCLONE_AB_CHASER_LEGACY") == "1"

_CELL = 12


class EntityDroneChaser(EntityDroneZap):
    """Type 15. A zap drone that commits to a chase when it sees down a corridor."""

    KILL_CAUSE = "drone_chaser"
    #: Only the CHASER's think reads +0x30/+0x38 (xtarget/ytarget) as the
    #: acquisition origin; the shooter drones use +0x10/+0x18 for their firing
    #: test, and `ninja_in_corridor` is shared between them.
    USES_TARGET_ORIGIN = True
    #: The game allows a chaser within 13.526 px of a LIVING ninja
    #: (`killradius.py`), so the kill radius is at most 3.5 -- not the
    #: 7.5 movement radius inherited from the zap drone.
    KILL_RADIUS = (
        float(os.environ["NCLONE_AB_CHASER_KILL_RADIUS"])
        if os.environ.get("NCLONE_AB_CHASER_KILL_RADIUS") else None
    )

    MAX_COUNT_PER_LEVEL = 256

    def __init__(self, entity_type, sim, xcoord, ycoord, orientation, mode):
        super().__init__(entity_type, sim, xcoord, ycoord, orientation, mode)
        self.is_thinkable = True
        #: `+0x78`, saved by the ctor, and `+0x80`, which is twice it.
        self.speed_slow = self.speed
        self.speed_chase = 2 * self.speed
        #: `+0x70`. While set, `think` does not run its detection at all.
        self.chasing = False
        #: `+0x88`. The direction committed to on acquisition; the game's -1.
        self.chase_dir = None
        #: Kept only so the legacy model still has something to aim at.
        self.goal = None

    # -- the shared four-gate acquisition -----------------------------------

    def _acquire_from(self, ox, oy, start, ninja):
        """`dir-1`, `dir`, `dir+1` through all four gates. The direction, or None.

        `ox`/`oy` is the origin the caller measures from: `think` uses
        `+0x30`/`+0x38` and `FUN_103FEC60` uses `+0x10`/`+0x18`. They coincide at
        a cell centre, which is the only place the second one runs, so the split
        only shows up in `think`.
        """
        for k in range(-1, 2):
            direction = (start + 4 + k) % 4
            xdir, ydir = self.DIR_TO_VEC[direction]
            dx = ninja.xpos - ox
            dy = ninja.ypos - oy
            # 1. forward
            if not (0.0 < dy * ydir + dx * xdir):
                continue
            # 2. inside the corridor
            if not (abs(dy * xdir - dx * ydir) <= self.CORRIDOR_HALF_WIDTH):
                continue
            # 3. the ninja's cell lies between the drone's and the corridor's end
            extent = self._extent_from(direction, ox, oy)
            if ydir:
                here, there = math.floor(oy / _CELL), math.floor(ninja.ypos / _CELL)
            else:
                here, there = math.floor(ox / _CELL), math.floor(ninja.xpos / _CELL)
            if not (min(extent, here) <= there <= max(extent, here)):
                continue
            # 4. the step is legal. NON-DESTRUCTIVE here -- see the docstring.
            sx, sy = self.xtarget, self.ytarget
            ok = bool(self.test_next_direction_and_goal(direction))
            self.xtarget, self.ytarget = sx, sy
            if ok:
                return direction
        return None

    def _extent_from(self, direction, ox, oy):
        """`FUN_103FF070` / `FUN_103FF100`, walked from (ox, oy)."""
        xdir, ydir = self.DIR_TO_VEC[direction]
        sign = 1 if direction < 2 else -1
        if not ydir:
            span = math.floor(oy / _CELL)
            start = cur = math.floor(ox / _CELL)
            for _ in range(self.MAX_CELL_WALK):
                if not is_empty_column(self.sim, cur, span, span, sign):
                    return cur
                cur += sign
            return start
        span = math.floor(ox / _CELL)
        start = cur = math.floor(oy / _CELL)
        for _ in range(self.MAX_CELL_WALK):
            if not is_empty_row(self.sim, span, span, cur, sign):
                return cur
            cur += sign
        return start

    def _latch(self, direction, turn_now):
        """The five writes every acquisition site shares."""
        self.speed = self.speed_chase
        self.chase_dir = direction
        self.chasing = True
        if turn_now:
            # `FUN_103FEC60` writes `+0x40` directly; `think` does not touch it.
            self.dir = direction

    # -- FUN_103FF1B0 -------------------------------------------------------

    def think(self):
        """Detect from `+0x30`/`+0x38`, every frame, while not already chasing."""
        if _LEGACY:
            self._think_legacy()
            return
        if self.chasing:
            return
        ninja = self.sim.ninja
        if ninja is None or not ninja.is_valid_target():
            return
        direction = self._acquire_from(self.xtarget, self.ytarget, self.dir, ninja)
        if direction is not None:
            self._latch(direction, turn_now=False)

    # -- FUN_103FEC60 -------------------------------------------------------

    def choose_next_direction_and_goal(self):
        """Continue the chase, or start one, or fall back to the patrol search."""
        if _LEGACY:
            return self._choose_legacy()
        if self.chasing:
            if self.chase_dir is not None and 0 <= self.chase_dir <= 3:
                self.dir = self.chase_dir
            if self.test_next_direction_and_goal(self.dir):
                return True
            # Blocked. The chase ends AT THE WALL, which is what `chaseend.py`
            # measured: runs end at walls, not at the ninja's old cell.
            self.dir = (self.dir - self.DIR_LIST[self.mode][0] + 4) % 4
            self.speed = self.speed_slow
            self.chasing = False

        # The committed direction outlives the latch by exactly this long: it
        # seeds the next search and is then cleared.
        start = self.dir
        if self.chase_dir is not None:
            start = self.chase_dir
            self.chase_dir = None

        ninja = self.sim.ninja
        if ninja is not None and ninja.is_valid_target():
            direction = self._acquire_from(self.xpos, self.ypos, start, ninja)
            if direction is not None:
                self._latch(direction, turn_now=True)
                # The second `FUN_103FE0D0` call, the one that is not a test:
                # here the buffer it fills IS `+0x30`/`+0x38`.
                self.test_next_direction_and_goal(direction)
                return True
        return super().choose_next_direction_and_goal()

    def reset_state(self):
        """Reset for fast environment reset (same level, no reallocation)."""
        self.chasing = False
        self.chase_dir = None
        self.goal = None
        self.speed = self.speed_slow
        super().reset_state()

    # -- the superseded model, kept behind NCLONE_AB_CHASER_LEGACY ----------

    def _think_legacy(self):
        if self.chasing:
            return
        ninja = self.sim.ninja
        if not ninja.is_valid_target():
            return
        for k in range(-1, 2):
            direction = (k + 4 + self.dir) % 4
            if not self.ninja_in_corridor(direction, ninja):
                continue
            self.chasing = True
            self.chase_dir = direction
            self.goal = (
                (math.floor(ninja.xpos / self.GRID_WIDTH) + 0.5) * self.GRID_WIDTH,
                (math.floor(ninja.ypos / self.GRID_WIDTH) + 0.5) * self.GRID_WIDTH,
            )
            self.speed = self.speed_chase
            return

    def _choose_legacy(self):
        if self.chasing and self.goal is not None:
            gx, gy = self.goal
            dx, dy = gx - self.xpos, gy - self.ypos
            candidates = []
            if abs(dx) >= self.GRID_WIDTH / 2:
                candidates.append(0 if dx > 0 else 2)
            if abs(dy) >= self.GRID_WIDTH / 2:
                candidates.append(1 if dy > 0 else 3)
            if abs(dy) > abs(dx):
                candidates.reverse()
            for direction in candidates:
                if self.test_next_direction_and_goal(direction):
                    self.turn(direction)
                    return True
            self.chasing = False
            self.chase_dir = None
            self.goal = None
            self.speed = self.speed_slow
        return super().choose_next_direction_and_goal()
