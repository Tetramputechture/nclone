"""Floor Guard Entity (Type 16).

Reconstructed from `npp.dll` -- see `docs/ENTITY_INTEGRATION_SPEC.md` §8.1
(`move` = 0x1040B900, `think` = 0x1040CDD0, ctor = 0x1040B730).

A floor guard sits dormant on a platform until a ninja crosses its foot-level
line of sight, then charges in that direction until a wall or a ledge stops it.

Three behaviours are easy to get wrong and all three are load-bearing:

1. **It latches.** `think` early-outs forever once `dir != 0`; a floor guard
   never re-acquires a target. Only `move` resets `dir` to 0, at which point
   `think` can fire again. (Spec: "[V] It latches.")
2. **Line of sight is the half-cell grid-edge test, not a raycast.** The binary
   walks cells horizontally and consults the two "blocked" grids -- the same
   machinery drones use -- so a floor guard sees through anything that is not a
   grid edge (e.g. a one-way platform).
3. **Wall-ahead and end-of-floor share one branch.** Both stop the guard AND
   snap it to the cell boundary; there is no separate "turn around" path. The
   guard stops, and only re-acquires on a later frame.

FloorGuard derives straight from `BaseEntity` in the game -- it is *not* a
drone (spec §1) and shares no machinery with the drone family despite the
superficially similar grid stepping.
"""

import math
import os

from ..entities import Entity
from ..physics import (
    clamp_half_cell,
    is_empty_column,
    is_empty_row,
    overlap_circle_vs_circle,
)
from ..ninja import NINJA_RADIUS


class EntityFloorGuard(Entity):
    """Type 16. Dormant until it sees a ninja at foot level, then charges."""

    # ctor 0x1040B730 writes these; see spec §8.1 "Constants are now bound".
    #   +0x20 = 24/7  travel distance per move  (the n/7 speed family, §6.4)
    #   +0x28 = 6.0   step base AND line-of-sight band offset
    #   +0x38 = 0.0   second step component, so step == 6.0 at spawn
    SPEED = 24 / 7
    OFFSET = 6.0
    #: Height of the foot-level detection band, `0x1056C360` (= the cell size).
    BAND_HEIGHT = 24.0
    #: Added to `step` on the frame the guard stops, `0x10571CB8`.
    STOP_NUDGE = 0.01
    #: Loop bound on the horizontal cell walk, `0x10572340`.
    MAX_CELL_WALK = 100
    #: The guard walks the **half-cell** grid, not the 24px tile grid. `move` and
    #: `think` both divide by `*(double *)(param_1 + 0x3203F8)`, a field of the
    #: grid object at `sim+0x3203E8`, and that object is constructed
    #: `FUN_1050CDB0(sim + 0x3203E8, 44*2, 25*2, 24.0 * 0.5)` -- 88 x 50 cells at
    #: **12.0** (`TILE_SEGMENT_TABLES.md` §7). The port used 24 and so probed one
    #: edge per tile instead of one per half-tile, which is why it needed "both
    #: halves solid" to approximate a test the binary makes on a single value.
    CELL = 12.0
    #: `NCLONE_AB_LEGACY_EDGES=1` reverts §10 whole: the 0/1 table, the old
    #: combine rule, the 24 px walk and the `!= 0` floor test. Partial reverts do
    #: not measure anything -- the four are one change.
    _LEGACY = os.environ.get("NCLONE_AB_LEGACY_EDGES") == "1"
    if _LEGACY:
        CELL = 24.0

    #: Radius passed to the circle-vs-ninja overlap test. The decompiled
    #: `logical_collision` (0x1040B790) hands `+0x28` to the shared circle helper,
    #: i.e. 6.0 -- NOT the 12.0 that spec §8.0.1 summarises for this family. The
    #: literal reading is used here because the already-validated zap drone
    #: settles the same conflict the same way: §8.0.1 claims 12.0 for the drone
    #: family too, yet the shipped Rust/Python drones use their own 7.5 radius and
    #: reproduce demos correctly. So 12.0 is not the ninja-contact radius.
    RADIUS = 6.0

    MAX_COUNT_PER_LEVEL = 512

    def __init__(self, entity_type, sim, xcoord, ycoord):
        super().__init__(entity_type, sim, xcoord, ycoord)
        self.log_positions = self.sim.sim_config.full_export
        self.is_movable = True
        self.is_thinkable = True
        self.is_logical_collidable = True
        #: -1 / 0 / +1. Zero means dormant; this single field is the whole state
        #: machine (`+0x30`).
        self.dir = 0
        #: `+0x38`, the second step component. Zero at spawn and never written by
        #: `move`/`think`, but kept as a field because `step` is defined as
        #: `+0x38 + +0x28` and a future reading may find a writer.
        self.step2 = 0.0
        #: `+0x40`, a one-shot consumed by `move` to fire the activation effect.
        self.just_activated = False
        self._initial_xpos = self.xpos
        self._initial_ypos = self.ypos
        self._initial_cell = self.cell

    # -- grid queries ------------------------------------------------------
    #
    # The binary indexes three HALF-cell arrays, all 88 x 50 (spec §5):
    #   sim+0x320400 / sim+0x328D80  vertical edges, static / doors; nonzero blocks
    #   sim+0x3248C0                 horizontal edges, static; must be exactly 2
    # The wall test reads half-cell `c_from` travelling right and `c_from - 1`
    # travelling left, which is only symmetric if the array holds "the RIGHT
    # vertical edge of this half-cell is solid" -- so the pair is our
    # `ver_grid_edge_dic` / `door_ver_grid_edge_dic`, and the floor array is
    # `hor_grid_edge_dic`. All three are ternary (§10 of the handoff): the wall
    # test only asks `!= 0`, but the floor test asks `== 2`, and that is the one
    # place the 1-vs-2 distinction decides anything.

    def _cell_right_edge_solid(self, cx, cy):
        """True if the vertical boundary on the right of HALF-cell `(cx, cy)` is solid.

        Static tile edges or dynamic door edges, matching the binary's pair of
        probes into `grid_320400` and `grid_328D80`.
        """
        if self._LEGACY:
            return not is_empty_column(self.sim, 2 * cx + 1, 2 * cy, 2 * cy + 1, 1)
        k = clamp_half_cell(cx + 1, cy)
        return bool(self.sim.ver_grid_edge_dic[k] or self.sim.door_ver_grid_edge_dic[k])

    def _cell_has_floor(self, cx, cy):
        """True if cell `(cx, cy)` has a FULL FLAT floor to walk onto.

        `FUN_1040B900` tests `grid_3248C0[cell(c_to, cy)] != 2` and stops. The
        edge grid is ternary -- `1` is a partial face, a slope or a curve, and
        only `2` is a full flat one -- so this is "walk onto flat ground only",
        not "is anything solid down there".

        That distinction is the whole bug. On `10136880` the flat floor at
        `y = 72` ends at `x = 840` and a quarter-pipe takes over; the game's guard
        stops dead against that boundary and pins itself there, while the port --
        which could only ask `!= 0` off a 0/1 table -- walked straight onto the
        curve and ran the rest of the map down. See the handoff §10.

        Doors are deliberately not consulted: the binary's floor test reads only
        the static array `grid_3248C0`, unlike its wall test which reads both.
        """
        if self._LEGACY:
            return not is_empty_row(self.sim, 2 * cx, 2 * cx + 1, 2 * cy + 1, 1)
        return self.sim.hor_grid_edge_dic[clamp_half_cell(cx, cy + 1)] == 2

    def _row_clear(self, cx_from, cx_to, cy):
        """True if every cell boundary strictly between the two cells is open."""
        lo, hi = (cx_from, cx_to) if cx_to > cx_from else (cx_to, cx_from)
        if hi - lo > self.MAX_CELL_WALK:
            return False
        for cx in range(lo, hi):
            if self._cell_right_edge_solid(cx, cy):
                return False
        return True

    # -- lifecycle ---------------------------------------------------------

    def think(self):
        """Acquire a target. Runs only while dormant, and latches on success."""
        if self.dir != 0:
            return
        ninja = self.sim.ninja
        if not ninja.is_valid_target():
            return

        # Foot-level band: the ninja must sit within 24px above `y + 6`.
        band = self.ypos + self.OFFSET
        if not (band - self.BAND_HEIGHT <= ninja.ypos <= band):
            return

        cy = math.floor(self.ypos / self.CELL)
        if not self._row_clear(
            math.floor(self.xpos / self.CELL), math.floor(ninja.xpos / self.CELL), cy
        ):
            return

        # The game loops over 8 co-op ninjas and takes the NEAREST (§6.5). With a
        # single ninja the loop degenerates to this, but the tie-break rule is
        # kept verbatim: facing is decided by `self.x <= px`, so a ninja exactly
        # on top of the guard sends it right.
        self.dir = 1 if self.xpos <= ninja.xpos else -1
        self.just_activated = True

    def move(self):
        """Charge one step. Runs only while active; stops at a wall or a ledge."""
        if self.dir == 0:
            return
        direction = self.dir
        step = self.step2 + self.OFFSET
        target = self.xpos + direction * self.SPEED

        c_from = math.floor((self.xpos + direction * step) / self.CELL)
        c_to = math.floor((target + direction * step) / self.CELL)
        new_x = target
        if c_from != c_to:
            cy = math.floor(self.ypos / self.CELL)
            lead = c_from if direction > 0 else c_from - 1
            if self._cell_right_edge_solid(lead, cy) or not self._cell_has_floor(
                c_to, cy
            ):
                step += self.STOP_NUDGE
                self.dir = 0
                # Snap so the guard's leading edge rests just inside the boundary
                # it failed to cross. `direction` is the PRE-stop value: using the
                # freshly-zeroed `self.dir` here would collapse both cases to the
                # cell's left edge.
                new_x = int((c_from + max(direction, 0)) * self.CELL) - step * direction

        self.xpos = new_x
        self.grid_move()
        if self.dir != 0 and self.just_activated:
            # One-shot activation FX in the game; nothing to do in the sim beyond
            # consuming the flag so it fires once.
            self.just_activated = False

    def logical_collision(self):
        """Lethal on contact."""
        ninja = self.sim.ninja
        if not ninja.is_valid_target():
            return
        if overlap_circle_vs_circle(
            self.xpos, self.ypos, self.RADIUS, ninja.xpos, ninja.ypos, NINJA_RADIUS
        ):
            ninja.kill(0, 0, 0, 0, 0, cause="floor_guard")

    def reset_state(self):
        """Reset for fast environment reset (same level, no reallocation)."""
        self.xpos = self._initial_xpos
        self.ypos = self._initial_ypos
        self.cell = self._initial_cell
        self.dir = 0
        self.step2 = 0.0
        self.just_activated = False
        self.active = True
