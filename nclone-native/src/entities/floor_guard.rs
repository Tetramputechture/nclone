//! Floor Guard (Type 16).
//! Port of nclone/entity_classes/entity_floor_guard.py; reconstructed from
//! `npp.dll` per `docs/ENTITY_INTEGRATION_SPEC.md` §8.1.
//!
//! Dormant on a platform until a ninja crosses its foot-level line of sight, then
//! it charges in that direction until a wall or a ledge stops it.
//!
//! **FloorGuard is not a drone.** It derives straight from `BaseEntity` and shares
//! no machinery with the drone family (spec §1), despite the superficially similar
//! grid stepping -- so it lives in its own module rather than as a `DroneKind`.
//!
//! Three behaviours are load-bearing:
//!   1. **It latches.** `think` early-outs forever once `dir != 0`; only `move`
//!      resets it, at which point `think` can fire again.
//!   2. **Line of sight is the half-cell grid-edge test, not a raycast**, so a
//!      floor guard sees through anything that is not a grid edge.
//!   3. **Wall-ahead and end-of-floor share one branch.** Both stop the guard AND
//!      snap it to the cell boundary; there is no separate turn-around path.
//!
//! `+0x34` (a float lerped toward `dir` at a frame-delta rate) is render state and
//! is deliberately NOT a field here -- porting it would add frame-rate dependence
//! to the simulation (spec §6.2).

use crate::collision::intersection::overlap_circle_vs_circle;
use crate::constants::{
    FLOOR_GUARD_BAND_HEIGHT, FLOOR_GUARD_CELL, FLOOR_GUARD_MAX_CELL_WALK, FLOOR_GUARD_OFFSET,
    FLOOR_GUARD_RADIUS,
    FLOOR_GUARD_SPEED, FLOOR_GUARD_STOP_NUDGE, NINJA_RADIUS,
};
use crate::entities::EntityCtx;
use crate::grid_edges::GridEdges;
use crate::ninja::Ninja;
use crate::types::{clamp_cell, Cell};

/// `NCLONE_AB_LEGACY_EDGES=1` reverts the whole of handoff §10: the 0/1
/// table, the old combine rule, the 24 px walk and the `!= 0` floor test.
/// Partial reverts measure nothing -- the four are one change.
#[inline]
fn cell_size() -> f64 {
    if crate::ab::ab().legacy_edges {
        24.0
    } else {
        FLOOR_GUARD_CELL
    }
}

#[derive(Debug, Clone)]
pub struct FloorGuard {
    pub xpos: f64,
    pub ypos: f64,
    /// `+0x30`. -1 / 0 / +1; zero means dormant. This single field is the whole
    /// state machine.
    pub dir: i32,
    /// `+0x38`, the second step component. Zero at spawn and never written by
    /// `move`/`think`, but kept because `step` is defined as `+0x38 + +0x28`.
    pub step2: f64,
    /// `+0x40`, a one-shot consumed by `move` to fire the activation effect.
    pub just_activated: bool,
    pub cell: Cell,
    initial_xpos: f64,
    initial_ypos: f64,
}

impl FloorGuard {
    pub fn new(xcoord: i32, ycoord: i32) -> Self {
        let xpos = (xcoord * 6) as f64;
        let ypos = (ycoord * 6) as f64;
        FloorGuard {
            xpos,
            ypos,
            dir: 0,
            step2: 0.0,
            just_activated: false,
            cell: clamp_cell((xpos / 24.0).floor() as i32, (ypos / 24.0).floor() as i32),
            initial_xpos: xpos,
            initial_ypos: ypos,
        }
    }

    // -- grid queries ------------------------------------------------------
    //
    // The binary indexes three per-CELL arrays (spec §5): two "nonzero == blocked"
    // and one "floor kind, must == 2 to be walkable". Their semantics were tagged
    // [I]. The mapping used here falls out of how `move` indexes them: the wall
    // test reads cell `c_from` travelling right and `c_from - 1` travelling left,
    // which is only symmetric if the array holds "the RIGHT vertical edge of this
    // cell is solid". The two arrays are then the static tile edges and the dynamic
    // door edges -- exactly the pair `is_empty_column` already consults. "Floor
    // kind == 2" becomes "the bottom edge of this cell is solid".

    /// Is the vertical boundary on the right of HALF-cell `(cx, cy)` solid?
    ///
    /// Static tile edges or dynamic door edges, matching the binary's pair of
    /// probes into `grid_320400` and `grid_328D80`.
    fn cell_right_edge_solid(cx: i32, cy: i32, edges: &GridEdges) -> bool {
        if crate::ab::ab().legacy_edges {
            return !edges.is_empty_column(2 * cx + 1, 2 * cy, 2 * cy + 1, 1);
        }
        !edges.is_empty_column(cx, cy, cy, 1)
    }

    /// True if half-cell `(cx, cy)` has a FULL FLAT floor to walk onto.
    ///
    /// `FUN_1040B900` tests `grid_3248C0[cell(c_to, cy)] != 2` and stops. The
    /// edge grid is ternary -- `1` is a partial face, a slope or a curve, and
    /// only `2` is a full flat one -- so this is "walk onto flat ground only",
    /// not "is anything solid down there". See the handoff §10; on `10136880`
    /// the port used to walk onto a quarter-pipe the game stops dead against.
    ///
    /// Doors are deliberately not consulted: the floor test reads only the static
    /// array, unlike the wall test above which reads both.
    fn cell_has_floor(cx: i32, cy: i32, edges: &GridEdges) -> bool {
        if crate::ab::ab().legacy_edges {
            return !edges.is_empty_row(2 * cx, 2 * cx + 1, 2 * cy + 1, 1);
        }
        edges.hor_at(cx, cy + 1) == 2
    }

    /// Is every cell boundary strictly between the two cells open?
    fn row_clear(cx_from: i32, cx_to: i32, cy: i32, edges: &GridEdges) -> bool {
        let (lo, hi) = if cx_to > cx_from {
            (cx_from, cx_to)
        } else {
            (cx_to, cx_from)
        };
        if hi - lo > FLOOR_GUARD_MAX_CELL_WALK {
            return false;
        }
        for cx in lo..hi {
            if Self::cell_right_edge_solid(cx, cy, edges) {
                return false;
            }
        }
        true
    }

    // -- lifecycle ---------------------------------------------------------

    /// Acquire a target. Runs only while dormant, and latches on success.
    pub fn think(&mut self, ninja: &Ninja, edges: &GridEdges) {
        if self.dir != 0 || !ninja.is_valid_target() {
            return;
        }
        // Foot-level band: the ninja must sit within 24px above `y + 6`.
        let band = self.ypos + FLOOR_GUARD_OFFSET;
        if !(band - FLOOR_GUARD_BAND_HEIGHT <= ninja.ypos && ninja.ypos <= band) {
            return;
        }
        let cy = (self.ypos / cell_size()).floor() as i32;
        let cx_from = (self.xpos / cell_size()).floor() as i32;
        let cx_to = (ninja.xpos / cell_size()).floor() as i32;
        let clear = Self::row_clear(cx_from, cx_to, cy, edges);
        if std::env::var("NCLONE_DEBUG_FG").as_deref() == Ok("1") {
            eprintln!(
                "[fg] guard=({:.6},{:.1}) ninja=({:.6},{:.6}) band={:.1} cy={} \
                 cx_from={} cx_to={} row_clear={}",
                self.xpos, self.ypos, ninja.xpos, ninja.ypos, band, cy, cx_from, cx_to, clear
            );
        }
        if !clear {
            return;
        }
        // The game loops 8 co-op ninjas and takes the NEAREST (spec §6.5); with a
        // single ninja that degenerates to this. The tie-break is kept verbatim:
        // `self.x <= px` sends a guard directly under a ninja to the right.
        self.dir = if self.xpos <= ninja.xpos { 1 } else { -1 };
        self.just_activated = true;
    }

    /// Charge one step. Runs only while active; stops at a wall or a ledge.
    pub fn move_entity(&mut self, ctx: &mut EntityCtx) {
        if self.dir == 0 {
            return;
        }
        let direction = self.dir;
        let mut step = self.step2 + FLOOR_GUARD_OFFSET;
        let target = self.xpos + direction as f64 * FLOOR_GUARD_SPEED;

        let c_from = ((self.xpos + direction as f64 * step) / cell_size()).floor() as i32;
        let c_to = ((target + direction as f64 * step) / cell_size()).floor() as i32;
        let mut new_x = target;
        if c_from != c_to {
            let cy = (self.ypos / cell_size()).floor() as i32;
            let lead = if direction > 0 { c_from } else { c_from - 1 };
            if Self::cell_right_edge_solid(lead, cy, ctx.grid_edges)
                || !Self::cell_has_floor(c_to, cy, ctx.grid_edges)
            {
                step += FLOOR_GUARD_STOP_NUDGE;
                self.dir = 0;
                // Snap so the leading edge rests just inside the boundary it
                // failed to cross. `direction` is the PRE-stop value: using the
                // freshly-zeroed `self.dir` would collapse both cases onto the
                // cell's left edge.
                new_x = ((c_from + direction.max(0)) as f64 * cell_size()) as i32 as f64
                    - step * direction as f64;
            }
        }

        self.xpos = new_x;
        let new_cell = clamp_cell(
            (self.xpos / 24.0).floor() as i32,
            (self.ypos / 24.0).floor() as i32,
        );
        let old_cell = self.cell;
        self.cell = new_cell;
        ctx.grid_move(old_cell, new_cell);

        if self.dir != 0 && self.just_activated {
            // One-shot activation FX in the game; nothing to do here beyond
            // consuming the flag so it fires once.
            self.just_activated = false;
        }
    }

    /// Lethal on contact.
    pub fn logical_collision(&self, ninja: &mut Ninja) -> bool {
        if !ninja.is_valid_target() {
            return false;
        }
        if overlap_circle_vs_circle(
            self.xpos,
            self.ypos,
            FLOOR_GUARD_RADIUS,
            ninja.xpos,
            ninja.ypos,
            NINJA_RADIUS,
        ) {
            ninja.kill(0, 0.0, 0.0, 0.0, 0.0, Some("floor_guard"));
            return true;
        }
        false
    }

    pub fn reset_state(&mut self) {
        self.xpos = self.initial_xpos;
        self.ypos = self.initial_ypos;
        self.dir = 0;
        self.step2 = 0.0;
        self.just_activated = false;
        self.cell = clamp_cell(
            (self.xpos / 24.0).floor() as i32,
            (self.ypos / 24.0).floor() as i32,
        );
    }
}
