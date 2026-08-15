//! Shove Thwump Entity (Type 28).
//! Direct port of nclone/entity_classes/entity_shove_thwump.py.
//!
//! Contact-triggered rather than sight-triggered: shove it hard enough and it
//! launches AWAY along the contact normal at 4 px/frame until it hits a wall,
//! then crawls back at 1. Its outer square is solid, but it hides a lethal
//! 8px core.
//!
//! States: 0 immobile, 1 activated (direction latched, not yet moving),
//! 2 launching, 3 retreating.
//!
//! Unlike the thwump this is NOT in the `move` pass -- Python drives its motion
//! from `think` via `move_if_possible`, and the distinction is observable
//! because the two passes straddle other entities' updates.

use crate::collision::intersection::{overlap_circle_vs_circle, penetration_square_vs_point};
use crate::constants::{NINJA_RADIUS, SHOVE_THWUMP_RADIUS, SHOVE_THWUMP_SEMI_SIDE};
use crate::entities::EntityCtx;
use crate::ninja::Ninja;
use crate::types::{clamp_cell, Cell};

#[derive(Debug, Clone)]
pub struct ShoveThwump {
    pub xpos: f64,
    pub ypos: f64,
    pub xorigin: f64,
    pub yorigin: f64,
    pub xdir: f64,
    pub ydir: f64,
    /// 0 immobile, 1 activated, 2 launching, 3 retreating.
    pub state: u8,
    /// Set by `logical_collision`, consumed by `think`: keeps the shwump in state
    /// 1 for as long as the ninja stays in contact.
    pub activated: bool,
    pub cell: Cell,
}

impl ShoveThwump {
    pub fn new(xcoord: i32, ycoord: i32) -> Self {
        let xpos = (xcoord * 6) as f64;
        let ypos = (ycoord * 6) as f64;
        ShoveThwump {
            xpos,
            ypos,
            xorigin: xpos,
            yorigin: ypos,
            xdir: 0.0,
            ydir: 0.0,
            state: 0,
            activated: false,
            cell: clamp_cell((xpos / 24.0).floor() as i32, (ypos / 24.0).floor() as i32),
        }
    }

    /// Port of `EntityShoveThwump.think`.
    ///
    /// Note the FALL-THROUGH: Python uses `if state == 1: ... set_state(2)`
    /// followed by a separate `if state == 3 / elif state == 2`, so the frame a
    /// shwump leaves state 1 it ALSO takes its first launch step. An `else if`
    /// chain here would silently cost it that frame.
    pub fn think(&mut self, ctx: &mut EntityCtx) {
        if self.state == 1 {
            if self.activated {
                self.activated = false;
                return;
            }
            self.state = 2;
        }
        if self.state == 3 {
            let origin_dist =
                (self.xpos - self.xorigin).abs() + (self.ypos - self.yorigin).abs();
            if origin_dist >= 1.0 {
                self.move_if_possible(self.xdir, self.ydir, 1.0, ctx);
            } else {
                self.xpos = self.xorigin;
                self.ypos = self.yorigin;
                self.state = 0;
            }
        } else if self.state == 2 {
            self.move_if_possible(-self.xdir, -self.ydir, 4.0, ctx);
        }
    }

    /// Step along one axis, flipping to `retreating` if the next cell is walled.
    ///
    /// The axis is chosen by `self.ydir` (the LATCHED contact direction) while the
    /// step uses the passed `xdir`/`ydir`, which are negated for the launch. That
    /// asymmetry is in the original and is why launch and retreat share a path.
    /// `DAT_10572340`. `EntityShoveThwomp::moveIfPossible`
    /// (`docs/decomp/full/entity_104135D0.c`) walks half-cells toward the target
    /// and gives up the moment the walked index passes 100 in absolute value:
    ///
    /// ```text
    /// if (DAT_10572340 < |iVar2|) return 0;
    /// ```
    ///
    /// Nothing else bounds the launch. Every edge lookup outside the 88x51 grid
    /// reads empty -- `clamp_half_cell` caps y at 49 and so never even reaches
    /// the solid border row 50 -- so without this a launched shove thwump leaves
    /// the level and never comes back. It is the only thing that turns one
    /// around, and a shove thwump that does not come back is not there to be
    /// landed on later.
    ///
    /// Confirmed against two captures on two different tile layouts
    /// (`out14/10138350`, `out15/10138974`): four launches, every one of them
    /// blocked walking into half-cell 101, resting at y=1208 after exactly 236
    /// frames at 4 px, then retreating home at 1 px/frame.
    const CELL_WALK_LIMIT: i32 = 100;

    fn move_if_possible(&mut self, xdir: f64, ydir: f64, speed: f64, ctx: &mut EntityCtx) {
        if self.ydir == 0.0 {
            let xpos_new = self.xpos + xdir * speed;
            let cell_x = (self.xpos / 12.0).floor() as i32;
            let cell_x_new = (xpos_new / 12.0).floor() as i32;
            if cell_x != cell_x_new {
                let cell_y1 = ((self.ypos - 8.0) / 12.0).floor() as i32;
                let cell_y2 = ((self.ypos + 8.0) / 12.0).floor() as i32;
                if !ctx
                    .grid_edges
                    .is_empty_column(cell_x, cell_y1, cell_y2, xdir as i32)
                    || cell_x_new.abs() > Self::CELL_WALK_LIMIT
                {
                    self.state = 3;
                    return;
                }
            }
            self.xpos = xpos_new;
        } else {
            let ypos_new = self.ypos + ydir * speed;
            let cell_y = (self.ypos / 12.0).floor() as i32;
            let cell_y_new = (ypos_new / 12.0).floor() as i32;
            if cell_y != cell_y_new {
                let cell_x1 = ((self.xpos - 8.0) / 12.0).floor() as i32;
                let cell_x2 = ((self.xpos + 8.0) / 12.0).floor() as i32;
                if !ctx
                    .grid_edges
                    .is_empty_row(cell_x1, cell_x2, cell_y, ydir as i32)
                    || cell_y_new.abs() > Self::CELL_WALK_LIMIT
                {
                    self.state = 3;
                    return;
                }
            }
            self.ypos = ypos_new;
        }
        let new_cell = clamp_cell(
            (self.xpos / 24.0).floor() as i32,
            (self.ypos / 24.0).floor() as i32,
        );
        let old_cell = self.cell;
        self.cell = new_cell;
        ctx.grid_move(old_cell, new_cell);
    }

    /// Solid while immobile or activated. Once activated, only the face the ninja
    /// shoved stays solid -- the other sides become passable.
    pub fn physical_collision(&self, ninja: &Ninja) -> Option<((f64, f64), (f64, f64))> {
        if self.state > 1 {
            return None;
        }
        let depen = penetration_square_vs_point(
            self.xpos,
            self.ypos,
            ninja.xpos,
            ninja.ypos,
            SHOVE_THWUMP_SEMI_SIDE + NINJA_RADIUS,
        )?;
        let (depen_x, depen_y) = depen.0;
        if self.state == 0 || self.xdir * depen_x + self.ydir * depen_y >= 0.01 {
            Some(depen)
        } else {
            None
        }
    }

    /// Latches the shove direction, or kills the ninja on the inner core.
    ///
    /// The core check is reached ONLY when the outer square did not register --
    /// Python returns early from the depenetration branch, so a ninja against a
    /// solid side is never core-killed on the same frame.
    pub fn logical_collision(&mut self, ninja: &mut Ninja) -> Option<f64> {
        let depen = penetration_square_vs_point(
            self.xpos,
            self.ypos,
            ninja.xpos,
            ninja.ypos,
            SHOVE_THWUMP_SEMI_SIDE + NINJA_RADIUS + 0.1,
        );
        if let Some(((depen_x, depen_y), (_, depen_other))) = depen {
            if self.state <= 1 {
                if self.state == 0 {
                    self.activated = true;
                    // Only a firm shove (>0.2 on the other axis) launches it.
                    if depen_other > 0.2 {
                        self.xdir = depen_x;
                        self.ydir = depen_y;
                        self.state = 1;
                    }
                } else if self.xdir * depen_x + self.ydir * depen_y >= 0.01 {
                    self.activated = true;
                } else {
                    return None;
                }
                return Some(depen_x);
            }
        }
        if overlap_circle_vs_circle(
            ninja.xpos,
            ninja.ypos,
            NINJA_RADIUS,
            self.xpos,
            self.ypos,
            SHOVE_THWUMP_RADIUS,
        ) {
            ninja.kill(0, 0.0, 0.0, 0.0, 0.0, Some("shove_thwump"));
        }
        None
    }

    pub fn reset_state(&mut self) {
        self.xpos = self.xorigin;
        self.ypos = self.yorigin;
        self.xdir = 0.0;
        self.ydir = 0.0;
        self.state = 0;
        self.activated = false;
        self.cell = clamp_cell(
            (self.xpos / 24.0).floor() as i32,
            (self.ypos / 24.0).floor() as i32,
        );
    }
}
