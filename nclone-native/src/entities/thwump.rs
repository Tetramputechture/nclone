//! Thwump Entity (Type 20).
//! Direct port of nclone/entity_classes/entity_thwump.py.
//!
//! Sits still until the ninja enters its line of sight along the facing axis,
//! then charges at 20/7 px/frame until it hits a wall, retreats at 8/7 back to
//! its origin, and rearms. The charging FACE is lethal; the other three sides are
//! safe and can be stood on or wall-slid against.
//!
//! State is -1 (retreating) / 0 (immobile) / 1 (charging) -- signed, and used
//! arithmetically as a direction multiplier, so it is `i8` here rather than the
//! usual `u8`.

use crate::collision::intersection::{overlap_circle_vs_segment, penetration_square_vs_point};
use crate::constants::{
    NINJA_RADIUS, THWUMP_BACKWARD_SPEED, THWUMP_FORWARD_SPEED, THWUMP_SEMI_SIDE,
};
use crate::entities::EntityCtx;
use crate::grid_edges::GridEdges;
use crate::ninja::Ninja;
use crate::types::{clamp_cell, Cell};

#[derive(Debug, Clone)]
pub struct Thwump {
    pub xpos: f64,
    pub ypos: f64,
    pub xorigin: f64,
    pub yorigin: f64,
    pub orientation: u8,
    pub is_horizontal: bool,
    /// +1 or -1: which way along its axis the thwump charges.
    pub direction: f64,
    /// -1 retreating, 0 immobile, 1 charging.
    pub state: i8,
    pub cell: Cell,
}

impl Thwump {
    pub fn new(xcoord: i32, ycoord: i32, orientation: u8) -> Self {
        let xpos = (xcoord * 6) as f64;
        let ypos = (ycoord * 6) as f64;
        Thwump {
            xpos,
            ypos,
            xorigin: xpos,
            yorigin: ypos,
            orientation,
            is_horizontal: orientation == 0 || orientation == 4,
            direction: if orientation == 0 || orientation == 2 {
                1.0
            } else {
                -1.0
            },
            state: 0,
            cell: clamp_cell((xpos / 24.0).floor() as i32, (ypos / 24.0).floor() as i32),
        }
    }

    /// Advance along the charge axis. Only runs while moving.
    ///
    /// Two early exits matter: retreating PAST the origin snaps exactly onto it
    /// and stops (so the thwump never drifts), and hitting a solid half-cell edge
    /// while charging flips straight to retreating.
    pub fn move_entity(&mut self, ctx: &mut EntityCtx) {
        if self.state == 0 {
            return;
        }
        let speed = if self.state == 1 {
            THWUMP_FORWARD_SPEED
        } else {
            THWUMP_BACKWARD_SPEED
        };
        let speed_dir = self.direction * (self.state as f64);

        if !self.is_horizontal {
            let ypos_new = self.ypos + speed * speed_dir;
            if self.state == -1 && (ypos_new - self.yorigin) * (self.ypos - self.yorigin) < 0.0 {
                self.ypos = self.yorigin;
                self.state = 0;
                return;
            }
            let cell_y = ((self.ypos + speed_dir * 11.0) / 12.0).floor() as i32;
            let cell_y_new = ((ypos_new + speed_dir * 11.0) / 12.0).floor() as i32;
            if cell_y != cell_y_new {
                let cell_x1 = ((self.xpos - 11.0) / 12.0).floor() as i32;
                let cell_x2 = ((self.xpos + 11.0) / 12.0).floor() as i32;
                if !ctx
                    .grid_edges
                    .is_empty_row(cell_x1, cell_x2, cell_y, speed_dir as i32)
                {
                    self.state = -1;
                    return;
                }
            }
            self.ypos = ypos_new;
        } else {
            let xpos_new = self.xpos + speed * speed_dir;
            if self.state == -1 && (xpos_new - self.xorigin) * (self.xpos - self.xorigin) < 0.0 {
                self.xpos = self.xorigin;
                self.state = 0;
                return;
            }
            let cell_x = ((self.xpos + speed_dir * 11.0) / 12.0).floor() as i32;
            let cell_x_new = ((xpos_new + speed_dir * 11.0) / 12.0).floor() as i32;
            if cell_x != cell_x_new {
                let cell_y1 = ((self.ypos - 11.0) / 12.0).floor() as i32;
                let cell_y2 = ((self.ypos + 11.0) / 12.0).floor() as i32;
                if !ctx
                    .grid_edges
                    .is_empty_column(cell_x, cell_y1, cell_y2, speed_dir as i32)
                {
                    self.state = -1;
                    return;
                }
            }
            self.xpos = xpos_new;
        }

        let new_cell = clamp_cell(
            (self.xpos / 24.0).floor() as i32,
            (self.ypos / 24.0).floor() as i32,
        );
        let old_cell = self.cell;
        self.cell = new_cell;
        ctx.grid_move(old_cell, new_cell);
    }

    /// Start charging if the ninja is within the activation band AND there is an
    /// unobstructed run of half-cells between them.
    ///
    /// The sight scan walks up to 100 half-cells forward. Python's `if i > 0`
    /// guard means a ninja in the very first cell does NOT trigger the charge --
    /// preserved here via `steps > 0`.
    pub fn think(&mut self, ninja: &Ninja, edges: &GridEdges) {
        if self.state != 0 || !ninja.is_valid_target() {
            return;
        }
        let activation_range = 2.0 * (THWUMP_SEMI_SIDE + NINJA_RADIUS);
        let dir = self.direction;

        if !self.is_horizontal {
            if (self.xpos - ninja.xpos).abs() >= activation_range {
                return;
            }
            let ninja_ycell = (ninja.ypos / 12.0).floor() as i32;
            let mut thwump_ycell = ((self.ypos - dir * 11.0) / 12.0).floor() as i32;
            let x1 = ((self.xpos - 11.0) / 12.0).floor() as i32;
            let x2 = ((self.xpos + 11.0) / 12.0).floor() as i32;
            let mut dy = ninja_ycell - thwump_ycell;
            if (dy as f64) * dir < 0.0 {
                return;
            }
            let mut steps = 0;
            for i in 0..100 {
                steps = i;
                if !edges.is_empty_row(x1, x2, thwump_ycell, dir as i32) {
                    dy = ninja_ycell - thwump_ycell;
                    break;
                }
                thwump_ycell += dir as i32;
            }
            if steps > 0 && (dy as f64) * dir <= 0.0 {
                self.state = 1;
            }
        } else {
            if (self.ypos - ninja.ypos).abs() >= activation_range {
                return;
            }
            let ninja_xcell = (ninja.xpos / 12.0).floor() as i32;
            let mut thwump_xcell = ((self.xpos - dir * 11.0) / 12.0).floor() as i32;
            let y1 = ((self.ypos - 11.0) / 12.0).floor() as i32;
            let y2 = ((self.ypos + 11.0) / 12.0).floor() as i32;
            let mut dx = ninja_xcell - thwump_xcell;
            if (dx as f64) * dir < 0.0 {
                return;
            }
            let mut steps = 0;
            for i in 0..100 {
                steps = i;
                if !edges.is_empty_column(thwump_xcell, y1, y2, dir as i32) {
                    dx = ninja_xcell - thwump_xcell;
                    break;
                }
                thwump_xcell += dir as i32;
            }
            if steps > 0 && (dx as f64) * dir <= 0.0 {
                self.state = 1;
            }
        }
    }

    /// Solid on all sides -- the thwump pushes the ninja, and is the only entity
    /// that can crush it.
    pub fn physical_collision(&self, ninja: &Ninja) -> Option<((f64, f64), (f64, f64))> {
        penetration_square_vs_point(
            self.xpos,
            self.ypos,
            ninja.xpos,
            ninja.ypos,
            THWUMP_SEMI_SIDE + NINJA_RADIUS,
        )
    }

    /// Returns the wall normal for side contact, and kills the ninja if it is
    /// touching the lethal charging face.
    ///
    /// The lethal region is a segment inset 2px from the face's corners and
    /// pushed 2px proud of it, tested against a ninja circle inflated by 2 --
    /// slightly more generous than the face itself.
    pub fn logical_collision(&self, ninja: &mut Ninja) -> Option<f64> {
        if !ninja.is_valid_target() {
            return None;
        }
        let depen = penetration_square_vs_point(
            self.xpos,
            self.ypos,
            ninja.xpos,
            ninja.ypos,
            THWUMP_SEMI_SIDE + NINJA_RADIUS + 0.1,
        )?;
        let (px1, py1, px2, py2) = if self.is_horizontal {
            let dx = (THWUMP_SEMI_SIDE + 2.0) * self.direction;
            let dy = THWUMP_SEMI_SIDE - 2.0;
            (self.xpos + dx, self.ypos - dy, self.xpos + dx, self.ypos + dy)
        } else {
            let dx = THWUMP_SEMI_SIDE - 2.0;
            let dy = (THWUMP_SEMI_SIDE + 2.0) * self.direction;
            (self.xpos - dx, self.ypos + dy, self.xpos + dx, self.ypos + dy)
        };
        if overlap_circle_vs_segment(
            ninja.xpos,
            ninja.ypos,
            NINJA_RADIUS + 2.0,
            px1,
            py1,
            px2,
            py2,
        ) {
            ninja.kill(0, 0.0, 0.0, 0.0, 0.0, Some("thwump"));
        }
        Some(depen.0 .0)
    }

    pub fn reset_state(&mut self) {
        self.xpos = self.xorigin;
        self.ypos = self.yorigin;
        self.state = 0;
        self.cell = clamp_cell(
            (self.xpos / 24.0).floor() as i32,
            (self.ypos / 24.0).floor() as i32,
        );
    }
}
