//! Toggle Mine Entity (Types 1, 21).
//! Direct port of nclone/entity_classes/entity_toggle_mine.py.

use crate::collision::intersection::overlap_circle_vs_circle;
use crate::constants::{
    NINJA_RADIUS, TOGGLE_MINE_RADIUS_TOGGLED, TOGGLE_MINE_RADIUS_TOGGLING,
    TOGGLE_MINE_RADIUS_UNTOGGLED,
};
use crate::ninja::Ninja;

/// Toggle mine states.
pub const STATE_TOGGLED: u8 = 0; // Deadly
pub const STATE_UNTOGGLED: u8 = 1; // Safe
pub const STATE_TOGGLING: u8 = 2; // Transitioning

#[derive(Debug, Clone)]
pub struct ToggleMine {
    pub entity_type: u8, // 1 or 21
    pub xpos: f64,
    pub ypos: f64,
    pub active: bool,
    pub state: u8,
    pub radius: f64,

    // Initial state for reset
    initial_state: u8,
    initial_xpos: f64,
    initial_ypos: f64,
}

impl ToggleMine {
    pub fn new(entity_type: u8, xcoord: i32, ycoord: i32) -> Self {
        let initial_state = if entity_type == 21 {
            STATE_TOGGLED
        } else {
            STATE_UNTOGGLED
        };
        let radius = radius_for_state(initial_state);
        let xpos = (xcoord * 6) as f64;
        let ypos = (ycoord * 6) as f64;
        ToggleMine {
            entity_type,
            xpos,
            ypos,
            active: true,
            state: initial_state,
            radius,
            initial_state,
            initial_xpos: xpos,
            initial_ypos: ypos,
        }
    }

    /// Per-frame think: handle ninja-mine interaction state transitions.
    pub fn think(&mut self, ninja: &Ninja) {
        if ninja.is_valid_target() {
            if self.state == STATE_UNTOGGLED {
                // Untoggled -> Toggling if ninja touches
                if overlap_circle_vs_circle(
                    self.xpos,
                    self.ypos,
                    self.radius,
                    ninja.xpos,
                    ninja.ypos,
                    NINJA_RADIUS,
                ) {
                    self.set_state(STATE_TOGGLING);
                }
            } else if self.state == STATE_TOGGLING {
                // Toggling -> Toggled if ninja stops touching
                if !overlap_circle_vs_circle(
                    self.xpos,
                    self.ypos,
                    self.radius,
                    ninja.xpos,
                    ninja.ypos,
                    NINJA_RADIUS,
                ) {
                    self.set_state(STATE_TOGGLED);
                }
            }
        } else {
            // Toggling -> Untoggled if ninja dies
            if self.state == STATE_TOGGLING && ninja.state == 6 {
                self.set_state(STATE_UNTOGGLED);
            }
        }
    }

    /// Logical collision: kill ninja if touching a toggled mine.
    pub fn logical_collision(&mut self, ninja: &mut Ninja) -> bool {
        if ninja.is_valid_target() && self.state == STATE_TOGGLED {
            if overlap_circle_vs_circle(
                self.xpos,
                self.ypos,
                self.radius,
                ninja.xpos,
                ninja.ypos,
                NINJA_RADIUS,
            ) {
                self.set_state(STATE_UNTOGGLED);
                ninja.kill(0, 0.0, 0.0, 0.0, 0.0, Some("mine"));
                return true;
            }
        }
        false
    }

    fn set_state(&mut self, state: u8) {
        if state <= 2 {
            self.state = state;
            self.radius = radius_for_state(state);
        }
    }

    pub fn reset_state(&mut self) {
        self.xpos = self.initial_xpos;
        self.ypos = self.initial_ypos;
        self.active = true;
        self.set_state(self.initial_state);
    }
}

fn radius_for_state(state: u8) -> f64 {
    match state {
        STATE_TOGGLED => TOGGLE_MINE_RADIUS_TOGGLED,
        STATE_UNTOGGLED => TOGGLE_MINE_RADIUS_UNTOGGLED,
        STATE_TOGGLING => TOGGLE_MINE_RADIUS_TOGGLING,
        _ => TOGGLE_MINE_RADIUS_TOGGLED,
    }
}
