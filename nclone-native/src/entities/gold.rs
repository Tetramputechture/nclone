//! Gold Entity (Type 2).
//! Direct port of nclone/entity_classes/entity_gold.py.
//!
//! Optional collectible: collected on contact with a ninja that is NOT in the
//! winning state (state 8); increments `ninja.gold_collected` and deactivates.
//! No physical interaction.

use crate::collision::intersection::overlap_circle_vs_circle;
use crate::constants::{GOLD_RADIUS, NINJA_RADIUS};
use crate::ninja::Ninja;

#[derive(Debug, Clone)]
pub struct Gold {
    pub xpos: f64,
    pub ypos: f64,
    pub active: bool,

    // Initial state for reset
    initial_xpos: f64,
    initial_ypos: f64,
}

impl Gold {
    pub fn new(xcoord: i32, ycoord: i32) -> Self {
        let xpos = (xcoord * 6) as f64;
        let ypos = (ycoord * 6) as f64;
        Gold {
            xpos,
            ypos,
            active: true,
            initial_xpos: xpos,
            initial_ypos: ypos,
        }
    }

    /// Collected if touched by a ninja not in the winning state (state 8).
    /// Increments `ninja.gold_collected` and deactivates. Returns true if
    /// collected this call.
    pub fn logical_collision(&mut self, ninja: &mut Ninja) -> bool {
        if !self.active {
            return false;
        }
        if ninja.state != 8
            && overlap_circle_vs_circle(
                self.xpos,
                self.ypos,
                GOLD_RADIUS,
                ninja.xpos,
                ninja.ypos,
                NINJA_RADIUS,
            )
        {
            self.active = false;
            ninja.gold_collected += 1;
            return true;
        }
        false
    }

    pub fn reset_state(&mut self) {
        self.xpos = self.initial_xpos;
        self.ypos = self.initial_ypos;
        self.active = true;
    }
}
