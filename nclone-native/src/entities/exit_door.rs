//! Exit Door Entity (Type 3).
//! Direct port of nclone/entity_classes/entity_exit.py.

use crate::collision::intersection::overlap_circle_vs_circle;
use crate::constants::{EXIT_DOOR_RADIUS, NINJA_RADIUS};
use crate::ninja::Ninja;

#[derive(Debug, Clone)]
pub struct ExitDoor {
    pub xpos: f64,
    pub ypos: f64,
    pub switch_hit: bool,

    // Initial state for reset
    initial_xpos: f64,
    initial_ypos: f64,
}

impl ExitDoor {
    pub fn new(xcoord: i32, ycoord: i32) -> Self {
        let xpos = (xcoord * 6) as f64;
        let ypos = (ycoord * 6) as f64;
        ExitDoor {
            xpos,
            ypos,
            switch_hit: false,
            initial_xpos: xpos,
            initial_ypos: ypos,
        }
    }

    /// Logical collision: ninja wins if touching an active (switch hit) exit door.
    pub fn logical_collision(&mut self, ninja: &mut Ninja) -> bool {
        if !self.switch_hit {
            return false;
        }
        if overlap_circle_vs_circle(
            self.xpos,
            self.ypos,
            EXIT_DOOR_RADIUS,
            ninja.xpos,
            ninja.ypos,
            NINJA_RADIUS,
        ) {
            ninja.win();
            return true;
        }
        false
    }

    pub fn reset_state(&mut self) {
        self.xpos = self.initial_xpos;
        self.ypos = self.initial_ypos;
        self.switch_hit = false;
    }
}
