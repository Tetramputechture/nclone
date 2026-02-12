//! Exit Switch Entity (Type 4).
//! Direct port of nclone/entity_classes/entity_exit_switch.py.

use crate::collision::intersection::overlap_circle_vs_circle;
use crate::constants::{EXIT_SWITCH_RADIUS, NINJA_RADIUS};
use crate::ninja::Ninja;

#[derive(Debug, Clone)]
pub struct ExitSwitch {
    pub xpos: f64,
    pub ypos: f64,
    pub active: bool,

    // Index of the parent exit door in the entities list
    pub parent_door_index: usize,

    // Initial state for reset
    initial_xpos: f64,
    initial_ypos: f64,
}

impl ExitSwitch {
    pub fn new(xcoord: i32, ycoord: i32, parent_door_index: usize) -> Self {
        let xpos = (xcoord * 6) as f64;
        let ypos = (ycoord * 6) as f64;
        ExitSwitch {
            xpos,
            ypos,
            active: true,
            parent_door_index,
            initial_xpos: xpos,
            initial_ypos: ypos,
        }
    }

    /// Logical collision: deactivate switch and activate parent door when ninja touches.
    /// Returns true if switch was collected (caller must update parent door).
    pub fn logical_collision(&mut self, ninja: &mut Ninja) -> bool {
        if !self.active {
            return false;
        }
        if overlap_circle_vs_circle(
            self.xpos,
            self.ypos,
            EXIT_SWITCH_RADIUS,
            ninja.xpos,
            ninja.ypos,
            NINJA_RADIUS,
        ) {
            self.active = false;
            return true; // Caller must set parent door's switch_hit = true
        }
        false
    }

    pub fn reset_state(&mut self) {
        self.xpos = self.initial_xpos;
        self.ypos = self.initial_ypos;
        self.active = true;
    }
}
