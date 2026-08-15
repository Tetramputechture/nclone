//! Regular Door (Type 5) and Trap Door (Type 8).
//! Direct ports of nclone/entity_classes/entity_door_regular.py and
//! entity_door_trap.py, both `EntityDoorBase` subclasses.
//!
//! Both own a 24px collision segment plus a pair of half-cell grid edges, and
//! differ only in what flips them:
//!
//! * **Regular** — proximity-activated, radius 10 around the door itself. Opens
//!   on contact and auto-closes 5 frames after the ninja stops touching it. Its
//!   "switch" position IS the door position.
//! * **Trap** — starts OPEN and closes PERMANENTLY when its switch (radius 5,
//!   somewhere else on the map) is collected. One-way: nothing reopens it.
//!
//! The locked door (type 6) is the third sibling but predates this module and
//! lives in `locked_door.rs`.
//!
//! Neither entity flips its own segment or grid edges — they cannot reach them.
//! They only move `closed`, and `Simulator::sync_door_barriers` reconciles the
//! two blocking layers afterwards.

use crate::collision::intersection::overlap_circle_vs_circle;
use crate::constants::{NINJA_RADIUS, REGULAR_DOOR_RADIUS, TRAP_DOOR_RADIUS};
use crate::ninja::Ninja;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchDoorKind {
    Regular,
    Trap,
}

#[derive(Debug, Clone)]
pub struct SwitchDoor {
    pub kind: SwitchDoorKind,
    /// Switch / trigger position. For a regular door this equals the door
    /// position; for a trap door it is a separate point on the map.
    pub xpos: f64,
    pub ypos: f64,
    /// The physical barrier.
    pub door_xpos: f64,
    pub door_ypos: f64,
    pub orientation: u8,
    pub closed: bool,
    /// Switch still collectable. Always true for a regular door (it re-triggers
    /// forever); false for a trap door once its switch has been taken.
    pub active: bool,
    /// Frames since the regular door was last touched; it shuts above 5.
    pub open_timer: u32,
    pub door_segment_index: usize,
    /// Whether this door's grid edges are currently counted, so the simulator can
    /// apply only the DELTA when `closed` flips.
    pub registered_closed: bool,
}

impl SwitchDoor {
    pub fn new(
        kind: SwitchDoorKind,
        door_xcoord: i32,
        door_ycoord: i32,
        orientation: u8,
        sw_xcoord: i32,
        sw_ycoord: i32,
        door_segment_index: usize,
    ) -> Self {
        SwitchDoor {
            kind,
            xpos: (sw_xcoord * 6) as f64,
            ypos: (sw_ycoord * 6) as f64,
            door_xpos: (door_xcoord * 6) as f64,
            door_ypos: (door_ycoord * 6) as f64,
            orientation,
            // A trap door is constructed and then immediately opened
            // (`change_state(closed=False)` in its __init__), so it starts open.
            closed: kind == SwitchDoorKind::Regular,
            active: true,
            open_timer: 0,
            door_segment_index,
            // Both layers start matching `closed`; the simulator registers the
            // initial state at load.
            registered_closed: kind == SwitchDoorKind::Regular,
        }
    }

    pub fn radius(&self) -> f64 {
        match self.kind {
            SwitchDoorKind::Regular => REGULAR_DOOR_RADIUS,
            SwitchDoorKind::Trap => TRAP_DOOR_RADIUS,
        }
    }

    pub fn type_id(&self) -> u8 {
        match self.kind {
            SwitchDoorKind::Regular => 5,
            SwitchDoorKind::Trap => 8,
        }
    }

    /// Regular doors only: shut again once 5 frames have passed without contact.
    /// The timer is reset by every `logical_collision` hit, so holding position
    /// next to the door keeps it open.
    pub fn think(&mut self) {
        if self.kind != SwitchDoorKind::Regular {
            return;
        }
        if !self.closed {
            self.open_timer += 1;
            if self.open_timer > 5 {
                self.closed = true;
            }
        }
    }

    /// Regular: open (or hold open) while the ninja is inside the trigger radius.
    /// Trap: collect the switch and slam shut, permanently.
    ///
    /// Note the regular door has NO `is_valid_target` guard in Python -- a dead
    /// ninja's corpse position still opens it.
    pub fn logical_collision(&mut self, ninja: &Ninja) {
        if !overlap_circle_vs_circle(
            self.xpos,
            self.ypos,
            self.radius(),
            ninja.xpos,
            ninja.ypos,
            NINJA_RADIUS,
        ) {
            return;
        }
        match self.kind {
            SwitchDoorKind::Regular => {
                self.closed = false;
                self.open_timer = 0;
            }
            SwitchDoorKind::Trap => {
                self.closed = true;
                self.active = false;
            }
        }
    }

    pub fn reset_state(&mut self) {
        self.closed = self.kind == SwitchDoorKind::Regular;
        self.active = true;
        self.open_timer = 0;
    }
}
