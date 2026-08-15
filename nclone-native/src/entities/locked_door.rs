//! Locked Door Entity (Type 6).
//! Direct port of nclone/entity_classes/entity_door_locked.py (+ EntityDoorBase).
//!
//! A locked door is a 24px wall segment that starts CLOSED (blocking) and opens
//! PERMANENTLY when the ninja touches its associated switch (radius 5). The entity
//! position (xpos/ypos) is the SWITCH location, matching Python where the door entity
//! moves its own position to the switch after building the door segment. The door's
//! collision segment lives in the shared segment pool; opening the door simply flips
//! that segment's `active` flag to false (done simulator-side, since the entity cannot
//! reach `self.segments`).

use crate::collision::intersection::overlap_circle_vs_circle;
use crate::constants::{LOCKED_DOOR_SWITCH_RADIUS, NINJA_RADIUS};
use crate::ninja::Ninja;
use crate::segments::Segment;
use crate::types::Cell;

#[derive(Debug, Clone)]
pub struct LockedDoor {
    // Entity position == switch trigger position (mirrors Python, which moves
    // xpos/ypos to the switch after constructing the door segment).
    pub xpos: f64,
    pub ypos: f64,

    // Door segment midpoint (the physical barrier), kept for snapshots/planner.
    pub door_xpos: f64,
    pub door_ypos: f64,
    pub orientation: u8,

    /// True while the switch is uncollected. Becomes false once collected.
    pub active: bool,
    /// True while the door segment blocks. Becomes false (permanently) once opened.
    pub closed: bool,
    /// Index into `Simulator.segments` of this door's collision segment.
    pub door_segment_index: usize,
    /// Whether this door's grid edges are currently counted, so the simulator can
    /// apply only the DELTA when `closed` flips. See `Simulator::sync_door_barriers`.
    pub registered_closed: bool,

    // Initial state for fast_reset.
    initial_active: bool,
    initial_closed: bool,
}

impl LockedDoor {
    /// `door_xcoord/door_ycoord` are the door position; `sw_xcoord/sw_ycoord` the switch.
    /// Coordinates are in map units (multiply by 6 to get world pixels), matching the
    /// other entity constructors. `door_segment_index` is the index the caller assigned
    /// when it pushed this door's segment into the shared pool.
    pub fn new(
        door_xcoord: i32,
        door_ycoord: i32,
        orientation: u8,
        sw_xcoord: i32,
        sw_ycoord: i32,
        door_segment_index: usize,
    ) -> Self {
        let door_xpos = (door_xcoord * 6) as f64;
        let door_ypos = (door_ycoord * 6) as f64;
        let sw_xpos = (sw_xcoord * 6) as f64;
        let sw_ypos = (sw_ycoord * 6) as f64;
        LockedDoor {
            xpos: sw_xpos,
            ypos: sw_ypos,
            door_xpos,
            door_ypos,
            orientation,
            active: true,
            closed: true,
            door_segment_index,
            registered_closed: true,
            initial_active: true,
            initial_closed: true,
        }
    }

    /// Build the door's collision segment and its grid cell from the door position
    /// and orientation. Mirrors EntityDoorBase.__init__ (entity_door_base.py:60-92):
    /// the door cell is found by offsetting the door position by DOOR_RADIUS along the
    /// orientation vector, then clamping. The segment is a two-sided (oriented=false)
    /// 24px linear wall, active (=closed) on creation.
    pub fn make_door_segment(door_xcoord: i32, door_ycoord: i32, orientation: u8) -> (Segment, Cell) {
        crate::entities::door_base::make_door_segment(door_xcoord, door_ycoord, orientation)
    }

    /// Logical collision against the switch (radius 5). Returns true if the ninja
    /// collects the switch this frame; the caller is responsible for opening the door
    /// segment (flipping `segments[door_segment_index].active = false`) and bumping
    /// `ninja.doors_opened`.
    pub fn logical_collision(&mut self, ninja: &mut Ninja) -> bool {
        if !self.active {
            return false;
        }
        if overlap_circle_vs_circle(
            self.xpos,
            self.ypos,
            LOCKED_DOOR_SWITCH_RADIUS,
            ninja.xpos,
            ninja.ypos,
            NINJA_RADIUS,
        ) {
            self.active = false;
            self.closed = false;
            return true;
        }
        false
    }

    pub fn reset_state(&mut self) {
        self.active = self.initial_active;
        self.closed = self.initial_closed;
    }
}

