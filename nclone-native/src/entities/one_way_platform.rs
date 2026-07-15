//! One-Way Platform Entity (Type 11).
//! Direct port of nclone/entity_classes/entity_one_way_platform.py.
//!
//! A directional 24px barrier (semi-side 12): the ninja passes through freely
//! from one side but is depenetrated (and has its velocity projected onto the
//! platform tangent) when approaching from the blocking side, determined by the
//! platform's orientation normal. Unlike every other entity ported so far this
//! is PHYSICALLY collidable (`physical_collision`) — it pushes the ninja — and
//! is the reason `Ninja::collide_vs_objects` exists.

use crate::constants::{NINJA_RADIUS, ONE_WAY_PLATFORM_SEMI_SIDE};
use crate::ninja::Ninja;

/// Map an 8-way orientation (0..7) to a unit normal vector, matching
/// `map_orientation_to_vector` in nclone/physics.py. `diag = sqrt(2)/2` is
/// computed identically to Python's `math.sqrt(2)/2` (bit-identical f64).
pub fn map_orientation_to_vector(orientation: u8) -> (f64, f64) {
    let diag = (2.0_f64).sqrt() / 2.0;
    match orientation & 7 {
        0 => (1.0, 0.0),
        1 => (diag, diag),
        2 => (0.0, 1.0),
        3 => (-diag, diag),
        4 => (-1.0, 0.0),
        5 => (-diag, -diag),
        6 => (0.0, -1.0),
        7 => (diag, -diag),
        _ => unreachable!(),
    }
}

#[derive(Debug, Clone)]
pub struct OneWayPlatform {
    pub xpos: f64,
    pub ypos: f64,
    pub orientation: u8,
    pub normal_x: f64,
    pub normal_y: f64,
}

impl OneWayPlatform {
    pub fn new(xcoord: i32, ycoord: i32, orientation: u8) -> Self {
        let (normal_x, normal_y) = map_orientation_to_vector(orientation);
        OneWayPlatform {
            xpos: (xcoord * 6) as f64,
            ypos: (ycoord * 6) as f64,
            orientation,
            normal_x,
            normal_y,
        }
    }

    /// Return the depenetration between the ninja and the one-way as
    /// `((normal_x, normal_y), (depen_len, 0.0))`, or `None` if no penetration.
    /// Faithful port of `EntityOneWayPlatform.calculate_depenetration`.
    pub fn calculate_depenetration(&self, ninja: &Ninja) -> Option<((f64, f64), (f64, f64))> {
        let dx = ninja.xpos - self.xpos;
        let dy = ninja.ypos - self.ypos;
        let lateral_dist = dy * self.normal_x - dx * self.normal_y;
        let direction =
            (ninja.yspeed * self.normal_x - ninja.xspeed * self.normal_y) * lateral_dist;
        // Wider collision zone when the ninja moves towards the platform centre.
        let radius_scalar = if direction < 0.0 { 0.91 } else { 0.51 };
        if lateral_dist.abs() < radius_scalar * NINJA_RADIUS + ONE_WAY_PLATFORM_SEMI_SIDE {
            let normal_dist = dx * self.normal_x + dy * self.normal_y;
            if normal_dist > 0.0 && normal_dist <= NINJA_RADIUS {
                let normal_proj = ninja.xspeed * self.normal_x + ninja.yspeed * self.normal_y;
                if normal_proj <= 0.0 {
                    let dx_old = ninja.xpos_old - self.xpos;
                    let dy_old = ninja.ypos_old - self.ypos;
                    let normal_dist_old = dx_old * self.normal_x + dy_old * self.normal_y;
                    if NINJA_RADIUS - normal_dist_old <= 1.1 {
                        return Some((
                            (self.normal_x, self.normal_y),
                            (NINJA_RADIUS - normal_dist, 0.0),
                        ));
                    }
                }
            }
        }
        None
    }

    /// Physical collision: depenetration vector applied to the ninja.
    pub fn physical_collision(&self, ninja: &Ninja) -> Option<((f64, f64), (f64, f64))> {
        self.calculate_depenetration(ninja)
    }

    /// Logical collision: return the wall normal (`±1`) iff the ninja entered a
    /// walled state from a horizontal-facing one-way, else `None`. Mirrors
    /// `EntityOneWayPlatform.logical_collision` (only fires when `|normal_x| == 1`).
    pub fn logical_collision(&self, ninja: &Ninja) -> Option<f64> {
        if self.calculate_depenetration(ninja).is_some() && self.normal_x.abs() == 1.0 {
            Some(self.normal_x)
        } else {
            None
        }
    }

    /// Static entity: nothing to reset.
    pub fn reset_state(&mut self) {}
}
