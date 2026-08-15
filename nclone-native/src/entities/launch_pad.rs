//! Launch Pad Entity (Type 10).
//! Direct port of nclone/entity_classes/entity_launch_pad.py.
//!
//! A static directional impulse: touch the pad from the front and the ninja's
//! velocity is REPLACED (not added to) by a boost along the pad's orientation
//! normal. Logical-only -- it never pushes the ninja physically.
//!
//! The response lives in the caller (`Simulator::post_collision_with_entities`),
//! mirroring the `entity.type == 10` branch of `Ninja.post_collision`
//! (nclone/ninja.py:434-451), because it rewrites ninja state wholesale
//! (position, velocity, floor state, launch-pad buffer, movement state) rather
//! than contributing a wall normal like every other logical entity.

use crate::collision::intersection::overlap_circle_vs_circle;
use crate::constants::{LAUNCH_PAD_BOOST, LAUNCH_PAD_RADIUS, NINJA_RADIUS};
use crate::geometry::map_orientation_to_vector;
use crate::ninja::Ninja;

#[derive(Debug, Clone)]
pub struct LaunchPad {
    pub xpos: f64,
    pub ypos: f64,
    pub orientation: u8,
    pub normal_x: f64,
    pub normal_y: f64,
}

impl LaunchPad {
    pub fn new(xcoord: i32, ycoord: i32, orientation: u8) -> Self {
        let (normal_x, normal_y) = map_orientation_to_vector(orientation);
        LaunchPad {
            xpos: (xcoord * 6) as f64,
            ypos: (ycoord * 6) as f64,
            orientation,
            normal_x,
            normal_y,
        }
    }

    /// Return the raw `(xboost, yboost)` if the ninja is touching the pad's
    /// semi-circular hitbox from the front, else `None`.
    ///
    /// Two details worth not "simplifying": the front-facing test offsets the
    /// ninja by its radius along the normal and accepts `>= -0.1` (a small
    /// tolerance, so grazing contact still fires), and an UPWARD-facing pad
    /// (`normal_y < 0`) scales its y-boost by `1 - normal_y`, making upward
    /// launches stronger than the nominal BOOST.
    pub fn logical_collision(&self, ninja: &Ninja) -> Option<(f64, f64)> {
        if !ninja.is_valid_target() {
            return None;
        }
        if !overlap_circle_vs_circle(
            self.xpos,
            self.ypos,
            LAUNCH_PAD_RADIUS,
            ninja.xpos,
            ninja.ypos,
            NINJA_RADIUS,
        ) {
            return None;
        }
        let facing = (self.xpos - (ninja.xpos - NINJA_RADIUS * self.normal_x)) * self.normal_x
            + (self.ypos - (ninja.ypos - NINJA_RADIUS * self.normal_y)) * self.normal_y;
        if facing < -0.1 {
            return None;
        }
        let yboost_scale = if self.normal_y < 0.0 {
            1.0 - self.normal_y
        } else {
            1.0
        };
        Some((
            self.normal_x * LAUNCH_PAD_BOOST,
            self.normal_y * LAUNCH_PAD_BOOST * yboost_scale,
        ))
    }

    /// Static entity: nothing to reset.
    pub fn reset_state(&mut self) {}
}
