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
use crate::geometry::map_orientation_to_vector;
use crate::ninja::Ninja;

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
        // `0.9` and the inclusive `<= 0` are `DAT_10571cd8` and the comparison in
        // `docs/decomp/full/entity_1040ED60.c`; the inherited value was 0.91, and
        // that hundredth is worth a tenth of a radius of reach. It cost seven
        // demos: on `10139391` the ninja clears a platform edge by
        // `|lateral| = 21.07` against the true bound of `21.0`, and 0.91 turned
        // the miss into a ceiling on frame 183, which is where five demos of one
        // map family left the game's trajectory. 6488 -> 6495 on the 8000-demo
        // gate, no demo lost.
        let radius_scalar = if direction <= 0.0 { 0.9 } else { 0.51 };
        if lateral_dist.abs() < radius_scalar * NINJA_RADIUS + ONE_WAY_PLATFORM_SEMI_SIDE {
            let normal_dist = dx * self.normal_x + dy * self.normal_y;
            // `FUN_1040ED60` gates on `|normal_dist| < RADIUS + arg8` instead,
            // which both admits a ninja that overshot to the back side and
            // rejects the exact-touch case. Ported literally with `arg8 = 0` --
            // the physical caller `FUN_1040EF70` passes a literal zero -- it
            // costs 599 demos, so one of the two readings is wrong. The logical
            // caller `FUN_1040EEC0` forwards a *real* eighth argument, so the
            // game likely runs this test at two different slacks and our one
            // shared helper cannot express that. Left as the inherited form
            // until the logical caller is decompiled.
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
