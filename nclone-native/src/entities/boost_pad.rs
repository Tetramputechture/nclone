//! Boost Pad Entity (Type 24).
//! Direct port of nclone/entity_classes/entity_boost_pad.py.
//!
//! Amplifies whatever direction the ninja is already travelling: on the frame
//! contact BEGINS, 2 units are added to the velocity norm. Unlike the launch pad
//! this is direction-agnostic and additive.
//!
//! It runs in the `move` phase, not as a logical collision -- so the boost is
//! applied BEFORE the ninja integrates that frame, which is why an entry-speed
//! boost carries into the same frame's motion.
//!
//! `is_touching_ninja` latches the one-shot: the ninja must break contact before
//! it can be boosted again.

use crate::collision::intersection::overlap_circle_vs_circle;
use crate::constants::{BOOST_PAD_RADIUS, NINJA_RADIUS};
use crate::entities::EntityCtx;

#[derive(Debug, Clone)]
pub struct BoostPad {
    pub xpos: f64,
    pub ypos: f64,
    /// Latch preventing a re-boost until the ninja leaves the pad.
    pub is_touching_ninja: bool,
}

impl BoostPad {
    pub fn new(xcoord: i32, ycoord: i32) -> Self {
        BoostPad {
            xpos: (xcoord * 6) as f64,
            ypos: (ycoord * 6) as f64,
            is_touching_ninja: false,
        }
    }

    pub fn move_entity(&mut self, ctx: &mut EntityCtx) {
        let ninja = &mut *ctx.ninja;
        // An invalid target (dead/celebrating) also CLEARS the latch, so the pad
        // is armed again on respawn.
        if !ninja.is_valid_target() {
            self.is_touching_ninja = false;
            return;
        }
        if overlap_circle_vs_circle(
            self.xpos,
            self.ypos,
            BOOST_PAD_RADIUS,
            ninja.xpos,
            ninja.ypos,
            NINJA_RADIUS,
        ) {
            if !self.is_touching_ninja || crate::ab::ab().boost_nolatch {
                let vel_norm = (ninja.xspeed * ninja.xspeed + ninja.yspeed * ninja.yspeed).sqrt();
                if vel_norm > 0.0 {
                    let mag = crate::ab::ab().boost_mag;
                    ninja.xspeed += mag * ninja.xspeed / vel_norm;
                    ninja.yspeed += mag * ninja.yspeed / vel_norm;
                }
                self.is_touching_ninja = true;
            }
        } else {
            self.is_touching_ninja = false;
        }
    }

    pub fn reset_state(&mut self) {
        self.is_touching_ninja = false;
    }
}
