//! Player-Controlled Rocket (Type 29).
//! Port of nclone/entity_classes/entity_pc_rocket.py; reconstructed from
//! `npp.dll` per `docs/ENTITY_INTEGRATION_SPEC.md` §8.9. This type was absent
//! from every entity table the project had before the RE pass -- a genuine gap.
//!
//! **It is not map-placed, so `map/loader.rs` deliberately has no arm for it.**
//! The spec flagged this as an open question ([I] in §2): type 29's factory call
//! is `FUN_1040f560(iVar20, iVar20)`, taking no position or scale arguments unlike
//! every other case, suggesting a runtime spawn. That is now settled empirically --
//! type 29 does not occur once in the entity streams of 8000 decoded attract maps
//! (nor do the phantom types 7 and 9), while every other official type does.
//!
//! It is implemented anyway because it completes the official type set and costs a
//! driver rather than an entity: it shares its whole projectile integrator with
//! type 18 (spec: "every helper is shared"). The difference is **steering** -- the
//! think reads player input and applies it to the projectile's heading each frame,
//! and on teardown releases the controller by writing `+0x24 = -1`.
//!
//! Because nothing can construct one today, this code is unexercised by the
//! attract corpus. The exact input-to-direction mapping is [U]; `PC_ROCKET_TURN_RATE`
//! is a real think constant applied as radians per frame per unit of horizontal
//! input, but that binding is inferred.

use crate::collision::intersection::overlap_circle_vs_circle;
use crate::constants::{NINJA_RADIUS, PC_ROCKET_TURN_RATE, ROCKET_RADIUS};
use crate::entities::rocket::{aim_at, Projectile, STATE_FLYING, STATE_IDLE, STATE_LAUNCH};
use crate::entities::EntityCtx;
use crate::ninja::Ninja;
use crate::types::{clamp_cell, Cell};

#[derive(Debug, Clone)]
pub struct PcRocket {
    pub xpos: f64,
    pub ypos: f64,
    /// `+0xB0` -- type 18's `+0x90` under a different name.
    pub state: u8,
    /// `+0x24` / `+0x28`. -1 means released.
    pub controller: i32,
    pub controlled: bool,
    pub active: bool,
    pub projectile: Projectile,
    pub cell: Cell,
}

impl PcRocket {
    pub fn new(xcoord: i32, ycoord: i32) -> Self {
        let xpos = (xcoord * 6) as f64;
        let ypos = (ycoord * 6) as f64;
        PcRocket {
            xpos,
            ypos,
            state: STATE_LAUNCH,
            controller: 0,
            controlled: true,
            active: true,
            projectile: Projectile::default(),
            cell: clamp_cell((xpos / 24.0).floor() as i32, (ypos / 24.0).floor() as i32),
        }
    }

    pub fn think(&mut self, ctx: &mut EntityCtx) {
        if self.state == STATE_LAUNCH {
            let (xdir, ydir) = aim_at(self.xpos, self.ypos, ctx.ninja.xpos, ctx.ninja.ypos);
            self.projectile.launch(self.xpos, self.ypos, xdir, ydir);
            self.state = STATE_FLYING;
            return;
        }
        if self.state != STATE_FLYING {
            return;
        }

        // Steering: rotate the heading by the player's horizontal input.
        if self.controlled && ctx.ninja.hor_input != 0 {
            let angle = self.projectile.ydir.atan2(self.projectile.xdir)
                + ctx.ninja.hor_input as f64 * PC_ROCKET_TURN_RATE;
            self.projectile.xdir = angle.cos();
            self.projectile.ydir = angle.sin();
        }

        if self.projectile.advance(ctx.segment_grid, ctx.segments) {
            self.release();
        }
    }

    /// `FUN_1040F7F0`: tear down and hand the controller back.
    fn release(&mut self) {
        self.state = STATE_IDLE;
        self.controller = -1;
        self.controlled = false;
        self.active = false;
    }

    pub fn logical_collision(&mut self, ninja: &mut Ninja) -> bool {
        if self.state != STATE_FLYING || !ninja.is_valid_target() {
            return false;
        }
        if overlap_circle_vs_circle(
            self.projectile.xpos,
            self.projectile.ypos,
            ROCKET_RADIUS,
            ninja.xpos,
            ninja.ypos,
            NINJA_RADIUS,
        ) {
            ninja.kill(0, 0.0, 0.0, 0.0, 0.0, Some("pc_rocket"));
            self.release();
            return true;
        }
        false
    }

    pub fn reset_state(&mut self) {
        self.state = STATE_LAUNCH;
        self.controller = 0;
        self.controlled = true;
        self.active = true;
        self.projectile = Projectile::default();
    }
}
