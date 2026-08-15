//! Bounce Block Entity (Type 17).
//! Direct port of the PHYSICS half of nclone/entity_classes/entity_bounce_block.py.
//!
//! A spring-mass square anchored to its spawn point: it is displaced by the
//! ninja, then pulled back by a Hooke force with per-frame dampening. On
//! contact it absorbs 80% of the depenetration into itself and returns 20% to
//! the ninja, which is what makes bounce-block boosting possible.
//!
//! Deliberately NOT ported: the Python class's `_update_data_tracking`,
//! `nearby_bounce_blocks` and `clearance_directions` machinery. That is RL
//! telemetry computed every frame over all entities -- it does not feed the
//! simulation, and running it in the hot loop would cost far more than the
//! physics.

use crate::collision::intersection::penetration_square_vs_point;
use crate::constants::{
    BOUNCE_BLOCK_DAMPENING, BOUNCE_BLOCK_SEMI_SIDE, BOUNCE_BLOCK_STIFFNESS,
    BOUNCE_BLOCK_STRENGTH, NINJA_RADIUS,
};
use crate::entities::EntityCtx;
use crate::ninja::Ninja;
use crate::types::{clamp_cell, Cell};

#[derive(Debug, Clone)]
pub struct BounceBlock {
    pub xpos: f64,
    pub ypos: f64,
    pub xspeed: f64,
    pub yspeed: f64,
    pub xorigin: f64,
    pub yorigin: f64,
    pub cell: Cell,
}

impl BounceBlock {
    pub fn new(xcoord: i32, ycoord: i32) -> Self {
        let xpos = (xcoord * 6) as f64;
        let ypos = (ycoord * 6) as f64;
        BounceBlock {
            xpos,
            ypos,
            xspeed: 0.0,
            yspeed: 0.0,
            xorigin: xpos,
            yorigin: ypos,
            cell: clamp_cell((xpos / 24.0).floor() as i32, (ypos / 24.0).floor() as i32),
        }
    }

    /// Spring step: dampen, integrate, then apply the restoring force to BOTH
    /// position and velocity.
    ///
    /// The force being added to position as well as velocity is not a mistake in
    /// the port -- Python does exactly this, which is why the block converges to
    /// its origin faster than a textbook spring would.
    pub fn move_entity(&mut self, ctx: &mut EntityCtx) {
        self.xspeed *= BOUNCE_BLOCK_DAMPENING;
        self.yspeed *= BOUNCE_BLOCK_DAMPENING;
        self.xpos += self.xspeed;
        self.ypos += self.yspeed;

        let xforce = BOUNCE_BLOCK_STIFFNESS * (self.xorigin - self.xpos);
        let yforce = BOUNCE_BLOCK_STIFFNESS * (self.yorigin - self.ypos);

        self.xpos += xforce;
        self.ypos += yforce;
        self.xspeed += xforce;
        self.yspeed += yforce;

        let new_cell = clamp_cell(
            (self.xpos / 24.0).floor() as i32,
            (self.ypos / 24.0).floor() as i32,
        );
        let old_cell = self.cell;
        self.cell = new_cell;
        ctx.grid_move(old_cell, new_cell);
    }

    /// Apply 80% of the depenetration to the block, hand 20% back to the ninja.
    pub fn physical_collision(&mut self, ninja: &Ninja) -> Option<((f64, f64), (f64, f64))> {
        let depen = penetration_square_vs_point(
            self.xpos,
            self.ypos,
            ninja.xpos,
            ninja.ypos,
            BOUNCE_BLOCK_SEMI_SIDE + NINJA_RADIUS,
        )?;
        let ((depen_x, depen_y), (depen_len, depen_other)) = depen;
        let share = depen_len * (1.0 - BOUNCE_BLOCK_STRENGTH);
        self.xpos -= depen_x * share;
        self.ypos -= depen_y * share;
        self.xspeed -= depen_x * share;
        self.yspeed -= depen_y * share;
        Some((
            (depen_x, depen_y),
            (depen_len * BOUNCE_BLOCK_STRENGTH, depen_other),
        ))
    }

    /// Wall-normal contribution when the ninja is against a side (note the extra
    /// 0.1 of reach compared to the physical test, so the ninja can wall-slide on
    /// a block it is not quite penetrating).
    pub fn logical_collision(&self, ninja: &Ninja) -> Option<f64> {
        penetration_square_vs_point(
            self.xpos,
            self.ypos,
            ninja.xpos,
            ninja.ypos,
            BOUNCE_BLOCK_SEMI_SIDE + NINJA_RADIUS + 0.1,
        )
        .map(|((depen_x, _), _)| depen_x)
    }

    pub fn reset_state(&mut self) {
        self.xpos = self.xorigin;
        self.ypos = self.yorigin;
        self.xspeed = 0.0;
        self.yspeed = 0.0;
        self.cell = clamp_cell(
            (self.xpos / 24.0).floor() as i32,
            (self.ypos / 24.0).floor() as i32,
        );
    }
}
