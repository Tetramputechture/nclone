//! Evil Ninja (Type 22).
//! Port of nclone/entity_classes/entity_evil_ninja.py; reconstructed from
//! `npp.dll` per `docs/ENTITY_INTEGRATION_SPEC.md` §8.10.
//!
//! **The evil ninja is not an AI. It is a position-tape replayer.** No ninja state
//! machine, no physics, no steering -- it replays recorded ninja positions out of a
//! ring buffer the Simulator fills every frame. The original guide's reading
//! ("mirrors the player; probably reuses the whole ninja state machine; large but
//! mostly reuse") was wrong in mechanism and wrong in cost: the entity is trivial
//! and the work is the Simulator-side recorder (`Simulator::ninja_tape`), which
//! nothing else in the port needs and which is invisible from the entity side.
//!
//! Only the position is read out of each record; everything else in the game's
//! 0x78-byte snapshot is ignored by this entity.
//!
//! **Why the cursor starts at 0 and the wait is 120 frames:** the cursor trails
//! the recorder's write index, and `think` derives a render fade from `lag/120`.
//! Starting in WAITING with a 120-frame timer and a cursor of 0 makes the lag
//! exactly 120 on the first playback frame -- so the evil ninja retraces your path
//! from two seconds ago and the fade divisor is exactly the activation delay. That
//! self-consistency is the argument for this reading; the actual trigger (`+0x50`,
//! "recording source") is unread, as is any outgoing transition from state 3.

use crate::collision::intersection::overlap_circle_vs_circle;
use crate::constants::{
    EVIL_NINJA_FADE_DIVISOR, EVIL_NINJA_POST_FRAMES, EVIL_NINJA_RADIUS_ACTIVE,
    EVIL_NINJA_RADIUS_DORMANT, EVIL_NINJA_WAIT_FRAMES, NINJA_RADIUS,
};
use crate::entities::EntityCtx;
use crate::ninja::Ninja;
use crate::types::{clamp_cell, Cell};

pub const DORMANT: u8 = 0;
pub const WAITING: u8 = 1;
pub const PLAYBACK: u8 = 2;
pub const POST: u8 = 3;

#[derive(Debug, Clone)]
pub struct EvilNinja {
    pub xpos: f64,
    pub ypos: f64,
    /// `+0x20`. Spawn position, the restore target on reset and on POST.
    pub spawn_xpos: f64,
    pub spawn_ypos: f64,
    /// `+0x48`.
    pub state: u8,
    /// `+0x4C`.
    pub timer: i32,
    /// `+0x58`. Tape cursor; wraps with the recorder.
    pub cursor: usize,
    /// `+0x74`.
    pub post_timer: i32,
    /// `+0x40`. 10.0 dormant, 6.0 once active.
    pub radius: f64,
    /// `+0x64`. Render-only fade factor.
    pub fade: f64,
    pub cell: Cell,
}

impl EvilNinja {
    pub fn new(xcoord: i32, ycoord: i32) -> Self {
        let xpos = (xcoord * 6) as f64;
        let ypos = (ycoord * 6) as f64;
        EvilNinja {
            xpos,
            ypos,
            spawn_xpos: xpos,
            spawn_ypos: ypos,
            // DORMANT, not WAITING. All 151 captured evil ninjas carry
            // `+0x48 = 0` and `+0x4C = 120` on their first frame, without
            // exception; starting them armed made this port teleport them onto
            // the ninja's own path on frame 120 whether the game did or not.
            // The trigger out of DORMANT is still unread -- see the Python
            // class. `NCLONE_AB_EVIL_NINJA_ARMED=1` restores the old behaviour.
            state: if crate::ab::ab().evil_ninja_armed { WAITING } else { DORMANT },
            timer: EVIL_NINJA_WAIT_FRAMES,
            cursor: 0,
            post_timer: 0,
            radius: EVIL_NINJA_RADIUS_DORMANT,
            fade: 1.0,
            cell: clamp_cell((xpos / 24.0).floor() as i32, (ypos / 24.0).floor() as i32),
        }
    }

    /// `tape` and `write_index` come from the Simulator's recorder. `tape` is empty
    /// on maps that carry no evil ninja, which is the binary's "buffer gone" branch
    /// and triggers a full reset.
    pub fn think(&mut self, ctx: &mut EntityCtx, tape: &[(f64, f64)], write_index: usize) {
        if self.state == DORMANT {
            return;
        }
        if tape.is_empty() {
            self.reset_to_spawn();
            return;
        }
        if self.timer > 0 {
            self.timer -= 1;
        }

        if self.state == WAITING {
            if self.timer > 0 {
                return;
            }
            self.radius = EVIL_NINJA_RADIUS_ACTIVE;
            self.state = PLAYBACK;
            // falls through to PLAYBACK on this same frame, as the binary does
        }

        if self.state == PLAYBACK {
            if self.cursor != write_index {
                let lag = write_index as f64 - self.cursor as f64;
                if lag < EVIL_NINJA_FADE_DIVISOR {
                    self.fade = (lag / EVIL_NINJA_FADE_DIVISOR).clamp(0.0, 1.0);
                }
                let (x, y) = tape[self.cursor];
                self.xpos = x;
                self.ypos = y;
                self.cursor = if self.cursor + 1 > tape.len() - 1 {
                    0
                } else {
                    self.cursor + 1
                };
                self.update_cell(ctx);
                return;
            }
            // Tape exhausted: the replayer has caught up with the live ninja.
            self.state = POST;
            self.xpos = self.spawn_xpos;
            self.ypos = self.spawn_ypos;
            self.post_timer = EVIL_NINJA_POST_FRAMES;
            self.timer = EVIL_NINJA_WAIT_FRAMES;
            self.fade = 1.0;
            self.update_cell(ctx);
            return;
        }

        if self.state == POST {
            self.post_timer -= 1;
            if self.post_timer > 0 {
                return;
            }
            self.post_timer = 0;
            // Re-arm. State 3 has no read transition; the alternative reading is
            // that the entity parks at spawn forever once it catches up.
            self.timer = EVIL_NINJA_POST_FRAMES;
            self.state = WAITING;
            self.radius = EVIL_NINJA_RADIUS_DORMANT;
            self.fade = 1.0;
        }
    }

    fn update_cell(&mut self, ctx: &mut EntityCtx) {
        let new_cell = clamp_cell(
            (self.xpos / 24.0).floor() as i32,
            (self.ypos / 24.0).floor() as i32,
        );
        let old_cell = self.cell;
        self.cell = new_cell;
        ctx.grid_move(old_cell, new_cell);
    }

    /// Lethal on contact, at whichever radius the current state carries.
    pub fn logical_collision(&self, ninja: &mut Ninja) -> bool {
        if self.state != PLAYBACK || !ninja.is_valid_target() {
            return false;
        }
        if overlap_circle_vs_circle(
            self.xpos,
            self.ypos,
            self.radius,
            ninja.xpos,
            ninja.ypos,
            NINJA_RADIUS,
        ) {
            ninja.kill(0, 0.0, 0.0, 0.0, 0.0, Some("evil_ninja"));
            return true;
        }
        false
    }

    fn reset_to_spawn(&mut self) {
        self.radius = EVIL_NINJA_RADIUS_DORMANT;
        self.state = DORMANT;
        self.xpos = self.spawn_xpos;
        self.ypos = self.spawn_ypos;
        self.cursor = 0;
        self.timer = 0;
        self.post_timer = 0;
    }

    pub fn reset_state(&mut self) {
        self.xpos = self.spawn_xpos;
        self.ypos = self.spawn_ypos;
        self.state = WAITING;
        self.timer = EVIL_NINJA_WAIT_FRAMES;
        self.cursor = 0;
        self.post_timer = 0;
        self.radius = EVIL_NINJA_RADIUS_DORMANT;
        self.fade = 1.0;
        self.cell = clamp_cell(
            (self.xpos / 24.0).floor() as i32,
            (self.ypos / 24.0).floor() as i32,
        );
    }
}
