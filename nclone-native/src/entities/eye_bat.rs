//! Eye Bat (Type 27).
//! Port of nclone/entity_classes/entity_eye_bat.py; reconstructed from `npp.dll`
//! per `docs/ENTITY_INTEGRATION_SPEC.md` §8.2. This was the guide's "unknown
//! type 27".
//!
//! A flying hazard that homes on the ninja whenever it has line of sight and
//! coasts to a halt when it loses it. Structurally the death ball with a different
//! control law, so `death_ball.rs` is the template -- but three things differ in
//! kind, not just value:
//!
//!   * **`0.85` is a per-axis multiplicative decay, not a max speed.** The death
//!     ball's `0.85` is its speed cap; here the same number damps each velocity
//!     component independently while no target is visible.
//!   * **The cap is normalize-then-scale by `min(speed, 3.0)`**, not the death
//!     ball's soft "damp only the excess" drag.
//!   * **Two distinct radii**: `12.0` against the ninja (its own `.rdata` const
//!     block) and `5.0` against tiles. Do not conflate them.
//!
//! EyeBat is thinkable but NOT movable (spec §4), so position integration happens
//! inside `think`, after the velocity update and before tile depenetration.

use crate::collision::intersection::overlap_circle_vs_circle;
use crate::collision::raycast::find_visible_ninja;
use crate::collision::sweep::get_single_closest_point;
use crate::constants::{
    EYE_BAT_ACCELERATION, EYE_BAT_DRAG_NO_TARGET, EYE_BAT_MAX_CHASE_TIMER, EYE_BAT_MAX_SPEED,
    EYE_BAT_RADIUS, EYE_BAT_TILE_RADIUS, NINJA_RADIUS,
};
use crate::entities::{EntityCtx, SegmentGrid};
use crate::ninja::Ninja;
use crate::segments::Segment;
use crate::types::{clamp_cell, Cell};

/// Depenetration iteration cap. The binary's loop bound was unread; 16 is the
/// death ball's, and geometry needing more than 16 passes has already trapped the
/// entity.
const DEPEN_ITERATIONS: usize = 16;

#[derive(Debug, Clone)]
pub struct EyeBat {
    pub xpos: f64,
    pub ypos: f64,
    pub xspeed: f64,
    pub yspeed: f64,
    /// `+0x34`. Counts UP to 12 while chasing and DOWN to 0 while idle. What it
    /// gates is unread ([U]); it drives no physics.
    pub chase_timer: i32,
    /// `+0x38`. Set on any frame the bat had a target.
    pub chasing: bool,
    pub cell: Cell,
    initial_xpos: f64,
    initial_ypos: f64,
}

impl EyeBat {
    pub fn new(xcoord: i32, ycoord: i32) -> Self {
        let xpos = (xcoord * 6) as f64;
        let ypos = (ycoord * 6) as f64;
        // The ctor zeroes +0x20..+0x38 and holds no constants of its own -- every
        // number is read from .rdata at use time.
        EyeBat {
            xpos,
            ypos,
            xspeed: 0.0,
            yspeed: 0.0,
            chase_timer: 0,
            chasing: false,
            cell: clamp_cell((xpos / 24.0).floor() as i32, (ypos / 24.0).floor() as i32),
            initial_xpos: xpos,
            initial_ypos: ypos,
        }
    }

    /// Seek + accelerate + clamp, with decay on target loss, then integrate.
    ///
    /// `+0x30` (a float facing angle) is render state and is deliberately not a
    /// field here (spec §6.2).
    pub fn think(&mut self, ctx: &mut EntityCtx) {
        let visible = find_visible_ninja(
            ctx.segment_grid,
            ctx.segments,
            self.xpos,
            self.ypos,
            ctx.ninja,
        )
        .map(|n| (n.xpos, n.ypos));

        match visible {
            None => {
                self.chase_timer = (self.chase_timer - 1).max(0);
                self.xspeed *= EYE_BAT_DRAG_NO_TARGET;
                self.yspeed *= EYE_BAT_DRAG_NO_TARGET;
                self.chasing = false;
            }
            Some((nx, ny)) => {
                let mut dx = nx - self.xpos;
                let mut dy = ny - self.ypos;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist > 0.0 {
                    dx /= dist;
                    dy /= dist;
                }
                self.xspeed += dx * EYE_BAT_ACCELERATION;
                self.yspeed += dy * EYE_BAT_ACCELERATION;
                let speed = (self.xspeed * self.xspeed + self.yspeed * self.yspeed).sqrt();
                if speed > 0.0 {
                    // Normalize then scale by min(speed, 3.0): the identity below
                    // the cap, pinned at exactly 3.0 above it.
                    let clamp = speed.min(EYE_BAT_MAX_SPEED);
                    self.xspeed = clamp * self.xspeed / speed;
                    self.yspeed = clamp * self.yspeed / speed;
                }
                self.chase_timer = (self.chase_timer + 1).min(EYE_BAT_MAX_CHASE_TIMER);
                self.chasing = true;
            }
        }

        self.xpos += self.xspeed;
        self.ypos += self.yspeed;

        self.depenetrate(ctx.segment_grid, ctx.segments);

        let new_cell = clamp_cell(
            (self.xpos / 24.0).floor() as i32,
            (self.ypos / 24.0).floor() as i32,
        );
        let old_cell = self.cell;
        self.cell = new_cell;
        ctx.grid_move(old_cell, new_cell);
    }

    /// Push out of tile geometry at radius 5, then PROJECT the velocity onto the
    /// surface -- the bat does not bounce, but it does not keep driving into the
    /// wall either.
    ///
    /// Without the projection a bat that reaches a ceiling keeps its full
    /// approach speed and rams the same surface every frame forever: the
    /// depenetration puts it back at exactly `EYE_BAT_TILE_RADIUS` each time, so
    /// it is pinned, and a pinned bat sits wherever it stalled for the rest of
    /// the demo. On `10160699` bat 0 pins under a ceiling at frame 139 holding
    /// `yspeed = -2.656`; the game's `+0x28` reads **exactly 0.0** on that frame
    /// and 0.084 on the next. By frame 225 ours had not moved since and was
    /// sitting 20 px from the ninja against the game's 50, which is what killed
    /// the demo. See the Python twin in `entity_eye_bat.py`.
    fn depenetrate(&mut self, segment_grid: &SegmentGrid, segments: &[Segment]) {
        let mut xnormal = 0.0;
        let mut ynormal = 0.0;
        for _ in 0..DEPEN_ITERATIONS {
            let (result, closest) = get_single_closest_point(
                segment_grid,
                segments,
                self.xpos,
                self.ypos,
                EYE_BAT_TILE_RADIUS,
            );
            if result == 0 {
                break;
            }
            let Some((a, b)) = closest else { break };
            let dx = self.xpos - a;
            let dy = self.ypos - b;
            let dist = (dx * dx + dy * dy).sqrt();
            let depen_len = EYE_BAT_TILE_RADIUS - dist * (result as f64);
            if depen_len < 0.0000001 || dist == 0.0 {
                break;
            }
            let xnorm = dx / dist;
            let ynorm = dy / dist;
            self.xpos += xnorm * depen_len;
            self.ypos += ynorm * depen_len;
            xnormal += xnorm;
            ynormal += ynorm;
        }

        // Cancel the velocity component pointing INTO the surface: the death
        // ball's projection with `bounce_strength` fixed at 1, i.e. slide, never
        // rebound.
        let normal_len = (xnormal * xnormal + ynormal * ynormal).sqrt();
        if normal_len > 0.0 {
            let nx = xnormal / normal_len;
            let ny = ynormal / normal_len;
            let dot_product = self.xspeed * nx + self.yspeed * ny;
            if dot_product < 0.0 {
                self.xspeed -= nx * dot_product;
                self.yspeed -= ny * dot_product;
            }
        }
    }

    /// Lethal on contact.
    pub fn logical_collision(&self, ninja: &mut Ninja) -> bool {
        if !ninja.is_valid_target() {
            return false;
        }
        if overlap_circle_vs_circle(
            self.xpos,
            self.ypos,
            EYE_BAT_RADIUS,
            ninja.xpos,
            ninja.ypos,
            NINJA_RADIUS,
        ) {
            ninja.kill(0, 0.0, 0.0, 0.0, 0.0, Some("eye_bat"));
            return true;
        }
        false
    }

    pub fn reset_state(&mut self) {
        self.xpos = self.initial_xpos;
        self.ypos = self.initial_ypos;
        self.xspeed = 0.0;
        self.yspeed = 0.0;
        self.chase_timer = 0;
        self.chasing = false;
        self.cell = clamp_cell(
            (self.xpos / 24.0).floor() as i32,
            (self.ypos / 24.0).floor() as i32,
        );
    }
}
