//! Turret (Type 19).
//! Port of nclone/entity_classes/entity_turret.py; reconstructed from `npp.dll`
//! per `docs/ENTITY_INTEGRATION_SPEC.md` §8.7. At `0x110` bytes this is the
//! largest entity in the game.
//!
//! **`move` (0x10419140) is render-only and must not be ported** -- it
//! interpolates a rotation using a frame delta and `1/60` (spec §6.1). All
//! simulation is in `think`.
//!
//! **The turret is hitscan.** It fires through the shared raycast helper
//! (`FUN_1050E0E0`, the same one the dual laser uses) and spawns no projectile, so
//! it needs no spawning machinery -- only raycasting. And it has **no logical
//! collision of its own** (spec §6.6, the `return 0` default), so its body is
//! harmless and damage is applied inside `think` at the moment of the shot.
//!
//! Two things a naive turret gets wrong:
//!   1. **The aim point is stateful and eased** -- it converges on the predicted
//!      position at a rate chosen by distance band, rather than being recomputed
//!      each frame. "Aim straight at the ninja" will not reproduce demos.
//!   2. **The shot leads the target** by `ninja_velocity * TURRET_LEAD_TIME`.
//!
//! `charge` does double duty (accumulator while tracking, fire timer while firing)
//! and is zeroed on the transition, exactly as `+0x98` does.
//!
//! The nine ctor constants are all real, but WHICH field each lands in was not
//! recovered -- see `constants.rs` for the inferred binding.

use crate::collision::raycast::{
    beam_hits_ninja, find_visible_ninja, get_raycast_distance, raycast_vs_player, seg_vs_circle,
};
use crate::constants::{
    NINJA_RADIUS, TURRET_AIM_CONVERGENCE, TURRET_BANDS, TURRET_CHARGE_RATES,
    TURRET_CHARGE_THRESHOLD, TURRET_FIRE_DELAY, TURRET_LEAD_TIME, TURRET_MUZZLE_TIMER,
    TURRET_RECOVER_DELAY,
};
use crate::entities::EntityCtx;
use crate::types::{clamp_cell, Cell};

pub const SEARCHING: u8 = 0;
pub const TRACKING: u8 = 1;
pub const FIRING: u8 = 2;
/// `+0xA4 == 3`. The state round 9 did not know about -- see `think`.
pub const RECOVERING: u8 = 3;

#[derive(Debug, Clone)]
pub struct Turret {
    pub xpos: f64,
    pub ypos: f64,
    pub orientation: u8,
    /// `+0xA4`.
    pub state: u8,
    pub has_target: bool,
    /// `+0x80`/`+0x88`, the eased aim point.
    pub aim_x: f64,
    pub aim_y: f64,
    /// `+0x90`.
    pub band: usize,
    /// `+0x98`. Charge while TRACKING, fire timer while FIRING and RECOVERING.
    pub charge: f64,
    /// `+0x108`. Whether the last shot connected. RECOVERING reads it: a turret
    /// that hit drops its target instead of resuming the track.
    pub hit_ninja: bool,
    /// `+0xF0`, decremented every think and clamped at 0.
    pub timer: i32,
    /// Last traced beam, for rendering.
    pub xend: f64,
    pub yend: f64,
    pub cell: Cell,
}

impl Turret {
    pub fn new(xcoord: i32, ycoord: i32, orientation: u8) -> Self {
        let xpos = (xcoord * 6) as f64;
        let ypos = (ycoord * 6) as f64;
        Turret {
            xpos,
            ypos,
            orientation,
            state: SEARCHING,
            has_target: false,
            aim_x: xpos,
            aim_y: ypos,
            band: 0,
            charge: 0.0,
            hit_ninja: false,
            timer: 0,
            xend: xpos,
            yend: ypos,
            cell: clamp_cell((xpos / 24.0).floor() as i32, (ypos / 24.0).floor() as i32),
        }
    }

    /// Port of the binary's band walk:
    /// `band = 0; i = 0; while (i < 3 && thresh[i] > d2) { i++; band = i; }`.
    ///
    /// `i` counts how many thresholds the error is inside, so **0 is the widest
    /// band and 3 the tightest**; the returned index is what the tables are read
    /// with.
    ///
    /// `+0x90` is zeroed *before* the walk, not only written from inside it.
    /// Round 9 read it the other way and had the field keep the previous frame's
    /// value when the error fell outside every threshold; the game's own struct
    /// shows it drop to 0 (`10138417`, frames 179 -> 180). Nothing reads the
    /// field, so this is a fidelity fix, not a behaviour change.
    fn select_band(&mut self, d2: f64) -> usize {
        self.band = 0;
        let mut i = 0usize;
        while i < 3 && TURRET_BANDS[i] > d2 {
            i += 1;
            self.band = i;
        }
        i
    }

    /// `FUN_104190B0`: the target is alive and the path to it is clear.
    fn target_visible(&self, ctx: &EntityCtx) -> bool {
        self.has_target
            && ctx.ninja.is_valid_target()
            && raycast_vs_player(
                ctx.segment_grid,
                ctx.segments,
                self.xpos,
                self.ypos,
                ctx.ninja.xpos,
                ctx.ninja.ypos,
            )
    }

    /// The four-state machine of `FUN_1041A150`.
    ///
    /// Round 9 modelled three states and had the turret drop to SEARCHING the
    /// instant it fired. The binary has a fourth, and the round-9 capture of
    /// `10138417` shows it plainly: the turret commits on frame 149, winds up
    /// through 163, **fires on 164** and moves to state 3, sits there 15 frames,
    /// and returns to TRACKING on 179 with its aim untouched.
    ///
    /// That last detail is the whole bug. Dropping to SEARCHING re-latches the
    /// aim onto the turret's own muzzle, and the muzzle is by definition close to
    /// anything the turret can see -- so the aim error collapses into band 3, the
    /// charge rate jumps to 3.5/frame, and the port re-fires roughly 26 frames
    /// after a shot. The game keeps the *stale* aim, which the ninja has by then
    /// run away from, so the error sits in band 0 where the charge rate is `0.0`
    /// and it cannot fire again until it has eased all the way back. On
    /// `10138417` the port is back to charge 26.5 by frame 200 while the game is
    /// still at 0.
    ///
    /// Round 9 read the same capture as "fires on 149, 164 and 179, every 15
    /// frames" and concluded a firing turret re-fires with no cooldown. Those are
    /// the 1->2, 2->3 and 3->1 transitions of one single shot.
    pub fn think(&mut self, ctx: &mut EntityCtx) {
        self.timer -= 1;
        if self.timer < 0 {
            self.timer = 0;
        }

        match self.state {
            SEARCHING => {
                if find_visible_ninja(
                    ctx.segment_grid,
                    ctx.segments,
                    self.xpos,
                    self.ypos,
                    ctx.ninja,
                )
                .is_none()
                {
                    return;
                }
                self.has_target = true;
                // Latch the aim origin to the turret's own position, so the first
                // frames of tracking swing out from the barrel rather than
                // teleporting onto the ninja.
                self.aim_x = self.xpos;
                self.aim_y = self.ypos;
                self.charge = 0.0;
                self.state = TRACKING;
            }
            TRACKING => {
                if !self.target_visible(ctx) {
                    self.state = SEARCHING;
                    self.has_target = false;
                    return;
                }
                // Lead the target, then ease the aim point toward the prediction.
                let dx = (ctx.ninja.xspeed * TURRET_LEAD_TIME + ctx.ninja.xpos) - self.aim_x;
                let dy = (ctx.ninja.yspeed * TURRET_LEAD_TIME + ctx.ninja.ypos) - self.aim_y;
                let d2 = dx * dx + dy * dy;
                let band = self.select_band(d2);
                self.charge += TURRET_CHARGE_RATES[band];
                let factor = TURRET_AIM_CONVERGENCE[band];
                self.aim_x += factor * dx;
                self.aim_y += factor * dy;
                if self.charge <= TURRET_CHARGE_THRESHOLD {
                    return;
                }
                self.state = FIRING;
                self.charge = 0.0;
            }
            FIRING => {
                // The aim FREEZES while the shot winds up -- the game's
                // +0x80/+0x88 hold still for the whole cycle. Converging it here
                // diverges at frame 150 instead of 165.
                self.charge += 1.0;
                if self.charge < TURRET_FIRE_DELAY {
                    return;
                }
                self.fire(ctx);
                self.charge = 0.0;
                if crate::ab::ab().turret_legacy {
                    self.state = SEARCHING;
                    self.has_target = false;
                    return;
                }
                self.state = RECOVERING;
                self.timer = TURRET_MUZZLE_TIMER;
            }
            _ => {
                self.charge += 1.0;
                if self.charge < TURRET_RECOVER_DELAY {
                    return;
                }
                // A turret that CONNECTED drops its target; one that missed
                // resumes tracking the same ninja, aim intact, from a zeroed
                // charge.
                if !self.target_visible(ctx) || self.hit_ninja {
                    self.state = SEARCHING;
                    self.has_target = false;
                    return;
                }
                self.hit_ninja = false;
                self.state = TRACKING;
                self.charge = 0.0;
            }
        }
    }

    /// Hitscan along the eased aim direction.
    ///
    /// Because the aim point is eased and leads the target, the ray genuinely
    /// misses when the ninja changes direction mid-charge. That is the intended
    /// behaviour, not something to correct by re-aiming at the last moment.
    ///
    /// The hit test is `seg_vs_circle` (`FUN_1050C690`), **not** the 11-limb
    /// `beam_hits_ninja` -- see `seg_vs_circle` for how the binary's callers
    /// split. Limbs are a strict subset of the circle, so the port was
    /// under-lethal per shot: the opposite sign to the state-machine bug in
    /// `think`, which is why neither showed up alone.
    fn fire(&mut self, ctx: &mut EntityCtx) {
        self.hit_ninja = false;
        if !ctx.ninja.is_valid_target() {
            return;
        }
        let mut dx = self.aim_x - self.xpos;
        let mut dy = self.aim_y - self.ypos;
        let length = (dx * dx + dy * dy).sqrt();
        if length == 0.0 {
            return;
        }
        dx /= length;
        dy /= length;
        let Some(ray_len) =
            get_raycast_distance(ctx.segment_grid, ctx.segments, self.xpos, self.ypos, dx, dy)
        else {
            self.xend = self.xpos;
            self.yend = self.ypos;
            return;
        };
        self.xend = self.xpos + dx * ray_len;
        self.yend = self.ypos + dy * ray_len;
        let hit = if crate::ab::ab().turret_limbs {
            beam_hits_ninja(self.xpos, self.ypos, self.xend, self.yend, ctx.ninja, Some(ray_len))
        } else {
            // `ray_len`, not `hypot(xend - xpos, yend - ypos)`: the game passes
            // the raycast's own return value as the segment length.
            seg_vs_circle(
                ctx.ninja.xpos,
                ctx.ninja.ypos,
                NINJA_RADIUS,
                self.xpos,
                self.ypos,
                self.xend,
                self.yend,
                ray_len,
            )
        };
        if hit {
            self.hit_ninja = true;
            ctx.ninja.kill(0, 0.0, 0.0, 0.0, 0.0, Some("turret"));
        }
    }

    pub fn reset_state(&mut self) {
        self.state = SEARCHING;
        self.has_target = false;
        self.aim_x = self.xpos;
        self.aim_y = self.ypos;
        self.band = 0;
        self.charge = 0.0;
        self.hit_ninja = false;
        self.timer = 0;
        self.xend = self.xpos;
        self.yend = self.ypos;
    }
}
