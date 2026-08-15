//! Rocket (Type 18) and the shared projectile integrator.
//! Port of nclone/entity_classes/entity_rocket.py + entity_projectile.py;
//! reconstructed from `npp.dll` per `docs/ENTITY_INTEGRATION_SPEC.md` §8.8.
//!
//! The largest single gate in the corpus: type 18 is on 15% of attract maps.
//!
//! **No runtime entity spawning is required.** The in-flight projectile is stored
//! as fields inside the launcher -- no allocation, no new entity, no lifetime
//! management. One rocket entity owns at most one live projectile. This overturned
//! the original guide's headline warning that the sim would need a projectile
//! subsystem before type 18 could land.
//!
//! **`move` (0x104116D0) is render-only and must not be ported**: it reads a frame
//! delta and `1/60`, the discriminator from spec §6.1. All simulation is in
//! `think`, and the launcher never moves.
//!
//! Type 29 (`EntityPlayerControlledRocket`) reuses [`Projectile`] wholesale --
//! every helper is shared -- and differs only in being steered by player input.
//! It is not map-placed (confirmed: zero occurrences in 8000 decoded attract
//! maps), so it has no loader arm; see `pc_rocket.rs`.

use crate::collision::intersection::overlap_circle_vs_circle;
use crate::collision::raycast::find_visible_ninja;
use crate::collision::sweep::sweep_circle_vs_tiles;
use crate::constants::{
    NINJA_RADIUS, ROCKET_INITIAL_STEP, ROCKET_LAUNCH_DELAY, ROCKET_LEAD_TIME,
    ROCKET_MAX_SPEED, ROCKET_MAX_TRAVEL, ROCKET_TURN_RATE,
    ROCKET_RADIUS, ROCKET_SPEED_MULTIPLIER,
};
use crate::entities::{EntityCtx, SegmentGrid};
use crate::ninja::Ninja;
use crate::segments::Segment;
use crate::types::{clamp_cell, Cell};

/// One in-flight rocket: position, unit direction, speed and odometer.
///
/// The speed model is an accumulator with a ceiling, not a geometric series --
/// `speed` starts at zero and gains a *step* that itself grows by 1.1^(2/3) a
/// frame, capped at `16/7`. See the Python twin in
/// `nclone/entity_classes/entity_projectile.py` for the derivation from
/// `docs/decomp/ROCKET_think_10412780.c`; `16/7` is the cap (`+0x20`), not the
/// initial speed, and the initial step is `+0x28 = 2/45`.
#[derive(Debug, Clone)]
pub struct Projectile {
    pub xpos: f64,
    pub ypos: f64,
    pub xdir: f64,
    pub ydir: f64,
    pub speed: f64,
    pub step: f64,
    pub travel: f64,
}

impl Default for Projectile {
    fn default() -> Self {
        Projectile {
            xpos: 0.0,
            ypos: 0.0,
            xdir: 1.0,
            ydir: 0.0,
            speed: 0.0,
            step: 0.0,
            travel: 0.0,
        }
    }
}

impl Projectile {
    pub fn launch(&mut self, xpos: f64, ypos: f64, xdir: f64, ydir: f64) {
        self.xpos = xpos;
        self.ypos = ypos;
        self.xdir = xdir;
        self.ydir = ydir;
        self.speed = 0.0;
        self.step = ROCKET_INITIAL_STEP;
        self.travel = 0.0;
    }

    /// Integrate one frame. Returns true if the projectile should detonate.
    pub fn advance(&mut self, segment_grid: &SegmentGrid, segments: &[Segment]) -> bool {
        if self.speed < ROCKET_MAX_SPEED {
            self.step *= ROCKET_SPEED_MULTIPLIER;
        }
        self.speed = (self.speed + self.step).min(ROCKET_MAX_SPEED);
        let step = self.speed;
        self.travel += step;

        let (xpos_old, ypos_old) = (self.xpos, self.ypos);
        let dx = self.xdir * step;
        let dy = self.ydir * step;
        // Sweep so a 40px/frame rocket cannot tunnel through a wall.
        let time = sweep_circle_vs_tiles(
            segment_grid,
            segments,
            xpos_old,
            ypos_old,
            dx,
            dy,
            ROCKET_RADIUS,
        );
        self.xpos = xpos_old + time * dx;
        self.ypos = ypos_old + time * dy;
        time < 1.0 || self.travel > ROCKET_MAX_TRAVEL
    }

    /// Rotate the heading toward where the ninja will be in `ROCKET_LEAD_TIME`
    /// frames. Proportional steering, run every frame after the move -- see the
    /// Python twin `ProjectileState.steer_toward`.
    pub fn steer_toward(&mut self, ninja: &Ninja) {
        let lead = ROCKET_LEAD_TIME;
        let dx = (ninja.xspeed * lead + ninja.xpos) - (self.xdir * self.speed * lead + self.xpos);
        let dy = (ninja.yspeed * lead + ninja.ypos) - (self.ydir * self.speed * lead + self.ypos);
        let length = (dx * dx + dy * dy).sqrt();
        if length == 0.0 {
            return;
        }
        let cross = (dy / length) * self.xdir + (dx / length) * -self.ydir;
        let xdir_new = -self.ydir * cross * ROCKET_TURN_RATE + self.xdir;
        let ydir_new = self.xdir * cross * ROCKET_TURN_RATE + self.ydir;
        self.xdir = xdir_new;
        self.ydir = ydir_new;
        let norm = (self.xdir * self.xdir + self.ydir * self.ydir).sqrt();
        if norm != 0.0 {
            self.xdir /= norm;
            self.ydir /= norm;
        } else {
            self.xdir = 0.0;
            self.ydir = 0.0;
        }
    }
}

/// Unit vector toward a point, with the game's degenerate-case fallback.
///
/// When the target direction has zero length the game falls back to `(1.0, 0.0)`.
/// That branch is real and is the guard behind the truncated assertion string
/// `"EntityRocket::think() found a player coincid…"` in `.rdata` (spec §8.8
/// fact 2), so it is reproduced rather than treated as unreachable.
pub fn aim_at(from_x: f64, from_y: f64, to_x: f64, to_y: f64) -> (f64, f64) {
    let dx = to_x - from_x;
    let dy = to_y - from_y;
    let length = (dx * dx + dy * dy).sqrt();
    if length != 0.0 {
        (dx / length, dy / length)
    } else {
        (1.0, 0.0)
    }
}

/// Launcher state. `+0x90` in the binary.
pub const STATE_IDLE: u8 = 0;
/// Type 18 charges here for `ROCKET_LAUNCH_DELAY` frames and launches on the
/// last one; type 29 is spawned straight into it and launches immediately,
/// which is why the two names share a value.
pub const STATE_CHARGING: u8 = 1;
pub const STATE_LAUNCH: u8 = STATE_CHARGING;
pub const STATE_FLYING: u8 = 2;

#[derive(Debug, Clone)]
pub struct Rocket {
    pub xpos: f64,
    pub ypos: f64,
    /// `+0x90`.
    pub state: u8,
    /// `+0x98`, the previous state. Render/audio only; kept so exports match.
    pub prev_state: u8,
    pub projectile: Projectile,
    /// `+0x88`, zeroed on entering LAUNCH. Charge toward the next launch.
    pub charge: f64,
    pub cell: Cell,
}

impl Rocket {
    pub fn new(xcoord: i32, ycoord: i32) -> Self {
        let xpos = (xcoord * 6) as f64;
        let ypos = (ycoord * 6) as f64;
        Rocket {
            xpos,
            ypos,
            state: STATE_IDLE,
            prev_state: STATE_IDLE,
            projectile: Projectile::default(),
            charge: 0.0,
            cell: clamp_cell((xpos / 24.0).floor() as i32,
                             (ypos / 24.0).floor() as i32),
        }
    }

    /// Three-state machine: IDLE acquires by sight in one frame, CHARGING counts
    /// `ROCKET_LAUNCH_DELAY` frames and launches on the last one, FLYING
    /// integrates and steers.
    ///
    /// **Only IDLE tests line of sight.** State 1 gates on `FUN_10513C30`, which
    /// asks whether the target is still valid, not whether it is still visible.
    /// Keep in step with `entity_rocket.py`.
    pub fn think(&mut self, ctx: &mut EntityCtx) {
        let mut prev = self.state;
        self.prev_state = prev;

        // LAUNCH falls through into FLYING on the SAME frame; see the arm below.
        // Written as a loop so the fall-through runs at most once.
        loop {
        match prev {
            STATE_IDLE => {
                // `FUN_1051E690` -- acquire. One frame, and it does NOT charge.
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
                self.state = STATE_CHARGING;
                self.charge = 0.0;
                return;
            }
            STATE_CHARGING => {
                // `FUN_10513C30` is a target-validity test, NOT a sight test:
                // once the launcher has acquired, the telegraph runs to
                // completion even if the ninja breaks line of sight. Re-testing
                // sight here and resetting the charge made every launch late by
                // however many frames the ninja spent behind cover -- nine of
                // them on `10134378`, a whole rocket-length of lead.
                if !ctx.ninja.is_valid_target() {
                    self.state = STATE_IDLE;
                    return;
                }
                self.charge += 1.0;
                if self.charge < ROCKET_LAUNCH_DELAY {
                    return;
                }
                // Aim at where the ninja is NOW -- `FUN_10513690` reads the live
                // position, with no lead. The port used to aim a think behind,
                // which `launchdir.py` had fitted at 88.6% over 105 captured
                // launches; that was an artefact of launching late, and once the
                // telegraph above was corrected the stale aim cost a demo
                // instead of saving one.
                let (xdir, ydir) = aim_at(self.xpos, self.ypos, ctx.ninja.xpos, ctx.ninja.ypos);
                self.projectile
                    .launch(self.xpos, self.ypos, xdir, ydir);
                self.state = STATE_FLYING;
                // The launch frame seeds and RETURNS -- `FUN_10412780`'s state-1
                // arm ends in `state = 2; return` and never touches the state-2
                // code. Confirmed against the captures: on the frame the game
                // enters state 2, `+0x78` still holds the raw ctor seed `2/45`,
                // so the integrator has not run. The projectile spends one frame
                // sitting on the tube. The fall-through this replaces was a
                // compensation for the late launch fixed in the arm above.
                //
                // `NCLONE_AB_ROCKET_FAST_LAUNCH=1` restores it.
                if !crate::ab::ab().rocket_fast_launch {
                    return;
                }
                prev = STATE_FLYING;
                continue;
            }
            _ => {
                if self.projectile.advance(ctx.segment_grid, ctx.segments) {
                    self.detonate();
                } else {
                    if ctx.ninja.is_valid_target() {
                        // Continuous homing, every frame after the move -- see
                        // the Python twin in `entity_rocket.py`.
                        self.projectile.steer_toward(ctx.ninja);
                    }
                    // THE KILL TEST LIVES HERE, not in `logical_collision`.
                    //
                    // `logical_collision` is only reached from
                    // `gather_entities_from_neighbourhood(entity_grid, ninja)`,
                    // which returns entities registered in the nine cells around
                    // the NINJA. A rocket registers at its LAUNCHER's cell --
                    // the projectile is a field inside the launcher, not an
                    // entity -- and native rebuilds `entity_grid` only at init
                    // and restore, never inside `tick`. So the dispatch fires
                    // only if the ninja walks up to the launcher itself.
                    //
                    // Measured before this existed: a projectile passed within
                    // 2.0 px of a stationary ninja (radius 10) over 600 frames
                    // with clear line of sight, and the ninja did not die.
                    //
                    // Turret (19) and DualLaser (23) already do exactly this --
                    // "their bodies do not collide, and their beams kill inside
                    // think" (spec §6.6). A rocket is the same shape of hazard:
                    // its lethal region is not at its body.
                    self.projectile_kill_test(ctx.ninja);
                }
            }
        }
        break;
        }
    }

    /// `FUN_10411400` is unread ([U]); returning to IDLE is the only transition
    /// that keeps the machine live, and it matches the shipped game re-firing
    /// after a rocket bursts on a wall.
    fn detonate(&mut self) {
        self.state = STATE_IDLE;
    }

    /// The PROJECTILE kills, not the launcher -- the binary's logical collision
    /// (0x104112B0) tests `+0x50…+0x5C`, which is projectile state.
    ///
    /// A discrete circle test, matching the binary: the projectile registers in the
    /// spatial grid and the ninja's collision pass calls this. So the original can
    /// tunnel a fast rocket past the ninja, and this reproduces that. Adding a
    /// swept kill test would make the sim MORE lethal than the game and start
    /// failing demos the real engine passed.
    /// Discrete circle test, projectile vs ninja; kills and detonates. Run every
    /// frame from `think` -- see the note there for why the grid dispatch cannot
    /// reach a flying projectile.
    pub fn projectile_kill_test(&mut self, ninja: &mut Ninja) -> bool {
        self.logical_collision(ninja)
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
            ninja.kill(0, 0.0, 0.0, 0.0, 0.0, Some("rocket"));
            self.detonate();
            return true;
        }
        false
    }

    pub fn reset_state(&mut self) {
        self.state = STATE_IDLE;
        self.prev_state = STATE_IDLE;
        self.projectile = Projectile::default();
        self.charge = 0.0;
    }
}
