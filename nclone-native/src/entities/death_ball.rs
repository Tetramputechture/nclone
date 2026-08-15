//! Death Ball Entity (Type 25).
//! Direct port of nclone/entity_classes/entity_death_ball.py.
//!
//! A homing hazard. It accelerates toward the ninja, is capped at 0.85 px/frame
//! by a soft drag, bounces off tile geometry, repels sibling death balls, and
//! kills (and recoils off) the ninja on contact.
//!
//! Two collision radii: 5 against the ninja, 8 against tiles and other balls.
//!
//! Ball-vs-ball repulsion mutates a SIBLING entity, which `EntityCtx` cannot
//! express, so `think` here does everything except that and
//! `Simulator::death_ball_think` runs the pairwise pass immediately afterwards
//! -- preserving Python's ordering, where ball *i* repels balls *i+1..* before
//! ball *i+1* takes its own turn.

use crate::collision::intersection::overlap_circle_vs_circle;
use crate::collision::sweep::{get_single_closest_point, sweep_circle_vs_tiles};
use crate::constants::{
    DEATH_BALL_ACCELERATION, DEATH_BALL_DRAG_MAX_SPEED, DEATH_BALL_DRAG_NO_TARGET,
    DEATH_BALL_MAX_SPEED, DEATH_BALL_RADIUS, DEATH_BALL_RADIUS2, NINJA_RADIUS,
};
use crate::entities::{SegmentGrid, EntityCtx};
use crate::ninja::Ninja;
use crate::segments::Segment;
use crate::types::{clamp_cell, Cell};

#[derive(Debug, Clone)]
pub struct DeathBall {
    pub xpos: f64,
    pub ypos: f64,
    pub xspeed: f64,
    pub yspeed: f64,
    /// Index among death balls only, used for the "repel every LATER ball" pass.
    pub index_in_type: usize,
    pub cell: Cell,
    initial_xpos: f64,
    initial_ypos: f64,
}

impl DeathBall {
    pub fn new(xcoord: i32, ycoord: i32, index_in_type: usize) -> Self {
        let xpos = (xcoord * 6) as f64;
        let ypos = (ycoord * 6) as f64;
        DeathBall {
            xpos,
            ypos,
            xspeed: 0.0,
            yspeed: 0.0,
            index_in_type,
            cell: clamp_cell((xpos / 24.0).floor() as i32, (ypos / 24.0).floor() as i32),
            initial_xpos: xpos,
            initial_ypos: ypos,
        }
    }

    /// `sqrt(a^2 + b^2)`. Every distance in this file goes through here.
    ///
    /// **Known hazard, deliberately NOT worked around.** Python spells all seven
    /// of these `math.sqrt(a**2 + b**2)`, and CPython's `**2` calls libm `pow`
    /// rather than emitting a multiply. `pow(x, 2)` and `x * x` are each
    /// correctly rounded but not to the same value -- they disagree on ~0.08% of
    /// doubles (164 of 200,000 sampled); `4.5000015184551483` squares to
    /// `...642` by pow and `...639` by multiply.
    ///
    /// So the two ports can differ by an ULP anywhere Python writes `**2`. That
    /// is a real divergence class across the whole codebase, not just here.
    ///
    /// Writing `a.powf(2.0)` does NOT fix it: LLVM folds `powf(2.0)` back into a
    /// multiply, so the emitted code is identical and `10225427` diverged by the
    /// same 9.66e-13 on the same frame 53. Matching CPython would mean forcing a
    /// real libm `pow` call, which is slower on a hot path and unverified against
    /// the game -- and the game itself, being compiled C++, near-certainly
    /// multiplies. Left as a multiply; see `docs/FULL_CORPUS_SWEEP.md` §15.
    #[inline]
    fn norm(a: f64, b: f64) -> f64 {
        (a * a + b * b).sqrt()
    }

    /// Seek, integrate, and resolve tile collisions. Everything in Python's
    /// `think` up to (but not including) the ball-vs-ball loop and `grid_move`,
    /// both of which the simulator drives.
    /// Returns **false** when the depenetration bailed out on `dist == 0`, i.e.
    /// the ball's centre landed exactly on a segment. Python `return`s from the
    /// whole of `think` there, which skips three things -- the bounce, the
    /// ball-vs-ball repulsion, and `grid_move` -- and the caller has to skip the
    /// last two on our behalf. See `simulator.rs::death_ball_think`.
    pub fn think_motion(
        &mut self,
        ninja: &Ninja,
        segment_grid: &SegmentGrid,
        segments: &[Segment],
    ) -> bool {
        if !ninja.is_valid_target() {
            // No target: coast to a halt.
            self.xspeed *= DEATH_BALL_DRAG_NO_TARGET;
            self.yspeed *= DEATH_BALL_DRAG_NO_TARGET;
        } else {
            let mut dx = ninja.xpos - self.xpos;
            let mut dy = ninja.ypos - self.ypos;
            let dist = Self::norm(dx, dy);
            if dist > 0.0 {
                dx /= dist;
                dy /= dist;
            }
            self.xspeed += dx * DEATH_BALL_ACCELERATION;
            self.yspeed += dy * DEATH_BALL_ACCELERATION;
            let speed = Self::norm(self.xspeed, self.yspeed);
            if speed > DEATH_BALL_MAX_SPEED {
                // Soft cap: only the EXCESS is damped, and a negligible excess is
                // dropped outright so the ball settles exactly at max speed.
                let mut new_speed = (speed - DEATH_BALL_MAX_SPEED) * DEATH_BALL_DRAG_MAX_SPEED;
                if new_speed <= 0.01 {
                    new_speed = 0.0;
                }
                new_speed += DEATH_BALL_MAX_SPEED;
                self.xspeed = self.xspeed / speed * new_speed;
                self.yspeed = self.yspeed / speed * new_speed;
            }
        }

        let xpos_old = self.xpos;
        let ypos_old = self.ypos;
        self.xpos += self.xspeed;
        self.ypos += self.yspeed;

        // Interpolate against tiles so a fast ball cannot tunnel.
        let time = sweep_circle_vs_tiles(
            segment_grid,
            segments,
            xpos_old,
            ypos_old,
            self.xspeed,
            self.yspeed,
            DEATH_BALL_RADIUS2 * 0.5,
        );
        self.xpos = xpos_old + time * self.xspeed;
        self.ypos = ypos_old + time * self.yspeed;

        // Depenetration, accumulating a summed contact normal.
        let mut xnormal = 0.0;
        let mut ynormal = 0.0;
        for _ in 0..16 {
            let (result, closest) =
                get_single_closest_point(segment_grid, segments, self.xpos, self.ypos, DEATH_BALL_RADIUS2);
            if result == 0 {
                break;
            }
            let Some((a, b)) = closest else { break };
            let dx = self.xpos - a;
            let dy = self.ypos - b;
            let dist = Self::norm(dx, dy);
            let depen_len = DEATH_BALL_RADIUS2 - dist * (result as f64);
            if depen_len < 0.0000001 {
                break;
            }
            if dist == 0.0 {
                // Python `return`s here -- abandoning the whole think, so no
                // bounce, no ball-vs-ball pass and NO `grid_move` this frame.
                // Not a `break`, and not merely a local return: the skipped
                // `grid_move` is the load-bearing part. A ball whose centre
                // lands exactly on a segment is stranded there for the rest of
                // the demo (the sweep then reports time 0 every frame, so it
                // never moves again), and because its cell is never updated it
                // stays registered where it spawned. The ninja gathers entities
                // by cell, so it never even tests the stranded ball.
                //
                // Updating the cell anyway makes those stranded balls lethal:
                // 26 of the 44 balls on `10150766` are registered in the wrong
                // cell, and native killed the ninja on frame 23 against one of
                // them while python walked past. That single difference is 889
                // demos of the 250k corpus.
                return false;
            }
            let xnorm = dx / dist;
            let ynorm = dy / dist;
            self.xpos += xnorm * depen_len;
            self.ypos += ynorm * depen_len;
            xnormal += xnorm;
            ynormal += ynorm;
        }

        // Bounce: reflect velocity out of the surface, harder above 1.35 px/frame.
        let normal_len = Self::norm(xnormal, ynormal);
        if normal_len > 0.0 {
            let dx = xnormal / normal_len;
            let dy = ynormal / normal_len;
            let dot_product = self.xspeed * dx + self.yspeed * dy;
            if dot_product < 0.0 {
                let speed = Self::norm(self.xspeed, self.yspeed);
                let bounce_strength = if speed <= 1.35 { 1.0 } else { 2.0 };
                self.xspeed -= dx * dot_product * bounce_strength;
                self.yspeed -= dy * dot_product * bounce_strength;
            }
        }
        true
    }

    /// Update the occupancy cell after the simulator has finished the frame's
    /// ball-vs-ball pass.
    pub fn finish_move(&mut self, ctx: &mut EntityCtx) {
        let new_cell = clamp_cell(
            (self.xpos / 24.0).floor() as i32,
            (self.ypos / 24.0).floor() as i32,
        );
        let old_cell = self.cell;
        self.cell = new_cell;
        ctx.grid_move(old_cell, new_cell);
    }

    /// Kill the ninja and recoil hard off it.
    pub fn logical_collision(&mut self, ninja: &mut Ninja) {
        if !ninja.is_valid_target() {
            return;
        }
        if overlap_circle_vs_circle(
            self.xpos,
            self.ypos,
            DEATH_BALL_RADIUS,
            ninja.xpos,
            ninja.ypos,
            NINJA_RADIUS,
        ) {
            let dx = self.xpos - ninja.xpos;
            let dy = self.ypos - ninja.ypos;
            let dist = Self::norm(dx, dy);
            self.xspeed += dx / dist * 10.0;
            self.yspeed += dy / dist * 10.0;
            ninja.kill(0, 0.0, 0.0, 0.0, 0.0, Some("death_ball"));
        }
    }

    pub fn reset_state(&mut self) {
        self.xpos = self.initial_xpos;
        self.ypos = self.initial_ypos;
        self.xspeed = 0.0;
        self.yspeed = 0.0;
        self.cell = clamp_cell(
            (self.xpos / 24.0).floor() as i32,
            (self.ypos / 24.0).floor() as i32,
        );
    }
}
