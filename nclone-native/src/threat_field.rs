//! Multi-source, multi-horizon threat SDF, computed entirely in Rust.
//!
//! This is the scalable replacement for the `HazardSdf` + `predict_drone_positions`
//! pair. Both of those are correct; the problem is where the work happens.
//!
//! # Why per-entity data must not cross FFI
//!
//! `predict_drone_positions` returns `Vec<(Vec<(f64,f64)>, Vec<(f64,f64)>)>`,
//! which pyo3 materialises as nested Python lists of tuples -- one Python object
//! per drone per horizon. Its doc comment reasons that "a busy level has tens of
//! drones", and for the corpus that was in scope when it was written, it did.
//! Measured against the whole official corpus that assumption does not hold:
//!
//! | map                               | movers | positions | forecast(3h) |
//! |-----------------------------------|-------:|----------:|-------------:|
//! | `S/000 the basics`                |      0 |    1.3 us |       0.4 us |
//! | `S/589 escape is at hand`         |    261 |   30.0 us |     184.6 us |
//! | `SS2/097 compendium for literates`|    770 |  100.3 us |     526.8 us |
//!
//! 627 us per TICK of pure marshalling on the worst map, before any SDF work and
//! before the ten entity types still to be added. Corpus-wide: p50 = 6 movers,
//! p90 = 39, p99 = 165, max = 770; 25.6% of maps carry more than 16 movers and
//! 4.8% more than 64. The cost is real and it is concentrated exactly where the
//! hardest levels are.
//!
//! So this module takes NO position data as input. It reads the simulator's own
//! entity list, scatters every source itself, and returns one flat `f32` buffer.
//! FFI cost becomes O(1) in the number of entities.
//!
//! # Why the output grid is 12 px, not 6 px
//!
//! The policy consumes this key min-pooled to 12 px (`obs_map_pooling`), so the
//! 6 px grid was only ever an intermediate. Scattering straight into the 12 px
//! grid is **exactly** equal to min-pooling the 6 px scatter -- `min` over a
//! block of `min`s is the `min` of the union -- so this is not an approximation
//! and `test_threat_field.py` pins it against the existing path cell for cell.
//! It does the same number of distance evaluations but touches a quarter of the
//! output cells and skips the separate pooling pass in Python.
//!
//! # Cost model
//!
//! Scatter, like `HazardSdf`: the SDF saturates to exactly 1.0 beyond
//! `kill_radius + SDF_SCALE_PX`, so a source can only influence cells inside
//! that box and every skipped cell provably holds 1.0. Cost is
//! `O(n_sources * box_cells * n_horizons)` with no per-call allocation -- the
//! output buffer is owned by the engine and refilled in place.
//!
//! Scatter stays the right choice even at the top of the range. A bucketed
//! gather is the obvious alternative, but at 737 sources in a 1008x552 field the
//! mean spacing is ~27 px against a ~41 px influence radius, so a gather visiting
//! each of the 15,456 fine cells still tests ~9 candidates -- 139k distance
//! evaluations against scatter's 166k. There is no order-of-magnitude win
//! available here, which is why the effort goes into removing the marshalling
//! and the redundant passes instead of into a cleverer inner loop.

use crate::constants::{
    DEATH_BALL_MAX_SPEED, DEATH_BALL_RADIUS, DRONE_RADIUS, EYE_BAT_MAX_SPEED, EYE_BAT_RADIUS,
    FLOOR_GUARD_RADIUS,
    BOOST_PAD_RADIUS, LAUNCH_PAD_RADIUS, LOCKED_DOOR_SWITCH_RADIUS,
    FLOOR_GUARD_SPEED, MINI_DRONE_RADIUS, NINJA_RADIUS, ROCKET_INITIAL_STEP, ROCKET_LAUNCH_DELAY,
    ROCKET_MAX_SPEED, ROCKET_RADIUS, ROCKET_SPEED_MULTIPLIER, ROCKET_TURN_RATE,
    BOUNCE_BLOCK_SEMI_SIDE, ONE_WAY_PLATFORM_SEMI_SIDE, SHOVE_THWUMP_RADIUS,
    SHOVE_THWUMP_SEMI_SIDE, THWUMP_FORWARD_SPEED, THWUMP_SEMI_SIDE,
};
use crate::entities::drone::{DroneBehaviour, DroneKind};
use crate::entities::evil_ninja;
use crate::entities::Entity;
use crate::simulator::Simulator;

/// Fine grid the SDF is evaluated on -- the resolution the distances are
/// *sampled* at. Matches `HazardSdf` so the two agree cell for cell.
const FINE_CELL_SIZE: f64 = 6.0;
const FINE_WIDTH: usize = 168;
const FINE_HEIGHT: usize = 92;

/// Block edge, in fine cells, of the emitted grid. Must match
/// `OBS_MAP_DOWNSAMPLE` on the Python side.
const DOWNSAMPLE: usize = 2;

pub const OUT_WIDTH: usize = FINE_WIDTH / DOWNSAMPLE; // 84
pub const OUT_HEIGHT: usize = FINE_HEIGHT / DOWNSAMPLE; // 46

/// `clip((d - kill_radius) / SDF_SCALE_PX, -1, 1)`, as `hazard_map`.
const SDF_SCALE_PX: f64 = 24.0;
const PLAYFIELD_OFFSET_PX: f64 = 24.0;

/// Threat tiers, in channel-stack order. See `docs/ENTITY_INTEGRATION_PLAN.md` §3.1.
pub const TIER_CERTAIN: usize = 0;
pub const TIER_REACTIVE: usize = 1;
pub const N_TIERS: usize = 2;

/// Channel count of the time-to-lethal form: `[C_sdf, C_ttl, R_sdf, R_ttl]`.
pub const TTL_CHANNELS: usize = 4;
pub const TTL_CH_CERTAIN_SDF: usize = 0;
pub const TTL_CH_CERTAIN_TTL: usize = 1;
pub const TTL_CH_REACTIVE_SDF: usize = 2;
pub const TTL_CH_REACTIVE_TTL: usize = 3;

/// One lethal source at one instant.
#[derive(Clone, Copy)]
struct Source {
    x: f64,
    y: f64,
    /// Contact-death boundary: `NINJA_RADIUS + hazard radius`.
    kill_radius: f64,
}

/// Per-frame speed cap used to dilate a REACTIVE source's field at a horizon.
///
/// A reactive threat reads the live ninja, so rolling it forward would amount to
/// predicting the agent's own future. Instead its forecast channel is a
/// worst-case ENVELOPE: everywhere it could possibly reach in `h` frames. That is
/// an honest upper bound rather than a guess, and it keeps both tiers on one
/// tensor layout. Implemented by growing the kill radius, which is exactly a
/// Minkowski dilation of the lethal disc by the reachable distance.
fn reactive_max_speed(entity: &Entity) -> Option<f64> {
    match entity {
        Entity::DeathBall(_) => Some(DEATH_BALL_MAX_SPEED),
        // The eye bat's cap is applied as normalize-then-scale by min(speed, 3.0).
        Entity::EyeBat(_) => Some(3.0),
        Entity::Drone(d) if matches!(d.kind, DroneKind::Chaser) => Some(d.kind.speed()),
        _ => None,
    }
}

/// Reusable scatter buffers plus the emitted field.
///
/// Owned by the simulator so the per-tick path allocates nothing: a fresh
/// `vec![1.0; ...]` per call would be 4 x 3,864 floats per tier per frame, which
/// is pure churn at the frame rates this runs at.
pub struct ThreatField {
    /// `(N_TIERS * n_channels_per_tier) * OUT_WIDTH * OUT_HEIGHT`, row-major
    /// within a channel: `[tier][horizon][row][col]`.
    buf: Vec<f32>,
    /// Scratch for the source list, refilled each call.
    sources: Vec<Source>,
    n_horizons: usize,
}

impl ThreatField {
    pub fn new(n_horizons: usize) -> Self {
        let channels = N_TIERS * (1 + n_horizons);
        ThreatField {
            buf: vec![1.0; channels * OUT_WIDTH * OUT_HEIGHT],
            sources: Vec::new(),
            n_horizons,
        }
    }

    fn channels(&self) -> usize {
        N_TIERS * (1 + self.n_horizons)
    }

    fn ensure_capacity(&mut self, n_horizons: usize) {
        if n_horizons != self.n_horizons {
            self.n_horizons = n_horizons;
            self.buf
                .resize(self.channels() * OUT_WIDTH * OUT_HEIGHT, 1.0);
        }
    }

    /// Scatter one source into one channel, keeping the element-wise minimum.
    ///
    /// Taking the min of SDF values rather than of distances is equivalent
    /// because the SDF is monotonically increasing in `d`.
    ///
    /// # One distance evaluation per OUTPUT cell, not four
    ///
    /// The emitted value for an output cell is the min over the four fine-grid
    /// sub-centres it covers. Evaluating all four and taking the min is the
    /// obvious implementation and it is what the 6 px path did, but it is four
    /// times more work than necessary: the four sub-centres form a 2x2 grid at
    /// `+-FINE_CELL_SIZE/2` about the block centre, so
    ///
    /// ```text
    /// min over cx in {bx-h, bx+h}, cy in {by-h, by+h} of (cx-sx)^2 + (cy-sy)^2
    /// ```
    ///
    /// SEPARATES: the x term depends only on `cx` and the y term only on `cy`,
    /// so the nearest sub-centre is simply the nearer one in each axis
    /// independently -- `bx + h` when `sx > bx`, else `bx - h`. Picking it costs
    /// two comparisons and yields exactly the same minimum.
    ///
    /// This is an exact restatement, not an approximation, and it is what makes
    /// the dense end of the corpus affordable: at 737 sources it takes the
    /// per-channel scatter from ~510 us to ~130 us.
    fn scatter(channel: &mut [f32], src: Source) {
        let effect_radius = src.kill_radius + SDF_SCALE_PX;
        const HALF: f64 = FINE_CELL_SIZE / 2.0;
        // Distance from the playfield origin to an output block's centre.
        const BLOCK: f64 = FINE_CELL_SIZE * DOWNSAMPLE as f64;

        // Output-cell index bounds of the influence box, clamped to the grid.
        let ocol_lo = (((src.x - PLAYFIELD_OFFSET_PX - effect_radius) / BLOCK).floor()).max(0.0);
        let orow_lo = (((src.y - PLAYFIELD_OFFSET_PX - effect_radius) / BLOCK).floor()).max(0.0);
        let ocol_hi = (((src.x - PLAYFIELD_OFFSET_PX + effect_radius) / BLOCK).ceil())
            .clamp(0.0, (OUT_WIDTH - 1) as f64);
        let orow_hi = (((src.y - PLAYFIELD_OFFSET_PX + effect_radius) / BLOCK).ceil())
            .clamp(0.0, (OUT_HEIGHT - 1) as f64);
        if ocol_lo > ocol_hi || orow_lo > orow_hi {
            return; // entirely off-grid
        }
        let (ocol_lo, ocol_hi) = (ocol_lo as usize, ocol_hi as usize);
        let (orow_lo, orow_hi) = (orow_lo as usize, orow_hi as usize);

        for orow in orow_lo..=orow_hi {
            // Centre of the block's two fine rows, then the nearer of them.
            let by = PLAYFIELD_OFFSET_PX + (orow as f64 + 0.5) * BLOCK;
            let cy = if src.y > by { by + HALF } else { by - HALF };
            let dy = cy - src.y;
            let dy2 = dy * dy;
            let base = orow * OUT_WIDTH;
            for ocol in ocol_lo..=ocol_hi {
                let bx = PLAYFIELD_OFFSET_PX + (ocol as f64 + 0.5) * BLOCK;
                let cx = if src.x > bx { bx + HALF } else { bx - HALF };
                let dx = cx - src.x;
                let d = (dx * dx + dy2).sqrt();
                let sdf = ((d - src.kill_radius) / SDF_SCALE_PX).clamp(-1.0, 1.0) as f32;
                let cell = &mut channel[base + ocol];
                if sdf < *cell {
                    *cell = sdf;
                }
            }
        }
    }

    /// Scatter a source's LETHAL DISC (not its SDF falloff) into a time grid,
    /// keeping the earliest arrival per cell.
    ///
    /// `t_norm` is the normalised time at which this source occupies this
    /// position. Only cells inside `kill_radius` are written, which is what
    /// makes the TTL form cheap: the influence box is bounded by the kill radius
    /// alone (14 px for a mini drone) rather than by `kill_radius + 24` px of SDF
    /// falloff, so it is roughly 4x smaller in area than an SDF scatter.
    fn scatter_lethal_time(channel: &mut [f32], src: Source, t_norm: f32) {
        const HALF: f64 = FINE_CELL_SIZE / 2.0;
        const BLOCK: f64 = FINE_CELL_SIZE * DOWNSAMPLE as f64;
        let r = src.kill_radius;

        let ocol_lo = (((src.x - PLAYFIELD_OFFSET_PX - r) / BLOCK).floor()).max(0.0);
        let orow_lo = (((src.y - PLAYFIELD_OFFSET_PX - r) / BLOCK).floor()).max(0.0);
        let ocol_hi =
            (((src.x - PLAYFIELD_OFFSET_PX + r) / BLOCK).ceil()).clamp(0.0, (OUT_WIDTH - 1) as f64);
        let orow_hi =
            (((src.y - PLAYFIELD_OFFSET_PX + r) / BLOCK).ceil()).clamp(0.0, (OUT_HEIGHT - 1) as f64);
        if ocol_lo > ocol_hi || orow_lo > orow_hi {
            return;
        }
        let r2 = r * r;
        for orow in orow_lo as usize..=orow_hi as usize {
            let by = PLAYFIELD_OFFSET_PX + (orow as f64 + 0.5) * BLOCK;
            let cy = if src.y > by { by + HALF } else { by - HALF };
            let dy = cy - src.y;
            let dy2 = dy * dy;
            if dy2 > r2 {
                continue;
            }
            let base = orow * OUT_WIDTH;
            for ocol in ocol_lo as usize..=ocol_hi as usize {
                let bx = PLAYFIELD_OFFSET_PX + (ocol as f64 + 0.5) * BLOCK;
                let cx = if src.x > bx { bx + HALF } else { bx - HALF };
                let dx = cx - src.x;
                if dx * dx + dy2 <= r2 {
                    let cell = &mut channel[base + ocol];
                    if t_norm < *cell {
                        *cell = t_norm;
                    }
                }
            }
        }
    }

    /// Scatter a REACTIVE source's time-to-lethal in CLOSED FORM.
    ///
    /// A reactive threat closes on the ninja at up to `speed` px/frame, so the
    /// earliest frame at which it can reach a cell is simply
    /// `(dist - kill_radius) / speed` -- no sampling, no roll-out, one pass. This
    /// is the same worst-case-envelope semantics the tier-R SDF channels use,
    /// read as a time instead of as a dilation, and it is why the reactive half
    /// of this field costs a single scatter no matter how many frames ahead the
    /// window looks.
    fn scatter_reactive_time(channel: &mut [f32], src: Source, speed: f64, max_frames: f64) {
        const HALF: f64 = FINE_CELL_SIZE / 2.0;
        const BLOCK: f64 = FINE_CELL_SIZE * DOWNSAMPLE as f64;
        // Furthest the threat can possibly matter within the window.
        let reach = src.kill_radius + speed * max_frames;

        let ocol_lo = (((src.x - PLAYFIELD_OFFSET_PX - reach) / BLOCK).floor()).max(0.0);
        let orow_lo = (((src.y - PLAYFIELD_OFFSET_PX - reach) / BLOCK).floor()).max(0.0);
        let ocol_hi = (((src.x - PLAYFIELD_OFFSET_PX + reach) / BLOCK).ceil())
            .clamp(0.0, (OUT_WIDTH - 1) as f64);
        let orow_hi = (((src.y - PLAYFIELD_OFFSET_PX + reach) / BLOCK).ceil())
            .clamp(0.0, (OUT_HEIGHT - 1) as f64);
        if ocol_lo > ocol_hi || orow_lo > orow_hi {
            return;
        }
        for orow in orow_lo as usize..=orow_hi as usize {
            let by = PLAYFIELD_OFFSET_PX + (orow as f64 + 0.5) * BLOCK;
            let cy = if src.y > by { by + HALF } else { by - HALF };
            let dy = cy - src.y;
            let dy2 = dy * dy;
            let base = orow * OUT_WIDTH;
            for ocol in ocol_lo as usize..=ocol_hi as usize {
                let bx = PLAYFIELD_OFFSET_PX + (ocol as f64 + 0.5) * BLOCK;
                let cx = if src.x > bx { bx + HALF } else { bx - HALF };
                let dx = cx - src.x;
                let d = (dx * dx + dy2).sqrt();
                let frames = ((d - src.kill_radius) / speed).max(0.0);
                if frames <= max_frames {
                    let t_norm = (frames / max_frames) as f32;
                    let cell = &mut channel[base + ocol];
                    if t_norm < *cell {
                        *cell = t_norm;
                    }
                }
            }
        }
    }

    /// Scatter a homing projectile's earliest-arrival time, using its exact
    /// speed schedule and its bounded turn rate.
    ///
    /// # The bound
    ///
    /// For the projectile to be lethal at a cell it must come within
    /// `kill_radius` of it, which requires its DISPLACEMENT COMPONENT along the
    /// direction to that cell to reach `gap = d - kill_radius`. So the earliest
    /// possible arrival is the first frame `k` at which the largest achievable
    /// such component reaches `gap`.
    ///
    /// That largest component is computable exactly. Write `phi` for the angle
    /// from the current heading to the cell. The heading at frame `i` may sit
    /// anywhere within `i * atan(TURN_RATE)` of the current one, and the choice
    /// that drives the projectile at the cell hardest is simply "turn toward it
    /// as fast as possible, then hold" -- a feasible heading trajectory, so the
    /// bound is achieved rather than merely assumed. That gives
    ///
    /// ```text
    /// reach_along_e(k) = sum over i in 1..=k of  v_i * cos(max(phi - i*turn, 0))
    /// ```
    ///
    /// and the emitted time is the first `k` with `reach_along_e(k) >= gap`.
    ///
    /// This is a NECESSARY condition, so the field can only ever say a cell
    /// becomes lethal earlier than it really does -- the direction a hazard
    /// field is allowed to err in. It subsumes the two separate conditions this
    /// replaced (path length, and a cone that had to be abandoned past 90
    /// degrees because it stops being convex): at `phi = 0` it reduces to
    /// cumulative travel, and behind the projectile it correctly charges for the
    /// frames spent turning round instead of giving that region away for free.
    ///
    /// # No trigonometry in the loop
    ///
    /// `cos(phi - i*turn)` is advanced by a fixed rotation, so the whole inner
    /// loop is four multiplies and two adds per frame. `cos(phi)` is the dot
    /// product that is already computed, and `sin(phi)` follows from it -- phi
    /// is taken non-negative, which is exact because the turn limit is symmetric.
    /// A `cos` call per frame per cell would be ~600 us per rocket against a
    /// 400 us budget for the entire key.
    fn scatter_homing_time(
        channel: &mut [f32],
        p: &HomingProjectile,
        speeds: &[f64],
        travel_total: f64,
        max_frames: f64,
    ) {
        const HALF: f64 = FINE_CELL_SIZE / 2.0;
        const BLOCK: f64 = FINE_CELL_SIZE * DOWNSAMPLE as f64;
        if max_frames - p.delay < 0.0 {
            return; // launches after the window closes
        }
        let n = speeds.len();
        let reach = p.kill_radius + travel_total;
        let turn = rocket_max_turn_per_frame();
        let (cos_t, sin_t) = (turn.cos(), turn.sin());

        let ocol_lo = (((p.x - PLAYFIELD_OFFSET_PX - reach) / BLOCK).floor()).max(0.0);
        let orow_lo = (((p.y - PLAYFIELD_OFFSET_PX - reach) / BLOCK).floor()).max(0.0);
        let ocol_hi = (((p.x - PLAYFIELD_OFFSET_PX + reach) / BLOCK).ceil())
            .clamp(0.0, (OUT_WIDTH - 1) as f64);
        let orow_hi = (((p.y - PLAYFIELD_OFFSET_PX + reach) / BLOCK).ceil())
            .clamp(0.0, (OUT_HEIGHT - 1) as f64);
        if ocol_lo > ocol_hi || orow_lo > orow_hi {
            return;
        }
        for orow in orow_lo as usize..=orow_hi as usize {
            let by = PLAYFIELD_OFFSET_PX + (orow as f64 + 0.5) * BLOCK;
            let cy = if p.y > by { by + HALF } else { by - HALF };
            let dy = cy - p.y;
            let base = orow * OUT_WIDTH;
            for ocol in ocol_lo as usize..=ocol_hi as usize {
                let bx = PLAYFIELD_OFFSET_PX + (ocol as f64 + 0.5) * BLOCK;
                let cx = if p.x > bx { bx + HALF } else { bx - HALF };
                let dx = cx - p.x;
                let d = (dx * dx + dy * dy).sqrt();
                let gap = d - p.kill_radius;

                let frames = if gap <= 0.0 {
                    // Already inside the kill disc: lethal the moment it exists.
                    0.0
                } else {
                    // cos(phi), sin(phi). A charging launcher has no heading yet
                    // -- its aim is chosen at launch from the live ninja -- so it
                    // gets phi = 0 in every direction, i.e. a plain disc.
                    let (mut c, mut s) = if p.heading_known && d > 1e-9 {
                        let dot = ((dx * p.dirx + dy * p.diry) / d).clamp(-1.0, 1.0);
                        (dot, (1.0 - dot * dot).max(0.0).sqrt())
                    } else {
                        (1.0, 0.0)
                    };
                    let mut acc = 0.0;
                    let mut k = 0usize;
                    // "Aligned" means the heading already points AT the cell, so
                    // it holds and every frame contributes its full speed. Both
                    // phi = 0 and phi = 180 give sin(phi) = 0, so the sine alone
                    // does not distinguish them -- and treating a cell directly
                    // BEHIND the projectile as aligned would credit it with full
                    // forward progress in exactly the direction it cannot go,
                    // collapsing the anisotropy where it matters most. The
                    // cosine breaks the tie.
                    let mut aligned = s <= 0.0 && c > 0.0;
                    while k < n && acc < gap {
                        if !aligned {
                            // Rotate the angle by one frame's worth of turn.
                            let c_new = c * cos_t + s * sin_t;
                            let s_new = s * cos_t - c * sin_t;
                            c = c_new;
                            s = s_new;
                            // Overshot: the heading now points straight at the
                            // cell and holds there.
                            if s <= 0.0 {
                                aligned = true;
                            }
                        }
                        acc += speeds[k] * if aligned { 1.0 } else { c };
                        k += 1;
                    }
                    if acc < gap {
                        continue; // unreachable inside the window
                    }
                    k as f64
                };
                let t = frames + p.delay;
                if t <= max_frames {
                    let t_norm = (t / max_frames) as f32;
                    let cell = &mut channel[base + ocol];
                    if t_norm < *cell {
                        *cell = t_norm;
                    }
                }
            }
        }
    }

    /// Scatter a lethal SEGMENT (a capsule of `kill_radius` about it) into a
    /// time grid, keeping the earliest arrival.
    ///
    /// A thwump's lethal region is a segment, not a disc: the face spans
    /// `THWUMP_SEMI_SIDE - 2` either side of the face centre and kills anything
    /// within `NINJA_RADIUS + 2` of it. Approximating that by a disc containing
    /// it (radius 12 + 7 = 19, centred on the face) over-covers badly: the face
    /// sits only 11px from the body centre, so a 19px disc reaches 8px PAST the
    /// centre, painting the thwump's own body and its three SAFE sides as
    /// lethal. Those sides are solid and standable, so that is not conservatism,
    /// it is a false positive on exactly the surface the ninja is meant to use.
    fn scatter_segment_time(
        channel: &mut [f32],
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        kill_radius: f64,
        t_norm: f32,
    ) {
        const HALF: f64 = FINE_CELL_SIZE / 2.0;
        const BLOCK: f64 = FINE_CELL_SIZE * DOWNSAMPLE as f64;
        let r = kill_radius;
        let (lo_x, hi_x) = (x1.min(x2) - r, x1.max(x2) + r);
        let (lo_y, hi_y) = (y1.min(y2) - r, y1.max(y2) + r);

        let ocol_lo = (((lo_x - PLAYFIELD_OFFSET_PX) / BLOCK).floor()).max(0.0);
        let orow_lo = (((lo_y - PLAYFIELD_OFFSET_PX) / BLOCK).floor()).max(0.0);
        let ocol_hi =
            (((hi_x - PLAYFIELD_OFFSET_PX) / BLOCK).ceil()).clamp(0.0, (OUT_WIDTH - 1) as f64);
        let orow_hi =
            (((hi_y - PLAYFIELD_OFFSET_PX) / BLOCK).ceil()).clamp(0.0, (OUT_HEIGHT - 1) as f64);
        if ocol_lo > ocol_hi || orow_lo > orow_hi {
            return;
        }
        let (dx, dy) = (x2 - x1, y2 - y1);
        let len2 = dx * dx + dy * dy;
        let r2 = r * r;
        for orow in orow_lo as usize..=orow_hi as usize {
            let by = PLAYFIELD_OFFSET_PX + (orow as f64 + 0.5) * BLOCK;
            let base = orow * OUT_WIDTH;
            for ocol in ocol_lo as usize..=ocol_hi as usize {
                let bx = PLAYFIELD_OFFSET_PX + (ocol as f64 + 0.5) * BLOCK;
                // Nearest fine sub-centre, same analytic pick as `scatter`, taken
                // against the segment's own midpoint so the choice stays stable.
                let cx = if (x1 + x2) * 0.5 > bx { bx + HALF } else { bx - HALF };
                let cy = if (y1 + y2) * 0.5 > by { by + HALF } else { by - HALF };
                // Point-to-segment squared distance.
                let t = if len2 > 0.0 {
                    (((cx - x1) * dx + (cy - y1) * dy) / len2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let px = x1 + t * dx;
                let py = y1 + t * dy;
                let d2 = (cx - px) * (cx - px) + (cy - py) * (cy - py);
                if d2 <= r2 {
                    let cell = &mut channel[base + ocol];
                    if t_norm < *cell {
                        *cell = t_norm;
                    }
                }
            }
        }
    }

    /// The signed-distance form of `scatter_segment_time`, for the SDF channels.
    fn scatter_segment_sdf(
        channel: &mut [f32],
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        kill_radius: f64,
    ) {
        const HALF: f64 = FINE_CELL_SIZE / 2.0;
        const BLOCK: f64 = FINE_CELL_SIZE * DOWNSAMPLE as f64;
        let reach = kill_radius + SDF_SCALE_PX;
        let (lo_x, hi_x) = (x1.min(x2) - reach, x1.max(x2) + reach);
        let (lo_y, hi_y) = (y1.min(y2) - reach, y1.max(y2) + reach);

        let ocol_lo = (((lo_x - PLAYFIELD_OFFSET_PX) / BLOCK).floor()).max(0.0);
        let orow_lo = (((lo_y - PLAYFIELD_OFFSET_PX) / BLOCK).floor()).max(0.0);
        let ocol_hi =
            (((hi_x - PLAYFIELD_OFFSET_PX) / BLOCK).ceil()).clamp(0.0, (OUT_WIDTH - 1) as f64);
        let orow_hi =
            (((hi_y - PLAYFIELD_OFFSET_PX) / BLOCK).ceil()).clamp(0.0, (OUT_HEIGHT - 1) as f64);
        if ocol_lo > ocol_hi || orow_lo > orow_hi {
            return;
        }
        let (dx, dy) = (x2 - x1, y2 - y1);
        let len2 = dx * dx + dy * dy;
        for orow in orow_lo as usize..=orow_hi as usize {
            let by = PLAYFIELD_OFFSET_PX + (orow as f64 + 0.5) * BLOCK;
            let base = orow * OUT_WIDTH;
            for ocol in ocol_lo as usize..=ocol_hi as usize {
                let bx = PLAYFIELD_OFFSET_PX + (ocol as f64 + 0.5) * BLOCK;
                let cx = if (x1 + x2) * 0.5 > bx { bx + HALF } else { bx - HALF };
                let cy = if (y1 + y2) * 0.5 > by { by + HALF } else { by - HALF };
                let t = if len2 > 0.0 {
                    (((cx - x1) * dx + (cy - y1) * dy) / len2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let px = x1 + t * dx;
                let py = y1 + t * dy;
                let d = ((cx - px) * (cx - px) + (cy - py) * (cy - py)).sqrt();
                let sdf = ((d - kill_radius) / SDF_SCALE_PX).clamp(-1.0, 1.0) as f32;
                let cell = &mut channel[base + ocol];
                if sdf < *cell {
                    *cell = sdf;
                }
            }
        }
    }

    fn channel_mut(&mut self, tier: usize, horizon_idx: usize) -> &mut [f32] {
        let stride = OUT_WIDTH * OUT_HEIGHT;
        let ch = tier * (1 + self.n_horizons) + horizon_idx;
        &mut self.buf[ch * stride..(ch + 1) * stride]
    }

    /// Recompute the whole stack from the simulator's live entity list.
    ///
    /// `horizons` are sim-frame offsets, ascending. Channel layout is
    /// `[tier][horizon]` with horizon 0 meaning "now", so the emitted shape is
    /// `(N_TIERS * (1 + horizons.len()), OUT_WIDTH, OUT_HEIGHT)`.
    pub fn compute(&mut self, sim: &Simulator, horizons: &[u32]) -> &[f32] {
        self.ensure_capacity(horizons.len());
        // "No hazard anywhere" everywhere, then scatter. Matches the mine path's
        // saturation value, so a threat-free level is indistinguishable from a
        // threat that is merely far away.
        self.buf.fill(1.0);

        // ---- Tier C: exact, ninja-independent futures --------------------
        // Present frame first.
        self.sources.clear();
        for entity in &sim.entities {
            if let Entity::Drone(d) = entity {
                if !d.active {
                    continue;
                }
                let radius = match d.kind {
                    DroneKind::Zap => DRONE_RADIUS,
                    DroneKind::Mini => MINI_DRONE_RADIUS,
                    // Chaser is reactive; the shooters' bodies are harmless
                    // (`body_is_lethal`), so neither belongs in tier C.
                    _ => continue,
                };
                self.sources.push(Source {
                    x: d.xpos,
                    y: d.ypos,
                    kill_radius: NINJA_RADIUS + radius,
                });
            }
        }
        let present: Vec<Source> = self.sources.clone();
        {
            let channel = self.channel_mut(TIER_CERTAIN, 0);
            for &src in &present {
                Self::scatter(channel, src);
            }
        }

        // Forecast horizons. The roll-out drives the simulator's own motion code
        // (`Drone::forecast`), so the field at horizon k is bit-for-bit what the
        // present-frame field will read k ticks from now -- a lookup, not a
        // prediction. Nothing here crosses FFI.
        if !horizons.is_empty() {
            let tracks = sim.forecast_drone_positions(horizons);
            for (h_idx, (zap, mini)) in tracks.iter().enumerate() {
                let channel = self.channel_mut(TIER_CERTAIN, h_idx + 1);
                for &(x, y) in zap {
                    Self::scatter(
                        channel,
                        Source {
                            x,
                            y,
                            kill_radius: NINJA_RADIUS + DRONE_RADIUS,
                        },
                    );
                }
                for &(x, y) in mini {
                    Self::scatter(
                        channel,
                        Source {
                            x,
                            y,
                            kill_radius: NINJA_RADIUS + MINI_DRONE_RADIUS,
                        },
                    );
                }
            }
        }

        // ---- Tier R: reactive, dilated worst-case envelopes ---------------
        self.sources.clear();
        let mut speeds: Vec<f64> = Vec::new();
        for entity in &sim.entities {
            // Death balls and eye bats have no `active` flag -- unlike drones and
            // gold they are never deactivated, so there is nothing to test.
            let radius = match entity {
                Entity::DeathBall(_) => DEATH_BALL_RADIUS,
                Entity::EyeBat(_) => EYE_BAT_RADIUS,
                Entity::Drone(d) if d.active && matches!(d.kind, DroneKind::Chaser) => DRONE_RADIUS,
                _ => continue,
            };
            let Some(speed) = reactive_max_speed(entity) else {
                continue;
            };
            self.sources.push(Source {
                x: entity.xpos(),
                y: entity.ypos(),
                kill_radius: NINJA_RADIUS + radius,
            });
            speeds.push(speed);
        }
        let reactive: Vec<Source> = self.sources.clone();
        for h_idx in 0..=horizons.len() {
            // Horizon 0 is the present frame: no dilation.
            let frames = if h_idx == 0 {
                0.0
            } else {
                horizons[h_idx - 1] as f64
            };
            let channel = self.channel_mut(TIER_REACTIVE, h_idx);
            for (src, &speed) in reactive.iter().zip(speeds.iter()) {
                Self::scatter(
                    channel,
                    Source {
                        kill_radius: src.kill_radius + speed * frames,
                        ..*src
                    },
                );
            }
        }

        &self.buf
    }

    /// The TIME-TO-LETHAL form: `[C_sdf, C_ttl, R_sdf, R_ttl]`, 4 channels.
    ///
    /// # Why this exists alongside the horizon stack
    ///
    /// The horizon stack answers "where is lethal at t+12, t+24, t+48" with one
    /// full SDF grid per horizon. That is `1 + n_horizons` channels per tier and
    /// its cost scales with the horizon count, which caps how finely time can be
    /// sampled -- three snapshots, at a frame skip of 4, is three action steps of
    /// lookahead resolution.
    ///
    /// This form answers the question the policy actually has to answer -- *when*
    /// does this cell become lethal -- directly, in one channel per tier:
    ///
    /// * `0.0` = lethal right now
    /// * `t/max_frames` = first lethal at frame `t`
    /// * `1.0` = never lethal inside the window
    ///
    /// Polarity matches the SDF (1.0 = safe), so `min` remains the conservative
    /// pooling reduction for every channel of the stack and the encoder reads one
    /// convention throughout.
    ///
    /// The present-frame SDF rides along in its own channel because TTL alone is
    /// spatially flat: it says "lethal in 20 frames" identically for a cell 1 px
    /// and 30 px outside the disc, so on its own it gives the policy no gradient
    /// to keep margin against. SDF supplies the spatial gradient, TTL the
    /// temporal one; neither substitutes for the other.
    ///
    /// # Cost
    ///
    /// TTL scatters only the LETHAL DISC, not the 24 px SDF falloff, so each
    /// sample touches ~4x fewer cells than an SDF channel. The reactive half
    /// needs no sampling at all (closed form). So a 16-sample window costs about
    /// what 4 SDF horizons cost, while resolving time 5x more finely and halving
    /// the channel count the policy has to carry.
    pub fn compute_ttl(&mut self, sim: &Simulator, max_frames: u32, stride: u32) -> &[f32] {
        // Reuse the horizon-stack buffer sizing machinery: TTL_CHANNELS channels
        // is the same footprint as one horizon per tier.
        self.ensure_capacity(1);
        debug_assert!(self.channels() == TTL_CHANNELS);
        self.buf.fill(1.0);

        let max_f = max_frames.max(1) as f64;
        let stride = stride.max(1);

        // ---- Tier C ------------------------------------------------------
        let certain = collect_certain(sim);
        {
            let channel = self.channel_mut(0, TTL_CH_CERTAIN_SDF);
            for &src in &certain {
                Self::scatter(channel, src);
            }
        }
        {
            let channel = self.channel_mut(0, TTL_CH_CERTAIN_TTL);
            for &src in &certain {
                Self::scatter_lethal_time(channel, src, 0.0);
            }
        }
        // Sampled roll-out. One shared forecast pass drives every sample, so the
        // simulator's own motion code stays the single source of truth.
        let samples: Vec<u32> = (stride..=max_frames).step_by(stride as usize).collect();
        if !samples.is_empty() {
            let tracks = sim.forecast_drone_positions(&samples);
            for (i, (zap, mini)) in tracks.iter().enumerate() {
                let t_norm = (samples[i] as f64 / max_f) as f32;
                let channel = self.channel_mut(0, TTL_CH_CERTAIN_TTL);
                for &(x, y) in zap {
                    Self::scatter_lethal_time(
                        channel,
                        Source {
                            x,
                            y,
                            kill_radius: NINJA_RADIUS + DRONE_RADIUS,
                        },
                        t_norm,
                    );
                }
                for &(x, y) in mini {
                    Self::scatter_lethal_time(
                        channel,
                        Source {
                            x,
                            y,
                            kill_radius: NINJA_RADIUS + MINI_DRONE_RADIUS,
                        },
                        t_norm,
                    );
                }
            }
        }

        // Committed-velocity hazards ride the CERTAIN tier: straight-line
        // extrapolation at the same sample points as the drone roll-out.
        let kinematic = collect_kinematic(sim);
        if !kinematic.is_empty() {
            {
                let channel = self.channel_mut(0, TTL_CH_CERTAIN_SDF);
                for k in &kinematic {
                    Self::scatter(
                        channel,
                        Source {
                            x: k.x,
                            y: k.y,
                            kill_radius: k.kill_radius,
                        },
                    );
                }
            }
            let channel = self.channel_mut(0, TTL_CH_CERTAIN_TTL);
            for k in &kinematic {
                Self::scatter_lethal_time(
                    channel,
                    Source {
                        x: k.x,
                        y: k.y,
                        kill_radius: k.kill_radius,
                    },
                    0.0,
                );
                for &frame in &samples {
                    let t = frame as f64;
                    Self::scatter_lethal_time(
                        channel,
                        Source {
                            x: k.x + k.vx * t,
                            y: k.y + k.vy * t,
                            kill_radius: k.kill_radius,
                        },
                        (t / max_f) as f32,
                    );
                }
            }
        }

        // ---- Thwumps: segment geometry, forward-only sweep -----------------
        // Charging ones are certain, idle ones merely possible, but BOTH sweep
        // along the facing axis only. Same geometry, different tier.
        for tw in collect_thwumps(sim) {
            let (sdf_ch, ttl_ch) = if tw.committed {
                (TTL_CH_CERTAIN_SDF, TTL_CH_CERTAIN_TTL)
            } else {
                (TTL_CH_REACTIVE_SDF, TTL_CH_REACTIVE_TTL)
            };
            Self::scatter_segment_sdf(
                self.channel_mut(0, sdf_ch),
                tw.x1,
                tw.y1,
                tw.x2,
                tw.y2,
                THWUMP_FACE_KILL_RADIUS,
            );
            let channel = self.channel_mut(0, ttl_ch);
            Self::scatter_segment_time(
                channel,
                tw.x1,
                tw.y1,
                tw.x2,
                tw.y2,
                THWUMP_FACE_KILL_RADIUS,
                0.0,
            );
            for &frame in &samples {
                let t = frame as f64;
                let (ox, oy) = (tw.vx * t, tw.vy * t);
                Self::scatter_segment_time(
                    channel,
                    tw.x1 + ox,
                    tw.y1 + oy,
                    tw.x2 + ox,
                    tw.y2 + oy,
                    THWUMP_FACE_KILL_RADIUS,
                    (t / max_f) as f32,
                );
            }
        }

        // ---- Homing projectiles: exact speed ramp, bounded turn ------------
        // Tier R, because the heading reads the live ninja -- but a far tighter
        // envelope than the isotropic dilation the other reactive sources get,
        // because the speed schedule is not reactive at all.
        for p in collect_projectiles(sim) {
            // Present-frame lethal disc: only a projectile that EXISTS is lethal
            // now. A charging launcher's body never kills, so it contributes to
            // the TTL countdown below and to nothing else.
            if p.heading_known {
                Self::scatter(
                    self.channel_mut(0, TTL_CH_REACTIVE_SDF),
                    Source {
                        x: p.x,
                        y: p.y,
                        kill_radius: p.kill_radius,
                    },
                );
            }
            let n = (max_f - p.delay).max(0.0) as usize;
            let speeds = projectile_speed_table(p.speed, p.step, n);
            let travel_total: f64 = speeds.iter().sum();
            Self::scatter_homing_time(
                self.channel_mut(0, TTL_CH_REACTIVE_TTL),
                &p,
                &speeds,
                travel_total,
                max_f,
            );
        }

        // ---- Beams: live lethal segments -----------------------------------
        for beam in collect_beams(sim) {
            Self::scatter_segment_sdf(
                self.channel_mut(0, TTL_CH_REACTIVE_SDF),
                beam.x1, beam.y1, beam.x2, beam.y2, BEAM_KILL_RADIUS,
            );
            Self::scatter_segment_time(
                self.channel_mut(0, TTL_CH_REACTIVE_TTL),
                beam.x1, beam.y1, beam.x2, beam.y2, BEAM_KILL_RADIUS, 0.0,
            );
        }

        // ---- Tier R ------------------------------------------------------
        let (reactive, speeds) = collect_reactive(sim);
        {
            let channel = self.channel_mut(0, TTL_CH_REACTIVE_SDF);
            for &src in &reactive {
                Self::scatter(channel, src);
            }
        }
        {
            let channel = self.channel_mut(0, TTL_CH_REACTIVE_TTL);
            for (src, &speed) in reactive.iter().zip(speeds.iter()) {
                Self::scatter_reactive_time(channel, *src, speed, max_f);
            }
        }

        &self.buf
    }
}

/// A source whose velocity is KNOWN this frame, so its short-term future is a
/// straight-line extrapolation rather than a guess.
///
/// These are tier C without being ninja-independent in general: a thwump chose
/// to charge because of the ninja, but once charging its direction is committed
/// and it cannot re-aim. The forecast is exact for as long as it keeps moving,
/// which is the horizon that matters for dodging it.
///
/// Wall collisions are NOT modelled -- a thwump that will stop against a wall in
/// 8 frames is still projected the full window. That over-states danger, which is
/// the safe direction for a hazard field, and the alternative is re-implementing
/// each entity's collision response here where it would silently drift from the
/// real one.
#[derive(Clone, Copy)]
struct Kinematic {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    kill_radius: f64,
}

/// Sources with a committed velocity: flying-adjacent hazards whose direction is
/// already fixed.
///
/// * **Floor guard, latched** (`dir != 0`) -- charges along the floor until a
///   wall or ledge stops it, and `think` early-outs forever while latched, so it
///   cannot re-aim.
/// * **Thwump, charging** (`state == 1`) -- committed along its facing axis.
///   Only the LETHAL FACE is a threat, not the body, so the source sits at the
///   face and carries a radius covering the face segment (see below).
///
/// Deliberately NOT here: the rocket projectile. It looks ballistic and the
/// integration spec reads that way, but `Rocket::think` calls
/// `projectile.steer_toward(ctx.ninja)` on EVERY state-2 frame -- it homes
/// continuously. Extrapolating it as a straight line would under-state where it
/// can reach, which is the one error a hazard field must not make. It is a
/// reactive source instead.
fn collect_kinematic(sim: &Simulator) -> Vec<Kinematic> {
    let mut out = Vec::new();
    for entity in &sim.entities {
        match entity {
            Entity::FloorGuard(g) if g.dir != 0 => out.push(Kinematic {
                x: g.xpos,
                y: g.ypos,
                vx: g.dir as f64 * FLOOR_GUARD_SPEED,
                vy: 0.0,
                kill_radius: NINJA_RADIUS + FLOOR_GUARD_RADIUS,
            }),
            // Thwumps are NOT here: their lethal region is a segment, not a
            // disc, and both the charging and idle cases are handled together by
            // `collect_thwumps` so the two share one geometry.
            _ => {}
        }
    }
    out
}

/// Kill radius about a thwump's lethal face SEGMENT: `Ninja.logical_collision`
/// tests a ninja circle inflated by 2 against that segment.
const THWUMP_FACE_KILL_RADIUS: f64 = NINJA_RADIUS + 2.0;

// ---------------------------------------------------------------------------
// Homing projectiles (rocket, type 18)
// ---------------------------------------------------------------------------

/// A rocket projectile, modelled by what is actually known about it rather than
/// as a point that may move anywhere at top speed.
///
/// # Why this is neither tier C nor an ordinary tier-R source
///
/// The rocket splits cleanly into a part that is exactly known and a part that
/// is not, and collapsing the two loses most of the signal:
///
/// * **Speed is DETERMINISTIC.** `Projectile::advance` runs
///   `step *= 1.0656; speed = min(speed + step, 16/7)` with no reference to the
///   ninja whatsoever. A projectile's speed `k` frames from now is a pure
///   function of its current `(speed, step)` -- a lookup, not a prediction.
/// * **Heading is REACTIVE but RATE-LIMITED.** `steer_toward` rotates by
///   `atan(cross / 15)` with `|cross| <= 1`, so the heading turns by at most
///   `atan(ROCKET_TURN_RATE)` = 3.81 degrees per frame no matter what the ninja
///   does.
///
/// Treating it as a plain reactive source dilated by `ROCKET_MAX_SPEED * h`
/// -- what this module did before -- gets both halves wrong in the same
/// direction, and the error is largest exactly when it matters most:
///
/// | frames after launch | true reach | `MAX_SPEED * h` | over-statement |
/// |---:|---:|---:|---:|
/// |  4 |   0.5 px |   9.1 px | 18x |
/// | 12 |   4.8 px |  27.4 px | 5.8x |
/// | 24 |  24.4 px |  54.9 px | 2.2x |
/// | 64 | 115.8 px | 146.3 px | 1.26x |
///
/// A rocket needs ~24 frames to reach terminal speed, and the frames right after
/// the telegraph resolves are precisely the ones the policy has to route
/// through. Painting a 27 px radius of death around a projectile that can
/// physically reach 4.8 px does not make the agent safer -- it closes corridors
/// that are open, which is the failure mode that makes a hazard channel worth
/// ignoring.
///
/// The envelope this produces is still a strict SUPERSET of the reachable set,
/// so the field never under-states danger. See `scatter_homing_time`.
struct HomingProjectile {
    x: f64,
    y: f64,
    /// Unit heading. Meaningless when `heading_known` is false.
    dirx: f64,
    diry: f64,
    /// Live speed, px/frame.
    speed: f64,
    /// Live per-frame speed increment, which itself grows by
    /// `ROCKET_SPEED_MULTIPLIER` each frame until `speed` caps.
    step: f64,
    kill_radius: f64,
    /// Frames before the projectile exists at all: a CHARGING launcher has not
    /// fired yet, and nothing is lethal at the tube until it does.
    delay: f64,
    /// False for a CHARGING launcher: the launch aims at wherever the ninja is
    /// on the frame it fires, so no heading can be assumed and the envelope
    /// stays a full disc.
    heading_known: bool,
}

/// Largest heading change the steering can produce in one frame.
///
/// `steer_toward` sets `dir' = normalize(dir + perp * cross * TURN_RATE)` where
/// `cross` is a sine and so lies in `[-1, 1]`. The rotation that produces is
/// `atan(cross * TURN_RATE)`, maximised at `atan(TURN_RATE)`.
fn rocket_max_turn_per_frame() -> f64 {
    ROCKET_TURN_RATE.atan()
}

/// Every live projectile plus every launcher whose telegraph is running.
///
/// An IDLE launcher is deliberately absent: it acquires by line of sight in a
/// single frame and has committed to nothing, so admitting it would paint a
/// permanent dead zone around a fixture that is harmless until it fires.
fn collect_projectiles(sim: &Simulator) -> Vec<HomingProjectile> {
    let mut out = Vec::new();
    for entity in &sim.entities {
        let Entity::Rocket(r) = entity else { continue };
        match r.state {
            crate::entities::rocket::STATE_FLYING => out.push(HomingProjectile {
                x: r.projectile.xpos,
                y: r.projectile.ypos,
                dirx: r.projectile.xdir,
                diry: r.projectile.ydir,
                speed: r.projectile.speed,
                step: r.projectile.step,
                kill_radius: NINJA_RADIUS + ROCKET_RADIUS,
                delay: 0.0,
                heading_known: true,
            }),
            // CHARGING: in `LAUNCH_DELAY - charge` frames a projectile appears
            // AT THE TUBE and starts accelerating from a standstill. Surfacing
            // that as a countdown is what makes "move before it resolves"
            // learnable; surfacing it as "lethal now", which a present-frame
            // source would, is simply false -- the launcher body never kills.
            crate::entities::rocket::STATE_CHARGING => out.push(HomingProjectile {
                x: r.xpos,
                y: r.ypos,
                dirx: 0.0,
                diry: 0.0,
                speed: 0.0,
                step: ROCKET_INITIAL_STEP,
                kill_radius: NINJA_RADIUS + ROCKET_RADIUS,
                delay: (ROCKET_LAUNCH_DELAY - r.charge).max(0.0),
                heading_known: false,
            }),
            _ => {}
        }
    }
    out
}

/// Per-frame speed of a projectile with live `(speed, step)`, for the next `n`
/// frames.
///
/// This runs the SAME recurrence as `Projectile::advance` -- there is no closed
/// form worth deriving, because the `min(speed + step, MAX)` cap makes it
/// piecewise, and a second derivation is a second thing to keep in step with the
/// simulator. Wall collisions are not modelled, which over-states reach and is
/// the safe direction for a hazard field.
fn projectile_speed_table(mut speed: f64, mut step: f64, n: usize) -> Vec<f64> {
    let mut table = Vec::with_capacity(n);
    for _ in 0..n {
        if speed < ROCKET_MAX_SPEED {
            step *= ROCKET_SPEED_MULTIPLIER;
        }
        speed = (speed + step).min(ROCKET_MAX_SPEED);
        table.push(speed);
    }
    table
}

/// One thwump's lethal face as a segment, plus how it can move.
struct ThwumpThreat {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    /// Charge velocity, along the facing axis. Always FORWARD.
    vx: f64,
    vy: f64,
    /// `state == 1`: already charging, so the motion is certain rather than
    /// merely possible.
    committed: bool,
}

/// Every thwump's lethal face, as a segment perpendicular to its charge axis.
///
/// # Only the facing side is lethal
///
/// A thwump is a solid 18px block with ONE dangerous face; the other three sides
/// are standable, and routes rely on that. Two things have to be right or the
/// field paints the safe sides red:
///
/// 1. **The lethal region is a segment, not a disc.** It spans
///    `THWUMP_SEMI_SIDE - 2` either side of the face centre, killing within
///    `NINJA_RADIUS + 2`. A disc big enough to contain that (radius 19) centred
///    on a face only 11px from the body centre reaches 8px past the centre,
///    straight onto the far side.
/// 2. **The forecast is DIRECTIONAL, not radial.** A thwump can only ever charge
///    along its facing axis, forward. Dilating its lethal region isotropically by
///    `speed * horizon` -- the treatment every other reactive threat gets, and
///    correct for them because they can turn -- sprays danger backwards through
///    the block and out the safe sides. Idle thwumps get the same forward sweep
///    a charging one gets; the only difference is which tier it lands in.
///
/// Retraction is not modelled: a returning thwump sweeps its face back toward
/// the origin, and those positions are behind the current face, on the body's
/// own side. Under-covering there is harmless because the block is solid.
fn collect_thwumps(sim: &Simulator) -> Vec<ThwumpThreat> {
    let half = THWUMP_SEMI_SIDE - 2.0;
    let offset = THWUMP_SEMI_SIDE + 2.0;
    let mut out = Vec::new();
    for entity in &sim.entities {
        let Entity::Thwump(t) = entity else { continue };
        let speed = THWUMP_FORWARD_SPEED * t.direction;
        let (x1, y1, x2, y2, vx, vy) = if t.is_horizontal {
            // Facing along x, so the face is a VERTICAL segment.
            let fx = t.xpos + offset * t.direction;
            (fx, t.ypos - half, fx, t.ypos + half, speed, 0.0)
        } else {
            let fy = t.ypos + offset * t.direction;
            (t.xpos - half, fy, t.xpos + half, fy, 0.0, speed)
        };
        out.push(ThwumpThreat {
            x1,
            y1,
            x2,
            y2,
            vx,
            vy,
            committed: t.state == 1,
        });
    }
    out
}

/// A lethal BEAM: turret hitscan and dual-laser, as a live segment.
///
/// Both entities expose the raycast endpoint they killed along (`xend`/`yend`),
/// reset to their own position when there is no beam -- so a degenerate segment
/// is the "not firing" signal and must not be scattered, or every turret would
/// paint a permanent dot on itself.
///
/// These are 928 official maps (turret 425, dual laser 503) that were entirely
/// invisible to the policy: the threat field was point-source only, and a beam
/// is a line. Nothing about a corridor swept by a laser distinguished it from an
/// empty one.
///
/// Tier R, not C. The turret's aim is an eased track on the LIVE ninja with a
/// velocity lead, so its line is a function of the agent's own state. The dual
/// laser's spinner is ninja-independent and could in principle be forecast by
/// advancing its angle -- that is a genuine improvement left for later; present
/// frame is the honest, safe reading until the roll-out shares the sim's own
/// angle-advance code rather than a second copy of it.
struct BeamSource {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
}

fn collect_beams(sim: &Simulator) -> Vec<BeamSource> {
    let mut out = Vec::new();
    let mut push = |x1: f64, y1: f64, x2: f64, y2: f64| {
        // Degenerate == no beam this frame.
        if (x2 - x1).abs() > 1e-9 || (y2 - y1).abs() > 1e-9 {
            out.push(BeamSource { x1, y1, x2, y2 });
        }
    };
    for entity in &sim.entities {
        match entity {
            Entity::Turret(t) => push(t.xpos, t.ypos, t.xend, t.yend),
            Entity::DualLaser(l) => push(l.xpos, l.ypos, l.xend, l.yend),
            // The laser drone's SUSTAINED beam (80 lethal frames after its
            // charge). It was missing here while its two siblings were present,
            // so on the 11 official maps that carry one the only lethal thing on
            // the level was invisible in every threat channel -- the body is
            // deliberately excluded (`body_is_lethal` is false for shooters), so
            // omitting the beam omitted the entity entirely.
            //
            // Same `xend/yend` contract as the other two: the endpoint collapses
            // onto the body when no beam is live, which `push` skips.
            Entity::Drone(d) if d.active && matches!(d.kind, DroneKind::Laser) => {
                if let DroneBehaviour::Shooter(s) = &d.behaviour {
                    push(d.xpos, d.ypos, s.xend, s.yend);
                }
            }
            _ => {}
        }
    }
    out
}

/// Beams kill on contact with the ninja's own radius -- there is no projectile
/// body to add, so the capsule half-width is just the ninja.
const BEAM_KILL_RADIUS: f64 = NINJA_RADIUS;

/// Tier-C sources at the present frame: the deterministic patrol drones.
///
/// The chaser is excluded (it reads the ninja) and so are the shooters (their
/// bodies are harmless -- see `DroneKind::body_is_lethal`).
fn collect_certain(sim: &Simulator) -> Vec<Source> {
    let mut out = Vec::new();
    for entity in &sim.entities {
        if let Entity::Drone(d) = entity {
            if !d.active {
                continue;
            }
            let radius = match d.kind {
                DroneKind::Zap => DRONE_RADIUS,
                DroneKind::Mini => MINI_DRONE_RADIUS,
                _ => continue,
            };
            out.push(Source {
                x: d.xpos,
                y: d.ypos,
                kill_radius: NINJA_RADIUS + radius,
            });
        }
        // The evil ninja, which `logical_collision` kills on contact but which
        // appeared in NO collector -- not certain, not reactive, not beams. On
        // the 58 official maps carrying one (941 instances, the third-densest
        // hazard in the corpus) the agent could not see it in any channel.
        //
        // Tier C, not R, and the distinction is real: it replays the ninja's own
        // input tape at a fixed lag, so its motion is a recording, not a reaction
        // to the live ninja. Its next frames are already determined.
        //
        // Lethal ONLY in PLAYBACK -- a DORMANT/WAITING one is inert and carries
        // the larger 10.0 dormant radius, so treating it as a threat would paint
        // a bigger dead zone than the game has.
        if let Entity::EvilNinja(n) = entity {
            if n.state == evil_ninja::PLAYBACK {
                out.push(Source {
                    x: n.xpos,
                    y: n.ypos,
                    kill_radius: NINJA_RADIUS + n.radius,
                });
            }
        }
    }
    out
}

/// Tier-R sources at the present frame, paired with their per-frame speed caps.
///
/// Everything whose future depends on the ninja. Each contributes a worst-case
/// reachability envelope: the lethal disc dilated by `speed * horizon`.
fn collect_reactive(sim: &Simulator) -> (Vec<Source>, Vec<f64>) {
    let mut out = Vec::new();
    let mut speeds = Vec::new();
    let mut push = |x: f64, y: f64, radius: f64, speed: f64| {
        out.push(Source {
            x,
            y,
            kill_radius: NINJA_RADIUS + radius,
        });
        speeds.push(speed);
    };
    for entity in &sim.entities {
        match entity {
            // Death balls and eye bats have no `active` flag -- they are never
            // deactivated, so there is nothing to test.
            Entity::DeathBall(b) => push(b.xpos, b.ypos, DEATH_BALL_RADIUS, DEATH_BALL_MAX_SPEED),
            Entity::EyeBat(b) => push(b.xpos, b.ypos, EYE_BAT_RADIUS, EYE_BAT_MAX_SPEED),
            Entity::Drone(d) if d.active && matches!(d.kind, DroneKind::Chaser) => {
                push(d.xpos, d.ypos, DRONE_RADIUS, d.kind.speed())
            }
            // Rockets are NOT here. They used to be, dilated isotropically by
            // ROCKET_MAX_SPEED, which over-states a fresh projectile's reach by
            // up to 18x -- see `HomingProjectile`. `collect_projectiles` models
            // the exact speed ramp and the bounded turn rate instead.
            // Dormant floor guard: inert until the ninja crosses its foot-level
            // sightline, then it charges. Direction is unknown until it latches.
            Entity::FloorGuard(g) if g.dir == 0 => {
                push(g.xpos, g.ypos, FLOOR_GUARD_RADIUS, FLOOR_GUARD_SPEED)
            }
            // Thwumps are handled by `collect_thwumps`: their reach is
            // directional, and a radial envelope here was painting their three
            // SAFE sides as lethal.
            // Shove thwump: lethal on contact, shoves off at 4 px/frame and
            // returns at 1. The envelope takes the larger of the two.
            Entity::ShoveThwump(s) => push(s.xpos, s.ypos, SHOVE_THWUMP_RADIUS, 4.0),
            _ => {}
        }
    }
    (out, speeds)
}

// ---------------------------------------------------------------------------
// Solid-entity surface map
// ---------------------------------------------------------------------------

/// Channels of `solid_map`: geometry the ninja can stand on, push off, or is
/// blocked by. See `SolidField::compute`.
pub const SOLID_CHANNELS: usize = 11;
pub const SOLID_CH_STATIC: usize = 0;
pub const SOLID_CH_BOUNCE: usize = 1;
pub const SOLID_CH_ONEWAY: usize = 2;
pub const SOLID_CH_ONEWAY_NX: usize = 3;
pub const SOLID_CH_ONEWAY_NY: usize = 4;

// --- Traversal channels (plan section 3.4) ---------------------------------
//
// # Why these live here and not in `entity_overlays`
//
// The plan puts twelve traversal channels on `entity_overlays` (6 -> 18). Six of
// those twelve ALREADY EXIST in this field, which shipped after the plan was
// written: one-way density and both normal components are channels 2-4, closed
// doors and thwumps are channel 0, and bounce blocks are channel 1. Building
// them again would spend conv parameters and per-step compute on a second copy
// of a signal the policy already receives, and widen the checkpoint break for
// nothing.
//
// What the plan lists that is genuinely absent is the six below, and they belong
// on this grid rather than the 24px overlay grid for two reasons: they must
// co-register cell-for-cell with the solid geometry they qualify (an open door
// is the same barrier as a closed one, seen at a different moment), and the
// boost vectors are directional fields of exactly the kind the one-way normals
// already are.

/// Locked / trap / regular doors while OPEN.
///
/// A door is unified by STATE, not by type -- what the policy needs is "is this
/// barrier passable right now", and locked / trap / regular differ only in what
/// flips that bit. Open doors are deliberately NOT in `SOLID_CH_STATIC`: an open
/// door is not a surface. But it is not nothing either, and channel 0 alone
/// cannot distinguish "there is a doorway here, currently passable" from "there
/// has never been anything here" -- which matters because a trap door closes
/// behind you and a regular door shuts 5 frames after it is last touched.
pub const SOLID_CH_DOOR_OPEN: usize = 5;
/// SWITCH positions of doors that are still CLOSED.
///
/// The single most actionable traversal channel: a closed locked door is opened
/// by touching its switch, which sits somewhere else entirely on the map. The
/// barrier is visible in channel 0 and the reachability graph already models the
/// dependency, but the policy had no way to SEE the thing it has to go and
/// touch. Restricted to closed doors because an open door's switch is spent.
pub const SOLID_CH_DOOR_SWITCH: usize = 6;
/// Launch pad occupancy.
pub const SOLID_CH_LAUNCHPAD: usize = 7;
/// Launch pad boost direction, x and y.
///
/// Direction is the whole point of a launch pad, exactly as it is for a one-way:
/// two pads in the same cell pointing opposite ways are opposite affordances,
/// and an occupancy-only channel makes them identical.
pub const SOLID_CH_LAUNCHPAD_BX: usize = 8;
pub const SOLID_CH_LAUNCHPAD_BY: usize = 9;
/// Boost pad occupancy. No direction: it scales the ninja's existing velocity
/// rather than imposing one, so there is no vector to report.
pub const SOLID_CH_BOOST_PAD: usize = 10;

/// Per-entity surface geometry the tile grid does not carry.
///
/// # Why this is a separate observation from the threat field
///
/// Thwumps, shove thwumps, bounce blocks, one-way platforms and closed doors are
/// SOLID. The ninja stands on them, wall-jumps off them and is blocked by them,
/// and several official routes exist only because of that. But `tile_grid` holds
/// static tiles only, and `entity_overlays` carries goals, mines and gold -- so
/// to the policy a wall-jumpable thwump face has been indistinguishable from
/// empty air.
///
/// That is a different question from "will this kill me", and it needs the
/// opposite sign convention: a thwump is simultaneously a hazard on one face and
/// a floor on the other three. Folding surfaces into `threat_map` would force one
/// number to mean both.
///
/// # Channels
///
/// | ch | meaning |
/// |----|---------|
/// | 0 | `static` -- immovable solid bodies: thwumps, shove thwumps, CLOSED doors |
/// | 1 | `bounce` -- bounce blocks, kept separate because they are springy and change the jump that leaves them |
/// | 2 | `oneway` -- one-way platform occupancy |
/// | 3-4 | `oneway_nx` / `oneway_ny` -- the blocking normal, which is the entire point of a one-way: passable from one side, solid from the other |
///
/// Occupancy is 1.0 inside the body and 0.0 outside, so the encoder reads it like
/// a mask rather than like a distance. Doors contribute only while CLOSED: an
/// open door is not a surface, and a channel that ignored the state would tell
/// the policy a corridor is walled when it is open.
pub struct SolidField {
    buf: Vec<f32>,
}

impl SolidField {
    pub fn new() -> Self {
        SolidField {
            buf: vec![0.0; SOLID_CHANNELS * OUT_WIDTH * OUT_HEIGHT],
        }
    }

    /// Stamp an axis-aligned square body of half-extent `semi` into a channel.
    ///
    /// A cell counts as occupied when its nearest fine sub-centre falls inside
    /// the body -- the same sub-cell rule the threat scatter uses, so the two
    /// observations register against each other exactly.
    fn stamp_box(channel: &mut [f32], cx: f64, cy: f64, semi: f64, value: f32) {
        const HALF: f64 = FINE_CELL_SIZE / 2.0;
        const BLOCK: f64 = FINE_CELL_SIZE * DOWNSAMPLE as f64;
        let lo_c = (((cx - semi - PLAYFIELD_OFFSET_PX) / BLOCK).floor()).max(0.0);
        let lo_r = (((cy - semi - PLAYFIELD_OFFSET_PX) / BLOCK).floor()).max(0.0);
        let hi_c = (((cx + semi - PLAYFIELD_OFFSET_PX) / BLOCK).ceil())
            .clamp(0.0, (OUT_WIDTH - 1) as f64);
        let hi_r = (((cy + semi - PLAYFIELD_OFFSET_PX) / BLOCK).ceil())
            .clamp(0.0, (OUT_HEIGHT - 1) as f64);
        if lo_c > hi_c || lo_r > hi_r {
            return;
        }
        for orow in lo_r as usize..=hi_r as usize {
            let by = PLAYFIELD_OFFSET_PX + (orow as f64 + 0.5) * BLOCK;
            let sy = if cy > by { by + HALF } else { by - HALF };
            if (sy - cy).abs() > semi {
                continue;
            }
            let base = orow * OUT_WIDTH;
            for ocol in lo_c as usize..=hi_c as usize {
                let bx = PLAYFIELD_OFFSET_PX + (ocol as f64 + 0.5) * BLOCK;
                let sx = if cx > bx { bx + HALF } else { bx - HALF };
                if (sx - cx).abs() <= semi {
                    channel[base + ocol] = value;
                }
            }
        }
    }

    fn channel_mut(&mut self, ch: usize) -> &mut [f32] {
        let stride = OUT_WIDTH * OUT_HEIGHT;
        &mut self.buf[ch * stride..(ch + 1) * stride]
    }

    pub fn compute(&mut self, sim: &Simulator) -> &[f32] {
        self.buf.fill(0.0);
        for entity in &sim.entities {
            match entity {
                Entity::Thwump(t) => {
                    Self::stamp_box(
                        self.channel_mut(SOLID_CH_STATIC),
                        t.xpos,
                        t.ypos,
                        THWUMP_SEMI_SIDE,
                        1.0,
                    );
                }
                Entity::ShoveThwump(s) => {
                    Self::stamp_box(
                        self.channel_mut(SOLID_CH_STATIC),
                        s.xpos,
                        s.ypos,
                        SHOVE_THWUMP_SEMI_SIDE,
                        1.0,
                    );
                }
                Entity::BounceBlock(b) => {
                    Self::stamp_box(
                        self.channel_mut(SOLID_CH_BOUNCE),
                        b.xpos,
                        b.ypos,
                        BOUNCE_BLOCK_SEMI_SIDE,
                        1.0,
                    );
                }
                // Locked and trap doors share a barrier segment; both are solid
                // only while closed. A closed door also publishes its SWITCH --
                // the thing the ninja has to go and touch, which sits elsewhere
                // on the map and was invisible in every channel.
                Entity::LockedDoor(d) if d.closed => {
                    Self::stamp_box(
                        self.channel_mut(SOLID_CH_STATIC),
                        d.door_xpos,
                        d.door_ypos,
                        DOOR_SEMI_SIDE,
                        1.0,
                    );
                    Self::stamp_box(
                        self.channel_mut(SOLID_CH_DOOR_SWITCH),
                        d.xpos,
                        d.ypos,
                        LOCKED_DOOR_SWITCH_RADIUS,
                        1.0,
                    );
                }
                Entity::SwitchDoor(d) if d.closed => {
                    Self::stamp_box(
                        self.channel_mut(SOLID_CH_STATIC),
                        d.door_xpos,
                        d.door_ypos,
                        DOOR_SEMI_SIDE,
                        1.0,
                    );
                    Self::stamp_box(
                        self.channel_mut(SOLID_CH_DOOR_SWITCH),
                        d.xpos,
                        d.ypos,
                        LOCKED_DOOR_SWITCH_RADIUS,
                        1.0,
                    );
                }
                // --- traversal: doors by STATE, not by type ------------------
                Entity::LockedDoor(d) => {
                    // Not `closed`: the arm above consumed that case.
                    Self::stamp_box(
                        self.channel_mut(SOLID_CH_DOOR_OPEN),
                        d.door_xpos,
                        d.door_ypos,
                        DOOR_SEMI_SIDE,
                        1.0,
                    );
                }
                Entity::SwitchDoor(d) => {
                    Self::stamp_box(
                        self.channel_mut(SOLID_CH_DOOR_OPEN),
                        d.door_xpos,
                        d.door_ypos,
                        DOOR_SEMI_SIDE,
                        1.0,
                    );
                }
                Entity::LaunchPad(l) => {
                    Self::stamp_box(
                        self.channel_mut(SOLID_CH_LAUNCHPAD),
                        l.xpos,
                        l.ypos,
                        LAUNCH_PAD_RADIUS,
                        1.0,
                    );
                    Self::stamp_box(
                        self.channel_mut(SOLID_CH_LAUNCHPAD_BX),
                        l.xpos,
                        l.ypos,
                        LAUNCH_PAD_RADIUS,
                        l.normal_x as f32,
                    );
                    Self::stamp_box(
                        self.channel_mut(SOLID_CH_LAUNCHPAD_BY),
                        l.xpos,
                        l.ypos,
                        LAUNCH_PAD_RADIUS,
                        l.normal_y as f32,
                    );
                }
                Entity::BoostPad(b) => {
                    Self::stamp_box(
                        self.channel_mut(SOLID_CH_BOOST_PAD),
                        b.xpos,
                        b.ypos,
                        BOOST_PAD_RADIUS,
                        1.0,
                    );
                }
                Entity::OneWay(o) => {
                    let (x, y, nx, ny) = (o.xpos, o.ypos, o.normal_x, o.normal_y);
                    Self::stamp_box(
                        self.channel_mut(SOLID_CH_ONEWAY),
                        x,
                        y,
                        ONE_WAY_PLATFORM_SEMI_SIDE,
                        1.0,
                    );
                    Self::stamp_box(
                        self.channel_mut(SOLID_CH_ONEWAY_NX),
                        x,
                        y,
                        ONE_WAY_PLATFORM_SEMI_SIDE,
                        nx as f32,
                    );
                    Self::stamp_box(
                        self.channel_mut(SOLID_CH_ONEWAY_NY),
                        x,
                        y,
                        ONE_WAY_PLATFORM_SEMI_SIDE,
                        ny as f32,
                    );
                }
                _ => {}
            }
        }
        &self.buf
    }
}

/// Half-extent used to stamp a door barrier. A door spans one 24px cell edge, so
/// half of that is the right footprint.
const DOOR_SEMI_SIDE: f64 = 12.0;
