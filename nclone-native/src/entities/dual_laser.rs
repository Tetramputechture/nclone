//! Dual Laser (Type 23).
//! Port of nclone/entity_classes/entity_dual_laser.py; reconstructed from
//! `npp.dll` per `docs/ENTITY_INTEGRATION_SPEC.md` §8.6. Second-biggest gate in
//! the corpus: type 23 is on 16% of attract maps.
//!
//! `think` is a dispatcher on a mode field, which is the two-mode split the
//! abandoned `entity_laser.py` draft never reconciled:
//!
//!   * mode 0  spinner   0x10401270   pi/600, pi/300, no sqrt
//!   * mode 1  surface   0x10401670   0.2, 0.1, sqrt, miss sentinel
//!
//! Mode identification is verified by constant fingerprint, and
//! `EntityDualLaser::projectSurfaceLaserOntoInitialSurface()` exists as an
//! assertion string, so surface mode is real. The two modes touch disjoint field
//! regions, so they are two code paths over one struct, not shared state.
//!
//! Type 23 is thinkable but NOT movable (spec §4) -- surface mode writes its own
//! position, so it must not be added to the move pass to compensate.
//!
//! **Two spinner ordering facts that cause divergence if missed:**
//!   1. The hit test uses the PRE-step angle; the angle is committed only after
//!      the ninja loop.
//!   2. Angle wrapping uses `(double)(float)2pi` = 6.283185005187988, not `2pi` in
//!      full f64 (spec §6.3) -- see `TAU_F32_WIDENED`.
//!
//! And: pi/600 and pi/300 are not two spin speeds. pi/300 is the per-frame step
//! and pi/600 is the hit tolerance, exactly half a step, so the accepted arc
//! equals the arc swept. That is why they differ by 2x.

use crate::collision::raycast::{beam_hits_ninja, get_raycast_distance, raycast_vs_player};
use crate::collision::spatial::gather_segments_from_region;
use crate::collision::sweep::get_single_closest_point;
use crate::constants::{
    DUAL_LASER_ANCHOR_EPSILON, DUAL_LASER_BEAM_LERP, DUAL_LASER_CORNER_SPEED,
    DUAL_LASER_DEAD_END_TOLERANCE, DUAL_LASER_HIT_TOLERANCE, DUAL_LASER_RADIUS,
    DUAL_LASER_SURFACE_GATHER, DUAL_LASER_WRAP_BOUND, DUAL_LASER_WRAP_STEP,
    DUAL_LASER_SPIN_STEP, DUAL_LASER_SURFACE_ITERATIONS, DUAL_LASER_SURFACE_SPEED,
    PI_F32_WIDENED, TAU_F32_WIDENED,
};
use crate::entities::EntityCtx;
use crate::geometry::map_orientation_to_vector;
use crate::types::{clamp_cell, Cell};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaserMode {
    Spinner,
    Surface,
}

#[derive(Debug, Clone)]
pub struct DualLaser {
    pub xpos: f64,
    pub ypos: f64,
    pub mode: LaserMode,
    /// `+0x50`. Spin direction/multiplier, from the map's `mode` byte.
    pub spin: f64,
    /// `+0x48`. Current beam angle (spinner mode).
    pub angle: f64,
    /// Outward surface normal (surface mode): the beam fires along it and the
    /// crawl runs perpendicular to it.
    pub xnormal: f64,
    pub ynormal: f64,
    /// `+0x58`/`+0x60`. The contact point the crawl last sat on. Whether it moves
    /// between sub-steps distinguishes sliding along a surface from being pinned
    /// on a convex corner.
    pub xanchor: f64,
    pub yanchor: f64,
    /// `+0xB0`. The beam's own angle, which LAGS the surface normal.
    pub beam_angle: f64,
    pub xend: f64,
    pub yend: f64,
    pub cell: Cell,
    initial: (f64, f64, f64, f64, f64),
    /// Surface mode only CRAWLS when the map's mode byte is 1; mode 0 sits
    /// still and only the beam moves. See `think_surface`.
    pub crawls: bool,
    /// Surface mode flips `spin` at a dead end, so the reset needs the original.
    initial_spin: f64,
}

impl DualLaser {
    /// Mode is decided at construction by proximity to geometry: a laser mounted
    /// on a wall crawls, one placed in open space spins. The ctor's `24.0` is the
    /// likely query radius; the 12/7 numbers are the abandoned draft's and are the
    /// only concrete reading available.
    pub fn new(
        xcoord: i32,
        ycoord: i32,
        orientation: u8,
        mode: u8,
        segment_grid: &crate::entities::SegmentGrid,
        segments: &[crate::segments::Segment],
        tile_grid: &[[u8; crate::MAP_HEIGHT]; crate::MAP_WIDTH],
    ) -> Self {
        let xpos = (xcoord * 6) as f64;
        let ypos = (ycoord * 6) as f64;

        // `get_single_closest_point` returns 0 for "nothing found", 1 for an
        // outside edge and -1 for an INSIDE edge -- -1 is a hit, not an error.
        // This port read it as an error and sent it to Surface, and sent the
        // genuine no-hit case there too, against the rule itself: a laser
        // mounted on a wall crawls, one in open space spins.
        //
        // On `1011932` four lasers sit at (744, 168), (672, 384), (456, 312) and
        // (264, 432) with no segment within 24px; the game leaves
        // `+0x58`..`+0x70` zero on all four, which only the spinner path does.
        // Scored over all 136 lasers in the capture batches: no-hit -> Spinner
        // with `dist < 7` gives 136/136 at every query radius from 8 to 32; the
        // old branch gives 128/136 and no threshold in that family closes it.
        let legacy_empty = crate::ab::ab().laser_empty_surface;
        let (result, closest) = get_single_closest_point(segment_grid, segments, xpos, ypos, 12.0);
        let laser_mode = match (result, closest) {
            (-1, _) if legacy_empty => LaserMode::Surface,
            (_, None) => {
                if legacy_empty {
                    LaserMode::Surface
                } else {
                    LaserMode::Spinner
                }
            }
            (_, Some((a, b))) => {
                let dist = ((a - xpos).powi(2) + (b - ypos).powi(2)).sqrt();
                if dist < 7.0 {
                    LaserMode::Surface
                } else {
                    LaserMode::Spinner
                }
            }
        };

        let (angle, mut xnormal, mut ynormal) = if laser_mode == LaserMode::Spinner {
            let (dx, dy) = map_orientation_to_vector(orientation);
            (dy.atan2(dx), 0.0, -1.0)
        } else {
            (0.0, 0.0, -1.0)
        };

        let (mut xanchor, mut yanchor) = (xpos - xnormal * DUAL_LASER_RADIUS,
                                          ypos - ynormal * DUAL_LASER_RADIUS);
        let (mut bodyx, mut bodyy) = (xpos, ypos);

        if laser_mode == LaserMode::Surface && !crate::ab::ab().laser_seat {
            // Legacy default: the normal comes from the closest POINT, which is the
            // zero vector for a flush-mounted laser -- so it stays (0, -1) and the
            // body sits one radius inside its own wall. Known wrong; see
            // `ab::laser_seat` for the measurement that keeps it the default.
            let (_r, near) = get_single_closest_point(segment_grid, segments, xpos, ypos, 24.0);
            if let Some((a, b)) = near {
                let dx = xpos - a;
                let dy = ypos - b;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist != 0.0 {
                    xnormal = dx / dist;
                    ynormal = dy / dist;
                }
            }
            xanchor = xpos - xnormal * DUAL_LASER_RADIUS;
            yanchor = ypos - ynormal * DUAL_LASER_RADIUS;
        } else if laser_mode == LaserMode::Surface {
            // `projectSurfaceLaserOntoInitialSurface()`, a one-time snap at spawn.
            // Read off the round-9 captures (findings §1.5): the ANCHOR is the
            // closest point on the surface (not the map coordinate), the BODY sits
            // one radius out along the outward normal, and the normal is the
            // SURFACE's -- orientation has nothing to do with it.
            //
            // Needs the winning SEGMENT, not just the closest point, because a
            // flush laser sits exactly on its surface and `pos - closest` is zero.
            // `get_single_closest_point` discards it, so the search is inlined;
            // the scoring matches it exactly.
            let idxs = crate::collision::spatial::gather_segments_from_region(
                segment_grid, segments, xpos - 24.0, ypos - 24.0, xpos + 24.0, ypos + 24.0,
            );
            let mut best: Option<(f64, usize, f64, f64, bool)> = None;
            let mut flush: Vec<(usize, f64, f64, bool)> = Vec::new();
            for i in idxs {
                let (back, a, b) = segments[i].get_closest_point(xpos, ypos);
                let raw = (xpos - a).powi(2) + (ypos - b).powi(2);
                let score = raw - if back { 0.0 } else { 0.1 };
                if raw == 0.0 {
                    flush.push((i, a, b, back));
                }
                if best.is_none() || score < best.unwrap().0 {
                    best = Some((score, i, a, b, back));
                }
            }
            if let Some((_, mut si, mut a, mut b, mut back)) = best {
                // A flush laser in a corner touches several segments at distance
                // zero, wound opposite ways; a strict `<` just keeps whichever the
                // gather yielded first, and that order is not the game's. Break the
                // tie on meaning: the outward normal points into open space.
                if flush.len() > 1 {
                    // Walk the flush candidates in the GAME's per-cell emission
                    // order. `FUN_1050D6E0` emits a cell's sides as left,
                    // bottom, right, top, so a VERTICAL face is offered first;
                    // this port's accumulator emits every horizontal face and
                    // then every vertical one, so on a convex corner -- where
                    // both point into open space and the loop takes the first
                    // that does -- the horizontal always won.
                    //
                    // `10139043` is that corner. The game's captured normal on
                    // frame 1 is (-0.9985, -0.0552): the vertical face, pivoted
                    // by ten sub-steps of `SURFACE_CORNER_SPEED` inside frame 1.
                    // `NCLONE_AB_LASER_HFIRST=1` restores the old walk.
                    if !crate::ab::ab().laser_hfirst {
                        flush.sort_by_key(|&(fi, _, _, _)| {
                            !matches!(&segments[fi],
                                crate::segments::Segment::Linear(s) if s.px == 0.0)
                        });
                    }
                    for &(fi, fa, fb, fback) in &flush {
                        if let Some((nx, ny)) = segment_normal(&segments[fi], fa, fb) {
                            let cx = ((fa + nx * 12.0) / 24.0).floor() as i32;
                            let cy = ((fb + ny * 12.0) / 24.0).floor() as i32;
                            let empty = cx < 0
                                || cy < 0
                                || cx as usize >= tile_grid.len()
                                || cy as usize >= tile_grid[0].len()
                                || tile_grid[cx as usize][cy as usize] == 0;
                            if empty {
                                si = fi;
                                a = fa;
                                b = fb;
                                back = fback;
                                break;
                            }
                        }
                    }
                }
                // The sign comes from the SEGMENT's winding, never from
                // `pos - closest`. This port used `pos - closest` and fell back
                // to the winding only for a flush mount, which is wrong whenever
                // the laser's map coordinate lies inside the solid -- common:
                // `10138522`'s laser 9 spawns six pixels inside the wall at
                // x = 144 and the game still seats the body at x = 149.9, on the
                // open side. Scored against the game's own `+0x68`/`+0x70` over
                // all 128 surface lasers in the capture batches: `pos - closest`
                // 63, winding 120, winding plus the flush tie-break above 126.
                let _ = (back, xpos, ypos);
                if crate::ab::ab().no_laser_winding {
                    let dx = xpos - a;
                    let dy = ypos - b;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist > 0.0 {
                        xnormal = dx / dist;
                        ynormal = dy / dist;
                    } else if let Some((nx, ny)) = segment_normal(&segments[si], a, b) {
                        xnormal = nx;
                        ynormal = ny;
                    }
                } else if let Some((nx, ny)) = segment_normal(&segments[si], a, b) {
                    xnormal = nx;
                    ynormal = ny;
                }
                xanchor = a;
                yanchor = b;
                bodyx = a + xnormal * DUAL_LASER_RADIUS;
                bodyy = b + ynormal * DUAL_LASER_RADIUS;
            }
        }

        let (xpos, ypos) = (bodyx, bodyy);

        // `NCLONE_DEBUG_LASER_SEAT=1`: the seat is not in the entity snapshot,
        // and the SIGN OF ZERO in `ynormal` decides `beam_angle`'s sign
        // (`atan2(+0.0, -1) = +pi`, `atan2(-0.0, -1) = -pi`), which tilts the
        // beam by +-1.2e-16 -- enough to flip a beam lying exactly one ninja
        // radius away between miss and kill.
        if std::env::var("NCLONE_DEBUG_LASER_SEAT").as_deref() == Ok("1") {
            eprintln!(
                "[laser_seat] pos=({xpos:.17e},{ypos:.17e}) n=({:.17e},{:.17e}) \
                 n_signbits=({},{}) beam={:.17e} mode={laser_mode:?}",
                xnormal,
                ynormal,
                xnormal.is_sign_negative(),
                ynormal.is_sign_negative(),
                ynormal.atan2(xnormal)
            );
        }

        DualLaser {
            xpos,
            ypos,
            mode: laser_mode,
            spin: if mode == 1 { -1.0 } else { 1.0 },
            angle,
            xnormal,
            ynormal,
            xanchor,
            yanchor,
            beam_angle: ynormal.atan2(xnormal),
            xend: xpos,
            yend: ypos,
            cell: clamp_cell((xpos / 24.0).floor() as i32, (ypos / 24.0).floor() as i32),
            initial: (xpos, ypos, angle, xnormal, ynormal),
            crawls: mode == 1,
            initial_spin: if mode == 1 { -1.0 } else { 1.0 },
        }
    }

    pub fn think(&mut self, ctx: &mut EntityCtx) {
        match self.mode {
            LaserMode::Spinner => self.think_spinner(ctx),
            LaserMode::Surface => self.think_surface(ctx),
        }
    }

    fn think_spinner(&mut self, ctx: &mut EntityCtx) {
        let angle = self.angle; // PRE-step angle: everything below uses this one.
        let mut angle_new = self.spin * DUAL_LASER_SPIN_STEP + angle;
        // Explicit wrap against the float-rounded TAU, not a modulo against 2pi.
        while angle_new >= TAU_F32_WIDENED {
            angle_new -= TAU_F32_WIDENED;
        }
        while angle_new < 0.0 {
            angle_new += TAU_F32_WIDENED;
        }

        let dx = angle.cos();
        let dy = angle.sin();
        let beam_len =
            match get_raycast_distance(ctx.segment_grid, ctx.segments, self.xpos, self.ypos, dx, dy) {
            Some(len) => {
                self.xend = self.xpos + dx * len;
                self.yend = self.ypos + dy * len;
                Some(len)
            }
            None => {
                self.xend = self.xpos;
                self.yend = self.ypos;
                None
            }
        };

        if ctx.ninja.is_valid_target()
            && raycast_vs_player(
                ctx.segment_grid,
                ctx.segments,
                self.xpos,
                self.ypos,
                ctx.ninja.xpos,
                ctx.ninja.ypos,
            )
        {
            let ninja_angle = (ctx.ninja.ypos - self.ypos).atan2(ctx.ninja.xpos - self.xpos);
            let mut diff = ninja_angle - angle;
            while diff >= PI_F32_WIDENED {
                diff -= TAU_F32_WIDENED;
            }
            while diff < -PI_F32_WIDENED {
                diff += TAU_F32_WIDENED;
            }
            // Two kill paths, matching spec §8.6's [V] control flow: the centre
            // inside the swept arc, or the drawn beam clipping the hitbox. The
            // second stands in for `FUN_1050B070`, which is [U].
            //
            // Measured: across 524 type-23 attract demos the spinner scores ZERO
            // kills by either path -- every laser death is surface mode. So this
            // is not the source of type 23's false-kill rate, and narrowing it to
            // a centre-only test changes nothing on the corpus. Left as the spec
            // reads it until `FUN_1050B070` is decompiled.
            // `NCLONE_DEBUG_LASER_KILL=1`: which of the two spinner branches
            // fired, and the angles it fired on. Neither is in the snapshot.
            if std::env::var("NCLONE_DEBUG_LASER_KILL").as_deref() == Ok("1") {
                let by_arc = diff.abs() <= DUAL_LASER_HIT_TOLERANCE;
                let by_beam =
                    beam_hits_ninja(self.xpos, self.ypos, self.xend, self.yend, ctx.ninja, beam_len);
                if by_arc || by_beam {
                    eprintln!(
                        "[laser_kill] spinner pos=({:.17e},{:.17e}) angle={:.17e} \
                         ninja_angle={:.17e} diff={:.17e} by_arc={by_arc} by_beam={by_beam} \
                         end=({:.17e},{:.17e}) beam_len={beam_len:?} \
                         anim_enabled={} ab_laser_limbs={} bone0=({:.6},{:.6}) bone1=({:.6},{:.6})",
                        self.xpos,
                        self.ypos,
                        angle,
                        ninja_angle,
                        diff,
                        self.xend,
                        self.yend,
                        ctx.ninja.anim_enabled,
                        crate::ab::ab().laser_limbs,
                        ctx.ninja.bones[0][0],
                        ctx.ninja.bones[0][1],
                        ctx.ninja.bones[1][0],
                        ctx.ninja.bones[1][1],
                    );
                }
            }
            if diff.abs() <= DUAL_LASER_HIT_TOLERANCE
                || beam_hits_ninja(self.xpos, self.ypos, self.xend, self.yend, ctx.ninja, beam_len)
            {
                ctx.ninja.kill(0, 0.0, 0.0, 0.0, 0.0, Some("laser_spinner"));
            }
        }

        self.angle = angle_new; // committed only now
    }

    /// Crawl along the surface, then fire a beam that EASES toward the normal.
    ///
    /// Ten sub-steps per frame, a hard bound. Must stay in step with
    /// `EntityDualLaser._think_surface`; see `ENTITY_RE_ROUND4_FINDINGS.md` §4.3
    /// for the decompiled original.
    fn think_surface(&mut self, ctx: &mut EntityCtx) {
        let g = DUAL_LASER_SURFACE_GATHER;
        let seg_indices = gather_segments_from_region(
            ctx.segment_grid,
            ctx.segments,
            self.xpos - g,
            self.ypos - g,
            self.xpos + g,
            self.ypos + g,
        );
        // EVERY surface laser crawls; the mode byte picks the spin SIGN only.
        //
        // A previous round gated this on `mode == 1`, because 10134679's eight
        // mode-0 lasers and 10138730's six "do not move a pixel in 200 frames".
        // They do not move because the crawl PINS them: the dead-end reversal
        // below flips `spin` every sub-step once the contact point stops
        // advancing, so a laser on a short face oscillates inside a fraction of
        // a pixel. Fifty of the 136 captured surface lasers move by 0.2 or less
        // per frame, and 10138727's lone laser crawls a full 1.0 (ten sub-steps)
        // on frame 1 with a mode byte of 0.
        //
        // Scored as the fraction of (laser, frame) pairs bit-equal to the game's
        // over the first 120 frames of every captured demo: mode-gated
        // 10440/15360 = 67.97%, crawl-always 13560/15360 = 88.28%.
        let may_crawl =
            !crate::ab::ab().laser_mode_gate || self.crawls || !crate::ab::ab().laser_seat;
        // A door in the gather region used to disable the crawl outright. That
        // was a proxy fitted to the captures -- every one of the 15 lasers the
        // game froze and this crawl moved had a door segment in region -- and it
        // is too strong. `FUN_10401670` gates on the door being the segment the
        // crawl would step ONTO, not on its being nearby: the `else` arm of the
        // dead-end test falls into the same handler when the winning segment's
        // kind is 3, and kind 3 is the door (it takes the linear closest-point
        // routine `FUN_1050ECA0` like an ordinary tile edge, but with `back`
        // forced false -- exactly `oriented=false`). See the tangent check below.
        //
        // On `10138435` the difference is the whole demo: a door edge at y=336
        // enters the 12 px region on frame 385 while the laser is crawling a tile
        // floor at y=324, so the region test freezes it 57 px short.
        if !seg_indices.is_empty() && may_crawl {
            for _ in 0..DUAL_LASER_SURFACE_ITERATIONS {
                // Probe one sub-step along the surface, perpendicular to the normal.
                let tx = -self.ynormal * self.spin * DUAL_LASER_SURFACE_SPEED;
                let ty = self.xnormal * self.spin * DUAL_LASER_SURFACE_SPEED;
                let xnew = self.xpos + tx;
                let ynew = self.ypos + ty;

                let mut best_d2 = f64::INFINITY;
                let mut best: Option<(f64, f64)> = None;
                let mut best_back = false;
                let mut best_door = false;
                for &idx in &seg_indices {
                    let (back, a, b) = ctx.segments[idx].get_closest_point(xnew, ynew);
                    let d2 = (xnew - a).powi(2) + (ynew - b).powi(2);
                    let door = match &ctx.segments[idx] {
                        crate::segments::Segment::Linear(s) => !s.oriented,
                        _ => false,
                    };
                    // Strict `<`, then one tie-break: on an EXACT tie a
                    // front-facing segment displaces a back-facing one.
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best = Some((a, b));
                        best_back = back;
                        best_door = door;
                    } else if d2 == best_d2 && best_back && !back {
                        best = Some((a, b));
                        best_back = false;
                        best_door = door;
                    }
                }
                let Some((ax, by)) = best else { break };

                let vx = xnew - ax;
                let vy = ynew - by;

                // Two ways into the tangent check, and the binary spells both: a
                // dead end -- the probe has walked past the anchor and is heading
                // INTO the surface -- or the winning segment being a DOOR (kind
                // 3), whose `else` arm re-enters the same handler. Either way the
                // laser advances only if the step still has a positive component
                // along the tangent; otherwise it reverses and stays put. A door
                // is crawlable; it is refused only when the step turns back on
                // itself.
                if ((self.xpos - self.xanchor) * vx + (self.ypos - self.yanchor) * vy
                    <= DUAL_LASER_DEAD_END_TOLERANCE
                    || best_door)
                    && tx * vx + ty * vy <= 0.0
                {
                    self.spin = -self.spin;
                    continue;
                }

                let moved = ((ax - self.xanchor).powi(2) + (by - self.yanchor).powi(2)).sqrt();
                if moved >= DUAL_LASER_ANCHOR_EPSILON {
                    // Contact point moved: snap the normal onto it, signed by
                    // which side of the directed segment we are on.
                    let dist = (vx * vx + vy * vy).sqrt();
                    if dist == 0.0 {
                        break;
                    }
                    let sign = if best_back { -1.0 } else { 1.0 };
                    self.xnormal = vx / dist * sign;
                    self.ynormal = vy / dist * sign;
                } else {
                    // Contact point did NOT move: pinned on a convex corner, so
                    // rotate the normal in place instead of sliding.
                    let a = self.ynormal.atan2(self.xnormal)
                        + crate::ab::ab().dl_corner_speed * self.spin;
                    self.xnormal = a.cos();
                    self.ynormal = a.sin();
                }

                self.xanchor = ax;
                self.yanchor = by;
                self.xpos = ax + self.xnormal * DUAL_LASER_RADIUS;
                self.ypos = by + self.ynormal * DUAL_LASER_RADIUS;
            }
            let new_cell = clamp_cell(
                (self.xpos / 24.0).floor() as i32,
                (self.ypos / 24.0).floor() as i32,
            );
            let old_cell = self.cell;
            self.cell = new_cell;
            ctx.grid_move(old_cell, new_cell);
        }

        // The beam does NOT point along the normal -- it eases toward it, 20% of
        // the remaining angle per frame, so it lags the crawl. A snapping beam
        // points straight at whatever the crawl just walked past, which lands it
        // on the ninja on the frame it rounds a corner.
        let mut diff = self.ynormal.atan2(self.xnormal) - self.beam_angle;
        while diff.abs() >= DUAL_LASER_WRAP_BOUND {
            diff -= DUAL_LASER_WRAP_STEP.copysign(diff);
        }
        self.beam_angle += diff * DUAL_LASER_BEAM_LERP;
        let bx = self.beam_angle.cos();
        let by = self.beam_angle.sin();

        let beam_len =
            match get_raycast_distance(ctx.segment_grid, ctx.segments, self.xpos, self.ypos, bx, by) {
            Some(len) => {
                self.xend = self.xpos + bx * len;
                self.yend = self.ypos + by * len;
                Some(len)
            }
            None => {
                self.xend = self.xpos;
                self.yend = self.ypos;
                None
            }
        };

        if !ctx.ninja.is_valid_target() {
            return;
        }
        if !raycast_vs_player(
            ctx.segment_grid,
            ctx.segments,
            self.xpos,
            self.ypos,
            ctx.ninja.xpos,
            ctx.ninja.ypos,
        ) {
            return;
        }
        // `beam_len`, not `hypot(xend - xpos, ...)`: `FUN_1050B070` takes the
        // raycast's own distance as `param_5`. See `check_lineseg_vs_ninja`.
        if beam_hits_ninja(self.xpos, self.ypos, self.xend, self.yend, ctx.ninja, beam_len) {
            ctx.ninja.kill(0, 0.0, 0.0, 0.0, 0.0, Some("laser_surface"));
        }
    }

    pub fn reset_state(&mut self) {
        let (x, y, angle, nx, ny) = self.initial;
        self.xpos = x;
        self.ypos = y;
        self.angle = angle;
        self.xnormal = nx;
        self.ynormal = ny;
        self.xend = x;
        self.yend = y;
        self.cell = clamp_cell((x / 24.0).floor() as i32, (y / 24.0).floor() as i32);
        if self.mode == LaserMode::Surface {
            self.spin = self.initial_spin;
            self.xanchor = x - nx * DUAL_LASER_RADIUS;
            self.yanchor = y - ny * DUAL_LASER_RADIUS;
            self.beam_angle = ny.atan2(nx);
        }
    }
}

/// A tile segment's outward normal at the contact point `(a, b)`: `(-dy, dx)`
/// from the consistent tile winding (round-7 §3), or the radial direction on an
/// arc, inverted when the arc is concave.
///
/// It does NOT depend on which side the query point is on -- `is_back_facing`
/// only says the laser spawned inside the solid, and the game seats it on the
/// open side regardless. See the seating comment in `DualLaser::new`.
fn segment_normal(seg: &crate::segments::Segment, a: f64, b: f64) -> Option<(f64, f64)> {
    match seg {
        crate::segments::Segment::Linear(s) => {
            // A DOOR states its facing outright. Its segment is wound low-to-high
            // like every other, because that winding is the numerically exact one
            // for `get_closest_point` and the ninja stands on these -- see
            // `entities/door_base.rs`. So the facing cannot be recovered from
            // `(-py, px)` and is carried alongside instead.
            if let Some(n) = s.door_normal {
                return Some(n);
            }
            let plen = (s.px * s.px + s.py * s.py).sqrt();
            if plen == 0.0 {
                return None;
            }
            Some((-s.py / plen, s.px / plen))
        }
        crate::segments::Segment::Circular(s) => {
            if s.radius == 0.0 {
                return None;
            }
            let sign = if s.convex { 1.0 } else { -1.0 };
            Some((
                (a - s.xpos) / s.radius * sign,
                (b - s.ypos) / s.radius * sign,
            ))
        }
    }
}
