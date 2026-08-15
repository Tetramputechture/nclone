//! Raycasting against tile geometry, and the beam/line-of-sight tests built on
//! it. Port of `get_raycast_distance`, `intersect_ray_vs_cell_contents`,
//! `raycast_vs_player`, `find_visible_ninja` and `check_lineseg_vs_ninja`
//! (nclone/physics.py).
//!
//! This module existed only in Python until now, and its absence was the single
//! blocker on two entity types: the Turret (19) is hitscan and the DualLaser (23)
//! raycasts in both of its modes (`docs/ENTITY_INTEGRATION_SPEC.md` §10 lists
//! `FUN_1050E0E0` as "the main blocker").
//!
//! The traversal is a DDA over 24px cells. Note two details that are easy to
//! "clean up" into a divergence:
//!
//!   * The ray is tested against each cell's segments at the FULL 2000-unit
//!     length, not clipped to the cell -- so a segment in cell N can be hit
//!     before the walk reaches it. The walk only decides which cells to consult
//!     and in what order; `shortest_time < 1` in any cell ends it.
//!   * Running off the map returns "no hit" (`None`), which is a different
//!     outcome from "hit at distance 2000". Callers treat `None` as an infinitely
//!     long beam, matching Python's falsy `None`.

use crate::constants::{MAP_HEIGHT, MAP_WIDTH, NINJA_RADIUS};
use crate::ninja::Ninja;
use crate::segments::Segment;
use crate::types::clamp_cell;

/// Maximum ray length. Beyond this the ray is treated as hitting nothing.
const RAY_LENGTH: f64 = 2000.0;

/// Shortest intersection time between a ray and one cell's tile segments.
/// Returns 1.0 when the ray misses everything in the cell.
fn intersect_ray_vs_cell_contents(
    segment_grid: &[[Vec<usize>; MAP_HEIGHT]; MAP_WIDTH],
    all_segments: &[Segment],
    xcell: i32,
    ycell: i32,
    xpos: f64,
    ypos: f64,
    dx: f64,
    dy: f64,
) -> f64 {
    let (cx, cy) = clamp_cell(xcell, ycell);
    let mut shortest_time = 1.0;
    for &seg_idx in &segment_grid[cx][cy] {
        // **Inactive segments do not block.** `FUN_1050E410` walks a cell in two
        // passes -- the static tile segments, always live, then a linked list of
        // the dynamic ones (the doors), each gated on `(*seg & 8) != 0` before
        // its hit counts. That bit is `is_active`, which the ninja's own gather
        // already honours; this loop was the one place that did not, so an OPEN
        // door still stopped every sightline. Mirrors `physics.py`.
        if !all_segments[seg_idx].is_active() {
            continue;
        }
        let time = all_segments[seg_idx].intersect_with_ray(xpos, ypos, dx, dy, 0.0);
        if time < shortest_time {
            shortest_time = time;
        }
    }
    shortest_time
}

/// Length of a ray from `(xpos, ypos)` along `(dx, dy)` until it hits a tile.
/// `None` means it travelled 2000 units or left the map without hitting anything.
///
/// `(dx, dy)` is expected to be a unit vector; the returned value is in pixels.
pub fn get_raycast_distance(
    segment_grid: &[[Vec<usize>; MAP_HEIGHT]; MAP_WIDTH],
    all_segments: &[Segment],
    xpos: f64,
    ypos: f64,
    dx: f64,
    dy: f64,
) -> Option<f64> {
    let mut xcell = (xpos / 24.0).floor() as i32;
    let mut ycell = (ypos / 24.0).floor() as i32;

    // Standard DDA setup. A zero component parks tmax at a value the other axis
    // can never beat, which pins the walk to a single row or column.
    let (step_x, delta_x, mut tmax_x) = if dx > 0.0 {
        (1i32, 24.0 / dx, ((xcell + 1) as f64 * 24.0 - xpos) / dx)
    } else if dx < 0.0 {
        (-1i32, -24.0 / dx, (xcell as f64 * 24.0 - xpos) / dx)
    } else {
        (0i32, 0.0, 999999.0)
    };
    let (step_y, delta_y, mut tmax_y) = if dy > 0.0 {
        (1i32, 24.0 / dy, ((ycell + 1) as f64 * 24.0 - ypos) / dy)
    } else if dy < 0.0 {
        (-1i32, -24.0 / dy, (ycell as f64 * 24.0 - ypos) / dy)
    } else {
        (0i32, 0.0, 999999.0)
    };

    loop {
        let result = intersect_ray_vs_cell_contents(
            segment_grid,
            all_segments,
            xcell,
            ycell,
            xpos,
            ypos,
            RAY_LENGTH * dx,
            RAY_LENGTH * dy,
        );
        if result < 1.0 {
            return Some(RAY_LENGTH * result);
        }
        if tmax_x < tmax_y {
            xcell += step_x;
            if xcell < 0 || xcell >= MAP_WIDTH as i32 {
                return None;
            }
            tmax_x += delta_x;
        } else {
            ycell += step_y;
            if ycell < 0 || ycell >= MAP_HEIGHT as i32 {
                return None;
            }
            tmax_y += delta_y;
        }
        // A direction of exactly (0, 0) would otherwise spin here forever.
        // Python relies on the caller never doing that; be explicit instead.
        if step_x == 0 && step_y == 0 {
            return None;
        }
    }
}

/// Is there an unobstructed line from `(xstart, ystart)` to the ninja's centre?
///
/// Port of `raycast_vs_player`. The degenerate case where the start point is
/// already inside the ninja's radius returns `true` -- there is nothing between
/// them to block.
pub fn raycast_vs_player(
    segment_grid: &[[Vec<usize>; MAP_HEIGHT]; MAP_WIDTH],
    all_segments: &[Segment],
    xstart: f64,
    ystart: f64,
    ninja_xpos: f64,
    ninja_ypos: f64,
) -> bool {
    let dx = ninja_xpos - xstart;
    let dy = ninja_ypos - ystart;
    let dist = (dx * dx + dy * dy).sqrt();
    if NINJA_RADIUS <= dist && dist > 0.0 {
        let length = get_raycast_distance(
            segment_grid,
            all_segments,
            xstart,
            ystart,
            dx / dist,
            dy / dist,
        );
        // `None` (nothing hit) means the sightline is clear.
        return match length {
            Some(len) => len > dist - NINJA_RADIUS,
            None => true,
        };
    }
    true
}

/// The nearest ninja with unobstructed line of sight, or `None`.
///
/// Port of the shared query `FUN_1051E690`, which the Rocket (18), Turret (19)
/// and EyeBat (27) all call. The game loops the 8 co-op ninja slots and takes the
/// nearest visible one (spec §6.5); this sim has one ninja, so the loop
/// degenerates -- but the contract is kept nearest-selecting rather than "the
/// ninja" so multi-ninja support would not have to touch every caller.
pub fn find_visible_ninja<'a>(
    segment_grid: &[[Vec<usize>; MAP_HEIGHT]; MAP_WIDTH],
    all_segments: &[Segment],
    xpos: f64,
    ypos: f64,
    ninja: &'a Ninja,
) -> Option<&'a Ninja> {
    if !ninja.is_valid_target() {
        return None;
    }
    if raycast_vs_player(
        segment_grid,
        all_segments,
        xpos,
        ypos,
        ninja.xpos,
        ninja.ypos,
    ) {
        Some(ninja)
    } else {
        None
    }
}

/// `FUN_1050C690`: the hitscan test the TURRET and the chaingun drone use.
///
/// Distinct from `beam_hits_ninja`. The binary's callers split cleanly:
/// `FUN_1050B070` (the 11-limb test) is reached only from the laser family --
/// dual laser (23, both modes) and drone laser (13) -- while `FUN_1050C690` is
/// called by the turret (19, `1041A150`) and the chaingun drone (12,
/// `103FE790`). Limbs are a strict subset of the circle, so routing the turret
/// through them made it under-lethal per shot.
///
/// Project the centre onto the segment, clamp to the endpoints, compare against
/// the radius with a **strict** `<`. The radius is the ninja's own
/// (`FUN_105136B0`); there is no separate shot radius.
///
/// `seg_len` is a parameter rather than recomputed because the game passes the
/// raycast's returned distance. `|B - A|` equals it only up to rounding, and the
/// orderings below -- the y term first in both dot products, and `A + d*t` with
/// `d` pre-divided -- are the binary's, not an algebraic simplification.
/// Mirrors `nclone/physics.py:seg_vs_circle` operation for operation.
#[allow(clippy::too_many_arguments)]
pub fn seg_vs_circle(
    cx: f64,
    cy: f64,
    radius: f64,
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
    seg_len: f64,
) -> bool {
    // `if (NAN(len) != (len == 0.0)) return false` -- reject a zero or NaN
    // length before the divide.
    if seg_len == 0.0 || seg_len.is_nan() {
        return false;
    }
    let dx = (bx - ax) / seg_len;
    let dy = (by - ay) / seg_len;
    let t = (cy - ay) * dy + (cx - ax) * dx;
    let (px, py) = if t > 0.0 {
        if t < seg_len {
            (dx * t + ax, dy * t + ay)
        } else {
            (bx, by)
        }
    } else {
        (ax, ay)
    };
    (cy - py) * (cy - py) + (cx - px) * (cx - px) < radius * radius
}

/// Does the beam `(x1,y1)-(x2,y2)` hit the ninja?
///
/// The laser family's test: dual laser (23) and drone laser (13). The turret and
/// the chaingun drone use `seg_vs_circle` instead -- see there.
///
/// With the animation skeleton on, this is the game's real test (`FUN_1050B070`):
/// a circle prefilter at `NINJA_RADIUS`, then an exact intersection against the
/// **11 limbs**, so a beam can pass within the radius and miss *between* them.
/// Without the skeleton it degrades to the circle test alone, which is strictly
/// more lethal than the game. `nclone/physics.py:beam_hits_ninja` makes exactly
/// the same choice on the same flag, which is what keeps the engines frame-exact.
pub fn beam_hits_ninja(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    ninja: &Ninja,
    seg_len: Option<f64>,
) -> bool {
    beam_hits_ninja_opt(x1, y1, x2, y2, ninja, seg_len, true)
}

/// `beam_hits_ninja` with an explicit per-entity opt-out of the skeleton.
///
/// §73: the chaingun's kill is `FUN_1050C690`, a bare circle test with no
/// skeleton in it, while the laser family calls the 11-limb `FUN_1050B070`. The
/// global `NCLONE_AB_NO_LASER_LIMBS` cannot tell the two apart; `use_limbs`
/// can. Mirrors `physics.beam_hits_ninja`'s parameter of the same name.
pub fn beam_hits_ninja_opt(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    ninja: &Ninja,
    seg_len: Option<f64>,
    use_limbs: bool,
) -> bool {
    // A zero-length beam hits nothing. Callers legitimately produce one -- every
    // entity collapses its endpoint onto its own position when the raycast finds
    // no wall -- and the overlap test divides by the segment length.
    if x1 == x2 && y1 == y2 {
        return false;
    }
    if !use_limbs || !ninja.anim_enabled || !crate::ab::ab().laser_limbs {
        // Opted out, or no skeleton: keep the old circle test EXACTLY, because
        // Python's fallback is a byte-for-byte copy of it. The prefilter inside
        // `check_lineseg_vs_ninja` is the same test algebraically but divides by
        // `seglen` rather than `seg_lensq`, so it is not bit-identical.
        return super::intersection::overlap_circle_vs_segment(
            ninja.xpos,
            ninja.ypos,
            NINJA_RADIUS,
            x1,
            y1,
            x2,
            y2,
        );
    }
    check_lineseg_vs_ninja(x1, y1, x2, y2, ninja, seg_len)
}

/// The game's real beam-vs-ninja test: a circle prefilter, then 11 limbs.
///
/// `FUN_1050B070` (`docs/decomp/full/physics_1050B070.c`). The per-limb predicate
/// is `FUN_1050B840`, the textbook proper-intersection test with **strict** `< 0`
/// on both cross-product products, so a beam that merely grazes a limb endpoint
/// is not a hit either. Returns on the first limb that connects.
///
/// Mirrors `nclone/physics.py:check_lineseg_vs_ninja` operation for operation.
/// `seg_len` is the raycast's returned distance, passed in. The binary takes it
/// as `param_5` and never recomputes it; `hypot(x2 - x1, y2 - y1)` equals it only
/// up to rounding, because the endpoint was built as `A + dir * seg_len` in the
/// first place. `None` recomputes, for callers with no raycast to hand.
///
/// Both dot products below put the **y term first**, which is the binary's order
/// and not this port's original one.
pub fn check_lineseg_vs_ninja(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    ninja: &Ninja,
    seg_len: Option<f64>,
) -> bool {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let seglen = seg_len.unwrap_or_else(|| (dx * dx + dy * dy).sqrt());
    if seglen == 0.0 || seglen.is_nan() {
        return false;
    }
    let (ux, uy) = (dx / seglen, dy / seglen);
    // The projection is NOT clamped before the branch; the branch does the
    // clamping.
    let t = (ninja.ypos - y1) * uy + (ninja.xpos - x1) * ux;
    let (cx, cy) = if t <= 0.0 {
        (x1, y1)
    } else if t >= seglen {
        (x2, y2)
    } else {
        (t * ux + x1, t * uy + y1)
    };
    if (ninja.ypos - cy) * (ninja.ypos - cy) + (ninja.xpos - cx) * (ninja.xpos - cx)
        >= NINJA_RADIUS * NINJA_RADIUS
    {
        return false;
    }

    let (nx, ny) = (ninja.xpos, ninja.ypos);
    for (ia, ib) in crate::anim::NINJA_LIMB_PAIRS {
        let ax = nx + crate::anim::BONE_SCALE * ninja.bones[ia][0];
        let ay = ny + crate::anim::BONE_SCALE * ninja.bones[ia][1];
        let bx = nx + crate::anim::BONE_SCALE * ninja.bones[ib][0];
        let by = ny + crate::anim::BONE_SCALE * ninja.bones[ib][1];
        let d1 = dx * (ay - y1) - dy * (ax - x1);
        let d2 = dx * (by - y1) - dy * (bx - x1);
        if d1 * d2 >= 0.0 {
            continue;
        }
        let (lx, ly) = (bx - ax, by - ay);
        let d3 = lx * (y1 - ay) - ly * (x1 - ax);
        let d4 = lx * (y2 - ay) - ly * (x2 - ax);
        if d3 * d4 < 0.0 {
            return true;
        }
    }
    false
}
