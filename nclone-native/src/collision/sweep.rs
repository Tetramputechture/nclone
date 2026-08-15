//! Sweep test: moving circle vs tile segments.
//! Direct port of sweep_circle_vs_tiles and get_single_closest_point from physics.py.

use crate::constants::{MAP_HEIGHT, MAP_WIDTH};
use crate::segments::Segment;

/// Sweep a circle from old position along (dx, dy) and find shortest intersection time.
/// Returns a value in [0.0, 1.0].
pub fn sweep_circle_vs_tiles(
    segment_grid: &[[Vec<usize>; MAP_HEIGHT]; MAP_WIDTH],
    all_segments: &[Segment],
    xpos_old: f64,
    ypos_old: f64,
    dx: f64,
    dy: f64,
    radius: f64,
) -> f64 {
    let xpos_new = xpos_old + dx;
    let ypos_new = ypos_old + dy;
    let width = radius + 1.0;

    let x1 = xpos_old.min(xpos_new) - width;
    let y1 = ypos_old.min(ypos_new) - width;
    let x2 = xpos_old.max(xpos_new) + width;
    let y2 = ypos_old.max(ypos_new) + width;

    let mut shortest_time = 1.0;
    super::spatial::for_each_segment_in_region(
        segment_grid,
        all_segments,
        x1,
        y1,
        x2,
        y2,
        |seg_idx| {
            let time = all_segments[seg_idx].intersect_with_ray(xpos_old, ypos_old, dx, dy, radius);
            if time == 0.0 {
                shortest_time = 0.0;
                return false;
            }
            if time < shortest_time {
                shortest_time = time;
            }
            true
        },
    );
    shortest_time
}

/// Find the single closest point on any collidable segment from the given position.
/// Returns (result_code, closest_x, closest_y).
/// result_code: 0 = no collision found, 1 = front-facing, -1 = back-facing.
pub fn get_single_closest_point(
    segment_grid: &[[Vec<usize>; MAP_HEIGHT]; MAP_WIDTH],
    all_segments: &[Segment],
    xpos: f64,
    ypos: f64,
    radius: f64,
) -> (i32, Option<(f64, f64)>) {
    // Python's initial bound (`physics.py:116`), not `f64::INFINITY`. Unreachable
    // on a real level -- the longest diagonal squares to ~1.5e6 -- but it is the
    // reference value, so it stays the reference value.
    let mut shortest_distance = 9999999.0f64;
    let mut result = 0i32;
    let mut closest_point: Option<(f64, f64)> = None;

    // NO AABB rejection against the query box. `gather_segments_from_region`
    // returns everything registered in the overlapped CELLS, and Python then
    // considers every one of them -- a segment can sit well outside the query box
    // yet still be the closest point returned. Filtering here made the native sim
    // miss a door segment 36px from a dual laser that Python sees, flipping the
    // laser's spinner-vs-surface mode at construction.
    super::spatial::for_each_segment_in_region(
        segment_grid,
        all_segments,
        xpos - radius,
        ypos - radius,
        xpos + radius,
        ypos + radius,
        |seg_idx| {
            let seg = &all_segments[seg_idx];

            let (is_back_facing, a, b) = seg.get_closest_point(xpos, ypos);
            let mut distance_sq = (xpos - a) * (xpos - a) + (ypos - b) * (ypos - b);

            if !is_back_facing {
                distance_sq -= 0.1;
            }
            if distance_sq < shortest_distance {
                shortest_distance = distance_sq;
                closest_point = Some((a, b));
                result = if is_back_facing { -1 } else { 1 };
            }
            true
        },
    );
    (result, closest_point)
}
