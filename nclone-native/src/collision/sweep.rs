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

    let seg_indices =
        super::spatial::gather_segments_from_region(segment_grid, all_segments, x1, y1, x2, y2);

    let mut shortest_time = 1.0;
    for seg_idx in seg_indices {
        let time = all_segments[seg_idx].intersect_with_ray(xpos_old, ypos_old, dx, dy, radius);
        if time == 0.0 {
            return 0.0;
        }
        if time < shortest_time {
            shortest_time = time;
        }
    }
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
    let seg_indices = super::spatial::gather_segments_from_region(
        segment_grid,
        all_segments,
        xpos - radius,
        ypos - radius,
        xpos + radius,
        ypos + radius,
    );

    let mut shortest_distance = f64::INFINITY;
    let mut result = 0i32;
    let mut closest_point: Option<(f64, f64)> = None;

    let query_min_x = xpos - radius;
    let query_min_y = ypos - radius;
    let query_max_x = xpos + radius;
    let query_max_y = ypos + radius;

    for seg_idx in seg_indices {
        let seg = &all_segments[seg_idx];
        let (seg_min_x, seg_min_y, seg_max_x, seg_max_y) = seg.get_bounds();

        // Fast AABB rejection
        if seg_max_x < query_min_x
            || seg_min_x > query_max_x
            || seg_max_y < query_min_y
            || seg_min_y > query_max_y
        {
            continue;
        }

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
    }
    (result, closest_point)
}
