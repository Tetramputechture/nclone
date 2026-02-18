//! Intersection time calculations for sweep tests.
//! Direct port of get_time_of_intersection_* functions from physics.py.

/// Return time of intersection by sweeping a circle onto another circle (given combined radius).
pub fn circle_vs_circle(
    xpos: f64,
    ypos: f64,
    vx: f64,
    vy: f64,
    a: f64,
    b: f64,
    radius: f64,
) -> f64 {
    let dx = xpos - a;
    let dy = ypos - b;
    let dist_sq = dx * dx + dy * dy;
    let vel_sq = vx * vx + vy * vy;
    let dot_prod = dx * vx + dy * vy;

    if dist_sq - radius * radius > 0.0 {
        let radicand = dot_prod * dot_prod - vel_sq * (dist_sq - radius * radius);
        if vel_sq > 0.0001 && dot_prod < 0.0 && radicand >= 0.0 {
            return (-dot_prod - radicand.sqrt()) / vel_sq;
        }
        return 1.0;
    }
    0.0
}

/// Return time of intersection by sweeping a circle onto a line segment.
pub fn circle_vs_lineseg(
    xpos: f64,
    ypos: f64,
    dx: f64,
    dy: f64,
    a1: f64,
    b1: f64,
    a2: f64,
    b2: f64,
    radius: f64,
) -> f64 {
    let wx = a2 - a1;
    let wy = b2 - b1;
    let seg_len = (wx * wx + wy * wy).sqrt();
    let nx = wx / seg_len;
    let ny = wy / seg_len;
    let normal_proj = (xpos - a1) * ny - (ypos - b1) * nx;
    let hor_proj = (xpos - a1) * nx + (ypos - b1) * ny;

    if normal_proj.abs() >= radius {
        let dir = dx * ny - dy * nx;
        if dir * normal_proj < 0.0 {
            let t = ((normal_proj.abs() - radius) / dir.abs()).min(1.0);
            let hor_proj2 = hor_proj + t * (dx * nx + dy * ny);
            if (0.0..=seg_len).contains(&hor_proj2) {
                return t;
            }
        }
    } else if (0.0..=seg_len).contains(&hor_proj) {
        return 0.0;
    }
    1.0
}

/// Return time of intersection by sweeping a circle onto a circular arc.
pub fn circle_vs_arc(
    xpos: f64,
    ypos: f64,
    vx: f64,
    vy: f64,
    a: f64,
    b: f64,
    hor: f64,
    ver: f64,
    radius_arc: f64,
    radius_circle: f64,
) -> f64 {
    let dx = xpos - a;
    let dy = ypos - b;
    let dist_sq = dx * dx + dy * dy;
    let vel_sq = vx * vx + vy * vy;
    let dot_prod = dx * vx + dy * vy;
    let radius1 = radius_arc + radius_circle;
    let radius2 = radius_arc - radius_circle;

    let t;
    if dist_sq > radius1 * radius1 {
        let radicand = dot_prod * dot_prod - vel_sq * (dist_sq - radius1 * radius1);
        if vel_sq > 0.0001 && dot_prod < 0.0 && radicand >= 0.0 {
            t = (-dot_prod - radicand.sqrt()) / vel_sq;
        } else {
            return 1.0;
        }
    } else if dist_sq < radius2 * radius2 {
        let radicand = dot_prod * dot_prod - vel_sq * (dist_sq - radius2 * radius2);
        if vel_sq > 0.0001 {
            t = ((-dot_prod + radicand.sqrt()) / vel_sq).min(1.0);
        } else {
            return 1.0;
        }
    } else {
        t = 0.0;
    }

    if (dx + t * vx) * hor > 0.0 && (dy + t * vy) * ver > 0.0 {
        t
    } else {
        1.0
    }
}

/// Check if two circles overlap.
pub fn overlap_circle_vs_circle(
    xpos1: f64,
    ypos1: f64,
    radius1: f64,
    xpos2: f64,
    ypos2: f64,
    radius2: f64,
) -> bool {
    let dx = xpos1 - xpos2;
    let dy = ypos1 - ypos2;
    let dist = (dx * dx + dy * dy).sqrt();
    dist < radius1 + radius2
}
