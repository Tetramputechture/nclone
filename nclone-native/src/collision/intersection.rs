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

/// Check if a circle overlaps a line segment. Port of
/// `overlap_circle_vs_segment` (nclone/physics.py:409). Used by the thwump to
/// test the ninja against its lethal charging face.
pub fn overlap_circle_vs_segment(
    xpos: f64,
    ypos: f64,
    radius: f64,
    px1: f64,
    py1: f64,
    px2: f64,
    py2: f64,
) -> bool {
    let px = px2 - px1;
    let py = py2 - py1;
    let dx = xpos - px1;
    let dy = ypos - py1;
    let seg_lensq = px * px + py * py;
    let mut u = (dx * px + dy * py) / seg_lensq;
    u = u.max(0.0);
    u = u.min(1.0);
    let a = px1 + u * px;
    let b = py1 + u * py;
    (xpos - a) * (xpos - a) + (ypos - b) * (ypos - b) < radius * radius
}

/// If a point is inside an orthogonal square, return the orientation of the
/// shortest vector that depenetrates the point out of the square, plus the
/// penetration on both axes. Port of `penetration_square_vs_point`
/// (nclone/physics.py:131).
///
/// The square is given by its centre and semi-side. To depenetrate the ninja
/// out of a square entity (bounce block, thwump, shwump) the caller passes
/// `semi_side = entity semi-side + NINJA_RADIUS`.
///
/// Returns `Some((depen_normal, (depen_len, other_axis_pen)))`, or `None` when
/// the point is outside. The tuple shape matches `Entity::physical_collision`.
pub fn penetration_square_vs_point(
    s_xpos: f64,
    s_ypos: f64,
    p_xpos: f64,
    p_ypos: f64,
    semi_side: f64,
) -> Option<((f64, f64), (f64, f64))> {
    let dx = p_xpos - s_xpos;
    let dy = p_ypos - s_ypos;
    let penx = semi_side - dx.abs();
    let peny = semi_side - dy.abs();
    if penx > 0.0 && peny > 0.0 {
        // Shorter axis wins; ties (peny == penx) resolve to the vertical push,
        // matching Python's `if peny <= penx`.
        if peny <= penx {
            let normal = if dy < 0.0 { (0.0, -1.0) } else { (0.0, 1.0) };
            Some((normal, (peny, penx)))
        } else {
            let normal = if dx < 0.0 { (-1.0, 0.0) } else { (1.0, 0.0) };
            Some((normal, (penx, peny)))
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_penetration_picks_shortest_axis() {
        // Point 1 left of centre, 4 above: vertical pen (6) > horizontal (9)?
        // semi_side 10 -> penx = 10-1 = 9, peny = 10-4 = 6 -> peny <= penx.
        let (n, v) = penetration_square_vs_point(0.0, 0.0, -1.0, -4.0, 10.0).unwrap();
        assert_eq!(n, (0.0, -1.0));
        assert_eq!(v, (6.0, 9.0));

        // Mirror: 4 right, 1 below -> horizontal push.
        let (n, v) = penetration_square_vs_point(0.0, 0.0, 4.0, 1.0, 10.0).unwrap();
        assert_eq!(n, (1.0, 0.0));
        assert_eq!(v, (6.0, 9.0));
    }

    #[test]
    fn square_penetration_is_none_outside() {
        assert!(penetration_square_vs_point(0.0, 0.0, 11.0, 0.0, 10.0).is_none());
        // Exactly on the boundary is outside (penx == 0, not > 0).
        assert!(penetration_square_vs_point(0.0, 0.0, 10.0, 0.0, 10.0).is_none());
    }

    /// Reference values produced by the Python implementation
    /// (`nclone.physics.penetration_square_vs_point`) over pseudo-random inputs
    /// straddling the square boundary. Equality is EXACT: this helper drives
    /// bounce-block, thwump and shwump depenetration, where a 1-ulp drift
    /// compounds into a divergent trajectory within a few frames.
    #[test]
    fn square_penetration_matches_python_reference() {
        type Case = (f64, f64, f64, f64, f64, Option<((f64, f64), (f64, f64))>);
        let cases: &[Case] = &[
        (34.33879198468469, 17.21404900666596, 46.36479925336401, -1.133641074274447, 22.0, Some(((0.0, -1.0), (3.6523099190595936, 9.973992731320678)))),
        (37.236179589918905, -32.75391445605614, 29.884377841664758, -32.20055771579952, 12.0, Some(((-1.0, 0.0), (4.648198251745853, 11.446643259743382)))),
        (32.56096457587557, -4.271425813429417, 44.230480531281394, -12.56873124898677, 19.0, Some(((1.0, 0.0), (7.330484044594179, 10.702694564442647)))),
        (19.12269389648725, 17.855169649754984, 16.905828896913317, 5.545446748710395, 9.0, None),
        (0.7009718406529188, -19.968707266264772, -3.0028799211798898, -33.366442232169575, 19.0, Some(((0.0, -1.0), (5.602265034095197, 15.296148238167191)))),
        (7.090390454192672, 38.3530434800297, 35.78806823819676, 38.49845574824957, 22.0, None),
        (-24.946102413386072, 12.339742091403984, -36.639583317111246, 21.975390497213215, 9.0, None),
        (37.40523592519175, -33.323675409411614, 63.865728058671706, -15.711030773487781, 22.0, None),
        (-21.57242473195946, -6.088116360182603, -47.60226827971027, 5.5002809735216935, 19.0, None),
        (-19.615468677468062, -12.412558548527933, -14.776818901373627, -10.848751325108058, 19.0, Some(((1.0, 0.0), (14.161350223905565, 17.436192776580125)))),
        (5.008045859634137, 34.959509163344975, -17.72036724511907, 42.37794497804599, 19.0, None),
        (22.51568923251307, 16.34931576459718, 14.182890776548255, 28.27837620152962, 9.0, None),
        (-38.74913533406301, 0.4365805183851421, -38.38462726348441, -6.092643930902478, 9.0, Some(((0.0, -1.0), (2.4707755507123803, 8.635491929421399)))),
        (9.147247283097172, 11.189540935354934, 8.375328974101098, 30.997205993011743, 19.0, None),
        (21.856825543140317, 19.109267384663653, 8.544065711822084, -4.421700337807117, 19.0, None),
        (-32.29345302047221, 1.873072869027368, -22.314586987995696, -18.456404470645182, 22.0, Some(((0.0, -1.0), (1.6705226603274497, 12.02113396752349)))),
        (11.666470288807432, 19.182318010842465, 6.061530051628518, -9.178334242059869, 22.0, None),
        (-45.55716834708156, -2.562917598549639, -49.831319413981916, 0.8949525441489494, 22.0, Some(((-1.0, 0.0), (17.72584893309964, 18.54212985730141)))),
        (-8.059080445966458, -30.80637994591282, 14.008069855265646, -20.778597437040336, 19.0, None),
        (-31.322314609798198, -32.7513935216233, -50.77757186991424, -53.255385398171995, 19.0, None),
        (-30.315270131414184, -36.26153025572779, -16.375092652313715, -33.17353423724706, 22.0, Some(((1.0, 0.0), (8.05982252089953, 18.91200398151927)))),
        (-25.611932914712334, -38.61951078992685, -30.223832224805452, -51.08486686197461, 12.0, None),
        (-23.77235542702213, 30.908529746958962, -20.706941987974368, 29.198580432663675, 22.0, Some(((1.0, 0.0), (18.934586560952237, 20.290050685704713)))),
        (-5.9539605985543105, 45.36772305487327, -10.336384470126706, 18.061178404129166, 22.0, None),
        ];
        for (i, &(sx, sy, px, py, semi, expected)) in cases.iter().enumerate() {
            let got = penetration_square_vs_point(sx, sy, px, py, semi);
            assert_eq!(got, expected, "case {i}: ({sx}, {sy}, {px}, {py}, {semi})");
        }
    }

    #[test]
    fn circle_vs_segment_clamps_to_endpoints() {
        // Beyond the segment's far end: distance measured to the endpoint.
        assert!(overlap_circle_vs_segment(12.0, 0.0, 3.0, 0.0, 0.0, 10.0, 0.0));
        assert!(!overlap_circle_vs_segment(14.0, 0.0, 3.0, 0.0, 0.0, 10.0, 0.0));
        // Perpendicular to the middle.
        assert!(overlap_circle_vs_segment(5.0, 2.0, 3.0, 0.0, 0.0, 10.0, 0.0));
        assert!(!overlap_circle_vs_segment(5.0, 4.0, 3.0, 0.0, 0.0, 10.0, 0.0));
    }
}
