//! Collision segment types: linear and circular.
//! Direct port of GridSegmentLinear and GridSegmentCircular from entities.py.

use crate::collision::intersection;

/// A collision segment (linear or circular arc).
#[derive(Debug, Clone)]
pub enum Segment {
    Linear(LinearSegment),
    Circular(CircularSegment),
}

impl Segment {
    /// Find the closest point on this segment from the given position.
    /// Returns (is_back_facing, closest_x, closest_y).
    pub fn get_closest_point(&self, xpos: f64, ypos: f64) -> (bool, f64, f64) {
        match self {
            Segment::Linear(s) => s.get_closest_point(xpos, ypos),
            Segment::Circular(s) => s.get_closest_point(xpos, ypos),
        }
    }

    /// Return the time of intersection for a moving circle vs this segment.
    pub fn intersect_with_ray(
        &self,
        xpos: f64,
        ypos: f64,
        dx: f64,
        dy: f64,
        radius: f64,
    ) -> f64 {
        match self {
            Segment::Linear(s) => s.intersect_with_ray(xpos, ypos, dx, dy, radius),
            Segment::Circular(s) => s.intersect_with_ray(xpos, ypos, dx, dy, radius),
        }
    }

    /// Return the bounding box (min_x, min_y, max_x, max_y).
    pub fn get_bounds(&self) -> (f64, f64, f64, f64) {
        match self {
            Segment::Linear(s) => s.bounds,
            Segment::Circular(s) => s.bounds,
        }
    }

    pub fn is_active(&self) -> bool {
        match self {
            Segment::Linear(s) => s.active,
            Segment::Circular(s) => s.active,
        }
    }
}

/// Linear collision segment (tiles, doors).
#[derive(Debug, Clone)]
pub struct LinearSegment {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    pub oriented: bool,
    pub active: bool,
    // Pre-computed direction vector
    pub px: f64,
    pub py: f64,
    pub seg_lensq: f64,
    // Cached bounding box
    pub bounds: (f64, f64, f64, f64),
}

impl LinearSegment {
    pub fn new(x1: f64, y1: f64, x2: f64, y2: f64, oriented: bool) -> Self {
        let px = x2 - x1;
        let py = y2 - y1;
        let mut seg_lensq = px * px + py * py;
        if seg_lensq == 0.0 {
            seg_lensq = 1e-9;
        }
        let bounds = (x1.min(x2), y1.min(y2), x1.max(x2), y1.max(y2));
        LinearSegment {
            x1,
            y1,
            x2,
            y2,
            oriented,
            active: true,
            px,
            py,
            seg_lensq,
            bounds,
        }
    }

    pub fn get_closest_point(&self, xpos: f64, ypos: f64) -> (bool, f64, f64) {
        let dx = xpos - self.x1;
        let dy = ypos - self.y1;
        let mut u = (dx * self.px + dy * self.py) / self.seg_lensq;
        u = u.clamp(0.0, 1.0);
        let a = self.x1 + u * self.px;
        let b = self.y1 + u * self.py;
        let is_back_facing = (dy * self.px - dx * self.py) < 0.0 && self.oriented;
        (is_back_facing, a, b)
    }

    pub fn intersect_with_ray(
        &self,
        xpos: f64,
        ypos: f64,
        dx: f64,
        dy: f64,
        radius: f64,
    ) -> f64 {
        let time1 =
            intersection::circle_vs_circle(xpos, ypos, dx, dy, self.x1, self.y1, radius);
        let time2 =
            intersection::circle_vs_circle(xpos, ypos, dx, dy, self.x2, self.y2, radius);
        let time3 = intersection::circle_vs_lineseg(
            xpos, ypos, dx, dy, self.x1, self.y1, self.x2, self.y2, radius,
        );
        time1.min(time2).min(time3)
    }
}

/// Circular (arc) collision segment (tile corners).
#[derive(Debug, Clone)]
pub struct CircularSegment {
    pub xpos: f64,
    pub ypos: f64,
    pub hor: f64,
    pub ver: f64,
    pub radius: f64,
    pub convex: bool,
    pub active: bool,
    // Extremity points of the arc
    pub p_hor: (f64, f64),
    pub p_ver: (f64, f64),
    // Cached bounding box
    pub bounds: (f64, f64, f64, f64),
}

impl CircularSegment {
    pub fn new(
        center_x: f64,
        center_y: f64,
        hor: f64,
        ver: f64,
        convex: bool,
        radius: f64,
    ) -> Self {
        let p_hor = (center_x + radius * hor, center_y);
        let p_ver = (center_x, center_y + radius * ver);
        let min_x = center_x.min(p_hor.0);
        let max_x = center_x.max(p_hor.0);
        let min_y = center_y.min(p_ver.1);
        let max_y = center_y.max(p_ver.1);
        CircularSegment {
            xpos: center_x,
            ypos: center_y,
            hor,
            ver,
            radius,
            convex,
            active: true,
            p_hor,
            p_ver,
            bounds: (min_x, min_y, max_x, max_y),
        }
    }

    pub fn get_closest_point(&self, xpos: f64, ypos: f64) -> (bool, f64, f64) {
        let dx = xpos - self.xpos;
        let dy = ypos - self.ypos;
        let mut is_back_facing = false;

        if dx * self.hor > 0.0 && dy * self.ver > 0.0 {
            let dist = (dx * dx + dy * dy).sqrt();
            if dist == 0.0 {
                let (a, b) = if dx * self.hor > dy * self.ver {
                    self.p_hor
                } else {
                    self.p_ver
                };
                return (is_back_facing, a, b);
            }
            let a = self.xpos + self.radius * dx / dist;
            let b = self.ypos + self.radius * dy / dist;
            is_back_facing = if self.convex {
                dist < self.radius
            } else {
                dist > self.radius
            };
            (is_back_facing, a, b)
        } else {
            let (a, b) = if dx * self.hor > dy * self.ver {
                self.p_hor
            } else {
                self.p_ver
            };
            (is_back_facing, a, b)
        }
    }

    pub fn intersect_with_ray(
        &self,
        xpos: f64,
        ypos: f64,
        dx: f64,
        dy: f64,
        radius: f64,
    ) -> f64 {
        let time1 = intersection::circle_vs_circle(
            xpos, ypos, dx, dy, self.p_hor.0, self.p_hor.1, radius,
        );
        let time2 = intersection::circle_vs_circle(
            xpos, ypos, dx, dy, self.p_ver.0, self.p_ver.1, radius,
        );
        let time3 = intersection::circle_vs_arc(
            xpos,
            ypos,
            dx,
            dy,
            self.xpos,
            self.ypos,
            self.hor,
            self.ver,
            self.radius,
            radius,
        );
        time1.min(time2).min(time3)
    }
}
