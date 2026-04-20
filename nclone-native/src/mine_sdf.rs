//! Signed Distance Field (SDF) to mines for O(1) hazard lookups.
//!
//! Phase 4.5: Port of MineSignedDistanceField from Python.
//! Used for observation space features (escape direction).
//!
//! `MineThreatMap` (NPP-10): 3-channel 168×92 @ 6px. Uses an [`rstar::RTree`] spatial
//! index so each cell queries nearest deadly/safe mines in O(log n) instead of scanning
//! all mines (same role as spatial-hash pruning in the spec).

use pyo3::prelude::*;
use rstar::primitives::GeomWithData;
use rstar::RTree;

const SDF_CELL_SIZE: f64 = 6.0;
const SDF_WIDTH: usize = 176;
const SDF_HEIGHT: usize = 100;
const DANGER_RADIUS: f64 = 20.0;

/// Signed Distance Field to nearest deadly mine.
#[pyclass]
pub struct MineSDF {
    /// SDF grid: normalized distance [-1, 1] per 6px cell
    sdf_grid: Vec<f32>,
    /// Gradient: (dx, dy) escape direction per cell
    gradient_grid: Vec<(f32, f32)>,
}

#[pymethods]
impl MineSDF {
    #[new]
    fn new() -> Self {
        MineSDF {
            sdf_grid: vec![],
            gradient_grid: vec![],
        }
    }

    /// Build SDF from deadly mine positions (world coords).
    ///
    /// Args:
    ///   mine_positions: List of (x, y) in world pixels
    fn build_from_mines(&mut self, mine_positions: Vec<(f64, f64)>) {
        self.sdf_grid = vec![1.0_f32; SDF_WIDTH * SDF_HEIGHT];
        self.gradient_grid = vec![(0.0_f32, 0.0_f32); SDF_WIDTH * SDF_HEIGHT];

        if mine_positions.is_empty() {
            return;
        }

        for row in 0..SDF_HEIGHT {
            for col in 0..SDF_WIDTH {
                let cell_center_x = (col as f64 + 0.5) * SDF_CELL_SIZE;
                let cell_center_y = (row as f64 + 0.5) * SDF_CELL_SIZE;

                let mut min_dist = f64::INFINITY;
                let mut nearest_dx = 0.0_f64;
                let mut nearest_dy = 0.0_f64;

                for &(mine_x, mine_y) in &mine_positions {
                    let dx = cell_center_x - mine_x;
                    let dy = cell_center_y - mine_y;
                    let dist = (dx * dx + dy * dy).sqrt();

                    if dist < min_dist {
                        min_dist = dist;
                        nearest_dx = dx;
                        nearest_dy = dy;
                    }
                }

                let normalized_dist = if min_dist <= DANGER_RADIUS {
                    (min_dist / DANGER_RADIUS) - 1.0
                } else {
                    ((min_dist - DANGER_RADIUS) / (2.0 * DANGER_RADIUS)).min(1.0)
                };

                let idx = row * SDF_WIDTH + col;
                self.sdf_grid[idx] = normalized_dist as f32;

                if min_dist > 1e-6 {
                    self.gradient_grid[idx] = (
                        (nearest_dx / min_dist) as f32,
                        (nearest_dy / min_dist) as f32,
                    );
                }
            }
        }
    }

    /// Get SDF value at world position (nearest cell).
    fn get_sdf_value(&self, x: f64, y: f64) -> f64 {
        if self.sdf_grid.is_empty() {
            return 1.0;
        }
        let col = (x / SDF_CELL_SIZE) as i32;
        let row = (y / SDF_CELL_SIZE) as i32;
        let col = col.clamp(0, (SDF_WIDTH - 1) as i32);
        let row = row.clamp(0, (SDF_HEIGHT - 1) as i32);
        self.sdf_grid[row as usize * SDF_WIDTH + col as usize] as f64
    }

    /// Get bilinear interpolated SDF value.
    fn get_sdf_value_interp(&self, x: f64, y: f64) -> f64 {
        if self.sdf_grid.is_empty() {
            return 1.0;
        }
        let gx = x / SDF_CELL_SIZE;
        let gy = y / SDF_CELL_SIZE;
        let max_gx = (SDF_WIDTH - 1) as f64;
        let max_gy = (SDF_HEIGHT - 1) as f64;
        let gx = gx.clamp(0.0, max_gx);
        let gy = gy.clamp(0.0, max_gy);

        let ix0 = gx.floor() as usize;
        let iy0 = gy.floor() as usize;
        let ix1 = (ix0 + 1).min(SDF_WIDTH - 1);
        let iy1 = (iy0 + 1).min(SDF_HEIGHT - 1);
        let fx = gx - ix0 as f64;
        let fy = gy - iy0 as f64;

        let v00 = self.sdf_grid[iy0 * SDF_WIDTH + ix0] as f64;
        let v10 = self.sdf_grid[iy0 * SDF_WIDTH + ix1] as f64;
        let v01 = self.sdf_grid[iy1 * SDF_WIDTH + ix0] as f64;
        let v11 = self.sdf_grid[iy1 * SDF_WIDTH + ix1] as f64;

        v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy)
            + v01 * (1.0 - fx) * fy
            + v11 * fx * fy
    }

    /// Get gradient (escape direction) at world position.
    fn get_gradient(&self, x: f64, y: f64) -> (f64, f64) {
        if self.gradient_grid.is_empty() {
            return (0.0, 0.0);
        }
        let col = (x / SDF_CELL_SIZE) as i32;
        let row = (y / SDF_CELL_SIZE) as i32;
        let col = col.clamp(0, (SDF_WIDTH - 1) as i32);
        let row = row.clamp(0, (SDF_HEIGHT - 1) as i32);
        let (gx, gy) = self.gradient_grid[row as usize * SDF_WIDTH + col as usize];
        (gx as f64, gy as f64)
    }

    /// Get bilinear interpolated gradient.
    fn get_gradient_interp(&self, x: f64, y: f64) -> (f64, f64) {
        if self.gradient_grid.is_empty() {
            return (0.0, 0.0);
        }
        let gx = x / SDF_CELL_SIZE;
        let gy = y / SDF_CELL_SIZE;
        let max_gx = (SDF_WIDTH - 1) as f64;
        let max_gy = (SDF_HEIGHT - 1) as f64;
        let gx = gx.clamp(0.0, max_gx);
        let gy = gy.clamp(0.0, max_gy);

        let ix0 = gx.floor() as usize;
        let iy0 = gy.floor() as usize;
        let ix1 = (ix0 + 1).min(SDF_WIDTH - 1);
        let iy1 = (iy0 + 1).min(SDF_HEIGHT - 1);
        let fx = gx - ix0 as f64;
        let fy = gy - iy0 as f64;

        let (g00x, g00y) = self.gradient_grid[iy0 * SDF_WIDTH + ix0];
        let (g10x, g10y) = self.gradient_grid[iy0 * SDF_WIDTH + ix1];
        let (g01x, g01y) = self.gradient_grid[iy1 * SDF_WIDTH + ix0];
        let (g11x, g11y) = self.gradient_grid[iy1 * SDF_WIDTH + ix1];

        let out_x = g00x as f64 * (1.0 - fx) * (1.0 - fy)
            + g10x as f64 * fx * (1.0 - fy)
            + g01x as f64 * (1.0 - fx) * fy
            + g11x as f64 * fx * fy;
        let out_y = g00y as f64 * (1.0 - fx) * (1.0 - fy)
            + g10y as f64 * fx * (1.0 - fy)
            + g01y as f64 * (1.0 - fx) * fy
            + g11y as f64 * fx * fy;

        let mag = (out_x * out_x + out_y * out_y).sqrt();
        if mag > 1e-9 {
            (out_x / mag, out_y / mag)
        } else {
            (out_x, out_y)
        }
    }
}

// --- Mine threat map (L1.2, NPP-10): 3 × 168 × 92 @ 6px ---
//
// NPP-46 (Layer 1 toggle-mines chunk): adds a 1-channel `hazard_map`
// view on top of the existing 3-channel `MineThreatMap`. The 1-channel path
// exposes only the deadly-mine SDF (channel 0 of the 3-channel map) and is
// the shape the Layer 1 policy consumes. The 3-channel path is kept as a
// legacy fallback until all downstream consumers migrate.

const THREAT_CELL_SIZE: f64 = 6.0;
const THREAT_WIDTH: usize = 168;
const THREAT_HEIGHT: usize = 92;
const MINE_KILL_RADIUS_PX: f64 = 14.0;
const SDF_SCALE_PX: f64 = 24.0;

/// 3-channel mine threat map: deadly SDF, safe SDF, passability (sigmoid of deadly).
///
/// Output layout (channel-first, row-major within each channel): `[deadly | safe | pass]`,
/// each block length `THREAT_WIDTH * THREAT_HEIGHT`.
///
/// NPP-46 adds `compute_hazard_map{,_if_changed}` helpers that return only the
/// 1-channel deadly SDF buffer (`THREAT_WIDTH * THREAT_HEIGHT` floats).
#[pyclass]
pub struct MineThreatMap {
    last_deadly: Vec<(f64, f64)>,
    last_safe: Vec<(f64, f64)>,
    last_output: Vec<f32>,
    /// Last deadly-only inputs for the 1-channel `hazard_map` path (NPP-46).
    last_hazard_deadly: Vec<(f64, f64)>,
    /// Last 1-channel deadly SDF output (NPP-46). Empty until first compute.
    last_hazard_output: Vec<f32>,
}

impl MineThreatMap {
    fn positions_equal(a: &[(f64, f64)], b: &[(f64, f64)]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter()
            .zip(b.iter())
            .all(|(p, q)| p.0 == q.0 && p.1 == q.1)
    }

    fn build_tree(positions: &[(f64, f64)]) -> Option<RTree<GeomWithData<[f64; 2], ()>>> {
        if positions.is_empty() {
            return None;
        }
        let items: Vec<_> = positions
            .iter()
            .map(|(x, y)| GeomWithData::new([*x, *y], ()))
            .collect();
        Some(RTree::bulk_load(items))
    }

    fn nearest_dist_sq(cx: f64, cy: f64, tree: &Option<RTree<GeomWithData<[f64; 2], ()>>>) -> f64 {
        let Some(t) = tree else {
            return f64::INFINITY;
        };
        let Some(nearest) = t.nearest_neighbor(&[cx, cy]) else {
            return f64::INFINITY;
        };
        let p = nearest.geom();
        let dx = cx - p[0];
        let dy = cy - p[1];
        dx * dx + dy * dy
    }

    fn sdf_from_min_dist(min_dist: f64) -> f64 {
        if !min_dist.is_finite() {
            return 1.0;
        }
        let raw = (min_dist - MINE_KILL_RADIUS_PX) / SDF_SCALE_PX;
        raw.clamp(-1.0, 1.0)
    }

    fn passability_from_deadly_sdf(deadly_sdf: f32) -> f32 {
        // passability = sigmoid((deadly_sdf * 24) / 3) = sigmoid(deadly_sdf * 8)
        let x = deadly_sdf * (24.0_f32 / 3.0_f32);
        1.0 / (1.0 + (-x).exp())
    }

    /// 1-channel deadly SDF buffer (NPP-46).
    ///
    /// Returns `THREAT_WIDTH * THREAT_HEIGHT` floats in row-major order, matching
    /// channel 0 of `compute_threat_map_inner` bit-for-bit. Callers shape the
    /// result into `(1, THREAT_HEIGHT, THREAT_WIDTH)` for the `hazard_map` key.
    fn compute_hazard_map_inner(deadly_positions: &[(f64, f64)]) -> Vec<f32> {
        let deadly_tree = Self::build_tree(deadly_positions);
        let mut out = vec![0.0_f32; THREAT_WIDTH * THREAT_HEIGHT];

        for row in 0..THREAT_HEIGHT {
            for col in 0..THREAT_WIDTH {
                let cx = (col as f64 + 0.5) * THREAT_CELL_SIZE;
                let cy = (row as f64 + 0.5) * THREAT_CELL_SIZE;
                let idx = row * THREAT_WIDTH + col;

                let d_sq = Self::nearest_dist_sq(cx, cy, &deadly_tree);
                let min_d = d_sq.sqrt();
                out[idx] = Self::sdf_from_min_dist(min_d) as f32;
            }
        }
        out
    }

    fn compute_threat_map_inner(
        deadly_positions: &[(f64, f64)],
        safe_positions: &[(f64, f64)],
    ) -> Vec<f32> {
        let deadly_tree = Self::build_tree(deadly_positions);
        let safe_tree = Self::build_tree(safe_positions);
        let chunk = THREAT_WIDTH * THREAT_HEIGHT;
        let mut out = vec![0.0_f32; 3 * chunk];

        for row in 0..THREAT_HEIGHT {
            for col in 0..THREAT_WIDTH {
                let cx = (col as f64 + 0.5) * THREAT_CELL_SIZE;
                let cy = (row as f64 + 0.5) * THREAT_CELL_SIZE;
                let idx = row * THREAT_WIDTH + col;

                let d_sq = Self::nearest_dist_sq(cx, cy, &deadly_tree);
                let min_d = d_sq.sqrt();
                let deadly_sdf = Self::sdf_from_min_dist(min_d) as f32;
                out[idx] = deadly_sdf;

                let s_sq = Self::nearest_dist_sq(cx, cy, &safe_tree);
                let min_s = s_sq.sqrt();
                let safe_sdf = Self::sdf_from_min_dist(min_s) as f32;
                out[chunk + idx] = safe_sdf;

                out[2 * chunk + idx] = Self::passability_from_deadly_sdf(deadly_sdf);
            }
        }
        out
    }
}

#[pymethods]
impl MineThreatMap {
    #[new]
    fn new() -> Self {
        MineThreatMap {
            last_deadly: Vec::new(),
            last_safe: Vec::new(),
            last_output: Vec::new(),
            last_hazard_deadly: Vec::new(),
            last_hazard_output: Vec::new(),
        }
    }

    /// Always recompute and return flat `[3 * H * W]` (channel-first).
    fn compute_threat_map(
        &mut self,
        deadly_positions: Vec<(f64, f64)>,
        safe_positions: Vec<(f64, f64)>,
    ) -> Vec<f32> {
        let out = Self::compute_threat_map_inner(&deadly_positions, &safe_positions);
        self.last_deadly = deadly_positions;
        self.last_safe = safe_positions;
        self.last_output = out.clone();
        out
    }

    /// Recompute only if deadly or safe mine positions changed vs last call.
    /// Returns `None` when unchanged (Python may reuse cached ndarray).
    fn compute_threat_map_if_changed(
        &mut self,
        deadly_positions: Vec<(f64, f64)>,
        safe_positions: Vec<(f64, f64)>,
    ) -> Option<Vec<f32>> {
        if !self.last_output.is_empty()
            && Self::positions_equal(&self.last_deadly, &deadly_positions)
            && Self::positions_equal(&self.last_safe, &safe_positions)
        {
            return None;
        }
        let out = Self::compute_threat_map_inner(&deadly_positions, &safe_positions);
        self.last_deadly = deadly_positions;
        self.last_safe = safe_positions;
        self.last_output = out.clone();
        Some(out)
    }

    /// NPP-46: compute the 1-channel deadly-mine SDF buffer (length `H*W`).
    ///
    /// Always recomputes; returns a flat row-major buffer of floats in [-1, 1].
    /// Python reshapes to `(1, THREAT_HEIGHT, THREAT_WIDTH)` for `hazard_map`.
    fn compute_hazard_map(&mut self, deadly_positions: Vec<(f64, f64)>) -> Vec<f32> {
        let out = Self::compute_hazard_map_inner(&deadly_positions);
        self.last_hazard_deadly = deadly_positions;
        self.last_hazard_output = out.clone();
        out
    }

    /// NPP-46: recompute the 1-channel deadly-mine SDF only if deadly positions changed.
    fn compute_hazard_map_if_changed(
        &mut self,
        deadly_positions: Vec<(f64, f64)>,
    ) -> Option<Vec<f32>> {
        if !self.last_hazard_output.is_empty()
            && Self::positions_equal(&self.last_hazard_deadly, &deadly_positions)
        {
            return None;
        }
        let out = Self::compute_hazard_map_inner(&deadly_positions);
        self.last_hazard_deadly = deadly_positions;
        self.last_hazard_output = out.clone();
        Some(out)
    }

    /// NPP-46: width of the `hazard_map` grid (168).
    #[staticmethod]
    fn hazard_map_width() -> usize {
        THREAT_WIDTH
    }

    /// NPP-46: height of the `hazard_map` grid (92).
    #[staticmethod]
    fn hazard_map_height() -> usize {
        THREAT_HEIGHT
    }

    /// NPP-46: deadly kill radius in world pixels (14.0).
    #[staticmethod]
    fn hazard_kill_radius() -> f64 {
        MINE_KILL_RADIUS_PX
    }

    /// NPP-46: SDF normalization scale in world pixels (24.0).
    #[staticmethod]
    fn hazard_sdf_scale() -> f64 {
        SDF_SCALE_PX
    }

    /// NPP-46: cell size in world pixels (6.0).
    #[staticmethod]
    fn hazard_cell_size() -> f64 {
        THREAT_CELL_SIZE
    }
}
