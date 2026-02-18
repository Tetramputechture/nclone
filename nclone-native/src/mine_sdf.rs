//! Signed Distance Field (SDF) to mines for O(1) hazard lookups.
//!
//! Phase 4.5: Port of MineSignedDistanceField from Python.
//! Used for observation space features (escape direction).

use pyo3::prelude::*;

const SDF_CELL_SIZE: f64 = 12.0;
const SDF_WIDTH: usize = 88;
const SDF_HEIGHT: usize = 50;
const DANGER_RADIUS: f64 = 20.0;

/// Signed Distance Field to nearest deadly mine.
#[pyclass]
pub struct MineSDF {
    /// SDF grid: normalized distance [-1, 1] per 12px cell
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
