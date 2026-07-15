//! Mine threat map (NPP-10) and 1-channel deadly-mine SDF (NPP-46).
//!
//! `MineThreatMap` produces either a 3-channel (`deadly_sdf`, `safe_sdf`,
//! `passability`) or 1-channel (`deadly_sdf`) buffer at 168x92 on a 6px grid.
//! The Layer 1 policy consumes the 1-channel `hazard_map` view. An
//! [`rstar::RTree`] spatial index gives per-cell nearest-neighbor queries in
//! O(log n), and the 1-channel path recomputes only cells within the SDF
//! effective radius of toggled mines (see `compute_hazard_map_if_changed`).

use pyo3::prelude::*;
use rstar::primitives::GeomWithData;
use rstar::RTree;

const THREAT_CELL_SIZE: f64 = 6.0;
const THREAT_WIDTH: usize = 168;
const THREAT_HEIGHT: usize = 92;
const MINE_KILL_RADIUS_PX: f64 = 14.0;
const SDF_SCALE_PX: f64 = 24.0;
/// Raw sim coordinates include a 1-tile (24 px) solid border; the playable
/// area is [24, 1032) x [24, 576). The 168x92 grid covers exactly the playable
/// region, so cell (col, row) centers at `24 + (i + 0.5) * 6` in raw sim px —
/// co-registered with `tile_grid` / `entity_overlays` (which use the
/// border-stripped frame). Mine/ninja inputs stay in raw sim px.
const PLAYFIELD_OFFSET_PX: f64 = 24.0;

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
    /// NPP-48: profiling counters for hazard_map cache behaviour.
    hazard_cache_hits: u64,
    hazard_cache_misses: u64,
    /// Cell count recomputed on last miss (full grid = THREAT_WIDTH * THREAT_HEIGHT).
    hazard_cells_recomputed: usize,
    /// Reserved for future partial-incremental hazard updates (NPP-48); API hook only.
    #[allow(dead_code)]
    hazard_force_full: bool,
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
                let cx = PLAYFIELD_OFFSET_PX + (col as f64 + 0.5) * THREAT_CELL_SIZE;
                let cy = PLAYFIELD_OFFSET_PX + (row as f64 + 0.5) * THREAT_CELL_SIZE;
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
                let cx = PLAYFIELD_OFFSET_PX + (col as f64 + 0.5) * THREAT_CELL_SIZE;
                let cy = PLAYFIELD_OFFSET_PX + (row as f64 + 0.5) * THREAT_CELL_SIZE;
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
            hazard_cache_hits: 0,
            hazard_cache_misses: 0,
            hazard_cells_recomputed: 0,
            hazard_force_full: false,
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
        self.hazard_cells_recomputed = THREAT_WIDTH * THREAT_HEIGHT;
        self.last_hazard_deadly = deadly_positions;
        self.last_hazard_output = out.clone();
        out
    }

    /// NPP-46/48: recompute the 1-channel deadly-mine SDF only if deadly positions changed.
    ///
    /// When the deadly set changes, only cells within the SDF effective radius
    /// (``MINE_KILL_RADIUS_PX + SDF_SCALE_PX = 38 px``) of a toggled mine can
    /// actually change — beyond that radius the SDF saturates to ``1.0`` from
    /// either the old or new nearest neighbor. The incremental path rebuilds
    /// only those dirty cells, which is a ~60x reduction vs the full grid for
    /// the typical single-mine-toggle case. Calling ``set_hazard_force_full(true)``
    /// disables the fast path (for parity tests).
    fn compute_hazard_map_if_changed(
        &mut self,
        deadly_positions: Vec<(f64, f64)>,
    ) -> Option<Vec<f32>> {
        if !self.last_hazard_output.is_empty()
            && Self::positions_equal(&self.last_hazard_deadly, &deadly_positions)
        {
            self.hazard_cache_hits += 1;
            return None;
        }
        self.hazard_cache_misses += 1;

        // Cold start or forced full: compute the entire grid.
        if self.last_hazard_output.is_empty() || self.hazard_force_full {
            let out = Self::compute_hazard_map_inner(&deadly_positions);
            self.hazard_cells_recomputed = THREAT_WIDTH * THREAT_HEIGHT;
            self.last_hazard_deadly = deadly_positions;
            self.last_hazard_output = out.clone();
            return Some(out);
        }

        // Incremental path: only cells within EFFECT_RADIUS of a toggled mine
        // can change. Toggle = symmetric difference of prev and curr position
        // sets (positions themselves never move — only `state` toggles cause
        // entities to appear/disappear from the deadly set).
        let prev: std::collections::HashSet<(u64, u64)> = self
            .last_hazard_deadly
            .iter()
            .map(|&(x, y)| (x.to_bits(), y.to_bits()))
            .collect();
        let curr: std::collections::HashSet<(u64, u64)> = deadly_positions
            .iter()
            .map(|&(x, y)| (x.to_bits(), y.to_bits()))
            .collect();
        let changed_mines: Vec<(f64, f64)> = prev
            .symmetric_difference(&curr)
            .map(|&(xb, yb)| (f64::from_bits(xb), f64::from_bits(yb)))
            .collect();

        // Defensive fallback: if the diff is somehow larger than a full rebuild
        // (e.g., entire set replaced), just recompute the full grid.
        if changed_mines.len() * 256 > THREAT_WIDTH * THREAT_HEIGHT {
            let out = Self::compute_hazard_map_inner(&deadly_positions);
            self.hazard_cells_recomputed = THREAT_WIDTH * THREAT_HEIGHT;
            self.last_hazard_deadly = deadly_positions;
            self.last_hazard_output = out.clone();
            return Some(out);
        }

        // Brute-force nearest over the (small) current deadly set is faster
        // than rstar for the dirty-cell count we touch here (<~200 cells,
        // <~50 mines in practice). It also avoids the R-tree build per call.
        let effect_radius = MINE_KILL_RADIUS_PX + SDF_SCALE_PX;
        let effect_radius_sq = effect_radius * effect_radius;
        let mut cells_recomputed: usize = 0;

        for &(mx, my) in &changed_mines {
            let col_lo =
                ((mx - PLAYFIELD_OFFSET_PX - effect_radius) / THREAT_CELL_SIZE - 0.5).floor();
            let col_hi =
                ((mx - PLAYFIELD_OFFSET_PX + effect_radius) / THREAT_CELL_SIZE - 0.5).ceil();
            let row_lo =
                ((my - PLAYFIELD_OFFSET_PX - effect_radius) / THREAT_CELL_SIZE - 0.5).floor();
            let row_hi =
                ((my - PLAYFIELD_OFFSET_PX + effect_radius) / THREAT_CELL_SIZE - 0.5).ceil();
            let col_min = col_lo.max(0.0) as usize;
            let col_max = col_hi.min((THREAT_WIDTH - 1) as f64).max(0.0) as usize;
            let row_min = row_lo.max(0.0) as usize;
            let row_max = row_hi.min((THREAT_HEIGHT - 1) as f64).max(0.0) as usize;

            for row in row_min..=row_max {
                for col in col_min..=col_max {
                    let cx = PLAYFIELD_OFFSET_PX + (col as f64 + 0.5) * THREAT_CELL_SIZE;
                    let cy = PLAYFIELD_OFFSET_PX + (row as f64 + 0.5) * THREAT_CELL_SIZE;
                    let dx = cx - mx;
                    let dy = cy - my;
                    if dx * dx + dy * dy > effect_radius_sq {
                        continue;
                    }
                    let mut min_d_sq = f64::INFINITY;
                    for &(px, py) in &deadly_positions {
                        let ex = cx - px;
                        let ey = cy - py;
                        let d2 = ex * ex + ey * ey;
                        if d2 < min_d_sq {
                            min_d_sq = d2;
                        }
                    }
                    let idx = row * THREAT_WIDTH + col;
                    self.last_hazard_output[idx] =
                        Self::sdf_from_min_dist(min_d_sq.sqrt()) as f32;
                    cells_recomputed += 1;
                }
            }
        }

        self.hazard_cells_recomputed = cells_recomputed;
        self.last_hazard_deadly = deadly_positions;
        Some(self.last_hazard_output.clone())
    }

    /// Reset NPP-48 hazard cache counters (not the cached grid).
    fn reset_hazard_cache_stats(&mut self) {
        self.hazard_cache_hits = 0;
        self.hazard_cache_misses = 0;
        self.hazard_cells_recomputed = 0;
    }

    /// When set, the next recompute path still fills the full grid (placeholder for incremental parity tests).
    fn set_hazard_force_full(&mut self, v: bool) {
        self.hazard_force_full = v;
    }

    /// Returns ``(hazard_cache_hits, hazard_cache_misses, hazard_cells_recomputed)``.
    fn get_hazard_cache_stats(&self) -> (u64, u64, usize) {
        (
            self.hazard_cache_hits,
            self.hazard_cache_misses,
            self.hazard_cells_recomputed,
        )
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

    /// Raw-sim-px offset of hazard cell (0, 0): the playable area starts at
    /// 24 px (the sim's 1-tile solid border). Cell (col, row) center =
    /// `offset + (i + 0.5) * cell_size` in raw sim px.
    #[staticmethod]
    fn hazard_playfield_offset() -> f64 {
        PLAYFIELD_OFFSET_PX
    }
}
