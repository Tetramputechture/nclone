//! Spatial query functions for gathering segments and entities from the tile grid.
//! Direct port of gather_segments_from_region and gather_entities_from_neighbourhood from physics.py.

use crate::constants::{MAP_HEIGHT, MAP_WIDTH, TILE_SIZE};
use crate::segments::Segment;
use crate::types::Cell;

/// Gather active collision segments from tile cells overlapped by a rectangle.
pub fn gather_segments_from_region(
    segment_grid: &[[Vec<usize>; MAP_HEIGHT]; MAP_WIDTH],
    all_segments: &[Segment],
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
) -> Vec<usize> {
    let min_x = x1.min(x2);
    let min_y = y1.min(y2);
    let max_x = x1.max(x2);
    let max_y = y1.max(y2);

    let min_cell_x = (min_x / TILE_SIZE).floor().max(0.0).min((MAP_WIDTH - 1) as f64) as usize;
    let max_cell_x = (max_x / TILE_SIZE).floor().max(0.0).min((MAP_WIDTH - 1) as f64) as usize;
    let min_cell_y = (min_y / TILE_SIZE).floor().max(0.0).min((MAP_HEIGHT - 1) as f64) as usize;
    let max_cell_y = (max_y / TILE_SIZE).floor().max(0.0).min((MAP_HEIGHT - 1) as f64) as usize;

    let mut result = Vec::new();
    for xcell in min_cell_x..=max_cell_x {
        for ycell in min_cell_y..=max_cell_y {
            for &seg_idx in &segment_grid[xcell][ycell] {
                if all_segments[seg_idx].is_active() {
                    result.push(seg_idx);
                }
            }
        }
    }
    result
}

/// Gather entity indices from the 3x3 neighbourhood around a position.
pub fn gather_entities_from_neighbourhood(
    entity_grid: &[[Vec<usize>; MAP_HEIGHT]; MAP_WIDTH],
    xpos: f64,
    ypos: f64,
) -> Vec<usize> {
    let cx = ((xpos / TILE_SIZE).floor() as i32).clamp(0, (MAP_WIDTH - 1) as i32) as usize;
    let cy = ((ypos / TILE_SIZE).floor() as i32).clamp(0, (MAP_HEIGHT - 1) as i32) as usize;

    let x_start = if cx > 0 { cx - 1 } else { 0 };
    let x_end = (cx + 1).min(MAP_WIDTH - 1);
    let y_start = if cy > 0 { cy - 1 } else { 0 };
    let y_end = (cy + 1).min(MAP_HEIGHT - 1);

    let mut result = Vec::new();
    for x in x_start..=x_end {
        for y in y_start..=y_end {
            result.extend_from_slice(&entity_grid[x][y]);
        }
    }
    result
}

/// Get the cell for a world position, clamped to map bounds.
pub fn cell_from_pos(xpos: f64, ypos: f64) -> Cell {
    let cx = (xpos / TILE_SIZE).floor() as i32;
    let cy = (ypos / TILE_SIZE).floor() as i32;
    (
        cx.clamp(0, (MAP_WIDTH - 1) as i32) as usize,
        cy.clamp(0, (MAP_HEIGHT - 1) as i32) as usize,
    )
}
