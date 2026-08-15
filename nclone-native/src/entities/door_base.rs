//! Geometry shared by every door type (regular 5, locked 6, trap 8).
//! Port of the parts of `EntityDoorBase.__init__` that are independent of the
//! door's activation rule (nclone/entity_classes/entity_door_base.py:56-92).
//!
//! A door owns two things beyond its switch: a 24px collision segment in the
//! shared segment pool, and a pair of half-cell grid edges in
//! [`crate::grid_edges::GridEdges`]. The segment stops the ninja; the grid edges
//! stop grid-walking entities (drones, thwumps). Both must flip together
//! whenever the door opens or closes -- a door that only flips its segment is
//! solid to the ninja but transparent to drones.

use crate::constants::TILE_SIZE;
use crate::geometry::map_orientation_to_vector;
use crate::segments::{LinearSegment, Segment};
use crate::types::{clamp_cell, Cell};

/// Half of a door's 24px span, and the offset used to locate its cell.
pub const DOOR_RADIUS: f64 = 12.0;

/// Doors at orientation 0 or 4 are vertical; everything else is horizontal.
#[inline]
pub fn door_is_vertical(orientation: u8) -> bool {
    orientation == 0 || orientation == 4
}

/// The tile cell a door's segment belongs to: the door position pushed back
/// `DOOR_RADIUS` along its orientation vector, floored to a cell and clamped.
#[inline]
pub fn door_cell(door_xpos: f64, door_ypos: f64, orientation: u8) -> Cell {
    let (vx, vy) = map_orientation_to_vector(orientation);
    let x = ((door_xpos - DOOR_RADIUS * vx) / TILE_SIZE).floor() as i32;
    let y = ((door_ypos - DOOR_RADIUS * vy) / TILE_SIZE).floor() as i32;
    clamp_cell(x, y)
}

/// The two half-cell grid edges a door covers, as `(vertical, x, y)` keys into
/// [`crate::grid_edges::GridEdges`].
///
/// Mirrors entity_door_base.py:66-92 exactly, including the `2 * (cell + 1)`
/// half-cell conversion -- note the `+ 1`, which differs from the plain
/// `2 * tile_coord` used when folding tiles into the static grids.
pub fn door_grid_edges(door_xpos: f64, door_ypos: f64, orientation: u8) -> [(bool, usize, usize); 2] {
    let cell = door_cell(door_xpos, door_ypos, orientation);
    let half_x = 2 * (cell.0 + 1);
    let half_y = 2 * (cell.1 + 1);
    if door_is_vertical(orientation) {
        [
            (true, half_x, half_y - 2),
            (true, half_x, half_y - 1),
        ]
    } else {
        [
            (false, half_x - 2, half_y),
            (false, half_x - 1, half_y),
        ]
    }
}

/// Build a door's 24px collision segment and the cell it belongs to.
///
/// Shared by all three door types. The segment is UNORIENTED (`oriented=false`):
/// a door blocks from both sides, unlike a tile edge.
pub fn make_door_segment(door_xcoord: i32, door_ycoord: i32, orientation: u8) -> (Segment, Cell) {
    let door_xpos = (door_xcoord * 6) as f64;
    let door_ypos = (door_ycoord * 6) as f64;
    let cell = door_cell(door_xpos, door_ypos, orientation);
    // Wind the segment from the door's ORIENTATION, not unconditionally in the
    // +x / +y direction.
    //
    // Always wind LOW-to-HIGH, and carry the door's facing separately.
    //
    // A door is unoriented, so the winding is inert for collision -- except
    // numerically, and that turns out to matter more than anything it decides.
    // `get_closest_point` evaluates `x1 + t/lensq * px`, and when `x1` is the
    // FAR end that ends in a cancelling subtraction: on a 24 px face wound
    // high-to-low it misses the query point by up to 2^-47 on 725 positions in
    // 20000, where low-to-high is exact on all 20000.
    //
    // A ninja standing on a door floor then gets `floor_normalized_x` of ~-7e-16
    // instead of 0, and the flat-ground test `hor_input * floor_normalized_x >=
    // 0` is a SIGN test -- so a denormal tilt sends flat ground down the uphill
    // branch and hands out `GROUND_ACCEL / 2`. On `MET-S-E-11-04` that is +0.0333
    // px/frame of free acceleration from frame 11, compounding into a death at
    // 619 on a demo the game completes.
    //
    // The winding used to be flipped to `-vec` because the dual laser reads its
    // seating normal off `(-py, px)`: scored against the game's `+0x68`/`+0x70`
    // over all 190 captured surface lasers, the unconditional winding scores
    // 168/190 with 18 sign errors and the flipped one 186/190 with none. That
    // was two jobs on one field. The laser now reads `door_normal`, which states
    // that same `-vec` directly, so both sides get what they need -- dropping
    // the flip *without* it costs 30 demos, all laser deaths.
    //
    // `NCLONE_AB_DOOR_WINDING` no longer changes anything.
    let vec = crate::geometry::map_orientation_to_vector(orientation);
    let (ax, ay, bx, by) = if door_is_vertical(orientation) {
        (door_xpos, door_ypos - DOOR_RADIUS, door_xpos, door_ypos + DOOR_RADIUS)
    } else {
        (door_xpos - DOOR_RADIUS, door_ypos, door_xpos + DOOR_RADIUS, door_ypos)
    };
    // `+ 0.0` clears the SIGN OF ZERO, and it is not cosmetic.
    //
    // `map_orientation_to_vector` returns f64 here, so an axis-aligned door's
    // zero component negates to `-0.0`. Python's returns an int for the same
    // orientations, and `-0` is just `0` -- so Python's door normal is
    // `(-1, +0.0)` where this was `(-1, -0.0)`.
    //
    // A dual laser seats on this normal and takes `atan2(ynormal, xnormal)` for
    // its beam angle: `atan2(+0.0, -1) = +pi` but `atan2(-0.0, -1) = -pi`, so
    // the beam tilts `sin(+-pi) = +-1.22e-16` in y. That is invisible until a
    // beam runs exactly `NINJA_RADIUS` from the ninja's centre, where the hit
    // test is a strict `<` -- then one sign misses and the other kills. On
    // `10257573` two lasers sit 750 px away at y=186 with the ninja at y=176,
    // and native killed on frame 9 where the real game and the Python port both
    // play on. 70 demos in the corpus turned on this bit.
    //
    // For every non-zero component `x + 0.0 == x` exactly, so this only ever
    // rewrites `-0.0` to `+0.0`.
    let seg = LinearSegment::new(ax, ay, bx, by, false)
        .with_door_normal(-vec.0 + 0.0, -vec.1 + 0.0);
    (Segment::Linear(seg), cell)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertical_door_covers_two_vertical_edges_in_one_column() {
        // Door at tile-cell (5, 7), orientation 0 (vertical, normal +x).
        // pos - 12*(1,0) lands in cell (4, 7) -> half (10, 16).
        let edges = door_grid_edges(5.0 * TILE_SIZE + 12.0, 7.0 * TILE_SIZE + 12.0, 0);
        assert!(edges.iter().all(|e| e.0), "vertical door -> vertical edges");
        assert_eq!(edges[0].1, edges[1].1, "same column");
        assert_eq!(edges[1].2, edges[0].2 + 1, "adjacent rows");
    }

    #[test]
    fn horizontal_door_covers_two_horizontal_edges_in_one_row() {
        let edges = door_grid_edges(5.0 * TILE_SIZE + 12.0, 7.0 * TILE_SIZE + 12.0, 2);
        assert!(edges.iter().all(|e| !e.0), "horizontal door -> horizontal edges");
        assert_eq!(edges[0].2, edges[1].2, "same row");
        assert_eq!(edges[1].1, edges[0].1 + 1, "adjacent columns");
    }
}
