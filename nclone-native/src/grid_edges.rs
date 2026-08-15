//! Half-cell grid-edge occupancy, and the row/column emptiness queries built on
//! it. Port of the four `*_grid_edge_dic` dictionaries on the Python simulator
//! plus `is_empty_row` / `is_empty_column` (nclone/physics.py:158-187).
//!
//! Grid-walking entities (drones, thwumps, shove thwumps) do NOT collide against
//! tile segments. They ask a much cheaper question -- "is the half-cell edge I am
//! about to cross solid?" -- answered by these grids.
//!
//! Two independent layers, both consulted by every query:
//!   * `hor` / `ver`  -- STATIC, derived from the tile grid at load.
//!   * `door_hor` / `door_ver` -- DYNAMIC counters, incremented while a door is
//!     closed and decremented when it opens. A counter, not a flag, because
//!     several doors can cover the same half-edge.
//!
//! Dimensions mirror the Python dicts exactly: `hor` is 88x51, `ver` is 89x50.
//! Build-time keys are `(2*tile_x + i, 2*tile_y + j)` which stay in range for
//! every tile, so no clamping is needed there; queries clamp via
//! `clamp_half_cell` the way Python does.

use crate::map::tiles::{TILE_EDGE_GRID_TERNARY, TILE_GRID_EDGE_MAP};
use crate::types::clamp_half_cell;

pub const HOR_W: usize = 88;
pub const HOR_H: usize = 51;
pub const VER_W: usize = 89;
pub const VER_H: usize = 50;

#[derive(Clone)]
pub struct GridEdges {
    /// Static horizontal half-edges from tiles, indexed `[x][y]`.
    pub hor: Box<[[u8; HOR_H]; HOR_W]>,
    /// Static vertical half-edges from tiles, indexed `[x][y]`.
    pub ver: Box<[[u8; VER_H]; VER_W]>,
    /// Closed-door horizontal half-edge counters.
    pub door_hor: Box<[[i16; HOR_H]; HOR_W]>,
    /// Closed-door vertical half-edge counters.
    pub door_ver: Box<[[i16; VER_H]; VER_W]>,
}

impl Default for GridEdges {
    fn default() -> Self {
        Self::new()
    }
}

impl GridEdges {
    /// Borders solid, interior clear -- matching the dict comprehensions in
    /// `Simulator.reset_map_tile_data` (nclone/nsim.py:333-347). Tile processing
    /// then folds over the top of this with `combine_edge`, which is `max` unless
    /// two solid tiles collide, so the borders now survive; under the old toggle
    /// a border tile writing 1 over the pre-seeded 1 erased it.
    pub fn new() -> Self {
        let mut hor = Box::new([[0u8; HOR_H]; HOR_W]);
        for col in hor.iter_mut() {
            col[0] = 1;
            col[50] = 1;
        }
        let mut ver = Box::new([[0u8; VER_H]; VER_W]);
        for y in 0..VER_H {
            ver[0][y] = 1;
            ver[88][y] = 1;
        }
        GridEdges {
            hor,
            ver,
            door_hor: Box::new([[0i16; HOR_H]; HOR_W]),
            door_ver: Box::new([[0i16; VER_H]; VER_W]),
        }
    }

    /// Fold one tile's grid-edge contribution into the shared edge grid.
    ///
    /// `FUN_1050CE60` (`docs/decomp/TILE_SEGMENT_TABLES.md` §7) combines as
    ///
    /// ```text
    /// new = (old == 2 && v == 2) ? 0 : max(old, v)
    /// ```
    ///
    /// An edge cancels ONLY when two solid tiles both claim it -- the shared face
    /// between neighbours, which is interior and not a wall. The binary stores 2
    /// for "solid"; this port stores 1, so the test is just `old != 0 && v != 0`.
    ///
    /// This was a plain toggle (`if cur == 0 { new } else { 0 }`), which also
    /// cleared on `cur=1, new=0` -- so every solid tile's OUTER edge was wiped by
    /// the empty tile beside it and the edge grid came out empty. Only the
    /// drone/floor-guard/thwump cell walks read it, and they simply never saw a
    /// wall. Round 9 §1.4.
    /// The binary's tables are **ternary** -- `0` none, `1` a partial face (a
    /// slope or a curve), `2` a full flat one -- and it reads them at
    /// `outer*4 + inner*0xC` bytes, i.e. **`outer + 3*inner`** int32s, where
    /// `outer` is the 3-valued axis. The port used `2*outer + inner` over a 0/1
    /// table, which is wrong twice over; see `TILE_EDGE_GRID_TERNARY` and the
    /// handoff §10. Mirrors `nclone/utils/tile_segment_factory.py`.
    pub fn apply_tile(&mut self, tile_id: u8, xcoord: usize, ycoord: usize) {
        let t = tile_id as usize;
        if crate::ab::ab().legacy_edges {
            if t >= TILE_GRID_EDGE_MAP.len() {
                return;
            }
            let edges = &TILE_GRID_EDGE_MAP[t];
            for y_loop_idx in 0..3usize {
                for x_loop_idx in 0..2usize {
                    let cur = &mut self.hor[2 * xcoord + x_loop_idx][2 * ycoord + y_loop_idx];
                    *cur = combine_edge(*cur, edges[2 * y_loop_idx + x_loop_idx]);
                }
            }
            for x_loop_idx in 0..3usize {
                for y_loop_idx in 0..2usize {
                    let cur = &mut self.ver[2 * xcoord + x_loop_idx][2 * ycoord + y_loop_idx];
                    *cur = combine_edge(*cur, edges[2 * x_loop_idx + y_loop_idx + 6]);
                }
            }
            return;
        }
        if t >= TILE_EDGE_GRID_TERNARY.len() {
            return; // unknown tiles contribute nothing, as in Python
        }
        let (hor_row, ver_row) = &TILE_EDGE_GRID_TERNARY[t];

        // Horizontal edges: 3 rows (top / interior / bottom) x 2 columns.
        for y_loop_idx in 0..3usize {
            for x_loop_idx in 0..2usize {
                let cur = &mut self.hor[2 * xcoord + x_loop_idx][2 * ycoord + y_loop_idx];
                *cur = combine_edge(*cur, hor_row[y_loop_idx + 3 * x_loop_idx]);
            }
        }
        // Vertical edges: 3 columns (left / interior / right) x 2 rows.
        for x_loop_idx in 0..3usize {
            for y_loop_idx in 0..2usize {
                let cur = &mut self.ver[2 * xcoord + x_loop_idx][2 * ycoord + y_loop_idx];
                *cur = combine_edge(*cur, ver_row[x_loop_idx + 3 * y_loop_idx]);
            }
        }
    }

    /// Adjust the door counters for one half-edge. `delta` is `+1` when a door
    /// closes and `-1` when it opens, mirroring `EntityDoorBase.change_state`
    /// (nclone/entity_classes/entity_door_base.py:104-112).
    pub fn adjust_door_edge(&mut self, vertical: bool, x: usize, y: usize, delta: i16) {
        if vertical {
            if x < VER_W && y < VER_H {
                self.door_ver[x][y] += delta;
            }
        } else if x < HOR_W && y < HOR_H {
            self.door_hor[x][y] += delta;
        }
    }

    /// Reset every door counter to zero (used on `fast_reset`, before each door
    /// re-registers its closed state).
    pub fn clear_door_edges(&mut self) {
        for col in self.door_hor.iter_mut() {
            col.fill(0);
        }
        for col in self.door_ver.iter_mut() {
            col.fill(0);
        }
    }

    /// True if the row of half cells has no solid horizontal edge in direction
    /// `dir`. Port of `is_empty_row` (nclone/physics.py:158).
    ///
    /// Python returns `None` for a `dir` other than +/-1, which every caller
    /// treats as falsy (i.e. "blocked"); returning `false` matches that.
    /// The raw ternary value of one horizontal half-edge, clamped to the border
    /// like every other probe. Callers that need `== 2` (the floor guard's
    /// walkability test) must see the value, not a boolean -- `is_empty_row`
    /// collapses `1` and `2` and cannot answer it. Doors are deliberately not
    /// consulted: `FUN_1040B900`'s floor test reads only the static array
    /// `grid_3248C0`, unlike its wall test which reads both.
    pub fn hor_at(&self, xcoord: i32, ycoord: i32) -> u8 {
        let (cx, cy) = clamp_half_cell(xcoord, ycoord);
        self.hor[cx][cy]
    }

    pub fn is_empty_row(&self, xcoord1: i32, xcoord2: i32, ycoord: i32, dir: i32) -> bool {
        let y = match dir {
            1 => ycoord + 1,
            -1 => ycoord,
            _ => return false,
        };
        for xcoord in xcoord1..=xcoord2 {
            let (cx, cy) = clamp_half_cell(xcoord, y);
            if self.hor[cx][cy] != 0 || self.door_hor[cx][cy] != 0 {
                return false;
            }
        }
        true
    }

    /// True if the column of half cells has no solid vertical edge in direction
    /// `dir`. Port of `is_empty_column` (nclone/physics.py:174).
    pub fn is_empty_column(&self, xcoord: i32, ycoord1: i32, ycoord2: i32, dir: i32) -> bool {
        let x = match dir {
            1 => xcoord + 1,
            -1 => xcoord,
            _ => return false,
        };
        for ycoord in ycoord1..=ycoord2 {
            let (cx, cy) = clamp_half_cell(x, ycoord);
            if self.ver[cx][cy] != 0 || self.door_ver[cx][cy] != 0 {
                return false;
            }
        }
        true
    }
}

/// `new = (old == 2 && v == 2) ? 0 : max(old, v)`.
///
/// Cancel only when two **full** faces claim the edge -- the interior face
/// between solid neighbours. Two PARTIAL faces (`1`) meeting do not cancel, and
/// that asymmetry is the whole reason the grid is ternary. Mirrors
/// `nclone/utils/tile_segment_factory.py:_combine_edge`.
#[inline]
fn combine_edge(old: u8, new: u8) -> u8 {
    if crate::ab::ab().legacy_edges {
        // The legacy table is 0/1, so `== 2` would never cancel anything and the
        // flag would not be a revert at all. Old test with the old table.
        return if old != 0 && new != 0 { 0 } else { old.max(new) };
    }
    if old == 2 && new == 2 {
        0
    } else {
        old.max(new)
    }
}
