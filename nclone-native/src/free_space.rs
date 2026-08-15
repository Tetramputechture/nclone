//! Free-space grid cost-to-go floods (port of `nclone/value_function/free_space_distance.py`).
//!
//! Two edge-cost models over the same 6px sub-cell grid, both seeded from a
//! goal's activation disc and both consuming the caller's precomputed masks:
//!
//! * **geometric** — `1.0` orthogonal / `1.41` diagonal, mine-soft scaled. Pure
//!   pixel distance. Bit-identical to the Python implementation it replaces
//!   (same operation order, same `(cost, x, y)` heap tie-break).
//! * **physics** — direction-dependent costs in FRAMES, plus a climb rule that
//!   forbids rising into open air more than one jump above the floor beneath it.
//!   Falls are cheap and one-way, wall-adjacency is the only unbounded climb.
//!
//! Deliberately takes `free` / `mden` / the four blocked-edge masks as inputs
//! rather than deriving them: building those needs `SUBNODE_VALIDITY_TABLE` and
//! the tile collision tables, and duplicating that geometry here would open a
//! parity seam against the simulator for no throughput gain (the floods are the
//! dominant cost, the masks are not). The two derived climb fields DO live here
//! — they depend on `free` alone, so there is nothing to drift from.

use crate::map::tiles::{
    TILE_EDGE_GRID_TERNARY, TILE_SEGMENT_CIRCULAR_MAP, TILE_SEGMENT_DIAG_MAP,
};
use pyo3::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

const TILE_PX: f64 = 24.0;
const TILES_W: usize = 42;
const TILES_H: usize = 23;
/// Polyline segments approximating each quarter-circle arc. Must match the
/// Python `_ARC_CHORDS`, or the two implementations block different steps.
const ARC_CHORDS: usize = 6;

/// The 12 grid-edge half-segments in tile-local px, indexed exactly as
/// `TILE_GRID_EDGE_MAP`: first 6 horizontal (y in {0,12,24}, left then right),
/// last 6 vertical (x in {0,12,24}, top then bottom).
const GRID_EDGE_SEGS: [(f64, f64, f64, f64); 12] = [
    (0., 0., 12., 0.),
    (12., 0., 24., 0.),
    (0., 12., 12., 12.),
    (12., 12., 24., 12.),
    (0., 24., 12., 24.),
    (12., 24., 24., 24.),
    (0., 0., 0., 12.),
    (0., 12., 0., 24.),
    (12., 0., 12., 12.),
    (12., 12., 12., 24.),
    (24., 0., 24., 12.),
    (24., 12., 24., 24.),
];

/// Solid collision segments for a tile in tile-local px.
///
/// Union of the orthogonal grid-edge walls, the diagonal slope segment and a
/// chord approximation of the curved segment — the same three tables the
/// simulator collides against, so this cannot drift from the physics. Empty tile
/// (0) and the glitched tiles 34-37 have NO collision (the simulator treats
/// 34-37 as empty).
///
/// Straight walls are gated on [`TILE_EDGE_GRID_TERNARY`] `== 2` (a full flat
/// face). [`TILE_GRID_EDGE_MAP`] collapses that table's `1` (a PARTIAL face,
/// where a slope or curve meets the tile border) into the same `1` as a full
/// face; reading the collapsed map fabricated a solid 12px wall across the open
/// part of every sloped and curved border. For tiles 10-13 the collapsed map is
/// byte-identical to tile 1 (fully solid), so quarter circles became sealed
/// boxes. Where a face is partial the true geometry is the diagonal/circular
/// segment emitted below, so dropping the `1` cases loses no genuine wall.
/// Must stay in lockstep with `free_space_distance._tile_local_solid_segments`.
fn tile_local_solid_segments(tile_type: usize) -> Vec<(f64, f64, f64, f64)> {
    let mut segs = Vec::new();
    if tile_type == 0 || tile_type >= 34 {
        return segs;
    }
    let (tern_h, tern_v) = &TILE_EDGE_GRID_TERNARY[tile_type];
    for (idx, seg) in GRID_EDGE_SEGS.into_iter().enumerate() {
        // GRID_EDGE_SEGS is indexed 2*outer + inner within each axis block;
        // the ternary table is indexed outer + 3*inner.
        let (axis, rem) = (idx / 6, idx % 6);
        let (outer, inner) = (rem / 2, rem % 2);
        let tern = if axis == 0 { tern_h } else { tern_v };
        if tern[outer + 3 * inner] == 2 {
            segs.push(seg);
        }
    }
    let (has_diag, x1, y1, x2, y2) = TILE_SEGMENT_DIAG_MAP[tile_type];
    if has_diag {
        segs.push((x1, y1, x2, y2));
    }
    let (has_circ, cx, cy, sx, sy, _convex) = TILE_SEGMENT_CIRCULAR_MAP[tile_type];
    if has_circ {
        let mut prev: Option<(f64, f64)> = None;
        for k in 0..=ARC_CHORDS {
            let ang = (std::f64::consts::PI / 2.0) * (k as f64 / ARC_CHORDS as f64);
            let px = cx + sx * TILE_PX * ang.cos();
            let py = cy + sy * TILE_PX * ang.sin();
            if let Some((qx, qy)) = prev {
                segs.push((qx, qy, px, py));
            }
            prev = Some((px, py));
        }
    }
    segs
}

#[inline]
fn orient(px: f64, py: f64, qx: f64, qy: f64, rx: f64, ry: f64) -> f64 {
    (qx - px) * (ry - py) - (qy - py) * (rx - px)
}

/// True if open segment A->B strictly crosses segment C->D.
///
/// Strict sign test: a step that merely TOUCHES a wall endpoint (or runs
/// collinear along it) is NOT blocked — cell centres sit at odd multiples of 3px
/// so they never lie on the 12px-aligned ortho walls.
#[inline]
fn seg_crosses(
    ax: f64, ay: f64, bx: f64, by: f64,
    cx: f64, cy: f64, dx: f64, dy: f64,
) -> bool {
    let d1 = orient(cx, cy, dx, dy, ax, ay);
    let d2 = orient(cx, cy, dx, dy, bx, by);
    let d3 = orient(ax, ay, bx, by, cx, cy);
    let d4 = orient(ax, ay, bx, by, dx, dy);
    ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0))
}

/// Per-tile wall segments in GLOBAL pixels, indexed `tx * TILES_H + ty`.
struct Walls(Vec<Vec<(f64, f64, f64, f64)>>);

impl Walls {
    fn build(tiles: &[i32]) -> (Self, bool) {
        let mut v: Vec<Vec<(f64, f64, f64, f64)>> = vec![Vec::new(); TILES_W * TILES_H];
        let mut any = false;
        for rty in 0..TILES_H {
            let oy = (rty + 1) as f64 * TILE_PX;
            for rtx in 0..TILES_W {
                let tt = tiles[rtx + rty * TILES_W];
                if tt < 0 {
                    continue;
                }
                let local = tile_local_solid_segments(tt as usize);
                if local.is_empty() {
                    continue;
                }
                any = true;
                let ox = (rtx + 1) as f64 * TILE_PX;
                v[rtx * TILES_H + rty] = local
                    .into_iter()
                    .map(|(x1, y1, x2, y2)| (x1 + ox, y1 + oy, x2 + ox, y2 + oy))
                    .collect();
            }
        }
        (Walls(v), any)
    }

    /// Does the step A->B cross any solid segment?
    ///
    /// A short step near a tile corner can pass through up to 4 tiles and the
    /// segment it crosses may belong to any of them (a diagonal clipping a
    /// slope/arc corner whose bounding segment lives in the diagonal neighbour),
    /// so every tile in the step's bounding box is tested.
    fn crosses(&self, ax: f64, ay: f64, bx: f64, by: f64) -> bool {
        let tx0 = (ax.min(bx) / TILE_PX).floor() as i64 - 1;
        let tx1 = (ax.max(bx) / TILE_PX).floor() as i64 - 1;
        let ty0 = (ay.min(by) / TILE_PX).floor() as i64 - 1;
        let ty1 = (ay.max(by) / TILE_PX).floor() as i64 - 1;
        for tx in tx0..=tx1 {
            if tx < 0 || tx >= TILES_W as i64 {
                continue;
            }
            for ty in ty0..=ty1 {
                if ty < 0 || ty >= TILES_H as i64 {
                    continue;
                }
                for &(cx, cy, dx, dy) in &self.0[tx as usize * TILES_H + ty as usize] {
                    if seg_crosses(ax, ay, bx, by, cx, cy, dx, dy) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

/// The four directional blocked-edge masks, flattened `x * h + y`.
///
/// `block_r` = step (x,y)->(x+1,y); `block_u` = ->(x,y+1);
/// `block_dr` = ->(x+1,y+1); `block_ur` = ->(x+1,y-1). Diagonals additionally
/// fold in a squeeze past any surrounding ortho wall: the ninja can only cut a
/// corner when all four edges of the 2x2 block are open.
#[pyfunction]
pub fn free_space_blocked_edges(
    w: usize,
    h: usize,
    cell: f64,
    tiles: Vec<i32>,
    free: &[u8],
) -> PyResult<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)> {
    if free.len() != w * h {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "free has length {}, expected w*h = {}",
            free.len(),
            w * h
        )));
    }
    if tiles.len() != TILES_W * TILES_H {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "tiles has length {}, expected {}",
            tiles.len(),
            TILES_W * TILES_H
        )));
    }
    let n = w * h;
    let (mut br, mut bu) = (vec![0u8; n], vec![0u8; n]);
    let (mut bdr, mut bur) = (vec![0u8; n], vec![0u8; n]);
    let (walls, any) = Walls::build(&tiles);
    if !any {
        return Ok((br, bu, bdr, bur)); // fully open level: nothing to block
    }
    let idx = |x: usize, y: usize| x * h + y;
    let is_free = |x: usize, y: usize| free[x * h + y] != 0;
    let half = cell / 2.0;

    // Pass 1: orthogonal boundaries.
    for ix in 0..w {
        let ax = ix as f64 * cell + half;
        for iy in 0..h {
            if !is_free(ix, iy) {
                continue;
            }
            let ay = iy as f64 * cell + half;
            if ix + 1 < w && is_free(ix + 1, iy) && walls.crosses(ax, ay, ax + cell, ay) {
                br[idx(ix, iy)] = 1;
            }
            if iy + 1 < h && is_free(ix, iy + 1) && walls.crosses(ax, ay, ax, ay + cell) {
                bu[idx(ix, iy)] = 1;
            }
        }
    }
    // Pass 2: diagonals — direct crossing OR a squeeze past a surrounding wall.
    for ix in 0..w {
        let ax = ix as f64 * cell + half;
        for iy in 0..h {
            if !is_free(ix, iy) {
                continue;
            }
            let ay = iy as f64 * cell + half;
            if ix + 1 < w && iy + 1 < h && is_free(ix + 1, iy + 1) {
                bdr[idx(ix, iy)] = (walls.crosses(ax, ay, ax + cell, ay + cell)
                    || br[idx(ix, iy)] != 0
                    || br[idx(ix, iy + 1)] != 0
                    || bu[idx(ix, iy)] != 0
                    || bu[idx(ix + 1, iy)] != 0) as u8;
            }
            if ix + 1 < w && iy >= 1 && is_free(ix + 1, iy - 1) {
                bur[idx(ix, iy)] = (walls.crosses(ax, ay, ax + cell, ay - cell)
                    || br[idx(ix, iy - 1)] != 0
                    || br[idx(ix, iy)] != 0
                    || bu[idx(ix, iy - 1)] != 0
                    || bu[idx(ix + 1, iy - 1)] != 0) as u8;
            }
        }
    }
    Ok((br, bu, bdr, bur))
}

/// Per-cell frame costs and limits for the physics model. All in "per 6px cell"
/// except the two cell-count limits; the caller derives them from the sim's own
/// constants (see `_physics_cost_constants` on the Python side) so this struct
/// never hard-codes physics.
#[derive(Clone, Copy)]
pub struct PhysicsParams {
    pub run: f64,
    pub air: f64,
    pub fall: f64,
    pub jump: f64,
    pub wall: f64,
    pub jump_reach_cells: f64,
    pub lethal_drop_cells: f64,
    pub lethal_penalty: f64,
    pub air_unsupported_k: f64,
    pub air_unsupported_cap: f64,
}

/// Grid geometry + the caller's precomputed masks, indexed `x * h + y`
/// (C-order for a numpy `(W, H)` array, matching `ndarray.tobytes()`).
pub struct Grid<'a> {
    pub w: usize,
    pub h: usize,
    pub cell: f64,
    pub free: &'a [u8],
    pub mden: &'a [f32],
    pub block_r: &'a [u8],
    pub block_u: &'a [u8],
    pub block_dr: &'a [u8],
    pub block_ur: &'a [u8],
    pub mine_soft_k: f64,
}

impl<'a> Grid<'a> {
    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        x * self.h + y
    }

    #[inline]
    fn is_free(&self, x: usize, y: usize) -> bool {
        self.free[self.idx(x, y)] != 0
    }

    /// Does the step `(x, y) -> (x+dx, y+dy)` cross a solid collision segment?
    ///
    /// Mirrors `_step_blocked`: the masks are stored per FORWARD edge, so a
    /// reverse step reads the same mask entry at the shifted origin.
    #[inline]
    fn step_blocked(&self, x: usize, y: usize, dx: i32, dy: i32) -> bool {
        let xi = x as i32;
        let yi = y as i32;
        if dy == 0 {
            let ox = if dx == 1 { xi } else { xi - 1 };
            return self.block_r[self.idx(ox as usize, y)] != 0;
        }
        if dx == 0 {
            let oy = if dy == 1 { yi } else { yi - 1 };
            return self.block_u[self.idx(x, oy as usize)] != 0;
        }
        if dx == dy {
            let (ox, oy) = if dx == 1 { (xi, yi) } else { (xi - 1, yi - 1) };
            return self.block_dr[self.idx(ox as usize, oy as usize)] != 0;
        }
        let (ox, oy) = if dx == 1 { (xi, yi) } else { (xi - 1, yi + 1) };
        self.block_ur[self.idx(ox as usize, oy as usize)] != 0
    }
}

/// Min-heap entry. `f64::to_bits` is monotonic for non-negative finite values,
/// so ordering on the bit pattern reproduces Python's `(cost, ix, iy)` tuple
/// heap EXACTLY — including how ties break, which the reflection-equivariance
/// work showed is load-bearing for the nav hint's `argmin`.
type HeapItem = Reverse<(u64, u32, u32)>;

#[inline]
fn push(heap: &mut BinaryHeap<HeapItem>, c: f64, x: usize, y: usize) {
    heap.push(Reverse((c.to_bits(), x as u32, y as u32)));
}

/// Zero-cost sources inside the goal's activation disc, expanding the radius
/// until it captures free cells (port of `_seed_activation_disc`, including the
/// recessed-entity fallback that keeps the field from collapsing to one cell).
fn seed_activation_disc(
    g: &Grid,
    d: &mut [f64],
    goal: (f64, f64),
    contact_radius: f64,
) -> BinaryHeap<HeapItem> {
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
    let gx = (goal.0 / g.cell) as i64;
    let gy = (goal.1 / g.cell) as i64;
    for mult in [1.0f64, 2.0, 3.0, 4.0, 6.0] {
        let rr = contact_radius * mult;
        let rr2 = rr * rr;
        let rc = (rr / g.cell) as i64 + 1;
        let x0 = (gx - rc).max(0) as usize;
        let x1 = ((gx + rc + 1).min(g.w as i64)).max(0) as usize;
        let y0 = (gy - rc).max(0) as usize;
        let y1 = ((gy + rc + 1).min(g.h as i64)).max(0) as usize;
        for ix in x0..x1 {
            let cx = ix as f64 * g.cell + g.cell / 2.0;
            for iy in y0..y1 {
                let cy = iy as f64 * g.cell + g.cell / 2.0;
                if g.is_free(ix, iy)
                    && (cx - goal.0) * (cx - goal.0) + (cy - goal.1) * (cy - goal.1) <= rr2
                {
                    d[g.idx(ix, iy)] = 0.0;
                    push(&mut heap, 0.0, ix, iy);
                }
            }
        }
        if !heap.is_empty() {
            break;
        }
    }
    heap
}

/// Geometric flood: isotropic grid distance, mine-soft. Relaxes FORWARD edges,
/// which is sound only because the weights are symmetric — see `flood_physics`.
pub fn flood_geometric(g: &Grid, goal: (f64, f64), contact_radius: f64) -> Vec<f64> {
    let mut d = vec![f64::INFINITY; g.w * g.h];
    let mut heap = seed_activation_disc(g, &mut d, goal, contact_radius);
    while let Some(Reverse((cbits, xu, yu))) = heap.pop() {
        let c = f64::from_bits(cbits);
        let (x, y) = (xu as usize, yu as usize);
        if c > d[g.idx(x, y)] {
            continue;
        }
        for dx in -1i32..=1 {
            for dy in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= g.w as i32 || ny >= g.h as i32 {
                    continue;
                }
                let (nx, ny) = (nx as usize, ny as usize);
                if !g.is_free(nx, ny) {
                    continue;
                }
                // No corner cutting: a diagonal needs both orthogonal neighbours
                // free, or the ninja-sized body cannot pass the corner.
                if dx != 0 && dy != 0 {
                    let ax = (x as i32 + dx) as usize;
                    let ay = (y as i32 + dy) as usize;
                    if !g.is_free(ax, y) || !g.is_free(x, ay) {
                        continue;
                    }
                }
                if g.step_blocked(x, y, dx, dy) {
                    continue;
                }
                let base = if dx != 0 && dy != 0 { 1.41 } else { 1.0 };
                let step = base * (1.0 + g.mine_soft_k * g.mden[g.idx(nx, ny)] as f64);
                let nd = c + step;
                if nd < d[g.idx(nx, ny)] {
                    let i = g.idx(nx, ny);
                    d[i] = nd;
                    push(&mut heap, nd, nx, ny);
                }
            }
        }
    }
    d
}

/// `floor_rise[x][y]` — free cells directly BELOW before the first solid, i.e.
/// simultaneously "how high above the floor am I" (which bounds whether a jump
/// could have put me here) and "how far would I fall". `y` grows DOWNWARD, so
/// "below" is `y+1` and the scan runs bottom-up. Out of range counts as solid:
/// the sim's 1-tile border means no column is genuinely bottomless.
///
/// `wall_adj[x][y]` — a solid cell immediately left or right, so the ninja can
/// wall-jump/slide here. The escape hatch that keeps shafts climbable when the
/// open-air jump-reach rule would otherwise seal them.
pub fn climb_fields(g: &Grid) -> (Vec<i32>, Vec<u8>) {
    let (w, h) = (g.w, g.h);
    let mut rise = vec![0i32; w * h];
    let mut wall = vec![0u8; w * h];
    for x in 0..w {
        for y in (0..h).rev() {
            if !g.is_free(x, y) {
                continue;
            }
            let below_free = y + 1 < h && g.is_free(x, y + 1);
            rise[x * h + y] = if below_free { rise[x * h + y + 1] + 1 } else { 0 };
            let left_solid = x == 0 || !g.is_free(x - 1, y);
            let right_solid = x + 1 >= w || !g.is_free(x + 1, y);
            wall[x * h + y] = (left_solid || right_solid) as u8;
        }
    }
    (rise, wall)
}

/// Physics flood: anisotropic frame costs + the climb-admissibility rule.
///
/// Because the weights are ASYMMETRIC a backward flood must relax REVERSE
/// edges: popping `n` we consider each predecessor `p = n - (dx, dy)` and charge
/// the forward move `p -> n`. Relaxing `n -> p` (what the geometric flood does,
/// harmlessly, since its weights are symmetric) would price walking AWAY from
/// the goal.
pub fn flood_physics(
    g: &Grid,
    rise: &[i32],
    wall: &[u8],
    p: &PhysicsParams,
    goal: (f64, f64),
    contact_radius: f64,
    arcs: Option<&ArcEdges>,
) -> Vec<f64> {
    let mut d = vec![f64::INFINITY; g.w * g.h];
    let mut heap = seed_activation_disc(g, &mut d, goal, contact_radius);
    while let Some(Reverse((cbits, xu, yu))) = heap.pop() {
        let c = f64::from_bits(cbits);
        let (x, y) = (xu as usize, yu as usize);
        let ni = g.idx(x, y);
        if c > d[ni] {
            continue;
        }
        for dx in -1i32..=1 {
            for dy in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                // Predecessor of the forward move (dx, dy) that lands on (x, y).
                let px = x as i32 - dx;
                let py = y as i32 - dy;
                if px < 0 || py < 0 || px >= g.w as i32 || py >= g.h as i32 {
                    continue;
                }
                let (px, py) = (px as usize, py as usize);
                if !g.is_free(px, py) {
                    continue;
                }
                if dx != 0 && dy != 0 {
                    let ax = (px as i32 + dx) as usize;
                    let ay = (py as i32 + dy) as usize;
                    if !g.is_free(ax, py) || !g.is_free(px, ay) {
                        continue;
                    }
                }
                if g.step_blocked(px, py, dx, dy) {
                    continue;
                }

                let vcost = if dy < 0 {
                    // Rising into (x, y): only legal off a wall, or within a
                    // floor jump's reach of the floor beneath the target.
                    if wall[ni] != 0 {
                        p.wall
                    } else if rise[ni] as f64 <= p.jump_reach_cells {
                        p.jump
                    } else {
                        continue; // unreachable from below: no free climb
                    }
                } else if dy > 0 {
                    p.fall
                } else {
                    0.0
                };
                let hcost = if dx == 0 {
                    0.0
                } else if rise[ni] == 0 || wall[ni] != 0 {
                    p.run
                } else {
                    // Unsupported air: the higher above the floor, the less of
                    // this traverse the ninja could actually hold.
                    let hh = (rise[ni] as f64 / p.jump_reach_cells).min(p.air_unsupported_cap);
                    p.air * (1.0 + p.air_unsupported_k * hh)
                };
                let mut step = if dx != 0 && dy != 0 {
                    hcost.hypot(vcost)
                } else {
                    hcost + vcost
                };
                step *= 1.0 + g.mine_soft_k * g.mden[ni] as f64;
                let pi = g.idx(px, py);
                if p.lethal_penalty > 0.0
                    && rise[pi] <= 1
                    && rise[ni] as f64 > p.lethal_drop_cells
                    && wall[ni] == 0
                {
                    step += p.lethal_penalty;
                }

                let nd = c + step;
                if nd < d[pi] {
                    d[pi] = nd;
                    push(&mut heap, nd, px, py);
                }
            }
        }

        // Wall-launch arc predecessors that LAND on (x, y). These deliberately
        // skip the intervening air: every mid-flight cell fails the up-gate
        // above, so without them a wall jump is unrepresentable and a level
        // whose only exit is one gets severed. CSR keyed by landing cell,
        // precomputed on the Python side (see `_build_arc_edges`) and passed in
        // rather than re-derived, exactly like the blocked-edge masks.
        if let Some(a) = arcs {
            let node = g.idx(x, y);
            for e in a.start[node] as usize..a.start[node + 1] as usize {
                let li = a.src[e] as usize;
                let nd = c + a.cost[e];
                if nd < d[li] {
                    d[li] = nd;
                    push(&mut heap, nd, li / g.h, li % g.h);
                }
            }
        }
    }
    d
}

/// Reverse (landing-keyed) CSR adjacency of wall-launch arc edges.
pub struct ArcEdges<'a> {
    /// `start[i]..start[i+1]` indexes `src`/`cost` for landing cell `i`.
    pub start: &'a [i64],
    pub src: &'a [i64],
    pub cost: &'a [f64],
}

/// Flood both goals over one grid; returns `(d_switch, d_exit)` flattened
/// `x * h + y`. `physics_params` selects the model: `None` runs the geometric
/// flood, `Some(..)` the physics one.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    w, h, cell, free, mden, block_r, block_u, block_dr, block_ur, mine_soft_k,
    switch, exit, switch_radius, exit_radius, physics_params = None,
    arc_edges = None,
))]
pub fn free_space_flood(
    w: usize,
    h: usize,
    cell: f64,
    free: &[u8],
    mden: Vec<f32>,
    block_r: &[u8],
    block_u: &[u8],
    block_dr: &[u8],
    block_ur: &[u8],
    mine_soft_k: f64,
    switch: (f64, f64),
    exit: (f64, f64),
    switch_radius: f64,
    exit_radius: f64,
    physics_params: Option<[f64; 10]>,
    arc_edges: Option<(Vec<i64>, Vec<i64>, Vec<f64>)>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let n = w * h;
    for (name, len) in [
        ("free", free.len()),
        ("mden", mden.len()),
        ("block_r", block_r.len()),
        ("block_u", block_u.len()),
        ("block_dr", block_dr.len()),
        ("block_ur", block_ur.len()),
    ] {
        if len != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{name} has length {len}, expected w*h = {n}"
            )));
        }
    }
    let g = Grid {
        w,
        h,
        cell,
        free,
        mden: &mden,
        block_r,
        block_u,
        block_dr,
        block_ur,
        mine_soft_k,
    };
    match physics_params {
        None => Ok((
            flood_geometric(&g, switch, switch_radius),
            flood_geometric(&g, exit, exit_radius),
        )),
        Some(a) => {
            let p = PhysicsParams {
                run: a[0],
                air: a[1],
                fall: a[2],
                jump: a[3],
                wall: a[4],
                jump_reach_cells: a[5],
                lethal_drop_cells: a[6],
                lethal_penalty: a[7],
                air_unsupported_k: a[8],
                air_unsupported_cap: a[9],
            };
            let (rise, wall) = climb_fields(&g);
            let arcs = match &arc_edges {
                None => None,
                Some((start, src, cost)) => {
                    if start.len() != n + 1 {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "arc start has length {}, expected w*h + 1 = {}",
                            start.len(),
                            n + 1
                        )));
                    }
                    if src.len() != cost.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "arc src/cost length mismatch: {} vs {}",
                            src.len(),
                            cost.len()
                        )));
                    }
                    if src.iter().any(|&i| i < 0 || i as usize >= n) {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "arc src contains an out-of-range cell index",
                        ));
                    }
                    Some(ArcEdges {
                        start,
                        src,
                        cost,
                    })
                }
            };
            Ok((
                flood_physics(&g, &rise, &wall, &p, switch, switch_radius, arcs.as_ref()),
                flood_physics(&g, &rise, &wall, &p, exit, exit_radius, arcs.as_ref()),
            ))
        }
    }
}

/// `(floor_rise, wall_adj)` for the given free mask, flattened `x * h + y`.
/// Exposed so the Python side can cache and visualise the same fields the
/// physics flood consumed, without recomputing them in numpy.
#[pyfunction]
pub fn free_space_climb_fields(
    w: usize,
    h: usize,
    free: &[u8],
) -> PyResult<(Vec<i32>, Vec<u8>)> {
    if free.len() != w * h {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "free has length {}, expected w*h = {}",
            free.len(),
            w * h
        )));
    }
    let empty_f32: Vec<f32> = Vec::new();
    let empty_u8: Vec<u8> = Vec::new();
    let g = Grid {
        w,
        h,
        cell: 6.0,
        free,
        mden: &empty_f32,
        block_r: &empty_u8,
        block_u: &empty_u8,
        block_dr: &empty_u8,
        block_ur: &empty_u8,
        mine_soft_k: 0.0,
    };
    Ok(climb_fields(&g))
}
