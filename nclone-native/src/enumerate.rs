//! Native batch edge-enumeration kernel for the surface-graph lattice.
//!
//! This replaces the per-arc Python loop in
//! `nclone/value_function/edge_enumeration.py` with an exact, in-Rust sweep:
//! for every `(surface_point, velocity_bin)` node it injects the launch state
//! and simulates the same run / floor-jump / wall-jump / edge-fall / launch-pad
//! arcs the Python enumerator does, using the identical `tick_tiles_only`
//! (or `tick_physics_mines_only`) physics.  Because it calls the same physics
//! and injects the same launch state, it produces **bit-identical landings** to
//! the Python path — but with the per-frame loop, the three separate FFI
//! state-reads, and the multiprocess serialization removed, and with rayon
//! parallelism across nodes.
//!
//! The kernel returns only the *raw landings* (one `ArcRecord` per arc that the
//! Python `_try_edge` would process).  The landing -> graph-node mapping,
//! mine-cost inflation, and dominated-edge pruning stay in Python so they reuse
//! the existing, already-validated code unchanged (guaranteeing identical edges
//! given identical landings).
//!
//! Sweep tables (`run_durations`, `jump_holds`, `h_inputs`, `post_release`) are
//! passed in from Python so the canonical values live in one place
//! (`edge_enumeration.py`) and never drift out of sync.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::ninja::Ninja;
use crate::simulator::Simulator;

/// One landed arc, already mapped to graph nodes (the landing -> node mapping
/// that used to live in Python is now done in-kernel; see `NodeMapper`).  Only
/// arcs that map to a real, non-self destination node are emitted.
#[derive(Clone, Copy)]
pub struct ArcRecord {
    pub src_node: i32,
    pub dst_node: i32,
    pub frames: i32,
    pub jhold: i32,
    pub h_during: i32,
    pub h_after: i32,
    /// Landing velocity fell on/outside the first/last velocity bin (telemetry).
    pub clipped: bool,
}

/// Accumulated output for a batch (chunk) of nodes.
#[derive(Default)]
pub struct NodeBatchOut {
    pub arcs: Vec<ArcRecord>,
    /// Sparse per-(node, goal) minimum contact frame.  `gc_node` holds the
    /// source *node id* (not the launch index).
    pub gc_node: Vec<i32>,
    pub gc_goal: Vec<i32>,
    pub gc_frame: Vec<f64>,
}

impl NodeBatchOut {
    fn extend(&mut self, other: NodeBatchOut) {
        self.arcs.extend(other.arcs);
        self.gc_node.extend(other.gc_node);
        self.gc_goal.extend(other.gc_goal);
        self.gc_frame.extend(other.gc_frame);
    }
}

/// One launch node to enumerate from.
#[derive(Clone, Copy)]
pub struct NodeSpec {
    /// Source graph node id (`pt_idx * max_vel_bins + vel_bin`), precomputed by
    /// the caller so the kernel can emit fully-mapped edges.
    pub src_node: i32,
    pub x: f64,
    pub y: f64,
    pub vx0: f64,
    pub vy0: f64,
    pub surface_type: u8, // 0=floor, 1=wall, 2=slide, 3=launch_pad
    pub is_edge: bool,
}

const SURFACE_FLOOR: i32 = 0;
const SURFACE_WALL: i32 = 1;
const SURFACE_SLIDE: i32 = 2;
const SURFACE_LAUNCH_PAD: i32 = 3;
const STATE_WALL_SLIDING: u8 = 5;

#[inline]
fn is_wall_like(stype: i32) -> bool {
    stype == SURFACE_WALL || stype == SURFACE_SLIDE
}

/// Exact in-kernel port of `_landing_to_node` / `find_nearest_point` /
/// `velocity_to_bin` (edge_enumeration.py / surface_graph.py).
///
/// Precision is matched byte-for-byte to the Python path:
/// * nearest-point distances are f64 (SurfacePoint coords are Python floats);
/// * the velocity-bin `argmin` runs in **f32** (the bin-center arrays are
///   `np.float32`, so `np.abs(centers - vel)` is a float32 op) with first-min
///   tie-breaking;
/// * the clip test compares the f64 velocity against the f32 endpoints widened
///   to f64 (`vel <= float(centers[0])`).
///
/// The spatial index is built in surface-point index order, so within-cell lists
/// are ascending index order — matching `build_graph` — which together with the
/// `dx`-outer / `dy`-inner scan and strict `<` reproduces Python's tie-breaks.
pub struct NodeMapper<'a> {
    sp_x: &'a [f64],
    sp_y: &'a [f64],
    sp_type: &'a [i32],
    sp_tangent_x: &'a [f64],
    sp_tangent_y: &'a [f64],
    tangential: bool,
    spatial: HashMap<(i32, i32), Vec<usize>>,
    inv_cell: f64,
    floor_centers: Vec<f32>,
    wall_centers: Vec<f32>,
    slide_centers: Vec<f32>,
    launch_centers: Vec<f32>,
    max_vel_bins: i32,
}

impl<'a> NodeMapper<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sp_x: &'a [f64],
        sp_y: &'a [f64],
        sp_type: &'a [i32],
        sp_tangent_x: &'a [f64],
        sp_tangent_y: &'a [f64],
        tangential: bool,
        floor_centers: Vec<f32>,
        wall_centers: Vec<f32>,
        slide_centers: Vec<f32>,
        launch_centers: Vec<f32>,
        cell_size: f64,
        max_vel_bins: i32,
    ) -> Self {
        let inv_cell = 1.0 / cell_size;
        let mut spatial: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        for i in 0..sp_x.len() {
            let gx = (sp_x[i] * inv_cell).floor() as i32;
            let gy = (sp_y[i] * inv_cell).floor() as i32;
            spatial.entry((gx, gy)).or_default().push(i);
        }
        NodeMapper {
            sp_x,
            sp_y,
            sp_type,
            sp_tangent_x,
            sp_tangent_y,
            tangential,
            spatial,
            inv_cell,
            floor_centers,
            wall_centers,
            slide_centers,
            launch_centers,
            max_vel_bins,
        }
    }

    #[inline]
    fn centers(&self, stype: i32) -> &[f32] {
        match stype {
            SURFACE_FLOOR => &self.floor_centers,
            SURFACE_SLIDE => &self.slide_centers,
            SURFACE_LAUNCH_PAD => &self.launch_centers,
            _ => &self.wall_centers, // SURFACE_WALL and any fallthrough
        }
    }

    /// Nearest surface point to (x, y), optionally restricted to `stype`.
    /// Returns the point index or -1 (Python `None`).
    fn find_nearest_point(&self, x: f64, y: f64, stype: Option<i32>) -> i32 {
        let gx = (x * self.inv_cell).floor() as i32;
        let gy = (y * self.inv_cell).floor() as i32;
        let mut best_idx: i32 = -1;
        let mut best_d2 = f64::INFINITY;
        for dx in -3..=3 {
            for dy in -3..=3 {
                if let Some(indices) = self.spatial.get(&(gx + dx, gy + dy)) {
                    for &idx in indices {
                        if let Some(st) = stype {
                            if self.sp_type[idx] != st {
                                continue;
                            }
                        }
                        let ex = self.sp_x[idx] - x;
                        let ey = self.sp_y[idx] - y;
                        let d2 = ex * ex + ey * ey;
                        if d2 < best_d2 {
                            best_d2 = d2;
                            best_idx = idx as i32;
                        }
                    }
                }
            }
        }
        best_idx
    }

    /// `argmin(|centers - vel|)` in f32, first-min tie-break.
    #[inline]
    fn velocity_to_bin(&self, vel: f64, stype: i32) -> i32 {
        let centers = self.centers(stype);
        let v = vel as f32;
        let mut best_bin = 0i32;
        let mut best = f32::INFINITY;
        for (i, &c) in centers.iter().enumerate() {
            let d = (c - v).abs();
            if d < best {
                best = d;
                best_bin = i as i32;
            }
        }
        best_bin
    }

    /// Port of `_landing_to_node`.  Returns `(node, clipped)`; `node` is -1 for
    /// the Python `None` case.
    pub fn landing_to_node(&self, x: f64, y: f64, vx: f64, vy: f64, state: u8) -> (i32, bool) {
        let (mut stype, mut vel) = if state == STATE_WALL_SLIDING {
            (SURFACE_WALL, vy)
        } else {
            (SURFACE_FLOOR, vx)
        };
        let mut pt = self.find_nearest_point(x, y, Some(stype));
        if pt < 0 {
            pt = self.find_nearest_point(x, y, None);
            if pt < 0 {
                return (-1, false);
            }
            stype = self.sp_type[pt as usize];
            vel = if is_wall_like(stype) { vy } else { vx };
        }
        if self.tangential {
            // Mirror _landing_to_node: bin the signed along-surface speed of the
            // destination point.  f64 dot product (tangents passed from Python)
            // then the same f32 argmin in velocity_to_bin.
            let p = pt as usize;
            stype = self.sp_type[p];
            vel = vx * self.sp_tangent_x[p] + vy * self.sp_tangent_y[p];
        }
        let vel_bin = self.velocity_to_bin(vel, stype);
        let node = pt * self.max_vel_bins + vel_bin;
        let centers = self.centers(stype);
        let clipped =
            vel <= centers[0] as f64 || vel >= centers[centers.len() - 1] as f64;
        (node, clipped)
    }
}

/// Immutable sweep parameters shared across all nodes.
pub struct SweepParams<'a> {
    pub run_durations: &'a [i32],
    pub jump_holds: &'a [i32],
    pub h_inputs: &'a [i32],
    pub post_release: &'a [i32],
    pub max_arc_frames: i32,
    pub max_jump_duration: i32,
    pub tick_mines: bool,
    /// Flat goal triples: (gx, gy, r2). `r2` is the squared contact range.
    pub goals: &'a [(f64, f64, f64)],
}

const DEATH_STATES: [u8; 2] = [6, 7];
#[inline]
fn is_dead(st: u8) -> bool {
    st == DEATH_STATES[0] || st == DEATH_STATES[1]
}
#[inline]
fn is_surface(st: u8) -> bool {
    st == 0 || st == 1 || st == 2 || st == 5
}
#[inline]
fn is_air(st: u8) -> bool {
    st == 3 || st == 4
}

/// Update the per-goal minimum-contact-frame accumulator.  Mirrors
/// `_check_goal_contact`: strict `<` on both the squared distance and the frame.
#[inline]
fn check_goals(px: f64, py: f64, frame: i32, goals: &[(f64, f64, f64)], acc: &mut [f64]) {
    let f = frame as f64;
    for (g, &(gx, gy, r2)) in goals.iter().enumerate() {
        let dx = px - gx;
        let dy = py - gy;
        if dx * dx + dy * dy < r2 && f < acc[g] {
            acc[g] = f;
        }
    }
}

type Landing = (f64, f64, f64, f64, u8);

impl Simulator {
    #[inline]
    fn arc_tick(&mut self, h: i32, j: i32, tick_mines: bool) {
        // Experimental flood sizing: record the quantized (pre-state, input) key
        // *before* stepping.  `None` in production -> a single predicted branch.
        if self.flood_keys.is_some() {
            let k = flood_key(&self.ninja, h, j, self.flood_qpos, self.flood_qvel);
            self.flood_ticks += 1;
            self.flood_keys.as_mut().unwrap().insert(k);
            // Collision-relevant tick? (a segment within this frame's swept reach).
            let (px, py, vx, vy) = (
                self.ninja.xpos, self.ninja.ypos, self.ninja.xspeed, self.ninja.yspeed,
            );
            let pad = crate::constants::NINJA_RADIUS + 1.0;
            let (nx, ny) = (px + vx, py + vy);
            let near = crate::collision::spatial::region_has_segments(
                &self.segment_grid, &self.segments,
                px.min(nx) - pad, py.min(ny) - pad, px.max(nx) + pad, py.max(ny) + pad,
            );
            if near {
                self.flood_ticks_coll += 1;
                self.flood_keys_coll.as_mut().unwrap().insert(k);
            }
        }
        if tick_mines {
            self.tick_physics_mines_only(h, j);
        } else {
            self.tick_tiles_only(h, j);
        }

        // Collision-merge accuracy study: snap state to a grain-G grid at the
        // discrete *landing* event (airborne -> surface this tick) — the expensive
        // depenetration and the natural cache key.  NOT on continuous grounded /
        // sliding contact, which would corrupt acceleration dynamics frame-by-
        // frame.  Models a cache that merges landing/collision outcomes by grain.
        if self.snap_qpos > 0.0 {
            let (qp, qv) = (self.snap_qpos, self.snap_qvel);
            let n = &mut self.ninja;
            if n.airborn_old && !n.airborn {
                n.xpos = (n.xpos / qp).round() * qp;
                n.ypos = (n.ypos / qp).round() * qp;
                n.xspeed = (n.xspeed / qv).round() * qv;
                n.yspeed = (n.yspeed / qv).round() * qv;
            }
        }
    }

    /// Port of `_simulate_arc` (constant inputs until landing/death/timeout).
    /// `jump` is held constant (0 for runs/edge-falls/launch-pads).
    fn sim_run_arc(
        &mut self,
        h: i32,
        max_frames: i32,
        tick_mines: bool,
        goals: &[(f64, f64, f64)],
        acc: &mut [f64],
        goal_off: i32,
    ) -> (Option<Landing>, i32) {
        let mut was_air = false;
        for frame in 1..=max_frames {
            self.arc_tick(h, 0, tick_mines);
            let st = self.ninja.state;
            if is_dead(st) {
                return (None, frame);
            }
            let px = self.ninja.xpos;
            let py = self.ninja.ypos;
            if !goals.is_empty() {
                check_goals(px, py, frame + goal_off, goals, acc);
            }
            if is_air(st) {
                was_air = true;
            } else if was_air && is_surface(st) {
                return (
                    Some((px, py, self.ninja.xspeed, self.ninja.yspeed, st)),
                    frame,
                );
            }
        }
        let st = self.ninja.state;
        if is_surface(st) {
            return (
                Some((
                    self.ninja.xpos,
                    self.ninja.ypos,
                    self.ninja.xspeed,
                    self.ninja.yspeed,
                    st,
                )),
                max_frames,
            );
        }
        (None, max_frames)
    }

    /// Port of `_simulate_jump_arc`: hold the jump for `hold_duration` frames
    /// with `h_during`, then fall with `h_after` until the first surface state.
    /// Superseded by `jump_hold_sweep` + `sim_fall_phase` (kept as the reference
    /// spec the prefix-reuse path mirrors).
    #[allow(dead_code)]
    fn sim_jump_arc(
        &mut self,
        h_during: i32,
        hold_duration: i32,
        h_after: i32,
        max_arc_frames: i32,
        tick_mines: bool,
        goals: &[(f64, f64, f64)],
        acc: &mut [f64],
        goal_off: i32,
    ) -> (Option<Landing>, i32) {
        let mut total = 0i32;
        // Phase 1: jump hold.
        for _ in 0..hold_duration {
            total += 1;
            self.arc_tick(h_during, 1, tick_mines);
            let st = self.ninja.state;
            if is_dead(st) {
                return (None, total);
            }
            if !goals.is_empty() {
                check_goals(self.ninja.xpos, self.ninja.ypos, total + goal_off, goals, acc);
            }
        }
        // Phase 2: post-release fall.
        let fall = (max_arc_frames - total).max(0);
        let mut left_wall = false;
        for _ in 0..fall {
            total += 1;
            self.arc_tick(h_after, 0, tick_mines);
            let st = self.ninja.state;
            if is_dead(st) {
                return (None, total);
            }
            let px = self.ninja.xpos;
            let py = self.ninja.ypos;
            if !goals.is_empty() {
                check_goals(px, py, total + goal_off, goals, acc);
            }
            if is_surface(st) {
                return (
                    Some((px, py, self.ninja.xspeed, self.ninja.yspeed, st)),
                    total,
                );
            }
            // Fresh upward wall re-contact -> chainable wall node (see sim_fall_phase).
            if is_air(st) {
                if !self.ninja.walled {
                    left_wall = true;
                } else if left_wall && self.ninja.yspeed < 0.0 {
                    return (Some((px, py, self.ninja.xspeed, self.ninja.yspeed, 5u8)), total);
                }
            }
        }
        (None, total)
    }

    /// Run-arc sweep with PREFIX REUSE.  Simulates ONE constant-input arc to
    /// `max(durations)` and derives every duration's result from the single
    /// trace.  Byte-identical to calling `sim_run_arc(h, dur)` for each `dur`:
    /// the arc's first reland (or death) stops it, so every `dur >= stop` yields
    /// that one landing, while every `dur < stop` reads the recorded per-frame
    /// surface state at frame `dur`.  Goal contacts are min-accumulated over the
    /// single longest pass, which exactly covers the union of all per-dur arcs.
    fn sim_run_sweep(
        &mut self,
        snap: &Ninja,
        x: f64,
        y: f64,
        vx0: f64,
        vy0: f64,
        h: i32,
        durations: &[i32],
        tick_mines: bool,
        goals: &[(f64, f64, f64)],
        acc: &mut [f64],
        out: &mut Vec<(Landing, i32)>,
    ) {
        out.clear();
        let max_dur = match durations.iter().max() {
            Some(&m) => m,
            None => return,
        };
        self.ninja = snap.clone();
        self.ninja.inject_value_iteration_state(x, y, vx0, vy0, 0);
        // (is_surface, landing-tuple) for each simulated frame.
        let mut trace: Vec<(bool, Landing)> = Vec::with_capacity(max_dur.max(0) as usize);
        let mut was_air = false;
        let mut reland: Option<i32> = None;
        let mut death: Option<i32> = None;
        for frame in 1..=max_dur {
            self.arc_tick(h, 0, tick_mines);
            let st = self.ninja.state;
            if is_dead(st) {
                death = Some(frame);
                break;
            }
            let px = self.ninja.xpos;
            let py = self.ninja.ypos;
            if !goals.is_empty() {
                check_goals(px, py, frame, goals, acc);
            }
            let surf = is_surface(st);
            trace.push((surf, (px, py, self.ninja.xspeed, self.ninja.yspeed, st)));
            if is_air(st) {
                was_air = true;
            } else if was_air && surf {
                reland = Some(frame);
                break;
            }
        }
        for &dur in durations {
            // dur >= first reland: same landing as the longest run (the arc
            // stops at the reland regardless of remaining budget).
            if let Some(fr) = reland {
                if dur >= fr {
                    out.push((trace[(fr - 1) as usize].1, fr));
                    continue;
                }
            }
            // dur >= death frame (and no earlier reland): lethal -> no edge.
            if let Some(fd) = death {
                if dur >= fd {
                    continue;
                }
            }
            // dur strictly inside the trace: emit the grounded state at frame
            // `dur`, or nothing if airborne there (matches sim_run_arc tail).
            let idx = (dur - 1) as usize;
            if idx < trace.len() {
                let (surf, l) = trace[idx];
                if surf {
                    out.push((l, dur));
                }
            }
        }
    }

    /// Post-release fall phase of a jump arc, resuming after `start_total`
    /// frames have already elapsed (the hold phase, replayed once via snapshot).
    /// This is verbatim phase 2 of the old `sim_jump_arc`.
    fn sim_fall_phase(
        &mut self,
        h_after: i32,
        start_total: i32,
        max_arc_frames: i32,
        tick_mines: bool,
        goals: &[(f64, f64, f64)],
        acc: &mut [f64],
        goal_off: i32,
    ) -> (Option<Landing>, i32) {
        let mut total = start_total;
        let fall = (max_arc_frames - total).max(0);
        // `left_wall`: arc has fully separated from any wall since the jump fired.
        // A fresh upward wall re-contact (separated, now walled, vy < 0) is the
        // wall-jump chain-link the downward-only (state-5) landing never captured;
        // returned as state 5 so the NodeMapper maps it to the wall (upward) vy bin.
        // Mirrors edge_enumeration._simulate_jump_arc phase 2 exactly.
        let mut left_wall = false;
        for _ in 0..fall {
            total += 1;
            self.arc_tick(h_after, 0, tick_mines);
            let st = self.ninja.state;
            if is_dead(st) {
                return (None, total);
            }
            let px = self.ninja.xpos;
            let py = self.ninja.ypos;
            if !goals.is_empty() {
                check_goals(px, py, total + goal_off, goals, acc);
            }
            if is_surface(st) {
                return (Some((px, py, self.ninja.xspeed, self.ninja.yspeed, st)), total);
            }
            if is_air(st) {
                if !self.ninja.walled {
                    left_wall = true;
                } else if left_wall && self.ninja.yspeed < 0.0 {
                    return (Some((px, py, self.ninja.xspeed, self.ninja.yspeed, 5u8)), total);
                }
            }
        }
        (None, total)
    }

    /// Jump sweep with HOLD-PHASE PREFIX REUSE.  Simulates the hold phase ONCE
    /// for `hd` (snapshotting the ninja after each hold tick), then runs only
    /// the fall phase for each `(hold, ha)`.  Byte-identical to per-`(hold, ha)`
    /// jump arcs: the hold trajectory is shared (its goal contacts are
    /// min-accumulated once over the longest hold, which every shorter hold is a
    /// prefix of), and a hold-phase death prunes exactly the holds that reach it.
    fn jump_hold_sweep(
        &mut self,
        snap: &Ninja,
        x: f64,
        y: f64,
        vx0: f64,
        vy0: f64,
        hd: i32,
        post_release: &[i32],
        jump_holds: &[i32],
        max_jump_duration: i32,
        max_arc_frames: i32,
        tick_mines: bool,
        goals: &[(f64, f64, f64)],
        acc: &mut [f64],
        snaps: &mut Vec<Ninja>,
        out: &mut Vec<(i32, i32, Landing, i32)>,
    ) {
        out.clear();
        // reinject + init jump frame (always tile-only; see enumerate_node_arcs).
        self.ninja = snap.clone();
        self.ninja.inject_value_iteration_state(x, y, vx0, vy0, 0);
        self.tick_tiles_only(hd, 1);
        if is_dead(self.ninja.state) {
            return;
        }
        snaps.clear();
        snaps.push(self.ninja.clone()); // S[0]: state after init (total = 0)
        let max_hold_d = jump_holds
            .iter()
            .map(|&h| (h - 1).min(max_jump_duration - 1))
            .max()
            .unwrap_or(0)
            .max(0);
        let mut hold_death: Option<i32> = None;
        for t in 1..=max_hold_d {
            self.arc_tick(hd, 1, tick_mines);
            let st = self.ninja.state;
            if is_dead(st) {
                hold_death = Some(t);
                break;
            }
            if !goals.is_empty() {
                check_goals(self.ninja.xpos, self.ninja.ypos, t + 1, goals, acc);
            }
            snaps.push(self.ninja.clone()); // S[t]
        }
        for &hold in jump_holds {
            let hold_d = (hold - 1).min(max_jump_duration - 1);
            if let Some(td) = hold_death {
                if td <= hold_d {
                    continue; // dies during the hold phase -> no edge for any ha
                }
            }
            for &ha in post_release {
                self.ninja = snaps[hold_d as usize].clone();
                let (l, total) =
                    self.sim_fall_phase(ha, hold_d, max_arc_frames, tick_mines, goals, acc, 1);
                if let Some(l) = l {
                    out.push((hold, ha, l, total + 1));
                }
            }
        }
    }

    /// Enumerate all arcs from a single launch node, mirroring
    /// `_enumerate_node_edges`.  `snap` is the freshly-loaded ninja (restored
    /// before every arc, then re-injected — identical to the Python path).
    fn enumerate_node_arcs(
        &mut self,
        node: &NodeSpec,
        sw: &SweepParams,
        mapper: &NodeMapper,
        snap: &Ninja,
        out: &mut NodeBatchOut,
    ) {
        let NodeSpec { src_node, x, y, vx0, vy0, surface_type, is_edge } = *node;
        let n_goals = sw.goals.len();
        let mut acc = vec![f64::INFINITY; n_goals];

        macro_rules! emit {
            ($l:expr, $frames:expr, $jhold:expr, $hd:expr, $ha:expr) => {{
                let l: Landing = $l;
                // Landing -> graph-node mapping, in-kernel (was the Python hot loop).
                let (dst, clipped) = mapper.landing_to_node(l.0, l.1, l.2, l.3, l.4);
                if dst >= 0 && dst != src_node {
                    out.arcs.push(ArcRecord {
                        src_node,
                        dst_node: dst,
                        frames: $frames,
                        jhold: $jhold,
                        h_during: $hd,
                        h_after: $ha,
                        clipped,
                    });
                }
            }};
        }
        macro_rules! reinject {
            () => {{
                self.ninja = snap.clone();
                self.ninja
                    .inject_value_iteration_state(x, y, vx0, vy0, 0);
            }};
        }

        // Reusable scratch buffers (prefix-reuse sweeps).
        let mut run_out: Vec<(Landing, i32)> = Vec::new();
        let mut jump_out: Vec<(i32, i32, Landing, i32)> = Vec::new();
        let mut jsnaps: Vec<Ninja> = Vec::new();

        match surface_type {
            0 => {
                // FLOOR — runs (one pass per input, all durations derived).
                for &h in sw.h_inputs {
                    self.sim_run_sweep(
                        snap, x, y, vx0, vy0, h, sw.run_durations, sw.tick_mines,
                        sw.goals, &mut acc, &mut run_out,
                    );
                    for &(l, f) in &run_out {
                        emit!(l, f, 0, h, h);
                    }
                }
                // FLOOR — jumps (hold phase simulated once per h_during).
                for &hd in sw.h_inputs {
                    self.jump_hold_sweep(
                        snap, x, y, vx0, vy0, hd, sw.post_release, sw.jump_holds,
                        sw.max_jump_duration, sw.max_arc_frames, sw.tick_mines,
                        sw.goals, &mut acc, &mut jsnaps, &mut jump_out,
                    );
                    for &(hold, ha, l, f) in &jump_out {
                        emit!(l, f, hold, hd, ha);
                    }
                }
                // FLOOR — edge falls (one long fall per input; no sweep gain).
                if is_edge {
                    for &h in sw.h_inputs {
                        reinject!();
                        let (l, f) = self.sim_run_arc(
                            h, sw.max_arc_frames, sw.tick_mines, sw.goals, &mut acc, 0,
                        );
                        if let Some(l) = l {
                            emit!(l, f, 0, h, h);
                        }
                    }
                }
            }
            1 | 2 => {
                // WALL / SLIDE — wall jumps (h_after == h_during).  Python sweeps
                // ("regular","slide") which run identically here (only the bin's
                // initial vy differs, already baked in), so one pass is
                // dedup-equivalent.
                for &h in sw.h_inputs {
                    let pr = [h];
                    self.jump_hold_sweep(
                        snap, x, y, vx0, vy0, h, &pr, sw.jump_holds,
                        sw.max_jump_duration, sw.max_arc_frames, sw.tick_mines,
                        sw.goals, &mut acc, &mut jsnaps, &mut jump_out,
                    );
                    for &(hold, ha, l, f) in &jump_out {
                        emit!(l, f, hold, h, ha);
                    }
                }
                // WALL / SLIDE — slide falls (run sweep).
                for &h in sw.h_inputs {
                    self.sim_run_sweep(
                        snap, x, y, vx0, vy0, h, sw.run_durations, sw.tick_mines,
                        sw.goals, &mut acc, &mut run_out,
                    );
                    for &(l, f) in &run_out {
                        emit!(l, f, 0, h, h);
                    }
                }
            }
            3 => {
                // LAUNCH PAD — single kick swept over post-release input/duration.
                for &h in sw.post_release {
                    self.sim_run_sweep(
                        snap, x, y, vx0, vy0, h, sw.run_durations, sw.tick_mines,
                        sw.goals, &mut acc, &mut run_out,
                    );
                    for &(l, f) in &run_out {
                        emit!(l, f, 0, h, h);
                    }
                }
            }
            _ => {}
        }

        // Restore ninja (parity with the Python final restore_ninja_state).
        self.ninja = snap.clone();

        for (g, &fr) in acc.iter().enumerate() {
            if fr.is_finite() {
                out.gc_node.push(src_node);
                out.gc_goal.push(g as i32);
                out.gc_frame.push(fr);
            }
        }
    }
}

/// Run the full sweep over `nodes`, rayon-parallel across `n_threads` chunks.
/// Each chunk builds its own `Simulator` from `map_data` (cheap, a handful of
/// times), so no shared mutable simulator is needed.
#[allow(clippy::too_many_arguments)]
pub fn enumerate_arcs_parallel(
    map_data: &[i32],
    nodes: &[NodeSpec],
    sw: &SweepParams,
    mapper: &NodeMapper,
    n_threads: usize,
    snap_qpos: f64,
    snap_qvel: f64,
) -> NodeBatchOut {
    let n = nodes.len();
    if n == 0 {
        return NodeBatchOut::default();
    }
    let n_chunks = n_threads.max(1).min(n);
    let chunk_size = (n + n_chunks - 1) / n_chunks;

    let run_chunk = |chunk: &[NodeSpec]| -> NodeBatchOut {
        let mut sim = Simulator::load(map_data);
        sim.snap_qpos = snap_qpos;
        sim.snap_qvel = snap_qvel;
        let snap = sim.ninja.clone();
        let mut out = NodeBatchOut::default();
        for node in chunk {
            sim.enumerate_node_arcs(node, sw, mapper, &snap, &mut out);
        }
        out
    };

    if n_chunks == 1 {
        return run_chunk(nodes);
    }

    let chunks: Vec<&[NodeSpec]> = nodes.chunks(chunk_size).collect();
    let partials: Vec<NodeBatchOut> = chunks.par_iter().map(|c| run_chunk(c)).collect();

    let mut out = NodeBatchOut::default();
    for p in partials {
        out.extend(p);
    }
    out
}

// ---------------------------------------------------------------------------
// Experimental: reachability-flood sizing
// ---------------------------------------------------------------------------

/// FNV-1a hash of the quantized `(pre-tick state, input)`.  The fields are the
/// future-determining ones within a fixed level: position & velocity (quantized
/// to `qpos` / `qvel`), the discrete physics state/mode/timers, and the input.
/// Two ticks with the same key would (to within the quantization) compute the
/// same successor — so distinct keys / total ticks is the transposition-merge
/// redundancy.
#[inline]
fn flood_key(n: &Ninja, h: i32, j: i32, qpos: f64, qvel: f64) -> u64 {
    let grav = if n.applied_gravity > 0.03 { 1u64 } else { 0 }; // FALL vs JUMP
    let drag = if n.applied_drag > 0.95 { 1u64 } else { 0 }; // REGULAR vs SLOW
    // Conservative key: includes the buffer timers and surface normals that also
    // branch `think()`/collision, so merged states genuinely share a future
    // (no over-counting redundancy by ignoring future-determining state).
    let qn = |v: f64| -> u64 { (v * 8.0).round() as i64 as u64 }; // ~0.125 normal grain
    let fields: [u64; 19] = [
        (n.xpos / qpos).floor() as i64 as u64,
        (n.ypos / qpos).floor() as i64 as u64,
        (n.xspeed / qvel).floor() as i64 as u64,
        (n.yspeed / qvel).floor() as i64 as u64,
        n.state as u64,
        n.jump_duration as i64 as u64,
        n.jump_buffer as i64 as u64,
        n.floor_buffer as i64 as u64,
        n.wall_buffer as i64 as u64,
        n.launch_pad_buffer as i64 as u64,
        n.airborn as u64,
        n.walled as u64,
        grav,
        drag,
        qn(n.floor_normal_x),
        qn(n.floor_normal_y),
        qn(n.wall_normal),
        h as i64 as u64,
        j as i64 as u64,
    ];
    let mut hsh = 1469598103934665603u64; // FNV-1a offset basis
    for v in fields {
        hsh ^= v;
        hsh = hsh.wrapping_mul(1099511628211);
    }
    hsh
}

/// Run the same arc sweep as `enumerate_arcs_parallel` but, instead of emitting
/// edges, record every arc tick's quantized `(pre-state, input)` key.  Returns
/// `(total_ticks, distinct_keys)` — the ratio is the redundancy a transposition-
/// merging reachability flood could collapse (per-level, at this quantization).
#[allow(clippy::too_many_arguments)]
pub fn flood_stats_parallel(
    map_data: &[i32],
    nodes: &[NodeSpec],
    sw: &SweepParams,
    mapper: &NodeMapper,
    n_threads: usize,
    qpos: f64,
    qvel: f64,
) -> (u64, usize, u64, usize) {
    type Set = std::collections::HashSet<u64>;
    let n = nodes.len();
    if n == 0 {
        return (0, 0, 0, 0);
    }
    let n_chunks = n_threads.max(1).min(n);
    let chunk_size = (n + n_chunks - 1) / n_chunks;

    let run_chunk = |chunk: &[NodeSpec]| -> (u64, Set, u64, Set) {
        let mut sim = Simulator::load(map_data);
        sim.flood_keys = Some(Set::new());
        sim.flood_keys_coll = Some(Set::new());
        sim.flood_qpos = qpos;
        sim.flood_qvel = qvel;
        let snap = sim.ninja.clone();
        let mut out = NodeBatchOut::default();
        for node in chunk {
            sim.enumerate_node_arcs(node, sw, mapper, &snap, &mut out);
        }
        (
            sim.flood_ticks,
            sim.flood_keys.take().unwrap(),
            sim.flood_ticks_coll,
            sim.flood_keys_coll.take().unwrap(),
        )
    };

    let chunks: Vec<&[NodeSpec]> = nodes.chunks(chunk_size).collect();
    let partials: Vec<(u64, Set, u64, Set)> =
        chunks.par_iter().map(|c| run_chunk(c)).collect();

    let mut total_ticks = 0u64;
    let mut coll_ticks = 0u64;
    let mut merged: Set = Set::new();
    let mut merged_coll: Set = Set::new();
    for (ticks, set, cticks, cset) in partials {
        total_ticks += ticks;
        coll_ticks += cticks;
        merged.extend(set);
        merged_coll.extend(cset);
    }
    (total_ticks, merged.len(), coll_ticks, merged_coll.len())
}
