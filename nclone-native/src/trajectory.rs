//! Ballistic ghost-arc projection (Audit II P0-1 / §2 unifying primitive).
//!
//! Integrates the REAL airborne recurrence (`v *= drag; vy += g; pos += v`,
//! `vx += AIR_ACCEL * u` capped — byte-parity with `ninja.rs::integrate` +
//! the air branch of `think`) from a given (x, y, vx, vy) under a constant
//! horizontal-input hypothesis u ∈ {-1, 0, +1} until first contact with a
//! solid tile segment (exact `sweep_circle_vs_tiles`, the same sweep the sim
//! uses), a deadly-mine kill radius, or the frame cap. Exact up to collision
//! resolution (depenetration substeps are not simulated — the arc stops at
//! first sweep contact).
//!
//! One projector serves four consumers: the P0-1 obs bundle
//! (`project_bundle`, 3 arcs x 9 dims), the P0-2 ballistic airborne potential
//! (`project_landing`, u = 0), the P1-8 ghost-arc tau upgrade, and the P1-10
//! landing-prediction aux-head targets. Level geometry is loaded once per
//! level; the deadly-mine list is refreshed on the hazard-map signature
//! (mines toggle within an episode).

use pyo3::prelude::*;

use crate::collision::sweep::{get_single_closest_point, sweep_circle_vs_tiles};
use crate::constants::{
    AIR_ACCEL, DRAG_REGULAR, GRAVITY_FALL, GRAVITY_JUMP, MAP_HEIGHT, MAP_WIDTH,
    MAX_HOR_SPEED, MAX_SURVIVABLE_IMPACT, NINJA_RADIUS, TILE_SIZE,
};
use crate::map::loader::{load_map, MapData};
use crate::segments::Segment;

/// Ninja center-to-mine-center kill distance: ninja radius (10) + toggled
/// mine radius (4). Matches `mine_sdf::MINE_KILL_RADIUS_PX`.
const MINE_KILL_RADIUS_PX: f64 = 14.0;
/// Mine-adjacency margin used for the clear-surface-width probe: a landing
/// tile whose surface point is within this of a deadly mine is not "clear".
const MINE_CLEAR_MARGIN_PX: f64 = 18.0;
/// Spatial-hash cell for the deadly-mine index (3x3 neighborhood covers the
/// kill radius + margin).
const MINE_HASH_CELL_PX: f64 = 24.0;
/// Max contiguous clear tiles counted per side of the contact point.
const CLEAR_WIDTH_CAP_PER_SIDE: i32 = 4;
/// Normalizer for the vertical contact-velocity obs feature: the drag-terminal
/// fall speed `GRAVITY_FALL / (1 - DRAG_REGULAR) ≈ 9.98` px/frame. Vertical speed
/// is unbounded by MAX_HOR_SPEED, so terminal velocity is its natural scale;
/// MAX_SURVIVABLE_IMPACT (6.0) sits at ~0.6 of it, keeping the survivability
/// boundary mid-range for the policy MLP.
const VY_CONTACT_NORM_PX: f64 = 10.0;
/// v15 threading-margin: the graded "mine proximity" obs feature ramps from 0 at
/// (MINE_KILL_RADIUS_PX + this) px of closest approach to 1 at the kill radius.
/// 34 px puts the full danger zone at ~48 px, inside the deadly-mine spatial
/// hash's reliable lookup radius (nearest_mine_dist_sq returns INFINITY beyond
/// the 3x3 neighborhood, which maps to proximity 0 = "no mine near this arc").
const MINE_PROXIMITY_RANGE_PX: f64 = 34.0;
/// v16 (feature C): arc->goal proximity. The active objective (switch pre / exit
/// post) is fed in via `set_goal`. GOAL_HIT_RADIUS_PX is the closest-approach at
/// (or below) which the arc is treated as ARRIVING at the goal (boolean hit=1 +
/// graded=1); ~NINJA_RADIUS + entity margin. GOAL_PROXIMITY_RANGE_PX (4 tiles) is
/// the ramp beyond it — the graded "am I getting warmer" gradient the mostly-zero
/// boolean lacks. Wider than the mine range because the goal is a TARGET to steer
/// toward (useful from further out), not a point hazard to thread by a tile.
const GOAL_HIT_RADIUS_PX: f64 = 12.0;
const GOAL_PROXIMITY_RANGE_PX: f64 = 96.0;

/// Contact classes (returned raw; the obs bundle encodes mine/none as flags
/// and leaves floor/wall/ceiling readable from the surface normal).
pub const CONTACT_FLOOR: i32 = 0;
pub const CONTACT_WALL: i32 = 1;
pub const CONTACT_CEILING: i32 = 2;
pub const CONTACT_MINE: i32 = 3;
pub const CONTACT_NONE: i32 = 4;

/// Raw single-arc projection result.
pub struct ArcContact {
    pub class_: i32,
    /// Ninja center position at contact (raw sim px).
    pub x: f64,
    pub y: f64,
    /// Surface normal at contact (unit, pointing away from the surface into
    /// the ninja). Zero for mine/none contacts.
    pub nx: f64,
    pub ny: f64,
    /// Frames to contact (fractional at the sweep intersection).
    pub tau: f64,
    /// Speed INTO the surface at contact (>= 0; 0 for none).
    pub impact_speed: f64,
    /// Survivable-impact threshold at this surface (MAX_SURVIVABLE_IMPACT -
    /// 4/3 |ny|, the sim's floor/ceiling impact rule). 0 for mine/none.
    pub impact_threshold: f64,
    /// Contiguous mine-free standable tiles at the contact (1..=9), 0 when
    /// there is no surface contact.
    pub clear_width_tiles: i32,
    /// Ninja velocity at contact (raw sim px/frame). For a tile contact this is
    /// the post-gravity swept velocity that produced the impact (the same vx,vy
    /// `impact_speed` is projected from); for a mine/none contact it is the
    /// current-frame velocity. The tangential (sliding) component and the raw
    /// signed axes are invisible in `impact_speed` alone — this is the v15
    /// landing-velocity feature (F5 impact-death control).
    pub vx: f64,
    pub vy: f64,
    /// v15: the CLOSEST the arc's center comes to any deadly mine over its whole
    /// flight (raw px), INFINITY if no mine was ever near. Unlike the terminal
    /// mine-contact flag (binary "does this arc die to a mine"), this is the
    /// graded threading margin — how much clearance a SURVIVING arc keeps from
    /// the mines it passes. Targets the high-tier mine-death failure mass
    /// (VERTICAL_SHAFT/T3, MULTI_JUMP/T4, PRECISION) the held-out eval surfaced.
    pub min_mine_clearance_px: f64,
    /// v16 (feature C): the CLOSEST the arc's center comes to the ACTIVE goal
    /// (switch pre / exit post) over its whole flight (raw px), INFINITY when no
    /// goal is set. Action-conditioned goal lookahead — "does THIS commitment
    /// arrive at the objective" — the symmetric counterpart to
    /// `min_mine_clearance_px` (danger) that the bundle otherwise lacks.
    pub min_goal_clearance_px: f64,
}

impl ArcContact {
    /// Flatten to the 13-element list the Python single-arc API returns
    /// (class, cx, cy, nx, ny, tau, impact, threshold, clear_width, contact_vx,
    /// contact_vy, min_mine_clearance, min_goal_clearance). class_ and
    /// clear_width_tiles are widened to f64 (Python consumers cast them back).
    fn as_flat_list(&self) -> Vec<f64> {
        vec![
            self.class_ as f64,
            self.x,
            self.y,
            self.nx,
            self.ny,
            self.tau,
            self.impact_speed,
            self.impact_threshold,
            self.clear_width_tiles as f64,
            self.vx,
            self.vy,
            self.min_mine_clearance_px,
            self.min_goal_clearance_px,
        ]
    }
}

#[pyclass]
pub struct TrajectoryProjector {
    map: MapData,
    deadly_mines: Vec<(f64, f64)>,
    /// Uniform-grid spatial hash over `deadly_mines` (cell -> mine indices);
    /// keeps the per-frame mine check O(1) even on 13k-mine levels.
    mine_hash: std::collections::HashMap<(i32, i32), Vec<usize>>,
    /// v16 (feature C): the active objective point (switch pre-activation, exit
    /// post) fed in from Python each reset / phase flip. None = no goal set ->
    /// goal features read 0.
    goal: Option<(f64, f64)>,
}

impl TrajectoryProjector {
    fn hash_cell(x: f64, y: f64) -> (i32, i32) {
        (
            (x / MINE_HASH_CELL_PX).floor() as i32,
            (y / MINE_HASH_CELL_PX).floor() as i32,
        )
    }

    /// Squared distance to the nearest deadly mine within the 3x3 hash
    /// neighborhood (INFINITY when none nearby — sufficient for threshold
    /// checks up to ~MINE_HASH_CELL_PX).
    fn nearest_mine_dist_sq(&self, x: f64, y: f64) -> f64 {
        let (cx, cy) = Self::hash_cell(x, y);
        let mut best = f64::INFINITY;
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(idxs) = self.mine_hash.get(&(cx + dx, cy + dy)) {
                    for &i in idxs {
                        let (mx, my) = self.deadly_mines[i];
                        let d = (x - mx) * (x - mx) + (y - my) * (y - my);
                        if d < best {
                            best = d;
                        }
                    }
                }
            }
        }
        best
    }

    /// Squared distance from (x, y) to the active goal, INFINITY if none set.
    fn goal_dist_sq(&self, x: f64, y: f64) -> f64 {
        match self.goal {
            Some((gx, gy)) => (x - gx) * (x - gx) + (y - gy) * (y - gy),
            None => f64::INFINITY,
        }
    }

    fn segments(&self) -> (&[[Vec<usize>; MAP_HEIGHT]; MAP_WIDTH], &[Segment]) {
        (&self.map.segment_grid, &self.map.segments)
    }

    /// Surface normal at a contact position: unit vector from the closest
    /// segment point toward the ninja center (the sim's depenetration
    /// direction). Falls back to (0, -1) on a degenerate zero-length result.
    fn contact_normal(&self, x: f64, y: f64) -> (f64, f64) {
        let (grid, segs) = self.segments();
        let (found, closest) = get_single_closest_point(grid, segs, x, y, NINJA_RADIUS + 2.0);
        if found == 0 {
            return (0.0, -1.0);
        }
        let (a, b) = closest.unwrap();
        let dx = x - a;
        let dy = y - b;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-9 {
            (0.0, -1.0)
        } else {
            (dx / len, dy / len)
        }
    }

    /// Contiguous mine-free standable extent through the contact point, in
    /// tiles (1..=1+2*CAP). Marches the surface tangent in +-24 px steps; a
    /// step is "clear" iff a front-facing surface still exists within the
    /// probe radius AND no deadly mine sits within MINE_CLEAR_MARGIN_PX of
    /// the probed surface point. This is the quantity the audit found
    /// unrepresented at ANY range (coverage table row "contact-surface clear
    /// width").
    fn clear_surface_width(&self, cx: f64, cy: f64, nx: f64, ny: f64) -> i32 {
        let (grid, segs) = self.segments();
        // Surface tangent (unit, since the normal is unit).
        let tx = -ny;
        let ty = nx;
        let mut width = 1i32;
        // The contact tile itself must be mine-clear; otherwise width = 0 is
        // never emitted here because a mine contact classifies as MINE first.
        for side in [-1.0f64, 1.0] {
            for k in 1..=CLEAR_WIDTH_CAP_PER_SIDE {
                let px = cx + side * (k as f64) * TILE_SIZE * tx;
                let py = cy + side * (k as f64) * TILE_SIZE * ty;
                // Probe from one ninja-radius off the surface, like the ninja
                // would stand there: require a front-facing surface within
                // radius + 4 px (surface continues) ...
                let (found, closest) =
                    get_single_closest_point(grid, segs, px, py, NINJA_RADIUS + 4.0);
                let mut clear = found == 1;
                if clear {
                    let (a, b) = closest.unwrap();
                    // ... and no deadly mine adjacent to that surface point.
                    if self.nearest_mine_dist_sq(a, b)
                        < MINE_CLEAR_MARGIN_PX * MINE_CLEAR_MARGIN_PX
                        || self.nearest_mine_dist_sq(px, py)
                            < MINE_CLEAR_MARGIN_PX * MINE_CLEAR_MARGIN_PX
                    {
                        clear = false;
                    }
                } else {
                    clear = false;
                }
                if !clear {
                    break;
                }
                width += 1;
            }
        }
        width
    }

    /// Integrate one ghost arc to first contact. `jump_frames_left` frames of
    /// GRAVITY_JUMP remain (0 = falling; the caller derives this from the
    /// ninja's jump state/duration — release and the 45-frame cap both end
    /// the reduced-gravity phase, `ninja.think` air branch).
    pub fn project(
        &self,
        x0: f64,
        y0: f64,
        vx0: f64,
        vy0: f64,
        u: i32,
        jump_frames_left: i32,
        max_frames: i32,
    ) -> ArcContact {
        self.project_traced(x0, y0, vx0, vy0, u, jump_frames_left, max_frames, None)
    }

    /// `project` with an optional per-frame position trace (the dev-tool arc
    /// overlay draws the exact polyline the obs bundle was computed from).
    #[allow(clippy::too_many_arguments)]
    pub fn project_traced(
        &self,
        x0: f64,
        y0: f64,
        vx0: f64,
        vy0: f64,
        u: i32,
        jump_frames_left: i32,
        max_frames: i32,
        mut trace: Option<&mut Vec<(f64, f64)>>,
    ) -> ArcContact {
        let (grid, segs) = self.segments();
        let mut x = x0;
        let mut y = y0;
        let mut vx = vx0;
        let mut vy = vy0;
        let mut jf = jump_frames_left.max(0);
        let uu = u.clamp(-1, 1) as f64;
        // v15: track the closest approach to any deadly mine over the flight
        // (squared px). Seed with the start position so a launch already hugging
        // a mine registers. INFINITY = no mine ever within the hash lookup.
        let mut min_mine_sq = self.nearest_mine_dist_sq(x, y);
        // v16 (feature C): closest approach to the active goal over the flight.
        let mut min_goal_sq = self.goal_dist_sq(x, y);
        if let Some(t) = trace.as_deref_mut() {
            t.push((x, y));
        }

        for f in 0..max_frames.max(1) {
            // integrate(): drag, gravity, position — exact recurrence.
            let g = if jf > 0 { GRAVITY_JUMP } else { GRAVITY_FALL };
            vx *= DRAG_REGULAR;
            vy *= DRAG_REGULAR;
            vy += g;

            // First tile contact along this frame's displacement (the sim's
            // own sweep, radius = ninja radius).
            let t = sweep_circle_vs_tiles(grid, segs, x, y, vx, vy, NINJA_RADIUS);
            if t < 1.0 {
                x += t * vx;
                y += t * vy;
                if let Some(tr) = trace.as_deref_mut() {
                    tr.push((x, y));
                }
                let (nx, ny) = self.contact_normal(x, y);
                let class_ = if ny <= -0.3 {
                    CONTACT_FLOOR
                } else if ny >= 0.3 {
                    CONTACT_CEILING
                } else {
                    CONTACT_WALL
                };
                let impact = (-(nx * vx + ny * vy)).max(0.0);
                let threshold = MAX_SURVIVABLE_IMPACT - (4.0 / 3.0) * ny.abs();
                let cw = self.clear_surface_width(x, y, nx, ny);
                min_mine_sq = min_mine_sq.min(self.nearest_mine_dist_sq(x, y));
                min_goal_sq = min_goal_sq.min(self.goal_dist_sq(x, y));
                return ArcContact {
                    class_,
                    x,
                    y,
                    nx,
                    ny,
                    tau: f as f64 + t,
                    impact_speed: impact,
                    impact_threshold: threshold,
                    clear_width_tiles: cw,
                    vx,
                    vy,
                    min_mine_clearance_px: min_mine_sq.sqrt(),
                    min_goal_clearance_px: min_goal_sq.sqrt(),
                };
            }
            x += vx;
            y += vy;
            if let Some(tr) = trace.as_deref_mut() {
                tr.push((x, y));
            }

            // Deadly-mine kill radius at the new position (post-move check —
            // the same granularity as the sim's per-frame mine logic).
            let mine_sq = self.nearest_mine_dist_sq(x, y);
            min_mine_sq = min_mine_sq.min(mine_sq);
            min_goal_sq = min_goal_sq.min(self.goal_dist_sq(x, y));
            if mine_sq <= MINE_KILL_RADIUS_PX * MINE_KILL_RADIUS_PX {
                return ArcContact {
                    class_: CONTACT_MINE,
                    x,
                    y,
                    nx: 0.0,
                    ny: 0.0,
                    tau: (f + 1) as f64,
                    impact_speed: 0.0,
                    impact_threshold: 0.0,
                    clear_width_tiles: 0,
                    vx,
                    vy,
                    min_mine_clearance_px: min_mine_sq.sqrt(),
                    min_goal_clearance_px: min_goal_sq.sqrt(),
                };
            }

            // think() air branch: horizontal air accel under the input
            // hypothesis, applied for the NEXT frame, capped at MAX_HOR_SPEED.
            let vxn = vx + AIR_ACCEL * uu;
            if vxn.abs() < MAX_HOR_SPEED {
                vx = vxn;
            }
            if jf > 0 {
                jf -= 1;
            }
        }

        ArcContact {
            class_: CONTACT_NONE,
            x,
            y,
            nx: 0.0,
            ny: 0.0,
            tau: max_frames.max(1) as f64,
            impact_speed: 0.0,
            impact_threshold: 0.0,
            clear_width_tiles: 0,
            vx,
            vy,
            min_mine_clearance_px: min_mine_sq.sqrt(),
            min_goal_clearance_px: min_goal_sq.sqrt(),
        }
    }
}

#[pymethods]
impl TrajectoryProjector {
    /// Build the projector for a level from raw integer map data (tile
    /// segments only are used; entities are parsed but ignored — mines come
    /// in via `set_deadly_mines` because they toggle within an episode).
    #[staticmethod]
    fn load(map_data: Vec<i32>) -> Self {
        TrajectoryProjector {
            map: load_map(&map_data),
            deadly_mines: Vec::new(),
            mine_hash: std::collections::HashMap::new(),
            goal: None,
        }
    }

    /// v16 (feature C): set the ACTIVE objective point the arc->goal features
    /// measure closest-approach to. Call on reset and whenever the objective
    /// flips (switch -> exit); pass the switch center pre-activation and the exit
    /// center post-activation so the lookahead always tracks the current phase.
    fn set_goal(&mut self, x: f64, y: f64) {
        self.goal = Some((x, y));
    }

    /// v16 (feature C): clear the active goal (goal features read 0). Used when
    /// no objective point is available (e.g. a degenerate level).
    fn clear_goal(&mut self) {
        self.goal = None;
    }

    /// Refresh the DEADLY (toggled) mine set. Call on the hazard-map
    /// signature change, not per frame.
    fn set_deadly_mines(&mut self, xs: Vec<f64>, ys: Vec<f64>) {
        self.deadly_mines = xs.into_iter().zip(ys).collect();
        self.mine_hash.clear();
        for (i, (x, y)) in self.deadly_mines.iter().enumerate() {
            // Insert into every cell the kill+margin disc can reach so the
            // 3x3 neighborhood lookup is sufficient.
            let (cx, cy) = Self::hash_cell(*x, *y);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    self.mine_hash
                        .entry((cx + dx, cy + dy))
                        .or_default()
                        .push(i);
                }
            }
        }
    }

    /// Single-arc projection; returns
    /// (class, contact_x, contact_y, nx, ny, tau_frames, impact_speed,
    ///  impact_threshold, clear_width_tiles, contact_vx, contact_vy,
    ///  min_mine_clearance_px).
    /// class: 0=floor, 1=wall, 2=ceiling, 3=deadly mine, 4=none.
    /// contact_vx/vy + min_mine_clearance (v15) appended at the end so positional
    /// consumers of the first 9 fields are unaffected. min_mine_clearance_px is
    /// f64::INFINITY when no deadly mine came near the arc.
    #[pyo3(name = "project")]
    #[allow(clippy::too_many_arguments)]
    fn project_py(
        &self,
        x: f64,
        y: f64,
        vx: f64,
        vy: f64,
        u: i32,
        jump_frames_left: i32,
        max_frames: i32,
    ) -> Vec<f64> {
        // 13-element flat list (was a 12-tuple; pyo3 caps IntoPy tuples at 12, and
        // v16 appends min_goal_clearance). class_ and clear_width_tiles ride as
        // f64 — every Python consumer already casts them (int(c[0]), int(c[8])).
        let c = self.project(x, y, vx, vy, u, jump_frames_left, max_frames);
        c.as_flat_list()
    }

    /// Single-arc projection WITH the frame-by-frame position polyline —
    /// the dev-tool arc overlay's query (draws the exact arc the obs bundle
    /// was computed from). Returns
    /// ((class, cx, cy, nx, ny, tau, impact, threshold, clear_width,
    ///   contact_vx, contact_vy, min_mine_clearance_px), [(x, y), ...]).
    #[pyo3(name = "project_with_path")]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn project_with_path_py(
        &self,
        x: f64,
        y: f64,
        vx: f64,
        vy: f64,
        u: i32,
        jump_frames_left: i32,
        max_frames: i32,
    ) -> (Vec<f64>, Vec<(f64, f64)>) {
        // (13-element flat contact list, polyline). See project_py for the
        // tuple->list switch rationale.
        let mut path: Vec<(f64, f64)> = Vec::with_capacity(max_frames.max(1) as usize + 2);
        let c = self.project_traced(
            x, y, vx, vy, u, jump_frames_left, max_frames, Some(&mut path),
        );
        (c.as_flat_list(), path)
    }

    /// Landing point of the NO-INPUT arc (u = 0) — the P0-2 ballistic
    /// airborne potential's query. Returns (x, y, contact_class, tau).
    fn project_landing(
        &self,
        x: f64,
        y: f64,
        vx: f64,
        vy: f64,
        jump_frames_left: i32,
        max_frames: i32,
    ) -> (f64, f64, i32, f64) {
        let c = self.project(x, y, vx, vy, 0, jump_frames_left, max_frames);
        (c.x, c.y, c.class_, c.tau)
    }

    /// P0-1 observation bundle: u in {-1, 0, +1}, 15 dims per arc, 45 total.
    ///
    /// Per-arc layout (all normalized, policy-ready):
    ///   [0]  contact dx / 500 px, clipped [-1, 1]
    ///   [1]  contact dy / 500 px, clipped [-1, 1]
    ///   [2]  surface normal nx (0 for mine/none)
    ///   [3]  surface normal ny (0 for mine/none)
    ///   [4]  deadly-mine contact flag (0/1)
    ///   [5]  no-contact-within-horizon flag (0/1)
    ///   [6]  clear surface width / 9 tiles, in [0, 1]
    ///   [7]  impact-speed ratio vs survivable threshold, clipped [0, 2] / 2
    ///   [8]  time-to-contact / 60 frames, clipped [0, 1]
    ///   [9]  contact vx / MAX_HOR_SPEED, clipped [-1, 1]          (v15)
    ///   [10] contact vy / VY_CONTACT_NORM_PX, clipped [-1, 1]     (v15)
    ///   [11] surface_class: +1 floor / 0 wall / -1 ceiling;       (v15)
    ///        0 for mine/none (already flagged at [4]/[5])
    ///   [12] mine proximity (threading margin): 0 = no mine within range of    (v15)
    ///        the flight, ->1 as the arc's closest approach nears the kill
    ///        radius. Graded companion to the binary contact flag [4] — tells
    ///        the policy how much clearance each SURVIVING steering choice keeps.
    ///   [13] goal approach (graded): 0 = the active goal is beyond the proximity  (v16 C)
    ///        range of this arc's flight, ->1 as the arc's closest approach nears
    ///        the goal-hit radius. Action-conditioned "am I getting warmer" — the
    ///        gradient the mostly-zero boolean [14] lacks. 0 when no goal is set.
    ///   [14] goal hit (0/1): this arc's flight passes within the goal-collection  (v16 C)
    ///        radius of the active objective (switch pre / exit post) — the
    ///        decisive "this commitment REACHES the objective" bit.
    ///
    /// v15 APPEND-ONLY: indices 9-12 are new; 0-8 are byte-identical to the
    /// v14 layout so every positional consumer (arc-doomed flag reads [4]/[5]/
    /// [7]; arc-phi reads [0]/[1]/[4]/[5]) is unaffected. Indices 9-10 expose
    /// the full signed landing velocity (not just its normal projection in the
    /// [7] impact ratio) — the tangential/sliding component that F5 landing
    /// control needs. Index 11 makes "will land on a standable floor" vs "will
    /// smack a wall/ceiling" explicit rather than a threshold on [3]. Index 12
    /// is the graded mine-threading margin (the held-out eval's dominant failure
    /// is high-tier mine deaths, not impacts).
    ///
    /// The u = +-1 arcs ARE the bang-bang controllability envelope endpoints:
    /// their contact offsets span the reachable-landing interval (up to ~28
    /// tiles on a full fall) that no other observation carries.
    fn project_bundle(
        &self,
        x: f64,
        y: f64,
        vx: f64,
        vy: f64,
        jump_frames_left: i32,
        max_frames: i32,
    ) -> Vec<f64> {
        let mut out = Vec::with_capacity(45);
        for u in [-1i32, 0, 1] {
            let c = self.project(x, y, vx, vy, u, jump_frames_left, max_frames);
            let is_mine = c.class_ == CONTACT_MINE;
            let is_none = c.class_ == CONTACT_NONE;
            out.push(((c.x - x) / 500.0).clamp(-1.0, 1.0));
            out.push(((c.y - y) / 500.0).clamp(-1.0, 1.0));
            out.push(c.nx);
            out.push(c.ny);
            out.push(if is_mine { 1.0 } else { 0.0 });
            out.push(if is_none { 1.0 } else { 0.0 });
            out.push((c.clear_width_tiles as f64 / 9.0).clamp(0.0, 1.0));
            let ratio = if c.impact_threshold > 1e-9 {
                (c.impact_speed / c.impact_threshold).clamp(0.0, 2.0) / 2.0
            } else {
                0.0
            };
            out.push(ratio);
            out.push((c.tau / 60.0).clamp(0.0, 1.0));
            // v15: full signed contact velocity + explicit surface class.
            out.push((c.vx / MAX_HOR_SPEED).clamp(-1.0, 1.0));
            out.push((c.vy / VY_CONTACT_NORM_PX).clamp(-1.0, 1.0));
            let surface_class = match c.class_ {
                CONTACT_FLOOR => 1.0,
                CONTACT_CEILING => -1.0,
                _ => 0.0, // wall / mine / none
            };
            out.push(surface_class);
            // v15: graded mine-threading margin along the flight (0 = clear).
            let mine_prox = if c.min_mine_clearance_px.is_finite() {
                (1.0
                    - ((c.min_mine_clearance_px - MINE_KILL_RADIUS_PX)
                        / MINE_PROXIMITY_RANGE_PX)
                        .clamp(0.0, 1.0))
                .clamp(0.0, 1.0)
            } else {
                0.0
            };
            out.push(mine_prox);
            // v16 (feature C): graded arc->goal approach + boolean hit.
            let (goal_approach, goal_hit) = if c.min_goal_clearance_px.is_finite() {
                let clr = c.min_goal_clearance_px;
                let graded = (1.0
                    - ((clr - GOAL_HIT_RADIUS_PX) / GOAL_PROXIMITY_RANGE_PX)
                        .clamp(0.0, 1.0))
                .clamp(0.0, 1.0);
                let hit = if clr <= GOAL_HIT_RADIUS_PX { 1.0 } else { 0.0 };
                (graded, hit)
            } else {
                (0.0, 0.0)
            };
            out.push(goal_approach);
            out.push(goal_hit);
        }
        out
    }
}
