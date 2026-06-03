//! nclone_native: Rust physics engine for N++ simulation.
//! Exposes Simulator, state types, and constants to Python via PyO3.

use crate::constants::{MAP_HEIGHT, MAP_WIDTH};
use pyo3::types::PyDict;

pub mod collision;
pub mod constants;
pub mod entities;
pub mod enumerate;
pub mod level_cache;
pub mod mine_sdf;
pub mod pathfinding;
pub mod map;
pub mod ninja;
pub mod segments;
pub mod simulator;
pub mod types;
pub mod vecenv;

use pyo3::prelude::*;

use types::{EntitySnapshot, NinjaSnapshot, SimulatorState};

/// Python-facing Simulator wrapper.
#[pyclass(name = "Simulator")]
pub struct PySimulator {
    inner: simulator::Simulator,
}

#[pymethods]
impl PySimulator {
    /// Load a map from integer map data.
    #[staticmethod]
    fn load(map_data: Vec<i32>) -> Self {
        PySimulator {
            inner: simulator::Simulator::load(&map_data),
        }
    }

    /// Execute one full physics frame.
    fn tick(&mut self, hor_input: i32, jump_input: i32) {
        self.inner.tick(hor_input, jump_input);
    }

    /// Execute one physics frame without entity AI.
    fn tick_physics_only(&mut self, hor_input: i32, jump_input: i32) {
        self.inner.tick_physics_only(hor_input, jump_input);
    }

    /// Execute one physics frame with tile collisions only — no entity interactions.
    ///
    /// Use this for edge enumeration and arc simulation so that exit switches, exit
    /// doors, and toggle mines cannot mutate simulator entity state across arc runs.
    /// Geometrically equivalent to tick_physics_only for pure-tile geometry.
    fn tick_tiles_only(&mut self, hor_input: i32, jump_input: i32) {
        self.inner.tick_tiles_only(hor_input, jump_input);
    }

    /// Execute one physics frame with tile collisions + toggle-mine DEATH only.
    ///
    /// Same ninja trajectory as tick_tiles_only, but kills the ninja on contact
    /// with a currently-deadly toggle mine (state 0) WITHOUT mutating any entity
    /// (the mine is not toggled to safe), so ninja-only save/restore stays valid
    /// across edge-enumeration arcs.  Switch/door logical collisions are skipped.
    /// Use for entity-aware edge enumeration: prune arcs by real physics death
    /// instead of a geometric waypoint-clearance test.
    fn tick_physics_mines_only(&mut self, hor_input: i32, jump_input: i32) {
        self.inner.tick_physics_mines_only(hor_input, jump_input);
    }

    /// Fast reset for same-level training.
    fn fast_reset(&mut self) {
        self.inner.fast_reset();
    }

    /// Get the current simulator state as a snapshot.
    fn get_state(&self) -> SimulatorState {
        self.inner.get_state()
    }

    /// Get the flat tile grid (row-major, 1100 bytes).
    fn get_tile_grid(&self) -> Vec<u8> {
        self.inner.get_tile_grid_flat()
    }

    /// Get current frame number.
    #[getter]
    fn frame(&self) -> u32 {
        self.inner.frame
    }

    /// Set the position of an entity by index. Used by curriculum manager.
    fn set_entity_position(&mut self, entity_index: usize, x: f64, y: f64) {
        if entity_index < self.inner.entities.len() {
            match &mut self.inner.entities[entity_index] {
                entities::Entity::ToggleMine(e) => {
                    e.xpos = x;
                    e.ypos = y;
                }
                entities::Entity::Gold(e) => {
                    e.xpos = x;
                    e.ypos = y;
                }
                entities::Entity::ExitSwitch(e) => {
                    e.xpos = x;
                    e.ypos = y;
                }
                entities::Entity::ExitDoor(e) => {
                    e.xpos = x;
                    e.ypos = y;
                }
                entities::Entity::LockedDoor(e) => {
                    // Position is the switch; the door barrier segment is static.
                    e.xpos = x;
                    e.ypos = y;
                }
            }
            // Rebuild entity grid after position change
            self.inner.rebuild_entity_grid();
        }
    }

    /// Get ninja position as (x, y) tuple.
    fn get_ninja_pos(&self) -> (f64, f64) {
        (self.inner.ninja.xpos, self.inner.ninja.ypos)
    }

    /// Check if ninja has won.
    fn has_won(&self) -> bool {
        self.inner.ninja.has_won()
    }

    /// Number of gold pieces collected so far (fast getter for gold-aware search).
    fn get_ninja_gold_collected(&self) -> u32 {
        self.inner.ninja.gold_collected
    }

    /// Check if ninja has died.
    fn has_died(&self) -> bool {
        self.inner.ninja.has_died()
    }

    /// Save ninja state for transition enumeration (stateless queries).
    fn save_ninja_state(&self) -> Vec<u8> {
        self.inner.ninja.to_bytes()
    }

    /// Restore ninja state from bytes (from save_ninja_state).
    fn restore_ninja_state(&mut self, data: Vec<u8>) {
        self.inner.ninja = ninja::Ninja::from_bytes(&data);
    }

    /// Inject position/velocity for transition enumeration.
    fn inject_ninja_for_enumeration(&mut self, x: f64, y: f64, vx: f64, vy: f64) {
        self.inner.ninja.inject_enumeration_state(x, y, vx, vy);
    }

    /// Inject full 5D state for value iteration (position, velocity, jump_duration).
    /// Sets jump_duration and gravity (Jumping vs Falling) unlike inject_ninja_for_enumeration.
    fn inject_ninja_for_value_iteration(
        &mut self,
        x: f64,
        y: f64,
        vx: f64,
        vy: f64,
        jump_duration: i32,
    ) {
        self.inner
            .ninja
            .inject_value_iteration_state(x, y, vx, vy, jump_duration);
    }

    /// Inject grounded floor state for floor jump enumeration.
    ///
    /// Sets state=1 (Running) with active floor contact so that on the next
    /// tick with jump_input=1, think() fires floor_jump() with the correct
    /// floor normal (enabling slope-jump impulse branches).  Also fixes the
    /// ground-running simulation: inject then run tick_tiles_only(h, 0) for k
    /// frames to get exact k-frame running edges without the airborne artifact.
    ///
    /// floor_normalized_x/y: outward floor normal (unit vector, away from solid).
    fn inject_ninja_for_value_iteration_grounded(
        &mut self,
        x: f64,
        y: f64,
        vx: f64,
        vy: f64,
        floor_normalized_x: f64,
        floor_normalized_y: f64,
    ) {
        self.inner.ninja.inject_value_iteration_grounded(
            x, y, vx, vy, floor_normalized_x, floor_normalized_y,
        );
    }

    /// Inject wall-slide state for slide-wall-jump enumeration.
    ///
    /// Sets state=5 (WallSliding) with wall contact.  Call tick_tiles_only with
    /// jump_input=1 and h_input = -sign(wall_normal) (toward wall) to trigger
    /// the weak slide-wall-jump impulse (vx=±0.667, vy=−1.0).
    ///
    /// wall_normal: +1 if normal points right (left wall), -1 if left (right wall).
    fn inject_ninja_for_value_iteration_wall_slide(
        &mut self,
        x: f64,
        y: f64,
        vy: f64,
        wall_normal: f64,
    ) {
        self.inner
            .ninja
            .inject_value_iteration_wall_slide(x, y, vy, wall_normal);
    }

    /// Get ninja velocity as (xspeed, yspeed) tuple.
    fn get_ninja_velocity(&self) -> (f64, f64) {
        (self.inner.ninja.xspeed, self.inner.ninja.yspeed)
    }

    /// Get ninja jump duration.
    fn get_ninja_jump_duration(&self) -> i32 {
        self.inner.ninja.jump_duration
    }

    /// Get ninja state code (for death detection: 6=Dead, 7=AwaitingDeath).
    fn get_ninja_state_code(&self) -> u8 {
        self.inner.ninja.state
    }

    /// True while the ninja is airborne (not in floor contact this step).
    fn get_ninja_airborn(&self) -> bool {
        self.inner.ninja.airborn
    }

    /// True while the ninja is in wall contact this step.
    fn get_ninja_walled(&self) -> bool {
        self.inner.ninja.walled
    }

    /// Wall-jump buffer counter (-1 = inactive; 0..<5 = live). Surfaces a
    /// Corner-Double-Jump's wall buffer so corner detection runs natively.
    fn get_ninja_wall_buffer(&self) -> i32 {
        self.inner.ninja.wall_buffer
    }

    /// Floor-jump buffer counter (-1 = inactive; 0..<5 = live). `wall_jump`
    /// does NOT reset this, which is what lets the CDJ stack a floor jump.
    fn get_ninja_floor_buffer(&self) -> i32 {
        self.inner.ninja.floor_buffer
    }

    /// Debug: get total number of segments.
    fn debug_segment_count(&self) -> usize {
        self.inner.segments.len()
    }

    /// Debug: get segments in a specific cell as list of [type_str, x1, y1, x2, y2].
    fn debug_cell_segments(&self, cx: usize, cy: usize) -> Vec<Vec<f64>> {
        let mut result = Vec::new();
        if cx < MAP_WIDTH && cy < MAP_HEIGHT {
            for &idx in &self.inner.segment_grid[cx][cy] {
                let seg = &self.inner.segments[idx];
                match seg {
                    segments::Segment::Linear(s) => {
                        result.push(vec![0.0, s.x1, s.y1, s.x2, s.y2]);
                    }
                    segments::Segment::Circular(s) => {
                        result.push(vec![1.0, s.xpos, s.ypos, s.hor, s.ver]);
                    }
                }
            }
        }
        result
    }

    /// Debug: count segments per cell, returns Vec of (cx, cy, count).
    fn debug_segment_grid_counts(&self) -> Vec<(usize, usize, usize)> {
        let mut result = Vec::new();
        for x in 0..MAP_WIDTH {
            for y in 0..MAP_HEIGHT {
                let count = self.inner.segment_grid[x][y].len();
                if count > 0 {
                    result.push((x, y, count));
                }
            }
        }
        result
    }

    /// Get tile dict as Python dict {(col, row): tile_type} for GridClassifier.
    fn get_tile_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        for x in 0..MAP_WIDTH {
            for y in 0..MAP_HEIGHT {
                dict.set_item((x as i32, y as i32), self.inner.tile_grid[x][y])?;
            }
        }
        Ok(dict.unbind())
    }
}

/// Result of a batch arc enumeration (see `enumerate_arcs`).  Each per-edge
/// array has one entry per landed arc that mapped to a real, non-self
/// destination node (the landing -> node mapping is now done in-kernel); the
/// `gc_*` arrays are a sparse per-(node, goal) minimum contact-frame table for
/// air-reachable goals, keyed by source *node id*.
#[pyclass]
pub struct ArcBatch {
    #[pyo3(get)]
    pub src: Vec<i32>,
    #[pyo3(get)]
    pub dst: Vec<i32>,
    #[pyo3(get)]
    pub frames: Vec<i32>,
    #[pyo3(get)]
    pub jump_hold: Vec<i32>,
    #[pyo3(get)]
    pub h_during: Vec<i32>,
    #[pyo3(get)]
    pub h_after: Vec<i32>,
    #[pyo3(get)]
    pub clipped: Vec<bool>,
    #[pyo3(get)]
    pub gc_node: Vec<i32>,
    #[pyo3(get)]
    pub gc_goal: Vec<i32>,
    #[pyo3(get)]
    pub gc_frame: Vec<f64>,
}

/// Batch edge enumeration entirely in Rust (exact physics, rayon-parallel).
///
/// Mirrors `edge_enumeration._enumerate_node_edges` for every node, returning
/// the raw landed arcs (and air-goal contacts).  The landing -> graph-node
/// mapping, mine inflation and pruning stay in Python.  Sweep tables are passed
/// in so the canonical values live in `edge_enumeration.py`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn enumerate_arcs(
    py: Python<'_>,
    map_data: Vec<i32>,
    node_x: Vec<f64>,
    node_y: Vec<f64>,
    node_vx0: Vec<f64>,
    node_vy0: Vec<f64>,
    node_surface: Vec<u8>,
    node_is_edge: Vec<bool>,
    node_pt: Vec<i32>,
    node_vb: Vec<i32>,
    run_durations: Vec<i32>,
    jump_holds: Vec<i32>,
    h_inputs: Vec<i32>,
    post_release: Vec<i32>,
    max_arc_frames: i32,
    max_jump_duration: i32,
    tick_mines: bool,
    goals_flat: Vec<f64>,
    sp_x: Vec<f64>,
    sp_y: Vec<f64>,
    sp_type: Vec<i32>,
    floor_centers: Vec<f64>,
    wall_centers: Vec<f64>,
    slide_centers: Vec<f64>,
    launch_centers: Vec<f64>,
    spatial_cell_size: f64,
    max_vel_bins: i32,
    n_threads: usize,
    snap_qpos: f64,
    snap_qvel: f64,
    sp_tangent_x: Vec<f64>,
    sp_tangent_y: Vec<f64>,
    tangential: bool,
) -> PyResult<ArcBatch> {
    use enumerate::{enumerate_arcs_parallel, NodeMapper, NodeSpec, SweepParams};

    let n = node_x.len();
    let nodes: Vec<NodeSpec> = (0..n)
        .map(|i| NodeSpec {
            src_node: node_pt[i] * max_vel_bins + node_vb[i],
            x: node_x[i],
            y: node_y[i],
            vx0: node_vx0[i],
            vy0: node_vy0[i],
            surface_type: node_surface[i],
            is_edge: node_is_edge[i],
        })
        .collect();
    let goals: Vec<(f64, f64, f64)> = goals_flat
        .chunks_exact(3)
        .map(|c| (c[0], c[1], c[2]))
        .collect();

    // f32 bin centers (the Python arrays are np.float32; the argmin runs in f32).
    let to_f32 = |v: Vec<f64>| -> Vec<f32> { v.into_iter().map(|c| c as f32).collect() };
    let mapper = NodeMapper::new(
        &sp_x,
        &sp_y,
        &sp_type,
        &sp_tangent_x,
        &sp_tangent_y,
        tangential,
        to_f32(floor_centers),
        to_f32(wall_centers),
        to_f32(slide_centers),
        to_f32(launch_centers),
        spatial_cell_size,
        max_vel_bins,
    );

    let out = py.allow_threads(|| {
        let sw = SweepParams {
            run_durations: &run_durations,
            jump_holds: &jump_holds,
            h_inputs: &h_inputs,
            post_release: &post_release,
            max_arc_frames,
            max_jump_duration,
            tick_mines,
            goals: &goals,
        };
        enumerate_arcs_parallel(&map_data, &nodes, &sw, &mapper, n_threads, snap_qpos, snap_qvel)
    });

    let m = out.arcs.len();
    let mut src = Vec::with_capacity(m);
    let mut dst = Vec::with_capacity(m);
    let mut frames = Vec::with_capacity(m);
    let mut jump_hold = Vec::with_capacity(m);
    let mut h_during = Vec::with_capacity(m);
    let mut h_after = Vec::with_capacity(m);
    let mut clipped = Vec::with_capacity(m);
    for a in &out.arcs {
        src.push(a.src_node);
        dst.push(a.dst_node);
        frames.push(a.frames);
        jump_hold.push(a.jhold);
        h_during.push(a.h_during);
        h_after.push(a.h_after);
        clipped.push(a.clipped);
    }

    Ok(ArcBatch {
        src,
        dst,
        frames,
        jump_hold,
        h_during,
        h_after,
        clipped,
        gc_node: out.gc_node,
        gc_goal: out.gc_goal,
        gc_frame: out.gc_frame,
    })
}

/// Experimental: size the reachability-flood opportunity.  Runs the exact same
/// arc sweep as `enumerate_arcs` but, instead of emitting edges, counts total
/// arc ticks vs distinct quantized `(pre-state, input)` transitions.  Returns
/// `(total_ticks, distinct_transitions)`; `total/distinct` is the speedup a
/// transposition-merging flood could realize at this `(qpos, qvel)` grain.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn flood_stats(
    py: Python<'_>,
    map_data: Vec<i32>,
    node_x: Vec<f64>,
    node_y: Vec<f64>,
    node_vx0: Vec<f64>,
    node_vy0: Vec<f64>,
    node_surface: Vec<u8>,
    node_is_edge: Vec<bool>,
    node_pt: Vec<i32>,
    node_vb: Vec<i32>,
    run_durations: Vec<i32>,
    jump_holds: Vec<i32>,
    h_inputs: Vec<i32>,
    post_release: Vec<i32>,
    max_arc_frames: i32,
    max_jump_duration: i32,
    tick_mines: bool,
    goals_flat: Vec<f64>,
    sp_x: Vec<f64>,
    sp_y: Vec<f64>,
    sp_type: Vec<i32>,
    floor_centers: Vec<f64>,
    wall_centers: Vec<f64>,
    slide_centers: Vec<f64>,
    launch_centers: Vec<f64>,
    spatial_cell_size: f64,
    max_vel_bins: i32,
    n_threads: usize,
    qpos: f64,
    qvel: f64,
) -> PyResult<(u64, usize, u64, usize)> {
    use enumerate::{flood_stats_parallel, NodeMapper, NodeSpec, SweepParams};

    let n = node_x.len();
    let nodes: Vec<NodeSpec> = (0..n)
        .map(|i| NodeSpec {
            src_node: node_pt[i] * max_vel_bins + node_vb[i],
            x: node_x[i],
            y: node_y[i],
            vx0: node_vx0[i],
            vy0: node_vy0[i],
            surface_type: node_surface[i],
            is_edge: node_is_edge[i],
        })
        .collect();
    let goals: Vec<(f64, f64, f64)> = goals_flat
        .chunks_exact(3)
        .map(|c| (c[0], c[1], c[2]))
        .collect();
    let to_f32 = |v: Vec<f64>| -> Vec<f32> { v.into_iter().map(|c| c as f32).collect() };
    // flood_stats measures collision redundancy only; the tangential landing
    // projection is irrelevant here, so pass dummy tangents + tangential=false.
    let sp_tan_dummy = vec![0.0f64; sp_x.len()];
    let mapper = NodeMapper::new(
        &sp_x, &sp_y, &sp_type,
        &sp_tan_dummy, &sp_tan_dummy, false,
        to_f32(floor_centers), to_f32(wall_centers),
        to_f32(slide_centers), to_f32(launch_centers),
        spatial_cell_size, max_vel_bins,
    );

    let out = py.allow_threads(|| {
        let sw = SweepParams {
            run_durations: &run_durations,
            jump_holds: &jump_holds,
            h_inputs: &h_inputs,
            post_release: &post_release,
            max_arc_frames,
            max_jump_duration,
            tick_mines,
            goals: &goals,
        };
        flood_stats_parallel(&map_data, &nodes, &sw, &mapper, n_threads, qpos, qvel)
    });
    Ok(out)
}

/// Constants submodule for Python.
fn register_constants_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "constants")?;

    // Physics constants
    m.add("NINJA_RADIUS", constants::NINJA_RADIUS)?;
    m.add("GRAVITY_FALL", constants::GRAVITY_FALL)?;
    m.add("GRAVITY_JUMP", constants::GRAVITY_JUMP)?;
    m.add("GROUND_ACCEL", constants::GROUND_ACCEL)?;
    m.add("AIR_ACCEL", constants::AIR_ACCEL)?;
    m.add("DRAG_REGULAR", constants::DRAG_REGULAR)?;
    m.add("DRAG_SLOW", constants::DRAG_SLOW)?;
    m.add("FRICTION_GROUND", constants::FRICTION_GROUND)?;
    m.add("FRICTION_GROUND_SLOW", constants::FRICTION_GROUND_SLOW)?;
    m.add("FRICTION_WALL", constants::FRICTION_WALL)?;
    m.add("MAX_HOR_SPEED", constants::MAX_HOR_SPEED)?;
    m.add("MAX_VER_SPEED", constants::MAX_VER_SPEED)?;
    m.add("MAX_JUMP_DURATION", constants::MAX_JUMP_DURATION)?;
    m.add("MAX_SURVIVABLE_IMPACT", constants::MAX_SURVIVABLE_IMPACT)?;
    m.add("MIN_SURVIVABLE_CRUSHING", constants::MIN_SURVIVABLE_CRUSHING)?;
    m.add("TERMINAL_VELOCITY", constants::TERMINAL_VELOCITY)?;

    // Jump constants
    m.add("JUMP_FLOOR_Y", constants::JUMP_FLOOR_Y)?;
    m.add(
        "JUMP_SLOPE_DOWNHILL_X_MULTIPLIER",
        constants::JUMP_SLOPE_DOWNHILL_X_MULTIPLIER,
    )?;
    m.add(
        "JUMP_SLOPE_DOWNHILL_Y_MULTIPLIER",
        constants::JUMP_SLOPE_DOWNHILL_Y_MULTIPLIER,
    )?;
    m.add(
        "JUMP_SLOPE_UPHILL_DEFAULT_Y",
        constants::JUMP_SLOPE_UPHILL_DEFAULT_Y,
    )?;
    m.add(
        "JUMP_WALL_REGULAR_X_MULTIPLIER",
        constants::JUMP_WALL_REGULAR_X_MULTIPLIER,
    )?;
    m.add("JUMP_WALL_REGULAR_Y", constants::JUMP_WALL_REGULAR_Y)?;
    m.add(
        "JUMP_WALL_SLIDE_X_MULTIPLIER",
        constants::JUMP_WALL_SLIDE_X_MULTIPLIER,
    )?;
    m.add("JUMP_WALL_SLIDE_Y", constants::JUMP_WALL_SLIDE_Y)?;

    // Entity constants
    m.add("EXIT_SWITCH_RADIUS", constants::EXIT_SWITCH_RADIUS)?;
    m.add("EXIT_DOOR_RADIUS", constants::EXIT_DOOR_RADIUS)?;
    m.add("LOCKED_DOOR_RADIUS", constants::LOCKED_DOOR_RADIUS)?;
    m.add(
        "LOCKED_DOOR_SWITCH_RADIUS",
        constants::LOCKED_DOOR_SWITCH_RADIUS,
    )?;
    m.add(
        "TOGGLE_MINE_RADIUS_TOGGLED",
        constants::TOGGLE_MINE_RADIUS_TOGGLED,
    )?;
    m.add(
        "TOGGLE_MINE_RADIUS_UNTOGGLED",
        constants::TOGGLE_MINE_RADIUS_UNTOGGLED,
    )?;
    m.add(
        "TOGGLE_MINE_RADIUS_TOGGLING",
        constants::TOGGLE_MINE_RADIUS_TOGGLING,
    )?;

    // Map constants
    m.add("MAP_WIDTH", constants::MAP_WIDTH)?;
    m.add("MAP_HEIGHT", constants::MAP_HEIGHT)?;
    m.add("TILE_SIZE", constants::TILE_SIZE)?;

    parent.add_submodule(&m)?;
    Ok(())
}

/// The main Python module.
#[pymodule]
fn nclone_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySimulator>()?;
    m.add_class::<NinjaSnapshot>()?;
    m.add_class::<EntitySnapshot>()?;
    m.add_class::<SimulatorState>()?;
    m.add_class::<vecenv::NativeVecEnv>()?;
    m.add_class::<level_cache::LevelCache>()?;
    m.add_class::<mine_sdf::MineThreatMap>()?;
    m.add_class::<ArcBatch>()?;
    m.add_function(wrap_pyfunction!(enumerate_arcs, m)?)?;
    m.add_function(wrap_pyfunction!(flood_stats, m)?)?;
    register_constants_module(m)?;
    Ok(())
}
