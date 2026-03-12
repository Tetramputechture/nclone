//! nclone_native: Rust physics engine for N++ simulation.
//! Exposes Simulator, state types, and constants to Python via PyO3.

use crate::constants::{MAP_HEIGHT, MAP_WIDTH};
use pyo3::types::PyDict;

pub mod collision;
pub mod constants;
pub mod entities;
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
                entities::Entity::ExitSwitch(e) => {
                    e.xpos = x;
                    e.ypos = y;
                }
                entities::Entity::ExitDoor(e) => {
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
    m.add_class::<mine_sdf::MineSDF>()?;
    register_constants_module(m)?;
    Ok(())
}
