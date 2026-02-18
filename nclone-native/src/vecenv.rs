//! NativeVecEnv: in-process vectorized simulation.
//!
//! Runs N simulators and steps them all with a single Python call.
//! Uses sequential iteration by default because each physics tick is so fast
//! (~0.5μs) that rayon's per-call overhead (~1ms) makes parallelism slower
//! for typical batch sizes. Parallel stepping is available via step_parallel()
//! for future heavier-per-env workloads.

use pyo3::prelude::*;
use rayon::prelude::*;

use crate::entities::Entity;
use crate::simulator::Simulator;
use crate::types::SimulatorState;

/// A vectorized environment that runs N simulators with a single Python call.
#[pyclass(name = "NativeVecEnv")]
pub struct NativeVecEnv {
    sims: Vec<Simulator>,
}

#[pymethods]
impl NativeVecEnv {
    /// Create N simulators from the same map data.
    #[staticmethod]
    fn create(map_data: Vec<i32>, num_envs: usize) -> Self {
        let sims: Vec<Simulator> = (0..num_envs)
            .map(|_| Simulator::load(&map_data))
            .collect();
        NativeVecEnv { sims }
    }

    /// Get the number of environments.
    fn num_envs(&self) -> usize {
        self.sims.len()
    }

    /// Step all environments with the given actions.
    /// actions: list of (hor_input, jump_input) tuples, one per env.
    /// Returns list of SimulatorState snapshots.
    fn step(&mut self, actions: Vec<(i32, i32)>) -> Vec<SimulatorState> {
        assert_eq!(
            actions.len(),
            self.sims.len(),
            "actions length must match num_envs"
        );

        self.sims
            .iter_mut()
            .zip(actions.iter())
            .map(|(sim, &(hor, jump))| {
                sim.tick(hor, jump);
                sim.get_state()
            })
            .collect()
    }

    /// Step all environments WITHOUT returning state.
    /// Useful when state is only needed periodically.
    fn step_no_state(&mut self, actions: Vec<(i32, i32)>) {
        assert_eq!(
            actions.len(),
            self.sims.len(),
            "actions length must match num_envs"
        );

        for (sim, &(hor, jump)) in self.sims.iter_mut().zip(actions.iter()) {
            sim.tick(hor, jump);
        }
    }

    /// Step all environments using rayon for parallel execution.
    /// Only beneficial when per-env computation is heavy (e.g. many entities).
    /// For typical physics-only workloads, step() is faster due to lower overhead.
    fn step_parallel(&mut self, py: Python<'_>, actions: Vec<(i32, i32)>) -> Vec<SimulatorState> {
        assert_eq!(
            actions.len(),
            self.sims.len(),
            "actions length must match num_envs"
        );

        let sims = &mut self.sims;
        py.allow_threads(|| {
            sims.par_iter_mut()
                .zip(actions.par_iter())
                .map(|(sim, &(hor, jump))| {
                    sim.tick(hor, jump);
                    sim.get_state()
                })
                .collect()
        })
    }

    /// Step all environments (physics only, no entity AI).
    fn step_physics_only(&mut self, actions: Vec<(i32, i32)>) -> Vec<SimulatorState> {
        assert_eq!(
            actions.len(),
            self.sims.len(),
            "actions length must match num_envs"
        );

        self.sims
            .iter_mut()
            .zip(actions.iter())
            .map(|(sim, &(hor, jump))| {
                sim.tick_physics_only(hor, jump);
                sim.get_state()
            })
            .collect()
    }

    /// Fast reset all environments.
    fn reset_all(&mut self) {
        for sim in &mut self.sims {
            sim.fast_reset();
        }
    }

    /// Fast reset a specific environment by index.
    fn reset_env(&mut self, env_idx: usize) {
        if env_idx < self.sims.len() {
            self.sims[env_idx].fast_reset();
        }
    }

    /// Get states for all environments.
    fn get_states(&self) -> Vec<SimulatorState> {
        self.sims.iter().map(|sim| sim.get_state()).collect()
    }

    /// Get state for a specific environment.
    fn get_state(&self, env_idx: usize) -> SimulatorState {
        self.sims[env_idx].get_state()
    }

    /// Load a new map for a specific environment.
    fn load_env(&mut self, env_idx: usize, map_data: Vec<i32>) {
        if env_idx < self.sims.len() {
            self.sims[env_idx] = Simulator::load(&map_data);
        }
    }

    /// Load a new map for all environments (same map).
    fn load_all(&mut self, map_data: Vec<i32>) {
        self.sims = (0..self.sims.len())
            .map(|_| Simulator::load(&map_data))
            .collect();
    }

    /// Step a single environment (for auto-reset when done).
    fn step_env(&mut self, env_idx: usize, hor_input: i32, jump_input: i32) -> SimulatorState {
        self.sims[env_idx].tick(hor_input, jump_input);
        self.sims[env_idx].get_state()
    }

    /// Set entity position for a specific environment (for curriculum).
    fn set_entity_position(&mut self, env_idx: usize, entity_index: usize, x: f64, y: f64) {
        if env_idx < self.sims.len() {
            let sim = &mut self.sims[env_idx];
            if entity_index < sim.entities.len() {
                match &mut sim.entities[entity_index] {
                    Entity::ToggleMine(e) => {
                        e.xpos = x;
                        e.ypos = y;
                    }
                    Entity::ExitSwitch(e) => {
                        e.xpos = x;
                        e.ypos = y;
                    }
                    Entity::ExitDoor(e) => {
                        e.xpos = x;
                        e.ypos = y;
                    }
                }
                sim.rebuild_entity_grid();
            }
        }
    }
}
