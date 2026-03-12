//! Level-based path distance cache (Phase 4).
//!
//! Precomputes shortest-path distances from goal positions via Dijkstra over
//! the graph. Supports geometric costs with horizontal rule validation.
//! Used for O(1) path lookups in PBRS and reachability features.

use std::collections::HashMap;
use std::collections::BinaryHeap;
use std::collections::VecDeque;
use pyo3::prelude::*;

use crate::constants::{MAP_HEIGHT, MAP_WIDTH};

const PHI_GRID_WIDTH: usize = 88;
const PHI_GRID_HEIGHT: usize = 50;
const PHI_GRID_RESOLUTION: f64 = 12.0;
const NODE_WORLD_COORD_OFFSET: i32 = 24;

/// Level cache with precomputed geometric distances from goals.
#[pyclass]
pub struct LevelCache {
    /// Node positions (x, y) in tile data space, indexed by node_idx
    node_positions: Vec<(i32, i32)>,
    /// (node_idx, goal_idx) -> geometric distance
    distances: HashMap<(usize, usize), f64>,
    /// (node_idx, goal_idx) -> next_hop node_idx (toward goal)
    next_hop: HashMap<(usize, usize), usize>,
    /// (node_idx, goal_idx) -> multi_hop direction (dx, dy) normalized
    multi_hop_dir: HashMap<(usize, usize), (f64, f64)>,
    /// Goals: (goal_node_idx, goal_id)
    goal_info: Vec<(usize, String)>,
    /// Phi grid for switch (rm_state=0)
    phi_grid_switch: Vec<f32>,
    /// Phi grid for exit (rm_state=1)
    phi_grid_exit: Vec<f32>,
    max_dist_switch: f64,
    max_dist_exit: f64,
}

#[pymethods]
impl LevelCache {
    /// Build level cache from tile grid and goal positions (legacy).
    #[staticmethod]
    fn build(
        tile_grid: Vec<u8>,
        switch_x: f64,
        switch_y: f64,
        exit_x: f64,
        exit_y: f64,
    ) -> Self {
        let mut dist_to_switch = vec![f64::INFINITY; MAP_WIDTH * MAP_HEIGHT];
        let mut dist_to_exit = vec![f64::INFINITY; MAP_WIDTH * MAP_HEIGHT];
        let switch_cell = (
            (switch_x / 24.0).floor() as i32,
            (switch_y / 24.0).floor() as i32,
        );
        let exit_cell = ((exit_x / 24.0).floor() as i32, (exit_y / 24.0).floor() as i32);
        bfs_fill_legacy(&tile_grid, switch_cell, &mut dist_to_switch);
        bfs_fill_legacy(&tile_grid, exit_cell, &mut dist_to_exit);

        LevelCache {
            node_positions: vec![],
            distances: HashMap::new(),
            next_hop: HashMap::new(),
            multi_hop_dir: HashMap::new(),
            goal_info: vec![],
            phi_grid_switch: vec![-1.0; PHI_GRID_WIDTH * PHI_GRID_HEIGHT],
            phi_grid_exit: vec![-1.0; PHI_GRID_WIDTH * PHI_GRID_HEIGHT],
            max_dist_switch: 1.0,
            max_dist_exit: 1.0,
        }
    }

    /// Build level cache from serialized graph (nodes, edges, goals).
    ///
    /// Args:
    ///   node_positions: List of (x, y) in tile data space
    ///   edges: List of (from_idx, to_idx, geometric_cost)
    ///   physics_grounded: List of bool per node
    ///   goals: List of (x, y, goal_id) in tile data space
    #[staticmethod]
    fn build_from_graph(
        node_positions: Vec<(i32, i32)>,
        edges: Vec<(usize, usize, f64)>,
        physics_grounded: Vec<bool>,
        goals: Vec<(i32, i32, String)>,
    ) -> PyResult<Self> {
        if node_positions.is_empty() {
            return Err(py_err("node_positions cannot be empty"));
        }
        if goals.is_empty() {
            return Err(py_err("goals cannot be empty"));
        }

        let n_nodes = node_positions.len();
        let mut adjacency: Vec<Vec<(usize, f64)>> = vec![vec![]; n_nodes];
        for (from_idx, to_idx, cost) in edges {
            if from_idx < n_nodes && to_idx < n_nodes {
                adjacency[from_idx].push((to_idx, cost));
            }
        }

        let mut distances: HashMap<(usize, usize), f64> = HashMap::new();
        let mut next_hop: HashMap<(usize, usize), usize> = HashMap::new();
        let mut multi_hop_dir: HashMap<(usize, usize), (f64, f64)> = HashMap::new();
        let mut goal_info: Vec<(usize, String)> = Vec::new();

        for (goal_x, goal_y, goal_id) in goals {
            let Some(goal_idx) = find_closest_node(goal_x, goal_y, &node_positions) else {
                continue;
            };

            let (dists, parents) = dijkstra_from_goal(
                goal_idx,
                &adjacency,
                &node_positions,
                &physics_grounded,
            );

            let goal_idx_for_map = goal_info.len();
            goal_info.push((goal_idx, goal_id.clone()));

            for (node_idx, &dist) in dists.iter().enumerate() {
                if dist == f64::INFINITY {
                    continue;
                }
                distances.insert((node_idx, goal_idx_for_map), dist);
                if let Some(&parent_idx) = parents.get(&node_idx) {
                    next_hop.insert((node_idx, goal_idx_for_map), parent_idx);
                }
                let dir = compute_multi_hop_direction(
                    node_idx,
                    &parents,
                    goal_idx,
                    &node_positions,
                    4,
                );
                if let Some(d) = dir {
                    multi_hop_dir.insert((node_idx, goal_idx_for_map), d);
                }
            }
        }

        let mut phi_switch = vec![-1.0_f32; PHI_GRID_WIDTH * PHI_GRID_HEIGHT];
        let mut phi_exit = vec![-1.0_f32; PHI_GRID_WIDTH * PHI_GRID_HEIGHT];
        let mut max_switch = 1.0_f64;
        let mut max_exit = 1.0_f64;

        for (idx, (_goal_idx, goal_id)) in goal_info.iter().enumerate() {
            let mut max_d = 0.0_f64;
            for node_idx in 0..n_nodes {
                if let Some(&d) = distances.get(&(node_idx, idx)) {
                    if d != f64::INFINITY {
                        max_d = max_d.max(d);
                    }
                }
            }
            if max_d < 1.0 {
                max_d = 1.0;
            }

            for node_idx in 0..n_nodes {
                if let Some(&d) = distances.get(&(node_idx, idx)) {
                    if d == f64::INFINITY {
                        continue;
                    }
                    let (nx, ny) = node_positions[node_idx];
                    let wx = nx + NODE_WORLD_COORD_OFFSET;
                    let wy = ny + NODE_WORLD_COORD_OFFSET;
                    let ix = (wx as f64 / PHI_GRID_RESOLUTION) as usize;
                    let iy = (wy as f64 / PHI_GRID_RESOLUTION) as usize;
                    if ix < PHI_GRID_WIDTH && iy < PHI_GRID_HEIGHT {
                        let phi = (-d / max_d) as f32;
                        if goal_id == "switch" || goal_id.starts_with("exit_switch_") {
                            phi_switch[iy * PHI_GRID_WIDTH + ix] = phi;
                            max_switch = max_switch.max(d);
                        } else if goal_id == "exit" || goal_id.starts_with("exit_door_") {
                            phi_exit[iy * PHI_GRID_WIDTH + ix] = phi;
                            max_exit = max_exit.max(d);
                        }
                    }
                }
            }
        }

        Ok(LevelCache {
            node_positions,
            distances,
            next_hop,
            multi_hop_dir,
            goal_info,
            phi_grid_switch: phi_switch,
            phi_grid_exit: phi_exit,
            max_dist_switch: max_switch,
            max_dist_exit: max_exit,
        })
    }

    fn get_distance(&self, x: f64, y: f64, goal_id: &str) -> f64 {
        let node_idx = self.world_to_nearest_node(x, y);
        let Some(ni) = node_idx else {
            return f64::INFINITY;
        };
        for (g_idx, (_, gid)) in self.goal_info.iter().enumerate() {
            if goal_matches(gid, goal_id) {
                return *self.distances.get(&(ni, g_idx)).unwrap_or(&f64::INFINITY);
            }
        }
        f64::INFINITY
    }

    fn get_phi(&self, x: f64, y: f64, rm_state: u8) -> f64 {
        if rm_state >= 2 {
            return 0.0;
        }
        let gx = x / PHI_GRID_RESOLUTION;
        let gy = y / PHI_GRID_RESOLUTION;
        let max_gx = (PHI_GRID_WIDTH - 1) as f64;
        let max_gy = (PHI_GRID_HEIGHT - 1) as f64;
        let gx = gx.clamp(0.0, max_gx);
        let gy = gy.clamp(0.0, max_gy);

        let grid = if rm_state == 0 {
            &self.phi_grid_switch
        } else {
            &self.phi_grid_exit
        };

        let ix0 = gx.floor() as usize;
        let iy0 = gy.floor() as usize;
        let ix1 = (ix0 + 1).min(PHI_GRID_WIDTH - 1);
        let iy1 = (iy0 + 1).min(PHI_GRID_HEIGHT - 1);
        let fx = gx - ix0 as f64;
        let fy = gy - iy0 as f64;

        let v00 = grid[iy0 * PHI_GRID_WIDTH + ix0];
        let v10 = grid[iy0 * PHI_GRID_WIDTH + ix1];
        let v01 = grid[iy1 * PHI_GRID_WIDTH + ix0];
        let v11 = grid[iy1 * PHI_GRID_WIDTH + ix1];

        let phi = v00 * (1.0 - fx) as f32 * (1.0 - fy) as f32
            + v10 * fx as f32 * (1.0 - fy) as f32
            + v01 * (1.0 - fx) as f32 * fy as f32
            + v11 * fx as f32 * fy as f32;

        phi as f64
    }

    fn get_next_hop(&self, x: f64, y: f64, goal_id: &str) -> Option<(f64, f64)> {
        let node_idx = self.world_to_nearest_node(x, y)?;
        for (g_idx, (_, gid)) in self.goal_info.iter().enumerate() {
            if goal_matches(gid, goal_id) {
                let nh_idx = *self.next_hop.get(&(node_idx, g_idx))?;
                let (nx, ny) = self.node_positions[node_idx];
                let (nhx, nhy) = self.node_positions[nh_idx];
                let dx = (nhx - nx) as f64;
                let dy = (nhy - ny) as f64;
                let mag = (dx * dx + dy * dy).sqrt();
                if mag < 1e-9 {
                    return None;
                }
                return Some((dx / mag, dy / mag));
            }
        }
        None
    }

    fn get_multi_hop_direction(&self, x: f64, y: f64, goal_id: &str) -> Option<(f64, f64)> {
        let node_idx = self.world_to_nearest_node(x, y)?;
        for (g_idx, (_, gid)) in self.goal_info.iter().enumerate() {
            if goal_matches(gid, goal_id) {
                return self.multi_hop_dir.get(&(node_idx, g_idx)).copied();
            }
        }
        None
    }
}

fn goal_matches(stored: &str, query: &str) -> bool {
    stored == query
        || (query == "switch" && stored.starts_with("exit_switch_"))
        || (query == "exit" && stored.starts_with("exit_door_"))
}

impl LevelCache {
    fn world_to_nearest_node(&self, x: f64, y: f64) -> Option<usize> {
        let tx = (x - NODE_WORLD_COORD_OFFSET as f64) as i32;
        let ty = (y - NODE_WORLD_COORD_OFFSET as f64) as i32;
        let mut best_idx = None;
        let mut best_d2 = i32::MAX;
        for (i, &(nx, ny)) in self.node_positions.iter().enumerate() {
            let dx = tx - nx;
            let dy = ty - ny;
            let d2 = dx * dx + dy * dy;
            if d2 < best_d2 {
                best_d2 = d2;
                best_idx = Some(i);
            }
        }
        best_idx
    }
}

fn find_closest_node(goal_x: i32, goal_y: i32, nodes: &[(i32, i32)]) -> Option<usize> {
    let mut best = None;
    let mut best_d2 = i64::MAX;
    for (i, &(nx, ny)) in nodes.iter().enumerate() {
        let dx = (goal_x - nx) as i64;
        let dy = (goal_y - ny) as i64;
        let d2 = dx * dx + dy * dy;
        if d2 < best_d2 {
            best_d2 = d2;
            best = Some(i);
        }
    }
    best
}

fn violates_horizontal_rule(
    current: usize,
    neighbor: usize,
    parents: &HashMap<usize, usize>,
    node_positions: &[(i32, i32)],
    physics_grounded: &[bool],
) -> bool {
    let (cx, cy) = node_positions[current];
    let (nx, ny) = node_positions[neighbor];
    if cy != ny || cx == nx {
        return false;
    }
    if physics_grounded[current] && physics_grounded[neighbor] {
        return false;
    }
    let Some(&parent_idx) = parents.get(&current) else {
        return false;
    };
    let (px, py) = node_positions[parent_idx];
    py == cy && px != cx
}

fn dijkstra_from_goal(
    goal_idx: usize,
    adjacency: &[Vec<(usize, f64)>],
    node_positions: &[(i32, i32)],
    physics_grounded: &[bool],
) -> (Vec<f64>, HashMap<usize, usize>) {
    let n = adjacency.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut parents: HashMap<usize, usize> = HashMap::new();
    dist[goal_idx] = 0.0;

    #[derive(Clone, Copy, PartialEq)]
    struct State(f64, usize);
    impl Eq for State {}
    impl PartialOrd for State {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            o.0.partial_cmp(&self.0)
        }
    }
    impl Ord for State {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            self.partial_cmp(o).unwrap_or(std::cmp::Ordering::Equal)
        }
    }

    let mut heap = BinaryHeap::new();
    heap.push(State(0.0, goal_idx));

    while let Some(State(d, u)) = heap.pop() {
        if d > dist[u] {
            continue;
        }
        for &(v, cost) in &adjacency[u] {
            if violates_horizontal_rule(v, u, &parents, node_positions, physics_grounded) {
                continue;
            }
            let new_d = dist[u] + cost;
            if new_d < dist[v] {
                dist[v] = new_d;
                parents.insert(v, u);
                heap.push(State(new_d, v));
            }
        }
    }

    (dist, parents)
}

fn compute_multi_hop_direction(
    start: usize,
    parents: &HashMap<usize, usize>,
    goal: usize,
    node_positions: &[(i32, i32)],
    max_hops: usize,
) -> Option<(f64, f64)> {
    let weights = [0.45, 0.25, 0.15, 0.08];
    let mut total_dx = 0.0;
    let mut total_dy = 0.0;
    let mut current = start;
    for &w in weights.iter().take(max_hops) {
        let parent_idx = *parents.get(&current)?;
        let (cx, cy) = node_positions[current];
        let (px, py) = node_positions[parent_idx];
        total_dx += w * (px - cx) as f64;
        total_dy += w * (py - cy) as f64;
        current = parent_idx;
        if current == goal {
            break;
        }
    }
    let mag = (total_dx * total_dx + total_dy * total_dy).sqrt();
    if mag < 1e-9 {
        return None;
    }
    Some((total_dx / mag, total_dy / mag))
}

fn py_err(msg: impl Into<String>) -> pyo3::PyErr {
    pyo3::exceptions::PyValueError::new_err(msg.into())
}

fn cell_to_idx(cx: i32, cy: i32) -> Option<usize> {
    if cx < 0 || cx >= MAP_WIDTH as i32 || cy < 0 || cy >= MAP_HEIGHT as i32 {
        return None;
    }
    Some(cy as usize * MAP_WIDTH + cx as usize)
}

fn bfs_fill_legacy(tile_grid: &[u8], start: (i32, i32), dist: &mut [f64]) {
    fn is_walkable(tile_grid: &[u8], cx: i32, cy: i32) -> bool {
        if let Some(i) = cell_to_idx(cx, cy) {
            let t = tile_grid[i];
            t == 0 || (t >= 2 && t <= 33)
        } else {
            false
        }
    }
    if !is_walkable(tile_grid, start.0, start.1) {
        return;
    }
    let start_idx = cell_to_idx(start.0, start.1).unwrap();
    dist[start_idx] = 0.0;
    let mut queue = VecDeque::new();
    queue.push_back(start);
    const DX: [i32; 4] = [1, -1, 0, 0];
    const DY: [i32; 4] = [0, 0, 1, -1];

    while let Some((cx, cy)) = queue.pop_front() {
        let curr_idx = cell_to_idx(cx, cy).unwrap();
        let curr_dist = dist[curr_idx];
        for i in 0..4 {
            let nx = cx + DX[i];
            let ny = cy + DY[i];
            if !is_walkable(tile_grid, nx, ny) {
                continue;
            }
            if let Some(next_idx) = cell_to_idx(nx, ny) {
                let new_dist = curr_dist + 24.0;
                if new_dist < dist[next_idx] {
                    dist[next_idx] = new_dist;
                    queue.push_back((nx, ny));
                }
            }
        }
    }
}
