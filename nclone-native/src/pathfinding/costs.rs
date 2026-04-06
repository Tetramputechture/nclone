//! Cost functions for physics-aware pathfinding.
//!
//! Edge weight computation, aerial chain multipliers, and grounding utilities.
//! Ported from nclone/graph/reachability/cost_functions.py.

// Aerial upward movement cost constants
// N++ physics: A jump from ground with held jump button can realistically cover
// ~5-6 sub-nodes (60-72px) of upward travel before gravity overcomes jump velocity.
pub const AERIAL_UPWARD_CHAIN_THRESHOLD: i32 = 2;
pub const AERIAL_UPWARD_BLOCKING_COST: f64 = 500.0;
pub const AERIAL_UPWARD_BASE_MULTIPLIER: f64 = 3.0;

// Momentum-aware pathfinding cost constants
pub const MOMENTUM_CONTINUE_MULTIPLIER: f64 = 0.7;
pub const MOMENTUM_REVERSE_MULTIPLIER: f64 = 2.5;
pub const MOMENTUM_BUILDING_THRESHOLD: i32 = 6;

// Sub-node spacing in pixels
pub const SUB_NODE_SIZE: i32 = 6;

// Aerial descent chain constants
pub const AERIAL_DESCENT_MILD_THRESHOLD: i32 = 5;
pub const AERIAL_DESCENT_MODERATE_THRESHOLD: i32 = 15;
pub const AERIAL_DESCENT_HARD_LIMIT: i32 = 28;

// Cost cap for impossible edges when allow_high_cost_edges is enabled
pub const IMPOSSIBLE_EDGE_COST_CAP: f64 = 10000.0;

/// Multiplicative cost scaling for consecutive aerial upward moves.
///
/// Chain 0-5: Progressive cost scaling (within realistic jump physics)
/// Chain 6+: Beyond physics limits - apply blocking cost
pub fn get_aerial_chain_multiplier(chain_count: i32) -> f64 {
    if chain_count <= AERIAL_UPWARD_CHAIN_THRESHOLD {
        AERIAL_UPWARD_BASE_MULTIPLIER.powi(chain_count + 1)
    } else {
        let excess_chain = chain_count - AERIAL_UPWARD_CHAIN_THRESHOLD;
        AERIAL_UPWARD_BLOCKING_COST * AERIAL_UPWARD_BASE_MULTIPLIER.powi(excess_chain)
    }
}

/// Cost multiplier for consecutive airborne edges (any direction).
///
/// Models drag-limited horizontal range during sustained aerial traversal.
pub fn get_aerial_descent_chain_multiplier(chain_count: i32) -> f64 {
    if chain_count <= AERIAL_DESCENT_MILD_THRESHOLD {
        1.0
    } else if chain_count <= AERIAL_DESCENT_MODERATE_THRESHOLD {
        let t = (chain_count - AERIAL_DESCENT_MILD_THRESHOLD) as f64
            / (AERIAL_DESCENT_MODERATE_THRESHOLD - AERIAL_DESCENT_MILD_THRESHOLD) as f64;
        1.0 + 2.0 * t
    } else if chain_count <= AERIAL_DESCENT_HARD_LIMIT {
        let t = (chain_count - AERIAL_DESCENT_MODERATE_THRESHOLD) as f64
            / (AERIAL_DESCENT_HARD_LIMIT - AERIAL_DESCENT_MODERATE_THRESHOLD) as f64;
        3.0 + 17.0 * t
    } else {
        let excess = chain_count - AERIAL_DESCENT_HARD_LIMIT;
        50.0 * 2.0f64.powi(excess)
    }
}

/// Compute geometric edge cost between two nodes (pixels).
///
/// Cardinal moves: 6px. Diagonal moves: 6 * sqrt(2) ≈ 8.49px.
pub fn geometric_edge_cost(from: (i32, i32), to: (i32, i32)) -> f64 {
    let dx = (to.0 - from.0) as f64;
    let dy = (to.1 - from.1) as f64;
    if dx != 0.0 && dy != 0.0 {
        (dx * dx + dy * dy).sqrt()
    } else {
        6.0 // SUB_NODE_SIZE
    }
}
