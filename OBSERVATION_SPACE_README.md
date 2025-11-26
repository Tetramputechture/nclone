# Observation Space Reference

Multi-modal observation space for the nclone N++ environment (optimized for graph-based RL).

## Validation

Run validation script to verify observations:

```bash
# Quick validation
python tools/validate_observations.py

# Detailed validation
python tools/validate_observations.py --episodes 5 --max-steps 200 --verbose
```

## Structure

```python
observation_space = SpacesDict({
    # Visual (NOT USED - disabled for graph-only training)
    'player_frame': Box(0, 255, (84, 84, 1), np.uint8),
    'global_view': Box(0, 255, (176, 100, 1), np.uint8),
    
    # Game state (OPTIMIZED)
    'game_state': Box(-1, 1, (41,), np.float32),  # 40 ninja physics + 1 time_remaining
    'reachability_features': Box(0, 1, (7,), np.float32),  # 4 base + 2 mine context + 1 phase
    
    # Graph (GCN-optimized: minimal features)
    'graph_node_feats': Box(-inf, inf, (max_nodes, 6), np.float32),  # GCN-optimized: 6 dims
    'graph_edge_index': Box(0, max_nodes-1, (2, max_edges), np.int32),
    'graph_node_mask': Box(0, 1, (max_nodes,), np.int32),
    'graph_edge_mask': Box(0, 1, (max_edges,), np.int32),
})
```

## Design Principles (Phases 1-6 Optimization)

### Removed Redundancy
- **Tile encoding**: Removed 34-dim one-hot from nodes (position is sufficient)
- **Path objectives**: Removed 15 dims from game_state (now in graph)
- **Mine features**: Removed 8 dims from game_state (now in graph nodes/edges)
- **Progress features**: Removed 3 dims from game_state (computed from graph)
- **Sequential goals**: Removed 3 dims from game_state (in graph structure)
- **Action death probabilities**: Removed 6 dims (no expensive physics precomputation)
- **Entity positions**: Removed separate 6-dim vector (in graph nodes)
- **Switch states**: Removed separate 25-dim vector (in graph nodes)

### Graph-Centric Design
All spatial, entity, and mine information consolidated into comprehensive graph representation with explicit mine state encoding and geometric features for navigation.

## Visual Observations (NOT USED)

Visual modalities are available but not used for training. The agent learns purely from graph structure and state vectors.

### `player_frame`
`(84, 84, 1)` uint8, range [0, 255]

84×84 grayscale frame centered on ninja.

### `global_view`
`(176, 100, 1)` uint8, range [0, 255]

176×100 grayscale frame showing full level at 1/6 resolution.

## State Vectors

### `game_state`
`(41,)` float32, range [-1, 1] or [0, 1]

**Only contains ninja physics state** - all other features moved to graph.

**Indices 0-40: Ninja Physics State (40 features) + Time Remaining (1 feature)**
- `[0]` Velocity magnitude (normalized)
- `[1:3]` Velocity direction (unit vector x, y)
- `[3:7]` Movement categories (ground/air/wall/special states)
- `[7]` Airborne status
- `[8]` Horizontal input {-1, 0, 1}
- `[9]` Jump input {-1, 1}
- `[10:13]` Buffer states (jump/floor/wall buffers, normalized timing windows)
- `[13:16]` Contact strength (floor/wall/ceiling)
- `[16]` Floor normal strength (magnitude)
- `[17]` Wall direction (-1 or 1)
- `[18]` Surface slope (floor normal y-component)
- `[19:21]` Recent acceleration (x, y)
- `[21]` Applied gravity (normalized between GRAVITY_JUMP and GRAVITY_FALL)
- `[22]` Jump duration (normalized by MAX_JUMP_DURATION)
- `[23]` Walled status (boolean to -1/1)
- `[24]` Floor normal x-component (full x-component)
- `[25:26]` Ceiling normal vector (x, y)
- `[27]` Applied drag (normalized between DRAG_SLOW and DRAG_REGULAR)
- `[28]` Applied friction (normalized between FRICTION_GROUND_SLOW and FRICTION_GROUND)
- `[29:39]` Enhanced physics features (kinetic energy, potential energy, force magnitude, etc.)
- `[40]` Time remaining [0, 1] (curriculum-aware dynamic truncation)

### `reachability_features`
`(7,)` float32, range [0, 1]

Graph-based reachability features with mine context and explicit phase indicator.

**Base Features (4)**
- `[0]` Reachable area ratio [0, 1] (reachable / total graph nodes)
- `[1]` Distance to nearest switch [0, 1] (normalized, inverted)
- `[2]` Distance to exit [0, 1] (normalized, inverted)
- `[3]` Exit reachable flag {0, 1}

**Mine Context (2)**
- `[4]` Total mines normalized [0, 1] (count / 256 max)
- `[5]` Deadly mine ratio [0, 1] (deadly / total mines)

**Phase Indicator (1)**
- `[6]` Switch activated flag {0, 1} - **CRITICAL for Markov property**
  - Explicit indicator of which objective to pursue (switch vs exit)
  - Enables proper credit assignment for +2.0 milestone reward
  - Ensures agent observes two-phase task structure clearly

### `time_remaining`
`()` float32 scalar, range [0, 1]

Curriculum-aware dynamic truncation time pressure feature - **CRITICAL for Markov property**.

- Normalized time remaining: `(curriculum_limit - current_frame) / curriculum_limit`
- **Dynamic truncation**: Base limit calculated per level based on complexity
  - Formula: `sqrt(surface_area) * BASE_TIME_PER_NODE * TRUNCATION_MULTIPLIER`
  - Ensures fair time allocation: small levels get ~600-800 frames, large levels get ~3000-4000 frames
- **Curriculum-aware multipliers**: Aligns with reward config phases for consistency
  - Early phase (no time penalty): 2.0x generous limit (exploration focus)
  - Mid phase (optional penalty): 1.5x moderate limit (balanced approach)
  - Late phase (full time penalty): 1.0x standard limit (efficiency focus)
- **Robust calculation**: Handles edge cases (division by zero, negative values)
- **Agent consistency**: Agent observes the SAME time horizon that determines actual truncation

This feature ensures agents understand the actual time constraints they will face, maintaining consistency between observation and environment behavior across curriculum phases.

## Graph Observations

See [GRAPH_FEATURES.md](docs/GRAPH_FEATURES.md) for detailed feature descriptions.

### `graph_node_feats`
`(max_nodes, 6)` float32

6-dimensional GCN-optimized node features:

**Spatial (2)**: x, y position normalized

**Mine-Specific (2)**: 
- mine_state: -1.0 (deadly), 0.0 (transitioning), +1.0 (safe), 0.0 (non-mine)
- mine_radius: Normalized collision radius (0.0 for non-mines)

**Entity State (2)**:
- entity_active: For switches/doors (1.0=active, 0.0=inactive)
- door_closed: For locked doors (1.0=closed, 0.0=open)

**Note**: Removed features for GCN optimization:
- Type one-hot encoding (7 dims): GCN learns types from features and structure
- Topological features (6 dims): Redundant with PBRS shortest paths
- Reachability (1 dim): All nodes in graph are reachable (flood fill filtered)

### `graph_edge_index`
`(2, max_edges)` int32

COO format edge list: `[[sources], [targets]]`

All edges represent simple adjacency between reachable nodes. GCN doesn't use edge features or types.

### `graph_node_mask` / `graph_edge_mask`
`(max_nodes,)` / `(max_edges,)` int32

Binary masks {0, 1} indicating valid nodes/edges (vs padding).

## Performance Characteristics

- **Graph building**: <200ms per level (target maintained)
- **Node features**: 6 dims (GCN-optimized) - minimal, efficient processing
- **Edge features**: None (GCN uses graph structure only)
- **Game state**: 41 dims (40 ninja physics + 1 time_remaining)
- **Total observation**: GCN-optimized with minimal redundancy
- **Memory savings**: 65% reduction from original 17-dim node features

All computations are vectorized using numpy for performance. Observation space optimized for sparse batching in GCN encoder.
