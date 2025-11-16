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
    'game_state': Box(-1, 1, (29,), np.float32),  # Only ninja physics
    'reachability_features': Box(0, 1, (6,), np.float32),  # 4 base + 2 mine context
    
    # Graph (ENHANCED with mine navigation features)
    'graph_node_feats': Box(-inf, inf, (max_nodes, 21), np.float32),  # Added topological features
    'graph_edge_index': Box(0, max_nodes-1, (2, max_edges), np.int32),
    'graph_edge_feats': Box(-inf, inf, (max_edges, 14), np.float32),  # Added geometric + mine danger
    'graph_node_mask': Box(0, 1, (max_nodes,), np.int32),
    'graph_edge_mask': Box(0, 1, (max_edges,), np.int32),
    'graph_node_types': Box(0, 10, (max_nodes,), np.int32),
    'graph_edge_types': Box(0, 10, (max_edges,), np.int32),
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
`(29,)` float32, range [-1, 1] or [0, 1]

**Only contains ninja physics state** - all other features moved to graph.

**Indices 0-28: Ninja Physics State (29 features)**
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

### `reachability_features`
`(6,)` float32, range [0, 1]

Graph-based reachability features with mine context.

**Base Features (4)**
- `[0]` Reachable area ratio [0, 1] (reachable / total graph nodes)
- `[1]` Distance to nearest switch [0, 1] (normalized, inverted)
- `[2]` Distance to exit [0, 1] (normalized, inverted)
- `[3]` Exit reachable flag {0, 1}

**Mine Context (2)**
- `[4]` Total mines normalized [0, 1] (count / 256 max)
- `[5]` Deadly mine ratio [0, 1] (deadly / total mines)

## Graph Observations

See [GRAPH_FEATURES.md](docs/GRAPH_FEATURES.md) for detailed feature descriptions.

### `graph_node_feats`
`(max_nodes, 21)` float32

21-dimensional node features with mine-specific and topological information:

**Spatial (2)**: x, y position normalized

**Type (7)**: One-hot node type (EMPTY, WALL, TOGGLE_MINE, LOCKED_DOOR, SPAWN, EXIT_SWITCH, EXIT_DOOR)

**Mine-Specific (3)**: 
- is_mine: Binary flag
- mine_state: -1.0 (deadly), 0.0 (transitioning), +1.0 (safe)
- mine_radius: Normalized collision radius

**Entity State (2)**:
- entity_active: For switches/doors
- door_closed: For locked doors

**Reachability (1)**: Flood-fill reachability from ninja

**Topological (6)**:
- in_degree: Normalized in-degree
- out_degree: Normalized out-degree
- objective_dx: X-distance to objective
- objective_dy: Y-distance to objective
- objective_hops: Graph hops to objective
- betweenness: Centrality score

### `graph_edge_index`
`(2, max_edges)` int32

COO format edge list: `[[sources], [targets]]`

### `graph_edge_feats`
`(max_edges, 14)` float32

14-dimensional edge features with geometric and mine danger information:

**Edge Type (4)**: One-hot (ADJACENT, REACHABLE, FUNCTIONAL, BLOCKED)

**Connectivity (2)**:
- weight: Traversal weight
- reachability_confidence: From flood-fill system

**Geometric (4)**:
- dx_norm: Normalized x-direction [-1, 1]
- dy_norm: Normalized y-direction [-1, 1]
- distance: Normalized Euclidean distance [0, 1]
- movement_category: [0, 1] (0=horizontal, 0.33=mixed, 0.66=upward, 1.0=downward)

**Mine Danger (4)**:
- nearest_mine_distance: Distance to nearest mine [0, 1]
- passes_deadly_mine: Binary flag if edge passes through deadly mine
- mine_threat_level: Aggregate danger score [0, 1]
- num_mines_nearby: Count of mines near edge [0, 1]

### `graph_node_mask` / `graph_edge_mask`
Binary masks {0, 1} indicating valid nodes/edges (vs padding).

### `graph_node_types` / `graph_edge_types`
Integer type IDs for each node/edge.

## Performance Characteristics

- **Graph building**: <200ms per level (target maintained)
- **Node features**: 21 dims (down from 50) - faster processing
- **Edge features**: 14 dims (up from 6) - richer information
- **Game state**: 29 dims (down from 64) - minimal state MLP
- **Total observation**: Graph-centric with no redundancy

All geometric and mine danger computations are vectorized using numpy for performance.
