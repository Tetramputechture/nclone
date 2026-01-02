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
    'reachability_features': Box(-1, 1, (38,), np.float32),  # 13 base + 8 path + 1 difficulty + 3 curvature + 5 exit + 8 directional (Phases 1-5)
    'spatial_context': Box(-1, 1, (96,), np.float32),  # 64 local_tile_grid + 32 mine_overlay
    
    # Graph (GCN-optimized: spatial + entity features) - OPTIONAL with spatial_context
    'graph_node_feats': Box(-inf, inf, (max_nodes, 6), np.float32),  # Spatial(2) + Mine(2) + Entity(2)
    'graph_edge_index': Box(0, max_nodes-1, (2, max_edges), np.uint16),
    'graph_node_mask': Box(0, 1, (max_nodes,), np.uint8),
    'graph_edge_mask': Box(0, 1, (max_edges,), np.uint8),
})
```

## Configuration Flags

### `enable_graph_observations` (EnvironmentConfig)

Controls whether graph observation arrays are included in the observation space:

- **True (default)**: Full graph arrays included (`graph_node_feats`, `graph_edge_index`, `graph_node_mask`, `graph_edge_mask`)
- **False**: Graph arrays excluded from observation space (memory optimization)

**Important**: When `False`, the graph is still built internally for PBRS reward calculation. This flag only controls whether graph arrays appear in the observation space returned to the agent.

**Use case**: Set to `False` for architectures that don't use graph modality (e.g., `graph_free`) to save ~21KB per observation (~1.3GB for 64 envs × 1024 steps).

```python
from nclone.gym_environment.config import EnvironmentConfig

# Graph-free training (saves ~21KB per observation)
config = EnvironmentConfig.for_training(
    enable_graph_observations=False,  # Exclude graph arrays
)
```

The `EnvironmentFactory` in npp-rl automatically sets this flag based on `architecture_config.modalities.use_graph`.

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
`(30,)` float32, range [-1, 1]

Graph-based reachability features with path distances, direction vectors, and phase indicator.

**Base Features (4)**
- `[0]` Reachable area ratio [0, 1] (reachable / total graph nodes)
- `[1]` Distance to nearest switch [0, 1] (normalized, inverted)
- `[2]` Distance to exit [0, 1] (normalized, inverted)
- `[3]` Exit reachable flag {0, 1}

**Path Distances (2)** - raw normalized distances for learning
- `[4]` Path distance to switch [0, 1] (normalized)
- `[5]` Path distance to exit [0, 1] (normalized)

**Direction Vectors (4)** - unit vectors toward goals (Euclidean)
- `[6]` Direction to switch X [-1, 1]
- `[7]` Direction to switch Y [-1, 1]
- `[8]` Direction to exit X [-1, 1]
- `[9]` Direction to exit Y [-1, 1]

**Mine Context (2)**
- `[10]` Total mines normalized [0, 1] (count / 10 max)
- `[11]` Deadly mine ratio [0, 1] (deadly / total mines)

**Phase Indicator (1)**
- `[12]` Switch activated flag {0, 1} - **CRITICAL for Markov property**
  - Explicit indicator of which objective to pursue (switch vs exit)
  - Enables proper credit assignment for milestone reward
  - Ensures agent observes two-phase task structure clearly

**Path Direction to Current Goal (8)** - Phase 1.1
- `[13]` Next hop direction X [-1, 1] (optimal path to current goal)
- `[14]` Next hop direction Y [-1, 1]
- `[15]` Waypoint direction X [-1, 1] (toward active waypoint)
- `[16]` Waypoint direction Y [-1, 1]
- `[17]` Waypoint distance [0, 1] (normalized)
- `[18]` Path requires detour {0, 1} (binary flag if next_hop points away from goal)
- `[19]` Mine clearance direction X [-1, 1] (safe direction from SDF)
- `[20]` Mine clearance direction Y [-1, 1]

**Path Difficulty (1)** - Phase 3.3
- `[21]` Path difficulty ratio [0, 1] (physics_cost / geometric_distance, log-normalized)

**Path Curvature to Current Goal (3)** - Phase 3.4
- `[22]` Multi-hop direction X [-1, 1] (8-hop lookahead to current goal)
- `[23]` Multi-hop direction Y [-1, 1]
- `[24]` Path curvature [0, 1] (dot product of next_hop and multi_hop, 1.0=straight, 0.0=90° turn)

**Exit Lookahead (5)** - Phase 4 (Switch Transition Continuity)
- `[25]` Exit next hop direction X [-1, 1] (always computed, even pre-switch)
- `[26]` Exit next hop direction Y [-1, 1]
- `[27]` Exit multi-hop direction X [-1, 1] (8-hop lookahead to exit, always computed)
- `[28]` Exit multi-hop direction Y [-1, 1]
- `[29]` Near-switch transition indicator [0, 1] (1.0 at switch, 0.0 at 50px+ away)

**Directional Connectivity (8)** - Phase 5 (Blind Jump Verification)
- `[30-37]` Platform distance in 8 directions [0, 1] (E, NE, N, NW, W, SW, S, SE)
  - Distance to nearest grounded platform in each compass direction
  - Normalized: 0.0 = adjacent, 1.0 = >500px or unreachable
  - Solves blind jump problem: verifies landing platforms beyond spatial context visibility (192px)
  - Computed from graph using physics cache to identify grounded nodes only
  - Example: 15-tile gap (360px) → F30=0.72 confirms platform exists before blind jump

### `spatial_context`
`(96,)` float32, range [-1, 1]

Graph-free local spatial context for platforming decisions. **Alternative to full graph observation** with 99% memory reduction.

**Local Tile Grid (64)** - 8×8 tiles centered on ninja
- Simplified tile categories: Empty(0), Solid(1), Half(2), Slope(3), Curved(4)
- Normalized to [0, 0.25, 0.5, 0.75, 1.0]
- Captures local geometry for movement decisions

**Mine Overlay (32)** - 8 nearest mines
- Per mine (4 features): relative_x, relative_y, state, radius
- State: -1.0 (deadly), 0.0 (transitioning), 1.0 (safe)
- Radius: normalized collision radius [0, 1]

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

6-dimensional node features optimized for goal-directed navigation:

**Spatial (2)**:
- x_normalized: x / LEVEL_WIDTH_PX [0-1] - explicit position for distance estimation
- y_normalized: y / LEVEL_HEIGHT_PX [0-1] - enables goal-directed navigation

**Mine-Specific (2)**: 
- mine_state: -1.0 (deadly), 0.0 (transitioning), +1.0 (safe), 0.0 (non-mine)
- mine_radius: Normalized collision radius (0.0 for non-mines)

**Entity State (2)**:
- entity_active: For switches/doors (1.0=active, 0.0=inactive)
- door_closed: For locked doors (1.0=closed, 0.0=open)

**Removed for memory optimization:**
- Type one-hot encoding (7 dims): GCN learns types from features and structure
- Topological features (6 dims): Redundant with PBRS shortest paths
- Reachability (1 dim): All nodes in graph are reachable (flood fill filtered)

### `graph_edge_index`
`(2, max_edges)` uint16

COO format edge list: `[[sources], [targets]]`

All edges represent simple adjacency between reachable nodes. GCN doesn't use edge features or types.

### `graph_node_mask` / `graph_edge_mask`
`(max_nodes,)` / `(max_edges,)` uint8

Binary masks {0, 1} indicating valid nodes/edges (vs padding).

## Performance Characteristics

- **Graph building**: <200ms per level (target maintained)
- **Node features**: 6 dims (spatial + mine + entity) - optimized for navigation
- **Edge features**: None (GCN uses graph structure only)
- **Game state**: 41 dims (40 ninja physics + 1 time_remaining)
- **Reachability features**: 38 dims (extended with path directions, curvature, exit lookahead, and directional connectivity)
- **Spatial context**: 96 dims (graph-free alternative - 64 tile grid + 32 mine overlay)

### Memory Comparison

| Mode | Per-Observation | 64 envs × 1024 steps |
|------|-----------------|----------------------|
| Full Graph | ~162 KB | ~10.4 GB |
| Graph-Free (spatial_context) | ~0.5 KB | ~33 MB |
| **Savings** | **99.7%** | **99.7%** |

The `spatial_context` observation provides equivalent local geometry information while using 99.7% less memory than full graph observations. This enables larger rollout buffers and more parallel environments.

All computations are vectorized using numpy for performance. Observation space optimized for sparse batching in GCN encoder.
