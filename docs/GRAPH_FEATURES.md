# Graph Features Reference

Comprehensive guide to node and edge features in the N++ level graph representation.

## Overview

The graph representation captures the level structure, entities, and navigation constraints in a form optimized for Graph Neural Networks (GNNs). All features are designed to enable the agent to learn effective navigation strategies, especially for mine-heavy puzzles, without requiring expensive physics simulation.

## Node Features (21 dimensions)

Each node in the graph represents a reachable position in the level. Nodes are created at 12-pixel resolution with 2×2 sub-nodes per 24-pixel tile.

### Feature Index Reference

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | x_position | [0, 1] | Normalized x-coordinate in level |
| 1 | y_position | [0, 1] | Normalized y-coordinate in level |
| 2-8 | node_type | {0, 1} | One-hot: EMPTY, WALL, TOGGLE_MINE, LOCKED_DOOR, SPAWN, EXIT_SWITCH, EXIT_DOOR |
| 9 | is_mine | {0, 1} | Binary flag: 1 if node has a mine |
| 10 | mine_state | {-1, 0, +1} | -1=deadly (toggled), 0=transitioning, +1=safe (untoggled) |
| 11 | mine_radius | [0, 1] | Normalized collision radius of mine |
| 12 | entity_active | {0, 1} | For switches/doors: 1 if active |
| 13 | door_closed | {0, 1} | For locked doors: 1 if closed |
| 14 | reachable_from_ninja | {0, 1} | Flood-fill reachability: 1 if reachable |
| 15 | in_degree | [0, 1] | Normalized incoming edge count |
| 16 | out_degree | [0, 1] | Normalized outgoing edge count |
| 17 | objective_dx | [-1, +1] | Normalized x-distance to current objective |
| 18 | objective_dy | [-1, +1] | Normalized y-distance to current objective |
| 19 | objective_hops | [0, 1] | Normalized graph hops to objective (BFS distance) |
| 20 | betweenness | [0, 1] | Betweenness centrality (approximated via sampling) |

### Feature Groups Explained

#### Spatial Features (Indices 0-1)

Position encoding normalized to [0, 1] by level dimensions (1056×600 pixels).

```python
x_norm = x_position / 1056.0
y_norm = y_position / 600.0
```

**Design note**: Tile type encoding was removed (previously 34 dims) as position is sufficient for learning spatial patterns.

#### Node Type (Indices 2-8)

One-hot encoding of strategic node types:

- `EMPTY` (0): Traversable space
- `WALL` (1): Obstacle (rarely used as graph only includes reachable nodes)
- `TOGGLE_MINE` (2): Mine entity
- `LOCKED_DOOR` (3): Locked door entity
- `SPAWN` (4): Player spawn point
- `EXIT_SWITCH` (5): Exit switch (must be collected)
- `EXIT_DOOR` (6): Exit door (level completion)

#### Mine-Specific Features (Indices 9-11)

**Critical for mine navigation puzzles.** Explicit encoding of mine state enables the agent to distinguish safe vs deadly mines and plan accordingly.

- `is_mine`: Clear binary flag (redundant with type one-hot but improves feature interpretability)
- `mine_state`: 
  - `-1.0`: Deadly (toggled state, red in game)
  - `0.0`: Transitioning (animated, changing state)
  - `+1.0`: Safe (untoggled state, gold in game)
- `mine_radius`: Normalized collision radius (typically ~4-8 pixels normalized by ninja radius×2)

**Example**: For a node with a deadly mine at position (100, 200) with radius 6:
```python
features[9] = 1.0   # is_mine
features[10] = -1.0  # deadly
features[11] = 6.0 / (2 * NINJA_RADIUS)  # normalized radius
```

#### Entity State Features (Indices 12-13)

General entity state for switches and doors:

- `entity_active`: Used for switches (1 if switch is active/uncollected) and doors (1 if switch collected but door not yet opened)
- `door_closed`: Specific to locked doors (1 if door is closed, 0 if open)

#### Reachability (Index 14)

Binary flag from flood-fill reachability analysis. Indicates whether the node is reachable from the ninja's current position given current game state (switch states, mine positions, etc.).

**Note**: This is connectivity-based, not physics-based. The agent learns action requirements from experience.

#### Topological Features (Indices 15-20)

Graph-theoretic features that help the agent understand node importance and navigation structure.

**Degree Centrality (15-16)**:
- `in_degree`: How many nodes can reach this node
- `out_degree`: How many nodes this node can reach
- High degree → junction/hub node, low degree → dead-end or bottleneck

**Objective-Relative (17-19)**:
- `objective_dx/dy`: Geometric distance to current objective (exit switch or exit door)
- `objective_hops`: Graph distance (BFS shortest path length)
- Helps agent understand progress toward goal

**Betweenness Centrality (20)**:
- Measures how often a node lies on shortest paths between other nodes
- High betweenness → critical junction or bottleneck
- Computed via sampling (100 random sources) for performance

## Edge Features (14 dimensions)

Each edge represents a potential movement between two nodes. Edges capture connectivity, direction, and danger (especially mine proximity).

### Feature Index Reference

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0-3 | edge_type | {0, 1} | One-hot: ADJACENT, REACHABLE, FUNCTIONAL, BLOCKED |
| 4 | weight | [0, 1] | Traversal weight (distance-based) |
| 5 | reachability_confidence | [0, 1] | Confidence from flood-fill system |
| 6 | dx_norm | [-1, +1] | Normalized x-direction of movement |
| 7 | dy_norm | [-1, +1] | Normalized y-direction of movement |
| 8 | distance | [0, 1] | Normalized Euclidean distance |
| 9 | movement_category | [0, 1] | 0=horizontal, 0.33=mixed, 0.66=upward, 1.0=downward |
| 10 | nearest_mine_distance | [0, 1] | Distance from edge to nearest mine |
| 11 | passes_deadly_mine | {0, 1} | 1 if edge passes through deadly mine radius |
| 12 | mine_threat_level | [0, 1] | Aggregate danger score from nearby mines |
| 13 | num_mines_nearby | [0, 1] | Count of mines near edge (normalized) |

### Feature Groups Explained

#### Edge Type (Indices 0-3)

One-hot encoding of connectivity type:

- `ADJACENT` (0): Direct neighbor in grid (12-pixel spacing)
- `REACHABLE` (1): Reachable via movement (may require jump/fall)
- `FUNCTIONAL` (2): Entity relationship (e.g., switch→door connection)
- `BLOCKED` (3): Currently blocked (e.g., locked door without key)

#### Connectivity (Indices 4-5)

- `weight`: Graph traversal weight, typically Euclidean distance normalized
- `reachability_confidence`: From flood-fill system, indicates reliability of connection

#### Geometric Features (Indices 6-9)

**Enables direction-aware navigation.**

- `dx_norm, dy_norm`: Direction vector normalized by map dimensions
  ```python
  dx_norm = (target_x - source_x) / FULL_MAP_WIDTH_PX
  dy_norm = (target_y - source_y) / FULL_MAP_HEIGHT_PX
  ```
  
- `distance`: Euclidean distance normalized by map diagonal
  ```python
  distance_norm = euclidean_dist / sqrt(1056² + 600²)
  ```

- `movement_category`: Categorical encoding of movement type
  - `0.0`: Horizontal (|dx| > |dy| × 2)
  - `0.33`: Mixed/diagonal
  - `0.66`: Upward (|dy| > |dx| × 2 and dy < 0)
  - `1.0`: Downward (|dy| > |dx| × 2 and dy > 0)
  
  **Design rationale**: Downward edges are easier (can fall), upward edges are harder (require jump), helping agent learn physics-aware navigation.

#### Mine Danger Features (Indices 10-13)

**Critical for safe navigation through mine puzzles.**

All mine computations use vectorized geometric calculations for performance.

- `nearest_mine_distance`: Minimum distance from edge line segment to any mine center
  ```python
  # Uses point-to-line-segment distance for all mines
  # Normalized by map diagonal
  min_dist = min(point_to_line_distance(mine_pos, edge_src, edge_tgt))
  ```

- `passes_deadly_mine`: Binary flag indicating if edge passes through deadly mine radius
  ```python
  # Edge dangerous if: line_distance <= mine_radius AND mine_state == -1.0
  passes_deadly = any(line_dist <= mine.radius for deadly_mines)
  ```

- `mine_threat_level`: Aggregate danger score from nearby deadly mines
  ```python
  # Sum of (radius / distance) for mines within 2×radius of edge midpoint
  threat = sum(mine.radius / max(distance_to_midpoint, 1.0))
  threat_norm = min(threat / 5.0, 1.0)  # Normalize
  ```

- `num_mines_nearby`: Count of mines (any state) within 2×radius of edge midpoint
  ```python
  nearby = count(edge_midpoint_dist < mine.radius * 2.0)
  nearby_norm = min(nearby / 5.0, 1.0)  # Normalize
  ```

**Example**: Edge passing through deadly mine with radius 6:
```python
features[10] = 0.1   # Very close to mine
features[11] = 1.0   # Passes through deadly mine
features[12] = 0.8   # High threat (mine very close)
features[13] = 0.2   # 1 mine nearby (normalized)
```

## Graph Construction

### Node Generation

Nodes are created at 12-pixel resolution with 2×2 sub-nodes per 24-pixel tile:

```python
# For each 24×24 tile, create 4 sub-nodes
SUB_NODE_OFFSETS = [(6,6), (18,6), (6,18), (18,18)]
```

Only reachable positions are included (via flood-fill from spawn).

### Edge Generation

Edges are created using 4-connectivity (N, E, S, W):

```python
DIRECTIONS = {
    "N": (0, -12),   # North
    "E": (12, 0),    # East
    "S": (0, 12),    # South
    "W": (-12, 0),   # West
}
```

Additional edges for entity relationships (switch→door connections).

### Performance Optimizations

- **Vectorized computations**: All geometric and mine features use numpy broadcasting
- **Spatial hashing**: Mine proximity uses spatial hash for O(1) lookups
- **Caching**: Graph structure cached per level, only mine states updated per step
- **Sampling**: Betweenness centrality approximated via 100-sample random walk

**Target performance**: <200ms graph building per level (achieved).

## Usage in GNN Models

### Message Passing

Node and edge features flow through GNN layers:

```python
# Simplified message passing
for layer in gnn_layers:
    # Aggregate neighbor information
    messages = aggregate_neighbors(
        node_features,
        edge_features,
        edge_index,
    )
    # Update node representations
    node_features = layer(node_features, messages)
```

### Attention Mechanisms

Rich edge features enable attention-based aggregation:

```python
# Edge features used as attention keys
attention_weights = softmax(
    edge_features @ query_projection
)
```

### Multi-Head Processing

Different heads can specialize in different feature groups:

- **Spatial head**: Position, direction, distance
- **Entity head**: Node type, entity state, reachability
- **Mine head**: Mine state, danger features
- **Topological head**: Degree, centrality, objective distance

## Design Rationale

### Why No Physics Pre-computation?

Physics simulation is too expensive to run ahead of time for every edge. Instead:

- **Geometric features** provide directional hints (upward vs downward)
- **Agent learns** action requirements through RL experience
- **Graph structure** captures connectivity without specifying how to traverse

### Why Explicit Mine Features?

Mine navigation is a core challenge in N++. Explicit encoding:

- Makes mine state unambiguous (-1/0/+1 is clearer than learned embeddings)
- Enables efficient mine-aware path planning
- Separates deadly from safe mines for strategic decision-making

### Why Topological Features?

Graph structure alone doesn't indicate node importance:

- **Degree** identifies hubs and bottlenecks
- **Betweenness** highlights critical junctions
- **Objective distance** provides goal-directed bias

These features help the agent learn to prioritize important nodes.

## Future Extensions

Potential enhancements (not yet implemented):

- **Historical features**: Edge traversal counts, success rates
- **Temporal features**: Time since node visited
- **Dynamic objectives**: Multi-gold-collection levels (not yet supported)
- **Enemy entities**: If future levels include enemies

See also: [OBSERVATION_SPACE_README.md](../OBSERVATION_SPACE_README.md) for full observation space documentation.

