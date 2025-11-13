# Observation Space Reference

Multi-modal observation space for the nclone N++ environment.

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
    # Visual
    'player_frame': Box(0, 255, (84, 84, 1), np.uint8),
    'global_view': Box(0, 255, (176, 100, 1), np.uint8),
    
    # Game state
    'game_state': Box(-1, 1, (52,), np.float32),
    'reachability_features': Box(0, 1, (8,), np.float32),
    'entity_positions': Box(0, 1, (6,), np.float32),
    'switch_states': Box(0, 1, (25,), np.float32),
    
    # Graph
    'graph_node_feats': Box(-inf, inf, (max_nodes, 50), np.float32),
    'graph_edge_index': Box(0, max_nodes-1, (2, max_edges), np.int32),
    'graph_edge_feats': Box(0, 1, (max_edges, 6), np.float32),
    'graph_node_mask': Box(0, 1, (max_nodes,), np.int32),
    'graph_edge_mask': Box(0, 1, (max_edges,), np.int32),
})
```

## Visual Observations

### `player_frame`
`(84, 84, 1)` uint8, range [0, 255]

84×84 grayscale frame centered on ninja. Covers approximately 1/6 of full level.

### `global_view`
`(176, 100, 1)` uint8, range [0, 255]

176×100 grayscale frame showing full level at 1/6 resolution (downsampled from 1056×600).

## State Vectors

### `game_state`
`(70,)` float32, range [-1, 1] or [0, 1]

Combined state vector: ninja physics (29) + path-aware objectives (15) + mine features (8) + progress (3) + sequential goal (3) + mine death probabilities (6) + terminal velocity death probabilities (6).

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

**Indices 29-43: Path-Aware Objectives (15 features)**
- `[29]` Exit switch collected {0, 1}
- `[30:32]` Exit switch relative position (x, y) and path distance
- `[33:35]` Exit door relative position (x, y) and path distance
- `[36:43]` Nearest locked door: present, switch_collected, switch_rel (x, y), switch_path_dist, door_rel (x, y), door_path_dist

**Indices 44-51: Mine Features (8 features)**
- `[44:45]` Nearest mine relative position (x, y)
- `[46]` Mine state {-1, 0, 0.5, 1} (deadly/safe/unknown)
- `[47]` Path distance to nearest mine
- `[48]` Deadly mines nearby count (normalized)
- `[49]` Mine state certainty [0, 1] (based on distance to nearest mine)
- `[50]` Safe mines nearby count (normalized)
- `[51]` Mine avoidance difficulty [0, 1] (spatial complexity metric)

**Indices 52-54: Progress Features (3 features)**
- `[52]` Current objective type (normalized)
- `[53]` Objectives completed ratio [0, 1]
- `[54]` Total path distance remaining (normalized)

**Indices 55-57: Sequential Goal Features (3 features)**
- `[55]` Goal phase {-1, 0, 1} (pre-switch, post-switch, at-door)
- `[56]` Switch priority {0, 1}
- `[57]` Door priority {0, 1}

**Indices 58-63: Mine Death Probabilities (6 features)**

Privileged information from physics simulation - probability of mine collision for each action.

- `[58]` NOOP death probability [0.0-1.0]
- `[59]` LEFT death probability [0.0-1.0]
- `[60]` RIGHT death probability [0.0-1.0]
- `[61]` JUMP death probability [0.0-1.0]
- `[62]` JUMP+LEFT death probability [0.0-1.0]
- `[63]` JUMP+RIGHT death probability [0.0-1.0]

Computed via `MineDeathPredictor.calculate_death_probability()` using 3-tier hybrid approach (spatial filter → distance check → physics simulation). Defaults to 0.0 when mine predictor is unavailable.

**Indices 64-69: Terminal Velocity Death Probabilities (6 features)**

Privileged information from physics simulation - probability of terminal impact death for each action.

- `[64]` NOOP terminal death probability [0.0-1.0]
- `[65]` LEFT terminal death probability [0.0-1.0]
- `[66]` RIGHT terminal death probability [0.0-1.0]
- `[67]` JUMP terminal death probability [0.0-1.0]
- `[68]` JUMP+LEFT terminal death probability [0.0-1.0]
- `[69]` JUMP+RIGHT terminal death probability [0.0-1.0]

Computed via `TerminalVelocityPredictor` using reachability-optimized lookup table for fast impact trajectory prediction. Defaults to 0.0 when predictor is unavailable.

### `reachability_features`
`(8,)` float32, range [0, 1]

Graph-based path planning features using adjacency graph.

- `[0]` Area ratio (reachable / total)
- `[1]` Distance to next objective (normalized, inverted) - uses objective hierarchy
- `[2]` Exit distance (normalized, inverted)
- `[3]` Objective path quality (path distance / Euclidean ratio)
- `[4]` Deadly mines on optimal path (normalized count)
- `[5]` Connectivity score (edge density)
- `[6]` Next objective reachable {0, 1}
- `[7]` Full completion path exists {0, 1} (switch→door→exit)

### `entity_positions`
`(6,)` float32, range [0, 1]

Key entity positions normalized to [0, 1].

- `[0:2]` Ninja (x, y)
- `[2:4]` Switch (x, y)
- `[4:6]` Exit (x, y)

### `switch_states`
`(25,)` float32, range [0, 1]

Locked door system state (5 doors × 5 features).

Per-door features (5 each):
- `[0:2]` Switch position (x, y)
- `[2:4]` Door position (x, y)
- `[4]` Collected/open state {0, 1}

Padded with zeros for levels with <5 locked doors.

```python
# Door indexing
door_0 = switch_states[0:5]
door_1 = switch_states[5:10]
door_2 = switch_states[10:15]
```

## Graph Observations

### `graph_node_feats`
`(max_nodes, 50)` float32

50-dimensional node features:
- Spatial (3): position (x, y) + resolution level
- Type (7): one-hot node type (EMPTY, WALL, TOGGLE_MINE, LOCKED_DOOR, SPAWN, EXIT_SWITCH, EXIT_DOOR)
- Entity (5): entity-specific attributes
- Tile (34): tile type one-hot (0-33, glitched tiles 34-37 treated as 0)
- Reachability (1): connectivity from flood-fill

### `graph_edge_index`
`(2, max_edges)` int32

COO format edge list: `[[sources], [targets]]`

### `graph_edge_feats`
`(max_edges, 6)` float32

6-dimensional edge features (type, distance, connectivity).

### `graph_node_mask` / `graph_edge_mask`
Binary masks {0, 1} indicating valid nodes/edges (vs padding).
