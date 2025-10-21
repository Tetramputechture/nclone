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
    
    # State vectors
    'game_state': Box(-1, 1, (26,), np.float32),
    'reachability_features': Box(0, 1, (8,), np.float32),
    'entity_positions': Box(0, 1, (6,), np.float32),
    'switch_states': Box(0, 1, (25,), np.float32),
    
    # Graph (optional, config.graph.enable_graph_updates=True)
    'graph_node_feats': Box(-inf, inf, (max_nodes, 55), np.float32),
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
`(26,)` float32, range [-1, 1]

Ninja physics and movement state.

**Indices 0-7: Movement**
- `[0]` Velocity magnitude (normalized)
- `[1:3]` Velocity direction (unit vector)
- `[3:7]` Movement categories (binary flags)
- `[7]` Airborne status

**Indices 8-12: Input & Buffers**
- `[8]` Horizontal input {-1, 0, 1}
- `[9]` Jump input {-1, 1}
- `[10:13]` Buffer states (timing windows)

**Indices 13-18: Surface Contact**
- `[13:16]` Contact strength (floor/wall/ceiling)
- `[16]` Floor normal
- `[17]` Wall direction
- `[18]` Surface slope

**Indices 19-23: Physics**
- `[19:21]` Recent acceleration (x, y)
- `[21]` Nearest hazard distance
- `[22]` Nearest collectible distance
- `[23]` Entity cooldown

**Indices 24-25: Level Progress**
- `[24]` Switch progress {-1, 1}
- `[25]` Exit accessible {-1, 1}

### `reachability_features`
`(8,)` float32, range [0, 1]

Fast path planning features using OpenCV flood fill (<1ms).

- `[0]` Area ratio (reachable / total)
- `[1]` Switch distance (normalized)
- `[2]` Exit distance (normalized)
- `[3]` Reachable switches (count)
- `[4]` Reachable hazards (count)
- `[5]` Connectivity score
- `[6]` Exit reachable {0, 1}
- `[7]` Path to exit exists {0, 1}

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

## Graph Observations (Optional)

Enable with `config.graph.enable_graph_updates = True`.

### `graph_node_feats`
`(max_nodes, 55)` float32

55-dimensional node features:
- Spatial (2): position
- Type (10): one-hot node type
- Entity (15): entity-specific attributes
- Tile (20): surrounding context
- Reachability (5): connectivity metrics
- Proximity (3): distances to key entities

### `graph_edge_index`
`(2, max_edges)` int32

COO format edge list: `[[sources], [targets]]`

### `graph_edge_feats`
`(max_edges, 6)` float32

6-dimensional edge features (type, distance, connectivity).

### `graph_node_mask` / `graph_edge_mask`
Binary masks {0, 1} indicating valid nodes/edges (vs padding).

## Usage Examples

### Vision-based CNN Policy

```python
import torch
import torch.nn as nn

class CNNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Process player frame
        self.player_cnn = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        
        # Process global view
        self.global_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
        )
        
    def forward(self, obs):
        player_feat = self.player_cnn(obs['player_frame'])
        global_feat = self.global_cnn(obs['global_view'])
        return torch.cat([player_feat.flatten(1), global_feat.flatten(1)], dim=1)
```

### MLP Policy (Vision-Free)

```python
class MLPPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # State dim: 26 + 8 + 6 + 25 = 65
        self.mlp = nn.Sequential(
            nn.Linear(65, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        
    def forward(self, obs):
        state = torch.cat([
            obs['game_state'],
            obs['reachability_features'],
            obs['entity_positions'],
            obs['switch_states'],
        ], dim=-1)
        return self.mlp(state)
```

## Changelog

**v1.2.0 (2025-10-21)**
- Removed entity counts from `game_state` (30→26 features)
- Simplified `switch_states` (30→25 features, removed redundant door_open)
- Added validation tool: `tools/validate_observations.py`

**v1.0.0 (2025-10-01)**
- Initial multi-modal observation space
