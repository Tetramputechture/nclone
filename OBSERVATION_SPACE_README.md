# N++ Observation Space Documentation

Complete reference for nclone's multi-modal observation space designed for Deep RL.

## Overview

The observation space provides complementary views of the game state optimized for different neural network architectures:

- **Visual modalities** (player_frame, global_view) → CNNs
- **State vectors** (game_state, reachability_features, entity_positions, switch_states) → MLPs
- **Graph structure** (node/edge features) → GNNs

All observations are synchronized and updated every step.

## Observation Space Definition

```python
from gymnasium.spaces import Dict as SpacesDict, Box

observation_space = SpacesDict({
    # === Visual Modalities ===
    'player_frame': Box(0, 255, (84, 84, 1), dtype=np.uint8),
    'global_view': Box(0, 255, (176, 100, 1), dtype=np.uint8),
    
    # === State Vectors ===
    'game_state': Box(-1, 1, (30,), dtype=np.float32),
    'reachability_features': Box(0, 1, (8,), dtype=np.float32),
    'entity_positions': Box(0, 1, (6,), dtype=np.float32),
    'switch_states': Box(0, 1, (25,), dtype=np.float32),  # 5 locked doors × 5 features
    
    # === Graph Modality (optional, enable with config.graph.enable_graph_updates=True) ===
    'graph_node_feats': Box(-np.inf, np.inf, (max_nodes, 55), dtype=np.float32),
    'graph_edge_index': Box(0, max_nodes-1, (2, max_edges), dtype=np.int32),
    'graph_edge_feats': Box(0, 1, (max_edges, 6), dtype=np.float32),
    'graph_node_mask': Box(0, 1, (max_nodes,), dtype=np.int32),
    'graph_edge_mask': Box(0, 1, (max_edges,), dtype=np.int32),
})
```

---

## Visual Modalities

### `player_frame` — Local Player View
**Shape:** `(84, 84, 1)` | **Type:** `uint8` | **Range:** `[0, 255]`

Player-centered view capturing local gameplay details.

**Properties:**
- 84×84 grayscale frame
- Centered on ninja position
- Covers ~1/6 of full level
- Updated every step
- Optimized for 2D CNNs

**Use cases:**
- Local obstacle detection
- Immediate threat awareness
- Fine-grained movement control
- Reaction to nearby entities

### `global_view` — Strategic Overview
**Shape:** `(176, 100, 1)` | **Type:** `uint8` | **Range:** `[0, 255]`

Downsampled full-level view for strategic planning.

**Properties:**
- 176×100 grayscale frame (1/6 resolution of 1056×600 level)
- Full level coverage
- Preserves spatial relationships
- Updated every step

**Use cases:**
- Path planning
- Goal localization
- Spatial reasoning
- Strategic decision-making

---

## State Vectors

### `game_state` — Physics & Entity State
**Shape:** `(30,)` | **Type:** `float32` | **Range:** `[-1, 1]`

Concatenated vector: `[ninja_state(26), entity_count(4)]`

#### Ninja State (indices 0-25)

**Core Movement (8 features):**
- `[0]` Velocity magnitude (normalized by max speed)
- `[1:3]` Velocity direction (x, y) — unit vector
- `[3:7]` Movement categories (ground/air/wall/special) — binary flags
- `[7]` Airborne status

**Input & Buffers (5 features):**
- `[8]` Horizontal input (-1=left, 0=none, 1=right)
- `[9]` Jump input (1=pressed, -1=not pressed)
- `[10:13]` Buffer states (jump/floor/wall) — timing windows

**Surface Contact (6 features):**
- `[13:16]` Contact strength (floor/wall/ceiling)
- `[16]` Floor normal strength
- `[17]` Wall direction
- `[18]` Surface slope

**Momentum & Physics (4 features):**
- `[19:21]` Recent acceleration (x, y)
- `[21]` Nearest hazard distance (computed by observation processor)
- `[22]` Nearest collectible distance (computed by observation processor)
- `[23]` Entity interaction cooldown

**Level Progress (2 features):**
- `[24]` Switch activation progress (1=activated, -1=not activated)
- `[25]` Exit accessibility (1=accessible, -1=locked)

#### Entity Counts (indices 26-29)

**WARNING:** These 4 values are extracted from `get_entity_states()[:4]` which contains:
- `[26]` Toggle mine count (normalized by max count)
- `[27:29]` First 3 attributes of first toggle mine (if exists, else 0)

**Note:** This is a legacy design artifact. For proper entity information, use `entity_states` or the graph modality.

---

### `reachability_features` — Path Planning
**Shape:** `(8,)` | **Type:** `float32` | **Range:** `[0, 1]`

Fast reachability analysis computed using OpenCV flood fill (<1ms).

**Features:**
- `[0]` **Area ratio** — Reachable area / total level area
- `[1]` **Switch distance** — Normalized distance to exit switch
- `[2]` **Exit distance** — Normalized distance to exit door
- `[3]` **Reachable switches** — Count of switches reachable from current position
- `[4]` **Reachable hazards** — Count of hazards in reachable area
- `[5]` **Connectivity score** — Path diversity metric (0=bottleneck, 1=open)
- `[6]` **Exit reachable** — Binary flag (1=can reach exit from here)
- `[7]` **Path to exit exists** — Binary flag (1=path exists to goal)

**Implementation details:**
- Uses collision-free space extraction
- OpenCV floodFill for reachability computation
- Cached and updated only on position change
- Normalized by level dimensions (1056×600)

**Use cases:**
- Path feasibility checking
- Exploration guidance
- Dead-end detection
- Goal reachability assessment

---

### `entity_positions` — Key Locations
**Shape:** `(6,)` | **Type:** `float32` | **Range:** `[0, 1]`

Normalized positions of critical entities.

**Layout:**
- `[0:2]` **Ninja position** (x, y) — Current player location
- `[2:4]` **Switch position** (x, y) — Exit activation switch
- `[4:6]` **Exit position** (x, y) — Level goal

**Normalization:**
- X coordinates normalized by `LEVEL_WIDTH` (1056)
- Y coordinates normalized by `LEVEL_HEIGHT` (600)

**Use cases:**
- Direct positional reasoning
- Distance calculations
- Relative positioning
- Goal-directed navigation

---

### `switch_states` — Locked Door System
**Shape:** `(25,)` | **Type:** `float32` | **Range:** `[0, 1]`

State of up to 5 locked doors and their activation switches.

**Structure:** 5 doors × 5 features = 25 total features

**Per-door features (5 each):**
- `[0]` **Switch X** — Normalized switch x position
- `[1]` **Switch Y** — Normalized switch y position  
- `[2]` **Door X** — Normalized door x position
- `[3]` **Door Y** — Normalized door y position
- `[4]` **Collected** — Switch collected = door open (0=closed, 1=open)

**Example indexing:**
```python
# Door 0: indices [0:5]
# Door 1: indices [5:10]
# Door 2: indices [10:15]
# Door 3: indices [15:20]
# Door 4: indices [20:25]

# Get door 2 state
door_2_switch_x = switch_states[10]
door_2_switch_y = switch_states[11]
door_2_door_x = switch_states[12]
door_2_door_y = switch_states[13]
door_2_open = switch_states[14]  # 1.0 means door is open
```

**Design notes:**
- Simplified from 6 to 5 features per door (removed redundant `door_open` field)
- `collected` field serves dual purpose: switch collected = door open
- Padding with zeros for levels with <5 locked doors
- Positions normalized to [0, 1] range

**Use cases:**
- Multi-step puzzle solving
- Door dependency tracking
- Switch collection planning
- Hierarchical task decomposition

---

## Graph Modality (Optional)

Enable with: `config.graph.enable_graph_updates = True`

### `graph_node_feats` — Node Features
**Shape:** `(max_nodes, 55)` | **Type:** `float32`

Node feature vector (55 dimensions per node):

**Spatial features (2):**
- Normalized (x, y) position

**Type features (10):**
- Node type one-hot encoding (geometry/entity/spawn/goal/etc.)

**Entity features (15):**
- Entity-specific attributes (state, activation, danger level, etc.)

**Tile features (20):**
- Surrounding tile context (walls, slopes, hazards)

**Reachability features (5):**
- Connectivity, accessibility, path metrics

**Proximity features (3):**
- Distance to key entities (player, goal, hazards)

### `graph_edge_index` — Edge Connectivity
**Shape:** `(2, max_edges)` | **Type:** `int32`

Edge list in COO format: `[[source_nodes], [target_nodes]]`

### `graph_edge_feats` — Edge Features
**Shape:** `(max_edges, 6)` | **Type:** `float32`

Edge feature vector (6 dimensions per edge):
- Edge type (spatial/entity/reachability)
- Distance metrics
- Connectivity strength

### `graph_node_mask` — Valid Nodes
**Shape:** `(max_nodes,)` | **Type:** `int32` | **Range:** `{0, 1}`

Binary mask indicating valid nodes (1=valid, 0=padding).

### `graph_edge_mask` — Valid Edges
**Shape:** `(max_edges,)` | **Type:** `int32` | **Range:** `{0, 1}`

Binary mask indicating valid edges (1=valid, 0=padding).

**Use cases:**
- Graph neural networks (GNN)
- Structural level understanding
- Relational reasoning
- Entity interaction modeling

---

## Observation Processing Pipeline

1. **Raw observation extraction** (`base_environment._get_observation()`)
   - Get ninja state from simulation
   - Get entity states from simulation
   - Extract positions and flags

2. **Visual processing** (`observation_processor.process_observations()`)
   - Crop player-centered frame
   - Downsample global view
   - Convert to grayscale

3. **State computation** (`observation_processor.process_game_state()`)
   - Update proximity features (nearest hazard/collectible)
   - Update progress features (switch/exit accessibility)
   - Normalize and clamp values

4. **Reachability analysis** (`observation_processor.process_reachability()`)
   - Compute flood fill from ninja position
   - Calculate area metrics
   - Update goal reachability

5. **Graph construction** (optional, `graph_builder.build_graph()`)
   - Extract geometry nodes
   - Add entity nodes
   - Compute edges and features

---

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
        # Concatenate all state vectors
        state_dim = 30 + 8 + 6 + 25  # game_state + reachability + positions + switches
        
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
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

### Multimodal Policy

```python
class MultimodalPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_encoder = CNNEncoder()  # Outputs 512-dim
        self.state_encoder = MLPEncoder()   # Outputs 256-dim
        self.fusion = nn.Linear(512 + 256, 512)
        
    def forward(self, obs):
        visual_feat = self.visual_encoder(obs['player_frame'], obs['global_view'])
        state_feat = self.state_encoder(obs['game_state'], obs['reachability_features'])
        combined = torch.cat([visual_feat, state_feat], dim=-1)
        return self.fusion(combined)
```

### GNN Policy

```python
import torch_geometric as pyg

class GNNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = pyg.nn.GATConv(55, 128, heads=4)
        self.global_pool = pyg.nn.global_mean_pool
        
    def forward(self, obs):
        node_feats = obs['graph_node_feats']
        edge_index = obs['graph_edge_index']
        node_mask = obs['graph_node_mask']
        
        # Apply node mask to filter padding
        valid_nodes = node_mask.bool()
        x = self.gnn(node_feats[valid_nodes], edge_index)
        
        # Global pooling for graph-level representation
        graph_feat = self.global_pool(x)
        return graph_feat
```

---

## Performance Considerations

### Memory Usage
- **Visual observations:** ~15KB per frame (player + global)
- **State vectors:** <0.5KB (all state vectors combined)
- **Graph observations:** ~50KB (with max_nodes=100, max_edges=400)
- **Total per step:** ~65KB with all modalities enabled

### Computation Time
- **Visual processing:** ~0.5ms (crop + downsample)
- **State computation:** ~0.1ms (array operations)
- **Reachability analysis:** ~0.8ms (OpenCV flood fill)
- **Graph construction:** ~2ms (optional, can disable)
- **Total per step:** ~1-3ms depending on modalities

### Optimization Tips
1. **Disable unused modalities:**
   ```python
   config.graph.enable_graph_updates = False  # Saves ~2ms + 50KB
   ```

2. **Use appropriate observation processing:**
   ```python
   # Vision-free training (fastest)
   extractor = VisionFreeExtractor()  # Only uses state vectors
   
   # CNN-only (moderate)
   extractor = CNNExtractor()  # Uses visual + minimal state
   
   # Full multimodal (slowest but most capable)
   extractor = MultimodalExtractor()  # Uses all modalities
   ```

3. **Leverage caching:**
   - Reachability is cached and only recomputed on position change
   - Graph is cached and only rebuilt on entity state change

---

## Validation and Debugging

### Check Observation Validity

```python
def validate_observation(obs):
    """Validate observation ranges and shapes."""
    assert obs['player_frame'].shape == (84, 84, 1)
    assert obs['global_view'].shape == (176, 100, 1)
    assert obs['game_state'].shape == (30,)
    assert obs['reachability_features'].shape == (8,)
    assert obs['entity_positions'].shape == (6,)
    assert obs['switch_states'].shape == (25,)
    
    # Check ranges
    assert obs['player_frame'].min() >= 0 and obs['player_frame'].max() <= 255
    assert obs['game_state'].min() >= -1 and obs['game_state'].max() <= 1
    assert obs['reachability_features'].min() >= 0
    assert obs['entity_positions'].min() >= 0 and obs['entity_positions'].max() <= 1
    
    print("✓ All observations valid!")
```

### Visualize Observations

```python
import matplotlib.pyplot as plt

def visualize_observations(obs):
    """Plot all observation modalities."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Visual
    axes[0, 0].imshow(obs['player_frame'][:, :, 0], cmap='gray')
    axes[0, 0].set_title('Player Frame')
    axes[0, 1].imshow(obs['global_view'][:, :, 0], cmap='gray')
    axes[0, 1].set_title('Global View')
    
    # State vectors
    axes[1, 0].bar(range(30), obs['game_state'])
    axes[1, 0].set_title('Game State')
    axes[1, 1].bar(range(8), obs['reachability_features'])
    axes[1, 1].set_title('Reachability')
    axes[1, 2].bar(range(25), obs['switch_states'])
    axes[1, 2].set_title('Switch States')
    
    plt.tight_layout()
    plt.show()
```

---

## Related Documentation

- **Reward System:** See `nclone/gym_environment/reward_calculation/` for reward computation
- **Graph Construction:** See `nclone/graph/` for graph building details
- **API Reference:** See `NPP_API_DOCUMENTATION.md` for complete API
- **Game Mechanics:** See `docs/sim_mechanics_doc.md` for physics details

---

## Changelog

### v1.1.0 (2025-10-21)
- **BREAKING:** Simplified `switch_states` from 30 to 25 features (removed redundant door_open field)
- **BREAKING:** Removed `hazard_threat_level` from ninja_state (index 23, redundant with nearest_hazard)
- **BREAKING:** Removed `completion_progress` from ninja_state (redundant with switch_progress + exit distance)
- **FIXED:** Clarified `game_state[26:30]` are NOT entity type counts but toggle_mine_count + first mine attributes
- **ADDED:** Comprehensive documentation of all observation components
- **ADDED:** Usage examples for CNN, MLP, and GNN policies
- **ADDED:** Performance benchmarks and optimization tips

### v1.0.0 (2025-10-01)
- Initial multi-modal observation space
- Visual, state vector, and graph modalities
- Reachability analysis integration
