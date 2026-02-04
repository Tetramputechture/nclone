# N++ Reinforcement Learning Observation Space

**Document Version:** 2.0  
**Last Updated:** February 2, 2026  
**Training Architecture:** `graph_free` (streamlined, no frame stacking)

---

## Table of Contents

1. [Version 2.0 Changes](#version-20-changes)
2. [Overview](#overview)
3. [Observation Components](#observation-components)
   - [Game State (27 dims)](#game-state-27-dims)
   - [Reachability Features (11 dims)](#reachability-features-11-dims)
   - [Spatial Context (112 dims)](#spatial-context-112-dims)
   - [Mine SDF Features (3 dims)](#mine-sdf-features-3-dims)
4. [Feature Extraction Pipeline](#feature-extraction-pipeline)
5. [Architecture Details](#architecture-details)

---

## Version 2.0 Changes

**February 2026 - Major Simplification Update**

### Summary
- **Dimensionality**: 301 → 153 (49% reduction)
- **Frame stacking**: Removed (velocity explicitly observed)
- **MDP completeness**: Added `jump_duration_remaining`
- **Memory**: 1.2 KB → 0.6 KB per observation (50% reduction)
- **Training speed**: 10-20% faster iterations

### Game State Changes (41 → 27 dims)

**Added:**
- `jump_duration_remaining` (MDP critical): Frames remaining until forced fall during jump

**Removed (14 features):**
- `velocity_magnitude`: Redundant (computable from velocity_x, velocity_y)
- `velocity_direction` (x, y): Redundant (normalized velocity)
- `kinetic_energy`: Redundant (0.5 × (vx² + vy²))
- `potential_energy`: Redundant (function of position_y)
- `floor_normal_strength`: Redundant (magnitude of floor_normal)
- `surface_slope`: Redundant (same as floor_normal_y)
- `ceiling_normal` (x, y): Low utility
- Temporal features (6): Redundant with explicit velocity observation
- `force_magnitude`, `energy_change_rate`: Derivative features
- `floor_contact_strength`, `wall_contact_strength`: Duplicate of binary contact
- `surface_slope_angle`, `wall_interaction_strength`: Redundant

### Reachability Changes (22 → 11 dims)

**Kept (11 features):**
- Path distances (2): A* path lengths to switch and exit
- Phase indicator (1): `switch_activated` (Markov property)
- A* guidance (2): `next_hop_direction` (optimal path)
- Detour detection (1): Flag when A* opposes Euclidean direction
- Mine clearance (2): SDF gradient (safe direction away from mines)
- Exit reachable (1): Can reach exit door
- Platform distances (2): Ahead and below (critical for blind jumps)

**Removed (11 features):**
- Euclidean directions (4): **MISLEADING** - can point into walls, contradict optimal policy
- Global statistics (3): `reachable_area_ratio`, `total_mines_norm`, `deadly_mine_ratio` - low information density
- Redundant inverses (2): `dist_to_next_objective_inv`, `dist_to_exit_inv`
- 6 of 8 directional connectivity features: Kept only 2 most relevant directions

### Rationale

**Frame Stacking Removal:** With velocity explicitly observed, frame stacking provides zero additional information while quadrupling observation size. Stacking is only necessary when velocity must be inferred from position differences (e.g., Atari pixels).

**Euclidean Direction Removal:** On levels requiring detours, Euclidean direction actively contradicts A* optimal path:
```
Example: [NINJA] ----WALL---- [EXIT]
             |                    |
             +----[PATH]----------+

Euclidean: → (toward wall, wrong)
A* direction: ↓ (toward path, correct)
```

**MDP Completeness:** Added `jump_duration_remaining` resolves POMDP where two states with identical observations have different optimal actions (early jump vs. late jump).

---

## Overview

The observation space is designed for **graph-free** reinforcement learning with **streamlined features**. This configuration achieves:

- **99.7% memory reduction** vs full graph observations
- **O(1) observation extraction** per timestep
- **True MDP** (velocity + jump_duration explicitly observed)
- **Path-guided navigation** using A*-derived features only

**Total Dimensions:** 153 (27 + 11 + 112 + 3)

**Architecture:** `graph_free` processes these into 256-dimensional feature vectors.

---

## Frame Stacking Configuration

### Enabled Components

| Component | Stack Size | Raw Dims | Stacked Dims | Purpose |
|-----------|------------|----------|--------------|---------|
| **Game State** | 4 frames | 41 | 164 | Infer velocity, acceleration from position differences |
| **Reachability** | 1 frame | 22 | 22 | Static path guidance (A* precomputed) |
| **Spatial Context** | 1 frame | 112 | 112 | Local geometry (8×8 tile grid) |
| **Mine SDF** | 1 frame | 3 | 3 | Global mine danger signal |

### Frame Order Convention

Frames are stored **oldest-first** in the stack:
```
[t-3, t-2, t-1, t] (current)
```

This allows the network to learn temporal patterns:
- Velocity: `position[t] - position[t-1]`
- Acceleration: `velocity[t] - velocity[t-1]`
- Trajectory prediction from 4-point history

---

## Observation Components

### Game State (27 dims)

**Source:** `nplay_headless.get_ninja_state()`  
**Range:** All features normalized to `[-1, 1]`  
**Stacking:** 4 frames (164 total dims)

#### Core Movement State (8 features)

| # | Feature | Description | Normalization |
|---|---------|-------------|---------------|
| 1 | `velocity_magnitude` | Speed magnitude | `[0, MAX_HOR_SPEED*2]` → `[-1, 1]` |
| 2-3 | `velocity_direction` | Velocity unit vector (x, y) | Already `[-1, 1]` |
| 4 | `ground_movement` | On ground: immobile/running/sliding | `{-1.0, 1.0}` binary |
| 5 | `air_movement` | In air: jumping/falling | `{-1.0, 1.0}` binary |
| 6 | `wall_interaction` | Wall sliding | `{-1.0, 1.0}` binary |
| 7 | `special_states` | Dead/celebrating/disabled | `{-1.0, 1.0}` binary |
| 8 | `airborne_status` | Whether ninja is airborne | `{-1.0, 1.0}` binary |

#### Input and Buffer State (5 features)

| # | Feature | Description | Normalization |
|---|---------|-------------|---------------|
| 9 | `horizontal_input` | Current horizontal input | `{-1.0, 0.0, 1.0}` discrete |
| 10 | `jump_input` | Current jump button state | `{-1.0, 1.0}` binary |
| 11 | `jump_buffer` | Jump input buffer window | `[0, 5]` frames → `[-1, 1]` |
| 12 | `floor_buffer` | Floor contact buffer | `[0, 5]` frames → `[-1, 1]` |
| 13 | `wall_buffer` | Wall contact buffer | `[0, 5]` frames → `[-1, 1]` |

**Note:** Input buffers provide temporal smoothing for responsive controls (coyote time, jump buffering).

#### Surface Contact Information (6 features)

| # | Feature | Description | Normalization |
|---|---------|-------------|---------------|
| 14 | `floor_contact` | Floor collision count | `{0, 1}` → `{-1, 1}` |
| 15 | `wall_contact` | Wall collision count | `{0, 1}` → `{-1, 1}` |
| 16 | `ceiling_contact` | Ceiling collision count | `{0, 1}` → `{-1, 1}` |
| 17 | `floor_normal_strength` | Floor normal magnitude | `[0, 1]` → `[-1, 1]` |
| 18 | `wall_direction` | Wall normal direction | `{-1.0, 0.0, 1.0}` |
| 19 | `surface_slope` | Floor normal Y component | `[-1, 1]` (upward positive) |

#### Additional Physics State (5 features)

| # | Feature | Description | Normalization |
|---|---------|-------------|---------------|
| 20 | `applied_gravity` | Current gravity constant | `[GRAVITY_JUMP, GRAVITY_FALL]` → `[-1, 1]` |
| 21 | `walled_status` | Against wall | `{-1.0, 1.0}` binary |
| 22 | `floor_normal_x` | Floor normal X component | `[-1, 1]` |
| 23-24 | `ceiling_normal` | Ceiling normal vector (x, y) | `[-1, 1]` per component |
| 25 | `applied_drag` | Current drag coefficient | `[DRAG_SLOW, DRAG_REGULAR]` → `[-1, 1]` |
| 26 | `applied_friction` | Current friction coefficient | `[FRICTION_GROUND_SLOW, FRICTION_GROUND]` → `[-1, 1]` |

#### Temporal Features (6 features)

| # | Feature | Description | Source |
|---|---------|-------------|--------|
| 27-32 | `temporal_features` | Frame-to-frame dynamics | `ninja.get_temporal_features()` |

These include acceleration, jerk, and short-term trajectory information computed from internal buffers.

#### Enhanced Physics Features (8 features)

| # | Feature | Description | Normalization |
|---|---------|-------------|---------------|
| 33 | `kinetic_energy` | `0.5 * (vx² + vy²)` | `[0, MAX_HOR_SPEED²]` → `[-1, 1]` |
| 34 | `potential_energy` | Height-based potential | `[0, LEVEL_HEIGHT]` → `[-1, 1]` |
| 35 | `force_magnitude` | Applied force magnitude | `sqrt(gravity² + accel²)` → `[-1, 1]` |
| 36 | `energy_change_rate` | Kinetic energy derivative | Rate of change → `[-1, 1]` |
| 37 | `floor_contact_strength` | Floor contact intensity | `[0, 5]` contacts → `[-1, 1]` |
| 38 | `wall_contact_strength` | Wall contact intensity | `[0, 3]` contacts → `[-1, 1]` |
| 39 | `surface_slope_angle` | Arctangent of floor normal | `atan2(ny, nx) / π` → `[-1, 1]` |
| 40 | `wall_interaction_strength` | Wall normal (repeated) | `[-1, 1]` |

#### Time Remaining (1 feature)

| # | Feature | Description | Normalization |
|---|---------|-------------|---------------|
| 41 | `time_remaining` | Frames until timeout | `[0, MAX_FRAMES]` → `[0, 1]` |

**Critical:** Time remaining is NOT stacked (would be redundant - already contains temporal information).

---

### Reachability Features (11 dims)

**Source:** `compute_reachability_features_from_graph()`  
**Range:** Features normalized to `[0, 1]` or `[-1, 1]`  
**Stacking:** 1 frame only (path guidance is static for current state)

These features provide **A*-derived optimal path guidance** without requiring graph observations.

#### Base Features (4 dims)

| # | Feature | Description | Range |
|---|---------|-------------|-------|
| 0 | `reachable_area_ratio` | Fraction of level reachable | `[0, 1]` |
| 1 | `dist_to_next_objective_inv` | Inverted distance to switch/locked door/exit | `[0, 1]` (1=close) |
| 2 | `dist_to_exit_inv` | Inverted distance to exit door | `[0, 1]` (1=close) |
| 3 | `exit_reachable` | Whether exit is in reachable graph | `{0, 1}` binary |

**Objective Priority:** Features 1 uses hierarchy: exit switch → locked door switches → exit door.

#### Path Distances (2 dims)

| # | Feature | Description | Range |
|---|---------|-------------|-------|
| 4 | `path_dist_to_switch_norm` | Normalized path distance to switch | `[0, 1]` |
| 5 | `path_dist_to_exit_norm` | Normalized path distance to exit | `[0, 1]` |

**Non-inverted:** These provide raw distance signal for learning.

#### Direction Vectors (4 dims)

| # | Feature | Description | Range |
|---|---------|-------------|-------|
| 6-7 | `direction_to_switch` | Euclidean unit vector to switch (x, y) | `[-1, 1]` |
| 8-9 | `direction_to_exit` | Euclidean unit vector to exit (x, y) | `[-1, 1]` |

**Approximation:** Euclidean direction (not optimal path direction for simplicity).

#### Mine Context (2 dims)

| # | Feature | Description | Range |
|---|---------|-------------|-------|
| 10 | `total_mines_norm` | Total mine count | `[0, 1]` (max 10 mines) |
| 11 | `deadly_mine_ratio` | Fraction of mines in deadly state | `[0, 1]` |

#### Phase Indicator (1 dim)

| # | Feature | Description | Range |
|---|---------|-------------|-------|
| 12 | `switch_activated` | Whether exit switch collected | `{0, 1}` binary |

**Critical for Markov property:** Indicates which objective to pursue (switch vs exit).

#### Path Direction Features (5 dims)

| # | Feature | Description | Range |
|---|---------|-------------|-------|
| 13-14 | `next_hop_direction` | Optimal path direction from A* (x, y) | `[-1, 1]` |
| 17 | `path_requires_detour` | Whether optimal path opposes Euclidean direction | `{0, 1}` binary |
| 15-16 | `mine_clearance_direction` | Safe direction from mine SDF gradient (x, y) | `[-1, 1]` |

**Inflection Point Navigation:** Features 13-14 and 17 solve "wrong direction" problem (e.g., go LEFT when goal is RIGHT due to level geometry).

#### Directional Connectivity (8 dims)

| # | Feature | Description | Range |
|---|---------|-------------|-------|
| 18 | `platform_distance_E` | Distance to platform in East direction | `[0, 1]` normalized |
| 19 | `platform_distance_NE` | Distance to platform in Northeast | `[0, 1]` |
| 20 | `platform_distance_N` | Distance to platform in North | `[0, 1]` |
| 21 | `platform_distance_NW` | Distance to platform in Northwest | `[0, 1]` |
| 22 | `platform_distance_W` | Distance to platform in West | `[0, 1]` |
| 23 | `platform_distance_SW` | Distance to platform in Southwest | `[0, 1]` |
| 24 | `platform_distance_S` | Distance to platform in South | `[0, 1]` |
| 25 | `platform_distance_SE` | Distance to platform in Southeast | `[0, 1]` |

**Blind Jump Verification:** Provides platform existence beyond 192px visibility radius.

**Removed Features (2026-02-03):**
- Waypoints (3 dims): Temporal LSTM-specific, redundant with frame stacking
- Path difficulty (1 dim): Not needed with frame stacking
- Path curvature (3 dims): LSTM lookahead, frame stacking handles anticipatory turning
- Exit lookahead (5 dims): LSTM switch discontinuity mitigation

---

### Spatial Context (112 dims)

**Source:** `spatial_context.compute_spatial_context_features()`  
**Range:** Normalized to `[0, 1]`  
**Stacking:** 1 frame only (local geometry changes slowly)

Provides **graph-free local geometry** with 99.7% memory reduction vs full graph observations.

#### Local Tile Grid (64 dims)

**Layout:** 8×8 grid centered on ninja (192×192 pixel coverage at 24px per cell)

**Encoding:** Each cell is a simplified tile category `[0-4]` normalized:
- `0.0`: Empty (air)
- `0.25`: Full tile (solid wall)
- `0.5`: Half tile (45° slope)
- `0.75`: Concave curve
- `1.0`: Convex curve

**Purpose:** Local obstacle awareness for collision avoidance and navigation.

#### Mine Overlay (48 dims)

**Layout:** 8 nearest mines × 6 features each

**Per-mine features:**
| Feature | Description | Range |
|---------|-------------|-------|
| `relative_x` | X offset from ninja | `[-1, 1]` (normalized by view radius) |
| `relative_y` | Y offset from ninja | `[-1, 1]` |
| `state` | Mine state (safe/deadly) | `{0.0, 1.0}` binary |
| `radius` | Mine collision radius | `[0, 1]` normalized |
| `velocity_dot_direction` | Alignment of ninja velocity with mine direction | `[-1, 1]` |
| `distance_rate` | Rate of distance change (approach/recede) | `[-1, 1]` |

**Markovian:** Uses only current state (no temporal history) to maintain MDP property.

**Purpose:** Mine avoidance and danger assessment for actor safety.

---

### Mine SDF Features (3 dims)

**Source:** `mine_sdf.get_gradient_at_position()`  
**Range:** Normalized values  
**Stacking:** 1 frame only

Provides **global mine danger signal** from ALL mines (complementing 8-nearest in spatial context).

| # | Feature | Description | Range |
|---|---------|-------------|-------|
| 0 | `sdf_value` | Signed distance to nearest mine | `[-1, 1]` (negative=danger) |
| 1-2 | `escape_gradient` | Direction away from nearest mine (x, y) | `[-1, 1]` unit vector |

**Purpose:** Actor safety awareness beyond local 8-mine visibility.

---

## Feature Extraction Pipeline

### 1. Input Processing

```
Raw Observations (with 4-frame stacking):
├── game_state: [batch, 4, 41] → flatten to [batch, 164]
├── reachability_features: [batch, 22]
├── spatial_context: [batch, 112]
└── mine_sdf_features: [batch, 3]
```

### 2. Feature Extractors

The `ConfigurableMultimodalExtractor` processes each modality:

#### Game State MLP (164 → 128 dims)

```python
StateMLP(
    input_dim=164,  # 41 * 4 frames
    hidden_dim=128,
    output_dim=128,
)
# Architecture:
#   Linear(164, 128) → ReLU → Dropout(0.1)
#   Linear(128, 128) → ReLU
```

**Handles Frame Stacking:** Automatically flattens `[batch, 4, 41]` to `[batch, 164]`.

#### Reachability MLP (22 → 96 dims)

```python
nn.Sequential(
    Linear(22, 128),     # Path guidance, high capacity
    ReLU(),
    Dropout(0.2),        # Higher dropout for rich features
    Linear(128, 128),
    ReLU(),
    Dropout(0.2),
    Linear(128, 96),     # Increased from 64 (v1.0: 50% boost)
    ReLU(),
)
```

**Information-Rich:** A*-derived features justify 3-layer deep network.

#### Spatial Context MLP (112 → 128 dims)

```python
nn.Sequential(
    Linear(112, 128),
    ReLU(),
    Dropout(0.2),
    Linear(128, 128),
    ReLU(),
    Dropout(0.2),
    Linear(128, 128),
    ReLU(),
)
```

**Local Geometry:** Processes tile grid + mine overlay for collision avoidance.

#### Mine SDF MLP (3 → 16 dims)

```python
nn.Sequential(
    Linear(3, 32),       # Compact hidden layer
    ReLU(),
    Linear(32, 16),      # Compact output
    ReLU(),
)
```

**Global Safety:** Lightweight projection of global mine danger signal.

### 3. Fusion

```python
# Concatenate all modality features
combined = torch.cat([
    state_features,           # 128 dims
    reachability_features,    # 96 dims
    spatial_context_features, # 128 dims
    mine_sdf_features,        # 16 dims
], dim=1)  # Total: 368 dims

# Simple concatenation fusion (graph_free architecture)
output = nn.Sequential(
    Linear(368, 384),    # Project to final feature dim
    ReLU(),
    Dropout(0.1),
    Linear(384, 384),    # Final feature vector
)(combined)
```

**Output:** `[batch, 384]` feature vectors for policy and value networks.

---

## Architecture Details

### Graph-Free Architecture (`graph_free`)

**Configuration:** From `architecture_configs.py`

```python
ArchitectureConfig(
    name="graph_free",
    modalities=ModalityConfig(
        use_player_frame=False,      # No visual observations
        use_global_view=False,       # No global view
        use_graph=False,             # No GNN (graph-free)
        use_game_state=True,         # ✓ Ninja physics
        use_reachability=True,       # ✓ Path guidance
        use_spatial_context=True,    # ✓ Local geometry
    ),
    state=StateConfig(
        hidden_dim=128,
        output_dim=128,
    ),
    fusion=FusionConfig(
        fusion_type=FusionType.CONCAT,  # Simple concatenation
        dropout=0.1,
    ),
    features_dim=384,                    # Final output
)
```

### Memory Efficiency

| Architecture | Obs Size | Rollout Buffer (64k steps) | Reduction |
|-------------|----------|---------------------------|-----------|
| Full Graph | ~162 KB | ~10.4 GB | Baseline |
| Graph-Free | ~0.5 KB | ~33 MB | **99.7%** |

**Key:** Spatial context (112 dims) replaces full graph (~800 nodes × 6 dims × 4 bytes).

### Training Configuration

**Frame Stacking:**
- Visual: Not used (graph-free)
- State: 4 frames (`--enable-state-stacking --state-stack-size 4`)
- Padding: Zero-padding for initial frames

**Observation Space:**
```python
gym.spaces.Dict({
    'game_state': Box(low=-1.0, high=1.0, shape=(4, 41)),  # Stacked
    'reachability_features': Box(low=0.0, high=1.0, shape=(22,)),
    'spatial_context': Box(low=0.0, high=1.0, shape=(112,)),
    'mine_sdf_features': Box(low=-1.0, high=1.0, shape=(3,)),
})
```

**Policy Network:** Uses extracted 384-dim features from `ConfigurableMultimodalExtractor`.

---

## Summary

**Total Observation Dimensions:**
- Raw (stacked): 246 dims (164 + 22 + 112 + 3)
- Processed (features): 384 dims
- Policy/Value input: 384 dims

**Design Principles:**
1. **Markovian State:** Frame stacking provides velocity/acceleration without explicit history
2. **Graph-Free:** 99.7% memory reduction using spatial context
3. **Path Guidance:** A*-derived features (reachability) provide optimal navigation hints
4. **Safety Awareness:** Global mine SDF + local 8-nearest mines for actor safety
5. **Normalized Inputs:** All features in `[-1, 1]` or `[0, 1]` for stable training

**Training Performance:**
- Memory: 33 MB rollout buffer (vs 10.4 GB full graph)
- Speed: ~0.3ms per observation extraction
- Parallelization: Supports 64+ environments per GPU

---

## References

- **Frame Stacking:** Mnih et al. (2015) - DQN with 4-frame stacking
- **Graph-Free Design:** Internal analysis showing 99.7% memory reduction
- **Architecture:** `npp-rl/npp_rl/training/architecture_configs.py`
- **Feature Computation:** `nclone/nclone/graph/reachability/feature_computation.py`
- **Spatial Context:** `nclone/nclone/gym_environment/spatial_context.py`
