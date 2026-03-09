# Global Level Context Specification: CNN-Encoded Tile Grid with Position Conditioning

**Document Version:** 1.0  
**Last Updated:** February 23, 2026  
**Parent Document:** TRAINING_PIPELINE_SPEC_v2.md  
**Purpose:** Provide the agent with full-level geometry awareness beyond the 8×8 local tile grid

---

## 1. Problem

The current spatial context provides an 8×8 tile grid centered on the ninja (192×192px coverage). The ninja's jump arcs can cover 200-300px horizontally and 150-200px vertically — regularly exceeding the visible grid. The agent must commit to jumps where the landing platform is invisible, plan routes through large open levels, and execute wall-jump chains requiring wall positions far above or below.

The existing reachability features (`platform_distance_ahead`, `platform_distance_below`) provide scalar distances in two directions but not the shape, gaps, or obstacle layout of distant geometry.

## 2. Design

Encode the full 42×23 tile grid through a small CNN computed **once per level**, then condition on the agent's position at each timestep via a lightweight MLP.

### Why Not Raw Flat Grid

42×23 = 966 scalar values. Feeding these directly as an MLP input works but wastes capacity — the network must independently learn spatial relationships (adjacency, corridors, platforms) that a CNN extracts naturally. A 3-layer CNN compresses 966 tiles into 64 semantically rich features at negligible cost.

### Why Not Per-Timestep CNN

The tile grid is static. Running the CNN every frame recomputes identical output ~2000 times per episode. Caching the CNN output and running only the position-conditioning MLP per timestep saves ~99.5% of the compute.

### Why Position Conditioning

A global level summary without position context is equally useful everywhere — it can't encode "the landing platform relative to *where I am now*." Concatenating the agent's normalized position before the final projection lets the network learn position-dependent interpretations: at position (0.3, 0.8), the relevant geometry differs from position (0.7, 0.2), even though the level is identical.

## 3. Architecture

```
Input:  tile_grid [42, 23] integer tile types (0-37)

Per-Level (cached):
  Normalize:     tile_grid / 37.0 -> [1, 1, 42, 23] float
  Conv2d(1, 32, 3, padding=1) + ReLU     -> [1, 32, 42, 23]
  Conv2d(32, 64, 3, stride=2, padding=1) + ReLU  -> [1, 64, 21, 12]
  Conv2d(64, 64, 3, stride=2, padding=1) + ReLU  -> [1, 64, 11, 6]
  AdaptiveAvgPool2d(4, 3)                -> [1, 64, 4, 3]
  Flatten                                -> [1, 768]
  Cache this vector for all timesteps in this level.

Per-Timestep:
  Concatenate: [grid_cache (768), agent_pos_norm (2)] -> [770]
  Linear(770, 128) + LayerNorm + ReLU
  Linear(128, 64) + LayerNorm + ReLU     -> [64]

Output: 64-dim position-conditioned level context vector
```

### Agent Position

Derived from game state (already available, no extra computation):

```python
agent_pos_norm = [ninja_x / 1056.0, ninja_y / 600.0]  # [0, 1] range
```

## 4. Integration with Feature Extractor

The 64-dim level context vector joins the existing modality features before fusion:

```
Existing modalities:
  Game state MLP:     27 -> 128
  Reachability MLP:   11 -> 64
  Spatial context MLP: 112 -> 128
  Mine SDF MLP:       3 -> 16
  Subtotal:           336

New:
  Level context:      770 -> 64
  
Fusion input:         400
Fusion output:        256 (unchanged)

Fusion layer:
  Linear(400, 256) + LayerNorm + ReLU
```

Each network (Q1, Q2, policy) has its own feature extractor including its own level context encoder. The CNN weights are shared within each network's encoder but separate across Q1/Q2/policy (consistent with the separate-extractor design in the training spec).

## 5. CNN Caching Strategy

```python
class LevelGridCache:
    """Caches CNN output per level. Invalidated on network update."""
    
    def __init__(self):
        self.cache = {}          # level_id -> [768] tensor
        self.cache_step = {}     # level_id -> global_step when cached
        self.refresh_interval = 500  # Re-encode every N gradient steps
    
    def get(self, level_id, tile_grid, encoder_cnn, global_step):
        needs_refresh = (
            level_id not in self.cache or
            global_step - self.cache_step.get(level_id, 0) > self.refresh_interval
        )
        if needs_refresh:
            with torch.no_grad():
                grid_tensor = torch.FloatTensor(tile_grid / 37.0)
                grid_tensor = grid_tensor.unsqueeze(0).unsqueeze(0).cuda()
                self.cache[level_id] = encoder_cnn(grid_tensor).flatten()
                self.cache_step[level_id] = global_step
        return self.cache[level_id]
```

The CNN weights change gradually during training, so cached embeddings remain accurate for hundreds of gradient steps. Refreshing every 500 steps balances accuracy against compute. With 50+ levels, each refresh is a single `[1, 1, 42, 23]` CNN forward pass (~0.05ms) — total refresh cost per cycle is ~2.5ms, negligible.

## 6. Replay Buffer Considerations

The replay buffer stores only the 153-dim dynamic observation. The tile grid is not stored per transition — it's looked up by level ID at training time. This requires the replay buffer to also store the level ID per transition:

```python
class ReplayBuffer:
    def __init__(self, capacity, obs_dim=153):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.level_ids = np.zeros(capacity, dtype=np.int32)    # NEW
        self.agent_positions = np.zeros((capacity, 2), dtype=np.float32)  # NEW: for pos conditioning
        self.next_agent_positions = np.zeros((capacity, 2), dtype=np.float32)  # NEW
        # ...
```

At training time, batch construction looks up the cached grid features by level ID and concatenates with agent positions:

```python
def prepare_batch(self, indices, grid_caches):
    batch = self.sample_by_indices(indices)
    
    # Look up cached grid features for each transition's level
    grid_features = torch.stack([
        grid_caches[level_id] for level_id in batch['level_ids']
    ])  # [batch, 768]
    
    batch['grid_cache'] = grid_features
    batch['agent_pos_norm'] = batch['agent_positions']
    batch['next_agent_pos_norm'] = batch['next_agent_positions']
    return batch
```

Additional replay buffer memory per transition: 4 bytes (level_id) + 8 bytes (pos) + 8 bytes (next_pos) = 20 bytes. For 1M transitions: 20 MB. Negligible.

## 7. Relationship to Existing 8×8 Local Grid

Both are retained. They serve complementary roles:

| Property | 8×8 Local Grid (existing) | 42×23 Global Grid (new) |
|---|---|---|
| Coverage | 192×192px around ninja | Full level (1056×600px) |
| Resolution | Per-tile (24px), 5 categories | Per-tile, compressed by CNN |
| Updates | Every timestep (position-centered) | Static per level |
| Role | Collision avoidance, precise jump timing | Route planning, landing platform awareness |
| Mine info | 8-nearest mine overlay (48 dims) | None (mine SDF handles global hazards) |
| Dims | 112 (64 grid + 48 mines) | 64 (CNN-compressed + position-conditioned) |

The local grid tells the agent "what's immediately around me." The global grid tells the agent "what does the whole level look like from where I am." Both are necessary for the full range of decisions: frame-precise collision avoidance (local) and multi-second route commitment (global).

## 8. Computational Cost

```
Per-level (cached, amortized to ~0):
  CNN forward:     [1, 1, 42, 23] -> [1, 768]     ~0.05 ms
  Refresh rate:    Every 500 gradient steps
  
Per-timestep (per environment):
  Grid MLP:        [770] -> [64]                    ~5-10 us
  Position lookup:  Indexing into game state          ~0 (already available)
  
Per training batch (batch=256):
  Grid cache lookup: 256 dict lookups               ~5 us
  Grid MLP forward:  [256, 770] -> [256, 64]        ~15-25 us
  
Total additional cost per training step:             ~20-35 us
Percentage of total step budget (~2650 us):          ~1%
```

## 9. CNN Parameter Count

```
Conv2d(1, 32, 3):     32 * (1*3*3 + 1) = 320
Conv2d(32, 64, 3):    64 * (32*3*3 + 1) = 18,496
Conv2d(64, 64, 3):    64 * (64*3*3 + 1) = 36,928
Linear(770, 128):     770 * 128 + 128 = 98,688
LayerNorm(128):        256
Linear(128, 64):      128 * 64 + 64 = 8,256
LayerNorm(64):         128

Total per network:     ~163K parameters
Total (Q1 + Q2 + policy): ~489K parameters

Added to existing ~3M total: ~16% increase. Negligible.
```
