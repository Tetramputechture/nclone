# Surface-Graph PBRS: Exact Cost-to-Go via Physics-Based Surface Graphs

This package precomputes optimal cost-to-go from every walkable surface state to level goals (switch and exit door) using a velocity-augmented surface graph and Dijkstra's algorithm. The result is used as a Potential-Based Reward Shaping (PBRS) signal during RL training.

## Key Advantages

1. **Exact Cost-to-Go**: Dijkstra on the surface graph gives frame-exact shortest paths
2. **Physics-Based Edges**: Graph edges are discovered by physics simulation, not heuristics
3. **Compact Representation**: ~5 MB per level vs ~50-200 MB for the old 5D grid
4. **Fast Precomputation**: ~1-5 minutes on 8 cores vs ~30+ minutes for value iteration
5. **Velocity-Aware**: Nodes are augmented with discretised velocity for momentum-sensitive routing
6. **Two-Phase Support**: Handles switch-then-exit levels with phase-dependent potentials

## Architecture Overview

```
Precomputation Pipeline:
  1. Extract walkable surfaces (floor/wall) from tile geometry
  2. Build velocity-augmented surface graph (~45K nodes)
  3. Enumerate edges via physics simulation (~5M sims, parallel)
  4. Backward Dijkstra to switch and exit goals (<1s each)
  5. Build arc cache for O(1) airborne lookups
  6. Save to single .npz file (~5 MB)

Runtime Evaluation:
  Surface contact → nearest node lookup + velocity interpolation → d*/d_max
  Airborne → arc cache (O(1)) or pessimistic fallback
  Potential: Φ(s) = -d*[node] / d_max + Φ_phase
```

## Components

### 1. Surface Extraction (`surface_extraction.py`)

Extracts walkable floor and wall surfaces from tile collision segments:
- Classifies segments by surface normal (floor: ny < -0.3, wall: |nx| > 0.7)
- Merges connected segments into continuous surface runs
- Samples SurfacePoint positions at 6px intervals (3px near edges)
- Positions are ninja-center (offset by NINJA_RADIUS from surface contact)

### 2. Surface Graph (`surface_graph.py`)

Builds the velocity-augmented directed graph:
- Floor nodes: 15 velocity bins over [-3.333, +3.333] px/frame
- Wall nodes: 12 velocity bins over [-2.0, +5.5] px/frame
- Spatial index for O(1) nearest-node lookup
- Backward Dijkstra from goal node sets
- Velocity interpolation between adjacent bins

### 3. Edge Enumeration (`edge_enumeration.py`)

Discovers graph edges by physics simulation:
- Running (floor): 3 h_inputs × 11 durations = 33 sims/node
- Floor jumps: 3 h_inputs × 12 hold durations × 3 post-release = 108 sims/node
- Wall jumps: 2 types × 3 h_inputs × 12 hold durations = 72 sims/node
- Edge falls: 3 h_inputs from platform-edge nodes
- Multiprocessing parallelization across nodes
- Collects arc waypoints for the arc cache

### 4. Arc Cache (`arc_cache.py`)

Spatial hash for O(1) airborne state cost lookup:
- Key: (x//12, y//12, quantize(vx, 0.5), quantize(vy, 0.5))
- Value: min cost-to-go from any arc waypoint passing through the cell
- Built from waypoints collected during edge enumeration

### 5. Airborne Evaluator (`airborne_evaluator.py`)

Forward simulation of ~45 representative control strategies:
- Jump-holding: 5 release times × 3 h_inputs = 15 strategies
- Falling: 3 h_inputs
- Mid-change: 3 switch times × 3×3 input combos ~27 strategies
- Returns min(frames_to_landing + d*[landing_node])

### 6. Runtime PBRS (`precomputed_pbrs.py`, `two_phase_pbrs.py`)

Runtime potential lookup for the gym environment:
- Surface contact: nearest-node lookup + velocity interpolation
- Airborne: arc cache hit → O(1); miss → pessimistic fallback
- Two-phase: switch d* (phase 0) / exit d* + C_PHASE (phase 1)
- Interface: `get_potential(state_dict)` → float

### 7. Physics Step Wrapper (`physics_step_wrapper.py`)

Stateless interface to the Rust physics engine:
- State injection via `inject_ninja_for_value_iteration(x, y, vx, vy, jd)`
- Frame-by-frame stepping via `tick_physics_only(h_input, jump_input)`
- Contact state queries for surface-graph PBRS

## Usage

### Precomputing (Offline)

```bash
# Basic (8 workers)
python -m nclone.value_function.precompute_cli \
    --map-path levels/rampjump.map --num-workers 8

# Single-process for debugging
python -m nclone.value_function.precompute_cli \
    --map-path levels/rampjump.map --num-workers 1 -v
```

Output: `value_functions/{level}_surface_pbrs.npz`

### Runtime (Training)

```python
from nclone.gym_environment import NPPEnvironment

env = NPPEnvironment(
    value_function_dir="value_functions/",
    use_precomputed_pbrs=True,
)
```

The environment automatically loads the appropriate surface graph PBRS for each level.

### Visualization

```bash
# Surface point heatmap coloured by d*
python -m nclone.value_function.visualize_value_function \
    --data value_functions/rampjump_surface_pbrs.npz --mode heatmap

# Graph edges overlay
python -m nclone.value_function.visualize_value_function \
    --data value_functions/rampjump_surface_pbrs.npz --mode edges

# Dashboard with statistics
python -m nclone.value_function.visualize_value_function \
    --data value_functions/rampjump_surface_pbrs.npz --mode dashboard
```

## Performance

### Precomputation Cost (per level, post-B1..B7 hardening, 4 workers)

Measured on the NPP-RL benchmark suite with the memory-hardened pipeline
(compact NumPy waypoints, shared reversed adjacency, vectorised arc-cache
build, subprocess-per-level batching).  See `docs/PRECOMPUTE_AUDIT.md` for
the current per-level audit run.

| Level (representative) | Surf. points | Graph nodes | Edges | Unique wp cells | Peak RSS (1 worker) | Wall (4 workers) |
|---|---:|---:|---:|---:|---:|---:|
| `simple-walk`   | 478 | 7,170 | ~46K | ~50K | ~3 GB  | ~80s |
| `only-jump`     | 438 | 6,570 | ~8K  | ~20K | ~1 GB  | ~55s |
| `switch-simple` | 2-3K | ~40K  | ~60K | ~50K | ~2 GB  | ~60s |
| `fall-required` | >3K  | ~50K  | ~80K | ~80K | ~4 GB  | ~120s |

**8 GB budget:** all benchmark levels complete within budget when run under
`npp-rl/scripts/precompute_all.py` (subprocess-per-level isolation).  On
very tight budgets pass `--arc-waypoint-interval 2` (halves wp memory
with minor arc-cache coverage loss) or `--no-waypoints` (graph + Dijkstra
only, for route-PBRS-only workflows).

### Runtime Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Surface potential lookup | ~5 μs | O(1) spatial hash + velocity interpolation |
| Arc cache lookup | ~1 μs | O(1) spatial hash |
| Airborne fallback | ~10 μs | Pessimistic estimate (no sim needed) |

## Supported Mechanics

The surface graph is built from the tile collision geometry plus toggle
mine positions; physics-based edges are discovered by running the same
native simulator used at training time.  The following mechanics are
**fully supported** by the precompute pipeline:

- Running/walking along floor surfaces (all horizontal inputs, arbitrary
  durations up to 60 frames per arc).
- Jumping from floors (12 hold durations × 3 horizontal inputs during
  hold × 3 post-release inputs).
- Wall-sliding and wall-jumping (regular + slide variants).
- Edge falls (3 horizontal inputs from is-edge nodes).
- Toggle mines (type 21) — proximity inflates edge cost via an inverse-
  linear penalty (`1 + min(β/clearance, cap)`) and arcs that enter the
  kill zone are pruned.  Both initially-deadly and initially-safe mines
  participate because the return trip through a toggled mine is lethal.

The following mechanics are **not yet modeled** in the Dijkstra layer and
fall back to conservative runtime fallbacks:

- **Locked doors / switches as graph state.**  The current pipeline emits
  two independent d_stars (`d_star_switch`, `d_star_exit`) and a
  phase-boundary constant; it does not enumerate the combinatorial state
  space of which doors are open.  On levels with locked doors the d_star
  to the exit assumes all doors are already open, so the resulting PBRS
  may be optimistic until the matching switch is collected.
- **Drones, thwumps, bounce blocks, launch pads.**  These dynamic
  hazards/helpers are absent from the graph; the agent is expected to
  learn them reactively from the observation stream.
- **Gold pickups.**  Not modeled; only switch and exit goals are
  precomputed.

For a full gap analysis and the plan to address them, see
`docs/LINEAR_PLAN_AUDIT.md` (observation/action/reward workstreams) and
the per-release notes in `docs/PRECOMPUTE_AUDIT.md`.

## Serialization Format

```
{level_id}_surface_pbrs.npz:
    surface_points_x: float32[N]      # Ninja center x positions
    surface_points_y: float32[N]      # Ninja center y positions
    surface_normals_x: float32[N]     # Surface outward normals
    surface_normals_y: float32[N]
    surface_types: uint8[N]           # 0=floor, 1=wall
    surface_segment_ids: int32[N]     # Merged surface run id
    surface_is_edge: bool[N]          # Platform edge flag
    velocity_bins_floor: float32[15]  # Floor velocity bin centers
    velocity_bins_wall: float32[12]   # Wall velocity bin centers
    edges_src: int32[E]               # Edge source node ids
    edges_dst: int32[E]               # Edge destination node ids
    edges_weight: float32[E]          # Edge frame counts
    d_star_switch: float32[M]         # Cost-to-go to switch
    d_star_exit: float32[M]           # Cost-to-go to exit
    arc_cache_keys_{phase}: int32[A,4]    # Spatial hash keys
    arc_cache_values_{phase}: float32[A]  # Cached costs
    metadata: dict                    # d_max values, counts
```

## Files

```
nclone/value_function/
├── __init__.py                  # Package exports
├── physics_step_wrapper.py      # Stateless physics interface
├── surface_extraction.py        # Tile → surface points
├── surface_graph.py             # Velocity-augmented graph + Dijkstra
├── edge_enumeration.py          # Physics-based edge discovery
├── arc_cache.py                 # Airborne state spatial hash
├── airborne_evaluator.py        # Forward-sim strategies
├── precomputed_pbrs.py          # Runtime PBRS lookup
├── two_phase_pbrs.py            # Two-phase wrapper + factory
├── precompute_cli.py            # CLI precomputation tool
├── visualize_value_function.py  # Visualization tool
└── README.md                    # This file
```

## References

- **Potential-Based Reward Shaping**: Ng et al. (1999) - "Policy Invariance Under Reward Shaping"
- **N++ Physics**: Metanet Software - N++ game documentation
