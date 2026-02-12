---
name: Native Physics Port Investigation
overview: Comprehensive plan for porting nclone's physics to Rust, with clear Rust/Python responsibility boundaries to eliminate duplication. Rendering is excluded from scope.
todos:
  - id: phase1-scaffold
    content: Scaffold nclone-native Rust crate with PyO3/maturin, Cargo.toml, pyproject.toml, basic module structure
    status: completed
  - id: phase1-types
    content: "Port core types and constants: Vec2, TileType enum, NinjaState enum, Segment enum, tile definition const tables, physics constants. Export all to Python via PyO3 submodule so Python graph/reward code imports from nclone_native.constants instead of nclone.constants"
    status: completed
  - id: phase1-collision
    content: "Port collision functions: sweep_circle_vs_tiles, get_single_closest_point, gather_segments_from_region, GridSegmentLinear, GridSegmentCircular"
    status: completed
  - id: phase1-ninja
    content: Port Ninja struct with full state machine. Expose NinjaSnapshot dataclass via PyO3 containing all 50+ fields needed by Python observation/reward code
    status: completed
  - id: phase1-parity
    content: "Build parity test framework: run same inputs through Python and Rust sims, assert identical ninja state snapshots frame-by-frame"
    status: completed
  - id: phase2-entities
    content: "Port entity system: base Entity + 3 entity types only (toggle mine, exit switch, exit door). No gold, drones, thwumps, bounce blocks, launch pads, locked doors, or other entities. Expose EntitySnapshot list via PyO3 for Python graph code"
    status: completed
  - id: phase3-simulator
    content: Port full Simulator with tick() + tick_physics_only(). Expose SimulatorState (ninja_snapshot + entity_snapshots + tile_grid + win/death/action_mask) as single cross-boundary call
    status: completed
  - id: phase3-headless
    content: Create NativePlayHeadless Python wrapper that implements identical API to NPlayHeadless but delegates to Rust Simulator. Python gym env uses this transparently
    status: completed
  - id: phase3-vecenv
    content: "Build NativeVecEnv: in-process parallel sims using rayon, single Python call steps N sims and returns N state snapshots"
    status: completed
  - id: phase4-precompute
    content: "Port graph building + pathfinding to Rust: sub-node generation, adjacency, BFS/Dijkstra, mine SDF, level cache. Expose as Python-callable functions returning numpy arrays"
    status: pending
isProject: false
---

# Native Physics Port: Rust/Python Responsibility Boundaries

## Guiding Principle

**Rust owns physics execution. Python owns intelligence (reward shaping, observation construction, graph-based reasoning).** The boundary is a flat state snapshot struct that crosses the FFI exactly once per step. No rendering in either language for the training path.

---

## System Inventory (What Exists Today)

### Code to be ported to Rust (~5,500 lines)

These files contain the physics simulation and will be **replaced** by Rust:

- [nclone/physics.py](nclone/nclone/physics.py) (518 lines) -- collision detection, sweep tests, spatial queries
- [nclone/entities.py](nclone/nclone/entities.py) (304 lines) -- `Entity` base, `GridSegmentLinear`, `GridSegmentCircular`
- [nclone/ninja.py](nclone/nclone/ninja.py) (1,571 lines) -- ninja state machine, collision resolution, jump mechanics
- [nclone/nsim.py](nclone/nclone/nsim.py) (348 lines) -- `Simulator.tick()` loop, 4 physics substeps
- [nclone/map_loader.py](nclone/nclone/map_loader.py) (145 lines) -- binary map data parsing, tile/entity instantiation
- [nclone/sim_config.py](nclone/nclone/sim_config.py) (29 lines) -- configuration dataclass
- [nclone/entity_classes/](nclone/nclone/entity_classes/) -- only 3 entity types in scope: toggle mine, exit switch, exit door (~300 lines). All other entities (gold, drones, thwumps, bounce blocks, launch pads, locked doors, etc.) are out of scope.
- [nclone/tile_definitions.py](nclone/nclone/tile_definitions.py) (234 lines) -- tile geometry tables (become Rust `const` arrays)
- [nclone/constants/](nclone/nclone/constants/) (~500 lines) -- physics constants, tile constants, entity types

### Code that stays in Python (~24,800 lines)

These systems read state from the Rust sim but contain no physics. They stay in Python:

**Reward Calculation (~7,000 lines)** -- reads ninja/entity state + queries graph caches:

- `gym_environment/reward_calculation/main_reward_calculator.py` (1,489 lines) -- PBRS reward computation
- `gym_environment/reward_calculation/pbrs_potentials.py` (1,885 lines) -- potential functions
- `gym_environment/reward_calculation/intermediate_goal_manager.py` (1,169 lines) -- curriculum entity repositioning
- `gym_environment/reward_calculation/reward_config.py` (745 lines) -- curriculum-aware config
- `gym_environment/reward_calculation/reward_constants.py` (510 lines) -- reward constants
- `gym_environment/reward_calculation/adaptive_pbrs.py` (570 lines) -- multi-path adaptive PBRS
- (+ 3 smaller files ~620 lines)

**Observation Processing (~4,100 lines)** -- constructs RL observation tensors from state:

- `gym_environment/base_environment.py` (4,104 lines) -- `step()`, `_get_observation()`, `_process_observation()`

**Graph Building + Reachability (~17,800 lines)** -- builds navigation graph, computes pathfinding features:

- `graph/reachability/graph_builder.py` (2,131 lines) -- sub-node graph construction
- `graph/reachability/path_distance_calculator.py` (2,035 lines) -- cached path distance queries
- `graph/reachability/pathfinding_algorithms.py` (1,451 lines) -- A*, Dijkstra, BFS
- `graph/reachability/shared_cache_builder.py` (965 lines) -- shared memory caches
- `graph/reachability/cost_functions.py` (632 lines) -- physics-aware edge costs
- (+ 15 other files ~10,500 lines)

### Code that is dropped entirely (rendering)

Not needed for training. Excluded from both Rust and the training Python path:

- `nplay_headless.py` rendering methods, `nsim_renderer.py`, `debug_overlay_renderer.py`, pygame initialization
- `NPlayHeadless._perform_grayscale_conversion()`, `_perform_rgb_conversion()`, `render()`

---

## The FFI Boundary: SimulatorState Snapshot

The central design decision is **what crosses the Rust-to-Python boundary**. A single flat struct per step minimizes FFI overhead:

### NinjaSnapshot (returned by Rust every tick)

All 50+ fields that Python reward/observation code currently reads from `sim.ninja`:

**Position/Velocity (8 floats):**

- `xpos`, `ypos`, `xspeed`, `yspeed`
- `xpos_old`, `ypos_old`, `xspeed_old`, `yspeed_old`

**State Machine (6 fields):**

- `state` (u8: 0-9), `airborn` (bool), `airborn_old` (bool)
- `walled` (bool), `wall_normal` (f64), `terminal_impact` (bool)

**Jump/Buffer State (7 fields):**

- `jump_duration` (i32), `jump_input_old` (i32)
- `jump_buffer`, `floor_buffer`, `wall_buffer`, `launch_pad_buffer` (i32 each)
- `last_jump_was_buffered` (bool)

**Collision Normals (8 floats):**

- `floor_normalized_x/y`, `ceiling_normalized_x/y`
- `floor_count`, `wall_count`, `ceiling_count` (u32 each, packed)

**Physics Parameters (3 floats):**

- `applied_gravity`, `applied_drag`, `applied_friction`

**Inputs (2 fields):**

- `hor_input` (i32), `jump_input` (i32)

**Temporal Tracking (5 fields):**

- `frames_airborne`, `state_change_frame`, `previous_state`
- `consecutive_floor_frames`, `consecutive_wall_frames`

**Game Events (3 fields):**

- `has_won` (bool), `has_died` (bool), `death_cause` (Option)

**Collectibles (2 fields):**

- `gold_collected` (u32), `doors_opened` (u32)

### EntitySnapshot (per entity, returned as Vec)

Only 3 entity types are in scope for the Rust port:

- **Toggle Mine**: `entity_type` (u8), `x` (f64), `y` (f64), `active` (bool), `state` (u8: 0=toggled/deadly, 1=untoggled, 2=toggling), `radius` (f64: varies by state)
- **Exit Switch**: `entity_type` (u8), `x` (f64), `y` (f64), `active` (bool: false = collected)
- **Exit Door**: `entity_type` (u8), `x` (f64), `y` (f64), `active` (bool: true = open after switch collected)

All other entity types (gold, drones, thwumps, bounce blocks, launch pads, locked doors, one-way platforms, lasers, death balls, etc.) are not ported. Levels using those entities are not supported by the native sim.

### SimulatorState (top-level return from tick)

```
SimulatorState {
    ninja: NinjaSnapshot,
    entities: Vec<EntitySnapshot>,
    tile_grid: &[u8; 44*25],      // shared reference, never changes mid-episode
    frame: u32,
    action_mask: [bool; 6],        // computed inside Rust (ninja.get_valid_action_mask)
}
```

Python receives this as a single PyO3 object. All observation construction, reward computation, and graph queries then operate on this snapshot without any further FFI calls.

---

## What Python Code Does With the Snapshot

### 1. Observation Construction (stays Python, ~153 dims)

`base_environment.py:_get_observation()` reads from the snapshot:

- **game_state (27 dims)**: Built from `NinjaSnapshot` fields -- velocity, state categories, airborne, buffers, normals, physics params, position, time_remaining. This is pure normalization math on the snapshot fields. Currently in `nplay_headless.py:get_ninja_state()`, will move to a Python function that takes `NinjaSnapshot`.
- **reachability_features (11 dims)**: Path distances, next-hop directions, mine SDF gradient, platform distances. Computed from **Python graph caches** using `ninja.xpos/ypos` from snapshot. No physics.
- **spatial_context (112 dims)**: 8x8 local tile grid (from `tile_grid`) + 8 nearest mines (from `entities` in snapshot). Pure spatial queries.
- **switch_states (25 dims)**: Locked door states from `entities` snapshot.
- **mine_sdf_features (3 dims)**: Precomputed SDF grid lookup using ninja position.
- **action_mask (6 dims)**: Directly from `SimulatorState.action_mask` (computed in Rust).
- **entity_positions (6 dims)**: Switch/exit positions from `entities` snapshot.

No duplication risk: observation code is pure Python math on the snapshot. It never calls back into Rust.

### 2. Reward Calculation (stays Python, ~7,000 lines)

`main_reward_calculator.py` reads from the snapshot:

- **Terminal rewards**: `ninja.has_won`, `ninja.has_died`, `ninja.death_cause`
- **PBRS potentials**: `ninja.xpos/ypos/xspeed/yspeed` + `ninja.jump_duration` fed into Python `path_calculator.get_path_cost()` which queries **Python-side graph caches**
- **Time penalty**: `frame` from snapshot
- **Milestone tracking**: Position deltas computed from snapshot

No physics simulation happens here. The reward code queries precomputed Python-side pathfinding caches (built from graph). No duplication risk.

### 3. Graph Building + Reachability (stays Python, ~17,800 lines)

The graph system needs from the simulator:

- **tile_grid** (once per level): 44x25 array of tile types 0-33. Used for sub-node generation and adjacency building.
- **entities** (once per level, updated when mines toggle): Entity positions and types for node classification, entity masking.
- **ninja position** (per frame): For node lookup and path distance queries.
- **ninja velocity** (per frame): For velocity-aware first-edge cost in pathfinding.

All of this is available from `SimulatorState`. The graph code does not call `tick()` or simulate physics. It builds a navigation graph from static level geometry and queries cached shortest paths.

**Phase 4 opportunity**: The graph building (sub-node generation, adjacency, BFS from goals) is the slowest per-level operation (~50-200ms). This is a strong candidate for Rust porting later. The Python reward/observation code would then call Rust graph functions instead of Python graph functions. But this is independent of the physics port.

---

## Duplication Risk Analysis

### Constants (HIGH risk -- must handle carefully)

Physics constants (`NINJA_RADIUS`, `GRAVITY_FALL`, `MAX_HOR_SPEED`, etc.) are used by:

- Rust physics simulation (core loop)
- Python observation code (normalization in `get_ninja_state`)
- Python graph code (edge cost functions, traversability)
- Python reward code (PBRS normalization)

**Solution**: Constants are defined in Rust as `pub const` values, exported to Python via a `nclone_native.constants` PyO3 submodule. Python code imports from there:

```python
# Before (Python constants):
from nclone.constants import NINJA_RADIUS, GRAVITY_FALL

# After (Rust-exported constants):
from nclone_native.constants import NINJA_RADIUS, GRAVITY_FALL
```

The existing `nclone/constants/` Python files become thin re-export wrappers (for backward compat):

```python
# nclone/constants/physics_constants.py (becomes a re-export shim)
try:
    from nclone_native.constants import *
except ImportError:
    # Fallback to Python definitions if native not installed
    NINJA_RADIUS = 10
    ...
```

### Tile Definitions (MEDIUM risk)

`TILE_GRID_EDGE_MAP`, `TILE_SEGMENT_ORTHO_MAP`, etc. are used by:

- Rust physics (segment creation during map loading)
- Python `TileConnectivityPrecomputer` (offline precomputation)
- Python `GraphBuilder` (sub-node validity checks via `SUBNODE_VALIDITY_TABLE`)

**Solution**: Same pattern -- Rust owns the canonical definitions, exports them via PyO3. The `TileConnectivityLoader` already loads from a precomputed binary file, so it doesn't need the raw definitions at runtime.

### Entity Types (LOW risk)

Only 3 entity type IDs are in scope for Rust: toggle mine, exit switch, exit door. These are also the only types used by the Python graph code for node classification (`EntityType.EXIT_SWITCH`, `EntityType.TOGGLE_MINE`, `EntityType.EXIT_DOOR`).

**Solution**: Rust defines an `EntityType` enum (`#[repr(u8)]`) with these 3 variants (plus an `Unknown` fallback), exports it to Python. The Python `EntityType` class becomes a re-export.

### Action Mask Logic (ZERO risk)

Currently in `ninja.py:get_valid_action_mask()`. This moves entirely to Rust. Python receives the computed mask in `SimulatorState.action_mask`. No duplication.

### get_ninja_state() Feature Extraction (ZERO risk -- Python only)

The 26-feature state normalization currently in `nplay_headless.py:get_ninja_state()` is **pure Python math** on ninja fields. It stays in Python, operating on `NinjaSnapshot`. The Rust sim has no awareness of RL features.

---

## Repo Structure

```
nclone/
  nclone/                         # Existing Python package
    constants/
      physics_constants.py        # Becomes re-export shim from nclone_native
      tile_constants.py           # Becomes re-export shim
      entity_types.py             # Becomes re-export shim
    gym_environment/              # UNCHANGED (reads from snapshot)
      base_environment.py         # step() calls NativePlayHeadless.tick()
      reward_calculation/         # UNCHANGED (pure Python + graph caches)
      mixins/                     # UNCHANGED (reachability features from graph)
    graph/                        # UNCHANGED (builds graph from tile_grid + entities)
      reachability/               # UNCHANGED (pathfinding, caching)
    native_headless.py            # NEW: Python wrapper around Rust sim
                                  #   Implements same API as NPlayHeadless
                                  #   Delegates tick/reset/state to nclone_native
    # DEPRECATED (kept for fallback):
    physics.py, ninja.py, nsim.py, entities.py, entity_classes/
    nplay_headless.py             # Kept for rendering/human mode only

  nclone-native/                  # NEW Rust crate
    Cargo.toml
    pyproject.toml                # maturin build config
    src/
      lib.rs                      # PyO3 module: exports Simulator, constants, types
      constants.rs                # All physics/tile/entity constants (single source of truth)
      types.rs                    # NinjaSnapshot, EntitySnapshot, SimulatorState
      ninja.rs                    # Ninja physics + state machine (~1,600 lines)
      simulator.rs                # Simulator.tick(), map loading (~400 lines)
      collision/
        mod.rs
        sweep.rs                  # sweep_circle_vs_tiles
        depenetration.rs          # 32-iteration closest-point loop
        spatial.rs                # gather_segments/entities from grid cells
      segments.rs                 # GridSegmentLinear, GridSegmentCircular
      entities/
        mod.rs                    # Entity trait + dispatch
        toggle_mine.rs            # Toggle mine (deadly/safe states, radius changes)
        exit_switch.rs            # Exit switch (collection trigger)
        exit_door.rs              # Exit door (opens when switch collected)
      map/
        loader.rs                 # Binary map data parsing
        tiles.rs                  # Tile segment factory
    tests/
      parity/                     # Frame-by-frame parity tests vs Python sim

  pyproject.toml                  # Updated: optional dep on nclone-native
```

---

## Data Flow Diagram

```
Per Step:
  Python gym env                    Rust nclone-native
  ─────────────────                 ──────────────────
  step(action)
    │
    ├─ action → (hor, jump) ──────► Simulator.tick(hor, jump)
    │                                  │ integrate()
    │                                  │ 4x { collide_vs_objects + collide_vs_tiles }
    │                                  │ post_collision()
    │                                  │ think()
    │                                  │ get_valid_action_mask()
    │                                  ▼
    │◄────── SimulatorState ─────── return snapshot
    │
    ├─ build observation (Python math on snapshot)
    │    ├─ game_state: normalize ninja fields (27 dims)
    │    ├─ reachability: query Python graph caches (11 dims)
    │    ├─ spatial_context: read tile_grid + entity positions (112 dims)
    │    ├─ switch_states: read entity states (25 dims)
    │    └─ action_mask: from snapshot (6 dims)
    │
    ├─ compute reward (Python, reads snapshot + graph caches)
    │    ├─ terminal check: has_won / has_died
    │    ├─ PBRS: path_calculator.get_path_cost(ninja_pos, goal)
    │    └─ time penalty, milestones
    │
    └─ return (obs, reward, terminated, truncated, info)

Per Level Load:
  Python                            Rust
  ──────                            ────
  load_map(map_data) ─────────────► Simulator.load(map_data)
                                       parse tiles, create segments
                                       create entities
                     ◄──────────── return tile_grid, entity_list

  build_graph(tile_grid, entities)  (Python-only, graph/ package)
    sub-node generation
    adjacency building
    BFS from goals → level cache
    mine SDF precomputation
```

---

## Phase 4: Graph Acceleration (Future)

The ~17,800 lines of graph/reachability code is the largest Python system. Key bottlenecks:

- **Level cache build** (~50-200ms per level): BFS from each goal to all nodes with physics-weighted edges. Embarrassingly parallel across goals.
- **Sub-node generation + adjacency** (~10-50ms): Iterates 44x25 tiles, checks 4 sub-nodes each, builds 8-connected edges.
- **Mine SDF computation** (~5-20ms): Grid-based signed distance field.

These are strong candidates for Rust + rayon parallelism in Phase 4. The Python reward/observation code would then call `nclone_native.build_level_cache()` instead of the Python graph builder. The reward/observation code itself stays in Python.

---

## Expected Performance Gains

- **Per-frame physics** (Phase 1-3): 50-200x faster (Python to Rust for tight numerical loops)
- **Vectorized training** (Phase 3): Additional 4-8x from in-process parallelism via `NativeVecEnv` (eliminating SubprocVecEnv IPC overhead)
- **Level precomputation** (Phase 4): 50-500x faster for graph building, tile connectivity, mine SDF
- **Net end-to-end RL throughput**: 200-1000x (physics speedup * parallelism * eliminated overhead)

The Python overhead for observation/reward per step is negligible (~0.1ms) compared to the current ~5-20ms per physics tick. After the Rust port, physics will be ~0.05ms and the Python observation/reward code becomes the bottleneck -- but at 0.1ms/step it supports >10,000 steps/sec per environment, which is sufficient for years of scaling.