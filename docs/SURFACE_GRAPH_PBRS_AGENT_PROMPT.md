# Agent Prompt: Surface Graph PBRS Precomputation for N++ RL Training

## Your Mission

You are building the precomputation pipeline that constructs a **velocity-augmented surface graph** for an N++ reinforcement learning agent. This graph is used to compute exact cost-to-go values (in frames) from every reachable state to the current objective. These values serve as the potential function Φ(s) in Potential-Based Reward Shaping (PBRS), providing a dense training signal for a FiGAR-PPO agent.

The core equation is: `F(s, s') = γΦ(s') − Φ(s)`, where Φ(s) = −d*(s) / d*_max (the negative normalized shortest-path distance from state s to the goal). This preserves the optimal policy exactly (Ng et al. 1999) as long as Φ(terminal) = 0.

**This is not an approximation.** The N++ simulation is fully deterministic with known physics constants, fixed geometry, and a grounded-jump-only constraint that decomposes all movement into surface-to-surface arcs. These properties make exact graph-based shortest-path computation tractable and strictly superior to any learned value function.

---

## Environment Specification

### Level Structure
- Grid: 42×23 tiles (1056×600 pixels), each tile is 24×24 pixels
- A 1-tile border of solid (type 1) tiles pads the level during simulation, giving an effective 44×25 tile area for collision
- Coordinate system: origin at top-left, X increases rightward, Y increases downward
- Tile data arrays exclude the padding border, so coordinates are offset by −1 tile (−24 pixels) from map_data coordinates

### Tile Types (38 total, indices 0–37)
- **Type 0**: Empty space (no collision)
- **Type 1**: Full solid block
- **Types 2–5**: Half tiles (top, right, bottom, left)
- **Types 6–9**: 45-degree slopes
- **Types 10–13**: Quarter moon (concave curves)
- **Types 14–17**: Quarter pipes (convex curves)
- **Types 18–25**: Mild slopes (various angles)
- **Types 26–33**: Steep slopes (various angles)
- **Types 34–37**: Glitched tiles (no collision, not used in real levels)

### Collision Geometry
Each tile type maps to a fixed, known set of collision segments. This mapping is stored in `segment_dic[(x,y)]` and never changes.
- **Linear segments**: Straight collision edges with orientation (surface normal)
- **Circular segments**: Curved collision surfaces with radius 24 pixels
- Physics gathers segments by iterating overlapped grid cells during collision resolution

### Agent (Ninja) Properties
- Circular collision shape, radius = 10 pixels
- 9 movement states: Immobile (0), Running (1), Ground Sliding (2), Jumping (3), Falling (4), Wall Sliding (5), Dead (6), Awaiting Death (7), Celebrating (8), Disabled (9)

### Physics Constants (all from physics_constants.py)
```
GRAVITY_FALL            = 0.06666666666666665   px/frame²
GRAVITY_JUMP            = 0.01111111111111111   px/frame²
GROUND_ACCEL            = 0.06666666666666665   px/frame²
AIR_ACCEL               = 0.04444444444444444   px/frame²
MAX_HOR_SPEED           = 3.333                 px/frame
MAX_JUMP_DURATION        = 45                    frames
DRAG_REGULAR            = 0.9933221725495059
DRAG_SLOW               = 0.8617738760127536
FRICTION_GROUND         = 0.9459290248857720
FRICTION_GROUND_SLOW    = 0.8617738760127536
FRICTION_WALL           = 0.9113380468927672
MAX_SURVIVABLE_IMPACT   = 6.0                   px/frame
MIN_SURVIVABLE_CRUSHING = 0.05
NINJA_RADIUS            = 10                     px
TILE_PIXEL_SIZE         = 24                     px
```

### Physics Integration Order (per frame)
1. Apply Drag: `velocity *= drag_coefficient`
2. Apply Gravity: `velocity_y += gravity`
3. Update Position: `position += velocity`
4. Collision Resolution: 4 physics substeps, each with 32 depenetration iterations
5. Post-Collision: Logical entity interactions, state updates
6. Think: Input processing and state transitions

### Death Conditions
- Impact death: hitting a surface at velocity > MAX_SURVIVABLE_IMPACT (6.0 px/frame)
- Impact formula: `impact_vel > MAX_SURVIVABLE_IMPACT - 4/3 * abs(normal_y)`
- Crushing death: compressed below MIN_SURVIVABLE_CRUSHING (0.05)
- Entity contact: touching mines, drones, thwumps, or other hazards
- Timeout: frame 20,000

### Jump Mechanics
- **Floor jump (flat ground)**: vy = −2.0, vx unchanged
- **Slope jump (downhill with input)**: vx = 2/3 × slope_x, vy = 2 × slope_y
- **Slope jump (uphill perpendicular)**: vx = 2/3 × slope_x, vy = 2 × slope_y (with speed reset)
- **Slope jump (default uphill)**: vy = −1.4
- **Wall jump (regular, not wall-sliding)**: vx = ±1.0 × wall_normal, vy = −1.4
- **Wall jump (from wall-slide state 5)**: vx = ±0.667 × wall_normal, vy = −1.0
- Jump hold: up to 45 frames of reduced gravity (GRAVITY_JUMP). After release or expiry, GRAVITY_FALL applies.

### The Grounded-Jump-Only Constraint
**This is the key structural property that makes the graph approach work.** The agent can only initiate jumps from surface contact (floor or wall). While airborne, the agent cannot gain new upward velocity. This means ALL trajectories decompose into surface-to-surface arcs: the agent leaves a surface, follows a deterministic ballistic arc with limited horizontal air control, and lands on another surface (or dies).

### Input Buffering
- Jump buffer: 5-frame window for jump inputs while airborne
- Floor buffer: 5-frame window after touching ground to still jump
- Wall buffer: 5-frame window after touching wall to wall jump
- Launch pad buffer: 4-frame window after touching launch pad

---

## Pipeline Overview

The pipeline has 7 stages. Each must be implemented correctly for the PBRS potential to be valid.

```
Stage 1: Surface Extraction
    Input: tile grid (42×23)
    Output: list of SurfaceSegments with classified type (FLOOR, WALL, SLOPE)

Stage 2: Surface Sampling
    Input: SurfaceSegments
    Output: list of SurfacePoints at regular intervals along segments

Stage 3: Velocity Augmentation
    Input: SurfacePoints
    Output: graph nodes = (SurfacePoint, velocity_bin) tuples

Stage 4: Edge Enumeration
    Input: graph nodes + physics simulator
    Output: directed weighted edges (source_node → dest_node, weight = frame_count)

Stage 5: Cost-to-Go Computation
    Input: graph + goal node sets
    Output: d*(node) for every node, for both switch and exit goals

Stage 6: Arc Cache Construction
    Input: all simulated trajectories from Stage 4
    Output: spatial hash mapping (x, y, vx, vy) → cost-to-go for airborne states

Stage 7: Runtime Evaluation API
    Input: agent state at each training frame
    Output: Φ(s) value for PBRS computation
```

---

## Stage 1: Surface Extraction

### What is a Surface

A surface is any contiguous run of collision segments where the ninja can maintain stable contact. There are three categories:

**Floor surfaces.** The ninja rests on top, supported against gravity. The surface normal has a significant upward (negative Y) component. These include:
- Horizontal tops of solid blocks (type 1)
- Top edges of half tiles (types 2–5)
- Slope surfaces (types 6–9, 18–33) where the normal is not too steep
- Upper surfaces of concave/convex curves (types 10–17)

**Wall surfaces.** The ninja contacts a vertical or near-vertical edge and can wall-slide. The surface normal is predominantly horizontal. Wall sliding occurs when the ninja is airborne, falling (vy > 0), and pressing toward the wall (input × wall_normal < 0). These include:
- Left/right edges of solid blocks
- Vertical edges of half tiles
- Steep slope surfaces where the normal is predominantly horizontal

**Slope surfaces or radial.** Floor surfaces with non-trivial normal vectors. The ninja can stand, run, and jump from slopes, but jump impulse and friction mechanics differ from flat ground. If a slope is steep enough that the ninja slides off under gravity, classify it as a wall instead of a floor.

### Surface Classification Rule

For each collision segment, compute the surface normal angle relative to gravity (straight down = positive Y direction):
- Normal with significant upward component (negative Y) → FLOOR or SLOPE
- Normal predominantly horizontal → WALL
- Normal pointing downward (positive Y) → CEILING (not a surface the ninja can stand on; skip for graph nodes, but ceilings still matter for collision during arc simulation)

### Segment Merging

Adjacent segments of the same type across tile boundaries should be merged into continuous surface runs. This avoids creating artificial segment boundaries at every tile edge. Two segments merge if they share an endpoint (or endpoints are within a small epsilon) and have compatible normals.

### Output Data Structure

```
SurfaceSegment:
    start_x, start_y: float       # segment start position (full map coordinates)
    end_x, end_y: float           # segment end position
    normal_x, normal_y: float     # surface normal (pointing away from solid, into free space)
    surface_type: enum             # FLOOR, WALL, SLOPE
    segment_id: int                # unique identifier
    length: float                  # arc length of segment
    tile_positions: list           # which tile cells this segment spans
    is_circular: bool              # linear or circular segment
    circle_center_x, circle_center_y: float  # only for circular segments
    circle_radius: float           # only for circular segments (24px for tile curves)
```

---

## Stage 2: Surface Sampling

### Sampling Resolution

Place graph node positions at regular intervals along each surface segment.

**Default resolution: 6px** (quarter-tile, slightly more than half the ninja radius of 10px). This yields approximately 2,000–4,000 surface sample points per typical level.

**Dense sampling at 3px** at critical points:
- Platform edges: within 12px of a surface segment endpoint (where the ninja can fall off)
- Slope transitions: where the surface normal changes discontinuously between adjacent segments
- Near interactive entities: within contact range of launch pads, bounce blocks, doors, switches, or the exit

### Surface Point Positioning

The ninja's center must be positioned at the surface point offset by NINJA_RADIUS (10px) along the surface normal. This is where the ninja's center would be when standing on / leaning against the surface.

For linear segments: `ninja_center = surface_point + 10 * normal`
For circular segments: compute the point on the curve, then offset along the local normal.

### Edge Flagging

Flag surface points within 12px of a segment endpoint as `is_edge = true`. These represent platform edges where the ninja can fall off, or wall tops/bottoms where behavior changes. Edge nodes receive denser sampling AND denser edge enumeration (Stage 4).

### Output Data Structure

```
SurfacePoint:
    x, y: float                # ninja center position when on this surface point
    normal_x, normal_y: float  # surface normal at this point
    surface_type: enum          # FLOOR, WALL, SLOPE
    segment_id: int             # which surface segment this belongs to
    is_edge: bool               # within 12px of segment endpoint
    surface_x, surface_y: float # the actual surface contact point (before normal offset)
    parameter_t: float          # position along segment [0, 1] for interpolation
```

---

## Stage 3: Velocity Augmentation

### Why Velocity Matters

Two agents at the same position with different velocities have vastly different cost-to-go values. An agent running rightward at MAX_HOR_SPEED toward a gap can clear it with a short jump. The same agent at rest must spend ~50 frames accelerating first. The cost-to-go difference can be hundreds of frames. A position-only graph produces directionally correct but magnitude-wrong potentials. Velocity augmentation fixes this.

### Node Definition

A graph node is a tuple: `(surface_point_index, velocity_bin_index)`.

**Floor/Slope velocity bins:** Discretize the tangential velocity component (horizontal velocity on floors, slope-projected velocity on slopes) into 15 uniformly spaced bins over [−MAX_HOR_SPEED, +MAX_HOR_SPEED]:

```
[-3.333, -2.857, -2.381, -1.905, -1.429, -0.952, -0.476, 0.0,
  0.476,  0.952,  1.429,  1.905,  2.381,  2.857,  3.333]
Bin width: ~0.476 px/frame
```

**Wall velocity bins:** Discretize vertical velocity (vy) into 12 bins over [−2.0, +5.5]:

```
Lower bound: wall jump impulse vy = −1.4 (with margin)
Upper bound: approaching MAX_SURVIVABLE_IMPACT (6.0)
Bin width: ~0.625 px/frame
```

### Node ID

```
node_id = surface_point_index * max_velocity_bins + velocity_bin_index
```

With ~3,000 surface points and ~15 velocity bins: **~45,000 nodes**. This is trivially small for Dijkstra.

### Reverse Lookup

Build a spatial index that maps (x, y, surface_type, velocity) to the nearest node ID. This is used at runtime to evaluate Φ for arbitrary agent states.

---

## Stage 4: Edge Enumeration via Physics Simulation

This is the core precomputation step and the most critical for graph quality. For each graph node, simulate all physically realizable transitions using the full physics engine to discover which other nodes are reachable and at what frame cost.

### CRITICAL: Use the Real Physics Engine

Every simulation must use the exact same physics engine used during RL training. Same integration order (drag → gravity → position → collision with 4 substeps × 32 depenetration iterations → state update). Same constants. Same collision resolution. The precomputed trajectories must match exactly what the agent will experience. No shortcuts, no simplified physics.

### Simulation Setup

For each node (surface_point, velocity_bin):
1. Place the ninja at the surface point's (x, y) position (already offset by NINJA_RADIUS along normal)
2. Set the ninja's velocity to the bin's representative value (bin center) in the surface-tangential direction
3. Set the appropriate contact state (grounded for floors, wall-sliding for walls)

### Action Category 1: Running Along Surface (Floor/Slope Only)

Hold horizontal input {−1, 0, +1} without jumping for k frames.

**Duration sweep:**
- Standard nodes: k ∈ {1, 2, 3, 5, 8, 12, 18, 24, 36, 48, 60} (11 values)
- Edge nodes (is_edge=true): additionally include k ∈ {4, 6, 7, 9, 10, 15} for finer coverage near platform edges

If the ninja is still on the surface after k frames, record the destination as a surface node. Edge weight = k frames. If the ninja falls off the surface during simulation, record the frame of departure and continue simulating the resulting fall until landing or death. If the ninja lands alive, record the full trajectory as an edge to the landing node with weight = total frames.

### Action Category 2: Floor/Slope Jumps

From floor/slope contact, press jump.

**Parameters to sweep:**
- Horizontal input during jump: {−1, 0, +1}
- Jump hold duration: see resolution tiers below
- Horizontal input after jump release: {same as during, opposite, neutral}

**Jump hold duration sweep resolution:**
- Standard nodes: {1, 3, 5, 8, 12, 16, 20, 25, 30, 35, 40, 45} (12 values)
- Edge nodes (is_edge=true): every integer from 1 to 45 (full 1-frame resolution, 45 values)

**For each combination, simulate the full arc:**
1. Frame 1: Apply the appropriate jump impulse based on surface type and movement direction:
   - Flat ground: vy = −2.0
   - Slope downhill with input: vx = 2/3 × slope_x, vy = 2 × slope_y
   - Slope uphill perpendicular: vx = 2/3 × slope_x, vy = 2 × slope_y (with speed reset)
   - Slope default uphill: vy = −1.4
2. Frames 2 through jump_duration: Apply GRAVITY_JUMP, horizontal AIR_ACCEL, drag
3. Frames after jump release: Apply GRAVITY_FALL, horizontal AIR_ACCEL (possibly changed direction), drag
4. Continue until the ninja contacts a surface alive or dies

If the ninja lands alive, create a directed edge from source to destination node. Edge weight = total frames of the arc.

**Mid-air input switching (edge nodes only):** For edge nodes, additionally enumerate trajectories where the horizontal input switches at frame k_switch ∈ {5, 10, 15, 20, 25, 30} during the jump hold phase. This captures precision air-control maneuvers critical near platform edges.

### Action Category 3: Wall Jumps

From wall-slide state, press jump.

**Wall jump types:**
- Regular wall jump (not in wall-slide state): vx = ±1.0 × wall_normal, vy = −1.4
- Wall-slide wall jump (state 5): vx = ±0.667 × wall_normal, vy = −1.0

**Parameters to sweep:**
- Wall jump type: {regular, slide}
- Horizontal input after jump: {toward wall, away from wall, neutral}
- Jump hold duration: same resolution tiers as floor jumps (standard vs edge nodes)

Simulate each full arc. Wall-jump arcs may land on another wall — this is expected and correct. The graph captures arbitrarily long wall-jump chains via path composition through Dijkstra.

### Action Category 4: Falling Off Edges

From surface nodes where is_edge=true, enumerate explicit fall-off transitions:
- Horizontal input during fall: {−1, 0, +1}
- Simulate the fall until landing or death
- Edge weight = frames from departure to landing

This is also discovered naturally during "running along surface" enumeration when the ninja runs past a segment endpoint, but explicit enumeration at edge nodes ensures coverage of all fall directions.

### Action Category 5: Launch Pad Trajectories

Launch pads (entity type 10) impart a velocity boost of (36/7) × (2/3) ≈ 3.429 px/frame in one of 8 fixed directions. The boost **replaces** the current velocity in the boosted components (it is not additive).

For each surface node within contact range of a launch pad (distance < NINJA_RADIUS + 6px):
1. Apply the launch pad boost velocity
2. Simulate until landing, with horizontal input sweep: {−1, 0, +1}
3. Additionally sweep jump hold durations if the ninja gains floor/wall contact mid-trajectory

These edges are high-value — launch pads may be the only viable path to certain level regions. Enumerate exhaustively.

### Action Category 6: Bounce Block Interactions

Bounce blocks (entity type 17) use spring-based physics. The outcome depends on approach velocity, contact angle, and block position.

Enumerate a representative set of approach velocities and contact angles for each bounce block. For each surface node within contact range:
1. Simulate the bounce interaction through the full physics engine
2. Record the resulting trajectory
3. If the ninja lands alive, create an edge

Accept that multi-bounce chains may not be fully captured. The RL agent learns to exploit these during training. The graph captures single-bounce trajectories as edges; chains compose via graph paths.

### Edge Pruning

After enumeration, prune:
- **Dominated edges:** For each (source, destination) pair, keep only the minimum-cost edge. If edge A goes X→Y in 30 frames and edge B goes X→Y in 45 frames, discard B.
- **Self-loops:** Edges where source = destination with weight > 0. Prune if shorter alternatives exist via other paths.
- **Death edges:** Trajectories where the simulation results in death are discarded entirely.
- **Out-of-bounds:** Trajectories exiting level boundaries (should not happen with the solid border, but verify).

### Recording Trajectories for Arc Cache

During edge enumeration, record the complete trajectory (every frame's position and velocity) for every edge, not just the endpoints. These trajectories are used in Stage 6 to build the arc cache for airborne state evaluation.

---

## Stage 5: Cost-to-Go Computation

### Backward Dijkstra

Given the directed weighted graph G = (V, E, w):
- V = velocity-augmented surface nodes
- E = edges from Stage 4
- w(e) = frame count for edge e

Run backward Dijkstra from the goal node set. The goal is a SET of nodes (the objective has a collision radius and can be reached from multiple surface positions at various velocities).

**Goal node identification:**
- Exit switch: all surface nodes where distance(surface_point, switch_position) < NINJA_RADIUS + 6px (switch radius)
- Exit door: all surface nodes where distance(surface_point, exit_position) < NINJA_RADIUS + 12px (exit radius)

All goal nodes receive d* = 0.

Run O(|E| log |V|) Dijkstra on the reversed graph. With ~45,000 nodes and ~2M estimated edges, runtime is under 1 second.

### Two Dijkstra Runs Per Level

1. **d*_switch[node]**: cost-to-go from each node to the exit switch. Used during Phase 0 (switch not yet collected).
2. **d*_exit[node]**: cost-to-go from each node to the exit door. Used during Phase 1 (switch collected).

### Unreachable Nodes

Nodes unreachable from the goal (disconnected components, one-way topology) receive d* = d*_max, where d*_max is the maximum finite d* in the graph. This produces Φ = Φ_min for unreachable states — a safe, conservative signal that points the PBRS gradient toward reachable regions.

### Locked/Trap Door Handling

If the level contains locked doors (type 6) or trap doors (type 8), the graph topology changes between phases (some edges appear/disappear when the switch is activated). Handle this with **two separate graphs** — one for Phase 0, one for Phase 1 — with different edge sets reflecting door states. Run Dijkstra once on each.

---

## Stage 6: Arc Cache Construction

### Purpose

During training, the agent spends significant time in airborne states (jumping, falling). Airborne states are NOT graph nodes. They must be evaluated by determining where the agent will land and looking up the cost-to-go at the landing node. The arc cache makes this fast.

### Building the Cache

During Stage 4, you recorded the complete trajectory for every edge. For each trajectory that includes an airborne phase (jumps, falls, launch pad arcs), extract every frame's state: (x, y, vx, vy). For each frame, compute:

```
cost_at_this_frame = remaining_frames_to_landing + d*(landing_node)
```

Store these in a spatial hash indexed by quantized (x, y, vx, vy):

```
Position bins: 12px resolution → 88 × 50 spatial cells
Velocity bins: 0.5 px/frame resolution → ~14 × ~24 velocity cells
Lookup: hash(x//12, y//12, vx_quantized, vy_quantized) → min cost-to-go
```

For each cell, store the **minimum** cost-to-go across all arc waypoints that pass through it. At runtime, if the cell is populated, return the cached value.

### Coverage Priority

The arc cache covers airborne states along optimal or near-optimal trajectories (those from graph edges). Exploratory states far from any cached arc fall back to forward simulation (see Stage 7).

---

## Stage 7: Runtime Evaluation API

### The Core Function

```python
def evaluate_phi(state, phase, precomputed_data) -> float:
    """
    Returns Φ(s, u) ∈ [-1, C_PHASE] for PBRS computation.
    Called at EVERY primitive frame during training.
    """
```

### Terminal States
Death, timeout, or success → return 0.0. Non-negotiable (PBRS constraint).

### Surface States (direct graph lookup)

If the agent is on a surface (floor, wall, slope):
1. Find the nearest surface node using the spatial index
2. Look up d* from the phase-appropriate Dijkstra map
3. Interpolate between adjacent velocity bins for smoothness:

```python
node_lo, node_hi, alpha = find_bracketing_velocity_nodes(x, y, surface_type, v_tangential)
cost = (1 - alpha) * d_star[node_lo] + alpha * d_star[node_hi]
phi_state = -cost / d_max
return phi_state + phi_phase
```

where phi_phase = 0.0 for Phase 0, C_PHASE (recommended: 0.5) for Phase 1.

### Airborne States (arc cache → forward simulation fallback)

If the agent is airborne:

**Step 1: Arc cache lookup (O(1)).** Hash the current (x, y, vx, vy) into the spatial hash. If the cell is populated, return the cached cost-to-go.

**Step 2: Forward simulation fallback.** If the cache misses, simulate multiple landing strategies:
- For each horizontal input {−1, 0, +1}:
  - If jump is currently held with frames remaining: simulate both "continue holding" and "release now"
  - Simulate until landing or death
  - If alive at landing, look up d*(landing_node) and compute total_cost = frames_to_land + d*(landing_node)
- Return Φ based on the minimum total_cost across all strategies

**Step 3: Pessimistic fallback (if forward simulation is too expensive).** Use the nearest-surface-below estimate:
```
Φ_fast = -(d*_nearest_surface_below + estimated_fall_frames) / d_max
```
Directionally correct but loses horizontal discrimination. Use only when per-frame simulation is budget-prohibitive.

### Smoothness Requirements

- **Position smoothness:** Bilinear interpolation between adjacent surface nodes along a segment (6px resolution is dense relative to max speed 3.3 px/frame)
- **Velocity smoothness:** Linear interpolation between adjacent velocity bins
- **Surface-to-air transition smoothness:** When the agent leaves a surface, the surface Φ and the first airborne Φ should be nearly identical (the agent has barely moved)
- **Phase transition smoothness:** The C_PHASE jump at switch collection is an intentional positive discontinuity, consistent with PBRS theory

### Performance Targets

- Surface states (graph lookup): < 1μs
- Airborne states (arc cache hit): < 5μs
- Airborne states (forward simulation): < 10ms
- Per-step overhead (all states averaged): < 100μs
- Memory per level: < 10 MB

---

## Entity Handling Rules

Entities are handled in three tiers based on whether they can be modeled in the static graph.

### Tier 1: Static (Included in Graph)

**Launch pads (type 10):** Fully static. Trajectories enumerated in Stage 4 (Action Category 5).

**One-way platforms (type 11):** Static directionality. Include as floor surfaces but prune edges that approach from the blocked direction.

**Regular doors (type 5):** Open when the ninja is within 10px, auto-close after 5 frames. The graph captures traversal naturally — the door opens as the agent approaches.

### Tier 2: State-Dependent (Phase-Based)

**Exit switch (type 4) and exit door (type 3):** Handled by the two-phase reward machine. Switch is the goal in Phase 0, exit is the goal in Phase 1.

**Locked doors (type 6):** Closed in Phase 0, open in Phase 1. Exclude edges through locked doors in the Phase 0 graph. Include them in the Phase 1 graph.

**Trap doors (type 8):** Open in Phase 0, closed in Phase 1. Opposite of locked doors.

**Toggle mines (types 1/21):** Assume initial state (safe/untoggled) for graph construction. The graph's edges should avoid paths through a mine's deadly-state radius using a conservative buffer of mine_radius + NINJA_RADIUS + 2px. Runtime mine awareness comes from observation-space features (Mine SDF), not the graph.

### Tier 3: Dynamic (NOT in Graph)

**Drones (types 14, 26):** Time-dependent patrol paths. The graph cannot capture drone timing. Option: add a cost penalty to edges whose trajectories overlap with a drone's sweep zone. The RL agent learns drone timing through experience.

**Death balls (type 25):** Agent-seeking, fully reactive. Ignored by the graph. The agent handles them via observation-space features.

**Thwumps (type 20):** Agent-reactive charging behavior. In resting position, treat non-deadly surfaces as walkable. Charge trajectories are not modeled. The agent learns thwump timing through RL.

**Bounce blocks (type 17):** Spring-based, state-dependent. Simplified enumeration in Stage 4 captures common single-bounce interactions. Multi-bounce chains are learned by the agent.

---

## Two-Phase Reward Machine

```
States:
    u0 = Phase 0: Switch not collected
    u1 = Phase 1: Switch collected, exit active
    u2 = Terminal: Exit reached (success)
    u_dead = Terminal: Death or timeout

Transitions:
    u0 →[ninja contacts exit switch] u1    (sparse reward: R_SWITCH = 1.0)
    u1 →[ninja contacts exit door]   u2    (sparse reward: R_EXIT = 5.0)
    any →[death]                     u_dead (sparse reward: R_DEATH = -0.5)
    any →[frame 20,000]             u_dead (sparse reward: 0)
```

### Phase-Dependent Potential

```
Phase 0 (switch not collected):
    Φ(s, u0) = -d*_switch[nearest_node(s)] / d*_switch_max + 0.0

Phase 1 (switch collected):
    Φ(s, u1) = -d*_exit[nearest_node(s)] / d*_exit_max + C_PHASE

Terminal:
    Φ(s, terminal) = 0.0
```

C_PHASE (recommended: 0.5) produces a positive shaping reward at switch collection, reinforcing it.

Normalization ensures Φ_state ∈ [−1, 0]. Combined with C_PHASE ∈ [0, 1], total Φ ∈ [−1, 1].

---

## PBRS Integration with FiGAR-PPO

PBRS is computed at every primitive frame (physics step), not at macro-action decision points.

```python
def compute_shaping_reward(state, next_state, phase, next_phase, gamma, alpha, precomputed_data):
    phi_current = evaluate_phi(state, phase, precomputed_data)
    phi_next = evaluate_phi(next_state, next_phase, precomputed_data)
    F_t = gamma * phi_next - phi_current
    return alpha * F_t
```

The telescoping property ensures this is mathematically equivalent to SMDP-level PBRS:
```
Σ_{i=t}^{t+k-1} γ^{i-t} × F_i = γ^k × Φ(s_{t+k}) − Φ(s_t)
```

### Reward Composition

```
r_total_i = r_sparse_i + α × F_i + β × r_exploration_i

where:
    r_sparse_i = reward machine output (R_SWITCH, R_EXIT, R_DEATH, or 0)
    F_i = γ × Φ(s_{i+1}) − Φ(s_i)
    r_exploration_i = RND intrinsic reward (separate value head)
```

### Discount Factor Consistency

The SAME γ must appear in:
1. PBRS formula: F_t = γ × Φ(s') − Φ(s)
2. GAE computation: δ_i = r_i + γ × V(s_{i+1}) − V(s_i)
3. Return computation: G_t = Σ γ^k × r_{t+k}

Recommended: γ = 0.99

### Tuning Guidance

```
Recommended α: start at 0.5, sweep [0.1, 0.3, 0.5, 1.0]
Recommended C_PHASE: 0.5, sweep [0.0, 0.25, 0.5, 1.0]

Target |α × F_t| per frame: ~0.001–0.005
Over ~500 frames to goal: cumulative ~0.5–2.5 (comparable to sparse rewards)

If value function loss oscillates → reduce α
If agent ignores sparse goals → reduce α
If agent makes no progress toward goals → increase α
```

---

## Data Structures Summary

```
PrecomputedPBRS:
    # Surface graph
    surface_points: array[N_surfaces] of SurfacePoint        # ~3,000 entries
    velocity_bins_floor: array[15] of float                   # bin centers
    velocity_bins_wall: array[12] of float                    # bin centers

    # Dijkstra distance maps
    d_star_switch: array[N_nodes] of float                    # ~45,000 entries
    d_star_exit: array[N_nodes] of float                      # ~45,000 entries
    d_max_switch: float
    d_max_exit: float

    # Arc cache for airborne evaluation
    arc_cache_spatial_hash: SpatialHash                       # (x, y, vx, vy) → cost

    # Spatial index for nearest-node lookup
    surface_spatial_index: GridIndex                           # (x, y) → nearby surface points
```

### Storage Estimates

```
Surface points:    3,000 × 32 bytes   = ~96 KB
Dijkstra maps:     45,000 × 2 × 4 bytes = ~360 KB
Arc cache:         ~500,000 entries × 8 bytes = ~4 MB
Spatial index:     88 × 50 cells × 16 bytes = ~70 KB
Total per level: ~5 MB
```

Serialize as compressed binary files. One per level, one per phase if locked/trap doors require separate graphs.

---

## Verification Requirements

### Graph Connectivity

For every level, verify:
- The switch position is reachable from the spawn point in the Phase 0 graph
- The exit position is reachable from the switch position in the Phase 1 graph
- If disconnected components exist, log which surface regions are isolated and why
- The fraction of reachable nodes should be >80% for typical levels (lower may indicate missing edges or entity-dependent paths)

### Arc Accuracy

For a sample of edges, independently verify by running the same trajectory through the physics engine and confirming the destination node and frame count match exactly. Zero tolerance — if there's a discrepancy, the simulation setup is wrong.

### Φ Smoothness

Walk the agent along a surface at constant velocity. Plot Φ vs position. It should be monotonically decreasing (toward the goal) with no discontinuities or spikes. Any spikes indicate graph artifacts or interpolation bugs.

### Arc Visualization

Overlay precomputed arcs on the level. Each arc connects a source surface point to a destination surface point. Verify arcs look physically reasonable — parabolic for jumps, appropriate apex heights for different hold durations.

### Precomputation Budget

- Edge enumeration + Dijkstra + arc cache: < 2 minutes per level on 8 cores
- For a 1,000-level curriculum: ~16.7 hours sequential, ~2 hours on 8 machines
- This is a one-time cost. Results are cached across all training runs.

---

## Summary of Non-Negotiable Requirements

1. **Use the real physics engine** for all edge enumeration simulations. No simplified physics.
2. **Φ(terminal) = 0** for all terminal states (death, timeout, success). PBRS guarantee depends on this.
3. **Same γ everywhere**: PBRS formula, GAE, returns. Using different values violates the PBRS guarantee.
4. **Complete coverage**: every state the agent can occupy during training must have a finite, well-defined Φ value. States without a potential produce NaN gradients or zero-information frames.
5. **Velocity augmentation**: the graph must be velocity-aware. Position-only graphs produce magnitude-wrong potentials that fail for momentum-dependent levels.
6. **Dense enumeration at edge nodes**: platform edges are where precision matters most. Full 1-frame resolution jump hold sweeps at is_edge=true nodes. This is where the graph is most likely to have coverage gaps.
7. **Two-phase support**: separate Dijkstra maps for switch and exit goals, with door-state-dependent edge sets when locked/trap doors are present.
8. **Smooth interpolation**: bilinear in position, linear in velocity. No discontinuities except at phase transitions.
