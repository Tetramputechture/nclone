# Value Precomputation Specification: Exact Cost-to-Go via Physics-Based Surface Graph

**Document Version:** 1.0  
**Last Updated:** February 13, 2026  
**Purpose:** Dense PBRS potential for FiGAR-PPO training on N++ platformer levels

---

## Table of Contents

1. [Design Rationale](#1-design-rationale)
2. [State Space Decomposition](#2-state-space-decomposition)
3. [Surface Extraction](#3-surface-extraction)
4. [Velocity-Augmented Surface Graph](#4-velocity-augmented-surface-graph)
5. [Edge Enumeration via Physics Simulation](#5-edge-enumeration-via-physics-simulation)
6. [Cost-to-Go Computation](#6-cost-to-go-computation)
7. [Mid-Air State Evaluation](#7-mid-air-state-evaluation)
8. [Entity Handling](#8-entity-handling)
9. [Two-Phase Reward Machine Integration](#9-two-phase-reward-machine-integration)
10. [Runtime Evaluation Pipeline](#10-runtime-evaluation-pipeline)
11. [Data Structures and Storage](#11-data-structures-and-storage)
12. [Computational Budget](#12-computational-budget)
13. [Integration with FiGAR-PPO](#13-integration-with-figar-ppo)
14. [Verification and Testing](#14-verification-and-testing)
15. [Physics Constants Reference](#15-physics-constants-reference)

---

## 1. Design Rationale

### Why Exact Precomputation

Four structural constraints of the N++ simulation eliminate the need for learned cost-to-go approximations:

1. **Fixed level geometry.** All levels are 42×23 tiles (1056×600 pixels) with known tile types (38 types, no novel tiles). The environment structure is fully observable before training begins.

2. **Deterministic physics.** Given a state and action, the next state is uniquely determined. There is no stochasticity to marginalize over.

3. **Grounded-jump-only constraint.** The agent can only initiate jumps from surface contact (floor or wall). It cannot gain vertical height while falling in mid-air. After the 45-frame jump hold expires or the jump button is released, gravity is the sole vertical force. This means all trajectories decompose into **surface-to-surface arcs** — the agent leaves a surface, follows a deterministic ballistic arc with limited horizontal air control, and lands on another surface.

4. **Known physics constants.** All gravity, friction, drag, acceleration, jump impulse, and death threshold values are fixed and known. Trajectory simulation is exact.

These constraints reduce the problem from "approximate a value function in a general MDP" to "compute shortest paths on a precomputed directed graph with exact edge weights." Every method surveyed in the research literature (quasimetric learning, contrastive RL, GNN heuristics, successor representations) exists to approximate something that is hard to compute exactly. Our constraints make exact computation tractable.

### PBRS Foundation

Potential-Based Reward Shaping (Ng et al. 1999) adds a shaping reward F(s, s') = γΦ(s') − Φ(s) to the environment reward. This preserves the optimal policy exactly, regardless of the quality of Φ, as long as Φ(terminal) = 0 for all terminal states. Using Φ(s) = −V*(s) (negative optimal cost-to-go) produces the strongest possible gradient, maximally accelerating convergence.

The requirement that drives this document: **every state the agent can occupy during training must have a finite, well-defined Φ value.** States without a potential value create undefined PBRS rewards, producing NaN gradients or zero-information frames. Complete coverage is non-negotiable.

### Edge Weight: Frame Count

Frame count is the natural edge weight. Every physics interaction — gravity work, friction loss, acceleration effort, wall-jump setup time — is paid in frames. A 200-frame path is strictly more expensive than a 100-frame path regardless of which mechanics are involved. Frame count naturally captures asymmetry: climbing costs more frames than falling, momentum-assisted jumps cost fewer frames than jumps from rest. No analytical energy-cost formula is needed — the simulator computes the exact frame cost for any trajectory.

---

## 2. State Space Decomposition

The agent's full MDP state includes position (x, y), velocity (vx, vy), contact state, jump duration remaining, and task phase. For the potential function, we decompose this into two regimes with different evaluation strategies.

### Surface States

The agent is in contact with a surface: standing on a floor, sliding on a wall, or resting on a slope. The agent can initiate actions from this state, including jumps. Surface states are the **nodes of the precomputed graph**.

A surface state is defined by: (surface_segment_id, position_along_segment, velocity_tangent_to_surface). Position along the segment is discretized at a fixed resolution. Velocity is discretized into bins. The combination gives a finite set of graph nodes with precomputed cost-to-go values.

### Airborne States

The agent is between surfaces: mid-jump, mid-fall, or in a ballistic arc after a wall jump. The agent cannot initiate a new jump (grounded-jump-only constraint) but can apply horizontal air control ({−1, 0, 1}) and, if currently holding jump with frames remaining, can choose to release or continue holding.

Airborne states are **not graph nodes**. They are evaluated at runtime by forward-simulating to the nearest surface landing and looking up the cost-to-go at the landing state. The grounded-jump-only constraint bounds the duration of any airborne phase — the agent must land on some surface (or die) within a finite number of frames determined by the level's vertical extent and physics constants.

### Terminal States

Death (impact, crushing, entity contact), timeout (frame 20,000), and success (exit reached after switch collection). All terminal states have Φ = 0 by the PBRS constraint. No precomputation needed.

---

## 3. Surface Extraction

### Input

The level's tile grid: 42×23 cells, each a known tile type (0–37). Tile collision geometry is fully defined — each tile type maps to a fixed set of linear and circular collision segments. An additional 1-tile border of solid (type 1) tiles pads the level during simulation, giving an effective 44×25 tile area for collision purposes.

### Surface Definition

A surface is any contiguous run of collision segments where the ninja (radius 10px, circular) can maintain stable contact. Three surface categories exist.

**Floor surfaces.** The ninja rests on top, supported against gravity. These include horizontal tops of solid blocks (type 1), top edges of half tiles (types 2–5), slope surfaces (types 6–9, 18–33), and the upper surfaces of concave/convex curves (types 10–17). The ninja's center is positioned at the surface point offset by its radius (10px) along the surface normal.

**Wall surfaces.** The ninja contacts a vertical or near-vertical edge and can wall-slide. These include the left/right edges of solid blocks, the vertical edges of half tiles, and steep slope surfaces where the surface normal is predominantly horizontal. Wall sliding occurs when the ninja is airborne, falling (vy > 0), and pressing toward the wall (input × wall_normal < 0).

**Slope surfaces.** These are floor surfaces with non-trivial normal vectors. The ninja can stand, run, and jump from slopes, but the jump impulse and friction mechanics differ from flat ground. Slopes with normals steep enough that the ninja slides off under gravity are walls, not floors.

### Surface Segment Catalog

For each tile position (tx, ty) with tile type t:

1. Retrieve the collision segments for tile type t from the known segment dictionary.
2. For each segment, classify as floor-capable, wall-capable, or neither, based on the surface normal angle relative to gravity. A segment is floor-capable if its normal has a significant upward (negative Y) component. A segment is wall-capable if its normal is predominantly horizontal.
3. Merge adjacent segments of the same type across tile boundaries into continuous surface runs.

### Sampling Along Surfaces

Sample graph node positions at regular intervals along each surface segment. The sampling resolution determines graph density and accuracy.

**Recommended resolution: 6px** (quarter-tile, slightly more than half the ninja radius of 10px). This gives approximately 2,000–4,000 surface sample points per typical level, depending on geometry complexity.

**Denser sampling** (3px) at critical points: platform edges (within 12px of a surface endpoint), slope transitions (where surface normal changes), and near interactive entities (launch pads, bounce blocks, doors, switches, exit).

**Surface point data structure:**

```
SurfacePoint:
    x: float              # position, full map coordinates
    y: float              # position, full map coordinates
    normal_x: float       # surface normal (pointing away from solid)
    normal_y: float       # surface normal
    surface_type: enum    # FLOOR, WALL, SLOPE
    segment_id: int       # which surface segment this belongs to
    is_edge: bool         # within 12px of segment endpoint (platform edge, wall top/bottom)
```

### One-Way Platforms

One-way platforms (type 11, 12×12px) are surfaces that only support the ninja from one approach direction. They are included as floor surfaces but their edges are tagged with directionality. Edges approaching from the blocked direction are excluded during edge enumeration.

### Bounce Blocks and Launch Pads

Bounce blocks (type 17, 9×9px) and launch pads (type 10, 6px radius) are not surfaces in the traditional sense. They are treated as **special transition sources** during edge enumeration (Section 5). The ninja can stand on top of a stationary bounce block (floor-like), but the spring interaction upon landing creates a unique transition. Launch pads impart a velocity boost in a fixed direction upon contact.

Surface points adjacent to bounce blocks and launch pads receive denser sampling and special-case edge enumeration.

---

## 4. Velocity-Augmented Surface Graph

### Why Velocity Matters

Two agents at the same surface position with different velocities have different cost-to-go values. An agent running rightward at MAX_HOR_SPEED (3.333 px/frame) toward a gap can clear it with a short jump. The same agent at rest must spend ~50 frames accelerating before the jump is viable, and the jump itself covers less horizontal distance. The cost-to-go difference can be hundreds of frames.

A position-only graph produces a potential that is directionally correct but magnitude-wrong. A velocity-augmented graph produces a potential that is accurate in both direction and magnitude, providing stronger PBRS gradients.

### Node Definition

A graph node is a tuple: (surface_point_index, velocity_bin_index).

**Velocity discretization for floor/slope surfaces:** Discretize the tangential velocity component into bins. On floors, this is horizontal velocity vx. On slopes, this is the velocity projected onto the slope direction.

```
Velocity bins for floors: 15 uniformly spaced bins over [-MAX_HOR_SPEED, +MAX_HOR_SPEED]
    = [-3.333, -2.857, -2.381, ..., 0.0, ..., +2.381, +2.857, +3.333]
    Bin width: ~0.476 px/frame
```

**Velocity discretization for wall surfaces:** Discretize vertical velocity vy. On walls, the ninja slides downward (vy ≥ 0 while wall-sliding) or arrives with upward velocity from a wall jump.

```
Velocity bins for walls: 12 bins over [-2.0, +5.5]
    Lower bound: wall jump impulse vy = -1.4 (with some margin)
    Upper bound: approaching MAX_SURVIVABLE_IMPACT (6.0)
    Bin width: ~0.625 px/frame
```

### Node Count Estimate

With ~3,000 surface points and ~15 velocity bins per point: **~45,000 nodes.** This is trivially small for Dijkstra (millisecond-scale computation).

### Node Identification

Each node receives a unique integer ID for graph indexing:

```
node_id = surface_point_index * max_velocity_bins + velocity_bin_index
```

A reverse lookup maps (x, y, surface_type, velocity) to the nearest node ID for runtime queries.

---

## 5. Edge Enumeration via Physics Simulation

This is the core precomputation step. For each graph node, we simulate all physically realizable transitions to discover which other nodes are reachable and at what frame cost.

### Simulation Setup

For each node (surface_point, velocity_bin):
1. Place the ninja at the surface point's (x, y) position, offset by NINJA_RADIUS along the surface normal.
2. Set the ninja's velocity to the bin's representative value (bin center) in the surface-tangential direction.
3. Set the appropriate contact state (grounded for floors, wall-sliding for walls).

### Action Categories

From each surface state, the agent can perform several categories of actions. Each is simulated using the full physics engine.

#### 5.1 Running Along Surface (Floor/Slope Only)

Hold a horizontal input {−1, 0, +1} without jumping, simulating for k frames.

**Duration sweep:** k ∈ {1, 2, 3, 5, 8, 12, 18, 24, 36, 48, 60} (11 values). Covers from single-frame adjustments to full runs across a surface. Longer durations are unnecessary — if the agent is still on the same surface after 60 frames, intermediate nodes along the surface provide the same information via graph path composition.

**For each (input, k):** Simulate k physics frames. If the ninja is still on a surface after k frames, record the destination as a surface node. The edge weight is k (frames). If the ninja falls off the surface during simulation (leaves the surface edge), record the frame of departure and the airborne state — this becomes a fall edge.

**Expected edges per node:** ~33 (3 inputs × 11 durations).

#### 5.2 Floor Jumps

From floor/slope contact, press jump. The ninja receives an initial vertical impulse and enters the jumping state with up to 45 frames of reduced gravity (GRAVITY_JUMP = 0.01111).

**Parameters to sweep:**
- Horizontal input during jump: {−1, 0, +1}
- Jump hold duration: {1, 3, 5, 8, 12, 16, 20, 25, 30, 35, 40, 45} (12 values, covering the full 1–45 frame range)
- Horizontal input after jump release: {same as during, opposite, neutral} (3 values, representing common air-control strategies)

**For each combination:** Simulate the full arc:
1. Frame 1: Apply jump impulse. For flat ground: vy = JUMP_FLAT_GROUND_Y = −2.0. For slopes, use the appropriate slope jump formula based on slope angle and movement direction.
2. Frames 2 through jump_duration: Apply GRAVITY_JUMP, horizontal AIR_ACCEL in the specified direction, drag.
3. Frames after jump release: Apply GRAVITY_FALL, horizontal AIR_ACCEL (possibly changed direction), drag.
4. Continue until the ninja contacts a surface or dies (impact velocity > MAX_SURVIVABLE_IMPACT).

**Record the result:** If the ninja lands on a surface alive, create a directed edge from the source node to the destination node (identified by landing surface point + landing velocity bin). Edge weight = total frames of the arc. If the ninja dies, no edge is created (this trajectory is not viable).

**Expected edges per node:** ~108 (3 inputs × 12 durations × 3 post-release inputs). Many will be pruned (landing on same surface, death, or duplicate destinations).

**Slope jump variants:** The jump impulse differs based on slope geometry and agent movement direction:
- Downhill with input: vx = 2/3 × slope_x, vy = 2 × slope_y (JUMP_SLOPE_DOWNHILL)
- Uphill perpendicular: vx = 2/3 × slope_x, vy = 2 × slope_y with speed reset (JUMP_SLOPE_UPHILL_PERP)
- Default uphill: vy = −1.4 (JUMP_SLOPE_UPHILL_FORWARD_Y)

Surface nodes on slopes enumerate jump edges using the slope-specific impulse formula, not the flat-ground impulse.

#### 5.3 Wall Jumps

From wall-slide state, press jump. The impulse direction depends on the wall jump type and the agent's input state.

**Regular wall jump** (not in wall-slide animation state): vx = ±1.0 × wall_normal, vy = −1.4. The ninja launches away from the wall with a strong horizontal and vertical impulse.

**Wall-slide wall jump** (in wall-slide state, state 5): vx = ±0.667 × wall_normal, vy = −1.0. Weaker impulse, the ninja peels off the wall more gently.

**Parameters to sweep:**
- Wall jump type: {regular, slide} (2 values)
- Horizontal input after jump: {toward wall, away from wall, neutral} (3 values)
- Jump hold duration: {1, 3, 5, 8, 12, 16, 20, 25, 30, 35, 40, 45} (12 values)

**For each combination:** Simulate the full arc from the wall, applying the appropriate impulse, then GRAVITY_JUMP during hold, GRAVITY_FALL after release, and AIR_ACCEL for horizontal input. Record landing surface and velocity.

**Wall-jump chains:** A wall jump arc may land on another wall rather than a floor. This is captured naturally — the destination node is on a wall surface, and subsequent edges from that wall node include further wall jumps. The graph encodes arbitrarily long wall-jump chains via path composition through Dijkstra.

**Expected edges per node (wall):** ~72 (2 types × 3 inputs × 12 durations). Many will terminate on the same wall (slide downward) or on the floor below.

#### 5.4 Falling Off Edges

When the agent runs past the endpoint of a surface (platform edge), it becomes airborne and falls. This is simulated during running-along-surface enumeration (Section 5.1) — when the simulation detects the ninja leaving the surface, it continues simulating the resulting fall until landing.

Additionally, from surface nodes near platform edges (within 12px, flagged as is_edge=true), enumerate explicit fall-off transitions:

**Parameters:**
- Horizontal input during fall: {−1, 0, +1}
- Direction of departure: {off the edge in the surface-tangential direction}

**Simulate** the fall until landing or death. Edge weight = frames from departure to landing.

#### 5.5 Launch Pad Trajectories

Launch pads (type 10) impart a velocity boost of (36/7) × (2/3) = ~3.429 px/frame in one of 8 fixed directions. When the ninja contacts a launch pad, its velocity is set to the boost vector (not added — it replaces the current velocity in the boosted components).

**For each surface node within contact range of a launch pad (distance < NINJA_RADIUS + 6px):** Simulate the launch pad trajectory:
1. Apply launch pad boost velocity.
2. Simulate until landing, with horizontal input sweep: {−1, 0, +1}.

These create high-value edges (potentially reaching distant surfaces cheaply) and must be enumerated exhaustively.

#### 5.6 Bounce Block Interactions

Bounce blocks (type 17) use spring-based physics. When the ninja contacts a bounce block, a force distribution system applies impulses based on the relative velocity and contact geometry. The bounce block also moves in response.

Bounce block interactions are harder to enumerate systematically because the outcome depends on approach velocity, contact angle, and the block's current position. For the precomputation graph:

**Simplified approach:** Enumerate a representative set of approach velocities and angles for each bounce block. Simulate each interaction through the full physics engine and record the resulting trajectory. Accept that some bounce block chains (multi-bounce sequences) will not be captured — the RL agent learns to exploit these during training.

**Contact states near bounce blocks** receive extra surface sampling points with wider velocity bin coverage.

#### 5.7 Edge Pruning

After enumeration, prune the edge set:

- **Dominated edges:** If edge A goes from node X to node Y in 30 frames, and edge B goes from X to Y in 45 frames, discard edge B. Keep only the minimum-cost edge for each (source, destination) pair.
- **Self-loops:** Edges where source = destination with weight > 0 are valid (representing time spent on a surface) but can be pruned if shorter alternatives exist via other paths.
- **Death edges:** Edges where the simulation results in death are discarded entirely.
- **Out-of-bounds:** Edges where the trajectory exits the level boundaries (should not happen with the solid border, but verify).

### Simulation Implementation

Each simulation uses the full N++ physics engine (the same engine used during RL training). This guarantees fidelity — the precomputed trajectories match exactly what the agent will experience. The simulation steps through the standard integration order: drag → gravity → position update → collision resolution (4 substeps, 32 depenetration iterations) → state update.

The simulation is run in "headless" mode: no rendering, no entity logic beyond tile collisions (entity handling is discussed separately in Section 8), no input buffering (direct state injection).

---

## 6. Cost-to-Go Computation

### Backward Dijkstra

Given the directed weighted graph G = (V, E, w) where V is the set of velocity-augmented surface nodes, E is the set of edges discovered by simulation, and w(e) is the frame count for edge e:

**Run backward Dijkstra from the goal node set.** The goal is a set of nodes, not a single node, because the switch/exit has a collision radius and can be reached from multiple surface positions at various velocities.

**Goal node identification:** Find all surface nodes (surface_point, velocity_bin) where the surface point is within contact range of the goal entity:
- Exit switch: distance(surface_point, switch_position) < NINJA_RADIUS + 6px (switch radius)
- Exit door: distance(surface_point, exit_position) < NINJA_RADIUS + 12px (exit radius)

All goal nodes receive cost-to-go = 0.

**Dijkstra execution:** Standard backward Dijkstra on the reversed graph. For each node v, compute d*(v) = minimum frame count from v to the nearest goal node. This runs in O(|E| log |V|) time. With ~45,000 nodes and ~2M edges (estimated), runtime is under 1 second.

### Two Goal Computations

Run Dijkstra twice per level:

1. **d*_switch[node]**: Cost-to-go from each node to the exit switch position. Used during phase 0 (switch not yet collected).
2. **d*_exit[node]**: Cost-to-go from each node to the exit door position. Used during phase 1 (switch collected, exit active).

### Unreachable Nodes

Some surface nodes may be unreachable from the goal (disconnected graph components, one-way topology). These receive d* = d*_max, where d*_max is the maximum finite d* in the graph. This produces Φ = Φ_min for unreachable states, which is a safe, conservative signal — the PBRS gradient will point toward reachable regions.

Verify that the switch and exit positions are themselves reachable from the majority of the level. If the goal is in an isolated component, the level may require entity interactions (doors, launch pads) not captured by static geometry. See Section 8.

### Velocity-Collapsed Lookup (Optional Optimization)

For faster runtime evaluation, compute a velocity-collapsed distance for each surface point:

```
d*_min[surface_point] = min over velocity_bins { d*[surface_point, velocity_bin] }
d*_max[surface_point] = max over velocity_bins { d*[surface_point, velocity_bin] }
```

The velocity-collapsed minimum gives a lower bound on cost-to-go (best-case velocity), useful for quick estimates. The full velocity-aware lookup is used for precise PBRS evaluation.

---

## 7. Mid-Air State Evaluation

This section addresses the critical requirement: the agent spends a significant fraction of training time in airborne states (jumping, falling, being launched). Every airborne state must have a well-defined Φ value.

### The Airborne Constraint

From the grounded-jump-only constraint, an airborne agent has limited control:
- **Horizontal input:** {−1, 0, +1}, applying AIR_ACCEL = 0.04444 px/frame² up to MAX_HOR_SPEED.
- **Jump hold release:** If currently holding jump with frames remaining, the agent can release (switching from GRAVITY_JUMP to GRAVITY_FALL). Once released or exhausted, jump cannot be re-initiated.
- **No vertical impulse.** The agent cannot gain new upward velocity while airborne. Gravity always wins.

This means every airborne trajectory terminates at a surface (floor, wall, or level boundary) within a bounded number of frames. The maximum airborne duration is bounded by the level's vertical extent divided by the minimum vertical acceleration:

```
Max fall height: ~600px (full level height)
Min vertical acceleration: GRAVITY_JUMP = 0.01111 px/frame²
Max fall time from top: ~330 frames (if holding jump the entire time)
Practical upper bound: ~200 frames (most arcs much shorter)
```

### Evaluation Strategy: Greedy Forward Simulation

From an airborne state s_air = (x, y, vx, vy, jump_held, jump_frames_remaining), compute cost-to-go as:

```
Φ(s_air) = -min over control strategies σ {
    frames_to_landing(σ) + d*[landing_node(σ)]
} / normalization_constant
```

Where each control strategy σ specifies the horizontal input and jump-release timing for each frame of the remaining arc.

**The strategy enumeration problem.** With 3 horizontal inputs and 2 jump decisions per frame, over ~100 frames, there are ~6^100 possible control sequences. Exhaustive enumeration is impossible.

**Practical approach: enumerate representative strategies.** The critical insight is that airborne control has low-dimensional structure. Horizontal air control is weak (AIR_ACCEL is small), so the horizontal landing position varies smoothly with input. Jump release timing determines the arc apex height and is a single scalar parameter. We can cover the space of useful trajectories with a small set of representative strategies:

**Strategy set for jump-holding states:**

```
For jump_release_time in {0, remaining/4, remaining/2, 3*remaining/4, remaining}:
    For horizontal_input in {-1, 0, +1}:
        Simulate: hold jump for jump_release_time frames with horizontal_input,
                  then fall with horizontal_input until landing.
        Record: (landing_surface_point, landing_velocity, total_frames)
```

This gives 15 strategies per airborne state. If the jump is already released (falling only), the jump_release_time dimension collapses and we have just 3 strategies (3 horizontal inputs).

**Strategy set for post-jump-release states (falling):**

```
For horizontal_input in {-1, 0, +1}:
    Simulate: fall with horizontal_input until landing.
    Record: (landing_surface_point, landing_velocity, total_frames)
```

**Extended strategies for mid-change maneuvers:**

```
For switch_time in {1/4, 1/2, 3/4 of estimated remaining time}:
    For input_before in {-1, 0, +1}:
        For input_after in {-1, 0, +1} where input_after ≠ input_before:
            Simulate: input_before for switch_time, then input_after until landing.
            Record landing.
```

This adds ~27 more strategies but captures the important case of "drift left to clear an obstacle, then drift right to reach the landing platform."

**Total strategies per airborne evaluation:** ~45. Each simulation is at most ~200 frames × ~1μs/frame = ~0.2ms. Total per airborne evaluation: ~9ms. This is acceptable for per-frame evaluation during training at moderate environment counts.

### Landing Resolution

When a strategy simulation results in landing on a surface:

1. Identify the nearest surface point to the landing position (within the 6px sampling resolution).
2. Identify the nearest velocity bin to the landing velocity.
3. Look up d*[landing_surface_point, landing_velocity_bin] from the precomputed graph.
4. Compute total cost: frames_to_landing + d*_at_landing.

If the simulation results in death (impact velocity > MAX_SURVIVABLE_IMPACT), that strategy is assigned cost = ∞ (excluded from the minimum).

If the simulation results in hitting a wall (wall contact), the agent enters wall-slide state — a surface state. Look up the wall surface node's d* value.

If no strategy results in a viable landing (all die or exit the level), assign Φ = Φ_min. This should be rare — the level has solid boundaries, so the agent must land somewhere.

### Performance Optimization: Arc Cache

During edge enumeration (Section 5), every jump and fall trajectory is already simulated. These arcs pass through many airborne states. Cache the intermediate states along each arc:

```
ArcCache entry:
    arc_id: int
    source_node: node_id
    destination_node: node_id
    total_frames: int
    waypoints: List[(frame_offset, x, y, vx, vy)]  # sampled every N frames
    cost_at_waypoint[i] = (total_frames - frame_offset) + d*[destination_node]
```

At runtime, when evaluating an airborne state, first check if it lies near a cached arc waypoint (within some distance threshold, e.g., 12px position and 0.5 px/frame velocity). If so, use the cached cost directly — O(1) lookup via spatial hash. Only fall back to forward simulation for states not near any cached arc.

The arc cache covers the most common airborne states (those along optimal or near-optimal trajectories). Exploratory states far from any precomputed arc require forward simulation, but these are also the states where PBRS precision matters least (the agent is clearly off-path, and any reasonable gradient pointing toward viable surfaces is sufficient).

### Arc Cache Data Structure

Store arc waypoints in a spatial hash indexed by (x_bin, y_bin, vx_bin, vy_bin):

```
Position bins: 12px resolution → 88 × 50 spatial cells
Velocity bins: 0.5 px/frame resolution → ~14 × ~24 velocity cells  
Total: ~1.5M cells (most empty)

Lookup: hash(x//12, y//12, vx_quantized, vy_quantized) → List[ArcCacheEntry]
```

For each cell, store the minimum cost-to-go across all arc waypoints that pass through it. At runtime, if the cell is populated, return the cached value. If not, fall back to forward simulation.

### Fallback: Pessimistic Estimate

If forward simulation is too expensive for a particular training configuration (very high environment count), use a fast pessimistic estimate:

```
Φ_fast(s_air) = -(d*_nearest_surface_below + estimated_fall_frames) / normalization
```

Where d*_nearest_surface_below is the cost-to-go at the surface directly below the agent's current position (simple vertical raycast into the tile grid), and estimated_fall_frames is a conservative estimate of fall time based on current vy and height above that surface.

This is directionally correct (points downward toward surfaces, which is where the agent must go) but loses horizontal discrimination. Use only when per-frame forward simulation is budget-prohibitive.

---

## 8. Entity Handling

Entities introduce state-dependent and time-dependent elements that complicate the static graph. They are handled in three tiers based on their impact on traversal cost.

### Tier 1: Static Transitions (Included in Graph)

**Launch pads (type 10):** Fully static. Position, direction, and boost magnitude are fixed. Launch pad trajectories are enumerated during edge construction (Section 5.5). Each launch pad creates high-value edges that may be the only viable path to certain regions.

**One-way platforms (type 11):** Static directionality. Included as floor surfaces with directional constraints on approach. Edges violating the approach direction are pruned.

**Regular doors (type 5):** Open when the ninja is within 10px, auto-close after 5 frames. Since the agent must be near the door to open it, the graph already captures traversal through doors — the agent passes through during normal movement. No special handling needed; the door opens as the agent approaches.

### Tier 2: State-Dependent (Modeled via Graph Phase or Penalty)

**Exit switch (type 4) and exit door (type 3):** Handled by the two-phase reward machine (Section 9). The switch is a goal in phase 0; the exit is a goal in phase 1. No special graph edges needed.

**Locked doors (type 6):** Require a switch activation (not necessarily the exit switch — some levels have multiple switches for locked doors). For levels with locked doors:
- Identify which switch controls each locked door.
- In phase 0 (before switch), graph edges through the locked door are excluded (door is closed).
- In phase 1 (after switch), graph edges through the locked door are included (door is permanently open).
- This requires separate graph construction for each phase, or dynamic edge masking.

**Trap doors (type 8):** Opposite of locked doors — start open, close permanently when their switch is activated. Handle similarly with phase-dependent edge masking.

**Toggle mines (types 1/21):** State changes when the ninja contacts them (safe → toggling → deadly → safe cycle). For graph precomputation, assume mines are in their initial state (typically safe/untoggled). The spatial cost penalty from the Mine SDF features (already in the observation space) provides runtime awareness of mine danger. The graph's edges avoid paths that would require passing through a mine's deadly-state radius, using a conservative buffer of mine_radius + NINJA_RADIUS + 2px.

### Tier 3: Dynamic (Not in Graph, Handled by RL Agent)

**Drones (types 14, 26):** Moving patrol hazards with deterministic but time-dependent positions. The precomputed graph cannot capture drone timing (the same path may be safe or lethal depending on when the agent traverses it). Handling:
- Exclude graph edges that pass through a drone's patrol path with zero clearance at any drone position. Alternatively, add a cost penalty proportional to the fraction of the edge's trajectory that overlaps with the drone's sweep zone.
- The RL agent learns drone timing through experience. The PBRS potential guides the agent toward the goal, and the agent learns to wait or dash past drones as part of its policy.

**Death balls (type 25):** Agent-seeking hazards. Fully reactive behavior — cannot be precomputed. The Mine SDF features in the observation space provide runtime avoidance gradient. The graph ignores death balls; the agent handles them.

**Thwumps (type 20):** Charge toward the agent when in line of sight. The non-deadly sides of thwumps can be used as surfaces (standing on top, pressing against the side). For the graph:
- Thwumps in their resting position are treated as static obstacles. Their non-deadly surfaces are included as walkable.
- Thwump charge trajectories are not modeled (time-dependent, agent-reactive).
- The agent learns to exploit thwump rides and avoid deadly faces through RL training.

**Bounce blocks (type 17):** Spring-based interaction with state (block position, velocity). The simplified enumeration from Section 5.6 captures common interactions. Multi-bounce chains are learned by the agent.

### Entity Summary

The graph provides exact cost-to-go for tile-geometry traversal and static entities. Dynamic entities are handled by the agent's policy, informed by observation-space features (mine SDF, spatial context mine overlay, drone positions). The PBRS potential guides the agent toward goals through the static geometry; the agent superimposes hazard avoidance on top.

---

## 9. Two-Phase Reward Machine Integration

### Reward Machine Definition

```
States:
    u0 = Initial (switch not collected)
    u1 = Switch collected (exit active)
    u2 = Terminal (exit reached)

Transitions:
    u0 →[agent contacts switch] u1   (sparse reward: R_SWITCH)
    u1 →[agent contacts exit]   u2   (sparse reward: R_EXIT)
    death/timeout → terminal         (sparse reward: 0, Φ = 0)
```

### Phase-Dependent Potential

```
Φ(s, u) = Φ_state(s, u) + Φ_phase(u)

Phase 0 (u0, switch not collected):
    Φ_state(s, u0) = -d*_switch[nearest_node(s)] / d*_switch_max
    Φ_phase(u0) = 0.0

Phase 1 (u1, switch collected):
    Φ_state(s, u1) = -d*_exit[nearest_node(s)] / d*_exit_max
    Φ_phase(u1) = C_PHASE    (recommended: 0.5)

Terminal (u2, death, timeout):
    Φ(s, u2) = 0.0           (PBRS constraint, non-negotiable)
```

**C_PHASE** provides a one-time potential jump at the moment of switch collection. When the agent touches the switch, the PBRS reward includes γΦ(s', u1) − Φ(s, u0) = γ(Φ_state(s', u1) + C_PHASE) − Φ_state(s, u0). The C_PHASE term produces a positive shaping reward at the phase transition, reinforcing switch collection. This is consistent with PBRS theory — the phase is part of the state, so the potential is still a function of state.

**Normalization** ensures Φ_state ∈ [−1, 0] regardless of level size. The normalization constant d*_max is the maximum finite cost-to-go in the respective graph. Combined with C_PHASE ∈ [0, 1], the total potential Φ ∈ [−1, 1].

### Phase-Dependent Graph Construction

**Option A: Single graph, two Dijkstra runs.** Build one velocity-augmented surface graph. Run backward Dijkstra from the switch position (phase 0 goal) and from the exit position (phase 1 goal) on the same graph. If locked/trap doors exist, the graph topology changes between phases (some edges appear/disappear), requiring Option B.

**Option B: Two graphs.** Build separate graphs for phase 0 and phase 1, with different edge sets reflecting door states. Run Dijkstra once on each. Higher memory (2× graph storage) but cleanly handles door interactions. Recommended when locked/trap doors are present.

---

## 10. Runtime Evaluation Pipeline

At every primitive timestep during FiGAR-PPO training, the environment must provide Φ(s) for the current state and Φ(s') for the next state to compute the PBRS reward F_t = γΦ(s') − Φ(s).

### Evaluation Algorithm

```python
def evaluate_phi(state, phase, precomputed_data):
    """
    Returns Φ(s, u) for PBRS computation.
    
    Args:
        state: (x, y, vx, vy, contact_state, jump_held, jump_frames_remaining)
        phase: 0 (pre-switch) or 1 (post-switch)
        precomputed_data: surface graph, Dijkstra distances, arc cache
    
    Returns:
        phi: float in [-1, C_PHASE]
    """
    # Terminal states
    if state.is_dead or state.is_timeout or state.is_success:
        return 0.0
    
    # Select phase-appropriate distance map
    d_star = precomputed_data.d_star_switch if phase == 0 else precomputed_data.d_star_exit
    d_max = precomputed_data.d_max_switch if phase == 0 else precomputed_data.d_max_exit
    phi_phase = 0.0 if phase == 0 else C_PHASE
    
    # Surface states: direct graph lookup with velocity interpolation
    if state.contact_state in (FLOOR, WALL, SLOPE):
        node = find_nearest_node(state.x, state.y, state.surface_type, 
                                  state.velocity_tangential)
        cost = d_star[node]
        
        # Interpolate between adjacent velocity bins for smoothness
        node_lo, node_hi, alpha = find_bracketing_velocity_nodes(
            state.x, state.y, state.surface_type, state.velocity_tangential)
        cost = (1 - alpha) * d_star[node_lo] + alpha * d_star[node_hi]
        
        phi_state = -cost / d_max
        return phi_state + phi_phase
    
    # Airborne states: check arc cache first, then forward simulate
    if state.contact_state == AIRBORNE:
        # Try arc cache (O(1) spatial hash lookup)
        cached_cost = arc_cache.lookup(state.x, state.y, state.vx, state.vy)
        if cached_cost is not None:
            phi_state = -cached_cost / d_max
            return phi_state + phi_phase
        
        # Forward simulate representative strategies
        min_cost = float('inf')
        for strategy in generate_strategies(state.jump_held, 
                                              state.jump_frames_remaining):
            landing, frames = simulate_strategy(state, strategy, 
                                                  precomputed_data.physics)
            if landing is not None and not landing.is_dead:
                landing_node = find_nearest_node(landing.x, landing.y,
                                                  landing.surface_type, 
                                                  landing.velocity_tangential)
                total_cost = frames + d_star[landing_node]
                min_cost = min(min_cost, total_cost)
        
        if min_cost == float('inf'):
            # No viable landing found — assign worst-case potential
            phi_state = -1.0
        else:
            phi_state = -min_cost / d_max
        
        return max(phi_state, -1.0) + phi_phase  # clamp to [-1, 0] before phase
```

### PBRS Reward Computation

```python
def compute_shaping_reward(state, next_state, phase, next_phase, 
                            gamma, alpha, precomputed_data):
    """
    Compute the total PBRS shaping reward for one primitive timestep.
    """
    phi_current = evaluate_phi(state, phase, precomputed_data)
    phi_next = evaluate_phi(next_state, next_phase, precomputed_data)
    
    F_t = gamma * phi_next - phi_current
    return alpha * F_t
```

### Smoothness

The potential function must be reasonably smooth for stable PBRS gradients. Discontinuities in Φ cause spike rewards that increase variance and can destabilize training.

**Position smoothness:** Surface nodes at 6px resolution are dense relative to agent speed (~3.3 px/frame max). Bilinear interpolation between adjacent surface nodes along a segment provides sub-pixel smoothness.

**Velocity smoothness:** Linear interpolation between adjacent velocity bins (as shown in evaluate_phi) provides smooth velocity dependence.

**Surface-to-airborne transition smoothness:** When the agent leaves a surface (jump or fall), the surface Φ value and the first airborne Φ value should be nearly identical (the agent has barely moved). The forward-simulation evaluation naturally provides this — the "simulate 0 remaining frames" strategy returns the current position's surface cost.

**Phase transition smoothness:** The C_PHASE jump occurs exactly once, at switch collection. This is an intentional discontinuity (positive reward spike) and is consistent with the sparse reward for switch collection.

---

## 11. Data Structures and Storage

### Per-Level Precomputed Data

```
PrecomputedPBRS:
    # Surface graph
    surface_points: array[N_surfaces] of SurfacePoint        # ~3000 entries
    velocity_bins_floor: array[15] of float                   # bin centers
    velocity_bins_wall: array[12] of float                    # bin centers
    
    # Dijkstra distance maps
    d_star_switch: array[N_nodes] of float                    # ~45,000 entries
    d_star_exit: array[N_nodes] of float                      # ~45,000 entries
    d_max_switch: float
    d_max_exit: float
    
    # Arc cache for airborne evaluation
    arc_cache_spatial_hash: SpatialHash                       # position+velocity → cost
    
    # Spatial index for nearest-node lookup
    surface_spatial_index: GridIndex                           # (x, y) → nearby surface points
```

### Storage Estimates

```
Surface points:    3,000 × 32 bytes   = ~96 KB
Dijkstra maps:     45,000 × 2 × 4 bytes = ~360 KB
Arc cache:         ~500,000 entries × 8 bytes = ~4 MB (variable, depends on arc density)
Spatial index:     88 × 50 cells × 16 bytes = ~70 KB

Total per level: ~5 MB (dominated by arc cache)
Without arc cache: ~0.5 MB per level
```

### Serialization

Save precomputed data as compressed binary files (one per level, one per phase if using separate graphs for locked/trap door levels).

```
Format: {level_id}_pbrs.npz
Contents:
    surface_points: structured array
    d_star_switch: float32 array
    d_star_exit: float32 array
    arc_cache_keys: int32 array (spatial hash keys)
    arc_cache_values: float32 array (costs)
    metadata: {d_max_switch, d_max_exit, n_surfaces, n_nodes}
```

---

## 12. Computational Budget

### Edge Enumeration (Dominant Cost)

```
Surface nodes:           ~3,000 points × ~15 velocity bins = ~45,000 nodes

Edges per floor node:    ~33 (running) + ~108 (jumps) + ~10 (falls) = ~151
Edges per wall node:     ~72 (wall jumps) + ~10 (sliding/falling) = ~82
Total edges to simulate: ~45,000 × ~120 (avg) = ~5,400,000 edge simulations

Per simulation:
    Average arc length:  ~60 frames
    Physics step cost:   ~1μs (headless, no rendering)
    Time per simulation: ~60μs

Total sequential:        5,400,000 × 60μs = ~324 seconds = ~5.4 minutes
```

### Parallelization

Edge simulations are independent (each starts from a known initial state). Parallelizable across CPU cores with linear speedup.

```
8 cores:  ~40 seconds
16 cores: ~20 seconds
64 cores: ~5 seconds
```

### Dijkstra

```
Nodes: ~45,000
Edges: ~2,000,000 (after pruning)
Dijkstra: O(E log V) = ~2M × 16 = ~32M operations
Runtime: <1 second
Two runs (switch + exit): <2 seconds
```

### Arc Cache Construction

```
Arcs from edge enumeration: ~2,000,000 (one per viable edge)
Waypoints per arc (sampled every 5 frames): ~12 average
Total waypoints: ~24,000,000
Spatial hash insertion: O(1) per waypoint
Construction time: ~10 seconds
```

### Total Per-Level Precomputation

```
Edge enumeration:    ~40 seconds (8 cores)
Dijkstra:            ~2 seconds
Arc cache:           ~10 seconds
I/O:                 ~2 seconds
────────────────────────────────
Total:               ~1 minute per level (8 cores)
```

This is negligible compared to RL training time (typically hours per level).

### Batch Processing

For a training curriculum of 1,000 levels:

```
Sequential: 1,000 × 1 min = ~16.7 hours
Parallel across 8 machines: ~2 hours
```

Precomputation is a one-time cost. Results are cached and reused across all training runs.

---

## 13. Integration with FiGAR-PPO

### Primitive-Level PBRS Application

PBRS is computed at every primitive frame (physics step), not at macro-action decision points. This is mathematically equivalent to SMDP-level PBRS via the telescoping property but provides denser value-function training signal.

```
For a macro-action of length k starting at time t:
    Primitive rewards at each frame i ∈ [t, t+k-1]:
        r_total_i = r_sparse_i + α × F_i + β × r_exploration_i
    
    where:
        F_i = γ × Φ(s_{i+1}, u_{i+1}) - Φ(s_i, u_i)
        r_sparse_i = reward machine output (R_SWITCH, R_EXIT, or 0)
        r_exploration_i = optional decaying exploration bonus

    Telescoping property:
        Σ_{i=t}^{t+k-1} γ^{i-t} × F_i = γ^k × Φ(s_{t+k}) - Φ(s_t)
        (SMDP-level shaping emerges naturally from primitive-level computation)
```

### GAE Computation

```
Primitive-level TD residuals:
    δ_i = r_total_i + γ × V(s_{i+1}) - V(s_i)

GAE advantages:
    A_i = Σ_{j=0}^{...} (γλ)^j × δ_{i+j}

Policy loss: computed only at macro-action decision points.
Value loss: computed at ALL primitive timesteps.
```

### Reward Magnitude Tuning

The shaping reward magnitude should be 0.1–0.5× the sparse reward magnitude to provide gradient without overwhelming the task signal.

```
Typical sparse rewards:
    R_SWITCH = 1.0
    R_EXIT = 5.0

Target |α × F_t| per frame:
    ~0.001–0.005 (shaping reward per frame)
    Over ~500 frames to goal: cumulative ~0.5–2.5 (comparable to sparse)

Recommended α:
    Start with α = 0.5, sweep [0.1, 0.3, 0.5, 1.0]
    Monitor: if value function loss oscillates, reduce α
    Monitor: if agent ignores sparse goals, reduce α
    Monitor: if agent makes no progress toward goals, increase α

C_PHASE:
    Recommended: 0.5 (half the switch-collection sparse reward)
    Sweep: [0.0, 0.25, 0.5, 1.0]
```

### γ Consistency

The same discount factor γ must appear in three places. Using different values violates the PBRS guarantee and introduces bias.

```
1. PBRS formula:        F_t = γ × Φ(s') - Φ(s)
2. GAE computation:     δ_i = r_i + γ × V(s_{i+1}) - V(s_i)
3. Return computation:  G_t = Σ γ^k × r_{t+k}

Recommended: γ = 0.99 (standard for long-horizon episodic tasks)
```

### Interrupt Handling

When a FiGAR macro-action is interrupted (contact state change: landing, wall contact, leaving surface), the macro-action ends early and the agent makes a new decision. The PBRS computation handles this naturally because it operates at the primitive level — each frame gets its own F_t regardless of macro-action boundaries. No special logic is needed at interrupt points.

---

## 14. Verification and Testing

### Correctness Tests

**Monotonicity test.** Along any shortest path in the graph (from a node to the goal), the cost-to-go should decrease monotonically. Verify: for each edge (u, v) on the shortest path tree, d*(u) > d*(v).

**Triangle inequality test.** For any three nodes a, b, c: d*(a, goal) ≤ d*(a, b) + d*(b, goal). Verify on a random sample of node triples.

**Terminal state test.** Verify Φ(terminal) = 0 for all terminal conditions: death, timeout, success. Run the environment to each terminal condition and check evaluate_phi returns 0.

**Phase transition test.** At the moment of switch collection, verify that the PBRS reward F_t includes the C_PHASE jump: F_t ≈ γ(Φ_exit(s') + C_PHASE) − Φ_switch(s) and is positive.

**Gradient direction test.** From any non-terminal surface state, verify that at least one action produces a positive F_t (the potential increases in the direction of the goal). If no action improves Φ, the state may be a local minimum in the potential — log these and verify they are genuine dead ends (disconnected from goal), not computation errors.

### Visualization

**2D heatmap.** Render d*_switch and d*_exit as 2D heatmaps overlaid on the level geometry. The goal should be bright (cost = 0) with smooth gradients extending outward. Walls and obstacles should create shadow regions with higher costs. Verify that the heatmap looks intuitively correct for the level's layout.

**Gradient vector field.** At each surface point, compute the gradient of Φ (direction of steepest cost decrease) and render as arrows overlaid on the level. The arrows should point toward the goal along viable paths, going around walls and through corridors. Verify arrows do not point into walls or toward dead ends.

**Arc visualization.** Overlay precomputed arcs (jump trajectories) on the level. Each arc connects a source surface point to a destination surface point. Verify arcs look physically reasonable — parabolic for jumps, appropriate apex heights for different hold durations.

### Performance Benchmarks

**Precomputation time.** Measure wall-clock time for edge enumeration, Dijkstra, and arc cache construction. Should be < 2 minutes per level on 8 cores.

**Runtime Φ evaluation time.** Measure average and P99 latency for evaluate_phi per frame during training. Targets:
- Surface states (graph lookup): < 1μs
- Airborne states (arc cache hit): < 5μs
- Airborne states (forward simulation): < 10ms
- Per-step overhead (all states averaged): < 100μs

**Memory.** Measure total memory consumption of precomputed data per level. Target: < 10 MB.

---

## 15. Physics Constants Reference

All values from `physics_constants.py`. Used throughout edge simulation.

```
# Gravity
GRAVITY_FALL            = 0.06666666666666665   px/frame²
GRAVITY_JUMP            = 0.01111111111111111   px/frame²

# Acceleration
GROUND_ACCEL            = 0.06666666666666665   px/frame²
AIR_ACCEL               = 0.04444444444444444   px/frame²

# Speed limits
MAX_HOR_SPEED           = 3.333                 px/frame
MAX_JUMP_DURATION        = 45                     frames

# Drag and friction
DRAG_REGULAR            = 0.9933221725495059
DRAG_SLOW               = 0.8617738760127536
FRICTION_GROUND         = 0.9459290248857720
FRICTION_GROUND_SLOW    = 0.8617738760127536
FRICTION_WALL           = 0.9113380468927672

# Death conditions
MAX_SURVIVABLE_IMPACT   = 6.0                   px/frame
MIN_SURVIVABLE_CRUSHING = 0.05

# Agent
NINJA_RADIUS            = 10                     px
TILE_PIXEL_SIZE         = 24                     px

# Level dimensions
MAP_TILE_WIDTH          = 42                     tiles
MAP_TILE_HEIGHT         = 23                     tiles
FULL_MAP_WIDTH_PX       = 1056                   px
FULL_MAP_HEIGHT_PX      = 600                    px
MAX_FRAMES              = 20000                  frames

# Jump impulses
JUMP_FLAT_GROUND_Y      = -2.0                  px/frame
JUMP_WALL_REGULAR_X     = 1.0                   px/frame (× wall_normal)
JUMP_WALL_REGULAR_Y     = -1.4                  px/frame
JUMP_WALL_SLIDE_X       = 0.6667                px/frame (× wall_normal)
JUMP_WALL_SLIDE_Y       = -1.0                  px/frame
JUMP_SLOPE_DOWNHILL_X   = 0.6667                (× slope_x)
JUMP_SLOPE_DOWNHILL_Y   = 2.0                   (× slope_y)
JUMP_SLOPE_UPHILL_PERP_X = 0.6667               (× slope_x)
JUMP_SLOPE_UPHILL_PERP_Y = 2.0                  (× slope_y)
JUMP_SLOPE_UPHILL_FORWARD_Y = -1.4              px/frame

# Launch pad
LAUNCH_PAD_BOOST        = 5.142857               px/frame (36/7)
LAUNCH_PAD_BOOST_SCALAR = 0.6667                 (2/3)

# Entity radii
EXIT_DOOR_RADIUS        = 12                     px
EXIT_SWITCH_RADIUS      = 6                      px
TOGGLE_MINE_RADII       = {untoggled: 3.5, toggling: 4.5, toggled: 4.0}  px

# One-way platform
ONE_WAY_PLATFORM_SEMI_SIDE = 12                  px (half-width of 12×12 square)

# Bounce block
BOUNCE_BLOCK_SIZE       = 9                      px (half-side of square)

# Thwump
THWUMP_SEMI_SIDE        = 9                      px
THWUMP_FORWARD_SPEED    = 2.857                  px/frame (20/7)
THWUMP_BACKWARD_SPEED   = 1.143                  px/frame (8/7)
THWUMP_ACTIVATION_RANGE = 38                     px

# Drone
DRONE_RADIUS            = 7.5                    px
MINI_DRONE_RADIUS       = 4                      px

# Input buffering
JUMP_BUFFER_WINDOW      = 5                      frames
FLOOR_BUFFER_WINDOW     = 5                      frames
WALL_BUFFER_WINDOW      = 5                      frames
LAUNCH_PAD_BUFFER       = 4                      frames
```

---

## Appendix A: Why Learned Approaches Are Unnecessary Given These Constraints

For reference and future architectural decisions, this section summarizes why learned cost-to-go approximations (quasimetric learning, contrastive RL, GNN heuristics, demo-based regression) are superseded by exact precomputation for this specific problem.

| Property | Exact Graph | Learned (Best Case) |
|---|---|---|
| Accuracy | Exact | Approximate |
| Asymmetry handling | By construction (gravity in simulation) | Must be architecturally enforced (IQE, quasimetric head) |
| Training data needed | None | Demos from many levels |
| Per-level precompute | ~1 min (8 cores) | ~0 (if universal) |
| Cross-level generalization | Run algorithm again on new level | Architecture-dependent |
| Failure mode | None (exact) | Approximation error, distribution shift |
| Mid-air state coverage | Forward-sim (exact) | Interpolation (approximate) |
| Velocity dependence | Exact via velocity-augmented nodes | Learned from data (approximate) |
| PBRS guarantee strength | Φ = −V* (optimal potential) | Φ ≈ −V̂ (approximate, weaker gradient) |
| Implementation complexity | Graph algorithms (well-understood) | Neural architecture + training pipeline |

The only advantage a learned approach retains is **amortized cost across many levels** — train once, deploy everywhere without per-level precomputation. At ~1 minute per level, this amortization provides negligible benefit relative to the hours of RL training that follow.

A learned universal V_nn may still have value as a **warm-start or fallback** for levels where precomputation has not yet completed, or as an additional observation feature for the agent (V_nn gradient provides physics-aware directional guidance complementing the geometric A* features). But for the PBRS potential itself, exact precomputation is strictly superior.

---

## Appendix B: Relationship to Existing Observation Space Features

The precomputed PBRS potential complements (does not replace) the existing observation space features described in OBSERVATION_SPACE_DOCUMENTATION.md v2.0.

**Features that overlap with PBRS:**
- `path_dist_to_switch_norm` / `path_dist_to_exit_norm` (reachability indices 4–5): These are geometric A* distances, which the PBRS potential supersedes with physics-aware frame-count distances. However, these remain valuable as agent observation features because they provide topology-aware navigation signal directly to the policy network.
- `next_hop_direction` (reachability indices 13–14): Geometric A* direction. Could be augmented with a Φ-gradient-derived direction for physics-aware guidance (see earlier discussion on V_nn gradient as observation feature).

**Features that complement PBRS:**
- Local tile grid (64 dims): Provides obstacle awareness for collision avoidance. PBRS provides goal-directed gradient; the tile grid provides local situational awareness.
- Mine SDF features (3 dims): Provides hazard avoidance gradient. PBRS provides goal-seeking gradient. The agent combines both.
- `switch_activated` (reachability index 12): This IS the reward machine phase indicator. It determines which Dijkstra distance map to use for Φ evaluation.

**PBRS does not enter the observation space.** The potential Φ and shaping reward F_t are computed in the reward pipeline, not the observation pipeline. The agent's policy network does not see Φ directly — it sees the observation features and learns to maximize total reward (which includes the shaped component). This separation is intentional: the agent should not overfit to the potential function.

**Optional: Φ-gradient as observation feature.** As discussed in earlier design sessions, the gradient of Φ at the agent's position could be added as an observation feature (2 dims: direction of steepest Φ increase). This provides physics-aware directional guidance to the policy network, complementing the geometric A* next_hop_direction. Implementation: finite-difference Φ evaluation at (x ± ε, y ± ε), cost is 4 additional evaluate_phi calls per step.
