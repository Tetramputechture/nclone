# Implementation Plan: Dense Reward Shaping via Physics-Informed PBRS + Reward Machine

## Goal

Replace sparse-only rewards (switch collection, exit reached) with a dense reward signal that provides per-timestep feedback. The system has three components:

1. **Energy-cost reachability graph** — precomputed per level, encoding physics-aware traversal costs between platform positions
2. **Reward machine (RM)** — 3-state automaton tracking switch→exit task progression
3. **Potential-based reward shaping (PBRS)** — dense reward derived from graph distances, applied at primitive timestep level within FiGAR-PPO

The PBRS guarantee: F(s,s') = γΦ(s') − Φ(s) preserves the optimal policy exactly (Ng et al. 1999). The potential Φ(s) = −d*(s, goal) where d* is minimum energy-cost path on the reachability graph.

---

## Component 1: Energy-Cost Reachability Graph

### What to build

A directed weighted graph G = (V, E, w) precomputed once per level where:
- **Vertices** = viable positions along surfaces (floors, walls, slopes, ceiling ledges)
- **Edges** = physically achievable transitions (walks, jumps, falls, wall-jumps, launch-pad trajectories)
- **Weights** = energy cost of traversal (NOT distance)

### Graph construction pipeline

**Step 1: Surface extraction.** Parse the level's tile data (42×23 grid, 24px cells) and identify all walkable/standable surfaces. A surface is any continuous run of tiles where the ninja (10px radius circle) can rest. Include:
- Floor surfaces (top edges of solid tiles, slopes)
- Wall surfaces (vertical edges where wall-sliding is possible)
- Ceiling surfaces (where ceiling-hugging or specific jumps apply)
- One-way platforms, bounce blocks (as special surface types)

Each surface node stores: position (x, y), surface normal, surface type.

Sample nodes along surfaces at intervals of ~12-24px (1-2 ninja radii). Denser sampling at slope transitions, platform edges, and near interactive entities.

**Step 2: Edge computation via physics simulation.** For each pair of nodes within plausible range, determine if a direct transition exists. Use the known physics constants to compute trajectories:

**Walk edges** (same surface): Cost = friction_loss × horizontal_distance. Use `FRICTION_GROUND = 0.9459` and `DRAG_REGULAR = 0.9933` from physics_constants.py. These are cheap, bidirectional (but asymmetric cost on slopes).

**Jump edges**: Simulate jump trajectories from each surface node using:
- Flat ground jump: initial vy = -2.0 px/frame (`JUMP_FLAT_GROUND_Y`)
- Max jump duration: 45 frames (`MAX_JUMP_DURATION`) at gravity 0.01111 px/frame² (`GRAVITY_JUMP`)
- Post-jump fall: gravity 0.06667 px/frame² (`GRAVITY_FALL`)
- Horizontal: ±`AIR_ACCEL` (0.04444) up to `MAX_HOR_SPEED` (3.333)
- Check for intermediate tile collisions using `sweep_circle_vs_tiles()` logic

For each launch point, sweep across jump durations (1-45 frames) and horizontal inputs (-1, 0, 1) to find reachable landing positions. Each valid trajectory becomes an edge.

**Wall-jump edges**: From wall-slide positions:
- Regular wall jump: vx = ±1.0 × wall_normal, vy = -1.4 (`JUMP_WALL_REGULAR_X/Y`)
- Slide wall jump: vx = ±0.667 × wall_normal, vy = -1.0 (`JUMP_WALL_SLIDE_X/Y`)
- Chain wall jumps by checking if trajectory reaches another wall surface

**Fall edges**: From platform edges, simulate falling with horizontal control. Mark edges where landing impact > 6 px/frame (`MAX_SURVIVABLE_IMPACT`) as **lethal** (infinite cost or excluded).

**Launch pad edges**: From launch pad positions, compute trajectory with boost = 36/7 px/frame in pad direction, scaled by 2/3 (`JUMP_LAUNCH_PAD_BOOST_SCALAR/FACTOR`). These inject free energy — assign zero or negative cost.

**Slope jump edges**: Use the slope jump formulas from sim_mechanics_doc.md (downhill, uphill perpendicular, default uphill variants).

**Step 3: Edge weight formula.** For each edge, compute energy cost as:

```
w(e) = gravity_work + friction_work + control_work

where:
  gravity_work = g × Δy  (positive going up, negative going down)
                 g = GRAVITY_FALL = 0.06667
  friction_work = μ × g × horizontal_distance
                  μ depends on surface type
  control_work = acceleration_effort × frames_of_active_input
```

For practical purposes, a simpler proxy that correlates well:

```
w(e) = α × frames_required + β × max(0, Δy) + penalty_if_near_fall_damage_threshold
```

where `frames_required` is the number of simulation frames the trajectory takes (directly measures time/effort), `Δy` penalizes upward movement (gravity opposition), and the fall-damage penalty adds cost for trajectories landing near the 6 px/frame threshold. Recommended: α=1.0, β=0.5. Tune these.

The frame-count proxy is attractive because it's trivially computed from trajectory simulation and naturally captures that harder maneuvers take more frames.

**Step 4: Precompute shortest paths.** Run backward Dijkstra from:
- Switch position → yields `d*_switch(v)` for all nodes
- Exit position → yields `d*_exit(v)` for all nodes

Store both distance maps. This is O(|E| log |V|), milliseconds for typical levels.

### Where this fits in the codebase

Look for existing graph/reachability infrastructure:
- `nclone/nclone/graph/reachability/` — likely has graph construction code already
- `nclone/nclone/graph/reachability/feature_computation.py` — computes reachability features, referenced in observation space docs
- The observation space already has A*-derived features (`next_hop_direction`, `path_dist_to_switch_norm`, etc.) — this means **a reachability graph already exists**

**Key action**: Examine the existing graph to determine:
1. Does it already encode edge weights? Are they distance-based or energy-based?
2. Does it already run Dijkstra/A* from switch and exit?
3. Can we add energy-cost edge weights to the existing graph, or do we need a parallel graph?

If the existing graph uses uniform or distance-based weights, the primary work is **re-weighting edges with energy cost** and storing the resulting Dijkstra distance maps for use by the reward function.

---

## Component 2: Reward Machine

### What to build

A 3-state finite automaton that tracks task phase:

```
States: u0 (start) → u1 (switch collected) → u2 (exit reached, terminal)

Transitions:
  u0 →[agent touches switch] u1    (reward: R_switch)
  u1 →[agent touches exit]   u2    (reward: R_exit)

No other transitions. Death resets to u0.
```

### Implementation

```python
class RewardMachine:
    def __init__(self):
        self.state = 0  # u0

    def step(self, events: set) -> float:
        """Returns sparse reward for this transition."""
        if self.state == 0 and 'switch_collected' in events:
            self.state = 1
            return R_SWITCH  # e.g., 1.0
        if self.state == 1 and 'exit_reached' in events:
            self.state = 2
            return R_EXIT    # e.g., 5.0
        return 0.0

    def reset(self):
        self.state = 0
```

**Observation augmentation**: Append the RM state (0 or 1; state 2 is terminal) to the observation. This is already partially handled — the observation space has `switch_activated` as a binary feature in reachability features (index 12). Verify this is passed through correctly. The RM state makes the problem fully Markov with respect to the task structure.

### Where this fits

- The gym environment wrapper (likely in `nclone/nclone/gym_environment/`) already tracks switch activation via `switch_activated`
- The environment already emits events when switch is collected and exit is reached
- The RM is a thin wrapper that formalizes what likely already exists implicitly in the reward function
- Look for the existing reward computation in the environment's `step()` method

---

## Component 3: Phase-Dependent PBRS

### Core formula

At every primitive timestep t:

```
F_t = γ · Φ(s_{t+1}, u_{t+1}) − Φ(s_t, u_t)
```

where:

```
Φ(s, u) = Φ_state(s, u) + Φ_phase(u)

Φ_state(s, u=0) = −d*_switch(nearest_node(s.x, s.y))   # guide toward switch
Φ_state(s, u=1) = −d*_exit(nearest_node(s.x, s.y))     # guide toward exit

Φ_phase(u=0) = 0.0
Φ_phase(u=1) = C_phase    # bonus for having collected switch (e.g., 0.5)
Φ_phase(u=2) = 0.0        # MUST be 0 at terminal states
```

**Critical constraint**: Φ(terminal_state) = 0. This includes death states and the u=2 success state. Violating this introduces bias that can alter the optimal policy.

### Normalization

Normalize Φ_state values to [−1, 0]:

```python
max_dist = max(d*_switch.values())  # or d*_exit, depending on phase
Φ_state_normalized(s, u) = −d*(s, target(u)) / max_dist
```

This keeps F_t in a bounded, predictable range regardless of level size.

### Total reward per primitive timestep

```
r_total = r_sparse + α · F_t + β · r_exploration

where:
  r_sparse    = reward machine output (R_SWITCH, R_EXIT, or 0)
  F_t         = PBRS shaping reward (γΦ(s') − Φ(s))
  r_exploration = optional decaying exploration bonus (RND or count-based)
  α ≈ 0.1–1.0 (tune so |α·F_t| ≈ 0.1–0.5× |r_sparse| in magnitude)
  β decays to 0 over training
```

### Nearest-node lookup

For runtime efficiency, precompute a spatial index (KD-tree or grid-based lookup) mapping continuous (x, y) positions to nearest graph nodes. The level is 1056×600 pixels — a simple 2D array at 12px resolution (88×50 = 4400 cells) with precomputed nearest-node IDs and Φ values is fast and sufficient.

```python
# Precompute once per level:
phi_grid_switch = np.zeros((88, 50))  # Φ values for phase u=0
phi_grid_exit = np.zeros((88, 50))    # Φ values for phase u=1
for ix in range(88):
    for iy in range(50):
        x, y = ix * 12, iy * 12
        node = nearest_node(x, y)
        phi_grid_switch[ix, iy] = -d_switch[node] / max_d_switch
        phi_grid_exit[ix, iy] = -d_exit[node] / max_d_exit

# At runtime (O(1) per step):
def get_phi(x, y, rm_state):
    ix, iy = int(x / 12), int(y / 12)
    ix, iy = clip(ix, 0, 87), clip(iy, 0, 49)
    if rm_state == 0:
        return phi_grid_switch[ix, iy]
    elif rm_state == 1:
        return phi_grid_exit[ix, iy] + C_PHASE
    else:
        return 0.0  # terminal
```

---

## Component 4: Integration with FiGAR-PPO

### Primitive-level GAE (critical)

PBRS must be applied at every primitive frame, not just at macro-action decision points. This is mathematically equivalent to SMDP-level PBRS (the intermediate terms telescope) but provides denser value function training signal.

```
For a macro-action of length k starting at time t:
  Primitive rewards: r_t, r_{t+1}, ..., r_{t+k-1}  (each includes F_i = γΦ(s_{i+1}) − Φ(s_i))
  
  GAE deltas at each primitive step:
    δ_i = r_i + γ·V(s_{i+1}) − V(s_i)
  
  Advantages via standard GAE(λ):
    A_i = Σ_{j=0}^{...} (γλ)^j · δ_{i+j}
```

**Policy loss**: Only computed at decision-point timesteps (start of each macro-action). The advantage for the decision at time t is A_t computed from the primitive-level GAE.

**Value loss**: Computed at ALL primitive timesteps. Train V(s) on every state visited, not just decision points. This gives the value function much richer supervision.

**Log probability at decision points**: `log π(a,w|s) = log π_action(a|s) + log π_repeat(w|s)` due to the factored FiGAR policy.

### Interrupt handling

When a macro-action is interrupted (state change: collision event, landing, reaching wall), the macro-action ends early. The agent:
- Receives primitive rewards up to the interruption point
- Makes a new decision at the interrupted state
- PBRS handles this naturally — no special logic needed since we're computing at primitive level

### Hyperparameter changes for FiGAR-PPO with PBRS

- **Entropy coefficient**: Increase from default ~0.01 to **0.03–0.05** (larger action space needs more exploration). Consider separate entropy coefficients for the action head and the repetition head.
- **Gradient clipping**: Keep or tighten (variable-length actions increase advantage variance)
- **GAE λ**: Standard 0.95 should work; reduce to 0.9 if high variance from variable macro-action lengths
- **γ**: Standard 0.99; the same γ must be used in both the PBRS formula and GAE

---

## Implementation Order

### Phase 1: Examine existing infrastructure
1. Scan `nclone/nclone/graph/` for existing reachability graph implementation
2. Scan `nclone/nclone/gym_environment/` for existing reward computation and step() logic
3. Scan `npp-rl/npp_rl/training/` for PPO training loop, rollout buffer, GAE computation
4. Identify where FiGAR is implemented (action space definition, policy heads, macro-action loop)
5. Check if Dijkstra/A* distances are already computed and stored

### Phase 2: Energy-cost graph (if needed)
1. If existing graph has distance-based weights: add energy-cost weight computation to edge creation
2. If no graph exists: build the graph construction pipeline (surface extraction → edge computation → Dijkstra)
3. Add backward Dijkstra from switch and exit positions
4. Build Φ grid lookup tables
5. Test: verify Φ values are sensible (higher near goals, lower far away, handles walls/obstacles correctly)

### Phase 3: Reward machine + PBRS
1. Implement RewardMachine class
2. Integrate into gym environment wrapper's step() method
3. Implement PBRS computation: `F_t = γ·Φ(s',u') − Φ(s,u)`
4. Combine: `r_total = r_sparse + α·F_t`
5. Ensure Φ=0 at terminal states (death, success)
6. Test: verify shaped rewards are positive when moving toward goal, negative when moving away

### Phase 4: FiGAR-PPO integration
1. Modify rollout buffer to store primitive-level (r_total, V(s)) at every frame
2. Modify GAE computation to work at primitive level
3. Modify policy loss to only use decision-point timesteps
4. Modify value loss to use all timesteps
5. Adjust entropy coefficients
6. Test: verify advantages and returns are computed correctly across macro-action boundaries

### Phase 5: Tuning
1. α (shaping scale): sweep [0.1, 0.3, 0.5, 1.0]
2. C_PHASE (phase bonus): sweep [0.0, 0.25, 0.5, 1.0]
3. Edge weight formula parameters (α_frames, β_height): sweep ratios
4. Monitor: |F_t|/|r_sparse| ratio should be ~0.1–0.5×. If >10×, reduce α.
5. Monitor: value function loss — should decrease steadily, not oscillate

---

## Key Invariants to Maintain

1. **Φ(terminal) = 0** — always, for all terminal states (death, timeout, success)
2. **Same γ everywhere** — PBRS formula, GAE, and value function must use identical γ
3. **Primitive-level application** — F_t computed every frame, not every macro-action
4. **Phase-correct targeting** — before switch: guide to switch; after switch: guide to exit
5. **Graph must be connected** — if Dijkstra can't reach a node from the goal, assign Φ = Φ_min (worst case), don't leave as infinity
6. **Normalization** — Φ ∈ [−1, 0] prevents reward scale issues across different levels

---

## Reference: Existing Observation Space Features (from OBSERVATION_SPACE_DOCUMENTATION.md)

The v2.0 observation space (153 dims) already includes A*-derived features that come from a reachability graph:
- `path_dist_to_switch_norm` / `path_dist_to_exit_norm` (reachability indices 4-5)
- `next_hop_direction` x,y (reachability indices 13-14)
- `path_requires_detour` (reachability index 17)
- `mine_clearance_direction` x,y (reachability indices 15-16)
- `switch_activated` (reachability index 12)
- `platform_distance` ahead/below (reachability indices for blind jump verification)

These features are computed in `nclone/nclone/graph/reachability/feature_computation.py` via `compute_reachability_features_from_graph()`. The existing graph infrastructure is the foundation — the main work is ensuring edge weights reflect energy cost and exposing the raw Dijkstra distance maps for PBRS.

## Reference: Key Physics Constants (from sim_mechanics_doc.md)

```
GRAVITY_FALL         = 0.06667 px/frame²
GRAVITY_JUMP         = 0.01111 px/frame²
GROUND_ACCEL         = 0.06667 px/frame²
AIR_ACCEL            = 0.04444 px/frame²
MAX_HOR_SPEED        = 3.333 px/frame
MAX_JUMP_DURATION    = 45 frames
DRAG_REGULAR         = 0.9933
FRICTION_GROUND      = 0.9459
FRICTION_WALL        = 0.9113
MAX_SURVIVABLE_IMPACT = 6.0 px/frame
NINJA_RADIUS         = 10 px
TILE_PIXEL_SIZE      = 24 px
LEVEL_SIZE           = 42×23 tiles (1056×600 px)
JUMP_FLAT_GROUND_Y   = -2.0
JUMP_WALL_REGULAR    = (±1.0, -1.4)
JUMP_WALL_SLIDE      = (±0.667, -1.0)
LAUNCH_PAD_BOOST     = 36/7 px/frame, scaled by 2/3
```