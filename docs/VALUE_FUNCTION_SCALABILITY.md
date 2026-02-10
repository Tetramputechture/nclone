# Precomputed Value Function PBRS: Current Approach, Limitations, and Scalability

## 1. Problem Context

We are training a reinforcement learning agent to play N++, a 2D physics-based platformer. The agent controls a ninja character through levels containing static geometry (tiles), objectives (switch and exit door), and various hazard entities. The agent must navigate from spawn to the exit switch, then to the exit door, without dying.

To accelerate learning, we use **Potential-Based Reward Shaping (PBRS)** as defined by Ng et al. (1999), which provides a dense reward signal derived from a potential function while provably preserving the optimal policy:

```
F(s, s') = gamma * Phi(s') - Phi(s)
```

The quality of the potential function directly determines how useful this shaping signal is. A perfect potential function would encode the exact optimal cost-to-go from every state, providing a gradient that always points toward the goal along the optimal trajectory.

## 2. Current Approach: Precomputed 5D Value Function

### 2.1 State Space and Discretization

We precompute an optimal value function `V*(s)` over a 5-dimensional state space:

```
s = (x, y, vx, vy, jump_frames)
```

- `(x, y)` -- ninja position in pixels, discretized at 12px resolution (half-tile)
- `(vx, vy)` -- ninja velocity in pixels/frame, discretized at 0.5 px/frame
- `jump_frames` -- frames since jump started (0 if not jumping), discretized in 5-frame buckets

This produces a grid of shape `89 x 51 x 15 x 27 x 10 ≈ 18.6M states`, stored as a float32 array (~75 MB per level).

### 2.2 Value Iteration Algorithm

We compute `V*(s)` using backward dynamic programming (Bellman value iteration):

1. **Initialization**: Set `V[s] = 0` for goal states (position within goal radius), `V[s] = inf` for all other states
2. **Position validity filtering**: Before iteration, mark positions where the ninja cannot physically exist as permanently `V = inf`:
   - Positions where the ninja center falls inside solid tile geometry (using the definitive tile segment definitions for all 34 tile types including slopes, curves, and the 1-tile solid border around the level)
   - Positions where the ninja overlaps with a toggled toggle mine (the only stationary lethal entity we currently handle)
3. **Bellman updates**: Iterate until convergence:
   ```
   V(s) = min over actions a of [ cost(s,a) + V(s') ]
   ```
   where `s' = physics_step(s, a)` is obtained by running the actual N++ physics simulation (gravity, drag, collision detection, depenetration, terminal velocity death, etc.)
4. **Action space**: 6 discrete actions `{left, left+jump, none, jump, right, right+jump}`
5. **Step cost**: `frames_per_step` (default 5) per transition, so `V*(s)` represents the minimum number of physics frames to reach the goal

### 2.3 Physics Integration

The physics step uses `Simulator.tick_physics_only()`, which executes the full ninja physics pipeline:

- Integration (drag, gravity, position update)
- Collision detection (sweep circle to prevent tunneling, 32-iteration depenetration loop against tile segments)
- Entity collisions (physical depenetration against bounce blocks, thwumps; logical collision for mine kills)
- Death detection (terminal velocity impact, crushing, toggled mine contact)
- State machine transitions (jumping, falling, wall sliding, etc.)

Entity AI (`entity.think()`, `entity.move()`) is **skipped** during value iteration. This means:

- Toggle mines remain in their initial state (type 1 = untoggled/safe, type 21 = toggled/deadly)
- Drones do not patrol
- Thwumps do not charge
- Bounce blocks do not oscillate

### 2.4 Runtime PBRS Lookup

At training time, the agent queries the precomputed value function:

```
Phi(s) = 1 - V*(s) / max_V
```

This provides a normalized potential in `[0, 1]`, where `Phi = 1.0` at the goal and `Phi = 0.0` at the most distant reachable state. Unreachable states (`V* = inf`) return `Phi = 0.0` (no gradient). Continuous state queries use 5D linear interpolation for smooth gradients.

### 2.5 What This Approach Handles Well

- **Static tile geometry**: All 34 tile types (full tiles, half tiles, 45-degree slopes, quarter circles, quarter pipes, mild slopes, steep slopes) are handled exactly through the physics simulation
- **Level boundaries**: The 1-tile solid border around the level is included in the tile dictionary and handled by position validity filtering
- **Velocity-aware costs**: Two states at the same position but different velocities correctly have different costs-to-go (e.g., moving toward the goal vs. away from it)
- **Jump mechanics**: The jump_frames dimension captures the cost difference between states with remaining jump duration vs. states that have exhausted their jump
- **Terminal velocity death**: The physics simulation correctly detects and rejects transitions where the ninja dies from high-speed surface impact
- **Stationary lethal entities**: Initially-toggled toggle mines (type 21) are correctly lethal via `post_collision() -> logical_collision()`

## 3. The Dynamic Entity Problem

### 3.1 Fundamental Limitation: The Environment State Is Not Fully Captured

Our 5D state space `(x, y, vx, vy, jump_frames)` captures only the **ninja's kinematic state**. It does not capture:

- The positions or states of any dynamic entities
- Whether toggle mines have been activated
- The phase of a drone's patrol cycle
- Whether a thwump has been triggered
- The displacement of bounce blocks from their origins
- Whether locked doors have been opened (changing the reachability graph)

The value function `V*(s)` implicitly assumes the environment is in a **single fixed configuration** -- specifically, the initial configuration at level start, with all entity AI frozen. Any deviation from this assumption makes the precomputed values incorrect.

### 3.2 Toggle Mines: State-Dependent Hazards

Toggle mines (entity types 1 and 21) are the simplest dynamic entity and already cause problems:

**Behavior**: Toggle mines cycle through three states based on ninja contact:
- **Untoggled** (safe, radius 3.5px) -- touching starts toggling
- **Toggling** (safe, radius 4.5px) -- breaking contact completes toggle
- **Toggled** (deadly, radius 4.0px) -- touching kills ninja and resets to untoggled

**Problem**: A mine that starts untoggled (safe to pass) becomes toggled (deadly) after the ninja passes through it. If the optimal path requires passing through the same mine location twice (e.g., backtracking), the value function does not know that the mine is now deadly. The correct cost-to-go depends on the mine's current state, which is part of the environment state not captured in our 5D space.

**Severity**: Moderate. For simple levels where the ninja only passes each mine once, the initial-state assumption is adequate. For levels requiring backtracking or complex mine sequences, the value function may provide misleading gradients.

To fully handle toggle mines, the state space would need to include a binary variable for each mine's toggle state. With `n` mines, this multiplies the state space by `2^n`. Even 8 mines would expand the space by 256x, making precomputation infeasible.

### 3.3 Drones: Deterministic but Time-Dependent Hazards

Zap drones (type 14) and mini drones (type 26) patrol along fixed grid-aligned paths at constant speed.

**Behavior** (from `EntityDroneBase`):
- Move between cell centers on a 24px grid at ~1.14 px/frame (zap) or ~1.71 px/frame (mini)
- Follow one of 4 patrol modes: wall-following CW/CCW, wandering CW/CCW
- Direction selection is deterministic based on tile geometry and mode
- Lethal on contact (circular collision, radius 7.5px for zap, 4.0px for mini)

**Problem**: Drone positions are time-dependent. The same spatial position `(x, y)` might be safe at frame `t` but lethal at frame `t + k` when a drone passes through. The correct cost-to-go at state `s` depends on the drone positions, which evolve deterministically over time.

**Why precomputation fails**: To incorporate drones, we would need to add drone positions to the state space. Each drone adds 2 continuous dimensions `(drone_x, drone_y)`. With just one drone, the state space becomes 7D; with several drones (levels can have up to 256), the state space explodes combinatorially. Even discretizing each drone to a small number of patrol phase buckets (e.g., 20), a level with 4 drones would multiply the state space by `20^4 = 160,000x`.

**Alternative: time dimension**: Since drone movement is deterministic and periodic, one could add a single `time mod patrol_period` dimension. But different drones may have different (or incommensurate) periods, and the combined period can be very large.

### 3.4 Thwumps: Reactive Hazards That Track the Player

Thwumps (type 20) are 18x18px squares that charge at high speed when the ninja enters their line of sight.

**Behavior** (from `EntityThwump`):
- Idle at origin position until ninja is detected in line of sight (38px activation range)
- Charges forward at ~2.86 px/frame until hitting a wall
- Retreats backward at ~1.14 px/frame to origin, then resets to idle
- Deadly only on the charging face; safe on sides and back
- Can be used as a moving platform (stand on top of a horizontal thwump)

**Problem**: Thwumps are **reactive to the ninja's position**. Their behavior depends on where the ninja is and was, creating a feedback loop between the ninja's trajectory and the environment state. The thwump's position is a function of the ninja's entire trajectory history, not just the current state.

**Why precomputation fails**: A thwump's charging behavior is triggered by the ninja's position. The same ninja state `(x, y, vx, vy, jump_frames)` may or may not trigger a thwump charge depending on whether the ninja approached from a direction within the thwump's line of sight. The resulting hazard pattern depends on the trajectory taken, which is not captured in any finite Markov state space that doesn't include the thwump's own state.

### 3.5 Bounce Blocks: Continuous-State Interactive Physics Objects

Bounce blocks (type 17) are 18x18px spring-mass systems that physically interact with the ninja.

**Behavior** (from `EntityBounceBlock`):
- Oscillate around an origin point via spring physics (stiffness 0.0222, dampening 0.98)
- When the ninja contacts them, 80% of the collision force goes to the block, 20% to the ninja
- Create momentum transfer that can be exploited for "block boosting" (rapid jump inputs during compression phase)
- Their displacement from origin is continuous and depends on interaction history
- Can be wall-jumped off of (logical collision provides wall normal)

**Problem**: Bounce blocks have continuous state `(block_x, block_y, block_vx, block_vy)` -- four additional continuous dimensions per block. Their state is tightly coupled with the ninja's actions: the block's future trajectory depends on how the ninja interacted with it. Block boosting, a critical advanced technique, requires precise timing of jump inputs relative to the block's compression phase, which is inherently sequential and state-dependent.

**Why precomputation fails**: Each bounce block adds 4 continuous dimensions to the state space. Even one bounce block would expand the discretized state from ~18.6M to ~18.6M * (position_bins * velocity_bins)^2. With multiple interacting blocks (up to 512 per level), the state space is utterly intractable for tabular value iteration.

### 3.6 Locked Doors: Discrete Reachability Graph Changes

Locked doors (type 6) are 24px collision segments that start closed and permanently open when the ninja collects their associated switch (radius 5px). There are also trap doors (type 8), which start open and permanently close on switch collection. Up to 256 locked doors can exist per level.

**Behavior** (from `EntityDoorLocked` / `EntityDoorBase`):
- A closed door adds a physical collision segment to the tile grid, blocking passage
- When the ninja touches the door's switch entity, the door opens permanently: the collision segment is deactivated and the grid edge values are decremented
- Opening a door changes the reachability graph of the level -- positions that were previously unreachable (behind the door) become reachable, and the optimal path may change fundamentally
- Trap doors work in reverse: collecting their switch closes them, blocking previously open passages

**Problem**: Each locked/trap door adds a binary state variable (open/closed) to the environment. A door's state determines whether entire regions of the level are accessible. The value function must account for which doors are open to compute correct costs-to-go: `V*(x, y, vx, vy, jump_frames | door_1=open, door_2=closed, ...)`.

**Why precomputation fails**: With `n` doors, the state space multiplies by `2^n`. Unlike toggle mines (which are localized hazards), doors have **global** effects on the value function -- opening a single door can change the optimal path for every state in the level, since it may create a shortcut or unlock the only route to the goal. A level with 8 locked doors would require 256 separate value function precomputations (one per door configuration), each ~75 MB and ~30 minutes of compute. This is technically feasible for small numbers of doors but does not scale.

**Severity**: High for levels where door order matters. The value function computed with all doors closed will show `V = inf` for positions behind locked doors, providing no gradient to guide the agent toward opening those doors.

### 3.7 Summary of Scalability Barriers

| Entity | State Dimensions Added | Behavior Type | Tractable for Precomputation? |
|--------|----------------------|---------------|-------------------------------|
| Static tiles | 0 | Fixed geometry | Yes (current approach) |
| Toggle mines (initial state) | 0 | Stationary, fixed state | Yes (current approach) |
| Toggle mines (full) | 1 binary per mine | State-dependent hazard | Only for very few mines |
| Locked/trap doors | 1 binary per door | Discrete reachability change | Only for very few doors (2^n configs) |
| Drones | 2 continuous per drone (or 1 periodic phase) | Deterministic patrol | Only for 1-2 drones with coarse discretization |
| Thwumps | 1 continuous per thwump + triggered flag | Reactive to player | No -- creates trajectory dependence |
| Bounce blocks | 4 continuous per block | Interactive physics | No -- continuous coupling with ninja state |

The core issue is the **curse of dimensionality**: every additional entity state variable multiplies the state space exponentially, making tabular value iteration infeasible beyond very simple configurations.

### 3.8 Key Insight: Visual Observation as Sufficient State Representation

Despite the combinatorial explosion of the explicit state space, it is worth noting that **a human player solves these levels using only visual input**. A human observes a rendered frame of the full level (1056x600 pixels) and infers all entity states, movement vectors, and hazard patterns from that image alone. They do not have access to the underlying state variables.

Concretely, a stack of 4 consecutive rendered frames (a standard technique in visual RL, following Mnih et al. 2015) would capture:

- **All entity positions** at the current moment (from the latest frame)
- **All entity velocities and movement directions** (from frame-to-frame differences)
- **Drone patrol direction and speed** (visible as positional shift across frames)
- **Thwump state** (idle, charging, or retreating -- visible from position changes)
- **Bounce block displacement and oscillation phase** (visible from position across frames)
- **Toggle mine state** (visually distinct between untoggled, toggling, and toggled states)
- **Locked door state** (open or closed -- visually distinct as the collision segment is present or absent)
- **Tile geometry and boundaries** (all visible in the rendered frame)

This suggests that a visual representation of the level (or an equivalent structured encoding that captures the same information) is a **sufficient statistic** for the full environment state. The challenge is not a lack of observable information, but rather that tabular methods cannot operate over this high-dimensional observation space. A function approximator -- such as a convolutional neural network processing rendered frames, or a structured encoder processing entity state vectors -- could in principle learn a value function or potential function over the full environment state without explicitly enumerating every entity configuration.

This observation motivates exploring learned value function approximators (rather than tabular precomputation) as the path to scalable PBRS in entity-rich levels.

## 4. What Our Current Value Function Still Provides

Despite these limitations, the static-environment value function is still valuable:

1. **Geometric guidance**: It correctly captures the cost of navigating the level's tile geometry, which is the primary challenge in most levels. The physics-aware value function knows about jump arcs, velocity requirements for reaching platforms, and terminal velocity dangers.

2. **Baseline potential**: Even in levels with dynamic entities, the static value function provides a reasonable "geometric cost-to-go" that the RL agent can learn to deviate from when dynamic hazards require it.

3. **Offline precomputation**: The value function is computed once per level and loaded at training time with O(1) lookup cost (~10 microseconds per query), adding negligible overhead to training.

4. **PBRS safety**: By Ng et al.'s theorem, any potential function preserves the optimal policy. An imperfect potential function from a static-environment assumption will still not prevent the agent from learning the true optimal policy -- it just provides a less informative shaping signal than a perfect one would.

## 5. Research Prompt

The following prompt is intended to be forwarded to a research-oriented model to explore state-of-the-art approaches for providing dense reward shaping or value function approximation in environments with dynamic, player-reactive, and/or high-dimensional entity states.

---

### Research Prompt: Scalable Value Function Approximation for PBRS in Dynamic 2D Platformer Environments

**Context**: We are training an RL agent (PPO) to play N++, a 2D physics platformer. We currently use Potential-Based Reward Shaping (PBRS) with a precomputed optimal value function `V*(x, y, vx, vy, jump_frames)` over a 5D discretized state space (~18.6M states) computed via tabular value iteration using the exact game physics. This works well for static tile geometry but cannot scale to levels containing dynamic entities.

**The dynamic entities we need to handle include**:

1. **Toggle mines**: Binary-state hazards that cycle safe/deadly based on player contact. Each mine adds 1 bit to the environment state. Levels can have up to 8,192 mines.
2. **Patrol drones**: Grid-aligned enemies following deterministic patrol routes at constant speed. Each drone's position is a periodic function of time. Drones are lethal on contact. Levels can have up to 256 drones.
3. **Thwumps**: Rectangular enemies that idle at an origin position, charge at high speed toward the player when line-of-sight is established, hit a wall, then retreat. Their behavior is reactive to the player's position and trajectory. Levels can have up to 128 thwumps.
4. **Bounce blocks**: Spring-mass physics objects with continuous 4D state `(x, y, vx, vy)` that physically interact with the player, enabling advanced momentum-transfer techniques. Levels can have up to 512 bounce blocks.
5. **Locked doors and trap doors**: 24px collision segments that permanently open (locked) or close (trap) when the player collects their associated switch. Each door adds 1 bit to the environment state and changes the reachability graph of the level. Levels can have up to 256 doors.

**Key observation about visual sufficiency**: A human player solves these levels from visual input alone. A stack of 4 consecutive rendered frames (1056x600 pixels each) is a sufficient statistic for the full environment state -- it captures all entity positions, velocities (from frame-to-frame differences), states (visually distinct), and geometry. This suggests that a learned function approximator operating over visual or structured observations could sidestep the curse of dimensionality that defeats tabular methods.

**The fundamental challenge**: Our tabular value function approach suffers from the curse of dimensionality when entity states are included. Adding even a few dynamic entities makes the state space intractable for exact precomputation.

**Our requirements for a solution**:

- Must be usable as a PBRS potential function `Phi(s)` (or produce an equivalent dense shaping signal)
- Must preserve the optimal policy (or come with guarantees about the degree of suboptimality introduced)
- Must be computationally feasible for offline precomputation and fast at runtime (~10-100 microsecond query time during PPO training)
- The game physics are deterministic and we have a fast, exact simulator available for generating arbitrary state transitions
- We have access to expert demonstration replays and can generate additional training data through simulation

**What we are looking for**:

1. **State-of-the-art approaches** for approximating value functions or potential functions in environments with mixed static/dynamic elements, particularly:
   - Neural network function approximators for value functions in high-dimensional state spaces
   - Methods for decomposing a value function into static (geometry) and dynamic (entity) components
   - Techniques from the multi-agent or entity-interaction literature for handling variable numbers of dynamic objects
   - Attention-based or graph-neural-network approaches for entity-aware value estimation

2. **Approaches for handling periodic/deterministic entity motion** (drones):
   - Time-indexed or phase-conditioned value functions
   - Factored MDPs where entity states evolve independently of the player
   - Temporal abstraction or option-based approaches that reason about "safe windows" in periodic hazard patterns

3. **Approaches for handling reactive entities** (thwumps, toggle mines) **and discrete reachability changes** (locked/trap doors):
   - History-conditioned or recurrent value functions
   - Methods from adversarial/competitive RL for handling entities that react to the player
   - Belief-space planning approaches where entity state is partially observable
   - Value function decomposition or conditioning for discrete environment modes (e.g., which doors are open/closed)

4. **Approaches for handling continuous interactive physics** (bounce blocks):
   - Hybrid approaches combining physics simulation with learned value approximation
   - Model-predictive control (MPC) style approaches that plan short horizons using the simulator
   - Object-centric representation learning for physics-based interactions

5. **Hybrid/hierarchical approaches** that combine:
   - A precomputed static value function (our current approach) as a base
   - A learned correction or residual that accounts for dynamic entities
   - Potentially different strategies for different entity types based on their complexity

6. **Visual or structured observation approaches** leveraging the fact that 4 stacked frames capture the full environment state:
   - Learning value functions from visual observations (CNN-based value networks trained on simulator rollouts)
   - Structured state encoders that process entity lists with attention or graph neural networks
   - Frame-stacking or temporal convolution approaches for inferring entity velocities and states
   - Pretraining a visual value function via offline RL or behavioral cloning from expert demonstrations, then using it as a PBRS potential

7. **Alternatives to PBRS** that might be more suitable:
   - Hindsight experience replay (HER) or goal-conditioned approaches
   - Curiosity-driven or count-based exploration bonuses calibrated to geometry
   - Demonstration-guided reward shaping that doesn't require an explicit value function
   - Successor features / successor representations that might decompose more naturally

Please provide specific paper references (with years), method names, and a brief assessment of each approach's applicability to our setting. Prioritize methods that have been demonstrated in similar physics-based or platformer domains, or that have strong theoretical foundations applicable to our constraints.
