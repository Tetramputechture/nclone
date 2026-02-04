# Optimal Precomputed Value Function for N++ PBRS

**Date:** February 3, 2026  
**Status:** Technical Specification

---

## Executive Summary

Given your constraints (static levels, deterministic physics, fast simulation), the theoretically optimal approach is:

**Precompute V*(s) via backward dynamic programming on the discretized full state space, then use V*(s) directly as the PBRS potential.**

This gives you:
- **Exact optimal cost-to-go** for every state (position, velocity, jump_frames)
- **Guaranteed policy invariance** (V* is the true optimal potential)
- **O(1) runtime lookup** via interpolation
- **No heuristic multipliers** - physics simulation determines all costs

---

## Part 1: Why This Is Theoretically Optimal

### The Fundamental Problem with Your Current Approach

Your current system computes edge costs using heuristic multipliers (0.15×, 0.5×, 1.5×, etc.) applied to geometric distances. These are **approximations** of the true physical cost. The A* path cost is then:

```
A* cost = Σ (geometric_distance × heuristic_multiplier)
```

This has two problems:
1. **Multipliers are hand-tuned**, not physics-derived
2. **A* operates on position-only graph**, ignoring velocity state

### The Optimal Solution: True Value Function

The **optimal value function V*(s)** for a deterministic system is:

```
V*(s) = min over all policies π [ cost to reach goal from state s ]
```

For your platformer, with state s = (x, y, vx, vy, jump_frames):

```
V*(s) = minimum time/energy to reach goal from (x, y, vx, vy, jump_frames)
```

**This is exactly what you want for PBRS.** If you have V*(s), then:

```
Φ(s) = -V*(s)  (negate because PBRS wants higher = better)
F(s, s') = γ × Φ(s') - Φ(s)
```

The agent receives positive reward when it reduces V* (gets closer to goal in a physics-optimal sense) and negative reward when it increases V*.

### Why Precomputation Is Feasible

**State space analysis for N++ level:**

| Dimension | Range | Grid Points | Notes |
|-----------|-------|-------------|-------|
| x | 0-1056 px | 88 | 12px resolution (half tile) |
| y | 0-600 px | 50 | 12px resolution |
| vx | -3.33 to +3.33 | 14 | 0.5 px/frame resolution |
| vy | -6 to +6 | 25 | 0.5 px/frame resolution |
| jump_frames | 0-45 | 10 | 5-frame buckets |

**Total states:** 88 × 50 × 14 × 25 × 10 = **15.4 million states**

**Memory:** 15.4M × 4 bytes = **62 MB** per level (float32 values)

**Computation time estimate:**
- 15.4M states × ~6 actions × ~100 iterations = 9.2 billion updates
- At 1μs per update (simple physics step) = ~2.5 hours
- With GPU/vectorization: **10-30 minutes per level**

This is completely tractable for offline precomputation.

---

## Part 2: The Algorithm - Backward Value Iteration

### Problem Formulation

**State:** s = (x, y, vx, vy, jump_remaining)

**Actions:** a ∈ {(left, no_jump), (left, jump), (none, no_jump), (none, jump), (right, no_jump), (right, jump)}

**Dynamics:** s' = physics_step(s, a)  — Your deterministic N++ simulation

**Cost function:** c(s, a) = 1 (minimum-time) or c(s, a) = |a| (minimum-effort)

**Goal set:** G = {s : distance(s.position, goal_position) < threshold}

### Value Iteration Algorithm

```python
def compute_optimal_value_function(level_data, goal_position, max_iterations=500):
    """
    Compute V*(s) for all states via backward dynamic programming.
    
    Returns:
        V: ndarray of shape (nx, ny, nvx, nvy, njump) containing V*(s)
    """
    # Initialize value function
    # V[s] = 0 for goal states, +inf elsewhere
    V = np.full(state_shape, np.inf, dtype=np.float32)
    
    # Set goal states to zero cost
    for s in goal_states(goal_position):
        V[s] = 0.0
    
    # Set obstacle/death states to +inf (unreachable)
    for s in death_states(level_data):
        V[s] = np.inf
    
    # Value iteration (Bellman updates)
    for iteration in range(max_iterations):
        V_new = V.copy()
        max_change = 0.0
        
        for s in all_states():
            if is_goal(s) or is_death(s):
                continue
            
            # Bellman update: V(s) = min_a [ c(s,a) + V(s') ]
            best_cost = np.inf
            for a in actions:
                s_next = physics_step(s, a)
                s_next_idx = discretize(s_next)
                
                # Interpolate V at s_next (may be off-grid)
                V_next = interpolate_value(V, s_next)
                
                cost = step_cost(s, a) + V_next
                best_cost = min(best_cost, cost)
            
            V_new[s] = best_cost
            max_change = max(max_change, abs(V_new[s] - V[s]))
        
        V = V_new
        
        # Convergence check
        if max_change < tolerance:
            print(f"Converged after {iteration} iterations")
            break
    
    return V
```

### Key Implementation Details

#### 1. Physics Step Function

Use your **exact** N++ physics simulation:

```python
def physics_step(state, action):
    """
    Execute one physics frame with given action.
    Returns the resulting state.
    
    This should be YOUR actual physics code, not an approximation.
    """
    x, y, vx, vy, jump_remaining = state
    h_input, jump_input = action
    
    # === YOUR EXACT PHYSICS FROM sim_mechanics_doc.md ===
    
    # Determine gravity based on jump state
    if jump_input and jump_remaining > 0:
        gravity = GRAVITY_JUMP  # 0.0111
        jump_remaining -= 1
    else:
        gravity = GRAVITY_FALL  # 0.0667
        jump_remaining = 0
    
    # Apply acceleration based on grounded state
    grounded = is_grounded(x, y, level_data)
    walled = is_walled(x, y, level_data)
    
    if grounded:
        vx += h_input * GROUND_ACCEL
        # Apply ground friction
        vx *= FRICTION_GROUND
        # Handle jump initiation
        if jump_input and jump_remaining == 0:
            vy = JUMP_FLOOR_Y  # -2.0
            jump_remaining = MAX_JUMP_DURATION
    elif walled:
        # Wall physics
        vx += h_input * AIR_ACCEL
        vy *= FRICTION_WALL
        # Wall jump logic
        if jump_input:
            vy = JUMP_WALL_Y
            vx = wall_normal * JUMP_WALL_X
    else:
        # Air physics
        vx += h_input * AIR_ACCEL
    
    # Clamp horizontal speed
    vx = np.clip(vx, -MAX_HOR_SPEED, MAX_HOR_SPEED)
    
    # Apply gravity
    vy += gravity
    
    # Apply drag
    vx *= DRAG_REGULAR
    vy *= DRAG_REGULAR
    
    # Update position
    x_new = x + vx
    y_new = y + vy
    
    # Collision detection and resolution
    x_new, y_new, vx, vy = resolve_collisions(x, y, x_new, y_new, vx, vy, level_data)
    
    # Check for death
    if check_death(x_new, y_new, vx, vy, level_data):
        return DEATH_STATE
    
    return (x_new, y_new, vx, vy, jump_remaining)
```

#### 2. State Discretization

```python
# Grid parameters
X_MIN, X_MAX = 0, 1056
Y_MIN, Y_MAX = 0, 600
VX_MIN, VX_MAX = -3.5, 3.5
VY_MIN, VY_MAX = -6.5, 6.5
JUMP_MIN, JUMP_MAX = 0, 45

# Grid resolutions
DX = 12  # Half tile
DY = 12
DVX = 0.5
DVY = 0.5
DJUMP = 5

# Grid sizes
NX = int((X_MAX - X_MIN) / DX) + 1
NY = int((Y_MAX - Y_MIN) / DY) + 1
NVX = int((VX_MAX - VX_MIN) / DVX) + 1
NVY = int((VY_MAX - VY_MIN) / DVY) + 1
NJUMP = int((JUMP_MAX - JUMP_MIN) / DJUMP) + 1

def state_to_index(x, y, vx, vy, jump):
    """Convert continuous state to grid indices."""
    ix = int(np.clip((x - X_MIN) / DX, 0, NX - 1))
    iy = int(np.clip((y - Y_MIN) / DY, 0, NY - 1))
    ivx = int(np.clip((vx - VX_MIN) / DVX, 0, NVX - 1))
    ivy = int(np.clip((vy - VY_MIN) / DVY, 0, NVY - 1))
    ijump = int(np.clip(jump / DJUMP, 0, NJUMP - 1))
    return (ix, iy, ivx, ivy, ijump)

def index_to_state(ix, iy, ivx, ivy, ijump):
    """Convert grid indices to continuous state (cell centers)."""
    x = X_MIN + (ix + 0.5) * DX
    y = Y_MIN + (iy + 0.5) * DY
    vx = VX_MIN + (ivx + 0.5) * DVX
    vy = VY_MIN + (ivy + 0.5) * DVY
    jump = ijump * DJUMP
    return (x, y, vx, vy, jump)
```

#### 3. Value Interpolation for Off-Grid States

```python
def interpolate_value(V, state):
    """
    Trilinear interpolation for continuous state lookup.
    
    This is critical for accuracy - physics_step produces continuous
    states that may not land exactly on grid points.
    """
    x, y, vx, vy, jump = state
    
    # Handle death/out-of-bounds
    if is_death_state(state):
        return np.inf
    
    # Get fractional grid coordinates
    fx = (x - X_MIN) / DX
    fy = (y - Y_MIN) / DY
    fvx = (vx - VX_MIN) / DVX
    fvy = (vy - VY_MIN) / DVY
    fjump = jump / DJUMP
    
    # Clamp to valid range
    fx = np.clip(fx, 0, NX - 1.001)
    fy = np.clip(fy, 0, NY - 1.001)
    fvx = np.clip(fvx, 0, NVX - 1.001)
    fvy = np.clip(fvy, 0, NVY - 1.001)
    fjump = np.clip(fjump, 0, NJUMP - 1.001)
    
    # Use scipy's ndimage interpolation or implement 5D linear interp
    # For efficiency, can use lower-order interpolation
    return scipy.ndimage.map_coordinates(
        V, 
        [[fx], [fy], [fvx], [fvy], [fjump]], 
        order=1,  # Linear interpolation
        mode='nearest'
    )[0]
```

#### 4. Cost Function Options

**Option A: Minimum Time (frames to reach goal)**
```python
def step_cost(state, action):
    return 1.0  # Each frame costs 1
```

**Option B: Minimum Energy (control effort)**
```python
def step_cost(state, action):
    h_input, jump_input = action
    return 1.0 + 0.1 * abs(h_input) + 0.2 * jump_input
```

**Option C: Minimum Time with Hazard Penalty**
```python
def step_cost(state, action):
    base_cost = 1.0
    x, y, _, _, _ = state
    
    # Add cost for proximity to mines/hazards
    min_mine_dist = get_min_mine_distance(x, y)
    if min_mine_dist < 50:
        base_cost += (50 - min_mine_dist) / 50 * 2.0
    
    return base_cost
```

Minimum time is recommended for PBRS because it directly corresponds to "how many steps to reach goal."

---

## Part 3: Vectorized GPU Implementation

For 15M+ states, CPU iteration is slow. Here's a vectorized approach:

```python
import cupy as cp  # GPU arrays

def compute_value_function_gpu(level_data, goal_position):
    """
    GPU-accelerated value iteration.
    
    Key insight: All states can be updated in parallel since we're
    using the OLD V values (synchronous Bellman update).
    """
    # Allocate on GPU
    V = cp.full((NX, NY, NVX, NVY, NJUMP), cp.inf, dtype=cp.float32)
    
    # Precompute which states are goals, obstacles, valid
    is_goal = precompute_goal_mask(goal_position)
    is_obstacle = precompute_obstacle_mask(level_data)
    
    V[is_goal] = 0.0
    V[is_obstacle] = cp.inf
    
    # Precompute physics transitions for ALL (state, action) pairs
    # This is a one-time cost that enables massive parallelization
    transitions = precompute_all_transitions(level_data)
    # transitions[s, a] = (s'_indices, s'_weights for interpolation)
    
    for iteration in range(500):
        V_old = V.copy()
        
        # Vectorized Bellman update over all states and actions
        # V_new[s] = min_a [ 1 + V_old[s'] ]
        
        for a in range(6):  # 6 actions
            s_next_idx = transitions[:, :, :, :, :, a, 0]  # Next state indices
            V_next = sample_value_function(V_old, s_next_idx)
            Q_a = 1.0 + V_next
            V = cp.minimum(V, Q_a)
        
        # Check convergence
        max_change = cp.max(cp.abs(V - V_old))
        if max_change < 0.01:
            print(f"Converged at iteration {iteration}")
            break
    
    return cp.asnumpy(V)  # Transfer back to CPU
```

### Precomputing Transitions

The key optimization is precomputing where each (state, action) pair transitions to:

```python
def precompute_all_transitions(level_data):
    """
    Precompute physics transitions for all states.
    
    Returns array of shape (NX, NY, NVX, NVY, NJUMP, 6, 6)
    where the last dimension stores [ix', iy', ivx', ivy', ijump', valid]
    """
    transitions = np.zeros((NX, NY, NVX, NVY, NJUMP, 6, 6), dtype=np.int16)
    
    for ix in range(NX):
        for iy in range(NY):
            for ivx in range(NVX):
                for ivy in range(NVY):
                    for ijump in range(NJUMP):
                        state = index_to_state(ix, iy, ivx, ivy, ijump)
                        
                        for ia, action in enumerate(ACTIONS):
                            s_next = physics_step(state, action)
                            
                            if is_death_state(s_next):
                                transitions[ix, iy, ivx, ivy, ijump, ia] = [-1, -1, -1, -1, -1, 0]
                            else:
                                idx = state_to_index(*s_next)
                                transitions[ix, iy, ivx, ivy, ijump, ia] = [*idx, 1]
    
    return transitions
```

**Time estimate:** 15.4M states × 6 actions × 1ms/physics_step = ~25 hours on CPU

But this only needs to be done **once per level** and can be parallelized across cores.

---

## Part 4: Runtime Usage for PBRS

Once you have V*(s), using it for PBRS is trivial:

```python
class PrecomputedValuePBRS:
    def __init__(self, value_function_path):
        """Load precomputed value function from disk."""
        self.V = np.load(value_function_path)  # Shape: (NX, NY, NVX, NVY, NJUMP)
        
        # Store grid parameters
        self.grid_params = load_grid_params(value_function_path)
    
    def get_potential(self, state):
        """
        O(1) lookup of potential for any state.
        
        Args:
            state: dict with player_x, player_y, player_xspeed, player_yspeed, jump_duration_remaining
        
        Returns:
            Potential value Φ(s) for PBRS
        """
        x = state["player_x"]
        y = state["player_y"]
        vx = state["player_xspeed"]
        vy = state["player_yspeed"]
        jump = int(state["jump_duration_remaining"] * 45)  # Denormalize
        
        # Interpolate value function
        V_s = self._interpolate(x, y, vx, vy, jump)
        
        # Handle unreachable states
        if V_s == np.inf or V_s > 10000:
            return 0.0  # No gradient for truly unreachable
        
        # Convert to potential: Φ(s) = -V*(s) / normalization
        # Negate because lower V* = closer to goal = higher potential
        max_V = self.V[self.V < np.inf].max()
        potential = 1.0 - (V_s / max_V)
        
        return np.clip(potential, 0.0, 1.0)
    
    def _interpolate(self, x, y, vx, vy, jump):
        """5D linear interpolation."""
        # Convert to fractional grid coordinates
        fx = (x - X_MIN) / DX
        fy = (y - Y_MIN) / DY
        fvx = (vx - VX_MIN) / DVX
        fvy = (vy - VY_MIN) / DVY
        fjump = jump / DJUMP
        
        # Clamp to valid range
        fx = np.clip(fx, 0, NX - 1.001)
        fy = np.clip(fy, 0, NY - 1.001)
        fvx = np.clip(fvx, 0, NVX - 1.001)
        fvy = np.clip(fvy, 0, NVY - 1.001)
        fjump = np.clip(fjump, 0, NJUMP - 1.001)
        
        # Fast 5D interpolation using scipy or custom implementation
        return scipy.ndimage.map_coordinates(
            self.V, 
            [[fx], [fy], [fvx], [fvy], [fjump]], 
            order=1,
            mode='nearest',
            cval=np.inf
        )[0]
```

### Integration with Your Reward Calculator

```python
# In your reward calculation code:

class PBRSCalculator:
    def __init__(self, level_id):
        # Load precomputed value function for this level
        self.value_pbrs = PrecomputedValuePBRS(f"value_functions/{level_id}.npz")
    
    def calculate_potential(self, state, adjacency, level_data, path_calculator, graph_data):
        """
        Use precomputed V*(s) directly as potential.
        
        This replaces ALL the A* pathfinding and heuristic cost calculation!
        """
        return self.value_pbrs.get_potential(state)
```

---

## Part 5: Two-Phase Extension (Switch + Exit)

Your current system has two objectives: reach switch, then reach exit. Handle this with two value functions:

```python
class TwoPhaseValuePBRS:
    def __init__(self, level_id):
        self.V_switch = np.load(f"value_functions/{level_id}_switch.npz")
        self.V_exit = np.load(f"value_functions/{level_id}_exit.npz")
    
    def get_potential(self, state):
        switch_activated = state["switch_activated"]
        
        if not switch_activated:
            # Phase 1: Navigate to switch
            V_s = self._interpolate(self.V_switch, state)
            # Scale to [0, 0.5]
            return 0.5 * (1.0 - V_s / self.max_V_switch)
        else:
            # Phase 2: Navigate to exit
            V_s = self._interpolate(self.V_exit, state)
            # Scale to [0.5, 1.0]
            return 0.5 + 0.5 * (1.0 - V_s / self.max_V_exit)
```

---

## Part 6: Comparison with Current Approach

| Aspect | Current (A* + Heuristics) | Optimal (Precomputed V*) |
|--------|--------------------------|-------------------------|
| **Accuracy** | Approximate (heuristic multipliers) | Exact (true physics) |
| **State awareness** | Position + velocity (first edge only) | Full state (x, y, vx, vy, jump) |
| **Impossible trajectories** | Blocked by inf multipliers | Naturally inf (no valid path) |
| **Runtime cost** | 0.1-0.5ms (A* search) | <0.01ms (interpolation) |
| **Precomputation** | None | 10-30 min per level |
| **Memory** | Graph structure (~10MB) | Value grid (~62MB) |
| **Tuning required** | Many multipliers to tune | Cost function only |
| **Path visualization** | Explicit edges | Gradient descent on V |

---

## Part 7: Implementation Roadmap

### Week 1: Core Infrastructure

1. **Day 1-2:** Implement state discretization and grid management
   - State-to-index and index-to-state functions
   - Grid parameter configuration
   - Test with simple level

2. **Day 3-4:** Implement physics step wrapper
   - Extract your existing physics code into standalone function
   - Verify it matches actual game behavior
   - Handle edge cases (death, collision)

3. **Day 5:** Implement value iteration (CPU, single-threaded)
   - Basic Bellman updates
   - Goal and obstacle masking
   - Convergence checking

### Week 2: Optimization

1. **Day 1-2:** Precompute transition table
   - Store next-state indices for all (state, action) pairs
   - Parallelize across CPU cores

2. **Day 3-4:** GPU acceleration (optional but recommended)
   - Port value iteration to CuPy/PyTorch
   - Vectorize Bellman updates

3. **Day 5:** Interpolation and runtime integration
   - Implement fast 5D interpolation
   - Create PrecomputedValuePBRS class
   - Test with your environment

### Week 3: Integration and Testing

1. **Day 1-2:** Replace existing PBRS with precomputed version
   - A/B test: heuristic vs precomputed
   - Compare training curves

2. **Day 3-4:** Two-phase extension
   - Separate value functions for switch and exit
   - Test phase transitions

3. **Day 5:** Documentation and tooling
   - Scripts to generate value functions for new levels
   - Visualization of value function slices

---

## Part 8: Alternative Approaches Considered

### Why Not Hamilton-Jacobi PDE?

HJ reachability solves a continuous-time PDE. Your game is discrete-time with specific frame-by-frame physics. Value iteration on the discrete system is more accurate and avoids time-discretization errors.

### Why Not Trajectory Optimization?

Trajectory optimization (shooting methods, direct collocation) finds optimal paths on-demand. But:
- Each query costs 10-100ms
- Doesn't naturally provide cost-to-go from all states
- Would need to be called every step

Precomputation amortizes this cost: compute once, query infinitely.

### Why Not Learn V*(s) with a Neural Network?

Neural value approximation (e.g., DeepReach) is useful when:
- State space is very high-dimensional (>6D)
- Dynamics are partially unknown
- Need to generalize across environments

Your state space is 5D and dynamics are perfectly known. Tabular value iteration is more accurate and interpretable.

---

## Conclusion

For your specific constraints (static levels, deterministic physics, offline precomputation allowed), **precomputing the true optimal value function V*(s) via backward dynamic programming is the theoretically optimal approach.**

This eliminates:
- Hand-tuned heuristic multipliers
- A* pathfinding at runtime
- Edge cost approximations
- The geometric vs physics cost disconnect

And provides:
- Exact optimal cost-to-go
- Full state awareness (position + velocity + jump)
- O(1) runtime queries
- Guaranteed policy invariance for PBRS

The main cost is precomputation time (10-30 minutes per level) and memory (62MB per level), both of which are acceptable given your constraints.

---

## Appendix: Quick Start Code

```python
# quick_start.py - Minimal implementation to get started

import numpy as np
from scipy.ndimage import map_coordinates

# Grid parameters (adjust for your level)
NX, NY, NVX, NVY, NJUMP = 88, 50, 14, 25, 10
ACTIONS = [(-1, 0), (-1, 1), (0, 0), (0, 1), (1, 0), (1, 1)]

def compute_value_function(physics_step_fn, is_goal_fn, is_death_fn, max_iters=500):
    """
    Compute V*(s) via value iteration.
    
    Args:
        physics_step_fn: (state, action) -> next_state
        is_goal_fn: state -> bool
        is_death_fn: state -> bool
    """
    V = np.full((NX, NY, NVX, NVY, NJUMP), np.inf, dtype=np.float32)
    
    # Initialize goal states
    for idx in np.ndindex(V.shape):
        state = index_to_state(idx)
        if is_goal_fn(state):
            V[idx] = 0.0
    
    # Value iteration
    for it in range(max_iters):
        V_old = V.copy()
        
        for idx in np.ndindex(V.shape):
            if V[idx] == 0.0:  # Goal state
                continue
            
            state = index_to_state(idx)
            if is_death_fn(state):
                continue
            
            best = np.inf
            for action in ACTIONS:
                s_next = physics_step_fn(state, action)
                if is_death_fn(s_next):
                    continue
                V_next = interpolate_value(V_old, s_next)
                best = min(best, 1.0 + V_next)
            
            V[idx] = best
        
        if np.max(np.abs(V - V_old)[V < np.inf]) < 0.1:
            print(f"Converged at iteration {it}")
            break
    
    return V

# Usage:
# V = compute_value_function(your_physics_step, your_is_goal, your_is_death)
# np.savez_compressed("level_value_function.npz", V=V)
```
