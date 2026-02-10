# Optimal Value Function for PBRS

This package implements exact physics-aware PBRS using precomputed optimal value functions V*(s). Instead of using A* pathfinding for potential estimation, this approach computes the exact optimal cost-to-go at every reachable state using backward dynamic programming (value iteration).

## Key Advantages

1. **Physics-Aware**: Incorporates velocity and jump mechanics into the value function
2. **Exact Optimal**: Provides true optimal costs, not heuristic approximations
3. **Smooth Gradients**: Continuous interpolation provides smooth PBRS signals
4. **No Graph Dependency**: Works without precomputed reachability graphs
5. **Curriculum Compatible**: Supports two-phase objectives (switch → exit)

## Architecture Overview

```
State: (x, y, vx, vy, jump_frames) ∈ ℝ⁵
Grid: 88 × 50 × 14 × 25 × 10 = 15.4M states
Method: Value Iteration with physics simulation
Result: V*(s) = optimal cost-to-go from state s
PBRS: Φ(s) = 1 - V*(s) / max_V (normalized)
```

## Components

### 1. Physics Step Wrapper (`physics_step_wrapper.py`)

Provides a stateless physics step interface for value iteration:
- Reuses existing `Ninja` and `Simulator` code (NO duplication)
- Uses state snapshot/restore via `Ninja.save_state()` and `Ninja.restore_state()`
- Executes physics via `Simulator.tick_physics_only()` (bypasses entity AI)

```python
from nclone.value_function import PhysicsStepWrapper

physics = PhysicsStepWrapper(simulator)
next_state, is_death = physics.step(state=(x, y, vx, vy, jump_dur), action=(h_input, jump))
```

### 2. State Discretization (`state_discretization.py`)

Converts between continuous 5D states and discrete grid indices:
- Configurable grid resolution via `GridConfig`
- Default: dx=12px, dy=12px, dvx=0.5px/frame, dvy=0.5px/frame, djump=5 frames
- Supports nearest-neighbor and fractional coordinate conversion

```python
from nclone.value_function import StateDiscretizer, GridConfig

config = GridConfig(dx=12.0, dy=12.0, dvx=0.5, dvy=0.5, djump=5)
discretizer = StateDiscretizer(config)

# Convert state to grid index
idx = discretizer.state_to_index((x, y, vx, vy, jump_dur))

# Convert index to state (cell center)
state = discretizer.index_to_state(idx)
```

### 3. Value Iteration (`value_iteration.py`)

Computes optimal value function V*(s) using Bellman updates:
- Initializes goal states to 0, death states to ∞
- Iteratively propagates values backward from goal
- Converges when max change < tolerance

```python
from nclone.value_function import ValueIterator

iterator = ValueIterator(
    physics=physics,
    discretizer=discretizer,
    goal_pos=(500.0, 300.0),
    goal_radius=20.0
)

V = iterator.compute_value_function(
    max_iterations=500,
    tolerance=0.01,
    verbose=True
)
```

### 4. Transition Table (`transition_table.py`)

Precomputes all state transitions for accelerated value iteration:
- Uses multiprocessing for parallel precomputation
- Stores `(s, a) → s'` mapping for all 15.4M × 6 transitions
- Enables O(1) transition lookups during value iteration
- Typically 10-100× faster than direct physics simulation

```python
from nclone.value_function import TransitionTable, value_iteration_with_table

# Phase 1: Precompute transitions
table = TransitionTable(discretizer)
transitions = table.precompute(physics, num_workers=8)
table.save("transitions.npz")

# Phase 2: Accelerated value iteration
V = value_iteration_with_table(
    V_init=V_init,
    transition_table=table,
    max_iterations=500,
    tolerance=0.01
)
```

### 5. Interpolation (`interpolation.py`)

Provides 5D linear interpolation for continuous state queries:
- Supports nearest, linear, and hybrid interpolation modes
- Uses `scipy.ndimage.map_coordinates` for efficient 5D interpolation
- Handles out-of-bounds states gracefully

```python
from nclone.value_function import ValueInterpolator

interpolator = ValueInterpolator(discretizer, method='linear')
value = interpolator.interpolate(V, state=(x, y, vx, vy, jump_dur))
```

### 6. Precomputed PBRS (`precomputed_pbrs.py`)

Runtime class for PBRS potential lookup:
- Loads precomputed V*(s) from disk
- Provides O(1) potential lookup via interpolation
- Normalizes to [0, 1] range: Φ(s) = 1 - V*(s) / max_V

```python
from nclone.value_function import PrecomputedValuePBRS

pbrs = PrecomputedValuePBRS("value_functions/006_exit.npz")
potential = pbrs.get_potential(state)
```

### 7. Two-Phase PBRS (`two_phase_pbrs.py`)

Handles levels with two sequential objectives:
- Loads separate V*(s) grids for switch and exit
- Blends potentials based on `switch_activated` flag
- Supports continuous and discrete blending modes

```python
from nclone.value_function import TwoPhaseValuePBRS

pbrs = TwoPhaseValuePBRS(
    switch_value_path="value_functions/006_switch.npz",
    exit_value_path="value_functions/006_exit.npz",
    blend_mode='continuous'
)

potential = pbrs.get_potential(state)
```

## Usage

### Precomputing Value Functions (Offline)

Use the CLI tool to precompute value functions for specific levels:

```bash
# Single-phase level (exit only)
python -m nclone.value_function.precompute_cli \
    --map-path levels/006.map \
    --goal exit \
    --output-dir value_functions/

# Two-phase level (switch + exit)
python -m nclone.value_function.precompute_cli --map-path levels/006.map --goal switch
python -m nclone.value_function.precompute_cli --map-path levels/006.map --goal exit

# With transition table precomputation (much faster)
python -m nclone.value_function.precompute_cli \
    --map-path levels/006.map \
    --goal exit \
    --precompute-transitions \
    --num-workers 8

# Custom grid resolution
python -m nclone.value_function.precompute_cli \
    --map-path levels/006.map \
    --goal exit \
    --dx 8 --dy 8 --dvx 0.25 --dvy 0.25
```

Output: `value_functions/{level_name}_{goal}.npz`

### Using Precomputed PBRS (Runtime)

Enable precomputed PBRS in the Gym environment:

```python
from nclone.gym_environment import NPPEnvironment

env = NPPEnvironment(
    value_function_dir="value_functions/",
    use_precomputed_pbrs=True,
)
```

The environment will automatically load the appropriate value function for each level.

## Performance

### Precomputation Cost (per level)

| Method | Time | Memory |
|--------|------|--------|
| Direct Value Iteration | ~2-4 hours | ~200 MB |
| With Transition Table (CPU) | ~30-60 min | ~2 GB |
| With Transition Table (GPU) | ~2-5 min | ~2 GB + ~2-4 GB VRAM |

### GPU Acceleration

GPU-accelerated Semi-Lagrangian value iteration provides **10-50x speedup** over CPU:

```bash
# Enable GPU acceleration for SL-VI method
python -m nclone.value_function.precompute_cli \
    --map-path levels/006.map \
    --goal exit \
    --method sl \
    --precompute-transitions \
    --use-gpu

# Run benchmark to compare GPU vs CPU on your hardware
python -m nclone.value_function.benchmark_gpu --iterations 50
```

**Requirements**: PyTorch with CUDA support (`pip install torch`)

| Hardware | CPU Time | GPU Time | Speedup |
|----------|----------|----------|---------|
| RTX 3080ti | ~25 min | ~2 min | ~12x |
| RTX 4090 | ~20 min | ~1 min | ~20x |

The GPU implementation fully vectorizes the 4D multilinear interpolation (16 corners per lookup) and processes all 6 actions in parallel.

### Runtime Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Potential Lookup | ~10 μs | O(1) with interpolation |
| PBRS Calculation | ~20 μs | Includes Φ(s') - Φ(s) |
| Legacy Path-Based | ~200 μs | A* pathfinding overhead |

**Speedup: 10-20× faster than path-based PBRS**

## Grid Resolution Trade-offs

| Resolution | Total States | File Size | Accuracy | Iteration Time |
|------------|--------------|-----------|----------|----------------|
| Coarse (dx=24, dvx=1.0) | ~1M | ~5 MB | Good | ~5 min |
| Default (dx=12, dvx=0.5) | ~15M | ~60 MB | Excellent | ~30 min |
| Fine (dx=6, dvx=0.25) | ~120M | ~500 MB | Perfect | ~4 hours |

**Recommendation**: Use default resolution (dx=12, dvx=0.5) for production training.

## Integration with Existing Systems

### Compatibility with Curriculum Learning

The two-phase PBRS system supports goal curriculum:
- Stage 0: Switch-only (uses `switch` value function)
- Stage 1+: Switch → Exit (uses blended value functions)

### Fallback to Path-Based PBRS

If precomputed value functions are not available, the system automatically falls back to path-based PBRS:

```python
env = NPPEnvironment(
    value_function_dir="value_functions/",
    use_precomputed_pbrs=True,  # Will fallback if files missing
)
```

## Visualization

Validate and analyze precomputed value functions with the interactive visualization tool:

```bash
# Interactive 2D heatmap with sliders (recommended for exploration)
python -m nclone.value_function.visualize_value_function \
    --value-function value_functions/006_exit.npz \
    --mode 2d

# 3D surface plot
python -m nclone.value_function.visualize_value_function \
    --value-function value_functions/006_exit.npz \
    --mode 3d \
    --vx 0.0 --vy 0.0 --jump 0

# Gradient field (shows potential gradients as arrows)
python -m nclone.value_function.visualize_value_function \
    --value-function value_functions/006_exit.npz \
    --mode gradient \
    --stride 15

# Interactive plotly (web-based, requires: pip install plotly)
python -m nclone.value_function.visualize_value_function \
    --value-function value_functions/006_exit.npz \
    --mode interactive

# Multi-view dashboard (comprehensive overview)
python -m nclone.value_function.visualize_value_function \
    --value-function value_functions/006_exit.npz \
    --mode dashboard
```

### Visualization Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `2d` | Interactive heatmap with sliders | Explore value function across different velocities/jump states |
| `3d` | 3D surface plot | Understand value function topology |
| `gradient` | Arrow field showing potential gradients | Validate PBRS gradient direction toward goal |
| `interactive` | Web-based plotly visualization | High-quality interactive exploration |
| `dashboard` | Multi-panel overview | Comprehensive validation and statistics |

### What to Look For

✅ **Good Value Function:**
- Values decrease smoothly toward goal (0 at goal)
- Gradients point consistently toward goal
- High reachability percentage (>10%)
- Smooth transitions between adjacent states
- No discontinuities or artifacts

❌ **Potential Issues:**
- Large unreachable regions far from obstacles
- Gradients pointing away from goal
- Discontinuities in value function
- Values not reaching 0 at goal
- Unexpected artifacts or patterns

## Testing

Run unit tests:

```bash
python -m pytest tests/test_value_function.py -v
```

Tests cover:
- Physics step wrapper (state snapshot/restore)
- State discretization (continuous ↔ discrete conversions)
- Value iteration convergence
- Interpolation accuracy
- PBRS potential normalization

## Future Enhancements

1. **Adaptive Grid Resolution**: Use fine grids near obstacles, coarse elsewhere
2. **Multi-Level Caching**: Precompute for entire level sets
3. ~~**GPU-Accelerated Iteration**~~: ✅ Implemented! Use `--use-gpu` flag with SL-VI
4. **Model-Based Compression**: Train neural network to approximate V*(s)

## References

- **Potential-Based Reward Shaping**: Ng et al. (1999) - "Policy Invariance Under Reward Shaping"
- **Value Iteration**: Bellman (1957) - "Dynamic Programming"
- **N++ Physics**: Metanet Software - N++ game documentation

## Files

```
nclone/value_function/
├── __init__.py                      # Package exports
├── physics_step_wrapper.py         # Stateless physics interface
├── state_discretization.py         # Continuous ↔ discrete conversions
├── value_iteration.py              # Bellman updates + convergence
├── transition_table.py             # Precomputed transition caching (discrete)
├── continuous_transition_table.py  # Continuous transitions + SL-VI
├── gpu_value_iteration.py          # GPU-accelerated SL-VI (PyTorch/CUDA)
├── benchmark_gpu.py                # GPU vs CPU benchmark script
├── interpolation.py                # 5D linear interpolation
├── position_validity.py            # Position validity mask computation
├── precomputed_pbrs.py             # Runtime PBRS potential lookup
├── two_phase_pbrs.py               # Switch + exit blending
├── precompute_cli.py               # Command-line precomputation tool
├── visualize_value_function.py     # Interactive visualization tool
└── README.md                       # This file
```
