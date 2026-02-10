# Value Function PBRS - Complete Usage Guide

This guide walks through the complete workflow: precomputation → validation → training.

## Quick Start

### 1. Precompute Value Function

For a simple level (exit only):

```bash
python -m nclone.value_function.precompute_cli \
    --map-path levels/006.map \
    --goal exit \
    --precompute-transitions \
    --num-workers 8 \
    -v
```

For a two-phase level (switch + exit):

```bash
# Phase 1: Precompute switch value function
python -m nclone.value_function.precompute_cli \
    --map-path levels/006.map \
    --goal switch \
    --precompute-transitions \
    --num-workers 8

# Phase 2: Precompute exit value function
python -m nclone.value_function.precompute_cli \
    --map-path levels/006.map \
    --goal exit \
    --precompute-transitions \
    --num-workers 8
```

**Expected Output:**
```
======================================================================
              Value Function Precomputation
======================================================================
Level:  006
Goal:   exit
Output: value_functions/006_exit.npz

Step 1: Loading Level
----------------------------------------------------------------------
📂 Loading level: levels/006.map
✓ Level loaded successfully

...

======================================================================
              ✓ Precomputation Complete
======================================================================
Output file:   value_functions/006_exit.npz
File size:     62.4 MB
Total time:    35m 12s
Reachability:  15.2%
```

### 2. Visualize & Validate

**Interactive 2D Exploration (Recommended First Step):**

```bash
python -m nclone.value_function.visualize_value_function \
    --value-function value_functions/006_exit.npz \
    --mode 2d
```

This opens an interactive window with:
- Heatmap showing cost-to-go at each position
- Sliders to adjust velocity (vx, vy) and jump state
- Goal marker showing target position

**What to check:**
- ✅ Values should be 0 (or very close) at the goal position
- ✅ Values should increase smoothly as you move away from goal
- ✅ Unreachable regions (inf values) should make sense given obstacles
- ✅ Different velocity states should show reasonable variations

**Dashboard View (Comprehensive):**

```bash
python -m nclone.value_function.visualize_value_function \
    --value-function value_functions/006_exit.npz \
    --mode dashboard
```

Shows:
- Main heatmap
- Value distribution histogram
- Reachability vs velocity
- Cross-sections through goal
- Statistics panel

**Gradient Field (Validate PBRS Direction):**

```bash
python -m nclone.value_function.visualize_value_function \
    --value-function value_functions/006_exit.npz \
    --mode gradient \
    --stride 15
```

Red arrows show the potential gradient. They should:
- ✅ Point toward the goal
- ✅ Show smooth flow patterns
- ❌ NOT point away from goal or show chaotic patterns

### 3. Use in Training

**Single-phase (exit only):**

```python
from nclone.gym_environment import NPPEnvironment

env = NPPEnvironment(
    value_function_dir="value_functions/",
    use_precomputed_pbrs=True,
    custom_map_path="levels/006.map",
)

# Environment automatically loads 006_exit.npz
obs, info = env.reset()
```

**Two-phase (switch + exit) with curriculum:**

```python
from nclone.gym_environment import NPPEnvironment
from nclone.gym_environment.goal_curriculum_config import GoalCurriculumConfig

curriculum_config = GoalCurriculumConfig(
    enable_curriculum=True,
    curriculum_stages=[0, 1],  # 0=switch only, 1=switch+exit
)

env = NPPEnvironment(
    value_function_dir="value_functions/",
    use_precomputed_pbrs=True,
    custom_map_path="levels/006.map",
    goal_curriculum_config=curriculum_config,
)

# Environment uses:
# - Stage 0: value_functions/006_switch.npz
# - Stage 1: blended value_functions/006_switch.npz + 006_exit.npz
```

## Advanced Usage

### Custom Grid Resolution

Trade off between accuracy and computation time:

```bash
# Coarse grid (faster, ~5 min, good for testing)
python -m nclone.value_function.precompute_cli \
    --map-path levels/006.map \
    --goal exit \
    --dx 24 --dy 24 --dvx 1.0 --dvy 1.0 --djump 10

# Default grid (balanced, ~30 min, recommended)
python -m nclone.value_function.precompute_cli \
    --map-path levels/006.map \
    --goal exit \
    --dx 12 --dy 12 --dvx 0.5 --dvy 0.5 --djump 5

# Fine grid (slow, ~4 hours, maximum accuracy)
python -m nclone.value_function.precompute_cli \
    --map-path levels/006.map \
    --goal exit \
    --dx 6 --dy 6 --dvx 0.25 --dvy 0.25 --djump 2
```

### Without Transition Table

Skip transition precomputation (slower iteration but less memory):

```bash
python -m nclone.value_function.precompute_cli \
    --map-path levels/006.map \
    --goal exit \
    --max-iterations 500 \
    -v
```

### Convergence Tuning

Adjust convergence criteria:

```bash
python -m nclone.value_function.precompute_cli \
    --map-path levels/006.map \
    --goal exit \
    --max-iterations 1000 \
    --tolerance 0.001 \
    --precompute-transitions
```

## Troubleshooting

### Issue: Low Reachability (<5%)

**Symptoms:** Very few states are reachable from the goal.

**Possible Causes:**
- Goal position is incorrect or inaccessible
- Physics wrapper has bugs
- Grid resolution too coarse

**Solutions:**
1. Visualize with `--mode 2d` and check goal marker location
2. Test with coarser grid first to verify basic functionality
3. Check that goal position matches actual level goal

### Issue: Values Don't Converge

**Symptoms:** Value iteration doesn't converge within max_iterations.

**Possible Causes:**
- max_change stays large
- Physics simulation has non-determinism
- Cycle in transition graph

**Solutions:**
1. Increase `--max-iterations` to 1000+
2. Relax `--tolerance` to 0.1
3. Add verbose logging with `-v` to see iteration progress
4. Verify physics determinism with unit tests

### Issue: Gradient Points Away From Goal

**Symptoms:** In gradient visualization, arrows point in wrong directions.

**Possible Causes:**
- Value function has artifacts
- Interpolation issues
- Goal position incorrect

**Solutions:**
1. Check value function with dashboard mode
2. Verify goal position in visualization
3. Try different interpolation method in `PrecomputedValuePBRS`
4. Reduce grid resolution for smoother gradients

### Issue: Training Performance Doesn't Improve

**Symptoms:** Agent doesn't learn better with precomputed PBRS.

**Possible Causes:**
- Value function doesn't match actual physics
- Normalization issues
- PBRS weight too low

**Solutions:**
1. Compare precomputed vs path-based PBRS in TensorBoard
2. Check `_pbrs_using_precomputed` flag in episode info
3. Increase PBRS weight in reward config
4. Verify value function visualization looks reasonable

## Performance Tips

### 1. Parallel Precomputation

Use all CPU cores:

```bash
python -m nclone.value_function.precompute_cli \
    --map-path levels/006.map \
    --goal exit \
    --precompute-transitions \
    --num-workers $(nproc)  # Linux
    # --num-workers $(sysctl -n hw.ncpu)  # macOS
```

### 2. Batch Precomputation

Precompute multiple levels:

```bash
#!/bin/bash
for level in 006 007 008; do
    echo "Processing level $level..."
    
    # Switch
    python -m nclone.value_function.precompute_cli \
        --map-path levels/${level}.map \
        --goal switch \
        --precompute-transitions \
        --num-workers 8
    
    # Exit
    python -m nclone.value_function.precompute_cli \
        --map-path levels/${level}.map \
        --goal exit \
        --precompute-transitions \
        --num-workers 8
done
```

### 3. Reuse Transition Tables

If grid config unchanged, reuse transition tables:

```bash
# First level: compute transitions
python -m nclone.value_function.precompute_cli \
    --map-path levels/006.map \
    --goal exit \
    --precompute-transitions \
    --num-workers 8

# Transition table saved to: value_functions/006_transitions.npz

# Second level: physics differs, but we could modify code to load
# existing transition table if grid is identical
```

## Comparison: Precomputed vs Path-Based PBRS

| Aspect | Path-Based PBRS | Precomputed Value Function PBRS |
|--------|-----------------|----------------------------------|
| **Accuracy** | Heuristic (A* path length) | Exact optimal (Bellman equation) |
| **Physics Awareness** | Position only | Position + velocity + jump state |
| **Gradient Smoothness** | Discrete jumps at grid cells | Continuous via interpolation |
| **Runtime Cost** | ~200 μs/step (A* search) | ~10 μs/step (lookup + interp) |
| **Precomputation** | Graph building (~10s) | Value iteration (~30 min) |
| **Memory** | Graph adjacency (~5 MB) | Dense grid (~60 MB) |
| **Curriculum Support** | Manual waypoint extraction | Native two-phase support |

## Expected Results

### Typical Metrics

**Level 006 (default grid):**
- Total states: ~15.4M
- Reachable: ~15-20%
- File size: ~60 MB
- Precomputation time: ~30-40 min (with transitions)
- Value range: [0, 1200] frames
- Mean cost-to-go: ~350 frames

### Training Improvements

Expected improvements over path-based PBRS:
- **Faster learning:** 20-30% fewer episodes to reach threshold
- **Better exploration:** Velocity-aware guidance prevents local minima
- **Smoother rewards:** Continuous interpolation reduces variance
- **Curriculum alignment:** Seamless two-phase progression

## Further Reading

- **PBRS Theory:** Ng et al. (1999) - "Policy Invariance Under Reward Shaping"
- **Value Iteration:** Sutton & Barto - "Reinforcement Learning" Chapter 4
- **N++ Physics:** `docs/sim_mechanics_doc.md`
- **Implementation Details:** `nclone/value_function/README.md`
