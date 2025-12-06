# Momentum-Aware PBRS System

## Overview

The momentum-aware PBRS system solves a critical problem in platformer RL: **how to reward agents for necessary momentum-building behavior** that temporarily moves away from the goal.

### The Problem

In N++, many levels require momentum-dependent maneuvers:
- Running left to build speed before jumping right over mines
- Building velocity on slopes for long jumps
- Backtracking to get a running start

Traditional geometric pathfinding computes the shortest position-only path, which:
- Goes directly toward the goal (ignoring momentum requirements)
- Causes PBRS to **penalize** necessary momentum-building (distance increases)
- Prevents agents from learning optimal momentum-based strategies

### The Solution

**Two-component system:**

1. **Momentum-Aware Pathfinding Costs** (Primary)
   - Tracks momentum state during graph search
   - Makes momentum-preserving paths 30% cheaper
   - Makes momentum-reversing paths 2.5x more expensive
   - No exploitation possible (it's a cost function, not reward bonus)

2. **Demonstration-Derived Waypoints** (Secondary)
   - Extracts "momentum waypoints" from expert demonstrations
   - Routes PBRS through waypoints when momentum is needed
   - Handles complex multi-stage momentum scenarios

## Component 1: Momentum-Aware Pathfinding

### Implementation

Located in: `nclone/graph/reachability/pathfinding_algorithms.py`

**Key Functions:**
- `_infer_momentum_direction()`: Infers momentum from trajectory (grandparent → parent → current)
- `_calculate_momentum_multiplier()`: Adjusts edge costs based on momentum preservation
- Modified `_calculate_physics_aware_cost()`: Applies momentum multiplier to grounded horizontal movement

**Cost Multipliers:**
```python
MOMENTUM_CONTINUE_MULTIPLIER = 0.7   # 30% discount for preserving momentum
MOMENTUM_REVERSE_MULTIPLIER = 2.5    # 2.5x penalty for reversing direction
MOMENTUM_BUILDING_THRESHOLD = 12     # Min displacement (1 sub-node) for momentum
```

**Example:**
```
Path 1 (Direct): spawn → goal
  - No momentum buildup
  - Cost: 100 units

Path 2 (Momentum-building): spawn → left detour → goal
  - Builds momentum going left (0.7x multiplier)
  - Fast approach to goal with momentum (0.7x multiplier)
  - Cost: 80 units (20% cheaper!)
```

### Integration

Momentum tracking is integrated into:
- `PathDistanceCalculator._bfs_distance()`: BFS with grandparent tracking
- `PathDistanceCalculator._astar_distance()`: A* with grandparent tracking
- `bfs_distance_from_start()` in pathfinding_utils.py: Dijkstra with grandparent tracking

**Data structures added:**
- `grandparents = {start: None}`: Tracks parent's parent for momentum inference
- Updated in parallel with `parents` dict during search

## Component 2: Momentum Waypoints

### Extraction

Located in: `nclone/analysis/momentum_waypoint_extractor.py`

**MomentumWaypoint dataclass:**
```python
@dataclass
class MomentumWaypoint:
    position: Tuple[float, float]      # Where momentum-building occurs
    velocity: Tuple[float, float]      # Required velocity at waypoint
    speed: float                       # Speed magnitude
    approach_direction: Tuple[float, float]  # Normalized approach vector
    frame_index: int                   # Frame in demonstration
    leads_to_jump: bool                # Whether followed by jump
    distance_to_goal: float            # Distance to goal at waypoint
```

**Detection Criteria:**
1. Moving away from goal (Euclidean distance increases by >5px)
2. Building velocity (speed increases by >0.8 px/frame)
3. Optionally followed by jump action within 20 frames

**Usage:**
```bash
# Extract waypoints from expert demonstrations
python nclone/tools/extract_momentum_waypoints.py \
    --replay-dir path/to/expert/replays \
    --output-dir momentum_waypoints_cache
```

### PBRS Integration

Located in: `nclone/gym_environment/reward_calculation/pbrs_potentials.py`

**New Functions:**
- `objective_distance_potential_with_waypoints()`: Multi-stage potential routing
- `_find_active_momentum_waypoint()`: Selects waypoint based on agent state

**Waypoint Selection Logic:**
```python
Waypoint is "active" if:
1. Agent hasn't reached it (distance > 20px)
2. Waypoint is between agent and goal
3. Agent doesn't have sufficient momentum (speed < 2.5 px/frame OR wrong direction)
```

**Potential Calculation:**
```python
if active_waypoint:
    # Two-stage routing: current → waypoint → goal
    potential = 1.0 - (dist_to_waypoint / combined_path_distance)
else:
    # Direct routing: current → goal
    potential = 1.0 - (dist_to_goal / combined_path_distance)
```

### Environment Integration

Located in: `nclone/gym_environment/base_environment.py`

**Waypoint Loading:**
- `_load_momentum_waypoints_if_available()`: Loads cached waypoints for current level
- `_update_momentum_waypoints_for_current_level()`: Updates PBRS calculator with waypoints
- Called during `reset()` after map is loaded

**Caching:**
- Waypoints cached per level_id
- Invalidated only when level changes
- Persists across episodes on same level

## Testing

### Unit Tests

Located in: `nclone/gym_environment/tests/test_momentum_aware_pbrs.py`

**Test Coverage:**
1. **Momentum Inference**: Validates trajectory-based momentum detection
2. **Cost Multipliers**: Validates momentum preservation/reversal costs
3. **Waypoint Extraction**: Validates detection of momentum-building segments
4. **Waypoint PBRS**: Validates waypoint routing logic

**Run tests:**
```bash
pytest nclone/gym_environment/tests/test_momentum_aware_pbrs.py -v
```

### Integration Testing

Test on momentum-dependent level:
1. Load level requiring momentum (e.g., sloped jump over mines)
2. Run demonstration through environment
3. Monitor PBRS values during momentum-building phase
4. Verify positive potential change (not negative)

**Expected Behavior:**
- **Without momentum-aware PBRS**: Negative PBRS during backtracking
- **With momentum-aware PBRS**: Positive or neutral PBRS during momentum-building

## Performance

**Computational Overhead:**
- Momentum tracking: +1 dict lookup per node (~0.1ms per pathfinding call)
- Waypoint routing: +2 pathfinding calls when waypoint active (~2-4ms)
- Total overhead: <5ms per step (acceptable for training)

**Memory:**
- Grandparent dict: O(nodes) = ~2000 nodes × 8 bytes = 16KB
- Waypoints cache: ~10 waypoints × 100 bytes = 1KB per level
- Negligible impact on overall memory usage

## Configuration

### Momentum Cost Constants

In `pathfinding_algorithms.py`:
```python
MOMENTUM_CONTINUE_MULTIPLIER = 0.7   # Tune for stronger/weaker momentum preference
MOMENTUM_REVERSE_MULTIPLIER = 2.5    # Tune for stronger/weaker reversal penalty
MOMENTUM_BUILDING_THRESHOLD = 12     # Min displacement to have "momentum"
```

### Waypoint Detection Thresholds

In `momentum_waypoint_extractor.py`:
```python
MIN_SPEED_FOR_MOMENTUM = 1.5          # Min speed to be "momentum"
SPEED_INCREASE_THRESHOLD = 0.8        # Required speed increase
EUCLIDEAN_DISTANCE_INCREASE_THRESHOLD = 5.0  # Min distance increase
LOOKAHEAD_WINDOW = 20                 # Frames to look ahead for jumps
MIN_WAYPOINT_SEPARATION = 50.0        # Min distance between waypoints
```

## Usage Guide

### Step 1: Extract Waypoints from Demonstrations

```bash
# Extract waypoints from your expert replays
python nclone/tools/extract_momentum_waypoints.py \
    --replay-dir /path/to/expert/replays \
    --output-dir momentum_waypoints_cache \
    --verbose
```

This creates cache files: `momentum_waypoints_cache/{level_id}.pkl`

### Step 2: Train with Momentum-Aware PBRS

The environment automatically loads waypoints during training:

```python
# Waypoints are loaded automatically in reset()
env = NppEnvironment(...)
obs, info = env.reset()

# PBRS now uses momentum-aware pathfinding and waypoint routing
# No code changes needed - it's automatic!
```

### Step 3: Monitor PBRS Behavior

Check TensorBoard for momentum-aware metrics:
- `_pbrs_using_waypoint`: Whether waypoint routing is active
- `_pbrs_dist_to_waypoint`: Distance to active waypoint
- `_pbrs_potential_change`: Should be positive during momentum-building

## Troubleshooting

### Waypoints Not Loading

**Symptom:** `No momentum waypoints cache found for level {level_id}`

**Solutions:**
1. Verify cache directory exists: `momentum_waypoints_cache/`
2. Check level_id matches: filename should be `{level_id}.pkl`
3. Run extraction script on demonstrations for this level

### PBRS Still Penalizes Momentum-Building

**Symptom:** Negative PBRS during backtracking despite momentum-aware system

**Debug Steps:**
1. Check if momentum tracking is working:
   - Add logging to `_infer_momentum_direction()`
   - Verify grandparent positions are being tracked
2. Verify cost multipliers are applied:
   - Check `_calculate_momentum_multiplier()` returns 0.7 for continuing momentum
3. Check waypoint routing:
   - Verify `_pbrs_using_waypoint` is True when expected
   - Check waypoint selection logic in `_find_active_momentum_waypoint()`

### Performance Issues

**Symptom:** Pathfinding is slower with momentum tracking

**Solutions:**
1. Momentum tracking adds minimal overhead (<0.1ms)
2. If waypoint routing is slow, reduce number of waypoints per level
3. Disable waypoints for levels that don't need them (automatic if no cache)

## Future Enhancements

### Velocity-Discretized Pathfinding (Not Implemented)

For even more accurate momentum modeling, consider:
- Augmented state space: (x, y, v_x_bin, v_y_bin)
- Discretize velocity into 3-4 bins per axis
- Trade-off: 16x larger graph, but handles all momentum scenarios

**When to implement:**
- If momentum-aware costs insufficient for complex levels
- If waypoint extraction doesn't capture all momentum patterns
- If willing to accept 16x memory increase and 4-8x pathfinding time

## References

- Ng et al. (1999): "Policy Invariance Under Reward Transformations"
- Hamilton et al. (2017): "Inductive Representation Learning on Large Graphs"
- N++ Physics Documentation: `sim_mechanics_doc.md`

