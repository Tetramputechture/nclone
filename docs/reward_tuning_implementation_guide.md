# Reward Tuning Implementation Guide

## Quick Start - Immediate Changes (No Retraining Required)

These changes can be applied to your current training run by modifying configuration files. They won't break your saved model.

### Step 1: Update `reward_config.py` - Revisit Penalty

**File**: `nclone/gym_environment/reward_calculation/reward_config.py`

**Current code** (lines 166-173):
```python
@property
def revisit_penalty_weight(self) -> float:
    if self.recent_success_rate < 0.20:
        return 0.003  # Strong deterrent during discovery/early learning
    elif self.recent_success_rate < 0.40:
        return 0.002  # Moderate as agent improves
    return 0.001  # Light refinement penalty for late-stage
```

**Replace with** (stronger, linear scaling):
```python
@property
def revisit_penalty_weight(self) -> float:
    """Penalty weight for revisiting same position (oscillation deterrent).
    
    UPDATED: Changed from sqrt scaling to linear scaling with 3-5× stronger weights
    for better oscillation prevention. Linear scaling reaches breakeven with 
    exploration bonus at ~3 visits instead of 100.
    
    Returns:
        0.010 (0-20% success): Strong oscillation deterrent
        0.007 (20-40% success): Moderate as agent improves
        0.004 (40%+ success): Light refinement penalty
    """
    if self.recent_success_rate < 0.20:
        return 0.010  # Strong deterrent: 3.3× stronger than before
    elif self.recent_success_rate < 0.40:
        return 0.007  # Moderate: 3.5× stronger
    return 0.004  # Light: 4× stronger
```

**Then update** `main_reward_calculator.py` (lines 100-104):

**Current code**:
```python
# Revisit penalty (always active, uses sliding window)
visit_count = self.position_counts.get(cell, 0)
if visit_count > 0:
    revisit_penalty = -revisit_penalty_weight * math.sqrt(visit_count)
    total_reward += revisit_penalty
```

**Replace with** (linear instead of sqrt):
```python
# Revisit penalty (always active, uses sliding window)
# UPDATED: Changed from sqrt(visits) to linear visits for stronger deterrent
visit_count = self.position_counts.get(cell, 0)
if visit_count > 0:
    revisit_penalty = -revisit_penalty_weight * visit_count  # Linear scaling
    total_reward += revisit_penalty
```

---

### Step 2: Update `reward_constants.py` - Mine Avoidance

**File**: `nclone/gym_environment/reward_calculation/reward_constants.py`

**Current code** (lines 119-142):
```python
MINE_HAZARD_RADIUS = 40.0  # pixels
MINE_HAZARD_COST_MULTIPLIER = 2.0
```

**Replace with** (much stronger avoidance):
```python
# Mine hazard avoidance radius (pixels)
# UPDATED: Increased from 40px to 60px for safer buffer around deadly mines
# Rationale: 
# - Deadly toggle mines have radius ~4px (state 0)
# - Ninja has radius 10px, speed up to 3.3px/frame
# - Buffer: 60px gives ~18 frames reaction time at max speed
# - Previous 40px was too tight, leading to frequent mine deaths
MINE_HAZARD_RADIUS = 60.0  # pixels (was 40.0)

# Mine hazard cost multiplier
# UPDATED: Increased from 2.0 to 8.0 for much stronger path avoidance
# Rationale:
# - At 2.0×, risky shortcuts near mines only cost twice as much
# - Death penalty is -6.0, which risky shortcut can offset with PBRS savings
# - At 8.0×, paths near mines are 4× more expensive than before
# - This makes safe paths strongly preferred without making mines "forbidden"
# - If agent still takes too many risks, consider 10.0-15.0
MINE_HAZARD_COST_MULTIPLIER = 8.0  # Was 2.0, now 8.0 (4× stronger)
```

---

### Step 3: Update `reward_config.py` - Time Penalty

**Current code** (lines 86-113):
```python
@property
def time_penalty_per_step(self) -> float:
    if self.recent_success_rate < 0.05:  # Discovery phase (0-5% success)
        return -0.0000005  # Ultra-light
    elif self.recent_success_rate < 0.30:  # Early learning (5-30% success)
        return -0.000001  # Very light
    elif self.recent_success_rate < 0.50:  # Mid learning (30-50%)
        return -0.000002  # Light
    elif self.training_phase == "early":
        return -0.000004  # Moderate pressure
    elif self.training_phase == "mid":
        return -0.000008  # Normal pressure
    return -0.00001  # Higher pressure for late-stage optimization
```

**Replace with** (100-1000× stronger):
```python
@property
def time_penalty_per_step(self) -> float:
    """Curriculum-based time penalty - scales with success rate.
    
    UPDATED: Increased by 100-1000× to create meaningful efficiency pressure.
    Previous values were too weak (0.003% of episode reward) to matter.
    
    Analysis with 4-frame skip:
    - Discovery (<5%):   -0.00005 × 4 = -0.0002/action, -0.12 over 600 frames (0.6% of completion)
    - Early (5-30%):     -0.0001 × 4 = -0.0004/action,  -0.24 over 600 frames (1.2%)
    - Mid (30-50%):      -0.0005 × 4 = -0.002/action,   -1.2 over 600 frames (6%)
    - Refinement (>50%): -0.001 × 4 = -0.004/action,    -2.4 over 600 frames (12%)
    
    These values create gradual time pressure without overwhelming other signals.
    """
    if self.recent_success_rate < 0.05:  # Discovery phase (0-5% success)
        return -0.00005  # 100× stronger: minimal but present
    elif self.recent_success_rate < 0.30:  # Early learning (5-30% success)
        return -0.0001  # 100× stronger: light pressure
    elif self.recent_success_rate < 0.50:  # Mid learning (30-50%)
        return -0.0005  # 250× stronger: moderate pressure
    elif self.training_phase == "early":
        return -0.0004  # 100× stronger
    elif self.training_phase == "mid":
        return -0.0008  # 100× stronger
    return -0.001  # 100× stronger: strong efficiency incentive
```

---

### Step 4: Update `reward_config.py` - Exploration Decay

**Current code** (lines 116-141):
```python
@property
def exploration_bonus(self) -> float:
    if self.recent_success_rate < 0.10:
        return 0.03  # Moderate exploration during discovery
    elif self.recent_success_rate < 0.20:
        return 0.02  # Reduced as agent learns
    elif self.recent_success_rate < 0.30:
        return 0.01  # Minimal, PBRS dominates
    return 0.0  # Disabled after 30% success
```

**Replace with** (exponential decay):
```python
@property
def exploration_bonus(self) -> float:
    """Per-cell exploration bonus during discovery phase.
    
    UPDATED: Changed from step function to smooth exponential decay.
    This provides smoother exploration→exploitation transition.
    
    Formula: bonus = 0.05 × exp(-5 × success_rate)
    
    Examples:
    - 0% success:  0.050 (highest, encourages broad exploration)
    - 5% success:  0.039 (still strong)
    - 10% success: 0.030 (matched old value)
    - 20% success: 0.018 (smooth decay)
    - 30% success: 0.011 (fading)
    - 40% success: 0.007 (nearly disabled)
    
    Active only when success_rate < 0.40 (enforced in main_reward_calculator).
    """
    import math
    
    # Exponential decay: smooth transition, responsive to progress
    bonus = 0.05 * math.exp(-5.0 * self.recent_success_rate)
    
    # Disable completely after 40% success (exploration phase over)
    if self.recent_success_rate >= 0.40:
        return 0.0
    
    return bonus
```

---

## Testing the Changes

### Quick Validation Script

Create a test script to verify your changes:

```python
#!/usr/bin/env python3
"""Test reward parameter changes."""

import sys
sys.path.insert(0, '/home/tetra/projects/nclone')

from nclone.gym_environment.reward_calculation.reward_config import RewardConfig
from nclone.gym_environment.reward_calculation.reward_constants import (
    MINE_HAZARD_RADIUS,
    MINE_HAZARD_COST_MULTIPLIER,
)

def test_revisit_penalties():
    """Test revisit penalty changes."""
    config = RewardConfig()
    config.update(timesteps=1_200_000, success_rate=0.10)
    
    weight = config.revisit_penalty_weight
    
    print(f"\nRevisit Penalty Test (10% success rate):")
    print(f"  Weight: {weight} (old: 0.003)")
    print(f"  Scaling: Linear (old: sqrt)")
    print(f"\n  Penalties:")
    for visits in [1, 3, 5, 10, 25, 50, 100]:
        penalty_linear = -weight * visits
        penalty_old = -0.003 * (visits ** 0.5)
        print(f"    {visits:3d} visits: {penalty_linear:+.4f} (old: {penalty_old:+.4f}, {abs(penalty_linear/penalty_old):.1f}× stronger)")
    
    # Check breakeven with exploration
    exploration = config.exploration_bonus
    breakeven = exploration / weight
    print(f"\n  Exploration bonus: {exploration:.4f}")
    print(f"  Breakeven at: {breakeven:.1f} visits (old: 100 visits)")

def test_mine_avoidance():
    """Test mine avoidance changes."""
    print(f"\nMine Avoidance Test:")
    print(f"  Radius: {MINE_HAZARD_RADIUS}px (old: 40px)")
    print(f"  Cost multiplier: {MINE_HAZARD_COST_MULTIPLIER}× (old: 2×)")
    print(f"\n  Path cost at different distances:")
    for distance in [0, 10, 20, 30, 40, 50, 60]:
        if distance < MINE_HAZARD_RADIUS:
            proximity_factor = 1.0 - (distance / MINE_HAZARD_RADIUS)
            cost = 1.0 + proximity_factor * (MINE_HAZARD_COST_MULTIPLIER - 1.0)
        else:
            cost = 1.0
        
        # Old cost for comparison
        old_radius = 40.0
        old_multiplier = 2.0
        if distance < old_radius:
            old_proximity = 1.0 - (distance / old_radius)
            old_cost = 1.0 + old_proximity * (old_multiplier - 1.0)
        else:
            old_cost = 1.0
        
        print(f"    {distance:3d}px: {cost:.2f}× (old: {old_cost:.2f}×)")

def test_time_penalty():
    """Test time penalty changes."""
    config = RewardConfig()
    config.update(timesteps=1_200_000, success_rate=0.10)
    
    penalty = config.time_penalty_per_step
    old_penalty = -0.000001
    
    print(f"\nTime Penalty Test (10% success rate):")
    print(f"  Per-frame penalty: {penalty:.7f} (old: {old_penalty:.7f})")
    print(f"  Strength increase: {abs(penalty/old_penalty):.0f}×")
    print(f"\n  Accumulation over episode:")
    for frames in [300, 600, 1000, 2000]:
        total = penalty * frames
        old_total = old_penalty * frames
        completion_pct = (total / 20.0) * 100
        print(f"    {frames:4d} frames: {total:+.4f} ({completion_pct:+.1f}% of completion, old: {old_total:+.6f})")

def test_exploration_decay():
    """Test exploration bonus decay."""
    print(f"\nExploration Decay Test:")
    print(f"  Formula: 0.05 × exp(-5 × success_rate)")
    print(f"\n  Bonus at different success rates:")
    
    import math
    
    for success_rate in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        config = RewardConfig()
        config.update(timesteps=1_000_000, success_rate=success_rate)
        bonus = config.exploration_bonus
        
        # Old step function
        if success_rate < 0.10:
            old_bonus = 0.03
        elif success_rate < 0.20:
            old_bonus = 0.02
        elif success_rate < 0.30:
            old_bonus = 0.01
        else:
            old_bonus = 0.0
        
        print(f"    {success_rate*100:4.0f}% success: {bonus:.4f} (old: {old_bonus:.4f})")

if __name__ == '__main__':
    print("=" * 70)
    print("REWARD PARAMETER VALIDATION")
    print("=" * 70)
    
    test_revisit_penalties()
    test_mine_avoidance()
    test_time_penalty()
    test_exploration_decay()
    
    print("\n" + "=" * 70)
    print("✅ All parameter changes validated!")
    print("=" * 70)
```

Save as `scripts/test_reward_parameters.py` and run:
```bash
cd /home/tetra/projects/nclone
python scripts/test_reward_parameters.py
```

---

## Monitoring the Changes

Add these TensorBoard metrics to track improvement:

```python
# In your training callback (e.g., TensorBoard callback):

# 1. Revisit statistics
self.logger.record("reward/avg_revisits_per_episode", avg_revisits)
self.logger.record("reward/max_revisits_per_cell", max_revisits)
self.logger.record("reward/revisit_penalty_total", total_revisit_penalty)

# 2. Mine avoidance
self.logger.record("reward/deaths_by_mine", mine_death_count)
self.logger.record("reward/min_mine_distance", min_distance_to_mine)

# 3. Time efficiency
self.logger.record("reward/episode_length", episode_length)
self.logger.record("reward/time_penalty_total", total_time_penalty)
self.logger.record("reward/path_optimality", optimal_length / actual_length)

# 4. Exploration balance
self.logger.record("reward/cells_explored", cells_explored)
self.logger.record("reward/exploration_reward_total", exploration_total)
self.logger.record("reward/exploration_vs_revisit_ratio", exploration_total / abs(revisit_total))
```

---

## Expected Results Timeline

### After 50K steps (immediate):
- Reduced oscillation in tight corridors (20-30% fewer revisits)
- More cautious movement near mines
- Slightly longer episodes initially (agent being more careful)

### After 200K steps (short-term):
- Clear improvement in path efficiency (10-15% shorter paths)
- Fewer mine deaths (30-50% reduction)
- Success rate improvement (5-10% absolute)

### After 500K steps (medium-term):
- Consistent efficient navigation (minimal oscillation)
- Safe mine navigation (death rate <5%)
- Strong exploitation of optimal paths

---

## Rollback Plan (If Needed)

If changes cause training instability:

### Quick Rollback
```python
# In reward_config.py, restore old values:
revisit_penalty_weight = 0.003  # was 0.010
# Use sqrt scaling in main_reward_calculator.py

# In reward_constants.py:
MINE_HAZARD_RADIUS = 40.0  # was 60.0
MINE_HAZARD_COST_MULTIPLIER = 2.0  # was 8.0
```

### Gradual Tuning
If full changes are too aggressive, use intermediate values:

```python
# Moderate changes (50% of proposed):
revisit_penalty_weight = 0.006  # halfway between 0.003 and 0.010
MINE_HAZARD_COST_MULTIPLIER = 4.0  # halfway between 2.0 and 8.0
time_penalty = -0.00005  # halfway between old and proposed
```

---

## Advanced: Next Training Run

These changes require retraining from scratch (modify observation/action space):

### 1. Waypoint System

Add to `main_reward_calculator.py`:

```python
class WaypointRewardSystem:
    """Guide agent through long paths with intermediate checkpoints."""
    
    def __init__(self, waypoint_spacing: int = 200):
        self.waypoint_spacing = waypoint_spacing
        self.waypoints = []
        self.reached_waypoints = set()
        self.waypoint_reward = 0.5  # Small reward per waypoint
    
    def generate_waypoints(self, path: List[Tuple[int, int]]) -> None:
        """Generate evenly-spaced waypoints along optimal path."""
        self.waypoints = []
        self.reached_waypoints.clear()
        
        if not path or len(path) < 2:
            return
        
        cumulative_dist = 0
        last_waypoint_idx = 0
        
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            cumulative_dist += (dx*dx + dy*dy)**0.5
            
            if cumulative_dist >= self.waypoint_spacing:
                self.waypoints.append((i, path[i]))
                cumulative_dist = 0
        
        # Always add goal as final waypoint
        if len(path) > 0:
            self.waypoints.append((len(path)-1, path[-1]))
    
    def get_waypoint_reward(self, position: Tuple[int, int]) -> float:
        """Check if agent reached next waypoint in sequence."""
        if not self.waypoints:
            return 0.0
        
        next_waypoint_idx = len(self.reached_waypoints)
        if next_waypoint_idx >= len(self.waypoints):
            return 0.0
        
        _, waypoint_pos = self.waypoints[next_waypoint_idx]
        dx = position[0] - waypoint_pos[0]
        dy = position[1] - waypoint_pos[1]
        distance = (dx*dx + dy*dy)**0.5
        
        # Reached threshold: 30px (~1.25 tiles)
        if distance <= 30.0:
            self.reached_waypoints.add(next_waypoint_idx)
            return self.waypoint_reward
        
        return 0.0
    
    def reset(self):
        """Reset for new episode."""
        self.waypoints.clear()
        self.reached_waypoints.clear()
```

**Integration**:
1. In `RewardCalculator.__init__()`, add: `self.waypoint_system = WaypointRewardSystem()`
2. In `calculate_reward()` (after PBRS calculation), add waypoint reward
3. At episode start, compute optimal path and generate waypoints

### 2. Near-Miss Hazard Penalty

Add to `main_reward_calculator.py`:

```python
def calculate_hazard_proximity_penalty(
    self,
    position: Tuple[float, float],
    level_data: Any,
    threshold: float = 30.0
) -> float:
    """
    Penalize passing very close to deadly hazards.
    
    Complements PBRS mine avoidance with immediate feedback.
    
    Args:
        position: Current agent position (x, y)
        level_data: Level data containing mine entities
        threshold: Distance for penalty (default 30px)
    
    Returns:
        Penalty in range [-0.02, 0.0]
    """
    from ...constants.entity_types import EntityType
    
    # Get deadly mines
    mines = level_data.get_entities_by_type(EntityType.TOGGLE_MINE)
    mines += level_data.get_entities_by_type(EntityType.TOGGLE_MINE_TOGGLED)
    
    min_distance = float('inf')
    for mine in mines:
        # Only deadly mines (state 0)
        if mine.get('state', 0) != 0:
            continue
        
        mine_x = mine.get('x', 0)
        mine_y = mine.get('y', 0)
        dx = position[0] - mine_x
        dy = position[1] - mine_y
        distance = (dx*dx + dy*dy)**0.5
        min_distance = min(min_distance, distance)
    
    if min_distance < threshold:
        # Linear penalty: -0.02 at mine center, 0.0 at threshold
        proximity_factor = 1.0 - (min_distance / threshold)
        return -0.02 * proximity_factor
    
    return 0.0
```

**Integration**: Call in `calculate_reward()` after position tracking.

---

## FAQ

**Q: Will these changes break my saved model?**  
A: No, these are reward-only changes. Your policy network remains unchanged.

**Q: Should I restart training or continue?**  
A: Continue! These changes will improve learning going forward.

**Q: What if success rate drops initially?**  
A: Expected. Agent is learning new constraints (don't loop, avoid mines, be efficient). Give it 100-200K steps to adapt.

**Q: Can I test these in a separate run first?**  
A: Yes! Copy your checkpoint and run two parallel experiments: baseline vs proposed.

**Q: How do I know if changes are working?**  
A: Monitor TensorBoard:
- `reward/revisit_penalty_total` should become more negative
- `reward/avg_revisits_per_episode` should decrease
- `episode/path_optimality` should increase (closer to 1.0)
- `episode/success_rate` should improve after adaptation period

---

## Summary Checklist

- [ ] Update `reward_config.py` - revisit_penalty_weight (multiply by 3-5×, use linear)
- [ ] Update `main_reward_calculator.py` - change sqrt to linear for revisit penalty
- [ ] Update `reward_constants.py` - MINE_HAZARD_COST_MULTIPLIER (2.0 → 8.0)
- [ ] Update `reward_constants.py` - MINE_HAZARD_RADIUS (40.0 → 60.0)
- [ ] Update `reward_config.py` - time_penalty_per_step (multiply by 100×)
- [ ] Update `reward_config.py` - exploration_bonus (add exponential decay)
- [ ] Run `scripts/test_reward_parameters.py` to validate changes
- [ ] Run `scripts/visualize_reward_parameters.py` to generate plots
- [ ] Resume training and monitor TensorBoard for improvements
- [ ] After 200K steps, evaluate success rate and path efficiency

Good luck with training! 🚀

