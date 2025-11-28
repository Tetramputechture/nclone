# Reward Structure Analysis - Training Step 1.2M

## Executive Summary

Analysis of replay trajectories at ~1.2M training steps reveals significant **meandering, backtracking, and position oscillation**. While the PBRS-based reward structure is theoretically sound, several implementation issues and parameter imbalances are preventing efficient navigation learning.

**Critical Issues Identified:**
1. ⚠️ **Weak revisit penalties** allow excessive oscillation (10-100+ visits to same cells)
2. ⚠️ **Mine avoidance insufficient** in PBRS pathfinding (2.0× multiplier too weak)
3. ⚠️ **No intermediate waypoints** for long paths (single goal provides sparse gradient)
4. ⚠️ **Time penalty negligible** compared to other signals (~0.02% of episode reward)
5. ⚠️ **Exploration bonus may dominate** early revisit penalties (ratio imbalanced)

---

## Current Reward Structure (at 1.2M steps)

### 1. Terminal Rewards (Always Active)
```python
LEVEL_COMPLETION_REWARD = 20.0      # Success
DEATH_PENALTY = -5.0 to -6.0        # Failure (type-dependent)
SWITCH_ACTIVATION_REWARD = 2.0      # Milestone
```
✅ **Assessment**: Appropriate magnitudes, good differentiation by death type.

### 2. PBRS (Potential-Based Reward Shaping)

**Current Configuration:**
- **Weight**: 100-200 at current success rate (<20%)
- **Normalization**: Capped at 800px (prevents gradient vanishing)
- **Formula**: `Φ(s) = 1 - (distance_to_goal / 800px)`
- **Per-step PBRS**: `F(s,s') = γ * Φ(s') - Φ(s)` where γ=1.0

**Gradient Analysis** (for typical 6px movement with weight=100):
```
ΔΦ = 6px / 800px = 0.0075
PBRS reward = 100 × 0.0075 = 0.75 per step
```

✅ **Strength**: Policy-invariant, dense signal at every step  
⚠️ **Issue 1**: Single goal (switch or exit) provides no guidance through long corridors  
⚠️ **Issue 2**: Mine avoidance in pathfinding too weak (only 2.0× cost multiplier)

### 3. Time Penalty

**Current Values** (with 4-frame skip):
```python
# Per frame penalty
-0.000001 (5-20% success) → -0.000004 per action (4 frames)

# Over typical 600-frame episode:
-0.000001 × 600 = -0.0006 total (0.003% of completion reward!)
```

❌ **Assessment**: **ORDERS OF MAGNITUDE TOO WEAK**  
- Should be at least 100-1000× stronger to matter
- Current penalty: ~-0.001 per episode vs +20 completion reward
- Agent has no incentive to minimize time

### 4. Exploration & Revisit System

**Exploration Bonus** (active when success < 30%):
```python
0.03 per new cell visited (at <10% success)
Max possible: ~29 per episode (966 cells in 42×23 level)
```

**Revisit Penalty** (always active, 100-step sliding window):
```python
Penalty = -0.003 × sqrt(visit_count)

Examples:
- 10 visits: -0.009 (allows navigation)
- 25 visits: -0.015 (mild deterrent)
- 50 visits: -0.021 (moderate)
- 100 visits: -0.030 (matches exploration bonus)
```

⚠️ **Critical Issues**:

1. **Breakeven occurs too late**: Agent must loop 100× before penalty equals exploration bonus
2. **Sqrt scaling too forgiving**: Linear or quadratic would be stronger
3. **Window size may be too small**: 100 steps at 4-frame skip = 25 actions, agent completes loops faster than penalty accumulates

**Observed Behavior in Trajectories**:
- Heavy oscillation in confined corridors (episodes 58, 161 in your data)
- Agent revisits same positions 20-50+ times without strong penalty
- Exploration bonus may encourage "touching every tile" rather than "reach goal efficiently"

---

## Detailed Analysis of Trajectory Failures

### Pattern 1: Corridor Oscillation
**Observation**: Agent oscillates horizontally in narrow passages (X: 100-250 range)

**Root Cause**:
1. PBRS gradient very weak in lateral movement perpendicular to goal
2. Revisit penalty too weak to prevent back-and-forth (need 25+ visits for -0.015)
3. No intermediate waypoints to guide through corridor

**Recommendation**:
- Increase revisit penalty weight: `0.003 → 0.01` (3.3× stronger)
- Change scaling: `sqrt(visits) → visits` (linear, much stronger)
- Add waypoint system for long corridors (see Section 5)

### Pattern 2: Mine-Dense Area Meandering
**Observation**: Agent takes very indirect paths, suggesting poor mine avoidance

**Root Cause**:
1. Mine cost multiplier in A* pathfinding only 2.0× (too weak)
2. Mine radius 40px may be too small for ninja radius (10px) + reaction time
3. Death penalty (-6.0) comparable to PBRS gain from risky shortcut

**Recommendation**:
- Increase mine cost multiplier: `2.0 → 5.0 or 10.0`
- Increase mine hazard radius: `40px → 60-80px` for safer paths
- Consider adding "near-miss" penalty for passing within 20px of deadly mines

### Pattern 3: Backtracking After Progress
**Observation**: Agent makes progress toward goal, then backtracks significantly

**Root Cause**:
1. No "progress preservation" mechanism
2. Revisit penalty only applies within 100-step window (can reset by waiting)
3. PBRS is symmetric (same penalty for backtrack as reward for progress)

**Recommendation**:
- Add episode-level "closest distance to goal" tracking
- Penalty for moving further than previous best: `-0.1 × distance_regressed`
- This breaks PBRS policy invariance but may be necessary for efficiency

---

## Comparison to Best Practices

### PBRS Theory (Ng et al. 1999)
✅ Your implementation is **theoretically correct**:
- Policy invariance holds with γ=1.0
- Dense rewards at every step
- Automatic backtracking penalty via potential decrease

❌ But **practical implementation has gaps**:
- No waypoint/subgoal system for long paths
- Mine avoidance not reflected strongly enough in potential function
- Gradient strength appropriate but guidance density insufficient

### Navigation RL Best Practices

**Standard Techniques You're Missing**:

1. **Waypoint/Subgoal Rewards** ([Human-Informed Subgoals, 2021](https://arxiv.org/abs/2104.06411))
   - Break long paths into intermediate checkpoints
   - Provide rewards for reaching waypoints in sequence
   - Your level has "long path before first goal" - perfect use case

2. **Progress Tracking** (Common in Atari/navigation tasks)
   - Track "furthest progress" in episode
   - Penalize regression beyond certain threshold
   - You removed this for policy invariance, but it may be necessary

3. **Adaptive Exploration** (Curriculum learning standard)
   - Your exploration bonus stays at 0.03 for entire 0-10% range
   - Should decay faster: exponential decay as success improves
   - Consider: `bonus = 0.05 × exp(-10 × success_rate)`

4. **Temporal Difference in Penalties**
   - Revisit penalty should consider *when* revisits occur
   - Rapid oscillation (same cell in consecutive steps) worse than returning after exploration
   - Add recency factor: `penalty = weight × sqrt(visits) × recency_multiplier`

---

## Quantitative Signal Strength Analysis

Let's calculate the relative magnitudes of different reward components over a typical episode:

### Typical Episode (600 frames, ~30% progress before death)

**PBRS Contribution**:
```
Initial distance: 800px (capped normalization)
Final distance: 560px (30% progress)
Distance reduced: 240px

Total PBRS = weight × (240 / 800) = 100 × 0.30 = 30.0
```

**Time Penalty**:
```
600 frames × -0.000001 = -0.0006 (NEGLIGIBLE!)
```

**Exploration Bonus** (assume 200 cells visited):
```
200 cells × 0.03 = 6.0
```

**Revisit Penalty** (assume 50 cells visited average 4 times each):
```
50 cells × (-0.003 × sqrt(4)) = 50 × -0.006 = -0.3
```

**Death Penalty**:
```
-6.0 (hazard death)
```

**Total Episode Reward**: `30.0 - 0.0006 + 6.0 - 0.3 - 6.0 = 29.7`

### Signal Strength Rankings (by magnitude):
1. **PBRS**: ~30 (100% of episode reward) ✅ DOMINANT
2. **Exploration**: ~6 (20%) ⚠️ TOO HIGH relative to revisit
3. **Death**: -6 (20%) ✅ APPROPRIATE
4. **Revisit**: -0.3 (1%) ❌ TOO WEAK
5. **Time**: -0.0006 (0.002%) ❌ IRRELEVANT

### Imbalance Issues:
- **Exploration:Revisit ratio is 20:1** → Agent prefers touching new tiles over efficient paths
- **Time penalty is 50,000× weaker than PBRS** → No time pressure whatsoever
- **PBRS from meandering (6.0 exploration) comparable to PBRS from direct progress** → Exploration can dominate gradient

---

## Specific Recommendations

### Priority 1: Fix Revisit Penalty (HIGH IMPACT)

**Current**:
```python
revisit_penalty = -0.003 × sqrt(visit_count)  # Too weak
```

**Recommended**:
```python
# Option A: Linear scaling (moderate)
revisit_penalty = -0.01 × visit_count

# Option B: Quadratic scaling (strong)
revisit_penalty = -0.005 × visit_count²

# Examples with Option A:
# 5 visits: -0.05 (mild)
# 10 visits: -0.10 (strong deterrent)
# 20 visits: -0.20 (very strong)
# 100 visits: -1.00 (matches exploration bonus much sooner!)
```

**Reasoning**: Current sqrt scaling lets agent oscillate 25-50 times before meaningful penalty. Linear scaling reaches breakeven at 3 visits instead of 100.

### Priority 2: Increase Mine Avoidance in PBRS (HIGH IMPACT)

**Current**:
```python
MINE_HAZARD_COST_MULTIPLIER = 2.0  # Paths near mines 2× more expensive
MINE_HAZARD_RADIUS = 40.0          # Penalty within 40px
```

**Recommended**:
```python
MINE_HAZARD_COST_MULTIPLIER = 8.0  # Much stronger avoidance
MINE_HAZARD_RADIUS = 60.0          # Larger safety margin

# Or for extreme safety (if deaths dominate):
MINE_HAZARD_COST_MULTIPLIER = 15.0
MINE_HAZARD_RADIUS = 80.0
```

**Reasoning**: With 2.0× multiplier, risky paths near mines only cost twice as much. But death penalty is -6.0, which can be offset by PBRS savings from shortcut. At 8-10×, agent will strongly prefer safer routes.

### Priority 3: Add Intermediate Waypoint System (MEDIUM IMPACT)

**Problem**: Single goal (switch or exit) provides no guidance through long corridors.

**Solution**: Identify waypoints along optimal path and provide micro-rewards:

```python
class WaypointRewardSystem:
    """Guide agent through long paths with intermediate checkpoints."""
    
    def __init__(self, waypoint_spacing: int = 200):
        """
        Args:
            waypoint_spacing: Distance between waypoints in pixels (default 200px = ~8 tiles)
        """
        self.waypoint_spacing = waypoint_spacing
        self.waypoints = []
        self.reached_waypoints = set()
    
    def generate_waypoints(self, path: List[Tuple[int, int]]) -> None:
        """Generate evenly-spaced waypoints along optimal path."""
        self.waypoints = []
        cumulative_dist = 0
        last_waypoint = path[0]
        
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            cumulative_dist += (dx*dx + dy*dy)**0.5
            
            if cumulative_dist >= self.waypoint_spacing:
                self.waypoints.append(path[i])
                cumulative_dist = 0
        
        # Always add goal as final waypoint
        self.waypoints.append(path[-1])
    
    def get_waypoint_reward(self, position: Tuple[int, int], threshold: float = 30.0) -> float:
        """
        Check if agent reached next waypoint.
        
        Args:
            position: Current agent position
            threshold: Distance to consider waypoint "reached" (default 30px)
        
        Returns:
            Small reward (0.5) if waypoint reached, else 0.0
        """
        if not self.waypoints:
            return 0.0
        
        next_waypoint_idx = len(self.reached_waypoints)
        if next_waypoint_idx >= len(self.waypoints):
            return 0.0  # All waypoints reached
        
        next_waypoint = self.waypoints[next_waypoint_idx]
        dx = position[0] - next_waypoint[0]
        dy = position[1] - next_waypoint[1]
        distance = (dx*dx + dy*dy)**0.5
        
        if distance <= threshold:
            self.reached_waypoints.add(next_waypoint_idx)
            return 0.5  # Small reward, ~2.5% of completion reward
        
        return 0.0
```

**Integration**:
1. Compute optimal path at episode start (spawn → switch → exit)
2. Generate waypoints every 200px along path
3. Award +0.5 when agent reaches each waypoint in sequence
4. This provides dense guidance without overwhelming PBRS

**Benefits**:
- Breaks long paths into manageable chunks
- Provides gradient in corridors where goal is far away
- Small reward (0.5) doesn't dominate PBRS (~30 per episode)
- Can identify when agent is "off-path" vs "exploring near path"

### Priority 4: Increase Time Penalty (MEDIUM IMPACT)

**Current**:
```python
time_penalty = -0.000001 per frame  # at 5-20% success
# Over 600 frames: -0.0006 total (irrelevant)
```

**Recommended** (gradual ramp-up):
```python
# Phase 1: <20% success - minimal time pressure (focus on completion)
time_penalty = -0.0001 per frame  # 100× stronger
# 600 frames: -0.06 total (~0.3% of completion reward)

# Phase 2: 20-50% success - moderate pressure
time_penalty = -0.001 per frame  # 1000× stronger than current
# 600 frames: -0.6 total (~3% of completion reward)

# Phase 3: >50% success - strong efficiency pressure
time_penalty = -0.005 per frame
# 600 frames: -3.0 total (~15% of completion reward)
```

**Reasoning**: Current penalty is so weak it's effectively disabled. Even at 100× stronger (-0.0001/frame), it only accounts for 0.3% of episode reward, but becomes meaningful at 1000× (-0.001/frame).

### Priority 5: Rebalance Exploration vs Revisit (MEDIUM IMPACT)

**Current Imbalance**:
```
Exploration: +0.03 per new cell
Revisit: -0.003 × sqrt(visits)
Ratio: 10:1 (20:1 after sqrt scaling)
```

**Recommended**:
```python
# Option A: Reduce exploration bonus
exploration_bonus = 0.01  # 3× weaker, matches revisit at 10 visits

# Option B: Faster exploration decay
exploration_bonus = 0.05 × exp(-5 × success_rate)
# At 0% success: 0.05
# At 10% success: 0.03
# At 20% success: 0.018
# At 30% success: 0.011
# At 40% success: 0.007 (becomes negligible)
```

**Reasoning**: Current exploration stays high (0.03) for entire 0-10% range. Exponential decay encourages early exploration but quickly shifts focus to exploitation.

### Priority 6: Consider "Near-Miss" Hazard Penalty (LOW IMPACT)

Add immediate feedback for risky navigation:

```python
def calculate_hazard_proximity_penalty(position, mines, threshold=30.0):
    """
    Penalize passing very close to deadly hazards.
    
    This complements PBRS mine avoidance with immediate feedback.
    """
    min_distance = float('inf')
    for mine in mines:
        if mine['state'] != 0:  # Skip non-deadly mines
            continue
        dx = position[0] - mine['x']
        dy = position[1] - mine['y']
        distance = (dx*dx + dy*dy)**0.5
        min_distance = min(min_distance, distance)
    
    if min_distance < threshold:
        # Linear penalty: -0.02 at mine center, 0.0 at threshold
        proximity_factor = 1.0 - (min_distance / threshold)
        return -0.02 * proximity_factor
    
    return 0.0
```

**Benefits**:
- Immediate feedback (unlike death penalty which requires collision)
- Complements PBRS mine avoidance
- Small magnitude (-0.02 max) doesn't dominate other signals

---

## Curriculum Learning Recommendations

Your current curriculum (reward_config.py) has good structure but needs tuning:

### Phase 1: Discovery (0-20% success, current state)
**Goal**: Learn basic navigation, reach goal occasionally

**Recommended Changes**:
```python
pbrs_objective_weight = 150.0      # Keep high for strong gradient (current: 100-200)
time_penalty = -0.0001             # Increase 100× (current: -0.000001)
exploration_bonus = 0.05 × exp(-5 × success)  # Decay faster (current: flat 0.03)
revisit_penalty = -0.01 × visits   # Linear scaling (current: -0.003 × sqrt)
mine_cost_multiplier = 8.0         # Much stronger (current: 2.0)
```

### Phase 2: Refinement (20-50% success)
**Goal**: Reduce meandering, improve efficiency

```python
pbrs_objective_weight = 80.0       # Still strong guidance
time_penalty = -0.001              # Start mattering (~3% of episode reward)
exploration_bonus = 0.005          # Mostly disabled
revisit_penalty = -0.015 × visits  # Stronger to enforce efficiency
mine_cost_multiplier = 10.0        # Very strong safety
```

### Phase 3: Optimization (>50% success)
**Goal**: Speed runs, optimal paths

```python
pbrs_objective_weight = 40.0       # Reduced, agent knows the way
time_penalty = -0.005              # Strong pressure (~15% of episode reward)
exploration_bonus = 0.0            # Fully disabled
revisit_penalty = -0.02 × visits   # Harsh penalty for any backtracking
mine_cost_multiplier = 15.0        # Extreme safety (never risk death)
```

---

## Implementation Priority

### Immediate (This Training Run)
Can be changed via `reward_config.py` without breaking saved models:

1. ✅ **Increase revisit penalty**: `0.003 → 0.01`, change to linear scaling
2. ✅ **Increase mine avoidance**: `MINE_HAZARD_COST_MULTIPLIER = 2.0 → 8.0`
3. ✅ **Increase time penalty**: `-0.000001 → -0.0001` (100× stronger)
4. ✅ **Adjust exploration decay**: Add exponential decay

### Next Training Run
Requires code changes, will need to retrain from scratch:

5. ⚠️ **Add waypoint system**: Implement intermediate checkpoint rewards
6. ⚠️ **Add near-miss penalty**: Immediate feedback for hazard proximity
7. ⚠️ **Add progress tracking**: Penalty for regressing past best distance

### Future Investigation

8. 🔬 **Dynamic PBRS**: Adjust potential function based on observed behavior
9. 🔬 **Multi-scale PBRS**: Combine long-range (goal) and short-range (obstacles) potentials
10. 🔬 **Bootstrapped shaping**: Use agent's own value function as potential

---

## Testing & Validation Plan

### Step 1: Validate PBRS Gradients (Diagnostic)
```python
# Log these metrics to TensorBoard every 100 steps:
- current_potential (should increase monotonically toward goal)
- pbrs_reward_per_step (should be positive when approaching goal)
- distance_to_goal (should decrease monotonically in successful episodes)
- revisit_count_distribution (histogram of cell visit counts)
```

### Step 2: A/B Test Parameter Changes (Scientific)
Run parallel experiments with different configurations:

**Baseline** (current):
- Revisit: -0.003 × sqrt
- Mine: 2.0×
- Time: -0.000001

**Variant A** (conservative):
- Revisit: -0.005 × sqrt
- Mine: 4.0×
- Time: -0.0001

**Variant B** (aggressive):
- Revisit: -0.01 × visits
- Mine: 8.0×
- Time: -0.001

**Variant C** (aggressive + waypoints):
- Variant B + waypoint system

Compare after 500K steps:
- Success rate
- Average episode length
- Revisit count per episode
- Path optimality (actual path length / optimal path length)

### Step 3: Trajectory Analysis (Qualitative)
For each variant, visualize 10 episodes and check for:
- Reduced oscillation in corridors
- Safer paths around mines
- More direct routes to goal
- Less backtracking

---

## Expected Improvements

### Conservative Estimate (Variants A/B)
- **Oscillation reduction**: 30-50% fewer revisits per episode
- **Path efficiency**: 10-20% shorter paths to goal
- **Success rate**: 5-10% absolute improvement
- **Training time to 50% success**: 20-30% reduction

### Optimistic Estimate (Variant C with waypoints)
- **Oscillation reduction**: 50-70% fewer revisits
- **Path efficiency**: 20-40% shorter paths
- **Success rate**: 10-20% absolute improvement
- **Training time to 50% success**: 40-50% reduction

---

## Conclusion

Your PBRS implementation is theoretically sound but has **practical parameter tuning issues**:

### What's Working ✅
- PBRS provides dense, policy-invariant guidance
- Terminal rewards appropriately scaled
- Curriculum structure is good (phase-based transitions)
- Mine avoidance system architecture is correct

### What Needs Fixing ⚠️
- **Revisit penalty 10-20× too weak** (highest priority fix)
- **Mine avoidance 4-5× too weak** (critical for safety)
- **Time penalty ~1000× too weak** (effectively disabled)
- **Exploration bonus too high relative to penalties** (3-5× too strong)
- **No intermediate waypoints** (limits gradient density in long corridors)

### Immediate Action Items
1. Change `revisit_penalty_weight = 0.01`, use linear scaling
2. Change `MINE_HAZARD_COST_MULTIPLIER = 8.0`
3. Change `time_penalty = -0.0001` (phase 1) to `-0.001` (phase 2)
4. Add exponential decay to exploration bonus
5. Implement waypoint system for next training run

These changes should significantly reduce meandering while maintaining the theoretical benefits of PBRS.

