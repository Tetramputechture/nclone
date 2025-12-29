# Reward System Simplification - Summary

**Date**: 2025-12-15  
**Objective**: Simplify reward system to focus purely on PBRS-based distance reduction

## Rationale

With a collision-accurate adjacency graph providing complete shortest-path information, the reward system had redundant components:

- **PBRS already handles oscillation**: Net-zero displacement = net-zero PBRS reward
- **PBRS already handles exploration**: Complete gradient field - nothing to "discover"  
- **PBRS already handles revisits**: Returning to higher-distance state = negative reward
- **PBRS already provides direction**: F(s,s') rewards any movement that reduces distance

## Final Reward Structure

```
Reward = Terminal + PBRS + TimePenalty

Components:
1. Terminal Rewards (always active)
   - Completion: +50.0
   - Switch activation: +30.0
   - Death: -10.0 to -40.0 (progress-gated)

2. PBRS Shaping (curriculum-scaled)
   - F(s,s') = γ * Φ(s') - Φ(s) where γ = 1.0
   - Φ(s) = 1 - (remaining_physics_cost / total_physics_cost)
   - Weight: 80.0 (discovery) → 5.0 (mastery)

3. Time Penalty (curriculum-scaled)
   - -0.002 (discovery) → -0.03 (mastery) per step
```

## Components Removed

### Exploration Mechanisms (Redundant)
- ❌ **RND (Random Network Distillation)**: Exploration unnecessary with complete gradient field
- ❌ **Go-Explore checkpoints**: No local minima to escape from

### Anti-Oscillation Penalties (Redundant)
- ❌ **Ineffective action penalty**: PBRS gives 0 for stationary behavior
- ❌ **Oscillation penalty**: PBRS gives 0 net reward for oscillation
- ❌ **Revisit penalty**: PBRS penalizes returning to higher-distance states

### Directional Guidance (Redundant)
- ❌ **Velocity alignment reward**: PBRS gradient provides directional signal
- ❌ **Alignment bonus**: Duplicate of velocity alignment

### Path Following (Redundant)
- ❌ **Waypoint bonuses**: PBRS rewards reaching any position that reduces distance
- ❌ **Waypoint approach gradient**: PBRS provides continuous gradient
- ❌ **Exit direction bonus**: PBRS rewards continuing along optimal path

## Files Modified

1. **npp-rl/scripts/lib/training.sh**
   - Removed `--enable-rnd` and all RND configuration flags
   - Removed `--enable-go-explore` and checkpoint selection flags

2. **nclone/gym_environment/reward_calculation/reward_config.py**
   - `revisit_penalty_weight` → returns 0.0 (disabled)
   - `velocity_alignment_weight` → returns 0.0 (disabled)

3. **nclone/gym_environment/reward_calculation/main_reward_calculator.py**
   - Removed `PositionTracker` class (revisit penalties)
   - Removed ineffective action penalty calculation
   - Removed oscillation detection and penalty
   - Removed velocity alignment reward
   - Removed waypoint bonus system
   - Removed alignment bonus
   - Simplified `last_pbrs_components` to core fields only
   - Disabled waypoint extraction methods (renamed with _SIMPLIFIED suffix)

4. **nclone/gym_environment/reward_calculation/pbrs_potentials.py**
   - Removed velocity alignment calculation from `calculate_combined_potential()`

5. **nclone/gym_environment/reward_calculation/reward_constants.py**
   - Marked removed constants as DEPRECATED with explanation

6. **nclone/gym_environment/base_environment.py**
   - Removed waypoint visualization code from `_build_episode_info()`
   - Disabled `_update_path_waypoints_for_current_level()` (now no-op)
   - Disabled `_mark_checkpoint_waypoints_as_collected()` (now no-op)

## Verification

The `last_pbrs_components` dict now contains only:

**Non-terminal steps**:
```python
{
    "pbrs_reward": float,           # F(s,s') = γ * Φ(s') - Φ(s)
    "time_penalty": float,          # Curriculum-scaled efficiency pressure
    "milestone_reward": float,      # Switch activation (30.0 or 0.0)
    "total_reward": float,          # Sum of above
    "is_terminal": False,
    # Plus diagnostic fields for debugging
}
```

**Terminal steps** (death):
```python
{
    "terminal_reward": float,       # Death penalty (progress-gated)
    "milestone_reward": float,      # Switch if activated on death
    "pbrs_reward": 0.0,
    "time_penalty": 0.0,
    "total_reward": float,
    "scaled_reward": float,
    "is_terminal": True,
    "terminal_type": "death",
    # Plus diagnostic fields
}
```

**Terminal steps** (win):
```python
{
    "terminal_reward": float,       # Completion + efficiency bonus
    "milestone_reward": float,      # Switch if activated on win
    "pbrs_reward": 0.0,
    "time_penalty": 0.0,
    "total_reward": float,
    "scaled_reward": float,
    "is_terminal": True,
    "terminal_type": "win",
    # Plus diagnostic fields
}
```

## Expected Benefits

1. **Clearer learning signal**: Every reward component directly relates to distance reduction
2. **Faster training**: No conflicting signals between RND (novelty) and PBRS (goal-directed)
3. **Simpler debugging**: Fewer components to tune and monitor
4. **Theoretical soundness**: Pure PBRS maintains policy invariance guarantees

## Rollback

All removed code is marked with `# SIMPLIFIED:` comments. To restore:
1. Search for `# SIMPLIFIED:` comments
2. Uncomment original code
3. Re-enable flags in training.sh
4. Restore weight properties in reward_config.py

## Next Steps

1. Test simplified system on single level
2. Monitor TensorBoard for cleaner reward signals
3. Verify learning progress without exploration mechanisms
4. If issues arise, investigate temporal credit assignment or action execution, not exploration

