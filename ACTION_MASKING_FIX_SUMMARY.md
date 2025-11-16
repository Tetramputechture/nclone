# Action Masking Bug Fix Summary

## Problem Identified

**Symptom:** During training, a masked action (JUMP+RIGHT, action 5) was selected when only NOOP (action 0) was valid. The validation in `base_environment.step()` caught this and raised a RuntimeError.

**Root Cause:** The action_mask was being **recomputed** when returning cached observations, causing it to change between when the policy selected an action and when the environment validated it.

### Detailed Timeline of the Bug

1. **T=0**: Policy requests observation from environment
   - Environment computes observation with `action_mask = [1, 1, 1, 1, 1, 1]` (all actions valid)
   - Observation is cached in `_cached_observation`
   
2. **T=1**: Policy selects action based on observation
   - Policy sees mask `[1, 1, 1, 1, 1, 1]`
   - Policy applies mask to logits
   - Policy selects action 5 (JUMP+RIGHT) - **valid** according to this mask
   
3. **T=2**: Environment steps with action 5
   - `step()` gets `prev_obs` from cache
   - **BUG**: Cache returns observation but **recomputes action_mask**
   - Ninja state has changed slightly (physics simulation)
   - New mask computed: `[1, 0, 0, 0, 0, 0]` (only NOOP valid now)
   
4. **T=3**: Validation fails
   - `_validate_action_against_mask()` checks if action 5 is valid
   - Mask says only action 0 is valid
   - **RuntimeError raised**: "Action 5 was selected but was masked!"

## The Critical Fix

**File:** `/home/tetra/projects/nclone/nclone/gym_environment/base_environment.py`

**Changed:** Lines 599-622 in `_get_observation()` method

**Before (BUGGY):**
```python
def _get_observation(self) -> Dict[str, Any]:
    """Get the current observation from the game state."""
    # Return cached observation if valid, BUT always recompute action_mask
    if self._cached_observation is not None:
        # CRITICAL: Always recompute action_mask fresh to ensure it reflects current state
        cached_obs = self._cached_observation.copy()
        
        # Recompute action mask and ninja debug state
        ninja_pos = self.nplay_headless.ninja_position()
        switch_pos = self.nplay_headless.exit_switch_position()
        exit_pos = self.nplay_headless.exit_door_position()
        
        cached_obs["action_mask"] = self._get_action_mask_with_path_update(
            ninja_pos, switch_pos, exit_pos
        )
        
        # Update ninja debug state
        ninja = self.nplay_headless.sim.ninja
        if hasattr(ninja, "_mask_debug_state"):
            cached_obs["_ninja_debug_state"] = ninja._mask_debug_state.copy()
        
        return cached_obs
```

**After (FIXED):**
```python
def _get_observation(self) -> Dict[str, Any]:
    """Get the current observation from the game state."""
    # Return cached observation if valid
    # CRITICAL: DO NOT recompute action_mask! It must remain exactly as it was computed.
    # If we recompute it, ninja state might have changed slightly between action selection
    # and validation, causing the mask to change and trigger false positives in
    # masked action detection. The action_mask must stay consistent throughout the step.
    if self._cached_observation is not None:
        return self._cached_observation
```

## Additional Improvements

### 1. Comprehensive Logging (Respecting Debug Mode)

**File:** `/home/tetra/projects/nclone/nclone/gym_environment/base_environment.py`

- Added detailed ninja state logging when masked action is detected
- Made verbose logging conditional on `enable_logging` flag or DEBUG level
- Core error always logged, detailed diagnostics only in debug mode

### 2. Validation Assertions in Policy

**Files:** 
- `/home/tetra/projects/npp-rl/npp_rl/agents/masked_actor_critic_policy.py`
- `/home/tetra/projects/npp-rl/npp_rl/agents/objective_attention_actor_critic_policy.py`

Added validation to ensure:
- `action_mask` is always present in observation dictionaries
- Batch dimensions match between masks and logits
- At least one action is always valid
- No NaN values after masking

### 3. Ninja State Tracking

**File:** `/home/tetra/projects/nclone/nclone/ninja.py`

Added `_mask_debug_state` dictionary to track:
- Ninja position and velocity
- Airborn/walled status
- Buffer states (jump, floor, wall, launch_pad)
- Computed mask and which actions were masked/valid

This enables detailed debugging when masked actions are detected.

### 4. Integration Tests

**File:** `/home/tetra/projects/nclone/tests/test_action_masking_integration.py`

Created comprehensive integration tests:
- Single environment with many steps
- Vectorized environments (DummyVecEnv)
- All actions eventually valid
- Mask consistency checks
- Mask updates after steps
- Masked action detection verification

## Why This Fix Works

### The Principle

**Action mask must be computed ONCE per observation and remain immutable until next step.**

The action_mask represents the valid actions **at the moment the observation is created**. If we recompute it later, we're answering a different question: "What actions are valid NOW?" vs "What actions were valid WHEN this observation was created?"

### The Guarantee

With this fix:
1. Observation is computed with action_mask at time T
2. Same observation (with same mask) is returned from cache
3. Policy selects action using this exact mask
4. Validation uses this same mask
5. **No false positives** because mask never changes

### Cache Invalidation

The cache is properly invalidated after each step:
```python
# In step() method, line 298
self._cached_observation = None
```

This ensures a fresh observation (with fresh action_mask) is computed for the next step.

## Verification

Run the integration tests:
```bash
cd /home/tetra/projects/nclone
python -m pytest tests/test_action_masking_integration.py -v
```

Or test with training:
```bash
cd /home/tetra/projects/npp-rl
python scripts/train_and_compare.py \
    --experiment-name "mask_test" \
    --architectures attention \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 10000 \
    --num-envs 2
```

The masked action error should no longer occur during training.

## Files Modified

1. `/home/tetra/projects/nclone/nclone/ninja.py` - Added debug state tracking
2. `/home/tetra/projects/nclone/nclone/gym_environment/base_environment.py` - Fixed cache bug, improved logging
3. `/home/tetra/projects/nclone/nclone/gym_environment/reward_calculation/main_reward_calculator.py` - Removed redundant validation
4. `/home/tetra/projects/npp-rl/npp_rl/agents/masked_actor_critic_policy.py` - Added validation assertions
5. `/home/tetra/projects/npp-rl/npp_rl/agents/objective_attention_actor_critic_policy.py` - Added validation assertions
6. `/home/tetra/projects/nclone/tests/test_action_masking_integration.py` - New comprehensive tests

## Summary

The bug was subtle but critical: recomputing the action_mask from a cached observation created a race condition where the mask could change between action selection and validation. The fix ensures the action_mask remains immutable once computed, guaranteeing consistency throughout the entire step cycle.


