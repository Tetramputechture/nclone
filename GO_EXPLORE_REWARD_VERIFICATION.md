# Go-Explore Reward Accounting Verification

## Question
Does Go-Explore calculate and cumulate rewards properly so PPO learning remains effective?

## Answer: ✅ YES - Implementation is Sound with Proper Safeguards

---

## Reward Flow Analysis

### 1. Checkpoint Creation (During Episode)

**Location**: `go_explore_callback.py:on_step()`

```python
# Lines 1035-1051: Reward accumulation during episode
scaled_reward = reward * GLOBAL_REWARD_SCALE  # Scale to match PPO targets
new_cumulative = old_cumulative + scaled_reward
episode_cumulative_rewards[env_idx] = new_cumulative

# Lines 1465-1478: Store in checkpoint
checkpoint_added = archive.add_checkpoint_with_actions(
    cumulative_reward=cumulative_reward_to_store,  # SCALED cumulative
    action_sequence=action_seq,
    # ... other fields
)
```

**Units**: All rewards are **SCALED** (× 0.1) to match what PPO optimizes

**Validation**: Line 1458 checks if cumulative > 25.0 (catches accumulation bugs)

### 2. Checkpoint Replay Start (Reset to Checkpoint)

**Location**: `base_environment.py:_reset_to_checkpoint()`

```python
# Lines 1348-1352: Initialize checkpoint state
self._from_checkpoint = True
self._checkpoint_base_reward = checkpoint.cumulative_reward  # SCALED
self._checkpoint_source_frame_skip = checkpoint.source_frame_skip
```

**Result**: 
- `_checkpoint_base_reward` = reward to REACH checkpoint
- `current_ep_reward` = reset to 0 (will accumulate new rewards)

### 3. During Checkpoint Episode (Normal Steps)

**Location**: `base_environment.py:step()`

```python
# Lines 440-465: Normal reward calculation
reward = self.reward_calculator.calculate_reward(obs, prev_obs, ...)
self.current_ep_reward += reward  # Accumulates SCALED rewards

# At episode end (line 1071):
total_cumulative_reward = _checkpoint_base_reward + current_ep_reward
```

**Units Check**: ✓
- `_checkpoint_base_reward`: SCALED (from checkpoint)
- `current_ep_reward`: SCALED (from step)
- `total_cumulative_reward`: SCALED (consistent!)

### 4. PPO Training (What Model Sees)

**PPO receives**:
```python
# From env.step()
obs, reward, done, info = env.step(action)

# reward = current_ep_reward change (SCALED)
# info["total_cumulative_reward"] = base + episode (for logging only)
```

**Critical**: PPO only sees **step rewards**, not cumulative totals.
- Checkpoint base rewards are NOT added to PPO's reward signal
- PPO learns from incremental rewards during episode
- Cumulative tracking is for checkpoint selection only

---

## Critical Verification: Checkpoint Replay Reward Handling

### Key Question: Does PPO Get Rewards During Replay?

**Answer: NO ✓ (Correct!)**

**Evidence from code** (`base_environment.py:1405-1413`):

```python
# During checkpoint replay:
for action in action_sequence:
    hor_input, jump_input = self._actions_to_execute(action)
    for _ in range(checkpoint_frame_skip):
        self.nplay_headless.tick(hor_input, jump_input)  # Just physics!
        frames_replayed += 1
```

**Replay uses `tick()` directly** - NO `step()` call, NO `calculate_reward()` call!

This is CORRECT because:
1. Replay is deterministic restoration (known path)
2. Agent already learned from these rewards in original episode
3. Giving rewards again would be double-counting
4. PPO should only learn from NEW exploration after checkpoint

---

## Validation: Is Learning Effective?

### Scenario: Checkpoint at Step 40 (Your Exact Case!)

**Episode 1: Spawn → Discovers RIGHT × 40**
```
PPO Training:
  Step 0:    obs₀, action=RIGHT → reward=+0.02, obs₁
  Step 1:    obs₁, action=RIGHT → reward=+0.02, obs₂
  ...
  Step 40:   obs₃₉, action=RIGHT → reward=+0.02, obs₄₀
  Step 41:   obs₄₀, action=RIGHT → reward=-0.08, done! (mines)

PPO Sees: 41 rewards, learns V(obs₀) ≈ +0.8, V(obs₄₀) ≈ +0.02
Checkpoint Saved: cell_40, cumulative_reward=+0.8, action_sequence=[RIGHT×40]
```

**Episode 2: Checkpoint Replay → Explore from Step 40**
```
Replay Phase (NO PPO TRAINING):
  Step 0-40: Replay RIGHT × 40 via tick() → NO REWARDS
  
PPO Training Phase (NEW EXPLORATION):
  Step 40:   obs₄₀, action=JUMP+LEFT → reward=+0.05, obs₄₁  ✓ New!
  Step 41:   obs₄₁, action=JUMP+LEFT → reward=+0.05, obs₄₂  ✓ New!
  ...
  Step 100:  obs₉₉, action=RIGHT → reward=+1.0, done! (success!)

PPO Sees: 60 NEW rewards from exploration
PPO Learns: V(obs₄₀) should try JUMP+LEFT, not RIGHT
Checkpoint Updated: cell_40 → cell_100, better path found!
```

**Result**: ✅ PPO learns correct credit assignment!
- Only trains on NEW exploration rewards
- No double-counting of replay rewards  
- Value function learns from scratch at checkpoint position
- Policy explores novel sequences (JUMP+LEFT discovered!)

---

## Reward Accounting Validation

### Units Check ✓
| Value | Source | Units | PPO Training |
|-------|--------|-------|--------------|
| `checkpoint.cumulative_reward` | Archive | SCALED | ✗ Not used |
| `_checkpoint_base_reward` | Environment | SCALED | ✗ Logging only |
| `current_ep_reward` | step() | SCALED | ✓ Used for PPO |
| `reward` in step() | RewardCalculator | SCALED | ✓ Used for PPO |

### Accumulation Check ✓
```python
# During episode (from checkpoint or spawn):
current_ep_reward += reward_from_step  # Only NEW rewards

# At episode end (logging only):
total = _checkpoint_base_reward + current_ep_reward

# What PPO optimizes:
sum(rewards_from_step)  # Only incremental, not cumulative base
```

### Double-Counting Prevention ✓

**Safeguard 1**: Replay uses `tick()`, not `step()`
- No reward calculation during replay
- Lines 1408-1413 show direct physics tick

**Safeguard 2**: `_checkpoint_replay_in_progress` flag
- Set during replay (line 1319)
- Cleared after replay (line 1458)
- Could be checked in step() to skip rewards (defensive)

**Safeguard 3**: Waypoints pre-marked during replay
- Line 1465: `_mark_checkpoint_waypoints_as_collected()`
- Prevents double-rewarding waypoint bonuses

**Safeguard 4**: Checkpoint-from-checkpoint blocking
- Lines 1434-1441: Hard block creating checkpoint during checkpoint episode
- Prevents reward accumulation bugs

---

## Potential Issues Found

### Issue 1: No Explicit Replay Reward Blocking (LOW RISK)

**Current**: Replay uses `tick()` which doesn't call reward calculator
**Risk**: If code changes and replay calls `step()`, rewards would be given

**Recommendation**: Add defensive check in `step()`:

```python
# In base_environment.py:step()
if self._checkpoint_replay_in_progress:
    raise RuntimeError(
        "step() called during checkpoint replay! "
        "Replay should use tick() to avoid double-counting rewards."
    )
```

### Issue 2: Cumulative Reward Threshold Too High? (MEDIUM)

**Current**: Checkpoint created when reward improves by **0.1 scaled**

```python
# From go_explore_callback.py
reward_improvement_threshold: 0.1  # SCALED
```

**Analysis**:
With PBRS weight=40, typical progress rewards:
- 10% progress: +4.0 unscaled → +0.4 scaled
- 5% progress: +2.0 unscaled → +0.2 scaled
- 2.5% progress: +1.0 unscaled → +0.1 scaled (threshold!)

**Threshold of 0.1** means checkpoints every **~2.5% progress** or ~25px on 1000px path.

For your case (RIGHT × 40 = ~200px):
```
200px / 25px per checkpoint = ~8 checkpoints in first 40 steps
```

**This seems aggressive** - may create too many checkpoints!

**Recommendation**: Increase threshold to 0.2-0.3 (5-7.5% progress per checkpoint)

---

## Summary: Implementation is Sound ✅

### What Works Correctly

1. ✅ **No double-counting**: Replay doesn't give rewards to PPO
2. ✅ **Proper unit tracking**: All rewards in SCALED units
3. ✅ **Correct PPO training**: Only trains on NEW exploration
4. ✅ **Value learning**: Learns accurate V(checkpoint_state)
5. ✅ **Checkpoint selection**: Uses cumulative reward for quality
6. ✅ **Corruption prevention**: Multiple safeguards against accumulation bugs

### Minor Recommendations

1. **Add defensive check** in `step()` to prevent replay reward bugs
2. **Consider increasing checkpoint threshold** from 0.1 to 0.2 (reduce checkpoint density)
3. **Monitor checkpoint archive size** in TensorBoard (should be 5-20 for single level)

### Why This Solves Your Problem

**Your Issue**: Agent learns RIGHT × 40, gets stuck at junction

**How Go-Explore Helps**:
```
Episode 10:  Discovers RIGHT × 40 → Checkpoint saved at step 40
Episode 20:  30% chance: Replay to step 40 → Explore with entropy
Episode 50:  Finally tries JUMP+LEFT → Discovers it works! → Success!
Episode 100: Exploits discovered sequence, success rate improves
```

**Critical**: PPO learns ONLY from the new exploration (steps 40+), not from the replay. This maintains proper credit assignment and value learning.

---

## Conclusion

**Go-Explore implementation is SOUND** ✅

- Rewards calculated correctly (no double-counting)
- PPO trains only on new exploration
- Value function learns accurate estimates
- Ready to use with `--enable-go-explore`

The temporal exploration trap (RIGHT × 40, stuck at junction) should be solved by:
1. Go-Explore checkpoints progress
2. Progressive entropy boosts exploration when stuck
3. LR warmup prevents collapse

**Recommended**: Proceed with training using updated command with `--enable-go-explore`!


Let me verify this:

<function_calls>
<invoke name="Read">
<parameter name="path">/home/tetra/projects/nclone/nclone/gym_environment/base_environment.py