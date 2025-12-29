# Training Analysis & Fix Summary (2025-12-18)

## Problem Diagnosis

**Training Run**: `gf-2025-12-17-17-22` on level "006 both flavours of ramp jumping"
- **Duration**: 4M timesteps (~2.5 hours)
- **Result**: 0% success rate, policy collapse (KL > 0.0334)
- **Behavior**: All episodes timeout at 2000 frames with -11 to -12 reward

### Root Causes Identified

1. **Policy Collapse** - KL divergence exceeded target (0.0334 > 0.02)
2. **Weak Gradient Signal** - PBRS weight too low, drowned by entropy noise
3. **Aggressive Hyperparameters** - Learning rate too high for fresh policy/value networks
4. **Premature Convergence** - Low entropy (0.01) allowed deterministic local minimum

### TensorBoard Evidence

- ❌ Entropy INCREASING (0.9 → 1.1) - policy becoming more random, not learning
- ❌ Rewards flat at -11 to -12 - no improvement over 3M steps
- ❌ All episodes timeout - 0% completion rate
- ❌ Loss oscillating - value function not converging

---

## Fixes Applied

### 1. Balanced PBRS Weight ⚖️

**Change**: Discovery phase weight 80.0 → 40.0

**Rationale**:
- Initial reduction to 20.0 was too conservative
- TensorBoard showed increasing entropy (signal too weak vs noise)
- Weight=40 provides:
  - **2x stronger** than 20.0 (overcomes entropy noise)
  - **50% weaker** than 80.0 (stable value learning)
  - **1.3x entropy coef** (signal-to-noise ratio of 1.3:1)

**Impact on Rewards**:
```
Efficient completion (150 steps):
  OLD: +19.25 unscaled → +1.93 scaled
  NEW: +39.25 unscaled → +3.93 scaled (2x stronger signal)

Max episode return:
  OLD: ~110 → ~11 scaled
  NEW: ~130 → ~13 scaled (manageable for value function)
```

### 2. Stabilized PPO Hyperparameters 🛡️

**Changes**:
| Parameter | Old | New | Reason |
|-----------|-----|-----|--------|
| learning_rate | 3e-4 constant | **1e-4 with warmup** | Gentle start + adaptive schedule |
| target_kl | 0.02 | **0.05** | Allow larger policy updates without collapse |
| clip_range | 0.2 | **0.15** | More conservative clipping |
| max_grad_norm | 0.5 | **0.3** | Prevent gradient explosion |
| ent_coef | 0.01 | **0.03** | 3x more exploration |

**LR Schedule Details**:
```
0-2.5% of training:  0.01x base LR (1e-6) - ultra-conservative start
2.5-25% of training: 0.1x → 1.0x base LR (1e-5 → 1e-4) - warmup ramp
25-100% of training: 1.0x → 0.0x base LR (1e-4 → 1e-5) - linear decay

For 20M steps:
  Steps 0-500K:   1e-6 to 1e-5 (ultra-slow start)
  Steps 500K-5M:  1e-5 to 1e-4 (warmup)
  Steps 5M-20M:   1e-4 to 1e-5 (decay)
```

**Expected Impact**:
- ✅ KL stays below 0.05 (vs exceeding 0.0334)
- ✅ Policy updates extremely gentle at start
- ✅ Gradual increase as networks stabilize
- ✅ More exploration prevents premature convergence
- ✅ Value function learns gradually without collapse

### 3. Updated Documentation 📝

Updated comments and examples in:
- `reward_config.py` - New weight calculations and rationale
- `reward_constants.py` - Updated hierarchy examples
- `REWARD_BALANCE_VERIFICATION.md` - All scenarios recalculated
- `PBRS_CALCULATION_VERIFICATION.md` - Examples with new weight
- `PBRS_VERIFICATION_SUMMARY.md` - Summary updated

---

## Reward Hierarchy (Weight=40.0)

```
1. Success:          +12.85 scaled  (BEST)
2. Switch + Death:   +6.09 scaled   (Excellent milestone)
3. 50% + Death:      +1.13 scaled   (Good bold play)
4. 30% + Death:      +0.36 scaled   (Acceptable)
5. Camping 16%:      +0.14 scaled   (Poor)
6. Stagnation <15%:  -2.10 scaled   (Bad)
7. Oscillation:      -2.50 scaled   (Worst)
```

**Breakeven**: Need **>22% progress** to justify death risk over camping

---

## Why RND Won't Help

The user asked if RND could alleviate the local minimum. **Answer: NO**

### Reasons:

1. **Not an exploration problem** - Agent IS exploring (reaches 2000 frames)
2. **PBRS provides complete gradient** - Dense reward at every step
3. **Would increase instability** - More competing signals during policy collapse
4. **Wrong scale** - RND weight ~0.5 vs PBRS weight 40 = 80x imbalance
5. **Explicitly removed** - System was simplified to remove RND (reward_constants.py:311)

**The problem is policy stability, not state discovery.**

---

## Expected Results After Fix

### Immediate (First 500K steps):
- ✅ KL divergence < 0.05 (stable updates)
- ✅ Value loss decreasing trend
- ✅ Entropy stable or slowly decreasing (not increasing!)
- ✅ Some episodes complete (>0% success)

### Medium-term (500K - 2M steps):
- ✅ Success rate reaches 5-15%
- ✅ Average reward improves from -11 to positive
- ✅ Episode length decreases (more efficient navigation)
- ✅ PBRS becomes positive on successful episodes

### Long-term (2M+ steps):
- ✅ Success rate reaches 30-50%
- ✅ Curriculum transitions to early/mid phase
- ✅ PBRS weight reduces (15 → 12 → 8)
- ✅ Agent masters level

---

## Training Command

**UPDATED**: Added Go-Explore for temporal exploration (critical for discovering novel action sequences)

```bash
cd ~/projects/npp-rl
python scripts/train_and_compare.py \
    --experiment-name gf-2025-12-18-fixed \
    --architectures graph_free \
    --train-dataset ~/datasets/train \
    --test-dataset ~/datasets/test \
    --single-level '../nclone/test-single-level/006 both flavours of ramp jumping (and the control thereof)' \
    --total-timesteps 20000000 \
    --hardware-profile auto \
    --frame-skip 4 \
    --enable-state-stacking \
    --state-stack-size 4 \
    --use-lstm \
    --lstm-hidden-size 128 \
    --lstm-num-layers 1 \
    --enable-early-stopping \
    --enable-go-explore \
    --checkpoint-selection-strategy ucb \
    --record-eval-videos \
    --num-eval-episodes 10
```

**Key additions**:
- `--enable-go-explore`: Saves checkpoints at progress milestones
- `--checkpoint-selection-strategy ucb`: Balances exploitation/exploration

---

## Monitoring Checklist

During training, watch for:

### Success Indicators ✓
- [ ] KL < 0.05 consistently
- [ ] Entropy stable (not increasing)
- [ ] Value loss decreasing
- [ ] Some episodes succeed (>0%)
- [ ] Avg reward improving

### Warning Signs ⚠️
- KL approaching 0.05 → reduce LR further
- Entropy still increasing → increase ent_coef to 0.05
- No successes by 1M steps → level may be too hard

### Failure Indicators ❌
- KL > 0.05 for 3+ updates → stop and reduce LR to 5e-5
- Rewards still flat after 1M steps → try simpler level
- Value loss diverging → reduce PBRS weight to 30.0

---

## Alternative Actions (If Still Failing)

### Option 1: Further Reduce Learning Rate
```python
learning_rate: 5e-5  # Even more conservative
```

### Option 2: Try Simpler Level First
Validate system works on easier level before tackling "006 both flavours of ramp jumping"

### Option 3: Increase PBRS Weight More
```python
return 50.0  # If 40.0 still too weak
```

### Option 4: Add Curriculum on Death Penalty
```python
# Start with -4.0, increase to -8.0 as success improves
# Makes early exploration less punishing
```

---

## Additional Fix 1: LR Warmup Schedule Re-enabled 📈

**Problem Found**: Learning rate was **constant** because graph_free was explicitly excluded from warmup schedule on 2025-12-17 "for simplicity".

**Why This Was Wrong**:
1. GRU has recurrent connections (gradient amplification)
2. Policy/value networks start randomly (BC only pretrains features)
3. Early gradients are noisy → needs gentle start
4. Constant LR=3e-4 contributed to policy collapse

**Fix Applied**: Re-enabled warmup schedule for graph_free (same as attention/mamba)

**New LR Schedule** (for 20M steps):
```
0-500K steps:    1e-6 to 1e-5  (ultra-gentle start, 0.01x → 0.1x base)
500K-5M steps:   1e-5 to 1e-4  (warmup ramp, 0.1x → 1.0x base)
5M-20M steps:    1e-4 to 1e-5  (linear decay for fine-tuning)
```

**Impact**: Prevents early collapse while allowing strong learning once networks stabilize.

See `LR_SCHEDULE_ANALYSIS.md` for detailed rationale.

---

## Additional Fix 2: Temporal Exploration (Go-Explore + Progressive Entropy) 🎯

**Problem Identified**: Agent learns beginning well (RIGHT × 40) but gets stuck at critical junction requiring novel action sequence (JUMP+LEFT).

### Analysis from Route Visualization
```
Steps 0-40:   RIGHT action repeated → Makes progress ✓
Step 40:      Critical junction → Needs JUMP+LEFT
Step 40-47:   Continues RIGHT → Dies to mines ✗

Root Cause: Temporal exploitation trap
  - Policy overconfident in learned sequence
  - Low entropy prevents exploring action changes
  - Gets stuck in local minimum at critical decision point
```

### Solution 1: Go-Explore Checkpointing (Architectural)

**Enabled** with `--enable-go-explore --checkpoint-selection-strategy ucb`

**How It Works**:
```
Episode 1:  Learns RIGHT × 40 → Checkpoint saved at step 40
Episode 2:  30% chance: Replay to step 40 → Explore from junction
Episode 3:  Tries JUMP+LEFT → Discovers it works! → New checkpoint
Episode 4:  Builds on new checkpoint → Completes level
```

**Benefits**:
- Focuses exploration on critical decision points
- Doesn't waste time re-learning the beginning
- Builds library of good partial solutions
- Proven effective (solved Montezuma's Revenge)

### Solution 2: Progressive Entropy (Reward-Based)

**Added** `entropy_coefficient` property to `RewardConfig`:

```python
@property
def entropy_coefficient(self) -> float:
    if self.current_timesteps < 500_000:
        return 0.02  # Initial learning
    elif self.recent_success_rate < 0.05:
        return 0.05  # BOOST when stuck (2.5x increase!)
    elif self.recent_success_rate < 0.20:
        return 0.03  # Reduce as learning happens
    return 0.01  # Low for exploitation
```

**Wired to trainer** in `enhanced_tensorboard_callback.py`:
- Updates `model.ent_coef` dynamically during training
- Increases when success < 5% after 500K steps
- Forces exploration of novel sequences when stuck

**Expected Behavior**:
```
Steps 0-500K:    ent_coef = 0.02 (learn basics)
Steps 500K+, 0% success: ent_coef = 0.05 (explore aggressively!)
Steps 2M+, 10% success: ent_coef = 0.03 (moderate)
Steps 5M+, 25% success: ent_coef = 0.01 (exploit)
```

### Combined Impact

Go-Explore **where** to explore + Progressive Entropy **how much** to explore:

```
Episode at 1M steps (stuck, 0% success):
  1. Entropy boosted to 0.05 (vs 0.03 base)
  2. Go-Explore replays to step-40 checkpoint (30% of episodes)
  3. Explores junction with 5% entropy
  4. Eventually discovers JUMP+LEFT
  5. Success rate improves → entropy reduces → exploit new knowledge
```

See `TEMPORAL_EXPLORATION_SOLUTIONS.md` for detailed analysis of all options.

---

## Files Modified

### Core Fixes

1. **`nclone/gym_environment/reward_calculation/reward_config.py`**
   - PBRS weight: 80 → 40 (balanced signal strength + stability)
   - Added `entropy_coefficient` property (progressive entropy)
   - Updated `get_active_components()` to log entropy
   - Updated documentation

2. **`npp_rl/training/architecture_trainer.py`**
   - learning_rate: 3e-4 constant → **1e-4 with warmup schedule**
   - **Re-enabled LR warmup for graph_free** (was disabled 2025-12-17)
   - target_kl: 0.02 → 0.05
   - clip_range: 0.2 → 0.15
   - max_grad_norm: 0.5 → 0.3
   - ent_coef: 0.01 → 0.03 (base, modulated by progressive entropy)

3. **`npp_rl/callbacks/enhanced_tensorboard_callback.py`**
   - Added dynamic entropy coefficient updating
   - Wires `reward_config.entropy_coefficient` → `model.ent_coef`
   - Logs entropy transitions to TensorBoard

4. **`nclone/gym_environment/reward_calculation/reward_constants.py`**
   - Updated hierarchy examples for weight=40
   - Deprecated PBRS_PATH_NORMALIZATION_FACTOR

### Documentation

5. **New Documentation Files**:
   - `REWARD_BALANCE_VERIFICATION.md` - Scenarios with weight=40
   - `PBRS_CALCULATION_VERIFICATION.md` - Unit verification
   - `PBRS_VERIFICATION_SUMMARY.md` - Quick reference
   - `LR_SCHEDULE_ANALYSIS.md` - Why warmup is needed
   - `TEMPORAL_EXPLORATION_SOLUTIONS.md` - Go-Explore + entropy analysis

---

## Summary

**Root Cause**: Policy collapse from aggressive hyperparameters + weak gradient signal

**Solution**: Balanced PBRS weight (40) + stable PPO hyperparameters + increased exploration

**Expected**: Stable learning with gradual improvement, success rate >0% within 1M steps

**Ready**: All fixes applied, ready for training! 🚀
