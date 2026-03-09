---
name: Implement training run fixes
overview: "Implement the remaining critical recommendations from the 100M-step training analysis: remove dual value heads (D2), increase death penalty (D5), add RND disable schedule (D6), and fix the eval unpack bug -- all justified by quantitative evidence in the analysis artifacts."
todos:
  - id: fix-eval-unpack
    content: "Fix CRITICAL eval unpack bug in architecture_trainer.py line 3500: change 4-value to 5-value unpack from env.step()"
    status: completed
  - id: disable-goal-curriculum
    content: "D1: Completely disable goal curriculum -- set enabled:false in profile, guard callback factory, update verification criteria"
    status: completed
  - id: remove-dual-heads-policy
    content: "D2: Remove intrinsic value head from figar_policy.py (value_net_int, _build_intrinsic_value_head, update forward/evaluate_actions/predict_values returns)"
    status: completed
  - id: remove-dual-heads-ppo
    content: "D2: Update figar_ppo.py to use single value head (remove values_int unpacking, single value loss, single advantage)"
    status: completed
  - id: remove-dual-heads-buffer
    content: "D2: Remove intrinsic fields from primitive_smdp_rollout_buffer.py (old_values_int, advantages_int, returns_int, intrinsic GAE)"
    status: completed
  - id: increase-death-penalty
    content: "D5: Change R_DEATH from -0.5 to -1.5 in reward_constants.py and update reward hierarchy comments"
    status: completed
  - id: add-rnd-disable
    content: "D6: Add RND disable after 50% progress in hwm_schedule_callback.py"
    status: completed
isProject: false
---

# Implement Critical Training Run Recommendations

Based on the analysis in `~/projects/npp-logs-220/analysis/decision_report.md` and supporting artifacts (`metrics_summary.json`, changepoints, and 8 diagnostic charts), the following changes from the 100M-step run `gf-2026-02-18-15-12` still need implementation.

## Already Implemented (No Action Needed)

These items are already reflected in the working tree:

- **D3 (Reduce n_epochs 6->3)**: [graph_free_stability_late_stage_figar.json](configs/hyperparameters/graph_free_stability_late_stage_figar.json) line 13 already has `n_epochs: 3`. Justified by value loss diverging 0.19->8.08 monotonically (chart `02_value_function_health.png`).
- **D4 (Time penalty)**: [reward_constants.py](nclone/gym_environment/reward_calculation/reward_constants.py) line 45 already has `TIME_PENALTY_PER_STEP = -0.0005`. Justified by 87-96% backtracking with 8-16% path optimality (chart `04_navigation_efficiency.png`).
- **D7 (No IQL escalation)**: No action needed. `switch_reach_rate=100%` means the problem is navigation efficiency, not goal discovery.
- **D8 (Entropy oscillation)**: [enhanced_tensorboard_callback.py](npp_rl/callbacks/enhanced_tensorboard_callback.py) lines 462-471 no longer sets `model.ent_coef` (only logs diagnostics). [hwm_schedule_callback.py](npp_rl/callbacks/hwm_schedule_callback.py) is the single owner. Justified by chart `07_entropy_stability.png` showing oscillation between 0.01/0.02/0.05 throughout training.

## Remaining Changes

### 1. CRITICAL BUG: Fix eval unpack error

**File:** [architecture_trainer.py](npp_rl/training/architecture_trainer.py) line 3500

**Evidence:** `metrics_summary.json` `bugs_found[0]`: "env.step() returns 5 values but evaluation loop unpacks 4; all 10 eval episodes fail silently with 0.0 reward". The environment's `step()` returns `(obs, reward, terminated, truncated, info)` per Gymnasium API ([base_environment.py](nclone/gym_environment/base_environment.py) line 812).

**Change:** Line 3500:

```python
# Before:
obs, reward, done, info = env.step(action)

# After:
obs, reward, terminated, truncated, info = env.step(action)
done = terminated | truncated
```

This is a DummyVecEnv wrapping, so `terminated` and `truncated` are arrays. The existing code on line 3504 already does `done = done[0]`, which will continue to work.

### 2. D1: Completely disable goal curriculum

**Evidence:** `goal_curriculum/global_stage`=0 for the entire 100M-step run. The agent peaked at 43.9% curriculum completion (chart `01_success_rates.png`) but the 50% advancement threshold prevented it from ever seeing the full level. `episode/success_rate_full_level`=0% for 100M steps because the agent never trained on the actual problem. `hwm/switch_reach_rate`=100% proves the 20% sub-path was trivially solved -- the agent spent 100M steps on a solved sub-problem.

The decision report recommends "remove curriculum OR use time-based escalation." The lowered threshold (35%) and time-based escalation already in the working tree are a half-measure. The stronger action is to disable it entirely: HWM A* shaping provides gradient signal along the full path, so the agent does not need the curriculum to learn.

**Three locations need changes:**

**a) [graph_free_stability_late_stage_figar.json](configs/hyperparameters/graph_free_stability_late_stage_figar.json)** -- set `enabled: false`:

```json
"goal_curriculum_config": {
    "enabled": false,
    "progress_stages": [0.2, 0.4, 0.6, 0.8, 1],
    "advancement_threshold": 0.85,
    "rolling_window": 5000
}
```

This flows through:

- [environment_factory.py](npp_rl/training/environment_factory.py) line 172: skips multi-stage cache building when `enabled=false`
- [environment_factory.py](npp_rl/training/environment_factory.py) line 568-579: still converts dict to `GoalCurriculumConfig(enabled=False)` and passes to environment
- [base_environment.py](nclone/gym_environment/base_environment.py) line 258: `goal_curriculum_config.enabled` is `False`, so `goal_curriculum_manager` stays `None`
- All code in [npp_environment.py](nclone/gym_environment/npp_environment.py) guarded by `if self.goal_curriculum_manager is not None` safely skips

**b) [callback_factory.py](npp_rl/training/callback_factory.py)** line 243 -- guard the callback creation:

The `_add_goal_curriculum_callback()` call at line 243 is currently unconditional. When curriculum is disabled, the `GoalCurriculumCallback` would still track completion rates and attempt stage advancement calls on environments that have no curriculum manager. Guard it:

```python
# Only add goal curriculum callback if curriculum is enabled
if self.goal_curriculum_config and self.goal_curriculum_config.get("enabled", False):
    self._add_goal_curriculum_callback(callbacks)
```

Also skip the Go-Explore and RND wiring blocks (lines 247-263) since `self.goal_curriculum_callback` will be `None`.

**c) Update the hyperparameter profile metadata** to document the change:

Update the `_metadata.removed_from_previous` list and description to note curriculum was disabled.

**Effect on success metrics:** With curriculum disabled, every level completion counts as `episode/success_rate_full_level` (since `goal_curriculum_manager is None`, `base_environment.py` line 1541-1543 returns `full_level_success = True`). This is the correct behavior -- we want the agent training on the real objective from step 1.

### 3. D2: Remove dual value heads from FiGAR-PPO

**Evidence:** `value_loss_int` was 0.000003 throughout 100M steps (chart `02_value_function_health.png`, blue flat line at zero). The intrinsic value head learned nothing. `rnd/intrinsic_reward_mean` reached 0.0005 by end (negligible). Dual heads add architectural complexity and gradient interference for zero benefit.

Three files need changes:

**a) [figar_policy.py](npp_rl/agents/figar_policy.py)**

- Remove `_build_intrinsic_value_head()` method (lines 189-211) and its call (line 132)
- Remove `value_net_int` from optimizer rebuild (line 137)
- Change `forward()` return from 6-tuple `(actions, duration_idx, values_ext, values_int, ...)` to 5-tuple `(actions, duration_idx, values, ...)` -- remove `values_int`
- Change `evaluate_actions()` return from 6-tuple to 5-tuple -- remove `values_int`
- Change `predict_values()` to return single tensor instead of `(values_ext, values_int)` tuple

**b) [figar_ppo.py](npp_rl/agents/figar_ppo.py)**

- Line 236-243: Unpack 5 values instead of 6 from `self.policy()` (drop `values_int`)
- Line 378-393: Call `compute_returns_and_advantage()` without `last_values_int`
- Lines 456-470: Unpack 5 values from `evaluate_actions()` (drop `values_int`)
- Lines 472-492: Replace dual value loss with single: `value_loss = F.mse_loss(rollout_data.returns, values_pred)`
- Lines 497-500: Use `rollout_data.advantages` only (drop `advantages_int`)
- Lines 627-629: Log single `train/value_loss` (remove `_ext`/`_int` split)
- Line 328-335: Stop storing `value_int` in rollout buffer

**c) [primitive_smdp_rollout_buffer.py](npp_rl/agents/primitive_smdp_rollout_buffer.py)**

- Lines 49-54: Remove `old_values_int`, `advantages_int`, `returns_int` from `SMDPRolloutSamples`
- Lines 201-206: Remove `last_values_int` parameter from `compute_returns_and_advantage()`
- Lines 244-268: Remove intrinsic GAE computation block
- Lines 345-351: Remove intrinsic fields from `_get_samples()` yield

### 4. D5: Increase death penalty from -0.5 to -1.5

**File:** [reward_constants.py](nclone/gym_environment/reward_calculation/reward_constants.py) line 37

**Evidence:** `mine_ratio` plateaued at 61% (chart `08_death_analysis.png`). Correlation `mine_ratio vs curriculum_success` has r=-0.924 (strongest correlation in the dataset), confirming mine deaths are the primary cause of episode failures. Current -0.5 penalty is insufficient to change behavior: the opportunity cost of detouring around mines exceeds the penalty.

**Change:**

```python
# Before:
R_DEATH = -0.5

# After:
R_DEATH = -1.5
```

Also update the reward hierarchy comment on line 29:

```
# Hierarchy: R_EXIT(5.0) >> R_SWITCH(1.0) >> |R_DEATH|(-1.5)
```

And update the reward math comment on lines 32-34 to reflect the new values.

### 5. D6: Add RND disable schedule after 50M steps

**File:** [hwm_schedule_callback.py](npp_rl/callbacks/hwm_schedule_callback.py)

**Evidence:** `rnd_predictor_loss` plateaus at ~2.1-2.4 from 36M steps onward (changepoints.json: `rnd_predictor_plateau` at 36.2M). `rnd/intrinsic_reward_mean` = 0.0005 at end (0.001x initial). RND contributes zero useful signal after 40M steps but still consumes compute (two MLP forward passes per step).

**Change:** In `_on_step()`, after the beta update (line 51-53), add RND disable logic:

```python
RND_DISABLE_PROGRESS = 0.5  # 50% of total = 50M steps at 100M budget

if progress >= RND_DISABLE_PROGRESS and self.rnd_callback is not None:
    if not getattr(self, '_rnd_disabled', False):
        self.rnd_callback.set_beta(0.0)
        self._rnd_disabled = True
        logger.info(f"RND disabled at progress={progress:.2f} (step {self.model.num_timesteps})")
else:
    # Normal beta decay only when RND is active
    beta = 0.01 * (0.1 ** progress)
    if self.rnd_callback is not None:
        self.rnd_callback.set_beta(beta)
```

## Verification Criteria (from decision_report.md)

At the **10M-step mark** of the next run, check:

- `train/value_loss` < 1.0 and decreasing (D3 fix)
- `episode/success_rate_full_level` > 0% (D1 curriculum removed -- every completion is full-level)
- `efficiency/backtracking_pct` < 75% (D4 time penalty)
- `train/current_entropy` monotonically non-increasing (D8 entropy fix)
- `death/by_cause/mine_ratio` < 50% (D5 death penalty)
- `goal_curriculum/`* metrics absent from TensorBoard (D1 curriculum callback removed)

