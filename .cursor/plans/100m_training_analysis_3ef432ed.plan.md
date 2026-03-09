---
name: 100M Training Analysis
overview: Comprehensive analysis of the 100M-step HWM+RND+FiGAR-PPO training run on the rampjump level, synthesizing findings from the crash diagnostics, training log, and 964MB TensorBoard event file. The run completed but hit several critical failures that need addressing before the next attempt.
todos:
  - id: write-analysis-script
    content: "Write scripts/analyze_training_run.py producing 5 outputs: (1) 8 PNG visualization panels, (2) metrics_summary.json with snapshot values at 1M/10M/25M/50M/75M/100M, (3) decision_report.md with quantitative evidence for 8 architectural decisions (D1-D8), (4) correlation analysis between key metric pairs, (5) changepoints.json identifying inflection points in value loss, curriculum success, and RND plateau"
    status: completed
  - id: fix-eval-bug
    content: Fix the evaluation loop 'too many values to unpack (expected 5)' error in architecture_trainer.py - the env.step() API signature changed
    status: completed
  - id: fix-entropy-oscillation
    content: Debug why train/current_entropy oscillates between 0.01-0.05 throughout training instead of following the monotonic HWM schedule (0.05->0.02->0.01)
    status: completed
  - id: fix-value-divergence
    content: "Reduce gradient updates per batch: either lower n_epochs from 6 to 3-4, or increase batch_size to 65536. Consider removing clip_range_vf=0.5 and lowering vf_coef to 0.25"
    status: completed
  - id: fix-curriculum-threshold
    content: Lower stage-0 advancement threshold from 0.50 to 0.35-0.40, or add time-based escalation to advance after N million steps regardless of success rate
    status: completed
  - id: add-time-penalty
    content: Add a small per-step time penalty (-0.0005/step) to the reward function to discourage the 87-96% backtracking/aimless wandering
    status: completed
  - id: compare-earlier-runs
    content: Check 3 earlier runs in npp-logs-220 for different patterns, confirm they are shorter/failed runs of same config or different configs, document in earlier_runs_comparison.md
    status: completed
isProject: false
---

# 100M Training Run Analysis: HWM+RND+FiGAR-PPO on Rampjump

## Executive Summary

The 100M step run completed (~37 hours on GH200) but **failed to produce a viable agent**. Full-level success rate was **0% across all 100M steps**. The goal curriculum never advanced past stage 0 (20% path distance), peaking at only 43% completion even at this easiest stage vs. the 50% threshold needed. The final evaluation also has a **code bug** ("too many values to unpack") that masked the real training performance, making the eval_results.json completely unreliable.

---

## Part 1: Key Findings From Data Already Extracted

### 1A. Success Rate Trajectory (from STUCK CHECK + TensorBoard)

- **Full-level success (`episode/success_rate_full_level`):** 0.0% for the entire 100M steps. The agent never once collected the switch AND reached the exit on the real level.
- **Curriculum success (`goal_curriculum/global_completion_rate`):**
  - 500K steps: 27.2%
  - 5M steps: 33.8% -- rapid initial learning
  - 20M steps: 33.1% -- plateau
  - 47M steps: 42.2% -- **peak**
  - 87M steps: 43.9% -- **all-time best**, slight oscillation
  - 100M steps: 43.3% -- **final value**
- The curriculum required **50% completion** (adaptive threshold for stage 0-2) to advance. Peak was 43.9% -- **never crossed the threshold**.
- `goal_curriculum/global_stage` = **0 for the entire 100M steps**. The agent was ALWAYS training on the 20%-path version of the level.

### 1B. HWM Shaping Metrics

- `**hwm/switch_reach_rate` = 100%** always -- the agent reliably reaches the switch at 20% path distance.
- `**hwm/watermark_switch_min` = 0.0** always -- confirmed switch contact.
- `**hwm/watermark_exit_min` = 0.0** from 5M onwards -- in curriculum stage 0, the exit is close enough that the agent sometimes reaches it.
- `**hwm/episode_total_mean`:** oscillates between 0.59-0.92 (normalized). This means the agent makes 59-92% of maximum possible HWM progress per episode. Healthy signal.

### 1C. Efficiency / Pathfinding (Critical Failure)

- `**efficiency/backtracking_pct` = 87-96%** -- the agent spends the vast majority of steps moving AWAY from the goal. Only 4-13% of steps are forward progress.
- `**efficiency/path_optimality` = 8-16%** -- the agent's path is wildly suboptimal; 6-12x longer than necessary.
- `**efficiency/forward_progress_pct` = 4-12%** -- corroborates the backtracking data.
- This is the **core pathology**: the agent discovered that jumping-right toward the switch earns HWM reward, but it hasn't learned efficient navigation. It stumbles forward through random movement.

### 1D. Value Function Instability

- `**train/value_loss`:** 0.19 at 1M -> **8.08 at 100M** -- monotonically increasing, indicating the value function is **diverging**, not converging.
- `**train/value_loss_ext`:** virtually identical to total value loss (intrinsic value loss is negligible at ~0.000003).
- `**value/estimate_mean`:** oscillates between -0.2 and 1.8 -- unstable, should converge to the expected return.
- The RND intrinsic value head effectively learned nothing useful (value_loss_int near zero throughout), suggesting the dual value head architecture isn't providing the intended benefit.

### 1E. RND Exploration

- `**rnd/intrinsic_reward_mean`:** decays from 0.005 to 0.0005 -- healthy decay pattern.
- `**rnd/beta`:** decays from 0.01 to 0.001 per schedule -- working as designed.
- `**rnd/prediction_error_mean`:** starts at 8.1, drops to ~2.1, then **plateaus at 2.1-2.4 for the last 60M steps** -- the predictor stops learning, suggesting the state space coverage has saturated.
- **Assessment:** RND is working correctly but providing minimal value after ~40M steps.

### 1F. Action Distribution

- **Jump+Right dominates at ~35%** -- the agent learned a strong rightward-jumping bias.
- **NOOP decreased from 17% to 7%** -- the agent learned to stay active.
- **Left at ~10%** -- minimal backward movement as an action (yet 87-96% backtracking by path distance -- suggesting the agent is bouncing/oscillating rather than deliberately walking left).
- `**actions/entropy`:** 1.78 -> 1.62 -- decreasing slowly, indicating some policy specialization.
- **Mine death ratio:** 100% initially -> 61% at end -- still majority mine deaths.

### 1G. FiGAR Macro-Actions

- `**primitive_buffer/mean_duration_frames` = 17-24 frames** -- reasonable macro-action durations.
- `**smdp/avg_steps_per_macro` = 89.04 CONSTANT** -- this is suspicious. Appears to be a logging artifact or static calculation, not reflective of actual macro-action lengths.
- `**primitive_buffer/decision_points` = 5888 CONSTANT** -- 256 envs x 23 decisions per rollout = 5888, which checks out for ~23 macro-actions per 2048-step rollout with ~89 avg duration.

### 1H. Training Infrastructure Issues

- **FPS: ~1600-1900** -- well below the spec target of 10,000 iter/s. The training took 37 hours instead of the ~10 hours expected at target throughput.
- **Adaptive entropy bug:** `train/current_entropy` oscillates between 0.01, 0.02, and 0.05 throughout training, including showing 0.05 at 90M steps when the HWM schedule should have set it to 0.01. The `schedule/entropy_coef` tag shows only 2 events (phase transitions at 41M and 82M). Something is overriding or incorrectly logging the entropy coefficient.
- **96 gradient updates per batch** (flagged as "HIGH" in the log) -- exceeds the recommended 20-80 PPO updates. This causes value function overfitting to stale data.
- **Evaluation bug:** "too many values to unpack (expected 5)" in all 10 eval episodes. The `env.step()` API signature changed but the evaluation loop wasn't updated. This **completely masks the true evaluation performance**.

### 1I. Performance Profile

- Training time: ~37 hours (22:15 Feb 18 -> 11:03 Feb 20)
- ~1,650 FPS average (100M steps / 37 hours)
- GPU memory: 0.04-0.09 GB allocated, 0.06-0.97 GB reserved
- CPU memory: grew from 1.7 GB to ~4.2 GB (leak or buffer growth)

---

## Part 2: Analysis Script Outputs

Write a Python script (`scripts/analyze_training_run.py`) that reads the TensorBoard event file and produces **three categories of output**, all saved to `~/projects/npp-logs-220/analysis/`:

### 2A. Visualization Panels (PNG)

Eight matplotlib figures with smoothed + raw curves:

1. **01_success_rates.png** -- `episode/success_rate_full_level`, `goal_curriculum/global_completion_rate`, `goal_curriculum/global_stage` with threshold line
2. **02_value_function_health.png** -- `train/value_loss_ext`, `value/estimate_mean`, `train/approx_kl`, `train/learning_rate`
3. **03_reward_composition.png** -- `episode/reward_mean`, `hwm/episode_total_mean`, `rnd/intrinsic_reward_mean`, `rnd/beta`
4. **04_navigation_efficiency.png** -- `efficiency/forward_progress_pct`, `efficiency/backtracking_pct`, `efficiency/path_optimality`
5. **05_action_distribution.png** -- stacked area of all 6 action frequencies, `actions/entropy` on secondary axis
6. **06_figar_duration.png** -- `primitive_buffer/mean_duration_frames`
7. **07_entropy_stability.png** -- `train/current_entropy`, `schedule/entropy_coef`, `train/entropy_boost_ratio`
8. **08_death_analysis.png** -- `death/by_cause/mine_ratio`

### 2B. Machine-Readable Metrics Summary (`metrics_summary.json`)

A single JSON file that any future agent can consume to make decisions without re-parsing TensorBoard. Contains snapshot values at key checkpoints (1M, 10M, 25M, 50M, 75M, 100M), peak/trough values with step numbers, trend direction (increasing/decreasing/stable/oscillating), and identified bugs. Key sections:

- `success` -- full_level and curriculum rates with peaks and finals
- `value_function` -- loss values at checkpoints, divergence rate, gradient updates per batch
- `efficiency` -- backtracking/forward progress ranges and trends
- `exploration` -- RND reward decay curve, prediction error plateau step
- `entropy` -- oscillation detected flag, observed vs expected schedule
- `deaths` -- mine ratio trend
- `hwm` -- episode totals, watermark minimums
- `figar` -- duration ranges
- `bugs_found` -- list of identified bugs with descriptions

### 2C. Decision Report (`decision_report.md`)

A structured markdown report with **quantitative evidence for/against each architectural decision**. This is the primary artifact for a future agent. For each decision, provide:

- **Question** -- the precise binary or multi-choice decision
- **Evidence FOR** -- specific metrics with step numbers and values
- **Evidence AGAINST** -- specific metrics supporting the other choice
- **Confidence** -- high/medium/low based on evidence strength
- **Recommendation** -- the recommended choice
- **Expected impact** -- what metric change we expect if followed
- **Verification criterion** -- how to confirm it worked in the next run at the 10M-step mark

Decisions to evaluate:

- **D1: Remove goal curriculum for single-level training?** -- Evidence: stuck at stage 0 for 100M steps; counterfactual: HWM provides gradient on full path
- **D2: Remove dual value heads, use single head?** -- Evidence: value_loss_int = 0.000003 throughout; intrinsic head learned nothing
- **D3: Reduce gradient updates per batch?** -- Evidence: 96 updates; value loss diverges 0.19->8.08; PPO paper recommends 3-30 epochs
- **D4: Add per-step time penalty?** -- Evidence: backtracking 87-96%; path optimality 8-16%; need efficiency incentive
- **D5: Increase death penalty?** -- Evidence: mine deaths = 61%; current -0.5 may be too mild
- **D6: Disable RND after exploration phase?** -- Evidence: prediction error plateaus at 2.2 after 40M; reward magnitude negligible
- **D7: Escalate to IQL?** -- Evidence: switch reach = 100%, watermarks = 0; problem is efficiency not discovery
- **D8: Fix or simplify entropy schedule?** -- Evidence: entropy oscillates erratically; action entropy barely decreased

### 2D. Correlation Analysis (in decision_report.md)

Compute from TensorBoard data to directly inform decisions:

- Curriculum success vs. value loss (expect negative correlation -- higher loss = worse performance)
- HWM episode total vs. curriculum success (expect positive -- more progress = more completions)
- Path optimality vs. curriculum success (expect positive -- efficient paths complete more)
- Mine death ratio vs. curriculum success (expect negative -- mines prevent completion)
- Value loss vs. training step (quantify divergence rate: linear or exponential?)
- Backtracking pct vs. training step (getting worse or better over time?)

### 2E. Changepoint Analysis (`changepoints.json`)

Identify inflection points in key metrics with step numbers and values:

- When did value loss start diverging? (helps link to LR/entropy schedule changes)
- When did curriculum success rate plateau? (identifies saturation point)
- When did RND prediction error plateau? (helps decide when to disable)
- Were there any correlated changepoints across multiple metrics?

---

## Part 3: Root Cause Chain and Priority Ordering

### Causal Model

```
[BUG] Entropy coefficient oscillation (train/current_entropy oscillates 0.01-0.05)
  -> Unstable policy optimization within each rollout
  -> Contributes to value loss divergence

[BUG] 96 gradient updates per batch (n_epochs=6, batch=32768, buffer=524288)
  -> Value function overfits to stale rollout data
  -> Value loss diverges: 0.19 -> 8.08 over 100M steps
  -> Poor advantage estimates -> noisy policy gradient signal

[DESIGN] Goal curriculum stuck at stage 0 for 100M steps
  -> Agent only ever trains on 20% of the level path
  -> Never sees the full switch+exit problem
  -> 0% full-level success rate

[DESIGN] No per-step time penalty
  -> No incentive for efficient paths
  -> 87-96% backtracking; 8-16% path optimality
  -> Agent times out or dies before reaching exit -> 43% max curriculum success

[DESIGN] -0.5 death penalty too mild
  -> 61% of deaths are from mines
  -> Episodes end early from mine collisions instead of progressing to exit
  -> Contributes to low success rate
```

### Recommended Fix Priority (ordered by expected impact on curriculum success rate)

1. **Fix n_epochs (6->3)** -- stabilize value function; biggest single lever for training health
2. **Fix entropy oscillation bug** -- remove source of policy instability; investigate HWMScheduleCallback `_get_progress()` and EnhancedTensorBoardCallback logging conflict
3. **Lower curriculum threshold (0.50->0.35) OR remove curriculum entirely** -- unblock progression past stage 0
4. **Add time penalty (-0.0005/step)** -- incentivize efficiency; expected to reduce backtracking from 90%+ toward 60-70%
5. **Increase death penalty (-0.5->-1.5)** -- reduce mine deaths from 61% toward 30-40%
6. **Remove dual value heads** -- simplify architecture; intrinsic value head is dead weight (loss=0.000003)
7. **Disable RND after 50M steps** -- reduce noise and save compute; prediction error plateaued at 40M
8. **Fix eval bug** -- get reliable evaluation numbers (env.step() API signature mismatch)

### What the Next Run Must Achieve by 10M Steps (Early Stopping Criteria)

A future agent should check these metrics at the 10M-step mark:

- `train/value_loss` should be DECREASING or stable, not increasing. **Target: < 1.0 at 10M**
- `goal_curriculum/global_stage` should be > 0 (curriculum advancing) or irrelevant if curriculum removed
- `efficiency/backtracking_pct` should be < 70% (improvement from time penalty)
- `episode/success_rate_full_level` should be > 0% (agent completes at least some full episodes)
- `train/current_entropy` should follow its schedule monotonically without oscillation
- `death/by_cause/mine_ratio` should be < 50%

**If these are not met at 10M steps, STOP THE RUN** and revisit configuration. Do not burn 90M more steps like this run did.

### Should We Move to IQL / Different Value Function?

Based on the spec's escalation criteria (REWARD_PIPELINE_SPEC.md Section 8):

- Criteria: "< 5% switch collection rate after 30% AND watermark stagnation"
- This run: switch reach = 100%, watermarks = 0 for both objectives
- The problem is **efficiency, not discovery**

**Verdict: IQL is NOT the right escalation.** The agent CAN find the goal -- it just navigates there extremely inefficiently. IQL addresses discovery failures. The fixes above address the actual bottleneck: value function stability, curriculum gating, and navigation incentives.

---

## Part 4: Comparison With Earlier Runs

Check the 3 earlier runs in `~/projects/npp-logs-220/` for patterns. Extract training logs and any TB data. Document in `earlier_runs_comparison.md`:

- Were they the same config or different?
- Did they show similar value loss divergence?
- Did they have different curriculum behavior?
- Were they aborted early due to the bugs now identified?

