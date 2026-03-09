# SAC + RFCL: Soft Actor-Critic with Reverse-Forward Curriculum Learning

## Overview

This document describes the SAC+RFCL approach to training the N++ ninja agent. It combines:

- **SAC-Discrete** (Soft Actor-Critic for discrete action spaces) as the core off-policy RL algorithm
- **RFCL** (Reverse-Forward Curriculum Learning, Tao et al. ICLR 2024) to solve the sparse reward discovery problem
- **RLPD** (Reinforcement Learning with Prior Data, Ball et al. 2023) demo mixing to seed the replay buffer

The central challenge in N++ is that the reward signal is extremely sparse: a typical level requires hundreds of frames of coordinated movement before any reward is received. SAC alone cannot discover this signal from random play. RFCL solves this by constructing a sequence of starting states that progress backward from the goal to the spawn, giving the agent a continuous stream of achievable sub-problems.

**Source files:**
- [`npp-rl/npp_rl/training/sac_trainer.py`](../../npp-rl/npp_rl/training/sac_trainer.py)
- [`npp-rl/npp_rl/training/rfcl_curriculum.py`](../../npp-rl/npp_rl/training/rfcl_curriculum.py)
- [`nclone/nclone/gym_environment/sac_environment.py`](../nclone/gym_environment/sac_environment.py)
- [`nclone/nclone/gym_environment/reward_calculation/sparse_reward_calculator.py`](../nclone/gym_environment/reward_calculation/sparse_reward_calculator.py)
- [`nclone/nclone/gym_environment/reward_calculation/dense_reward_calculator.py`](../nclone/gym_environment/reward_calculation/dense_reward_calculator.py)
- [`nclone/nclone/gym_environment/reward_calculation/reward_constants.py`](../nclone/gym_environment/reward_calculation/reward_constants.py)
- [`npp-rl/npp_rl/feature_extractors/sac_feature_extractor.py`](../../npp-rl/npp_rl/feature_extractors/sac_feature_extractor.py)
- [`npp-rl/npp_rl/feature_extractors/level_context_encoder.py`](../../npp-rl/npp_rl/feature_extractors/level_context_encoder.py)

---

## Algorithm

### SAC-Discrete

SAC-Discrete extends the continuous SAC algorithm (Haarnoja et al. 2018) to discrete action spaces following the formulation of Christodoulou (2019). The core objective balances expected return against an entropy bonus:

```
J(π) = E[Σ_t γ^t (r_t + α H(π(·|s_t)))]
```

where `α` (temperature) is automatically tuned to maintain a target entropy:

```
target_entropy = -target_entropy_ratio * log(1 / |A|)
target_entropy_ratio = 0.5
```

Two separate Q-networks (`Q1`, `Q2`) are maintained with target networks updated via Polyak averaging (`τ = 0.005`). The policy minimizes the KL divergence to the soft Q-function:

```
π* = argmin_π E[α log π(a|s) - Q(s,a)]
```

**Key hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| `gamma` | 0.99 | Standard long-horizon discount |
| `tau` (Polyak) | 0.005 | Slow target network smoothing for stability |
| `lr_q / lr_policy / lr_alpha` | 3e-4 | Adam learning rate (Ball et al. 2023 recommendation) |
| `target_entropy_ratio` | 0.5 | 50% of max entropy; tuned empirically |
| `utd_ratio` | 4 | 4 gradient updates per environment step |
| `utd_ratio` (warmup) | 2 | Halved for first 500k steps to prevent Q-overestimation |
| `batch_size` | 256 | Standard RLPD batch |
| `buffer_size` | 1,000,000 | 1M transition replay buffer |
| `min_buffer_size` | 10,000 | Minimum transitions before gradient updates begin |
| `initial_random_steps` | 5,000 | Random exploration per env to seed the buffer |
| `n_envs` | 32 | Parallel environments distributed across levels |
| `total_steps` | 100,000,000 | Total environment steps budget |

**UTD warm-up justification:** During the first 500k steps the buffer is nearly homogeneous with demo-like transitions. UTD=4 on a homogeneous buffer causes aggressive Q-overestimation before the policy has seen enough diversity. Halving UTD to 2 during warm-up reduces this without sacrificing sample efficiency later.

### Action Space: Semi-Markov Decision Process

The action space is `Discrete(48)`, encoding `(base_action, duration)` pairs:

```
action_index = base_action * 8 + (duration - 1)
```

| Axis | Values |
|---|---|
| Base actions (6) | NOOP, Left, Right, Jump, Jump+Left, Jump+Right |
| Durations (8) | 1–8 primitive frames |

This forms a Semi-Markov Decision Process (SMDP): the agent decides on both what to do and for how long at each decision point. The effective action space is much richer than 6 base actions alone, enabling momentum-based movement strategies without requiring the policy to output individual frames.

**Actual duration tracking:** Macro-actions can be cut short by contact state changes (grounded ↔ airborne ↔ wall-sliding). The re-encoded action index uses the *actual* duration executed — not the intended duration — to ensure the replay buffer stores accurate transitions:

```python
actual_action_idx = intended_base * 8 + (actual_dur - 1)
```

### RLPD Demo Mixing

Expert replay files are converted into buffer transitions and stored in a separate `demo_buffer`. During batch sampling, a fraction of each batch is drawn from demos:

```
demo_fraction = initial_demo_ratio * decay_schedule(t)
initial_demo_ratio = 0.15  (15%)
```

**Stall recovery:** When the level scheduler detects a stall (no progress for extended periods), demo fraction is temporarily boosted and position/velocity noise is injected into RFCL resets to escape local optima.

### Pre-Reset Terminal Observation

A subtle but critical implementation detail: when an episode ends due to a curriculum reset, the **pre-reset terminal observation** is stored in the replay buffer rather than the post-reset observation. Storing the post-reset observation would inject noise into Q-targets — the Q-value would be bootstrapped from a curriculum-reset state that is not actually reachable from the episode's final state.

---

## Reward Function

The total reward at each macro-action step combines sparse and optional dense components:

```
r_total = r_sparse + r_HWM + r_dense
```

### Sparse Rewards

Defined in `sparse_reward_calculator.py`. The design philosophy is minimal: "No distance shaping, no HWM. The RFCL curriculum provides the learning gradient via initial state distribution." Reward is a pure task signal; exploration shaping is delegated entirely to the curriculum.

| Event | Reward | Justification |
|---|---|---|
| Exit switch activated | +1.0 | Intermediate milestone; distinguishable from background noise |
| Exit door reached | +5.0 | Primary task signal; 5× switch to unambiguously distinguish phases |
| Death (any cause) | -0.5 | Risk aversion without paralysis |
| Per primitive frame | -0.001 | Time pressure: 500 frames = −0.5 (≈ one death) |

**Death penalty tuning history:** The original value was −1.5. Run-221 analysis showed this caused severe risk-aversion: 37% NOOP rate, 95% mine deaths (vs. 61% with −0.5), and a death/HWM ratio of 1500:1 that made exploration irrational. At −0.5, the death penalty no longer dominates the HWM learning signal.

### High-Water-Mark (HWM) Shaping

Defined in `hwm_shaping.py` and weighted by `HWM_ALPHA` in `reward_constants.py`.

HWM is a non-negative, non-exploitable distance-based shaping mechanism originally validated on momentum-based platformers by the Sonic OpenAI Retro Contest (2018). It fires only when the agent achieves a **new minimum A\* distance** to the current goal:

```
hwm_progress = max(0, watermark_prev - dist_current)
r_HWM = HWM_ALPHA * hwm_progress / initial_dist
```

**Properties:**
- **Non-negative:** backtracking yields zero reward, not a penalty
- **Bounded cumulative:** total HWM ≤ `HWM_ALPHA` per phase (normalized by initial distance)
- **Non-exploitable:** cycling distance down→up→down yields reward only on the first crossing

**Two-phase structure:**
- Phase 0 (pre-switch): watermark tracks A\* distance to the exit switch
- Phase 1 (post-switch): watermark resets and tracks A\* distance to the exit door
- On phase transition: `watermark_exit = dist_exit_now`; shaping reward = 0 (prevents discontinuity from switching distance metrics)

**HWM_ALPHA tuning history:** Originally 1.0. After run-221's death penalty reduction, HWM_ALPHA was raised to 3.0 to maintain signal strength. At HWM_ALPHA=3.0, the death/HWM ratio at stage-0 is approximately 28:1 (was 83:1 at ALPHA=1.0), making the learning signal substantially more tractable.

**Why HWM over classical PBRS:** Standard PBRS (`F = γΦ(s') - Φ(s)`) penalizes the agent when it moves away from the goal, even when doing so is necessary to build momentum for a wall-jump or running start. HWM's non-negative property eliminates this anti-momentum bias entirely.

**Full episode reward trajectory** (successful level completion):

```
R_SWITCH(1.0) + HWM_switch(~3.0) + R_EXIT(5.0) + HWM_exit(~3.0) ≈ 12.0
```

### Dense Shaping Components

Defined in `dense_reward_calculator.py`. All four components are intentionally small relative to `R_EXIT=5.0` — they act as "learning accelerants, not dominant objectives." The reward calculator has two call sites:

- `compute_frame_reward()` — called every primitive frame inside the macro-action loop
- `compute_macro_pbrs()` — called once after the full macro-action completes

#### Component 1: Macro-Level Progress Shaping (PBRS variant)

```
r_pbrs = alpha_pbrs * max(0, dist_before_macro - dist_after_macro)
alpha_pbrs = 0.5
```

Computed **once per macro-action** (not per frame). Positive-only: rewards progress but gives exactly zero for standing still or backtracking.

**Justification for positive-only over standard PBRS:** Standard PBRS `F = α(γΦ(s') - Φ(s))` at constant distance yields `α * (1−γ) * d` per step. At `α=0.5`, `d=0.9`: `0.5 * 0.01 * 0.9 = 0.0045/step`. This nearly cancels the −0.001 step penalty, making idle timeout profitable. Progress-only PBRS eliminates this idle bias entirely.

**Phase transition skip:** Skipped on macro-actions where the exit switch was activated. At that moment the distance metric switches from `dist_to_switch` to `dist_to_exit`, creating a large spurious discontinuity in the shaping signal.

#### Component 2: Velocity-Direction Alignment

```
r_vel = beta_vel * max(0, dot(v_unit, next_hop_dir))
beta_vel = 0.0001
```

Rewards the agent for moving in the direction indicated by the A\* graph's next-hop direction vector. Positive-only: setup movements (wall-jump approach, momentum building) are never penalized.

**Fallback:** When `next_hop_dir = (0, 0)` (BFS cache miss after an RFCL reset to a state not in the graph), fires `beta_vel * speed * path_progress` instead — rewarding fast movement that is also closing distance.

**Beta reduction history:** Originally 0.0005. Run-38 analysis showed that at this value, a 10k-frame timeout episode accumulated ~1.5 in velocity reward from random movement alone — approximately 35% of the total episode reward budget, causing velocity reward to dominate HWM. Reduced 5× to 0.0001.

#### Component 3: Distance Milestones

```
r_milestone = milestone_bonus  (once per new 10% progress threshold)
milestone_bonus = 0.1
milestone_interval = 0.1
```

Progress ratio: `1 - dist_current / initial_dist`. Each milestone (0%, 10%, 20%, ...) is awarded at most once per episode, tracked in `_milestones_crossed`. Prevents exploitation by oscillation: moving back and forth across a threshold yields one reward total.

**Interval reduction history:** Originally 0.2 (20% intervals). Run-38 showed that with RFCL near-goal starts, milestone 0 was crossed but milestones 1–4 were never reached in a single episode. 20% intervals were too coarse to provide density. Reduced to 0.1.

#### Component 4: Mine Hazard Avoidance

```
r_hazard = delta_hazard * max(0, sdf_curr - sdf_prev)  [only when sdf_prev < threshold]
delta_hazard = 0.0001
hazard_sdf_threshold = 0.3
```

Uses the global mine SDF (Signed Distance Field) value from `mine_sdf_features[0]`. Only fires when the agent is already **inside the mine danger zone** (`sdf < 0.3`). Rewards moving away from mines (SDF increasing) but does not reward simply being far away.

**Why conditional activation:** A reward that fires based on absolute distance to mines rewards all movement away from mines, including correct movement away from non-threatening mines. The threshold ensures the signal is only active when the agent is actually at risk, making it a targeted escape incentive rather than a global spatial preference.

### Dense Component Summary

| Component | Formula | Weight | Call site |
|---|---|---|---|
| Progress PBRS | `max(0, dist_before - dist_after)` | α=0.5 | Per macro-action |
| Velocity alignment | `max(0, dot(v_unit, next_hop_dir))` | β=0.0001 | Per primitive frame |
| Distance milestones | `+0.1` per new 10% threshold | 0.1 | Per primitive frame |
| Mine hazard avoidance | `max(0, sdf_curr - sdf_prev)` if sdf<0.3 | δ=0.0001 | Per primitive frame |

---

## Observation Features

The observation space is a **flat 119-dimensional float32 vector**. No images, no graphs. This compact representation was chosen for three reasons:
1. SAC's replay buffer scales linearly with observation size — images would require a ~50× larger buffer
2. Ball et al. 2023 (RLPD) demonstrates that compact vector observations are sufficient when combined with high UTD and demo mixing
3. The required geometric context is already computable from the physics simulation without rendering

```
game_state       [dims 0..23]    24 dims
reachability     [dims 24..31]    8 dims
spatial_context  [dims 32..115]  84 dims
mine_sdf         [dims 116..118]  3 dims
```

### game_state (24 dims)

Complete description of the ninja's physical state, normalized to [−1, 1]:

| Feature group | Dims | Description |
|---|---|---|
| Velocity | 2 | `xspeed`, `yspeed` (clipped to `[-MAX_HOR_SPEED, MAX_HOR_SPEED]`) |
| Movement state (one-hot) | 4 | Current locomotion mode (grounded, airborne, wall-sliding, etc.) |
| Airborne flag | 1 | Binary; critical for jump timing |
| Action buffers | 3 | Jump buffer, floor buffer, wall buffer (coyote time / input tolerance windows) |
| Jump duration remaining | 1 | Frames of jump hold remaining |
| Contact counts | 3 | Floor contact count, wall contact count, floor count |
| Surface normals | 3 | Floor normal (x, y) and wall normal |
| Walled flag | 1 | Binary; agent is touching a wall |
| Applied physics | 3 | Applied gravity, drag, friction (per-frame values) |
| Position | 2 | `xpos / LEVEL_WIDTH`, `ypos / LEVEL_HEIGHT` |
| Time remaining | 1 | Normalized frames remaining in episode |

**Version history:** Reduced from 26 dims in Feb 2026 by removing `hor_input` and `jump_input`. These were own-action features already encoded in the action index, creating redundancy that inflated the observation without adding information.

### reachability_features (8 dims)

Graph-derived path guidance, providing A\*-computed information the agent cannot infer from local perception:

| Index | Feature | Description |
|---|---|---|
| 0 | `path_dist_switch_norm` | Normalized A\* path distance to exit switch |
| 1 | `path_dist_exit_norm` | Normalized A\* path distance to exit door |
| 2 | `switch_activated` | Binary flag (0 or 1) |
| 3 | `next_hop_dir_x` | X component of A\* next-hop direction (unit vector) |
| 4 | `next_hop_dir_y` | Y component of A\* next-hop direction (unit vector) |
| 5 | `platform_ahead` | Normalized distance to platform in next-hop direction |
| 6 | `platform_below` | Normalized distance to platform below agent |
| 7 | `path_dist_current_goal` | Phase-aware distance: dist_to_switch (phase 0) or dist_to_exit (phase 1) |

**`switch_activated` is critical for the Markov property.** Without it, the same physical position before and after switch collection would appear identical to the policy, but have different optimal actions — violating the Markov assumption.

**`next_hop_dir` vs. Euclidean direction:** The A\*-derived direction accounts for geometry and obstacles; the Euclidean direction to the goal can point through walls or across gaps and was removed in v2.1 as "misleading."

**Removed features (v2.1, Feb 2026):** `mine_clearance_dir` (duplicate of the escape gradient already in `mine_sdf_features`), `exit_reachable` (near-constant once the graph is built; derivable from whether `path_dist_exit < inf`).

### spatial_context (84 dims)

Graph-free local geometry representation. Provides the agent with immediate environmental awareness without the memory cost of a full reachability graph:

| Sub-feature | Dims | Description |
|---|---|---|
| Local tile grid | 64 | 8×8 grid of tile categories (0–4) centered on the agent, normalized |
| Mine overlay | 20 | 5 nearest mines × 4 features: `rel_pos(2)`, `state(1)`, `velocity_dot_direction(1)` |

**Memory justification:** A full graph observation requires ~21KB per environment step. With 32 parallel environments and UTD=4 (128 observations per step), this becomes ~2.7MB/step or ~270GB over 100M steps. The spatial_context at 84 dims is a 99.7% reduction.

**Velocity-dot-direction in mine overlay:** `dot(ninja_velocity, unit_vec_to_mine)` indicates whether the agent is currently moving toward or away from each mine. This provides immediate collision-risk information without requiring the policy to compute it internally.

### mine_sdf_features (3 dims)

Global mine danger signal from a precomputed Signed Distance Field:

| Index | Feature | Description |
|---|---|---|
| 0 | `sdf_value` | Distance to nearest mine (negative = inside danger zone) |
| 1 | `grad_x` | X component of SDF gradient (escape direction) |
| 2 | `grad_y` | Y component of SDF gradient (escape direction) |

Computed from a precomputed 88×50 SDF grid at 12px resolution, covering **all mines globally**. This is distinct from the local 5-mine overlay in `spatial_context`, which only covers the nearest mines by Euclidean distance. The SDF captures mines that may be behind geometry and would otherwise be outside the local window.

### Level Context Encoder (separate branch)

The level context encoder provides global level structure awareness at negligible runtime cost (~5–10 µs per environment per step).

**Two-stage separation:**

1. **`LevelContextCNN`** — computed once per level load, cached:
   ```
   Input: [42, 23] tile grid (tile IDs / 37)
   Conv(1→32, k=3) → ReLU → Conv(32→64, k=3, s=2) → ReLU → Conv(64→64, k=3, s=2) → ReLU
   → AdaptiveAvgPool(4×3) → Flatten → [768]
   ```

2. **`LevelContextMLP`** — runs every timestep:
   ```
   Input: [grid_cache(768), agent_pos_norm(2)] → [770]
   → Linear(770→128) → LayerNorm → ReLU → Linear(128→64) → LayerNorm → ReLU → [64]
   ```

**Position conditioning justification:** The same 768-dim grid embedding is reused across all timesteps for a given level, but the position conditioning `agent_pos_norm` ensures the output reflects *where in the level* the agent currently is. A ninja near the top of the level should perceive different structural affordances than one at the bottom, even on the same level.

---

## Network Architecture

`SACFeatureExtractor` in `sac_feature_extractor.py`:

```
game_state (24)      → state_mlp   (2-layer) → 128
reachability (8)     → reach_mlp   (3-layer) → 64
spatial_context (84) → spatial_mlp (3-layer) → 128
mine_sdf (3)         → mine_mlp    (2-layer) → 16
level_context (770)  → level_ctx   (2-layer) → 64
                                        ↓
                     concat [128+64+128+16+64] = 400
                     → Linear(400→256) + LayerNorm + ReLU
                                        ↓
                                    [256-dim]
```

**LayerNorm throughout (not Dropout):** Ball et al. 2023 (RLPD) shows LayerNorm is critical for stability when UTD > 1. Dropout introduces stochastic noise into Q-value estimates that compounds across UTD=4 updates, destabilizing training. SAC's entropy bonus already handles exploration; Dropout's implicit ensemble effect is unnecessary and harmful here.

---

## RFCL Curriculum

RFCL (Tao et al. ICLR 2024) solves the exploration problem by providing a sequence of achievable starting states that bridge from near-goal to the level's actual spawn.

### Class Hierarchy

```
TwoPhaseRFCL
  └─ LevelCurriculum × 2  (phase 0: spawn→switch, phase 1: switch→exit)
       ├─ ReverseCurriculum × N_demos
       └─ ForwardCurriculum × 1
```

### ReverseCurriculum

Manages a single demo's sequence of snapshots (keyframes extracted from an expert replay). Index 0 = demo start (hardest, farthest from goal); index −1 = near goal (easiest).

**Frontier mechanics:**
- Starts at the last snapshot (closest to goal)
- **Advance:** requires ≥50 episodes of history AND ≥5 results at the frontier; success rate ≥ 40% triggers advancement by `max(1, int(n_snapshots * 0.05))` steps toward index 0
- **Retreat:** if success rate at the frontier stays below 5% for 20 consecutive results, frontier retreats one step toward the goal to re-establish competence
- **Sampling:** geometric distribution weighted toward the frontier: `P(i) ∝ 0.5^i` (where i=0 is frontier offset). Base 0.5 chosen over the paper's 0.7 for a broader distribution that reduces catastrophic forgetting of easy states.

**Noise injection:** Every sampled snapshot receives:
```
position += N(0, 2px),  clamped to level bounds
velocity += N(0, 0.15 px/frame)
```
This prevents the policy from overfitting to exact demo states, which would make the learned behavior brittle in levels that differ even slightly from the demos.

### ForwardCurriculum

Activated after all `ReverseCurriculum` instances complete (frontier at index 0 with ≥30% success). Mixes demo snapshot resets with true spawn resets:

```
spawn_mix_ratio starts at 0.2
+0.1 per advance step when spawn success rate ≥ 30%
Completes when spawn_mix_ratio ≥ 0.9 with ≥30% spawn success
```

### TwoPhaseRFCL

Manages two `LevelCurriculum` instances for a single level's two task phases:

| Condition | Phase 0 weight | Phase 1 weight |
|---|---|---|
| Both incomplete | 60% | 40% |
| Phase 0 done only | 20% | 80% |
| Phase 1 done only | 80% | 20% |
| Both complete | 50% | 50% |

**Phase 0 priority justification:** Phase 0 (reaching the switch) is a strict prerequisite for Phase 1. Spending 60% of early episodes on Phase 0 establishes the skill prerequisite before heavily investing in Phase 1.

**Stall recovery:** When the level scheduler detects a stall, `apply_stall_recovery(threshold=0.7)` raises the advance threshold across all reverse curricula — demanding stronger mastery before expanding further, rather than rushing to harder starting states prematurely.

### Curriculum Flow Diagram

```
Episode starts
      │
      ▼
TwoPhaseRFCL.sample_reset()
      │
      ├─ 60% → Phase 0 LevelCurriculum
      │              │
      │              ├─ ReverseCurriculum 1 (demo 1 snapshots)
      │              ├─ ReverseCurriculum 2 (demo 2 snapshots)
      │              └─ ForwardCurriculum (spawn starts)
      │
      └─ 40% → Phase 1 LevelCurriculum
                     │
                     ├─ ReverseCurriculum 1 (post-switch snapshots)
                     └─ ForwardCurriculum
                          │
                          ▼
               Snapshot + Gaussian noise
                          │
                          ▼
               SACEnvironment.reset_to_state()
               (restores full ninja physics state)
```

---

## Design Philosophy

### Why SAC over PPO for this task?

SAC's off-policy nature means every transition in the replay buffer can be reused for UTD=4 gradient updates. PPO discards data after each on-policy rollout. In a curriculum setting where each episode starts near the goal, the agent needs to extract maximum learning signal from each transition — off-policy SAC is far more sample-efficient in this regime.

### Why RFCL over standard curriculum?

The level spawn is often hundreds of frames from the nearest reward. Standard exploration (ε-greedy, entropy bonus) cannot bridge this distance in a reasonable number of steps. RFCL sidesteps the problem entirely by providing starting states that guarantee reward is achievable within the episode's time limit.

### Why not pure imitation learning?

Expert demos exist but the expert's movement is physics-optimal in a way that is difficult to imitate directly. RLPD demo mixing provides the demos as a bootstrap (15% of batches) without forcing the policy to exactly reproduce them. This allows the agent to discover its own movement strategies while still benefiting from the demos' coverage of important transitions.

### Reward scale considerations

The full completion trajectory yields approximately:

```
R_SWITCH(1.0) + HWM_switch(≤3.0) + R_EXIT(5.0) + HWM_exit(≤3.0) ≈ 12.0 total
```

Dense components at their maximum per-episode contribution are approximately 1.0–2.0, always smaller than the sparse terminal rewards. This ordering ensures the agent remains primarily task-motivated rather than shaped-reward-motivated.
