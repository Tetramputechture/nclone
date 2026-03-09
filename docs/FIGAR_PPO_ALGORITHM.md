# FiGAR-PPO: Fine Grained Action Repetition with Proximal Policy Optimization

## Overview

This document describes the FiGAR-PPO approach to training the N++ ninja agent. It combines:

- **FiGAR** (Fine Grained Action Repetition, Sharma et al. ICLR 2017) for temporal abstraction via jointly-predicted (action, duration) pairs
- **PPO** (Proximal Policy Optimization, Schulman et al. 2017) as the on-policy base algorithm, extended to support discrete multi-head outputs
- **Contact-state interruption** to give the agent reactive decision points aligned with the physics engine's state transitions
- Optional **RND** (Random Network Distillation, Burda et al. 2018) intrinsic rewards for exploration

The central challenge FiGAR addresses in N++ is temporal commitment: effective movement in a momentum-based platformer requires sustained actions (running starts, held jumps, wall-slide setup) that single-frame policies cannot express efficiently. FiGAR allows the policy to commit to an action for 1–8 frames at a time while still retaining fine-grained control via contact-state interruption.

**Source files:**
- [`npp-rl/npp_rl/agents/figar_ppo.py`](../../npp-rl/npp_rl/agents/figar_ppo.py)
- [`nclone/nclone/gym_environment/npp_environment.py`](../nclone/gym_environment/npp_environment.py)
- [`nclone/nclone/gym_environment/reward_calculation/main_reward_calculator.py`](../nclone/gym_environment/reward_calculation/main_reward_calculator.py)
- [`nclone/nclone/gym_environment/reward_calculation/hwm_shaping.py`](../nclone/gym_environment/reward_calculation/hwm_shaping.py)
- [`nclone/nclone/gym_environment/reward_calculation/reward_constants.py`](../nclone/gym_environment/reward_calculation/reward_constants.py)
- [`npp-rl/npp_rl/feature_extractors/configurable_extractor.py`](../../npp-rl/npp_rl/feature_extractors/configurable_extractor.py)

---

## Algorithm

### PPO Base

PPO updates the policy by maximizing a clipped surrogate objective:

```
L_CLIP(θ) = E_t[ min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t) ]
```

where `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` is the probability ratio and `A_t` is the advantage estimate from Generalized Advantage Estimation (GAE).

FiGAR-PPO extends this with a **multi-head output**: the policy simultaneously predicts an action distribution and a duration distribution, and both heads are trained with PPO's clipped objective.

### FiGAR: Action Repetition as Temporal Abstraction

FiGAR (Sharma et al. ICLR 2017) treats action repetition as a first-class policy output rather than an environment-level hyperparameter. The action space is:

```
(base_action, duration)  ∈  {NOOP, Left, Right, Jump, Jump+Left, Jump+Right} × {1, 2, ..., 8}
```

This forms a **Semi-Markov Decision Process (SMDP)**: the agent makes decisions at irregular time intervals (decision points), not every primitive frame. Between decision points, the selected base action is held.

**Key advantage over fixed frame-skip:** A fixed frame-skip parameter commits the same hold duration for all action types regardless of context. FiGAR allows the policy to learn that, for example, jump should be held for fewer frames on a low platform than on a high wall, without requiring any manual tuning.

### Split Training Signal (Value vs. Policy)

The most architecturally distinctive feature of this implementation is the asymmetric training signal:

- **Value loss:** computed on **all primitive frames** — every `env.step()` call within a macro-action
- **Policy loss + entropy:** computed at **decision points only** — the first frame of each macro-action

This is stored in a custom `PrimitiveSMDPRolloutBuffer` where `n_steps` is measured in primitive frames, not macro-actions. The rollout buffer effectively accumulates 4–10× more value training samples than policy training samples per wall-clock batch collection.

**Justification:** The value function `V(s)` must generalize across all states, including mid-macro-action states where the agent has no decision to make. Training `V` on every frame provides the densest possible supervision. The policy `π(a|s)` is only meaningful at decision points, so training it on non-decision frames would introduce gradient noise from states where the action is already determined.

### Contact-State Interruption

Macro-actions are cut short when the ninja's **contact regime changes**:

| Contact regime | Ninja states |
|---|---|
| GROUNDED (0) | States 0, 1, 2 (on floor) |
| AIRBORNE (1) | States 3, 4 (in air) — default |
| WALL_SLIDING (2) | State 5 (on wall) |

States 6–9 (dead, celebrating, etc.) are mapped to AIRBORNE to avoid spurious interrupts during terminal transitions.

**Why contact interrupts are necessary:** Consider a wall-jump: the agent must be in contact with the wall (WALL_SLIDING) when it issues the jump command. If a "Jump+Right for 6 frames" action places the agent against a wall partway through, the agent needs a new decision point *at the moment of contact* rather than 3 frames later. Contact-state interruption ensures decision points align with the physics engine's meaningful state boundaries.

### Loss Composition

```
L_total = L_policy              (action + duration, at decision points)
        + ent_coef      * L_action_entropy      (at decision points)
        + dur_ent_coef  * L_duration_entropy    (at decision points)
        + vf_coef       * L_value               (all primitive frames)
        + L_rnd_predictor                        (if RND attached)
```

| Parameter | Value | Rationale |
|---|---|---|
| `ent_coef` | 0.03 | Standard PPO entropy bonus for action exploration |
| `dur_ent_coef` | 0.05 | Higher than action entropy; duration exploration needed more than action exploration |
| `vf_coef` | 0.5 | Standard PPO value coefficient |
| Policy loss clamp | [−0.5, 0.5] | Hard clamp before combining to prevent large policy updates dominating |
| Advantage normalization | clip to [−5.0, 5.0] | Prevents extreme advantage estimates from corrupting early updates |

**Duration entropy coefficient justification:** The policy naturally converges to duration=1 (greedy temporal resolution) unless duration exploration is explicitly encouraged. A higher entropy coefficient for duration relative to action pushes the policy to explore longer commitment windows.

### Mixed Precision

Full FP16 mixed-precision training is supported via an `amp_helper` context manager, reducing memory usage and increasing throughput on GPU without affecting policy quality.

---

## Reward Function

FiGAR-PPO uses `NppEnvironment` directly (no SAC wrapper) with the full `RewardCalculator` defined in `main_reward_calculator.py`. The total reward per macro-action step is:

```
r_total = (r_sparse + r_HWM + r_time + r_backtrack) × GLOBAL_REWARD_SCALE
```

where `GLOBAL_REWARD_SCALE = 1.0` (identity; kept for backward compatibility with earlier reward-scaled training runs).

### Sparse Rewards

| Event | Reward | Justification |
|---|---|---|
| Exit switch activated | +1.0 | Intermediate milestone; distinguishable from background |
| Exit door reached | +5.0 | Primary task signal; 5× switch to unambiguously distinguish phases |
| Death (any cause) | −0.5 | Discourages recklessness without causing excessive risk-aversion |
| Per primitive frame | −0.001 | `×frames_executed` for multi-frame actions; 500 frames = −0.5 ≈ one death |

**Death penalty history:** Originally −1.5. Run-221 showed this caused 37% NOOP rate and 95% mine deaths (vs. 61% with −0.5). The oversized death penalty made any exploration near mines irrational, dominating the HWM reward signal 1500:1.

### Backtracking Penalty

```
r_backtrack = BACKTRACK_PENALTY × frames_executed  (when A* dist to goal increases)
BACKTRACK_PENALTY = -0.0002
```

Applied when the agent's A\* distance to the current goal is greater than the distance at the previous step. Skipped on the first step of an episode (no previous distance).

**Relationship to HWM:** HWM shaping gives zero reward for backtracking (non-negative property). The backtrack penalty makes backtracking actively negative, providing a directional nudge without harshly penalizing it. Run-221-2 observed 80–93% backtracking across all stages even with HWM active — the penalty was added as an additional signal.

**Why small (0.0002 vs. 0.001 step penalty):** Backtracking is sometimes necessary (momentum-building, wall-jump approach). A backtrack penalty equal to the step penalty would penalize necessary setup movements equivalently to wasted time. At 20% of the step penalty, it is a "nudge, not a hard constraint."

### High-Water-Mark (HWM) Shaping

HWM replaces classical PBRS for the same core reason: N++ requires momentum-building maneuvers that move *away* from the goal. Classical PBRS (`F = γΦ(s') - Φ(s)`) penalizes this necessary backtracking. HWM's non-negative property eliminates the anti-momentum bias.

```
hwm_progress = max(0, watermark_prev - dist_current)
r_HWM = HWM_ALPHA * hwm_progress / initial_dist
HWM_ALPHA = 3.0
```

**Properties:**
- **Non-negative:** backtracking and staying still both yield zero reward
- **Bounded cumulative:** total HWM ≤ `HWM_ALPHA = 3.0` per phase (normalized by spawn distance)
- **Non-exploitable:** cycling distance yields reward only on the first crossing of each new minimum
- **History-dependent:** the watermark is episode state, not part of the observation (intentional — the Markov property holds because the watermark can be reconstructed from the trajectory, not because it is given to the agent)

**Two-phase structure:** The level has two sequential objectives:
1. Phase 0 (pre-switch): watermark tracks A\* path distance to the exit switch
2. Phase 1 (post-switch): watermark resets; tracks A\* path distance to the exit door

On phase transition, shaping reward = 0 and `watermark_exit` is initialized to the current A\* distance to the exit. This prevents a spurious reward spike from the watermark resetting to a very large value.

**HWM_ALPHA tuning history:** Originally 1.0. After death penalty reduction to −0.5, HWM_ALPHA was raised to 3.0 to maintain signal strength relative to the death penalty. At ALPHA=3.0, the death/HWM ratio at stage-0 is approximately 28:1 (was 83:1), making the learning signal substantially more tractable.

**Reference:** HWM shaping was validated on momentum-based platformers by the Sonic OpenAI Retro Contest (2018).

### Reward Collection in the SMDP Buffer

Primitive rewards (one per frame executed) are summed across each macro-action:

```python
step_rewards = sum(info.get("primitive_rewards") or [r])
```

This means a macro-action held for `k` frames accumulates `k × r_frame` in step penalties, matching the time-pressure semantics of the reward function.

#### Optional RND Intrinsic Rewards

When RND is attached, per-episode intrinsic rewards from `info["intrinsic_reward"]` are stored in a **separate buffer stream** and processed with a separate GAE pass and value head. This dual-head architecture prevents intrinsic and extrinsic value estimates from interfering with each other.

---

## Observation Features

FiGAR-PPO uses `NppEnvironment` with a **multimodal Dict observation space**, processed by `ConfigurableMultimodalExtractor`. The multimodal design reflects the PPO era's goal of ablating which input modalities are necessary — each modality can be independently enabled or disabled for architecture comparison experiments.

### game_state (24 dims)

The complete ninja physics state, normalized to [−1, 1]. Identical to the SAC pipeline (version 2.1, Feb 2026):

| Feature group | Dims | Description |
|---|---|---|
| Velocity | 2 | `xspeed`, `yspeed` (normalized by `MAX_HOR_SPEED`) |
| Movement state (one-hot) | 4 | Current locomotion mode |
| Airborne flag | 1 | Binary; critical for jump timing |
| Action buffers | 3 | Jump buffer, floor buffer, wall buffer (coyote time windows) |
| Jump duration remaining | 1 | Remaining frames of variable-height jump hold |
| Contact counts | 3 | Floor count, wall count, overall floor contact |
| Surface normals | 3 | Floor normal (x, y), wall normal |
| Walled flag | 1 | Binary |
| Applied physics | 3 | Per-frame gravity, drag, friction |
| Position | 2 | `xpos / LEVEL_WIDTH`, `ypos / LEVEL_HEIGHT` |
| Time remaining | 1 | Normalized frames until episode truncation |

**Why action buffers are included:** N++ has coyote time (jump within N frames after leaving a platform) and wall-jump windows. These buffers are part of the game's hidden state that determines whether a jump action will succeed — without them, the observation is non-Markovian with respect to jump feasibility.

### reachability_features (7 dims, v2.1)

Graph-derived path guidance computed by the `ReachabilityMixin`. Provides A\*-computed information that local perception cannot provide:

| Index | Feature | Description |
|---|---|---|
| 0 | `path_dist_switch_norm` | Normalized A\* path distance to exit switch |
| 1 | `path_dist_exit_norm` | Normalized A\* path distance to exit door |
| 2 | `switch_activated` | Binary phase flag (critical for Markov property) |
| 3 | `next_hop_dir_x` | X component of A\* next-hop direction |
| 4 | `next_hop_dir_y` | Y component of A\* next-hop direction |
| 5 | `detour_detection` | Binary flag: A\* path routes around an obstacle |
| 6–7 | `platform_ahead`, `platform_below` | Raycast distances for blind-jump safety |

**Distinction from SAC's 8-dim reachability:** FiGAR-PPO's reachability is 7 dims (v2.1 before the final SAC-specific reduction). SAC's 8th dim (`path_dist_current_goal`) was added as a phase-aware convenience feature. The `detour_detection` flag present in PPO was removed in SAC in favor of the 8th phase-aware distance.

**Feature reduction history (v2.1, Feb 2026):** The reachability feature vector was significantly reduced from 22 dims. Removed:
- Global stats (mean/std of path distances) — not actionable per-step
- Euclidean directions to switch/exit — "misleading" (point through walls and gaps)
- Redundant inverse features (1 - dist_norm duplicates dist_norm)
- 6 of 8 directional connectivity features — too sparse to be informative
- Waypoints — better handled by the A\* next-hop direction
- Path difficulty, path curvature — LSTM-specific features not useful for feedforward policies
- Exit lookahead — redundant with phase-aware distance

**Why A\* direction rather than Euclidean:** Euclidean direction to the goal points through solid geometry and across wide gaps, leading the agent toward walls and off ledges. The A\*-derived next-hop direction accounts for the actual traversable path.

### spatial_context (96 dims for PPO, 84 dims for SAC)

Graph-free local geometry. The PPO version uses 8 nearest mines (96 dims) vs. SAC's 5 nearest mines (84 dims):

| Sub-feature | Dims | Description |
|---|---|---|
| Local tile grid | 64 | 8×8 grid of simplified tile categories (0–4), normalized |
| Mine overlay | 32 | 8 nearest mines × 4 features: `rel_pos(2)`, `state(1)`, `velocity_dot_direction(1)` |

The tile grid uses simplified categories (empty, solid, one-way platform, jump pad, etc.) rather than raw tile IDs, reducing the feature space while preserving the structurally relevant distinctions.

**Velocity-dot-direction in mine overlay:** `dot(ninja_velocity, unit_vec_to_mine)` gives the signed rate of approach to each mine. Positive = moving toward mine; negative = moving away. This is a one-step lookahead for collision risk that the policy can immediately act on.

**"99.7% memory reduction vs. full graph observations":** A full graph observation requires ~21KB per environment step (nodes + edges + masks). Spatial context at 96 dims is ~0.4KB, a 99.7% reduction. This matters significantly for PPO which keeps the full rollout buffer in GPU VRAM.

### mine_sdf_features (3 dims)

Global mine danger signal from a precomputed Signed Distance Field:

| Index | Feature | Description |
|---|---|---|
| 0 | `sdf_value` | Distance to nearest mine (negative = inside danger zone) |
| 1 | `grad_x` | X component of SDF gradient (escape direction) |
| 2 | `grad_y` | Y component of SDF gradient (escape direction) |

From a precomputed 88×50 SDF grid at 12px resolution covering **all mines globally**. Distinct from the local mine overlay in `spatial_context`:
- `spatial_context` mine overlay: 8 nearest mines by Euclidean distance (may miss mines around corners)
- `mine_sdf_features`: global minimum distance across all mines (complete coverage)

The SDF gradient provides an explicit escape direction usable by the SAC dense reward's hazard avoidance component and by the PPO policy directly.

### player_frame (84×84, optional)

A player-centered crop of the rendered game frame, grayscale, processed by a DQN-style CNN:

```
Conv2d(C_in, 32, 8, stride=4) → BN → ReLU → Dropout2d
Conv2d(32, 64, 4, stride=2)   → BN → ReLU → Dropout2d
Conv2d(64, 64, 3, stride=1)   → BN → ReLU
Flatten → Linear(64×7×7, player_frame_output_dim) → ReLU → Dropout
```

Supports **temporal frame stacking**: the stack dimension is folded into the input channel dimension following Mnih et al. 2015 (DQN). Frame stacking provides implicit velocity information without requiring the observation to include raw velocity features.

**When included:** Local visual context is particularly useful for levels where the tile geometry around the agent cannot be fully captured by the 8×8 local tile grid (e.g., complex diagonal surfaces, nearby entity positions not covered by the spatial context window).

**When excluded (most runs):** The CNN adds substantial computation per rollout step and requires frame rendering. Most training runs disable this modality in favor of the vector-based spatial context, which provides equivalent geometric information at a fraction of the cost.

### global_view (176×100, optional)

Full-level grayscale downsampled via `INTER_AREA` interpolation, processed by a 3-conv CNN with `AdaptiveAvgPool(4×4)`:

```
3 × (Conv2d, stride=2) → BN → ReLU → Dropout2d
AdaptiveAvgPool2d(4×4) → Flatten → Linear → ReLU → Dropout
```

**Purpose:** Provides full-layout spatial awareness — the agent can "see" the entire level including distant platforms, the switch location, and the exit door regardless of which part of the level it is currently in.

**Why dropped in most runs:** The global view requires rendering the full level every frame (expensive) and the resulting 176×100 grayscale image contains mostly redundant information at fine timescales. The reachability features (A\* distances and next-hop direction) provide functionally equivalent navigation guidance more efficiently.

### Graph Features (optional, deprecated)

A reachability graph can be encoded via GCN or GAT:

```
graph_node_feats: (N_MAX_NODES, 7)   — per-node platform/edge type features
graph_edge_index: (2, E_MAX_EDGES)   — adjacency in COO format
graph_node_mask:  (N_MAX_NODES,)     — valid node indicator
graph_edge_mask:  (E_MAX_EDGES,)     — valid edge indicator
```

Node features (7 dims per node): position (2), node type one-hot (3), platform length (1), elevation (1).

**Deprecation status:** HGT (Heterogeneous Graph Transformer) variants are fully deprecated — the observation space no longer includes node types, edge types, or heterogeneous edge features that HGT requires. GCN/GAT remain technically operational but are not used in current training configurations. The `spatial_context` graph-free alternative provides equivalent local geometry information with 99.7% less memory.

---

## Network Architecture

`ConfigurableMultimodalExtractor` in `configurable_extractor.py`. Designed as an **ablation framework** for architecture comparisons.

### Per-Modality Encoders

| Modality | Input | Encoder | Output |
|---|---|---|---|
| `game_state` | 24-dim | `StateMLP` (2-layer MLP) | 64-dim |
| `reachability_features` | 7-dim | 3-layer MLP | 96-dim |
| `spatial_context` | 96-dim | 3-layer MLP | 96-dim |
| `mine_sdf_features` | 3-dim | 2-layer MLP | 16-dim |
| `player_frame` | 84×84×(1 or stack) | `PlayerFrameCNN` (DQN-style 3-conv) | configurable |
| `global_view` | 176×100×1 | `GlobalViewCNN` (3-conv + AdaptiveAvgPool) | configurable |
| `graph` | (N_nodes, 7) | GCN or GAT | 256-dim |

### Fusion Mechanisms

Three options are supported:

**CONCAT:**
```
[modality_1 ‖ modality_2 ‖ ... ‖ modality_n] → 2-layer MLP → output_dim
```
Simple concatenation. Efficient; effective when modalities are roughly equal in importance.

**SINGLE_HEAD_ATTENTION:**
```
flat_concat → self-attention → output_dim
```
Single-head attention over the concatenated vector. Allows the policy to weight different feature regions differently.

**MULTI_HEAD_ATTENTION (`MultiHeadFusion`):**
```
Each modality → Linear → 256-dim
              + learnable modality embedding (position-encoding style)
              → Transformer encoder block (MHA + FFN + LayerNorm residuals)
              → mean-pool across modalities
              → output projection
```
The learnable modality embedding plays the same role as positional encoding in text transformers — it allows the attention mechanism to distinguish which modality a given 256-dim token came from. Mean pooling across modalities after the transformer block produces a fixed-size output regardless of which modalities are active.

### Dropout vs. LayerNorm

FiGAR-PPO uses **Dropout** (not LayerNorm). This is the opposite of the SAC pipeline's choice:

- **PPO with Dropout:** PPO's advantage estimates rely on the value function's uncertainty. Dropout provides an implicit ensemble effect that regularizes the value function and prevents it from fitting individual trajectories too precisely. The on-policy rollout ensures fresh data each update, so value overfitting is a greater risk than in off-policy SAC.
- **SAC with LayerNorm:** Ball et al. 2023 (RLPD) shows LayerNorm is critical for stability at high UTD (≥2). Dropout noise in Q-function estimates compounds across multiple gradient steps on the same data, destabilizing training. LayerNorm provides normalization without stochastic perturbation.

---

## Design Philosophy

### Why temporal abstraction?

N++ physics are smooth but momentum-dependent: accelerating to running speed requires 5–10 frames of sustained directional input. Expressing this with single-frame actions requires the policy to output the same action repeatedly, which a greedy policy may not do consistently. FiGAR lets the policy commit to "run right for 6 frames" as a single structured decision, dramatically simplifying the credit assignment problem.

### Why contact-state interruption rather than fixed frame-skip?

Fixed frame-skip (e.g., skip=4) uniformly quantizes time regardless of physics significance. Contact-state interruption creates decision points precisely at the moments the physics engine changes the agent's movement possibilities — landing on a floor (new jump available), leaving a wall (wall-jump window opening), becoming airborne (mid-air control changes). This aligns decision points with the game's natural decision boundaries.

### Why HWM over classical PBRS?

N++ requires backtracking for many movement techniques:
- Wall-jump approach: move *away* from the wall to build approach speed
- Running start: back up before a long jump
- Momentum conservation: arc away from a platform to land on another

Classical PBRS `F = γΦ(s') - Φ(s)` penalizes any increase in distance to the goal. This is correct in gridworld MDPs with step-optimal solutions but actively harmful in a momentum-based platformer where the optimal policy regularly increases A\* distance temporarily. HWM's zero-reward (rather than negative-reward) backtracking removes this anti-momentum bias.

### Why a configurable modality extractor?

The PPO pipeline was the research foundation for understanding which observation types are necessary and sufficient for N++. The `ConfigurableMultimodalExtractor` was built explicitly to run ablation studies: disable visual inputs, enable/disable graph observations, compare fusion mechanisms. The conclusions from these ablations informed the SAC pipeline's choice of a compact 119-dim flat vector (no images, no graph) as sufficient for the task.

### Reward hierarchy

A successful episode yields approximately:

```
R_SWITCH(1.0) + HWM_switch(≤3.0) + R_EXIT(5.0) + HWM_exit(≤3.0) ≈ 12.0 total
```

The ordering is intentional:
- `R_EXIT(5.0) >> R_SWITCH(1.0)`: completing the level is unambiguously more important than partial progress
- `HWM_phase(≤3.0) > |R_DEATH|(0.5)`: the reward for steady progress outweighs the cost of a single death, preventing excessive risk-aversion
- `|r_backtrack|(≤0.0002/frame) << |r_step|(0.001/frame)`: backtracking is a directional nudge, not a hard constraint

### Comparison with SAC+RFCL

| Dimension | FiGAR-PPO | SAC+RFCL |
|---|---|---|
| Learning paradigm | On-policy | Off-policy (UTD=4) |
| Exploration | Entropy bonus, optional RND | SAC entropy + RFCL curriculum |
| Action space | Discrete(48) via multi-head output | Discrete(48) via index encoding |
| Temporal abstraction | FiGAR (joint (action, duration) output) | SMDP macro-actions (same structure) |
| Observation | Multimodal Dict (images optional) | Flat 119-dim vector |
| Replay | On-policy rollout buffer (discarded after update) | 1M transition replay buffer |
| Normalizer | Dropout | LayerNorm (RLPD requirement) |
| Curriculum | Standard spawn resets | RFCL backward curriculum |
| Demo usage | Not used (PPO is pure RL) | RLPD 15% demo mixing |
| Sample efficiency | Lower (on-policy) | Higher (off-policy + UTD=4) |
| Hyperparameter sensitivity | Lower (PPO is robust) | Higher (SAC temperature, UTD balance) |
