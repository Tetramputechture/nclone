# Temporal Abstraction for N++ RL: From Per-Frame Actions to Macro-Actions

**Date:** February 6, 2026  
**Context:** N++ platformer RL with PPO, PBRS reward shaping, precomputed value functions  
**Core Problem:** Per-frame actions (6 actions × 60fps) create 20,000-frame horizons when human play involves ~200-400 actual decisions per level

---

## 1. Diagnosis: The Real Problem

### 1.1 What Semi-Lagrangian VI Fixed (and Didn't Fix)

Semi-Lagrangian value iteration solved the **spatial interpolation** problem — transitions that land between grid points are now properly captured via multilinear interpolation. But it didn't solve the **temporal** problem:

```
Human play:  "hold RIGHT for 40 frames" → moves 100+ pixels → crosses 8+ grid cells
Per-frame VI: "press RIGHT once"        → moves 1.5 pixels  → stays in same grid cell
```

The issue isn't just value function gradients. It's that the **fundamental decision-making timescale is wrong**. N++ has a natural decision frequency of roughly every 10-60 frames, not every frame. Between decision points, a human simply holds the same button combination.

### 1.2 The Dual Problem

This temporal mismatch hurts **both** sides of your system:

**PPO Training (online):**
- Effective horizon: ~20,000 frames ÷ γ=0.99 → value signal disappears around frame ~460
- To build momentum (run right for 60 frames), the agent must independently output `action=RIGHT` 60 times in a row — each time re-evaluating the policy network
- Exploration is catastrophically slow: random per-frame actions produce jittering in place
- Credit assignment across 60 identical decisions is wasted computation

**Value Function for PBRS (offline):**  
- Even with SL-VI, per-frame transitions of ~1.5px on a 12px grid produce tiny interpolation weight shifts
- Each VI iteration propagates value by ~1.5px, requiring ~700+ iterations minimum
- The value function has no concept of "commit to this action for N frames"

### 1.3 The Key Insight: N++ Has Natural Macro-Actions

Analyzing how N++ is actually played reveals a small vocabulary of temporally-extended actions:

| Macro-Action | Typical Duration | Effect |
|---|---|---|
| Run right/left | 20-100 frames | Build horizontal momentum (up to 3.33 px/frame) |
| Hold jump | 1-45 frames | Variable jump height (gravity changes at frame 45) |
| Wall slide | 10-40 frames | Controlled vertical descent on wall |
| Wait (no input) | 5-30 frames | Momentum-based slide or fall |
| Wall jump + direction | ~10 frames | Launch from wall with specific trajectory |
| Ground slide | 5-20 frames | Momentum carry with friction |

A typical level completion involves **200-400 such decisions**, not 20,000 per-frame actions. This is a 50-100× reduction in effective horizon.

---

## 2. Solution Landscape

### 2.1 Approach A: FiGAR-PPO (Recommended — Start Here)

**Paper:** Sharma et al., "Learning to Repeat: Fine Grained Action Repetition for Deep RL" (ICLR 2017)

**Core idea:** The policy outputs **two things** per decision step — an action AND a duration:

```
π(s) → (action, duration)
     → (a ∈ {0,...,5}, k ∈ {1,2,...,K_max})
```

**Architecture for N++:**

```python
class FiGARPolicyNetwork(nn.Module):
    def __init__(self, feature_dim=384, n_actions=6, n_durations=K):
        super().__init__()
        # Shared backbone (your existing multimodal extractor)
        self.backbone = ConfigurableMultimodalExtractor(...)  # → 384 dims
        
        # Action head: which of the 6 primitive actions
        self.action_head = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)  # logits for Categorical(6)
        )
        
        # Duration head: how many frames to hold it
        self.duration_head = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, n_durations)  # logits for Categorical(K)
        )
        
        # Value head (single value estimate)
        self.value_head = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, obs):
        features = self.backbone(obs)
        action_logits = self.action_head(features)
        duration_logits = self.duration_head(features)
        value = self.value_head(features)
        return action_logits, duration_logits, value
```

**Duration discretization options:**

```python
# Option 1: Linear spacing (simple, fine-grained for short durations)
durations = list(range(1, 101))  # K=100: [1, 2, 3, ..., 100]

# Option 2: Log-spaced (covers wide range with fewer bins — RECOMMENDED)
durations = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 45, 64, 90, 128]  # K=14

# Option 3: Hybrid (fine near 1, coarse for long holds)
durations = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 45, 60, 80, 100]  # K=15
```

**Recommendation:** Option 2 or 3. Log-spaced bins with K=14-15 are sufficient. The FiGAR paper used |W|=30 for Atari and noted the agent typically preferred shorter durations. For N++, you want finer resolution at short durations (1-6 frames matter for jump timing) and coarser resolution for long holds.

### 2.2 SMDP Discounting (Critical Implementation Detail)

When the agent selects (action=a, duration=k), the environment runs k frames. The SMDP Bellman equation is:

```
V(s) = max_{a,k} [ R(s, a, k) + γ^k · V(s') ]

where R(s, a, k) = Σ_{t=0}^{k-1} γ^t · r_t   (accumulated discounted reward over k frames)
      s' = state after executing action a for k frames
```

**For PPO's GAE computation**, each "step" in the rollout buffer corresponds to one (action, duration) decision, NOT one frame:

```python
# During rollout collection:
def collect_smdp_step(env, policy, obs):
    action, duration, value, log_prob = policy.act(obs)
    
    # Execute action for `duration` frames
    total_reward = 0.0
    gamma_power = 1.0
    done = False
    for t in range(duration):
        obs_next, reward, done, info = env.step(action)
        total_reward += gamma_power * reward
        gamma_power *= gamma
        if done:
            break
    
    actual_duration = t + 1  # may be shorter if episode ended
    
    # Store in rollout buffer:
    # - obs, action, duration, log_prob (for policy loss)
    # - total_reward (discounted sum over k frames)
    # - gamma^k (the effective discount to the next decision point)
    # - value (V(s) at the decision point)
    # - obs_next, done
    return obs_next, total_reward, gamma_power, done, actual_duration

# GAE computation with variable-length steps:
def compute_gae_smdp(rewards, values, dones, gamma_powers, gae_lambda=0.95):
    """
    rewards[i]: accumulated discounted reward for SMDP step i
    gamma_powers[i]: γ^{k_i} where k_i is the duration of step i
    """
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0  # or bootstrap
        else:
            next_value = values[t + 1]
        
        # SMDP TD error: different from standard because discount is γ^k, not γ
        delta = rewards[t] + gamma_powers[t] * (1 - dones[t]) * next_value - values[t]
        
        # GAE with variable discount
        gae = delta + gae_lambda * gamma_powers[t] * (1 - dones[t]) * gae
        advantages[t] = gae
    
    returns = advantages + values
    return advantages, returns
```

**Why γ^k matters:** If duration k=60 and γ=0.99, then γ^60 ≈ 0.547. This is a much stronger discount than a single step (γ=0.99), which correctly penalizes very long commitments and encourages the agent to re-evaluate periodically. The agent will naturally learn to use shorter durations near obstacles/mines and longer durations in open runs.

### 2.3 Approach B: Options Framework (More Powerful, More Complex)

**Paper:** Sutton, Precup, Singh, "Between MDPs and Semi-MDPs" (1999)

Instead of simple action repetition, define **options** — closed-loop policies with termination conditions:

```
Option = (I, π, β)
  I: initiation set (states where this option can start)
  π: intra-option policy (can vary action during execution)  
  β: termination condition β(s) ∈ [0,1]
```

Example options for N++:
- "Run right until near wall" (policy: RIGHT, terminates when wall_distance < threshold)
- "Jump to platform" (policy: JUMP→hold JUMP→release, terminates on floor_contact)
- "Wall jump sequence" (policy: context-dependent wall jump, terminates on launch)

**Advantage over FiGAR:** Options are closed-loop — they can react to state changes mid-execution. If the agent commits to "run right for 60 frames" via FiGAR but a mine activates at frame 20, it can't react. An option can.

**Disadvantage:** Much harder to implement. Requires defining or learning the option set, training intra-option policies, and the option-critic architecture is significantly more complex than adding a duration head.

**Recommendation:** Start with FiGAR (Approach A). If the agent exhibits problems with committed actions (dying because it couldn't interrupt a long hold), then add an **early termination mechanism**: allow the policy to re-evaluate at certain "interrupt points" during a macro-action (e.g., on state change triggers like floor contact, wall contact, mine activation).

### 2.4 Approach C: Hybrid — FiGAR with Interrupt

A practical middle ground between pure FiGAR and full Options:

```python
def execute_macro_action(env, policy, action, planned_duration, obs):
    """Execute action with early termination on critical state changes."""
    total_reward = 0.0
    gamma_power = 1.0
    
    initial_floor = obs['game_state'][..., FLOOR_CONTACT_IDX]
    initial_wall = obs['game_state'][..., WALL_CONTACT_IDX]
    
    for t in range(planned_duration):
        obs_next, reward, done, info = env.step(action)
        total_reward += gamma_power * reward
        gamma_power *= gamma
        
        if done:
            break
            
        # Interrupt conditions: re-evaluate policy early
        current_floor = obs_next['game_state'][..., FLOOR_CONTACT_IDX]
        current_wall = obs_next['game_state'][..., WALL_CONTACT_IDX]
        
        state_change = (
            (current_floor != initial_floor) or  # Landed or left ground
            (current_wall != initial_wall) or     # Hit or left wall
            info.get('mine_nearby', False)         # Mine threat
        )
        
        if state_change and t > 0:
            break  # Return to policy for new decision
    
    return obs_next, total_reward, gamma_power, done, t + 1
```

This gives you ~80% of the Options framework benefit with ~10% of the implementation complexity. The agent learns durations that represent "hold until something interesting happens," and the interrupt conditions handle the rest.

---

## 3. Impact on Value Function / PBRS

### 3.1 The PBRS Problem is Largely Solved by Temporal Abstraction

With FiGAR macro-actions, each SMDP transition moves the state by a large amount:

| Macro-Action | Duration | Displacement | Grid Cells (12px) |
|---|---|---|---|
| Run right (slow) | 30 frames | ~45 px | 3.75 cells |
| Run right (max speed) | 30 frames | ~100 px | 8.3 cells |
| Full jump | 45 frames | ~150px arc | 12+ cells |
| Wall jump + run | 20 frames | ~40 px lateral | 3.3 cells |

Each transition now reliably crosses multiple grid cells, producing substantial PBRS signals F(s,s') = γ^k·Φ(s') - Φ(s).

### 3.2 Revised Value Function Strategy

**Option 1: Skip precomputed V\* entirely (simplest)**

With macro-actions reducing the effective horizon to ~200-400 decisions, PPO's own learned value function may be sufficient. The exploration problem is dramatically reduced — random macro-actions with durations 10-60 frames will naturally traverse the level. Consider training without PBRS first and only adding it if learning stalls.

**Option 2: SMDP value iteration (if PBRS still needed)**

If you still want precomputed V\* for PBRS, the VI now operates over macro-action transitions:

```
For each grid state s:
    For each action a ∈ {0,...,5}:
        For each duration k ∈ {1, 2, 4, 8, 16, 32, 64}:
            s' = simulate(s, a, k)        # Run physics for k frames
            R = accumulated_reward(s, a, k) # Sum discounted rewards
            Q(s, a, k) = R + γ^k · V_interp(s')  # SL interpolation at s'
    
    V_new(s) = max_{a,k} Q(s, a, k)
```

Each iteration now propagates value by the full macro-action displacement — 30-100+ pixels instead of 1.5 pixels. Convergence drops from ~700+ iterations to ~50-100 iterations (since each "step" covers far more ground).

The tradeoff: VI is now more expensive per iteration (7 durations × 6 actions = 42 transitions per state instead of 6), but requires far fewer iterations. Net computation is likely **faster** than per-frame SL-VI.

**Option 3: Per-frame V\* with macro-action PBRS (hybrid)**

Compute a coarse V\*(position, phase) — just 2D position + switch_activated — using macro-action transitions. Use this as PBRS potential. The velocity/jump state dimensions become less critical because the macro-action implicitly handles momentum building.

State space: 88 × 50 × 2 (phase) = 8,800 states. Trivially small. VI converges in seconds.

---

## 4. Implementation Roadmap

### Phase 1: FiGAR-PPO (1-2 weeks)

**Changes to the policy network:**
1. Add duration head (parallel to action head, sharing backbone)
2. Duration bins: `[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 45, 64, 90]` (K=13)
3. Policy loss: sum of action log_prob + duration log_prob (joint policy)

**Changes to rollout collection:**
1. Execute selected (action, duration) for k frames
2. Accumulate discounted rewards over k frames  
3. Store γ^k per step for SMDP GAE

**Changes to GAE/advantage computation:**
1. Variable-step GAE using stored γ^k values
2. Returns computation uses γ^k discounting

**Changes to observation space:**
1. Add `frames_remaining_in_macro` to game state (so the agent knows when it's mid-execution during any interrupt-based scheme)
2. Or, simpler: only observe state at decision points (policy is only queried when a macro-action completes)

**Entropy bonus:**
- Action entropy + duration entropy (weighted separately)
- Duration entropy coefficient should be higher initially to encourage exploration of different timescales
- FiGAR paper recommends increasing entropy regularization when using action repetition (0.05 vs 0.02 for vanilla)

### Phase 2: Evaluate Without PBRS (1 week)

Train FiGAR-PPO on a set of levels without any PBRS. Compare:
- Learning speed vs per-frame PPO
- Final performance
- Distribution of learned durations (do they match human play patterns?)
- Explore/exploit behavior (does the agent discover momentum-building?)

### Phase 3: Add PBRS If Needed (1-2 weeks)

If learning stalls on complex levels (long detours, many mines):
1. Compute coarse V\*(x, y, phase) via macro-action SMDP-VI (tiny state space, fast)
2. Apply PBRS: F = γ^k · Φ(s') - Φ(s) at each SMDP decision point
3. Φ = V\* interpolated at continuous (x,y) position

### Phase 4: Refinements (ongoing)

- **Interrupt mechanism:** Add state-change triggers for early macro-action termination
- **Duration curriculum:** Start with small K_max (e.g., 16), gradually increase to 90+
- **Action-duration correlations:** Some (action, duration) combinations are physics-impossible or physics-useless. Consider masking or learned priors.

---

## 5. Key Literature

| Paper | Year | Key Contribution | Relevance to N++ |
|---|---|---|---|
| **FiGAR** (Sharma et al.) | 2017 | Action + duration head, works with A3C/TRPO/DDPG | Direct template for implementation |
| **Dynamic Action Repetition** (Lakshminarayanan et al.) | 2017 | Agent chooses between discrete frame skip rates | Simpler version of FiGAR |
| **Options Framework** (Sutton, Precup, Singh) | 1999 | Closed-loop temporally extended actions (SMDP theory) | Theoretical foundation, interrupt mechanism |
| **Action Persistence** (Metelli et al.) | 2020 | Theory of optimal persistence, "price of inertia" | Justifies why variable duration helps |
| **Frame-skip Analysis** (Kalyanakrishnan et al.) | 2021 | Frame-skip doesn't affect asymptotic optimality under mild conditions | Safety guarantee for approach |
| **AP-AC** (NeurIPS 2020) | 2020 | Action-Persistent Actor-Critic for multiple control frequencies | Scalable actor-critic with persistence |

---

## 6. Expected Impact

### Quantitative Predictions

| Metric | Per-Frame PPO | FiGAR-PPO (predicted) |
|---|---|---|
| Effective horizon | ~20,000 steps | ~200-400 decisions |
| Decisions per episode | 20,000 | 200-400 |
| Action space per step | 6 | 6 × 13 = 78 (but factored) |
| Exploration coverage (random) | Jitters in place | Traverses level sections |
| Policy evaluations per episode | 20,000 | 200-400 |
| GAE computation | 20,000 steps | 200-400 steps |
| PBRS signal per step | ~0 (sub-grid) | Large (multi-cell transitions) |
| Value propagation per VI iteration | ~1.5 px | ~30-100 px |

### Qualitative Predictions

1. **Exploration dramatically improves.** Random macro-actions naturally traverse the level. A random sequence of "right 30 frames, jump 20 frames, left 10 frames" actually moves through the level, unlike random per-frame actions which produce stationary jittering.

2. **Credit assignment becomes tractable.** Instead of asking "which of these 20,000 frame-level decisions led to success?", we ask "which of these 300 macro-decisions led to success?" — a 50-100× easier problem.

3. **Learned duration distribution will be informative.** Expect to see short durations (1-4 frames) near obstacles requiring precise timing, and long durations (30-90 frames) for open runs. This distribution itself reveals the agent's understanding of level structure.

4. **PBRS may become unnecessary.** With tractable exploration and short effective horizons, PPO's own value function may learn well enough without external shaping. This would eliminate the entire precomputed V* pipeline.

---

## 7. Risks and Mitigations

**Risk: Agent commits to fatal actions (holds RIGHT into a mine)**
- Mitigation: Interrupt mechanism (Section 2.4)
- Mitigation: Start with small K_max and increase gradually

**Risk: Duration head collapses to always predicting k=1 (degenerates to per-frame)**
- Mitigation: Higher entropy bonus on duration head
- Mitigation: Reward bonus for level progression (naturally rewards longer runs)
- Mitigation: Curriculum on K_max

**Risk: SMDP GAE computation is incorrect**
- Mitigation: Careful unit testing — verify that γ^k discounting is applied consistently
- Mitigation: Compare value estimates against Monte Carlo returns on short episodes

**Risk: Joint (action, duration) space is too large**
- Mitigation: Use factored policy (separate heads, not 6×13 joint distribution). FiGAR paper confirms this works.
- Mitigation: Log-spaced duration bins reduce K without losing expressiveness

---

## 8. Summary Recommendation

**Yes, fundamentally rethink the approach.** The per-frame action formulation is the root cause of both the flat value function gradients AND the slow PPO learning. Temporal abstraction via FiGAR-style (action, duration) predictions addresses both problems simultaneously.

**Implementation priority:**
1. **FiGAR-PPO with log-spaced durations** — this is the highest-impact, lowest-risk change
2. **SMDP GAE** — necessary for correct training with variable-length steps
3. **Interrupt mechanism** — safety improvement for committed actions
4. **Coarse SMDP V\*** — only if PPO still struggles on complex levels

The precomputed value function strategy should be **deprioritized** until you've evaluated FiGAR-PPO on its own. The 50-100× reduction in effective horizon may make external reward shaping unnecessary.
