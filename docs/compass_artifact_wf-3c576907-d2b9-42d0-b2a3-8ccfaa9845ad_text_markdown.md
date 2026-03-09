# Fixing random-walk PPO in N++: from sparse HWM to dense directional reward

**Your agent random-walks because High-Water-Mark reward is functionally sparse — 80–93% of steps produce zero gradient signal, and PPO (an on-policy method) cannot learn directed behavior from near-zero signal.** The fix is a three-pronged intervention: replace HWM with potential-based reward shaping that penalizes every backtracking step, add a velocity-alignment reward that provides signal on 100% of steps, and constrain FiGAR's duration head to stop amplifying wrong-direction commitments. These changes together transform the reward landscape from one where random walking is cost-free to one where every step carries directional information. The value function divergence (0.06 → 0.27) is a downstream symptom — it resolves once the reward provides learnable structure.

---

## The HWM reward creates a degenerate learning landscape

The core pathology is straightforward. HWM only fires when the agent achieves a new minimum A* distance to goal. Once the agent backtracks even one step, it must recover *all* lost ground before seeing another positive reward. With 80–93% backtracking, the agent receives reward on roughly **7–20% of steps**. For PPO — which discards experience after each update — this is catastrophic. The value function sees mostly zero targets, learns to predict ~0 everywhere, and advantages become noise. The policy update direction is essentially random, so the agent remains a random walker.

**The critical insight**: HWM treats backtracking as free (zero penalty). This means the agent has no gradient signal pushing it away from backward movement. A proper reward function must make backtracking explicitly costly — not equally costly as progress is rewarding (which would over-penalize necessary detours in platformer physics), but costly enough that the agent can distinguish "toward goal" from "away from goal" on every single step.

The Sibling Rivalry paper (Trott et al., NeurIPS 2019) documents this exact failure mode: "PPO with naive distance-to-goal shaped reward never succeeds" in U-shaped mazes. And Yang et al. (2021) showed that PBRS with A* distance in Sokoban yielded a **4× speedup** and took solve rates from 20% to 100%. The evidence strongly favors replacing HWM.

---

## Replace HWM with asymmetric potential-based reward shaping

The primary intervention is switching from HWM to **Potential-Based Reward Shaping (PBRS)** using your existing A* distance computation. PBRS (Ng, Harada & Russell, 1999) is the only reward shaping form that mathematically guarantees optimal policy invariance, and it provides signal on every step where the agent's A* distance changes.

**Core formulation:**

```python
def compute_pbrs_reward(d_current, d_next, gamma=0.99, is_terminal=False):
    """
    Potential-based reward shaping using A* distance.
    Φ(s) = -d_A*(s, goal) / max_distance  (normalized to [-1, 0])
    """
    phi_current = -d_current / MAX_ASTAR_DISTANCE
    phi_next = 0.0 if is_terminal else -d_next / MAX_ASTAR_DISTANCE
    return gamma * phi_next - phi_current
```

Standard PBRS gives a positive reward for progress and an *equal* negative reward for backtracking. In N++, where momentum-based physics and wall-jumping mean the agent sometimes *must* temporarily increase A* distance, this equal penalty can slow learning on detour segments. The fix is an **asymmetric modification** that reduces the backtracking penalty:

```python
def compute_asymmetric_pbrs(d_current, d_next, gamma=0.99, 
                             asymmetry=0.3, is_terminal=False):
    """
    Asymmetric PBRS: full reward for progress, reduced penalty for backtracking.
    asymmetry=0.3 means backtracking penalty is 30% of what PBRS would give.
    This breaks the PBRS optimality guarantee but works better in practice
    for environments requiring non-monotonic paths.
    """
    phi_current = -d_current / MAX_ASTAR_DISTANCE
    phi_next = 0.0 if is_terminal else -d_next / MAX_ASTAR_DISTANCE
    f = gamma * phi_next - phi_current
    
    if f < 0:  # Backtracking
        f *= asymmetry  # Reduce penalty (0.3 = 30% of full penalty)
    
    return f
```

**Critical implementation details:**

- **Terminal states must have Φ = 0.** Failing this creates incentives to seek or avoid termination regardless of task reward (Grześ, 2017; Müller et al., 2025).
- **Normalize the potential** by dividing A* distance by the maximum possible distance. This keeps the shaping reward in a predictable range and prevents scale issues with the value function.
- **The γ in PBRS should match your PPO discount factor.** Using γ=0.99 in PBRS with γ=0.999 in PPO creates a small bias (Canonaco et al., 2024).
- **Asymmetry parameter λ=0.3** is a good starting point. Tune between 0.1 (very soft backtracking penalty) and 0.5 (moderate penalty). Setting λ=1.0 recovers standard PBRS; λ=0.0 recovers HWM-like behavior with no backtracking penalty.

The comparison with HWM is stark:

| Property | HWM | Asymmetric PBRS |
|---|---|---|
| Steps with nonzero reward | 7–20% | **~100%** |
| Backtracking signal | Zero | Negative (scaled by λ) |
| Optimal policy guarantee | No | Approximate (exact if λ=1) |
| Value function learnability | Poor (mostly-zero targets) | **Strong (continuous targets)** |

---

## Add velocity-alignment reward for per-step dense signal

PBRS alone may not provide strong enough directional signal during the early chaotic phase of training. A **velocity-direction alignment reward** gives the agent immediate feedback on whether its current movement direction matches the A* next-hop direction — on every single frame.

The NVIDIA Isaac Lab documentation provides the best-validated formulation. The critical insight: use **multiplicative** combination (speed × alignment), not additive — addition allows the agent to maximize one term while ignoring the other.

```python
def compute_velocity_alignment_reward(agent_velocity, astar_direction, alpha=0.2):
    """
    Reward agent for moving in the A* direction.
    
    agent_velocity: 2D velocity vector (px/frame)
    astar_direction: 2D unit vector from A* next-hop
    alpha: scaling coefficient (0.1-0.5 relative to PBRS magnitude)
    """
    speed = np.linalg.norm(agent_velocity)
    if speed < 0.01:
        return -0.005  # Small penalty for standing still
    
    vel_dir = agent_velocity / speed
    alignment = np.dot(vel_dir, astar_direction)  # in [-1, 1]
    
    # Clamp to [0, 1] to prevent reward for moving backward
    alignment_positive = max(0.0, alignment)
    
    # Multiplicative: must be moving AND aligned
    # Normalize speed by max_speed to keep reward bounded
    normalized_speed = min(speed / MAX_SPEED, 1.0)
    
    return alpha * normalized_speed * alignment_positive
```

**The complete reward function** combining all components:

```python
def compute_total_reward(state, next_state, agent_velocity, astar_direction,
                          d_current, d_next, reached_switch, reached_exit, 
                          died, gamma=0.99):
    # 1. Sparse task rewards (keep existing)
    r_task = 0.0
    if reached_switch: r_task += 1.0
    if reached_exit:   r_task += 5.0
    if died:           r_task -= 0.5
    
    # 2. Asymmetric PBRS (replaces HWM)
    r_pbrs = compute_asymmetric_pbrs(d_current, d_next, gamma=gamma,
                                      asymmetry=0.3, is_terminal=died or reached_exit)
    
    # 3. Velocity-alignment (new dense signal)
    r_align = compute_velocity_alignment_reward(agent_velocity, astar_direction, 
                                                 alpha=0.15)
    
    # 4. Time penalty (keep existing but ensure it scales properly)
    r_time = -0.001
    
    # 5. RND exploration bonus (keep existing, decaying)
    # r_rnd = beta * rnd_bonus  (computed elsewhere)
    
    return r_task + r_pbrs + r_align + r_time  # + r_rnd
```

**Hyperparameter guidance**: The velocity-alignment coefficient (α=0.15) should be **10–30% of the PBRS signal magnitude**. Too high and the agent optimizes alignment over actual progress; too low and it provides insufficient signal during early training. Start at 0.15, increase to 0.3 if the agent still random-walks, decrease to 0.05 once directed movement is established.

---

## FiGAR's duration head is amplifying the failure mode

Your mean macro-action duration of ~19 frames is a red flag. The TempoRL paper (Biedenkapp et al., ICML 2021) directly identified that FiGAR "quickly begins to struggle and converges to worse policies, **always preferring large skip-values**" as maximum skip increases. This happens because FiGAR selects action and duration independently — the duration head cannot condition on whether the chosen action is good or bad.

With 19-frame mean duration, a wrong-direction action commits the agent to **~63 pixels of backtracking** (at max speed 3.333 px/frame). To recover, the agent must then execute another macro-action in the right direction — but the HWM reward won't fire until it recovers all that lost ground. This creates a vicious cycle.

**Four interventions, in priority order:**

**1. Cap maximum duration at 6–8 frames.** This immediately reduces worst-case backtracking from ~60px to ~25px per decision. The FiGAR paper used |W|=30 for Atari (which already runs at 4-frame skip). For a precision platformer, shorter commitments are safer.

```python
# Change from:
DURATION_RANGE = range(1, 31)  # 1-30 frames
# To:
DURATION_RANGE = range(1, 9)   # 1-8 frames
```

**2. Condition the duration head on the selected action** (TempoRL-style). This is the key architectural fix — it lets the agent learn "jump+right for 6 frames" separately from "noop for 6 frames":

```python
class FiGARPolicyFixed(nn.Module):
    def __init__(self, obs_dim, n_actions=6, max_duration=8, hidden=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.action_head = nn.Linear(hidden, n_actions)
        self.action_embed = nn.Embedding(n_actions, 32)
        # Duration head CONDITIONED on action
        self.duration_head = nn.Sequential(
            nn.Linear(hidden + 32, 64), nn.ReLU(),
            nn.Linear(64, max_duration)
        )
        self.value_head = nn.Linear(hidden, 1)
    
    def forward(self, obs):
        features = self.backbone(obs)
        action_logits = self.action_head(features)
        action = Categorical(logits=action_logits).sample()
        
        action_emb = self.action_embed(action)
        duration_input = torch.cat([features, action_emb], dim=-1)
        duration_logits = self.duration_head(duration_input)
        duration = Categorical(logits=duration_logits).sample() + 1  # 1-indexed
        
        value = self.value_head(features)
        return action, duration, action_logits, duration_logits, value
```

**3. Add a KL penalty toward a short-duration prior** to prevent the duration head from drifting to long commitments:

```python
# During loss computation:
prior = torch.ones(max_duration) / max_duration  # Uniform prior
# Or geometric prior favoring short durations:
prior = torch.tensor([0.15 * (0.85 ** i) for i in range(max_duration)])
prior = prior / prior.sum()

kl_penalty = kl_coeff * F.kl_div(
    F.log_softmax(duration_logits, dim=-1),
    prior.expand_as(duration_logits),
    reduction='batchmean'
)
total_loss += kl_penalty  # kl_coeff = 0.01 to 0.1
```

**4. Implement SMDP-adapted GAE.** Standard GAE assumes uniform timesteps. With variable-duration macro-actions, the discount between decision points must be γ^d, not γ:

```python
def smdp_gae(rewards, values, durations, gamma=0.99, lam=0.9):
    """
    GAE adapted for Semi-MDP with variable-duration macro-actions.
    
    rewards: discounted cumulative reward per macro-action
    values: V(s) at each decision point (len = n_decisions + 1)
    durations: frame count per macro-action
    """
    n = len(rewards)
    advantages = np.zeros(n)
    gae = 0.0
    for i in reversed(range(n)):
        gamma_d = gamma ** durations[i]  # Key: discount by ACTUAL duration
        delta = rewards[i] + gamma_d * values[i + 1] - values[i]
        gae = delta + gamma_d * lam * gae
        advantages[i] = gae
    return advantages
```

**Also increase your rollout length.** With ~19-frame mean duration and 2048-frame rollouts, you get only ~108 decision points — far too few for stable PPO updates. Increase to **8192–16384 frames** per rollout to maintain ~400–800 decision points.

---

## The value function divergence is a symptom, not the cause

The value loss increasing from 0.06 to 0.27 is expected when 80–93% of targets are zero and the remaining 7–20% are unpredictable positive spikes from random walk progress. The critic tries to fit these noisy, non-stationary targets and fails. Several fixes help, but **the primary fix is making the reward dense** (PBRS + velocity-alignment), which gives the critic learnable structure.

Supporting interventions that help the value function converge:

- **Separate actor and critic networks.** The "What Matters in On-Policy RL" paper (Andrychowicz et al., ICLR 2021) found this critical: "Use a wide value MLP (no layers shared with the policy)." Use `[256, 256]` for the critic, `[64, 64]` for the actor, with separate optimizers.
- **Separate learning rates.** Actor: `3e-4`, Critic: `1e-3`. The critic needs to learn faster to provide stable advantage estimates.
- **Use GAE λ=0.9**, not 0.95. The "What Matters" paper found 0.9 optimal across 250,000 experiments.
- **Do not use PPO-style value clipping.** The same paper explicitly recommends against it — value clipping prevents the critic from correcting bad predictions.
- **Add reward normalization.** Use a running mean/std normalizer on rewards so the critic always sees zero-mean, unit-variance targets. This is especially important when switching to PBRS, which changes the reward scale.

```python
# If using Stable Baselines 3 or similar:
from stable_baselines3.common.vec_env import VecNormalize
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

# If custom implementation:
class RewardNormalizer:
    def __init__(self, gamma=0.99):
        self.ret_rms = RunningMeanStd()
        self.returns = 0.0
    
    def normalize(self, reward, done):
        self.returns = self.returns * gamma + reward
        self.ret_rms.update(np.array([self.returns]))
        if done: self.returns = 0.0
        return reward / (np.sqrt(self.ret_rms.var) + 1e-8)
```

---

## The agent ignores A* direction because no reward correlates with it

Having the A* direction as an observation feature creates a chicken-and-egg problem. The PGF-MAPPO paper (arXiv:2512.09410) directly validates this: vanilla MAPPO with A* direction in observations alone achieved **60% success**, while adding A*-based PBRS reward shaping achieved **98% success**. The direction feature only becomes useful once the reward signal teaches the agent to attend to it.

The velocity-alignment reward (section above) solves half of this: it directly correlates the observation feature with reward. For the architecture side, use a **FiLM-style conditioning** where the direction modulates the grid features rather than just being concatenated:

```python
class DirectionConditionedPolicy(nn.Module):
    def __init__(self, grid_channels, state_dim, dir_dim=2, hidden=256):
        super().__init__()
        # Grid encoder
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(grid_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten()
        )
        grid_flat = 64 * 4 * 4  # = 1024
        
        # Direction encoder with FiLM conditioning
        self.film_gamma = nn.Linear(dir_dim, grid_flat)
        self.film_beta = nn.Linear(dir_dim, grid_flat)
        
        # State encoder (game state + mine features)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU()
        )
        
        # Fusion → policy/value
        self.fusion = nn.Sequential(
            nn.Linear(grid_flat + 64, hidden), nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden, 6)
        self.value_head = nn.Linear(hidden, 1)
    
    def forward(self, grid_obs, direction_obs, state_obs):
        grid_feat = self.grid_encoder(grid_obs)
        
        # FiLM: direction MODULATES grid features (can't be ignored)
        gamma = self.film_gamma(direction_obs)
        beta = self.film_beta(direction_obs)
        grid_feat = gamma * grid_feat + beta
        
        state_feat = self.state_encoder(state_obs)
        fused = self.fusion(torch.cat([grid_feat, state_feat], dim=-1))
        return self.policy_head(fused), self.value_head(fused)
```

An **auxiliary direction prediction loss** can further help. Train the grid encoder to predict the A* direction from the grid alone, forcing it to build spatial representations aligned with pathfinding:

```python
# Add an auxiliary head that predicts A* direction from grid features only
self.aux_dir_head = nn.Sequential(nn.Linear(grid_flat, 64), nn.ReLU(), nn.Linear(64, 2))

# During training:
aux_pred = self.aux_dir_head(grid_feat_before_film)  # BEFORE conditioning
aux_loss = 0.5 * F.mse_loss(aux_pred, true_astar_direction)
total_loss = ppo_loss + 0.1 * aux_loss  # λ_aux = 0.1
```

---

## A behavioral cloning warmstart can break the initial random walk

If the agent never experiences directed movement, it cannot learn from it. **BC pretraining on A* demonstrations** gives the policy an initial bias toward goal-directed behavior, which PPO then refines. The PIRLNav paper (CVPR 2023) showed that BC pretraining followed by RL finetuning achieved 65% success in ObjectNav versus near-zero for RL from scratch — and **90% of the benefit came from less than half the demonstrations**.

For N++, convert A* paths to action sequences and pretrain:

```python
def generate_demonstrations(level, astar_path):
    """Convert A* waypoint path to (observation, action) pairs."""
    demos = []
    env = make_env(level)
    obs = env.reset()
    for waypoint in astar_path:
        # Map waypoint direction to closest discrete action
        dx = waypoint.x - env.agent_x
        dy = waypoint.y - env.agent_y
        action = direction_to_action(dx, dy, env.agent_on_ground)
        demos.append((obs, action))
        obs, _, _, _ = env.step(action)
    return demos

def pretrain_bc(policy, demonstrations, epochs=50, lr=3e-4):
    optimizer = Adam(policy.parameters(), lr=lr)
    for epoch in range(epochs):
        for obs_batch, action_batch in dataloader(demonstrations, batch_size=256):
            logits = policy.get_action_logits(obs_batch)
            loss = F.cross_entropy(logits, action_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Usage: pretrain, then switch to PPO
pretrain_bc(ppo_agent.policy, all_demonstrations, epochs=50)
ppo_agent.learn(total_timesteps=10_000_000)  # RL finetuning
```

Note that A* paths won't perfectly map to platformer physics (jumps, momentum), so the BC policy will be imperfect. That's fine — the goal is to give the agent a directional bias, not a perfect policy. PPO finetuning handles the rest.

---

## Curriculum fix: add hysteresis and smooth metrics

The oscillation at curriculum stage 3 is partly caused by the sparse reward problem (which the above fixes address) and partly by the curriculum mechanism itself. Two quick fixes:

**Hysteresis thresholds** prevent rapid cycling. Use separate advance and demote thresholds with a wide gap:

```python
ADVANCE_THRESHOLD = 0.70  # Must sustain 70% success to advance
DEMOTE_THRESHOLD = 0.25   # Must drop to 25% to be demoted
SMOOTHING = 0.95          # EMA smoothing factor

smoothed_success = 0.95 * smoothed_success + 0.05 * episode_success
if smoothed_success > ADVANCE_THRESHOLD and episodes_at_stage > 500:
    current_stage += 1
    episodes_at_stage = 0
elif smoothed_success < DEMOTE_THRESHOLD:
    current_stage = max(0, current_stage - 1)
    episodes_at_stage = 0
```

**Reverse curriculum** (starting near the goal) is especially powerful combined with dense rewards. When the agent starts 100px from the exit, PBRS provides strong signal immediately, and the agent learns local navigation before tackling longer paths.

---

## Conclusion: a prioritized implementation plan

The root cause is a single design decision — HWM reward — that cascades into value function divergence, FiGAR duration drift, and random walking. The interventions are ordered by expected impact:

1. **Replace HWM with asymmetric PBRS** (λ=0.3). This alone transforms 80–93% zero-reward steps into informative signal. Expected to resolve the majority of the random-walk behavior.
2. **Add velocity-alignment reward** (α=0.15, multiplicative with speed). Provides per-step directional signal that correlates the A* observation feature with reward.
3. **Cap FiGAR duration to 8 frames** and condition duration head on action. Prevents catastrophic wrong-direction commitments and fixes the SMDP credit assignment.
4. **Implement SMDP-adapted GAE** with γ^d discounting between decision points. Increase rollout to 8K–16K frames.
5. **Separate actor/critic** with different learning rates (3e-4 / 1e-3) and remove value clipping.
6. **BC warmstart on A* paths** (~50 epochs) to break the initial random walk.
7. **Architecture: FiLM conditioning** for the direction feature, plus auxiliary direction prediction loss.

The first three changes address the fundamental reward sparsity, duration pathology, and credit assignment problems. They should be implemented together as a single intervention. Changes 4–7 provide additional stability and convergence speed but are secondary to getting the reward right.