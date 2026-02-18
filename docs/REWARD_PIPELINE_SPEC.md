# Reward Pipeline Specification: High-Water-Mark A* Shaping with RND Exploration

**Document Version:** 1.0  
**Last Updated:** February 18, 2026  
**Training Architecture:** FiGAR-PPO with 256 parallel environments  
**Target Throughput:** 10,000+ iterations/s

---

## Table of Contents

1. [Design Rationale](#1-design-rationale)
2. [Two-Phase Reward Machine](#2-two-phase-reward-machine)
3. [High-Water-Mark A* Distance Shaping](#3-high-water-mark-a-distance-shaping)
4. [RND Exploration Bonus](#4-rnd-exploration-bonus)
5. [Composite Reward Function](#5-composite-reward-function)
6. [Integration with FiGAR-PPO](#6-integration-with-figar-ppo)
7. [Hyperparameter Schedule](#7-hyperparameter-schedule)
8. [Escalation Path: IQL Learned Potential](#8-escalation-path-iql-learned-potential)
9. [Implementation Details](#9-implementation-details)
10. [Verification and Diagnostics](#10-verification-and-diagnostics)
11. [Computational Budget](#11-computational-budget)

---

## 1. Design Rationale

### The Problem

N++ levels require the agent to navigate from spawn → exit switch → exit door through complex tile geometry. The sparse rewards (switch collection, exit reached) can be hundreds of macro-actions apart. A* path distance provides topology-aware guidance, but standard PBRS with Φ(s) = −distance penalizes necessary backtracking — the agent that retreats to build momentum for a running jump receives negative shaping reward, biasing exploration away from the optimal trajectory.

### Why High-Water-Mark

High-water-mark (HWM) progress tracking replaces the PBRS formulation with a simpler mechanism: the agent receives positive reward only when it achieves a **new minimum** A* distance to the current objective. Moving backward yields zero reward, not negative reward. This eliminates the non-monotonic penalty entirely.

The key insight for N++ specifically: A* distance is topology-aware, not merely geometric. If the agent approaches a gap from the wrong side and can't cross, the A* distance doesn't reach 0 — it reaches some minimum determined by the shortest graph path, which routes through the actual passable geometry. The watermark stalls, and the agent must find an alternative approach that achieves a genuinely lower A* distance, which by the graph's construction corresponds to real topological progress.

When A* distance reaches 0, the agent is physically co-located with the objective and the sparse reward fires. There is no disconnect between the shaping signal and task completion.

### Why Not Classical PBRS

HWM shaping is **not** potential-based reward shaping. It depends on episode history (the watermark), not just current state, so the Ng et al. (1999) optimal-policy-preservation guarantee does not hold. In practice this distinction rarely matters for PPO:

- The shaped reward cannot be exploited by cycling (reducing distance then increasing it yields one positive reward followed by zeros, not repeatable gains).
- The Sonic OpenAI Retro Contest (2018) validated HWM on momentum-based platformers — multiple top-placing PPO agents used exactly this formulation.
- The theoretical risk (shaped optimal policy ≠ true optimal policy) is dominated by the practical risk of standard PBRS (learning is catastrophically slowed by negative backtracking gradients, making the true optimal policy undiscoverable within finite training budgets).

### Why RND as Complement

HWM provides zero signal during backtracking — the agent isn't penalized, but it also receives no guidance. For levels requiring extended momentum-building sequences (backing up 100+ pixels before a critical jump), the agent must discover these strategies through exploration. RND (Random Network Distillation) provides intrinsic reward for visiting novel states, naturally pushing the agent into unexplored regions including backed-up, high-velocity states. RND was designed specifically for PPO with 128+ parallel environments (Burda et al., ICLR 2019).

### Performance Constraint

256 parallel environments at 10,000 iterations/s leaves ~390 μs per step across all environments. The reward pipeline must add negligible overhead. HWM shaping requires one comparison and one max operation per environment per step. A* distances are already computed for the observation space (reachability features `path_dist_to_switch_norm` and `path_dist_to_exit_norm`). RND requires two small MLP forward passes per step, batched across all environments.

---

## 2. Two-Phase Reward Machine

### State Machine

```
States:
    u0 = Phase 0: Switch not collected
    u1 = Phase 1: Switch collected, exit active
    u2 = Terminal: Exit reached (success)
    u_dead = Terminal: Death or timeout

Transitions:
    u0 →[ninja contacts exit switch] u1    (sparse reward: R_SWITCH)
    u1 →[ninja contacts exit door]   u2    (sparse reward: R_EXIT)
    any →[death]                     u_dead (sparse reward: R_DEATH)
    any →[frame 20,000]             u_dead (sparse reward: 0)
```

### Sparse Reward Values

```
R_SWITCH = 1.0     # Collecting the exit switch
R_EXIT   = 5.0     # Reaching the exit door (level completion)
R_DEATH  = -0.5    # Death penalty (mild — avoid discouraging exploration)
```

The asymmetry (R_EXIT ≫ R_SWITCH) reflects that completing the level is the true objective; the switch is an intermediate milestone.

### Phase-Dependent Objective

The HWM shaping tracks distance to a **phase-dependent objective**:

- **Phase 0 (u0):** Track A* distance to the exit switch.
- **Phase 1 (u1):** Track A* distance to the exit door.

At the phase transition (switch collection), the watermark resets to the A* distance to the new objective (exit door). This prevents the watermark from carrying over stale progress from the previous phase.

---

## 3. High-Water-Mark A* Distance Shaping

### Core Mechanism

```python
class HighWaterMarkShaping:
    """Per-environment, per-episode state for HWM reward shaping."""
    
    def __init__(self):
        self.watermark: float = float('inf')  # Best (minimum) A* distance seen
        self.current_phase: int = 0            # Reward machine state
    
    def reset(self, initial_a_star_distance: float, phase: int = 0):
        """Called at episode start."""
        self.watermark = initial_a_star_distance
        self.current_phase = phase
    
    def step(self, a_star_distance: float, phase: int) -> float:
        """
        Called every primitive frame. Returns shaping reward.
        
        Args:
            a_star_distance: Current A* path distance to phase objective.
                Phase 0: distance to switch. Phase 1: distance to exit.
                Already available from reachability features.
            phase: Current reward machine state (0 or 1).
        
        Returns:
            Shaping reward (>= 0). Positive when new progress achieved, zero otherwise.
        """
        # Phase transition: reset watermark for new objective
        if phase != self.current_phase:
            self.watermark = a_star_distance
            self.current_phase = phase
            return 0.0  # No shaping reward at transition (sparse reward handles it)
        
        # Check for new progress
        if a_star_distance < self.watermark:
            progress = self.watermark - a_star_distance  # Positive value
            self.watermark = a_star_distance
            return progress
        
        return 0.0  # No penalty for backtracking
```

### A* Distance Source

The A* distance is already computed per frame for the observation space reachability features. Specifically:

- `path_dist_to_switch_norm` (reachability index 4): Normalized A* path distance to switch, range `[0, 1]`.
- `path_dist_to_exit_norm` (reachability index 5): Normalized A* path distance to exit, range `[0, 1]`.

These are normalized by a level-dependent maximum. For the HWM computation, use the **unnormalized** A* distance (in pixels or tile units) to preserve meaningful magnitude differences. The normalization constant is level-specific and known at environment initialization.

```python
# Denormalize from observation features
a_star_distance_px = path_dist_norm * max_path_distance_px
```

If unnormalized distances are not readily available from the reachability computation, using normalized distances is acceptable — the HWM mechanism works identically, and the shaping coefficient α absorbs the scale difference.

### Normalization of Shaping Reward

Raw HWM progress values scale with level size — a large level can yield progress deltas of 100+ pixels, while a small level yields deltas of 10 pixels. Normalize per level:

```python
def compute_normalized_shaping(progress_px: float, max_path_distance_px: float) -> float:
    """Normalize progress to [0, 1] range relative to level's maximum path distance."""
    return progress_px / max_path_distance_px
```

This ensures the shaping reward magnitude is consistent across levels, preventing large levels from dominating the gradient.

### Properties

**Non-negative:** HWM shaping reward is always ≥ 0. The agent is never punished for any action. This eliminates the backtracking penalty problem entirely.

**Bounded cumulative:** The total HWM shaping reward over an episode is at most `max_path_distance / max_path_distance = 1.0` (normalized). This is the case where the agent starts at maximum distance and achieves monotonic progress to distance 0. In practice, the cumulative HWM reward is ≤ 1.0, providing a predictable magnitude for tuning the shaping coefficient.

**Non-exploitable (in practice):** The agent cannot farm HWM reward by cycling. Reducing distance from 100→50 yields 50 units of progress. Returning to 100 yields 0. Reducing to 50 again yields 0 (watermark is already at 50). The only way to earn more shaping reward is to reach a genuinely new minimum distance.

**History-dependent:** The watermark constitutes episode-level state not present in the observation. The agent's policy is conditioned on observations, not the watermark, so the policy is still Markovian with respect to the observation space. The watermark only affects the reward signal, not the state representation. This is the standard formulation used in Sonic Retro Contest agents.

---

## 4. RND Exploration Bonus

### Purpose

HWM shaping provides positive-reward-or-nothing, which eliminates backtracking penalties but provides no gradient in regions where the agent hasn't yet achieved progress. RND fills this gap by rewarding visits to novel states, encouraging the agent to explore the state space — including high-velocity states far from the goal that are necessary precursors to momentum-dependent jumps.

### Architecture

RND uses two networks with identical architecture but different roles:

```
Target Network (frozen, random weights):
    Input:  observation (153 dims)
    Hidden: Linear(153, 256) → ReLU → Linear(256, 256) → ReLU
    Output: Linear(256, 128)
    
    Weights: Initialized randomly, NEVER updated.

Predictor Network (trained):
    Input:  observation (153 dims)
    Hidden: Linear(153, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 256) → ReLU
    Output: Linear(256, 128)
    
    Weights: Trained to minimize prediction error against target.
```

The predictor has one extra hidden layer than the target, following Burda et al.'s recommendation. Both networks are small — the critical constraint is that batched forward passes across 256 environments must be fast.

### Intrinsic Reward Computation

```python
def compute_rnd_reward(obs: Tensor, target_net: Module, predictor_net: Module,
                       obs_rms: RunningMeanStd) -> Tensor:
    """
    Compute RND intrinsic reward for a batch of observations.
    
    Args:
        obs: [256, 153] batch of observations
        target_net: Frozen random network
        predictor_net: Trained prediction network
        obs_rms: Running mean/std for observation normalization
    
    Returns:
        intrinsic_reward: [256] per-environment intrinsic rewards
    """
    # Normalize observations (critical for RND stability)
    obs_normalized = (obs - obs_rms.mean) / (obs_rms.std + 1e-8)
    obs_normalized = torch.clamp(obs_normalized, -5.0, 5.0)
    
    with torch.no_grad():
        target_features = target_net(obs_normalized)       # [256, 128]
    predictor_features = predictor_net(obs_normalized)      # [256, 128]
    
    # MSE per observation (not per feature)
    prediction_error = (target_features - predictor_features).pow(2).mean(dim=1)  # [256]
    
    # Normalize intrinsic reward by running statistics
    return prediction_error
```

### Observation Normalization

**Critical for RND stability.** Before training begins, run a random policy for 50 rollouts (~50 × 256 × average_episode_length observations) to initialize the running mean and standard deviation. Without this, the initial prediction errors are dominated by observation scale differences rather than state novelty.

The running mean/std is updated throughout training using the standard Welford online algorithm.

### Intrinsic Reward Normalization

Normalize intrinsic rewards by a separate running standard deviation to keep their magnitude stable relative to extrinsic rewards:

```python
intrinsic_reward_normalized = intrinsic_reward / (intrinsic_reward_rms.std + 1e-8)
```

### Dual Value Heads

**This is essential.** RND requires separate value function estimates for extrinsic and intrinsic rewards, with potentially different discount factors:

```
Extrinsic Value Head:
    Input:  shared feature extractor output (384 dims)
    Hidden: Linear(384, 256) → ReLU
    Output: Linear(256, 1)
    Discount: γ_ext = 0.99

Intrinsic Value Head:
    Input:  shared feature extractor output (384 dims)
    Hidden: Linear(384, 256) → ReLU
    Output: Linear(256, 1)
    Discount: γ_int = 0.99 (can also use 0.999 for longer exploration horizon)
```

The shared feature extractor (the existing `ConfigurableMultimodalExtractor` producing 384-dim features) feeds both heads. Separate GAE computations are run for each reward stream. The combined advantage for the policy gradient is:

```
A_combined = A_extrinsic + A_intrinsic
```

The value function losses are summed:

```
L_value = L_value_extrinsic + L_value_intrinsic
```

### Predictor Training

The predictor network is trained alongside PPO using the same batch of observations. The loss is:

```python
predictor_loss = (target_features.detach() - predictor_features).pow(2).mean()
```

This is added to the PPO loss with a coefficient of 1.0 (no tuning needed — the gradient scales are naturally balanced).

---

## 5. Composite Reward Function

### Per-Primitive-Frame Reward

At every physics frame (primitive step), the total reward is:

```python
def compute_total_reward(
    # Sparse rewards
    r_sparse: float,           # R_SWITCH, R_EXIT, R_DEATH, or 0
    
    # HWM shaping
    hwm_progress: float,       # From HighWaterMarkShaping.step(), normalized [0, 1]
    
    # RND exploration
    rnd_reward: float,         # Normalized intrinsic reward
    
    # Coefficients
    alpha: float,              # HWM shaping weight
    beta: float,               # RND intrinsic reward weight
) -> tuple[float, float]:
    """
    Returns:
        r_extrinsic: Reward for extrinsic value head (sparse + HWM)
        r_intrinsic: Reward for intrinsic value head (RND only)
    """
    r_extrinsic = r_sparse + alpha * hwm_progress
    r_intrinsic = beta * rnd_reward
    
    return r_extrinsic, r_intrinsic
```

### Reward Stream Separation

The extrinsic and intrinsic rewards feed separate value heads and separate GAE computations. This prevents the non-stationary RND signal from corrupting the extrinsic value estimates. As the predictor improves, intrinsic rewards decay naturally for frequently-visited states without affecting the extrinsic value landscape.

### Recommended Coefficient Values

```
α (HWM shaping weight):     1.0    # Start here; HWM is already [0,1]-normalized
β (RND intrinsic weight):   0.01   # Small relative to extrinsic — exploration aid, not primary signal

R_SWITCH:                    1.0
R_EXIT:                      5.0
R_DEATH:                    -0.5
```

### Reward Magnitude Analysis

Over a typical episode of ~2000 frames on a moderate-difficulty level:

```
Sparse rewards:
    Switch collection:       1.0 (once)
    Exit completion:         5.0 (once)
    Total sparse:            6.0

HWM shaping (α=1.0):
    Maximum cumulative:      1.0 (full path traversal, normalized)
    Typical cumulative:      ~0.6-0.9 (some backtracking)
    Per-frame average:       ~0.0003-0.0005

RND intrinsic (β=0.01):
    Per-frame:               ~0.001-0.05 (highly variable, decays with training)
    Cumulative per episode:  ~2-100 early, ~0.5-5 late training
```

The sparse rewards dominate the extrinsic signal, with HWM providing consistent intermediate gradient. RND operates on its own value head and decays naturally. This balance ensures the agent primarily optimizes for level completion while receiving navigational guidance and exploration pressure.

---

## 6. Integration with FiGAR-PPO

### Primitive-Level Computation

All rewards are computed at every primitive frame, not at macro-action decision points. This provides the densest possible value function training signal.

```
For a macro-action of duration k starting at frame t:
    Primitive rewards at each frame i ∈ [t, t+k-1]:
        r_ext_i = r_sparse_i + α × hwm_progress_i
        r_int_i = β × rnd_reward_i
    
    Extrinsic GAE deltas:
        δ_ext_i = r_ext_i + γ_ext × V_ext(s_{i+1}) − V_ext(s_i)
    
    Intrinsic GAE deltas:
        δ_int_i = r_int_i + γ_int × V_int(s_{i+1}) − V_int(s_i)
    
    Combined advantage at decision point t:
        A_t = Σ_{j=0}^{k-1} (γ_ext × λ)^j × δ_ext_{t+j}
            + Σ_{j=0}^{k-1} (γ_int × λ)^j × δ_int_{t+j}
```

### Policy Loss

Computed only at macro-action decision points (start of each macro-action):

```
L_policy = -E[ min(r_t(θ) × A_t, clip(r_t(θ), 1-ε, 1+ε) × A_t) ]

where r_t(θ) = π_θ(a_t, w_t | s_t) / π_θ_old(a_t, w_t | s_t)
      = [π_action(a_t | s_t) × π_repeat(w_t | s_t)] / [π_action_old × π_repeat_old]
```

### Value Losses

Computed at **all** primitive timesteps:

```
L_value = 0.5 × MSE(V_ext(s_i), target_ext_i) + 0.5 × MSE(V_int(s_i), target_int_i)
```

Both value heads receive supervision at every primitive frame, giving the value function much richer signal than macro-action-only supervision.

### Interrupt Handling

When a FiGAR macro-action is interrupted (contact state change: landing, leaving surface, wall contact), the macro-action ends early. The reward pipeline handles this naturally — each primitive frame within the interrupted macro-action received its own HWM + RND reward. No special correction is needed.

### Discount Factor Consistency

```
γ_ext = 0.99   # Used in extrinsic GAE and return computation
γ_int = 0.99   # Used in intrinsic GAE (can be tuned independently — 0.999 for longer horizon)
λ     = 0.95   # GAE lambda, shared across both streams
```

---

## 7. Hyperparameter Schedule

### Phase 1: Exploration-Heavy (0–30% of training)

The agent is learning basic movement and discovering level structure. RND drives exploration; HWM provides directional gradient.

```
α = 1.0        # Full HWM shaping
β = 0.01       # RND active
entropy_coeff = 0.05   # High entropy for action diversity
                        # Consider separate entropy coefficients:
                        #   action_head:  0.05
                        #   repeat_head:  0.03  (duration needs less exploration)
```

### Phase 2: Exploitation Transition (30–70% of training)

The agent has discovered viable routes. Reduce exploration pressure, let the policy refine.

```
α = 1.0        # HWM shaping maintained (never harmful)
β = 0.005      # RND reduced (predictor has learned common states anyway)
entropy_coeff = 0.02   # Moderate entropy
```

### Phase 3: Fine-Tuning (70–100% of training)

The agent is optimizing execution. Minimal exploration, shaping maintained for stability.

```
α = 1.0        # HWM shaping maintained
β = 0.001      # Minimal RND (effectively off for well-explored states)
entropy_coeff = 0.01   # Standard PPO entropy
```

### Schedule Implementation

```python
def get_schedule_params(progress: float) -> dict:
    """
    Args:
        progress: Training progress in [0, 1] (current_step / total_steps)
    """
    # α: constant throughout (HWM is never harmful)
    alpha = 1.0
    
    # β: exponential decay
    beta = 0.01 * (0.1 ** progress)  # 0.01 → 0.001 over training
    
    # Entropy: linear decay with floor
    entropy_coeff = max(0.01, 0.05 - 0.04 * progress)  # 0.05 → 0.01
    
    return {"alpha": alpha, "beta": beta, "entropy_coeff": entropy_coeff}
```

### Gradient Clipping

```
max_grad_norm = 0.5   # Standard PPO; tighten to 0.3 if advantage variance is high
                       # from variable-length FiGAR macro-actions
```

---

## 8. Escalation Path: IQL Learned Potential

### When to Escalate

Monitor per-level training success rate. If a level shows:

- **< 5% switch collection rate** after 30% of training budget, AND
- **Watermark stagnation** (the minimum A* distance achieved across all episodes is not decreasing over 1M+ steps)

...the level likely requires momentum-dependent strategies that the agent cannot discover through RND exploration alone. These are levels where the agent needs positive guidance saying "build speed here" rather than just "don't penalize backing up."

### IQL Escalation Procedure

Expert demonstrations are available. For stalled levels:

1. **Extract demonstrations for the specific level.** Even a few trajectories (5–20) suffice for IQL if they show diverse strategies.

2. **Train a small IQL value network** per level:
    ```
    Input:  27-dim game state (position, velocity, contact state, jump_duration_remaining)
    Hidden: Linear(27, 128) → ReLU → Linear(128, 128) → ReLU
    Output: Linear(128, 1)
    
    Hyperparameters:
        expectile τ = 0.9     # Aggressive stitching
        discount γ = 0.99
        learning rate = 3e-4
        batch_size = 256
        training steps: 50,000  (~2 minutes on GPU)
    ```

3. **Inject as additional shaping signal** alongside HWM, not replacing it:
    ```python
    # Only for escalated levels
    r_extrinsic = r_sparse + α × hwm_progress + α_iql × (γ × V_iql(s') − V_iql(s))
    ```
    With `α_iql = 0.3` initially, annealed to 0 over training as the PPO critic improves.

4. **Runtime cost:** One additional `[256, 27]` forward pass through a 2-layer MLP per step, cached across steps (Φ(s') at step t reused as Φ(s) at step t+1). Cost: ~20–30 μs per step. Well within budget.

### Decision Criteria Summary

```
Level success rate after 30% training:
    > 5%  → Continue with HWM + RND (working)
    < 5%, watermark improving → Continue, increase RND β temporarily
    < 5%, watermark stagnant → Escalate to IQL
```

---

## 9. Implementation Details

### Per-Environment Episode State

Each of the 256 parallel environments maintains:

```python
@dataclass
class EpisodeRewardState:
    # HWM tracking
    watermark_switch: float = float('inf')   # Best A* distance to switch
    watermark_exit: float = float('inf')     # Best A* distance to exit
    current_phase: int = 0                    # 0=pre-switch, 1=post-switch
    
    # Normalization
    max_path_dist_switch: float = 1.0        # Level's maximum A* dist to switch
    max_path_dist_exit: float = 1.0          # Level's maximum A* dist to exit
    
    def reset(self, initial_dist_switch: float, max_dist_switch: float,
              initial_dist_exit: float, max_dist_exit: float):
        self.watermark_switch = initial_dist_switch
        self.watermark_exit = float('inf')
        self.current_phase = 0
        self.max_path_dist_switch = max_dist_switch
        self.max_path_dist_exit = max_dist_exit
    
    def step(self, dist_switch: float, dist_exit: float, 
             switch_activated: bool) -> float:
        """Returns normalized HWM shaping reward."""
        new_phase = 1 if switch_activated else 0
        
        # Phase transition
        if new_phase != self.current_phase:
            self.current_phase = new_phase
            self.watermark_exit = dist_exit
            return 0.0
        
        if self.current_phase == 0:
            if dist_switch < self.watermark_switch:
                progress = self.watermark_switch - dist_switch
                self.watermark_switch = dist_switch
                return progress / self.max_path_dist_switch
            return 0.0
        else:
            if dist_exit < self.watermark_exit:
                progress = self.watermark_exit - dist_exit
                self.watermark_exit = dist_exit
                return progress / self.max_path_dist_exit
            return 0.0
```

### Batched RND Forward Pass

```python
class RNDModule:
    def __init__(self, obs_dim=153, hidden_dim=256, feature_dim=128):
        # Target: frozen random network
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        for p in self.target.parameters():
            p.requires_grad = False
        
        # Predictor: trained network (one extra layer)
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Observation normalization (initialize with random policy data)
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.reward_rms = RunningMeanStd(shape=())
    
    def compute_intrinsic_reward(self, obs: Tensor) -> Tensor:
        """
        Args:
            obs: [256, 153] batched observations
        Returns:
            intrinsic_reward: [256] normalized rewards
        """
        obs_norm = torch.clamp(
            (obs - self.obs_rms.mean) / (self.obs_rms.std + 1e-8),
            -5.0, 5.0
        )
        
        with torch.no_grad():
            target_features = self.target(obs_norm)
        predictor_features = self.predictor(obs_norm)
        
        error = (target_features - predictor_features).pow(2).mean(dim=1)
        
        # Normalize by running std
        return error / (self.reward_rms.std + 1e-8)
    
    def compute_predictor_loss(self, obs: Tensor) -> Tensor:
        """Training loss for predictor network. Called during PPO update."""
        obs_norm = torch.clamp(
            (obs - self.obs_rms.mean) / (self.obs_rms.std + 1e-8),
            -5.0, 5.0
        )
        target_features = self.target(obs_norm).detach()
        predictor_features = self.predictor(obs_norm)
        return (target_features - predictor_features).pow(2).mean()
```

### RND Observation Initialization

Before training begins, run a random policy for 50 rollouts to warm up `obs_rms`:

```python
def initialize_rnd_normalization(envs, rnd_module, n_rollouts=50):
    """Collect random policy observations to initialize RND normalization."""
    obs = envs.reset()
    for _ in range(n_rollouts):
        for step in range(envs.max_episode_length):
            actions = envs.action_space.sample()  # Random actions
            obs, _, dones, _ = envs.step(actions)
            rnd_module.obs_rms.update(obs)
            if dones.all():
                break
        obs = envs.reset()
```

---

## 10. Verification and Diagnostics

### Correctness Tests

**HWM monotonicity.** The watermark should never increase within a phase. Assert `watermark_new <= watermark_old` after every step call.

**HWM non-negativity.** The shaping reward should never be negative. Assert `hwm_progress >= 0` at every step.

**HWM bounded cumulative.** Over any episode, the sum of normalized HWM rewards should be ≤ 1.0 per phase (≤ 2.0 total for both phases). Log and alert if exceeded — indicates a normalization bug.

**Phase transition.** At the frame of switch collection, verify:
- `current_phase` transitions from 0 to 1
- `watermark_exit` is initialized to the current A* distance to exit
- HWM shaping reward at the transition frame is 0.0 (sparse reward handles it)

**Terminal states.** At death, timeout, or success, verify no shaping reward is applied. The episode ends; HWM state is discarded.

**RND reward decay.** Over training, the average RND intrinsic reward for frequently-visited states (e.g., spawn area) should decrease. If it doesn't, the predictor isn't learning — check observation normalization.

### Diagnostic Metrics (Log Every 1000 Steps)

```
# Per-level aggregates
hwm_progress_per_episode_mean     # Average total HWM reward per episode
hwm_progress_per_episode_max      # Best episode's total HWM reward
watermark_best_switch             # All-time minimum A* distance to switch
watermark_best_exit               # All-time minimum A* distance to exit

# RND health
rnd_reward_mean                   # Average intrinsic reward (should decrease over training)
rnd_reward_std                    # Should stabilize
rnd_predictor_loss                # Should decrease

# Success metrics
switch_collection_rate            # Fraction of episodes reaching switch
exit_completion_rate              # Fraction of episodes completing level
episode_length_mean               # Should decrease as agent improves
episode_length_at_success         # Frame count for successful episodes

# Escalation triggers
watermark_improvement_rate        # Rate of watermark decrease over last N steps
                                  # If near zero + low success → escalation candidate
```

### Visualization

**Watermark progression plot.** For each level, plot the minimum watermark achieved over training. Should show monotonic decrease toward 0. Stagnation indicates a stuck agent.

**Reward composition plot.** Per-episode stacked bar chart of sparse reward, HWM reward, and RND reward (cumulative per episode). Sparse should dominate in successful episodes; HWM should dominate in partial-progress episodes; RND should be largest in exploration-heavy early episodes.

**RND heatmap.** Render RND prediction error as a 2D heatmap over (x, y) positions. High-error regions are under-explored. Useful for diagnosing whether the agent is failing to explore critical regions.

---

## 11. Computational Budget

### Per-Step Cost Breakdown (256 Environments)

```
Component                  | Operation              | Time (μs)
---------------------------|------------------------|----------
A* distance lookup         | Already computed (obs)  | 0
HWM shaping                | 256 × (compare + max)  | < 1
RND forward (target)       | [256, 153] MLP         | ~40-60
RND forward (predictor)    | [256, 153] MLP         | ~40-60
RND reward normalization   | Running stats update    | < 1
Reward composition         | 256 × (add + multiply)  | < 1
---------------------------|------------------------|----------
Total reward pipeline      |                        | ~80-120
```

### Budget Context

```
Available per-step budget at 10,000 iter/s:  ~100 μs total
Existing cost (PPO policy + value forward):  ~50-80 μs (estimated)
Remaining budget for rewards:                ~20-50 μs (tight)
```

**Assessment:** The RND forward passes (~80–120 μs) are the most expensive component and may slightly exceed the per-step budget. Mitigations:

1. **Reduce RND hidden dimension** from 256 to 128. Cuts forward pass time roughly in half (~40–60 μs total for both networks). Minimal impact on exploration quality.

2. **Evaluate RND every N frames** instead of every frame. At N=4, the cost amortizes to ~20–30 μs per step. Intrinsic reward at non-evaluation frames is set to the last computed value. Exploration quality impact is minimal — nearby frames have similar novelty.

3. **Fuse RND with PPO forward pass.** Share the feature extractor output (384 dims) as input to RND instead of raw observations. The feature extractor forward pass is already paid for. RND then operates on a 384→128 projection (~20 μs for both networks combined).

**Recommended approach:** Option 3 (fused features). The 384-dim feature vector from `ConfigurableMultimodalExtractor` already captures the relevant state information. RND on features rather than raw observations is a known-effective variant.

```
Revised RND architecture (fused):
    Target:    Linear(384, 128) → ReLU → Linear(128, 64)
    Predictor: Linear(384, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 64)
    
    Forward pass (both): ~15-25 μs for [256, 384] batch
```

### Memory Budget

```
Component                          | Memory
-----------------------------------|--------
HWM state (256 envs)               | ~6 KB (trivial)
RND target network                 | ~200 KB (153×256 + 256×256 + 256×128)
RND predictor network              | ~300 KB (one extra layer)
RND observation running stats      | ~2.4 KB (mean + var for 153 dims)
RND reward running stats           | < 1 KB
-----------------------------------|--------
Total                              | ~0.5 MB
```

Negligible relative to the PPO rollout buffer and model parameters.

---

## Appendix A: Comparison with Alternative Approaches

| Property | HWM A* (this spec) | Classical PBRS (Φ = −distance) | IQL Learned Potential | BSRS (critic as Φ) |
|---|---|---|---|---|
| Backtracking penalty | None (zero) | Negative (harmful) | Positive (if demos show it) | Eventually learned |
| Implementation effort | Trivial | Trivial | Moderate (offline pipeline) | Trivial |
| Runtime cost | ~0 | ~0 | ~30 μs/step | ~0 (reuse critic) |
| Coverage | Full (A* everywhere) | Full | Demo-distribution only | Learned over training |
| Velocity awareness | None | None | Yes (from demos) | Eventually |
| Cold-start quality | Good (immediate gradient) | Good (but wrong sign) | Good (pre-trained) | Bad (random critic) |
| Theoretical guarantee | None (history-dependent) | Optimal policy preserved | Optimal policy preserved | Convergence proof (tabular) |
| Practitioner validation | Sonic Contest top-5 | Widely used, known failure modes | Research-stage | AAAI 2025 |

### When HWM A* Is Sufficient

Levels where the agent can discover momentum-dependent strategies through exploration (most levels). The zero-penalty for backtracking removes the primary learning barrier, and FiGAR macro-actions naturally produce the "hold direction for N frames" patterns needed for run-ups. RND provides additional exploration pressure for non-obvious strategies.

### When IQL Escalation Is Needed

Levels requiring specific, non-obvious momentum strategies that random exploration is unlikely to discover within the training budget. Signs: watermark stagnation + low success rate after 30% of training. These levels benefit from demonstration-derived value estimates that provide positive gradient for "build speed" states.

---

## Appendix B: Relationship to Existing Infrastructure

### Reachability Features (Observation Space)

The HWM shaping consumes the same A* distance values used to compute `path_dist_to_switch_norm` and `path_dist_to_exit_norm` in the observation space. The observation features continue to serve their purpose: providing the policy network with path-guidance signal. The HWM operates on the same data but in the reward pipeline, not the observation pipeline. No new computation is required.

### Mine SDF Features

Mine avoidance operates orthogonally to HWM goal-seeking. The agent receives HWM reward for progress toward the objective and learns mine avoidance through negative environmental rewards (death penalty) and the mine SDF/spatial context observation features. No interaction between the two systems.

### Phase Indicator (`switch_activated`)

The observation-space `switch_activated` feature (reachability index 12) is the same signal used to drive the reward machine phase transitions. The HWM shaping reads this to determine which objective's distance to track. Single source of truth, no duplication.

### A* Next-Hop Direction

The `next_hop_direction` observation feature (reachability indices 13–14) provides the policy network with frame-by-frame directional guidance from A*. This complements HWM shaping: the direction tells the policy *which way to go*, while HWM rewards the policy *for making progress*. Both are derived from the same A* graph but serve different roles (observation vs. reward).
