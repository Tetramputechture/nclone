# Training Pipeline Specification v2: SAC-Discrete + RFCL with Demonstration Replay

**Document Version:** 2.0  
**Last Updated:** February 22, 2026  
**Replaces:** REWARD_PIPELINE_SPEC.md (HWM + RND approach)  
**Architecture:** SAC-Discrete with Reverse Forward Curriculum Learning  
**Target:** 50+ N++ levels, 256 parallel environments

---

## Table of Contents

1. [Design Rationale](#1-design-rationale)
2. [Algorithm: SAC-Discrete](#2-algorithm-sac-discrete)
3. [Reverse Forward Curriculum Learning (RFCL)](#3-reverse-forward-curriculum-learning-rfcl)
4. [Demonstration Integration (RLPD)](#4-demonstration-integration-rlpd)
5. [Action Space: Discretized FiGAR](#5-action-space-discretized-figar)
6. [Reward Function](#6-reward-function)
7. [Network Architecture](#7-network-architecture)
8. [Replay Buffer Design](#8-replay-buffer-design)
9. [Two-Phase Task Structure](#9-two-phase-task-structure)
10. [Training Loop](#10-training-loop)
11. [State Reset Interface](#11-state-reset-interface)
12. [Hyperparameters](#12-hyperparameters)
13. [Curriculum Progression Logic](#13-curriculum-progression-logic)
14. [Observation Space Changes](#14-observation-space-changes)
15. [Verification and Diagnostics](#15-verification-and-diagnostics)
16. [Computational Budget](#16-computational-budget)
17. [Migration from PPO](#17-migration-from-ppo)

---

## 1. Design Rationale

### Why Replace PPO

The PPO + HWM pipeline has a fundamental structural problem: HWM provides zero reward signal on 80-93% of timesteps (all backtracking steps), and PPO discards all experience after each update. The agent random-walks and occasionally stumbles into goals, but the behavior never consolidates into directed navigation. The root cause is twofold:

1. **A* distance is geometrically misleading.** The A* graph traces topologically shortest paths through tile connectivity, but these paths are often physically impossible given the agent's momentum, jump physics, and surface contact requirements. Any reward signal derived from A* distance (HWM, PBRS, velocity-alignment) inherits this unreliability.

2. **PPO is on-policy.** The rare useful transitions (successful navigation segments) are used for one gradient update, then discarded. With 80-93% of experience being uninformative, the effective sample efficiency is catastrophically low.

### Why SAC-Discrete

SAC (Soft Actor-Critic) addresses both problems:

- **Off-policy replay.** All experience is stored in a replay buffer and resampled many times. The 7-20% of transitions containing useful navigation signal are replayed across dozens of gradient updates, cementing the lessons. Demonstration transitions are permanently available in the buffer.
- **Entropy-regularized.** SAC automatically balances exploration and exploitation via a learned temperature parameter alpha. No manual entropy coefficient scheduling.
- **Stable with function approximation.** Clipped double-Q prevents overestimation. Target networks provide stable bootstrapping targets.
- **Discrete variant is well-established.** Christodoulou (2019) adapted SAC for discrete actions, computing expectations over the action distribution analytically rather than via sampling.

### Why RFCL

Reverse Forward Curriculum Learning (Tao et al., ICLR 2024) eliminates the need for any distance-based reward shaping:

- **Demonstrations provide physics-valid waypoints.** Human demos show actual executable trajectories, bypassing the A*-vs-physics mismatch entirely.
- **Reverse curriculum provides the learning gradient.** By starting near the goal and working backward, even sparse rewards (+1 for reaching goal) provide dense learning signal -- the agent only needs to navigate 50-100px to succeed.
- **Forward curriculum generalizes to the spawn point.** Once the agent can solve the task from demo states, the initial state distribution expands to include the actual level spawn.
- **Works with few, suboptimal demos.** RFCL's authors validate on 1-10 demonstrations per task with human teleoperation data.
- **State resets are the key requirement.** You can reset to arbitrary states in your simulation, satisfying RFCL's core assumption.

### Why RLPD-Style Demonstration Mixing

RLPD (Ball et al., 2023) showed that simply mixing 50% demonstration data with 50% online data in a SAC replay buffer -- with high update-to-data ratio and LayerNorm -- achieves state-of-the-art on Adroit manipulation tasks. This is the simplest possible way to leverage demonstrations: no offline pretraining phase, no special loss functions, just data mixing. The demonstrations serve as a permanent source of high-quality transitions that the Q-function learns from alongside the agent's own experience.

---

## 2. Algorithm: SAC-Discrete

### Core Formulation

SAC-Discrete maximizes the entropy-regularized objective:

```
J(pi) = Sum_t E[ r_t + alpha * H(pi(.|s_t)) ]
```

Where alpha is the temperature parameter controlling the exploration-exploitation tradeoff, learned automatically to maintain a target entropy.

### Key Equations

**Soft Q-function update (clipped double-Q):**

```python
# Two Q-networks with separate parameters
Q_target_value = r + gamma * (1 - done) * (
    sum_a' pi(a'|s') * (min(Q1_targ(s',a'), Q2_targ(s',a')) - alpha * log(pi(a'|s')))
)

Q1_loss = MSE(Q1(s, a), Q_target_value.detach())
Q2_loss = MSE(Q2(s, a), Q_target_value.detach())
```

**Policy update:**

```python
# For discrete actions, policy is a categorical distribution
pi_logits = policy_net(s)           # [batch, n_actions]
pi_probs = softmax(pi_logits)       # [batch, n_actions]
log_pi = log_softmax(pi_logits)     # [batch, n_actions]

# Policy loss: maximize Q - alpha * log_pi, expectation over actions
Q_min = torch.min(Q1(s), Q2(s))    # [batch, n_actions]
policy_loss = (pi_probs * (alpha * log_pi - Q_min)).sum(dim=1).mean()
```

**Temperature update (automatic entropy tuning):**

```python
# Target entropy for discrete actions: -log(1/|A|) * target_ratio
# With |A| = 48 (6 base actions * 8 durations): target = -log(1/48) * 0.5 ~ 1.93
target_entropy = -np.log(1.0 / n_actions) * target_entropy_ratio

alpha_loss = -(log_alpha * (log_pi_mean + target_entropy).detach()).mean()
```

### Target Network Updates

Polyak averaging, not hard replacement:

```python
# After each gradient step:
for param, target_param in zip(Q.parameters(), Q_target.parameters()):
    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

With tau = 0.005 (standard SAC).

---

## 3. Reverse Forward Curriculum Learning (RFCL)

### Overview

Training proceeds in two phases per level:

1. **Reverse curriculum (Phase R):** Start near the goal, expand backward along demonstration trajectories.
2. **Forward curriculum (Phase F):** Start from the actual spawn, generalize using knowledge from Phase R.

### Demonstration State Extraction

From each human demonstration, extract snapshots at regular intervals:

```python
@dataclass
class DemoSnapshot:
    """A single state snapshot from a demonstration trajectory."""
    frame_index: int              # Frame number in the original demo
    # Full ninja state for simulation reset
    position_x: float
    position_y: float
    velocity_x: float
    velocity_y: float
    movement_state: int           # 0-9 (immobile, running, jumping, etc.)
    jump_duration_remaining: int  # Frames of jump hold remaining
    # Surface contact
    floor_contact: bool
    wall_contact: bool
    floor_normal_x: float
    floor_normal_y: float
    wall_normal_x: float
    # Entity state
    switch_activated: bool
    # Additional state needed for faithful reset
    applied_gravity: float
    applied_drag: float
    applied_friction: float

def extract_demo_snapshots(demo_replay, interval_frames=30):
    """
    Extract snapshots every `interval_frames` from a demo replay.
    
    Args:
        demo_replay: Full frame-by-frame recording of a human play session
        interval_frames: Sample every N frames (30 = 0.5 seconds at 60fps)
    
    Returns:
        List[DemoSnapshot] ordered from start (index 0) to end (index -1, near goal)
    """
    snapshots = []
    for frame_idx in range(0, len(demo_replay), interval_frames):
        state = demo_replay[frame_idx]
        snapshots.append(DemoSnapshot(
            frame_index=frame_idx,
            position_x=state.ninja_x,
            position_y=state.ninja_y,
            velocity_x=state.ninja_vx,
            velocity_y=state.ninja_vy,
            movement_state=state.ninja_state,
            jump_duration_remaining=state.jump_remaining,
            floor_contact=state.floor_count > 0,
            wall_contact=state.wall_count > 0,
            floor_normal_x=state.floor_nx,
            floor_normal_y=state.floor_ny,
            wall_normal_x=state.wall_nx,
            switch_activated=state.switch_collected,
            applied_gravity=state.gravity,
            applied_drag=state.drag,
            applied_friction=state.friction,
        ))
    return snapshots
```

**Snapshot density:** 30-frame intervals yield ~33-100 snapshots per demo (for 1000-3000 frame demos). This provides sufficient granularity for the reverse curriculum to expand gradually.

### Reverse Curriculum (Phase R)

```python
class ReverseCurriculum:
    """
    Per-level, per-demo reverse curriculum.
    Manages which demo snapshots are available as initial states.
    """
    def __init__(self, demo_snapshots: list[DemoSnapshot], 
                 goal_type: str = "switch"):
        self.snapshots = demo_snapshots  # Ordered: start -> end
        self.goal_type = goal_type       # "switch" or "exit"
        
        # Start at the last snapshot (nearest to goal)
        self.frontier_index = len(demo_snapshots) - 1
        self.success_history = deque(maxlen=200)
        
        # Per-snapshot success tracking
        self.snapshot_successes = defaultdict(lambda: deque(maxlen=50))
    
    def sample_reset_state(self) -> DemoSnapshot:
        """
        Sample an initial state for the next episode.
        Samples from a window around the frontier, biased toward
        snapshots with moderate success rate (not too easy, not too hard).
        """
        # Window: frontier to end of demo
        window_start = self.frontier_index
        window_end = len(self.snapshots) - 1
        
        if window_start >= window_end:
            return self._add_noise(self.snapshots[window_end])
        
        # Prioritize snapshots near the frontier (hardest reachable)
        # Geometric distribution: higher probability for harder (earlier) states
        window_size = window_end - window_start + 1
        weights = np.array([0.7 ** i for i in range(window_size)])
        weights /= weights.sum()
        
        idx = window_start + np.random.choice(window_size, p=weights)
        return self._add_noise(self.snapshots[idx])
    
    def _add_noise(self, snapshot: DemoSnapshot) -> DemoSnapshot:
        """Add small perturbation to avoid overfitting to exact demo states."""
        noisy = copy.deepcopy(snapshot)
        noisy.position_x += np.random.normal(0, 2.0)   # ~2px position noise
        noisy.position_y += np.random.normal(0, 2.0)
        noisy.velocity_x += np.random.normal(0, 0.15)   # Small velocity noise
        noisy.velocity_y += np.random.normal(0, 0.15)
        return noisy
    
    def report_result(self, snapshot_idx: int, success: bool):
        """Report episode outcome for curriculum advancement."""
        self.success_history.append(float(success))
        self.snapshot_successes[snapshot_idx].append(float(success))
        
        # Check if frontier should advance (move earlier in demo)
        if len(self.success_history) >= 50:
            frontier_rate = self._frontier_success_rate()
            if frontier_rate >= 0.5 and self.frontier_index > 0:
                # Advance frontier: expand to earlier (harder) demo states
                step = max(1, int(len(self.snapshots) * 0.05))  # 5% of demo
                self.frontier_index = max(0, self.frontier_index - step)
                self.success_history.clear()
    
    def _frontier_success_rate(self) -> float:
        """Success rate for snapshots near the frontier."""
        frontier_window = range(self.frontier_index, 
                                 min(self.frontier_index + 5, len(self.snapshots)))
        successes = []
        for idx in frontier_window:
            successes.extend(self.snapshot_successes[idx])
        return np.mean(successes) if successes else 0.0
    
    @property
    def is_complete(self) -> bool:
        """Reverse curriculum is complete when frontier reaches demo start."""
        return self.frontier_index == 0 and self._frontier_success_rate() >= 0.3
```

### Forward Curriculum (Phase F)

Once the reverse curriculum completes (agent can solve from the demo's starting state), transition to the forward curriculum that generalizes to the actual level spawn:

```python
class ForwardCurriculum:
    """
    Expands from demo-solvable states to the actual level spawn.
    Uses PLR-style sampling.
    """
    def __init__(self, level_spawn_state: DemoSnapshot,
                 demo_snapshots: list[DemoSnapshot]):
        self.spawn = level_spawn_state
        self.demo_snapshots = demo_snapshots
        self.spawn_mix_ratio = 0.2    # Start with 20% spawn resets
        self.success_history = deque(maxlen=200)
    
    def sample_reset_state(self) -> DemoSnapshot:
        """Mix between actual spawn and demo states."""
        if np.random.random() < self.spawn_mix_ratio:
            return self._add_noise(self.spawn)
        else:
            # Sample any demo state (uniform -- agent should solve from all)
            idx = np.random.randint(0, len(self.demo_snapshots))
            return self._add_noise(self.demo_snapshots[idx])
    
    def report_result(self, from_spawn: bool, success: bool):
        """Increase spawn ratio as agent improves."""
        if from_spawn:
            self.success_history.append(float(success))
        
        if len(self.success_history) >= 50:
            spawn_rate = np.mean(self.success_history)
            if spawn_rate >= 0.3:
                # Increase spawn ratio (more practice from actual start)
                self.spawn_mix_ratio = min(0.9, self.spawn_mix_ratio + 0.1)
                self.success_history.clear()
    
    @property
    def is_complete(self) -> bool:
        return self.spawn_mix_ratio >= 0.9 and np.mean(self.success_history) >= 0.3
```

### Multi-Demo Handling

When a level has multiple demonstrations:

```python
class LevelCurriculum:
    """Manages multiple demo curricula for a single level."""
    def __init__(self, level_id: str, demos: list[list[DemoSnapshot]],
                 spawn_state: DemoSnapshot):
        # One reverse curriculum per demo
        self.reverse_curricula = [
            ReverseCurriculum(demo) for demo in demos
        ]
        self.forward_curriculum = ForwardCurriculum(spawn_state, 
                                                      [s for d in demos for s in d])
        self.phase = "reverse"  # "reverse" or "forward"
    
    def sample_reset_state(self) -> DemoSnapshot:
        if self.phase == "reverse":
            # Sample from a random non-complete reverse curriculum
            active = [rc for rc in self.reverse_curricula if not rc.is_complete]
            if not active:
                self.phase = "forward"
                return self.forward_curriculum.sample_reset_state()
            rc = np.random.choice(active)
            return rc.sample_reset_state()
        else:
            return self.forward_curriculum.sample_reset_state()
```

---

## 4. Demonstration Integration (RLPD)

### Mechanism

Following RLPD (Ball et al., 2023), demonstration transitions are stored permanently in the replay buffer and sampled alongside online experience:

```python
class DemonstrationReplayMixer:
    """
    Mixes demonstration and online transitions during sampling.
    RLPD-style: fixed ratio, no special weighting.
    """
    def __init__(self, demo_ratio=0.25):
        self.demo_ratio = demo_ratio  # 25% demo, 75% online
    
    def sample_batch(self, online_buffer, demo_buffer, batch_size=256):
        n_demo = int(batch_size * self.demo_ratio)
        n_online = batch_size - n_demo
        
        demo_batch = demo_buffer.sample(min(n_demo, len(demo_buffer)))
        online_batch = online_buffer.sample(min(n_online, len(online_buffer)))
        
        return concatenate_batches(demo_batch, online_batch)
```

### Demo Transition Format

Convert demonstration replays into (s, a, r, s', done) transitions:

```python
def convert_demo_to_transitions(demo_replay, level_env):
    """
    Convert a demonstration replay into replay buffer transitions.
    
    Args:
        demo_replay: Frame-by-frame state+action recording
        level_env: Environment instance for observation computation
    
    Returns:
        List of (obs, action_idx, reward, next_obs, done) tuples
    """
    transitions = []
    for i in range(len(demo_replay) - 1):
        frame = demo_replay[i]
        next_frame = demo_replay[i + 1]
        
        # Compute observation from state (uses existing observation pipeline)
        obs = level_env.compute_observation(frame)
        next_obs = level_env.compute_observation(next_frame)
        
        # Map demo's (horizontal_input, jump_input) to discrete action index
        # Demos are frame-by-frame, so duration = 1
        action_idx = encode_action(frame.horizontal_input, frame.jump_input,
                                    duration=1)
        
        # Compute reward using the sparse reward function
        reward = compute_sparse_reward(frame, next_frame)
        done = next_frame.is_terminal
        
        transitions.append((obs, action_idx, reward, next_obs, done))
    
    return transitions
```

**Note on demo action durations:** Human demos record frame-by-frame actions (duration=1). This means demo transitions always use duration-1 actions. The agent learns longer durations from its own experience. This is acceptable -- the Q-function learns that duration-1 actions have certain values from demos, and discovers that longer durations can be more efficient through online exploration.

### Demo Ratio Schedule

Start with higher demo ratio, decay as online experience accumulates:

```python
def get_demo_ratio(total_steps: int) -> float:
    """
    Decay demo ratio as online experience grows.
    High initially (agent has little useful online data).
    Low later (agent's own experience is more relevant).
    """
    if total_steps < 1_000_000:
        return 0.50  # 50% demo early (bootstrapping)
    elif total_steps < 10_000_000:
        return 0.25  # 25% demo mid-training
    else:
        return 0.10  # 10% demo late (mostly online)
```

---

## 5. Action Space: Discretized FiGAR

### Joint Action-Duration Space

Flatten the FiGAR (action, duration) pair into a single discrete action space:

```
Base actions (6):
    0: Left
    1: Right
    2: Jump
    3: Jump+Left
    4: Jump+Right
    5: NOOP

Durations (8): 1, 2, 3, 4, 5, 6, 7, 8 frames

Joint actions: 6 x 8 = 48 discrete actions
    Action 0:  Left for 1 frame
    Action 1:  Left for 2 frames
    ...
    Action 7:  Left for 8 frames
    Action 8:  Right for 1 frame
    ...
    Action 47: NOOP for 8 frames
```

**Why cap at 8 frames:** The previous mean duration of ~19 frames caused catastrophic wrong-direction commitment (~63px backtracking per bad decision). Capping at 8 frames limits worst-case backtracking to ~27px and forces more frequent decision-making, improving credit assignment.

**Why flatten instead of factored heads:** SAC-Discrete computes Q(s, a) for all actions simultaneously. With a flat 48-action space, this is a single forward pass producing 48 Q-values -- simple, fast, and allows the Q-function to learn that "Jump+Right for 6 frames" has a completely different value than "Jump+Right for 1 frame" without any architectural coupling assumptions.

### Action Encoding/Decoding

```python
N_BASE_ACTIONS = 6
N_DURATIONS = 8
N_ACTIONS = N_BASE_ACTIONS * N_DURATIONS  # 48

def encode_action(base_action: int, duration: int) -> int:
    """Encode (base_action, duration) into flat action index."""
    assert 0 <= base_action < N_BASE_ACTIONS
    assert 1 <= duration <= N_DURATIONS
    return base_action * N_DURATIONS + (duration - 1)

def decode_action(action_idx: int) -> tuple[int, int]:
    """Decode flat action index into (base_action, duration)."""
    base_action = action_idx // N_DURATIONS
    duration = (action_idx % N_DURATIONS) + 1
    return base_action, duration
```

### Action Execution with Interrupts

```python
def execute_action(env, action_idx: int, gamma: float = 0.99):
    """
    Execute a macro-action: repeat base_action for duration frames.
    Interrupt on terminal or contact state change.
    
    Returns:
        next_obs: observation at the next decision point
        total_reward: gamma-discounted sum of primitive rewards
        done: whether episode ended
        info: dict with actual_duration, contact_state_changed, etc.
    """
    base_action, duration = decode_action(action_idx)
    total_reward = 0.0
    actual_duration = 0
    
    for frame in range(duration):
        obs, reward, done, info = env.step_primitive(base_action)
        total_reward += (gamma ** frame) * reward
        actual_duration += 1
        
        if done:
            break
        if info.get("contact_state_changed", False) and frame < duration - 1:
            break  # Interrupt: state change mid-action
    
    info["actual_duration"] = actual_duration
    return obs, total_reward, done, info
```

---

## 6. Reward Function

### Sparse-Only Design

With RFCL providing the learning gradient via curriculum structure, the reward function is purely sparse:

```python
def compute_reward(state, next_state, phase):
    """
    Minimal sparse reward. No distance-based shaping.
    The curriculum structure replaces dense reward shaping.
    """
    # Task completion
    if phase == 0 and next_state.switch_activated and not state.switch_activated:
        return R_SWITCH  # +1.0
    
    if phase == 1 and next_state.exit_reached:
        return R_EXIT    # +5.0
    
    # Death
    if next_state.is_dead:
        return R_DEATH   # -0.5
    
    # Time penalty (mild -- encourages shorter paths)
    return R_STEP        # -0.001
```

### Reward Constants

```
R_SWITCH =  1.0    # Collecting exit switch
R_EXIT   =  5.0    # Reaching exit door  
R_DEATH  = -0.5    # Death (any cause)
R_STEP   = -0.001  # Per-primitive-frame time penalty
```

### Why No Dense Shaping

A* distance is physically unreliable (established in prior analysis). HWM provides zero signal on 80-93% of steps. Velocity-alignment rewards penalize correct physics-based strategies. The RFCL curriculum provides the learning gradient that dense shaping was attempting to provide, but through initial state distribution rather than reward manipulation. This eliminates an entire class of reward engineering problems.

The time penalty (R_STEP = -0.001) provides mild pressure toward shorter paths. Over a 2000-frame episode, the cumulative penalty is -2.0, which is significant relative to R_SWITCH (1.0) but small relative to R_EXIT (5.0). This encourages efficiency without dominating the reward signal.

### Macro-Action Reward Accumulation

When executing a macro-action of actual duration d:

```
r_macro = sum_{i=0}^{d-1} gamma^i * r_primitive_i
```

This is the single scalar reward stored in the replay buffer for the transition.

## 7. Network Architecture

### Shared Feature Extractor

Reuse the existing multimodal extraction pattern, adapted for SAC:

```python
class FeatureExtractor(nn.Module):
    """
    Processes the 153-dim observation into a 256-dim feature vector.
    Separate instances for Q-networks and policy network.
    
    Changes from PPO version:
    - No dropout (SAC entropy regularization handles exploration)
    - LayerNorm added (critical for RLPD stability per Ball et al. 2023)
    - Reduced fusion dim: 384 -> 256 (sufficient for Q/policy heads)
    """
    def __init__(self):
        super().__init__()
        # Game state: 27 -> 128
        self.state_mlp = nn.Sequential(
            nn.Linear(27, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(),
        )
        # Reachability: 11 -> 64
        self.reach_mlp = nn.Sequential(
            nn.Linear(11, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(),
        )
        # Spatial context: 112 -> 128
        self.spatial_mlp = nn.Sequential(
            nn.Linear(112, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(),
        )
        # Mine SDF: 3 -> 16
        self.mine_mlp = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        )
        # Fusion: 336 -> 256
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64 + 128 + 16, 256),
            nn.LayerNorm(256), nn.ReLU(),
        )
    
    def forward(self, obs):
        """
        Args:
            obs: [batch, 153] flat tensor (game_state | reachability | spatial | mine_sdf)
        Returns:
            features: [batch, 256]
        """
        game_state = obs[..., :27]
        reachability = obs[..., 27:38]
        spatial = obs[..., 38:150]
        mine_sdf = obs[..., 150:153]
        
        s = self.state_mlp(game_state)
        r = self.reach_mlp(reachability)
        sp = self.spatial_mlp(spatial)
        m = self.mine_mlp(mine_sdf)
        
        return self.fusion(torch.cat([s, r, sp, m], dim=-1))
```

### Q-Networks (Twin)

```python
class QNetwork(nn.Module):
    """
    Discrete Q-network: observation -> Q-values for all 48 actions.
    Two independent copies for clipped double-Q.
    """
    def __init__(self, feature_dim=256, n_actions=48):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.q_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )
    
    def forward(self, obs):
        features = self.feature_extractor(obs)
        return self.q_head(features)  # [batch, 48]
```

### Policy Network

```python
class PolicyNetwork(nn.Module):
    """
    Discrete policy: observation -> action logits.
    Separate feature extractor from Q-networks (no shared parameters).
    """
    def __init__(self, feature_dim=256, n_actions=48):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )
    
    def forward(self, obs):
        features = self.feature_extractor(obs)
        return self.policy_head(features)  # [batch, 48] logits
    
    def get_action(self, obs, deterministic=False):
        logits = self.forward(obs)
        if deterministic:
            return logits.argmax(dim=-1)
        probs = F.softmax(logits, dim=-1)
        return Categorical(probs=probs).sample()
    
    def log_prob(self, obs):
        logits = self.forward(obs)
        return F.log_softmax(logits, dim=-1)  # [batch, 48]
```

### Why Separate Networks (Not Shared)

Shared feature extractors between policy and value networks create gradient interference (Andrychowicz et al., 2021). Separate extractors allow the Q-function to learn value estimation independently. Memory cost is ~3x (Q1, Q2, policy each have their own extractor), but at ~1M parameters each, total is ~3M parameters -- negligible.

### LayerNorm

LayerNorm is critical for RLPD stability (Ball et al. explicitly require it). It normalizes activations across the feature dimension, preventing Q-function divergence when demo and online data distributions differ.

---

## 8. Replay Buffer Design

### Structure

```python
class ReplayBuffer:
    """
    Standard circular replay buffer with uniform sampling.
    Separate instances for online data and demonstration data.
    """
    def __init__(self, capacity=1_000_000, obs_dim=153):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0
    
    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=torch.FloatTensor(self.obs[idxs]),
            actions=torch.LongTensor(self.actions[idxs]),
            rewards=torch.FloatTensor(self.rewards[idxs]),
            next_obs=torch.FloatTensor(self.next_obs[idxs]),
            dones=torch.FloatTensor(self.dones[idxs]),
        )
```

### Buffer Sizing

```
Online buffer:    1,000,000 transitions (~1.24 GB)
Demo buffer:      Variable per level, never overwritten
                  Typical: 5,000-50,000 transitions per level (~6-60 MB)

Per transition:   153*4 (obs) + 153*4 (next_obs) + 8 (action) + 4 (reward) + 4 (done)
                  = 1,240 bytes

Total memory:     ~1.3 GB (well within GPU memory budget)
```

### Macro-Action Transitions

Each macro-action produces one transition in the replay buffer:

```
obs:      Observation at decision point (153 dims)
action:   Joint (base_action, duration) index [0-47]
reward:   Discounted sum of primitive rewards over actual executed duration
next_obs: Observation at next decision point (after duration or interrupt)
done:     Whether episode terminated during this macro-action
```

This stores decision-level transitions, not frame-level. With 8-frame max duration, this is ~8x more compact than frame-level storage, and each transition carries more semantic meaning.

---

## 9. Two-Phase Task Structure

### Phase 0: Spawn -> Switch

The agent must navigate from its initial position to the exit switch. RFCL operates on demo states from the pre-switch portion of the demonstration.

### Phase 1: Switch -> Exit

After collecting the switch, the agent must navigate to the exit door. RFCL operates on demo states from the post-switch portion.

### Joint Phase Training (Recommended)

Include the `switch_activated` flag in the observation (already present as reachability feature index 12). Train both phases simultaneously, with the RFCL curriculum managing state resets for both phases. The Q-function learns phase-dependent values from the flag.

This is preferred over sequential training because:
- SAC's replay buffer naturally mixes Phase 0 and Phase 1 transitions
- No catastrophic forgetting risk (both phases trained continuously)
- The Q-function conditions on the phase flag, learning distinct value landscapes for each phase

### Implementation

```python
class TwoPhaseRFCL:
    """Manages RFCL for both phases of a level."""
    
    def __init__(self, demo_snapshots, switch_frame_idx, spawn_state):
        # Split demo at switch collection
        pre_switch = [s for s in demo_snapshots if s.frame_index < switch_frame_idx]
        post_switch = [s for s in demo_snapshots if s.frame_index >= switch_frame_idx]
        
        self.phase0_curriculum = LevelCurriculum(
            level_id="phase0", demos=[pre_switch], spawn_state=spawn_state
        )
        self.phase1_curriculum = LevelCurriculum(
            level_id="phase1", demos=[post_switch],
            spawn_state=pre_switch[-1] if pre_switch else spawn_state
        )
    
    def sample_episode_config(self):
        """
        Returns (reset_state, target_phase) for the next episode.
        Prioritizes the less-complete phase.
        """
        p0_done = self.phase0_curriculum.is_complete
        p1_done = self.phase1_curriculum.is_complete
        
        if not p0_done and not p1_done:
            phase_weight = 0.6  # Slight bias toward Phase 0 (must solve first)
        elif not p0_done:
            phase_weight = 0.8
        elif not p1_done:
            phase_weight = 0.2
        else:
            phase_weight = 0.5  # Both solved: maintain both
        
        if np.random.random() < phase_weight:
            return self.phase0_curriculum.sample_reset_state(), 0
        else:
            return self.phase1_curriculum.sample_reset_state(), 1
    
    def report_result(self, phase, snapshot_idx, success):
        if phase == 0:
            self.phase0_curriculum.report_result(snapshot_idx, success)
        else:
            self.phase1_curriculum.report_result(snapshot_idx, success)
```

### Full-Level Evaluation

Periodically run evaluation episodes from the actual spawn point through both phases (spawn -> switch -> exit) to measure end-to-end success rate. This is the true metric, separate from the curriculum's per-phase success tracking.

---

## 10. Training Loop

### Core Training Algorithm

```python
def train(levels, total_steps=100_000_000):
    """
    Main SAC-Discrete + RFCL training loop.
    
    Key differences from PPO:
    - Off-policy: replay buffer, not rollout buffer
    - High UTD ratio: multiple gradient steps per environment step
    - Demo mixing: permanent demo data in replay
    - Curriculum resets: RFCL manages initial state distribution
    """
    # --- Initialize networks ---
    policy = PolicyNetwork(n_actions=N_ACTIONS).cuda()
    Q1 = QNetwork(n_actions=N_ACTIONS).cuda()
    Q2 = QNetwork(n_actions=N_ACTIONS).cuda()
    Q1_target = copy.deepcopy(Q1)
    Q2_target = copy.deepcopy(Q2)
    
    # Freeze target network parameters
    for p in Q1_target.parameters(): p.requires_grad = False
    for p in Q2_target.parameters(): p.requires_grad = False
    
    # Learnable temperature
    log_alpha = torch.zeros(1, requires_grad=True, device='cuda')
    target_entropy = -np.log(1.0 / N_ACTIONS) * TARGET_ENTROPY_RATIO
    
    # --- Optimizers ---
    q_optimizer = Adam(list(Q1.parameters()) + list(Q2.parameters()), lr=Q_LR)
    policy_optimizer = Adam(policy.parameters(), lr=POLICY_LR)
    alpha_optimizer = Adam([log_alpha], lr=ALPHA_LR)
    
    # --- Replay buffers ---
    online_buffer = ReplayBuffer(capacity=BUFFER_SIZE)
    demo_buffers = {}
    for level in levels:
        demo_buffers[level.id] = ReplayBuffer(capacity=200_000)
        for demo in level.demos:
            transitions = convert_demo_to_transitions(demo, level.env)
            for t in transitions:
                demo_buffers[level.id].add(*t)
    
    # --- Curricula ---
    curricula = {}
    for level in levels:
        snapshots = extract_demo_snapshots(level.demos[0])
        curricula[level.id] = TwoPhaseRFCL(
            snapshots, level.switch_frame, level.spawn_state
        )
    
    # --- Parallel environments ---
    envs = VectorizedEnv(levels, n_envs=N_ENVS)
    
    # --- Initial exploration (random policy to seed buffer) ---
    obs = envs.reset_with_curricula(curricula)
    for _ in range(INITIAL_RANDOM_STEPS):
        actions = np.random.randint(0, N_ACTIONS, size=N_ENVS)
        next_obs, rewards, dones, infos = envs.step_macro(actions)
        for i in range(N_ENVS):
            online_buffer.add(obs[i], actions[i], rewards[i], 
                            next_obs[i], dones[i])
        obs = next_obs
    
    # --- Main loop ---
    global_step = 0
    while global_step < total_steps:
        # === Collect experience ===
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).cuda()
            actions = policy.get_action(obs_tensor).cpu().numpy()
        
        next_obs, rewards, dones, infos = envs.step_macro(actions)
        
        for i in range(N_ENVS):
            online_buffer.add(obs[i], actions[i], rewards[i], 
                            next_obs[i], dones[i])
            if dones[i]:
                level_id = infos[i]['level_id']
                curricula[level_id].report_result(
                    infos[i]['phase'], 
                    infos[i].get('snapshot_idx', -1),
                    infos[i]['success']
                )
                # Reset with curriculum-determined initial state
                reset_state, phase = curricula[level_id].sample_episode_config()
                envs.reset_single(i, reset_state, phase)
        
        obs = next_obs
        global_step += N_ENVS
        
        # === Gradient updates (high UTD ratio) ===
        if online_buffer.size >= MIN_BUFFER_SIZE:
            for _ in range(UTD_RATIO):
                # Sample mixed batch
                demo_ratio = get_demo_ratio(global_step)
                n_demo = int(BATCH_SIZE * demo_ratio)
                n_online = BATCH_SIZE - n_demo
                
                # Aggregate demos from all levels
                demo_batch = sample_from_all_demo_buffers(
                    demo_buffers, n_demo)
                online_batch = online_buffer.sample(n_online)
                batch = merge_batches(demo_batch, online_batch)
                
                # Move to GPU
                batch = {k: v.cuda() for k, v in batch.items()}
                
                # --- Q-function update ---
                with torch.no_grad():
                    alpha = log_alpha.exp()
                    next_log_pi = policy.log_prob(batch['next_obs'])
                    next_pi = next_log_pi.exp()
                    
                    q1_next = Q1_target(batch['next_obs'])
                    q2_next = Q2_target(batch['next_obs'])
                    q_next_min = torch.min(q1_next, q2_next)
                    
                    # Soft value of next state
                    v_next = (next_pi * (q_next_min - alpha * next_log_pi)).sum(dim=1)
                    q_target = batch['rewards'] + GAMMA * (1 - batch['dones']) * v_next
                
                q1_pred = Q1(batch['obs']).gather(1, batch['actions'].unsqueeze(1)).squeeze(1)
                q2_pred = Q2(batch['obs']).gather(1, batch['actions'].unsqueeze(1)).squeeze(1)
                
                q_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
                
                q_optimizer.zero_grad()
                q_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(Q1.parameters()) + list(Q2.parameters()), MAX_GRAD_NORM)
                q_optimizer.step()
                
                # --- Policy update ---
                log_pi = policy.log_prob(batch['obs'])
                pi = log_pi.exp()
                
                with torch.no_grad():
                    q1_vals = Q1(batch['obs'])
                    q2_vals = Q2(batch['obs'])
                    q_min = torch.min(q1_vals, q2_vals)
                
                policy_loss = (pi * (alpha.detach() * log_pi - q_min)).sum(dim=1).mean()
                
                policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                policy_optimizer.step()
                
                # --- Temperature update ---
                entropy = -(pi * log_pi).sum(dim=1).mean()
                alpha_loss = -(log_alpha * (entropy - target_entropy).detach())
                
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()
                
                # --- Target network update ---
                with torch.no_grad():
                    for p, p_targ in zip(Q1.parameters(), Q1_target.parameters()):
                        p_targ.data.mul_(1 - TAU)
                        p_targ.data.add_(TAU * p.data)
                    for p, p_targ in zip(Q2.parameters(), Q2_target.parameters()):
                        p_targ.data.mul_(1 - TAU)
                        p_targ.data.add_(TAU * p.data)
        
        # === Logging (every 10K steps) ===
        if global_step % 10_000 == 0:
            log_metrics(global_step, q_loss, policy_loss, alpha, entropy,
                       curricula, online_buffer)
```

---

## 11. State Reset Interface

### Required Environment API

The simulation must support resetting the ninja to an arbitrary state:

```python
class NppEnvironment:
    def reset_to_state(self, snapshot: DemoSnapshot, phase: int):
        """
        Reset the simulation to a specific ninja state.
        
        This must faithfully restore:
        1. Ninja position (x, y) -- placed at exact pixel coordinates
        2. Ninja velocity (vx, vy)
        3. Movement state (immobile/running/jumping/falling/wall_sliding)
        4. Jump duration remaining
        5. Surface contact state (floor/wall normals)
        6. Physics parameters (gravity, drag, friction matching the state)
        7. Entity states:
           - switch_activated flag (for phase management)
           - Toggle mine states (initial state for the level)
           - Door states (locked/open based on switch)
        8. Frame counter (set based on remaining time budget)
        
        Args:
            snapshot: DemoSnapshot with full state for restoration
            phase: 0 or 1 (determines switch/door states)
        
        Returns:
            obs: 153-dim observation from the restored state
        """
        # Reset simulation to level's initial state
        self.sim.reset_level()
        
        # Override ninja state
        self.sim.ninja.xpos = snapshot.position_x
        self.sim.ninja.ypos = snapshot.position_y
        self.sim.ninja.xspeed = snapshot.velocity_x
        self.sim.ninja.yspeed = snapshot.velocity_y
        self.sim.ninja.state = snapshot.movement_state
        self.sim.ninja.jump_duration = snapshot.jump_duration_remaining
        self.sim.ninja.applied_gravity = snapshot.applied_gravity
        self.sim.ninja.applied_drag = snapshot.applied_drag
        self.sim.ninja.applied_friction = snapshot.applied_friction
        
        # Set floor/wall contact state
        if snapshot.floor_contact:
            self.sim.ninja.floor_count = 1
            self.sim.ninja.floor_normal_x = snapshot.floor_normal_x
            self.sim.ninja.floor_normal_y = snapshot.floor_normal_y
        if snapshot.wall_contact:
            self.sim.ninja.wall_count = 1
            self.sim.ninja.wall_normal = snapshot.wall_normal_x
        
        # Set entity states for the target phase
        if phase == 1:
            self.sim.activate_exit_switch()
        
        # Compute and return observation
        return self._compute_observation()
    
    def step_primitive(self, base_action: int):
        """
        Execute one physics frame with the given base action.
        
        Args:
            base_action: 0-5 (Left, Right, Jump, Jump+Left, Jump+Right, NOOP)
        
        Returns:
            obs, reward, done, info
        """
        horizontal, jump = ACTION_MAP[base_action]
        self.sim.ninja.think(horizontal, jump)
        self.sim.step()
        
        obs = self._compute_observation()
        reward = self._compute_reward()
        done = self.sim.is_terminal()
        info = {
            'success': self.sim.ninja.state == 8,  # Celebrating
            'contact_state_changed': self._check_contact_change(),
            'phase': 1 if self.sim.switch_activated else 0,
        }
        return obs, reward, done, info
```

### State Reset Validation

After resetting to a snapshot, verify the state is physically valid:

```python
def validate_reset(env, snapshot):
    """Verify that the reset state is playable."""
    obs = env.reset_to_state(snapshot, phase=0)
    
    # Agent should not be dead immediately
    assert not env.sim.ninja.is_dead, "Reset to dead state"
    
    # Agent should not be inside a wall
    assert not env.sim.ninja.is_crushed, "Reset inside geometry"
    
    # Position should be within level bounds
    assert 24 <= snapshot.position_x <= 1032, "X out of bounds"
    assert 24 <= snapshot.position_y <= 576, "Y out of bounds"
    
    # Run one physics step to check stability
    obs2, _, done, _ = env.step_primitive(5)  # NOOP
    assert not done, "State immediately terminal after NOOP"
    
    return True
```

### Noise Bounds

Position noise (+-2px) and velocity noise (+-0.15 px/frame) must not place the ninja inside solid geometry. After adding noise, run a collision check and clamp to the nearest valid position:

```python
def safe_add_noise(snapshot, env):
    """Add noise to snapshot, ensuring physical validity."""
    noisy = add_noise(snapshot)
    
    # Quick collision check at noisy position
    if env.sim.check_collision(noisy.position_x, noisy.position_y, NINJA_RADIUS):
        # Fallback: use original position (no noise)
        noisy.position_x = snapshot.position_x
        noisy.position_y = snapshot.position_y
    
    return noisy
```

## 12. Hyperparameters

### SAC-Discrete Core

```
Parameter                     Value       Notes
----------------------------- ----------- ------------------------------------
GAMMA                         0.99        Discount factor
TAU                           0.005       Target network Polyak rate
Q_LR                          3e-4        Q-network learning rate
POLICY_LR                     3e-4        Policy learning rate
ALPHA_LR                      3e-4        Temperature learning rate
TARGET_ENTROPY_RATIO          0.5         target = -log(1/48) * 0.5 = 1.93
BATCH_SIZE                    256         Per gradient step
UTD_RATIO                     4           Gradient steps per env step
                                          (start 4, increase to 8 if stable)
MAX_GRAD_NORM                 1.0         Gradient clipping
N_ACTIONS                     48          6 base * 8 durations
```

### Replay Buffer

```
Parameter                     Value       Notes
----------------------------- ----------- ------------------------------------
BUFFER_SIZE                   1_000_000   Online buffer capacity
MIN_BUFFER_SIZE               10_000      Min transitions before training
INITIAL_RANDOM_STEPS          5_000       Random exploration steps (* N_ENVS)
```

### RFCL Curriculum

```
Parameter                     Value       Notes
----------------------------- ----------- ------------------------------------
SNAPSHOT_INTERVAL             30          Frames between demo snapshots
POSITION_NOISE_STD            2.0         Pixels, position perturbation
VELOCITY_NOISE_STD            0.15        px/frame, velocity perturbation
FRONTIER_ADVANCE_THRESHOLD    0.5         Success rate to expand frontier
FRONTIER_ADVANCE_STEP         0.05        Fraction of demo per advance
FORWARD_INITIAL_SPAWN_RATIO   0.2         Initial spawn-vs-demo mix
FORWARD_ADVANCE_THRESHOLD     0.3         Spawn success rate to increase ratio
FORWARD_ADVANCE_STEP          0.1         Spawn ratio increment per advance
```

### Demonstration Mixing

```
Parameter                     Value       Notes
----------------------------- ----------- ------------------------------------
DEMO_RATIO_EARLY              0.50        First 1M steps
DEMO_RATIO_MID                0.25        1M - 10M steps
DEMO_RATIO_LATE               0.10        After 10M steps
```

### Environment

```
Parameter                     Value       Notes
----------------------------- ----------- ------------------------------------
N_ENVS                        256         Parallel environments
MAX_EPISODE_FRAMES            5000        Reduced from 20000 during curriculum
                                          (increased as curriculum progresses)
```

### Hyperparameter Sensitivity Notes

**Most sensitive (tune carefully):**
- `UTD_RATIO`: Too high causes Q-function overfitting to current buffer. Too low wastes data. Start at 4, monitor Q-loss stability, increase to 8 if loss is stable.
- `TARGET_ENTROPY_RATIO`: Controls exploration. 0.5 means target entropy is half the maximum (uniform distribution). Lower = less exploration. Sweep [0.3, 0.5, 0.7] if behavior is too random or too deterministic.
- `FRONTIER_ADVANCE_THRESHOLD`: Controls how fast the reverse curriculum expands. 0.5 is conservative (need 50% success before advancing). Lower values (0.3) advance faster but risk advancing past unsolved segments.

**Least sensitive (defaults usually fine):**
- `GAMMA`, `TAU`, learning rates: Standard SAC values work well across domains.
- `BATCH_SIZE`: 256 is standard. 512 slightly more stable but slower per step.
- `POSITION_NOISE_STD`: Any value in [1.0, 4.0] works. Prevents overfitting to exact demo positions.

---

## 13. Curriculum Progression Logic

### Multi-Level Training

With 50+ levels, not all levels train simultaneously. Use a level selection strategy:

```python
class LevelScheduler:
    """
    Selects which levels to train on across 256 parallel environments.
    Prioritizes levels where the curriculum is actively progressing.
    """
    def __init__(self, levels, n_envs=256):
        self.levels = levels
        self.n_envs = n_envs
        self.level_states = {
            lvl.id: "active" for lvl in levels
            # States: "active" (training), "solved" (completed), "stalled" (no progress)
        }
        self.stall_counters = defaultdict(int)
    
    def assign_environments(self):
        """
        Assign levels to environments.
        Returns: dict mapping env_index -> level_id
        """
        active = [l for l in self.levels if self.level_states[l.id] == "active"]
        solved = [l for l in self.levels if self.level_states[l.id] == "solved"]
        
        assignments = {}
        n_active = min(len(active), self.n_envs)
        
        # 80% of envs to active levels (proportional to curriculum progress need)
        n_active_envs = int(self.n_envs * 0.8)
        for i in range(n_active_envs):
            lvl = active[i % len(active)]
            assignments[i] = lvl.id
        
        # 20% of envs to solved levels (prevent forgetting)
        n_maintenance_envs = self.n_envs - n_active_envs
        for i in range(n_maintenance_envs):
            if solved:
                lvl = solved[i % len(solved)]
                assignments[n_active_envs + i] = lvl.id
            else:
                lvl = active[i % len(active)]
                assignments[n_active_envs + i] = lvl.id
        
        return assignments
    
    def update_level_status(self, level_id, curriculum):
        """Check if a level is solved or stalled."""
        if curriculum.phase0_curriculum.is_complete and \
           curriculum.phase1_curriculum.is_complete:
            self.level_states[level_id] = "solved"
            return
        
        # Stall detection: no frontier advance in 500K steps
        # (tracked externally by comparing frontier_index over time)
        if self.stall_counters[level_id] > 500_000:
            self.level_states[level_id] = "stalled"
```

### Episode Time Limit Scaling

During the reverse curriculum, the agent starts near the goal and only needs a short time to reach it. Long time limits waste environment steps on futile wandering:

```python
def get_episode_time_limit(curriculum_stage, base_limit=5000):
    """
    Scale episode length based on curriculum progress.
    Near-goal starts need fewer frames; full-level starts need more.
    """
    if curriculum_stage == "reverse":
        # Distance from frontier to goal determines time budget
        # Rough heuristic: 50 frames per snapshot of distance
        return min(base_limit, max(300, frontier_distance_snapshots * 50))
    elif curriculum_stage == "forward":
        return base_limit
    else:
        return base_limit
```

### Stall Recovery

If a level stalls (no curriculum advancement for 500K+ steps):

1. **Increase noise.** Double POSITION_NOISE_STD and VELOCITY_NOISE_STD to explore more diverse initial conditions around demo states.
2. **Increase demo ratio.** Temporarily boost to 0.5 for this level's transitions.
3. **Slow the frontier.** Increase FRONTIER_ADVANCE_THRESHOLD to 0.7 (require stronger mastery before advancing).
4. **Log for manual inspection.** The level may require demo trajectories that better cover the difficult segment.

---

## 14. Observation Space Changes

### Retained (No Changes)

The 153-dimensional observation space from OBSERVATION_SPACE_DOCUMENTATION.md v2.0 is retained in full:

```
Game State:           27 dims (position, velocity, contact, physics state)
Reachability:         11 dims (A* distances, directions, phase flag)
Spatial Context:     112 dims (8x8 tile grid + 8-nearest mines)
Mine SDF:              3 dims (global mine danger gradient)
Total:               153 dims
```

### A* Features: Observation Only, Not Reward

The A* reachability features (path distances, next-hop direction, detour flag) remain as observation features for the policy and Q-networks. The critical distinction:

- **Old approach:** A* distance drives the reward (HWM/PBRS). The reward is wrong when A* is wrong.
- **New approach:** A* distance is an observation. The Q-function learns from experience which observations are reliable. If A* direction misleads on certain levels, the Q-function learns to weight other features (local tile grid, velocity, contact state) more heavily. The reward is independent of A*.

This is the same principle validated in PGF-MAPPO (2024): A* features in observations achieve 60% success; adding A*-based PBRS achieves 98%. The features are useful when the agent has learned to interpret them; the reward should be independent of their accuracy.

### Optional Addition: Demo Progress Feature (1 dim)

If using multiple demos per level, add a scalar observation indicating the agent's proximity to the nearest demo trajectory state:

```
demo_proximity: min distance to nearest demo snapshot / max_level_distance
Range: [0, 1] (0 = on a demo trajectory, 1 = far from any demo)
```

This provides a soft hint about whether the agent is in well-understood territory. Not critical but potentially helpful for the Q-function to assess state quality. This would increase the observation to 154 dims.

---

## 15. Verification and Diagnostics

### Correctness Tests

**State reset fidelity:** Reset to 100 random demo snapshots per level. For each, run a NOOP action and verify the ninja doesn't die, doesn't clip through geometry, and the observation is within expected ranges.

**Demo transition consistency:** For each demo, verify that executing the recorded actions from the recorded states produces states matching the next frame in the demo (within floating-point tolerance). This confirms the demo data is compatible with the simulation version.

**Q-function sanity:** After 100K steps of training, verify:
- Q-values for demo states near the goal should be higher than Q-values for demo states far from the goal.
- Q-values should be negative for states near hazards (mines, edges) and positive for states near objectives.
- Q-values should not diverge (stay within [-50, 50] for reasonable reward scales).

**Policy entropy:** Monitor the policy entropy. It should start near maximum (log(48) = 3.87) and decrease as training progresses, but should not collapse to 0 (the temperature alpha prevents this).

### Diagnostic Metrics (Log Every 10K Steps)

```
# --- Per-level curriculum ---
level_{id}_phase0_frontier_index      # How far back in demo (lower = more progress)
level_{id}_phase0_frontier_success    # Success rate at frontier
level_{id}_phase1_frontier_index
level_{id}_phase1_frontier_success
level_{id}_curriculum_stage           # "reverse" or "forward"
level_{id}_spawn_success_rate         # Success from actual spawn (ultimate metric)

# --- SAC health ---
q1_loss                               # Should decrease then stabilize
q2_loss
policy_loss                           # Should stabilize (not necessarily decrease)
alpha                                 # Learned temperature (should decrease over training)
policy_entropy                        # Should decrease but stay > 0
mean_q_value                          # Should increase as agent improves
max_q_value
min_q_value

# --- Buffer health ---
online_buffer_size                    # Should grow to capacity
demo_ratio_current                    # Should decrease per schedule
mean_episode_reward_online            # From agent's own experience
mean_episode_length_online

# --- Action distribution ---
action_base_distribution              # Fraction per base action [6 values]
action_duration_distribution          # Fraction per duration [8 values]
mean_actual_duration                  # After interrupts

# --- Per-level success (evaluation, not training) ---
eval_spawn_to_switch_rate             # From actual spawn
eval_spawn_to_exit_rate               # Full level completion
eval_mean_completion_frames           # Speed of successful episodes
```

### Visualization

**Curriculum progression plot:** For each level, plot frontier_index over training steps. Should show monotonic decrease toward 0 (reverse curriculum expanding to demo start), then transition to forward curriculum.

**Q-value heatmap:** Render Q-values (max over actions) as a 2D heatmap over (x, y) positions for a level. Should show high values near objectives and along viable paths, low values near hazards and dead ends.

**Policy action heatmap:** For each (x, y) position, render the most probable base action as a colored arrow. Should show coherent navigation patterns (arrows pointing toward objectives through valid paths).

**Demo vs. agent trajectory overlay:** On the level geometry, overlay the demo trajectory (blue) and the agent's trajectory from the same starting state (red). As training progresses, the agent trajectory should converge toward (but not necessarily match) the demo trajectory.

---

## 16. Computational Budget

### Per-Step Cost Breakdown

```
Component                       | Per env step | Notes
--------------------------------|-------------|------
Environment step (256 envs)     | ~200 us    | Physics simulation + obs computation
Policy forward pass             | ~30 us     | [256, 153] -> [256, 48]
Q-function updates (UTD=4)      |            |
  Q1+Q2 forward (batch=256)     | ~60 us     | x4 = 240 us
  Q1+Q2 target forward          | ~60 us     | x4 = 240 us
  Policy forward (for Q update) | ~30 us     | x4 = 120 us
  Backward passes               | ~200 us    | x4 = 800 us
  Total Q updates               | ~1400 us   |
Policy update (UTD=4)           |            |
  Forward + backward            | ~150 us    | x4 = 600 us
Alpha update                    | ~5 us      | x4 = 20 us
Target network Polyak           | ~20 us     | x4 = 80 us
Replay buffer sample            | ~30 us     | x4 = 120 us
Curriculum management           | ~5 us      | Negligible
--------------------------------|-------------|------
Total per 256-env step          | ~2650 us   | ~2.65 ms
```

### Throughput Estimate

```
Per 256-env step:     ~2.65 ms
Steps per second:     ~375 (256-env steps)
Transitions/second:   ~96,000 (375 * 256)
Decision points/sec:  ~96,000 (one decision per macro-action stored)

Comparison to PPO (10,000 iter/s * 256 envs):
    PPO: ~2,560,000 frame-steps/s (but most are wasted within rollouts)
    SAC: ~96,000 decision-steps/s (but each is reused UTD=4 times)
    Effective gradient steps: SAC = 384K/s vs PPO ~10K/s
```

SAC is slower in raw environment throughput but much higher in gradient steps per environment interaction, which is the metric that matters for sample efficiency.

### Scaling

- **Increase UTD ratio** to 8 or 16 if GPU has spare compute. This doubles/quadruples gradient steps without additional environment interaction.
- **Reduce N_ENVS** to 128 if GPU memory is tight. SAC is less dependent on parallel envs than PPO.
- **Increase BATCH_SIZE** to 512 or 1024 if training is unstable. Larger batches smooth gradient estimates.

### Memory Budget

```
Networks (Q1, Q2, Q1_target, Q2_target, policy):  ~15 MB total
Online replay buffer (1M transitions):             ~1.24 GB
Demo buffers (50 levels, ~10K transitions each):   ~0.6 GB
Environment state (256 parallel):                  ~50 MB
Optimizer states:                                  ~30 MB
------------------------------------------------------
Total GPU memory:                                  ~2.0 GB
```

Well within a single GPU's capacity.

---

## 17. Migration from PPO

### What to Keep

1. **Observation pipeline (153 dims).** The entire feature computation (game state extraction, reachability features, spatial context, mine SDF) is retained unchanged. The `compute_observation()` function is shared between PPO and SAC.

2. **Environment wrapper.** The base environment (physics simulation, entity management, observation computation) is unchanged. SAC replaces the training loop, not the environment.

3. **A* graph and reachability computation.** Still used for observation features. No longer used for reward.

4. **Level loading and curriculum levels.** The 50+ level set and any level metadata is retained.

5. **Evaluation infrastructure.** Success rate tracking, episode length metrics, per-level statistics.

### What to Replace

1. **PPO training loop** -> SAC-Discrete training loop (Section 10).
2. **Rollout buffer** -> Replay buffer (Section 8).
3. **Policy + value network (shared extractor)** -> Separate policy + twin Q-networks (Section 7).
4. **HWM reward shaping** -> Sparse rewards only (Section 6).
5. **RND exploration** -> SAC entropy regularization (automatic, no extra networks).
6. **FiGAR separate action+duration heads** -> Flat 48-action space (Section 5).
7. **Distance-based curriculum stages** -> RFCL reverse-forward curriculum (Section 3).
8. **Entropy coefficient schedule** -> Learned temperature alpha (automatic).

### What to Add

1. **State reset interface** (Section 11). The simulation must support `reset_to_state()`.
2. **Demonstration extraction pipeline.** Convert replay files to DemoSnapshot lists.
3. **Demo replay buffer.** Permanent storage for demonstration transitions.
4. **RFCL curriculum managers.** Per-level, per-phase reverse-forward curricula.
5. **Level scheduler.** Multi-level training assignment across 256 environments.

### Migration Priority

```
Priority 1 (critical path):
    - Implement state reset interface (enables RFCL)
    - Implement SAC-Discrete core (Q-networks, policy, temperature)
    - Implement replay buffer with demo mixing
    - Test on 1 level end-to-end

Priority 2 (scaling):
    - Implement RFCL curriculum managers
    - Implement level scheduler for multi-level training
    - Demo extraction pipeline for all 50+ levels

Priority 3 (optimization):
    - Tune UTD ratio, batch size, target entropy
    - Add curriculum progression logging
    - Stall detection and recovery
```

### Validation Milestones

**Milestone 1 (1 level, no curriculum):** Train SAC-Discrete on a single easy level, resetting always from spawn, with demo mixing. Target: match or exceed PPO's best success rate. This validates the algorithm works at all.

**Milestone 2 (1 level, with RFCL):** Add reverse-forward curriculum on the same level. Target: higher success rate than Milestone 1, and faster convergence. This validates RFCL provides meaningful learning gradient.

**Milestone 3 (5 levels, with RFCL):** Scale to 5 levels of varying difficulty. Target: all 5 levels reach forward curriculum stage. This validates multi-level training.

**Milestone 4 (50+ levels):** Full curriculum. Target: >50% of levels reach spawn-to-exit completion. This is the production goal.

---

## Appendix A: Comparison with Previous Approach

| Property | PPO + HWM + RND (v1) | SAC-Discrete + RFCL (v2) |
|---|---|---|
| Algorithm | On-policy (PPO) | Off-policy (SAC) |
| Data reuse | 1x (discarded after update) | UTD * buffer_lifetime |
| Demo leverage | None (or IQL escalation) | Permanent replay (RLPD) |
| Distance metric in reward | A* HWM (80-93% zero) | None (sparse only) |
| Dense reward signal source | HWM + RND | Curriculum structure |
| Exploration mechanism | RND (extra networks) + entropy schedule | SAC entropy (automatic, no extra networks) |
| Action space | Factored (6 actions * continuous duration) | Flat discrete (48 actions) |
| Max action duration | ~19 frames (learned, too long) | 8 frames (hard cap) |
| Curriculum type | Distance-based stages from spawn | Reverse-forward along demos |
| Physics validity | A* paths often invalid | Demos always valid |
| Per-level requirements | A* graph | 1+ demo + state reset |
| Networks | Policy + Value (shared extractor) | Policy + 2xQ + 2xQ_target (separate) |
| Memory | ~33 MB (rollout buffer) | ~2 GB (replay buffer) |
| Implementation complexity | Moderate | Moderate-high |

### Key Advantage

The v2 approach eliminates the dependency on A* distance as a reward signal. The only role of A* is as an observation feature, where its inaccuracies can be learned around by the Q-function. The learning gradient comes from demonstration-guided curriculum (physics-valid by construction) and off-policy replay (sample-efficient).

---

## Appendix B: Levels Without Demonstrations

For levels where no demonstration is available, two fallback strategies exist:

### Fallback 1: Spawn-Only Training with Demo-Free SAC

Train SAC from the level spawn with only sparse rewards and demo mixing from other levels (transfer learning). The Q-function may generalize navigation strategies learned on demo-equipped levels to similar geometry. Expected to work for levels with geometry similar to demo-equipped levels, and fail for levels requiring novel strategies.

### Fallback 2: Progressive Random Curriculum

Without demos, generate initial states by running a random policy and collecting states that achieve partial progress:

```python
def generate_random_curriculum(level, n_random_episodes=10000):
    """
    Run random policy, collect farthest-reached states.
    Use as a crude reverse curriculum.
    """
    reached_states = []
    for _ in range(n_random_episodes):
        obs = level.env.reset()
        trajectory = [level.env.get_state()]
        for _ in range(1000):
            action = np.random.randint(0, N_ACTIONS)
            obs, _, done, info = level.env.step_macro(action)
            trajectory.append(level.env.get_state())
            if done:
                break
        if info.get('success'):
            reached_states.extend(trajectory)
        else:
            # Keep the farthest-reached state as a curriculum anchor
            reached_states.append(trajectory[-1])
    
    # Sort by proximity to goal, use as crude demo snapshots
    reached_states.sort(key=lambda s: distance_to_goal(s))
    return reached_states[:100]  # Keep best 100 as curriculum
```

This is much weaker than demo-guided RFCL but provides some curriculum structure. The recommendation for production use is to ensure at least 1 human demo per level.

---

## Appendix C: References

- **SAC-Discrete:** Christodoulou, P. (2019). "Soft Actor-Critic for Discrete Action Settings." arXiv:1910.07207.
- **RFCL:** Tao, S., Shukla, A., Chan, T.K., Su, H. (2024). "Reverse Forward Curriculum Learning for Extreme Sample and Demonstration Efficiency in RL." ICLR 2024.
- **RLPD:** Ball, P.J., Smith, L., Kostrikov, I., Levine, S. (2023). "Efficient Online Reinforcement Learning with Offline Data." ICML 2023.
- **Reverse Curriculum:** Florensa, C., Held, D., Geng, X., Abbeel, P. (2017). "Reverse Curriculum Generation for Reinforcement Learning." CoRL 2017.
- **Observation Space:** OBSERVATION_SPACE_DOCUMENTATION.md v2.0.
- **Physics:** sim_mechanics_doc.md.
- **Previous Reward Pipeline:** REWARD_PIPELINE_SPEC.md v1.0 (superseded).
