---
name: RL Training Efficiency for Long-Distance Physics-Based Exploration
overview: ""
todos:
  - id: 2d477410-ca65-4ca1-96fa-da18e2a98bdd
    content: Implement RND module for intrinsic motivation (Priority 1A)
    status: pending
  - id: 883ebf5a-3ace-4674-9989-6b811e26c896
    content: Adjust curriculum thresholds to unblock progression (Priority 1C)
    status: pending
  - id: d255f4a9-5244-4405-a96d-54558af36300
    content: Implement Hindsight Experience Replay (Priority 1B)
    status: pending
  - id: e6896157-cb6d-45fe-a1aa-bc8aaa69b333
    content: Add asymmetric actor-critic with privileged information (Priority 2A)
    status: pending
  - id: fa4194d9-4c6c-404a-acab-ba74de208409
    content: Implement Go-Explore checkpointing system (Priority 2B)
    status: pending
  - id: babf8b15-1d51-41fc-b955-765ac30b0762
    content: Add demonstration replay mixing to training (Priority 2C)
    status: pending
  - id: 2744efb0-6616-4e19-a0a3-6db54df383b4
    content: Run Phase 1 training and validate 20-30% success rate
    status: pending
---

# RL Training Efficiency for Long-Distance Physics-Based Exploration

## Problem Context: Why <10% Success After 2M Steps

Your levels have **critical characteristics** that make this a hard exploration problem:

- **Long distances**: 200-1000px from spawn to switch/exit
- **Complex physics**: Platformer mechanics (jumping, wall-jumping, momentum)
- **Sparse rewards**: Only 2 goals per level
- **Large state space**: 1056×600px level = ~660,000 pixel positions

**Key Insight**: With random exploration, probability of reaching 1000px goal through complex physics = effectively zero. Your agent needs:

1. **Sustained exploration** (RND to explore 50-200 step chains)
2. **Checkpointing** (Go-Explore to build on partial progress)
3. **Physics primitives** (Demonstrations to learn jumping/movement)
4. **Sample efficiency** (HER to learn from 90% failed episodes)

---

## Phase 1: Foundation for Long-Distance Exploration

### Priority 1A: RND with Long-Chain Configuration

**Impact**: 🔥🔥🔥 CRITICAL

**Scope**: Medium (~500 lines)

**Time**: 3-4 days

**Long-Distance Adaptation**:

- **Strong initial weight**: 1.5-2.0 (sustain 50-200 step exploration chains)
- **Episodic novelty**: 2-3× bonus multiplier for new states within episode
- **Slow decay**: Linear decay over 1M steps (not 500k) to maintain exploration longer

**Implementation**:

```python
# npp_rl/exploration/rnd_module.py
class RNDModule(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256):
        # Target network (frozen) - encodes "ideal" features
        self.target_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Predictor network (trained) - learns to predict target
        self.predictor_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Freeze target network
        for param in self.target_net.parameters():
            param.requires_grad = False
        
        # Episodic memory for long-chain exploration
        self.episodic_memory = set()  # Track visited states per episode
        
    def compute_intrinsic_reward(self, state_features, episodic_bonus=2.0):
        # Prediction error = intrinsic curiosity
        with torch.no_grad():
            target = self.target_net(state_features)
        prediction = self.predictor_net(state_features)
        
        error = F.mse_loss(prediction, target, reduction='none').mean(dim=1)
        
        # Episodic novelty bonus for sustaining long chains
        state_hash = hash(state_features.cpu().numpy().tobytes())
        is_novel = state_hash not in self.episodic_memory
        if is_novel:
            self.episodic_memory.add(state_hash)
            error = error * episodic_bonus  # 2-3× bonus for new states
        
        return error
    
    def reset_episode(self):
        self.episodic_memory.clear()
```

**Reward Integration** ([`main_reward_calculator.py`](nclone/nclone/gym_environment/reward_calculation/main_reward_calculator.py:224)):

```python
# Add RND bonus scaled for long-distance exploration
rnd_weight = 1.5 * max(0.3, 1.0 - timesteps / 1_000_000)  # Decay to 0.45
intrinsic_reward = rnd_module.compute_bonus(graph_features) * rnd_weight
reward += intrinsic_reward
```

**Expected**: Agent discovers goals in 300k-700k steps (vs never discovering them)

---

### Priority 1B: Go-Explore - PROMOTED TO PHASE 1

**Impact**: 🔥🔥🔥 CRITICAL for long distances

**Scope**: Medium (~600 lines)

**Time**: 3-4 days

**Why Promoted**: With 200-1000px distances, agent CANNOT stumble to goals randomly. Go-Explore enables incremental progress by checkpointing.

**Implementation**:

```python
# npp_rl/exploration/checkpoint_archive.py
class CheckpointArchive:
    def __init__(self, grid_size=24):
        # Archive: cell → (state, distance_to_goal, novelty_score)
        self.archive = {}
        self.grid_size = grid_size
        
    def add_checkpoint(self, state, distance_to_goal, novelty):
        cell = self._discretize(state['player_x'], state['player_y'])
        
        # Only save if: (1) novel cell OR (2) closer to goal than existing
        if cell not in self.archive or distance_to_goal < self.archive[cell]['distance']:
            self.archive[cell] = {
                'state': self._serialize_state(state),
                'distance': distance_to_goal,
                'novelty': novelty,
            }
    
    def sample_checkpoint(self):
        # Prioritize: (1) close to goal (2) novel regions
        candidates = sorted(
            self.archive.items(),
            key=lambda x: x[1]['distance'] - 50 * x[1]['novelty']
        )
        return candidates[0][1]['state'] if candidates else None
```

**Environment Integration** ([`base_environment.py`](nclone/nclone/gym_environment/base_environment.py:873)):

```python
def reset(self, seed=None, options=None):
    # 20% of episodes: start from promising checkpoint
    if np.random.random() < 0.2 and self.checkpoint_archive:
        checkpoint = self.checkpoint_archive.sample_checkpoint()
        if checkpoint:
            self._load_checkpoint(checkpoint)
            logger.info(f"Reset to checkpoint at distance={checkpoint['distance']}")
    
    # Normal reset otherwise
    return super().reset(seed, options)
```

**Expected**: Agent makes incremental progress (900px → 700px → 500px → 300px → goal)

---

### Priority 1C: Curriculum Threshold Adjustment

**Impact**: 🔥🔥 HIGH

**Scope**: Minimal (1 file, 5 minutes)

Lower thresholds to 30%, 25%, 20% etc. (same as original plan)

---

## Phase 2: Sample Efficiency & Physics Learning

### Priority 2A: Demonstration Augmentation - PROMOTED TO PHASE 2

**Impact**: 🔥🔥🔥 CRITICAL for physics primitives

**Scope**: Medium (~500 lines)

**Time**: 2-3 days

**Why Promoted**: Complex physics navigation requires learning jumping, wall-jumping, momentum management. Expert demos provide these primitives.

**Implementation**:

```python
# npp_rl/training/demonstration_buffer.py
class DemonstrationBuffer:
    def __init__(self, replay_paths, max_demos=100):
        self.demonstrations = []
        self._load_replays(replay_paths, max_demos)
    
    def sample_batch(self, batch_size):
        # Sample from demonstration trajectories
        indices = np.random.choice(len(self.demonstrations), batch_size)
        return [self.demonstrations[i] for i in indices]

# Mix with on-policy data (20% demos, 80% rollouts)
demo_batch = demo_buffer.sample_batch(int(batch_size * 0.2))
rollout_batch = rollout_buffer.sample_batch(int(batch_size * 0.8))
combined_batch = demo_batch + rollout_batch
```

**Expected**: Agent learns jumping/movement primitives in first 200k steps

---

### Priority 2B: Hindsight Experience Replay (HER)

**Impact**: 🔥🔥🔥 CRITICAL

**Scope**: Large (~800 lines)

**Time**: 5-6 days

**Goal-Conditioning for Long Distances**:

- Add `current_goal` to [`reachability_features`](nclone/OBSERVATION_SPACE_README.md:97): expand from 7→9 dims
- New dims: `[7]` goal_x normalized, `[8]` goal_y normalized
- PBRS uses goal from observation (not hardcoded switch/exit)

**Implementation**: Same as original plan, but emphasize goal-conditioning is CRITICAL for long distances

**Expected**: 5-8× sample efficiency improvement (90% failures → learning signal)

---

### Priority 2C: Asymmetric Actor-Critic

**Impact**: 🔥🔥 HIGH

**Scope**: Medium (~400 lines)

**Time**: 2-3 days

Privileged info for critic:

- Shortest path distance (already computed)
- Estimated steps to goal
- Whether agent is moving toward/away from goal

Same as original plan.

---

## Phase 3: Optimization & Refinement

### Priority 3A: Temporal Action Abstraction

**Impact**: 🔥 MEDIUM-HIGH (more important for long distances)

**Scope**: Large (~800 lines)

**Time**: 1 week

**Why Important for Long Distances**: 1000px journey = ~300-500 actions. Temporal abstraction reduces to ~50-100 macro-actions.

Learn macro-actions from demonstrations:

- "Jump forward" (10-15 frames)
- "Wall jump" (8-12 frames)
- "Navigate corridor" (20-40 frames)

---

## Implementation Schedule

### Week 1-2: Phase 1 - Foundation

1. RND with long-chain config (3-4 days)
2. Go-Explore checkpointing (3-4 days)
3. Curriculum thresholds (1 hour)
4. **Training run** (2-3 days): Target 15-25% success

### Week 3-4: Phase 2 - Efficiency

5. Demonstration augmentation (2-3 days)
6. Hindsight Experience Replay (5-6 days)
7. **Training run** (2-3 days): Target 30-40% success

### Week 5-6: Phase 2 continued

8. Asymmetric Actor-Critic (2-3 days)
9. **Training run** (2-3 days): Target 40-50% success

### Week 7+ (Optional): Phase 3

10. Temporal abstraction (1 week)

---

## Critical Differences from Short-Distance Case

| Aspect | Short Distance (50-200px) | Long Distance (200-1000px) | Your Adjustments |

|--------|--------------------------|----------------------------|------------------|

| **RND Weight** | 0.3-0.5 | 1.0-2.0 | Higher to sustain long chains |

| **RND Decay** | 500k steps | 1M steps | Slower to maintain exploration |

| **Go-Explore** | Optional (Tier 2) | Critical (Phase 1) | REQUIRED for progress |

| **Demonstrations** | Nice-to-have | Critical | Learn physics primitives |

| **HER Priority** | High | Critical | 90% failures must become learning signal |

| **Success Target** | 30-40% Phase 1 | 15-25% Phase 1 | Lower initial target |

---

## Risk Mitigation

### Long-Distance Specific Risks

1. **Agent never reaches goals even with RND**

   - **Mitigation**: Go-Explore checkpointing ensures incremental progress
   - **Fallback**: Waypoint curriculum (start 400px from goal, increase to 1000px)

2. **Demonstration distribution shift**

   - **Mitigation**: Only use demos for first 30% of training, decay weight
   - **Expert trajectories may use different paths than learned policy**

3. **Checkpoints break curriculum**

   - **Mitigation**: Only enable Go-Explore within current curriculum stage
   - **Disable checkpoints when advancing to new stage**

---

## Expected Outcomes

### After Phase 1 (RND + Go-Explore + Curriculum)

- **Success Rate**: 15-25% (lower than short-distance case)
- **Exploration**: Agent discovers goals via checkpointing
- **Behavior**: Incremental progress toward goals

### After Phase 2 (HER + Demos + Asymmetric Critic)

- **Success Rate**: 30-40%
- **Navigation**: Learns physics primitives from demos
- **Sample Efficiency**: 5-8× improvement

### After Phase 3 (Temporal Abstraction)

- **Success Rate**: 45-55%
- **Efficiency**: Faster decision-making with macro-actions

---

## Monitoring Metrics

### Long-Distance Specific Metrics

1. **Progressive exploration**:

   - `exploration/best_distance_to_switch` (should decrease over time)
   - `exploration/checkpoints_archived` (should grow)
   - `exploration/distance_distribution` (histogram of how close agent gets)

2. **Physics learning**:

   - `physics/jump_success_rate` (from demonstrations)
   - `physics/wall_jump_frequency` (should match expert level)
   - `physics/average_velocity` (should approach expert level)

3. **Sample efficiency**:

   - `her/goals_reached_per_episode` (with relabeling)
   - `training/episodes_to_first_success` (should decrease)

---

## Conclusion

Long-distance exploration (200-1000px) fundamentally changes priority order:

1. **Go-Explore promoted to Phase 1** - Without checkpointing, agent cannot make incremental progress
2. **Demonstrations promoted to Phase 2** - Complex physics requires learning movement primitives
3. **Higher RND weights** - Must sustain 50-200 step exploration chains
4. **Lower success expectations** - 15-25% Phase 1 (not 30%) is realistic

The combination of RND (exploration motivation) + Go-Explore (incremental progress) + Demonstrations (physics skills) + HER (learning from failures) is ESSENTIAL for long-distance navigation. Any single technique alone will fail.