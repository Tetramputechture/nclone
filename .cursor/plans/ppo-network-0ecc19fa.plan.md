<!-- 0ecc19fa-5344-4906-8be8-85e7411ee54b 881fd1db-5a5d-40c5-a49a-9c28f1fd1993 -->
# PPO Policy Network Architecture Implementation Plan

## Overview

This plan implements state-of-the-art architectural improvements to the PPO policy network for robust sequential navigation in N++ levels. With H100 GPUs (80GB VRAM), we can implement the most sophisticated architectures without compromise.

## Executive Summary of Changes

### Current Architecture (Baseline)

- **Policy Head**: 3-layer MLP `[256, 256, 128]` → 6 action logits
- **Value Head**: 3-layer MLP `[256, 256, 128]` → 1 value estimate
- **Feature Extraction**: Shared 512-dim from ConfigurableMultimodalExtractor
- **Total Params**: ~724K (policy + value heads only)
- **Activation**: ReLU
- **Normalization**: None

### Target Architecture (Comprehensive)

- **Policy Head**: 5-layer ResNet `[512, 512, 384, 256, 256]` with residual connections
- **Value Head**: Dueling architecture (state value + advantage streams)
- **Feature Extraction**: Separate extractors for policy/value (gradient isolation)
- **Attention Mechanism**: Multi-head attention over objectives (16 locked doors + switches + exit)
- **Auxiliary Tasks**: Death prediction, time-to-goal, subgoal classification
- **Distributional Critic**: QR-DQN (51 quantiles) for uncertainty-aware value estimates
- **Total Params**: ~15-20M (feasible on H100, provides strong capacity)
- **Activation**: SiLU (Swish) - proven better than ReLU for deep networks
- **Normalization**: LayerNorm after each layer

## Phase 1: Foundation - Deep ResNet Policy with Separate Feature Extractors

### 1.1 Create Custom Actor-Critic Policy Class

**File**: `npp_rl/agents/deep_resnet_actor_critic_policy.py`

**Architecture**:

```python
class DeepResNetActorCriticPolicy(MaskedActorCriticPolicy):
    """
    Deep ResNet-based actor-critic with separate feature extractors.
    
    Policy Network (5 layers with residual connections):
    - Input: 512-dim features from policy feature extractor
    - Layer 1: Linear(512, 512) + LayerNorm + SiLU
    - Layer 2: Linear(512, 512) + LayerNorm + SiLU + Residual(from input)
    - Layer 3: Linear(512, 384) + LayerNorm + SiLU
    - Layer 4: Linear(384, 256) + LayerNorm + SiLU + Residual(from layer 3, projected)
    - Layer 5: Linear(256, 256) + LayerNorm + SiLU
    - Output: Linear(256, 6) → action logits
    
    Value Network (Dueling architecture):
    - Input: 512-dim features from value feature extractor
    - State Value Stream: [512, 384, 256] → 1 value
    - Advantage Stream: [512, 384, 256] → 6 advantages
    - Combine: V(s) + (A(s,a) - mean(A(s,*)))
    
    Total params: ~3.5M (policy) + ~3.5M (value) = ~7M parameters
    """
```

**Key Features**:

- Residual connections every 2 layers for gradient flow
- SiLU activation (proven superior to ReLU in modern architectures)
- LayerNorm for stable training with large batches
- Dueling architecture for better value decomposition
- Separate feature extractors (set `share_features_extractor=False`)

**Implementation**:

```python
# Override _build_mlp_extractor() to create deep ResNet
def _build_mlp_extractor(self):
    self.mlp_extractor = DeepResNetMLPExtractor(
        feature_dim=512,
        policy_layers=[512, 512, 384, 256, 256],
        value_layers=[512, 384, 256],
        activation_fn=nn.SiLU,
        use_residual=True,
        use_layer_norm=True,
        dueling=True,
    )
```

### 1.2 Implement ResNet MLP Extractor

**File**: `npp_rl/models/deep_resnet_mlp.py`

**Components**:

```python
class ResidualBlock(nn.Module):
    """Residual block with LayerNorm and SiLU activation."""
    def __init__(self, in_dim, out_dim, use_projection=True):
        # Main path: Linear → LayerNorm → SiLU
        # Residual path: Identity or Linear projection if dims differ
        
class DeepResNetMLPExtractor(BaseFeaturesExtractor):
    """Deep ResNet MLP for policy and value streams with dueling."""
    
    def __init__(
        self,
        feature_dim: int = 512,
        policy_layers: List[int] = [512, 512, 384, 256, 256],
        value_layers: List[int] = [512, 384, 256],
        activation_fn: nn.Module = nn.SiLU,
        use_residual: bool = True,
        use_layer_norm: bool = True,
        dueling: bool = True,
        num_actions: int = 6,
    ):
        # Build policy network with residual connections
        # Build value network with dueling streams
        # state_value: [512, 384, 256] → 1
        # advantage: [512, 384, 256] → 6
```

### 1.3 Update Architecture Trainer Configuration

**File**: `npp_rl/training/architecture_trainer.py`

**Changes**:

```python
# In setup_model()
self.policy_class = DeepResNetActorCriticPolicy

self.policy_kwargs = {
    "features_extractor_class": ConfigurableMultimodalExtractor,
    "features_extractor_kwargs": {
        "config": self.architecture_config,
        "frame_stack_config": self.frame_stack_config,
    },
    "share_features_extractor": False,  # CRITICAL: Separate extractors
    "net_arch": {
        "pi": [512, 512, 384, 256, 256],  # Policy: 5 layers
        "vf": [512, 384, 256],  # Value: 3 layers (dueling splits internally)
    },
    "activation_fn": nn.SiLU,  # Modern activation
    "normalize_images": False,
    "optimizer_kwargs": {"eps": 1e-5},
    "use_residual": True,  # Enable residual connections
    "use_layer_norm": True,  # Enable LayerNorm
    "dueling": True,  # Enable dueling architecture
}
```

## Phase 2: Attention-Based Policy Head for Variable Objectives

### 2.1 Create Attention Over Objectives Module

**File**: `npp_rl/models/objective_attention.py`

**Purpose**: Handle variable number of objectives (1-16 locked doors + switches + exit)

**Architecture**:

```python
class ObjectiveAttentionPolicy(nn.Module):
    """
    Attention-based policy that conditions on variable objectives.
    
    Query: Current state embedding (256-dim from policy MLP)
    Keys/Values: Objective embeddings (locked doors, switches, exit)
    
    Process:
    1. Extract objective features from observation
       - Exit switch position + activation state (3 dims)
       - Exit door position + accessibility (3 dims)
       - Locked door positions + states (16 × 4 dims = 64 dims, padded)
       - Locked door switch positions + states (16 × 4 dims = 64 dims, padded)
    
    2. Encode each objective type separately
       - Exit objectives: MLP(6) → 64-dim embedding
       - Locked doors: MLP(4) → 64-dim embedding per door (max 16)
       - Locked switches: MLP(4) → 64-dim embedding per switch (max 16)
    
    3. Multi-head attention (8 heads)
       - Query: policy_features (256-dim) → project to 512-dim
       - Keys: all objective embeddings (1+1+16+16 = 34 max)
       - Values: same as keys
       - Attention mask: mask out padded objectives
    
    4. Combine attended objectives with policy features
       - Concatenate: [policy_features(256) + attended(512)] = 768-dim
       - Project: Linear(768, 256) → final policy embedding
    
    5. Output action logits: Linear(256, 6)
    
    Benefits:
    - Handles variable number of locked doors (1-16)
    - Learns to focus on relevant objectives (e.g., nearest unfinished door)
    - Permutation invariant over locked doors
    - Explicit sequential reasoning (which door to prioritize)
    """
```

**Implementation**:

```python
class ObjectiveAttentionPolicy(nn.Module):
    def __init__(
        self,
        policy_feature_dim: int = 256,
        objective_embed_dim: int = 64,
        attention_dim: int = 512,
        num_heads: int = 8,
        max_locked_doors: int = 16,
        num_actions: int = 6,
    ):
        # Objective encoders
        self.exit_encoder = nn.Sequential(
            nn.Linear(6, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, objective_embed_dim)
        )
        self.locked_door_encoder = nn.Sequential(
            nn.Linear(4, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, objective_embed_dim)
        )
        self.locked_switch_encoder = nn.Sequential(
            nn.Linear(4, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, objective_embed_dim)
        )
        
        # Multi-head attention
        self.query_proj = nn.Linear(policy_feature_dim, attention_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        
        # Fusion and output
        self.fusion = nn.Sequential(
            nn.Linear(policy_feature_dim + attention_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.action_head = nn.Linear(256, num_actions)
    
    def forward(self, policy_features, obs_dict):
        # Extract objective information from observations
        # Apply attention over objectives
        # Combine with policy features
        # Output action logits
```

### 2.2 Update Observation Space for Objective Encoding

**File**: `npp_rl/training/environment_factory.py`

**Add to observations**:

```python
# Already in observations from nclone:
# - switch_activated (bool)
# - exit_door_accessible (bool) 
# - entities (list of entity objects)
# - locked_doors (list of locked door objects)
# - locked_door_switches (list of switch objects)

# Need to add structured encoding:
obs["objective_features"] = {
    "exit_switch_pos": np.array([switch_x, switch_y], dtype=np.float32),
    "exit_switch_activated": np.array([activated], dtype=np.float32),
    "exit_door_pos": np.array([door_x, door_y], dtype=np.float32),
    "exit_door_accessible": np.array([accessible], dtype=np.float32),
    "locked_door_positions": np.zeros((16, 2), dtype=np.float32),  # Padded to 16
    "locked_door_states": np.zeros((16, 2), dtype=np.float32),  # [open, distance]
    "locked_switch_positions": np.zeros((16, 2), dtype=np.float32),
    "locked_switch_states": np.zeros((16, 2), dtype=np.float32),  # [collected, distance]
    "num_locked_doors": np.array([actual_count], dtype=np.int32),
}
```

## Phase 3: Distributional Value Function (QR-DQN)

### 3.1 Implement Quantile Regression Value Head

**File**: `npp_rl/models/distributional_value.py`

**Purpose**: Model value distribution instead of point estimate for better uncertainty quantification

**Architecture**:

```python
class QuantileValueHead(nn.Module):
    """
    Quantile Regression DQN value head.
    
    Instead of outputting single value V(s), outputs distribution over values
    using N=51 quantiles (standard from QR-DQN paper).
    
    Input: 256-dim value features
    Output: 51 quantile values (representing distribution)
    
    Loss: Quantile Huber loss instead of MSE
    
    Benefits:
    - Captures uncertainty in returns (e.g., risky vs safe paths)
    - More robust to outliers than MSE
    - Better gradient flow (no squashing from MSE)
    - Proven to improve performance in Atari domains
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_quantiles: int = 51,
        hidden_dim: int = 512,
    ):
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_quantiles),  # Output 51 quantiles
        )
        
        # Quantile midpoints for τ (tau): [0.01, 0.03, ..., 0.99]
        self.register_buffer(
            "quantile_tau",
            torch.linspace(0.01, 0.99, num_quantiles)
        )
    
    def forward(self, value_features):
        # Shape: [batch, 51] - one value per quantile
        quantile_values = self.value_net(value_features)
        return quantile_values
    
    def get_value_estimate(self, quantile_values):
        # For policy updates, use mean of quantiles as value estimate
        return quantile_values.mean(dim=-1, keepdim=True)
```

### 3.2 Implement Quantile Huber Loss

**File**: `npp_rl/training/quantile_loss.py`

**Implementation**:

```python
def quantile_huber_loss(
    quantile_values: torch.Tensor,  # [batch, num_quantiles]
    target_values: torch.Tensor,    # [batch, num_quantiles]
    quantile_tau: torch.Tensor,     # [num_quantiles]
    kappa: float = 1.0,             # Huber threshold
) -> torch.Tensor:
    """
    Quantile Huber loss from QR-DQN paper.
    
    For each quantile τ, computes asymmetric Huber loss:
    - If error > 0: weight by τ
    - If error < 0: weight by (1 - τ)
    
    This encourages different quantiles to capture different parts
    of the return distribution.
    """
    # Compute TD errors for each quantile
    td_errors = target_values - quantile_values  # [batch, num_quantiles]
    
    # Huber loss element
    huber_loss = torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (td_errors.abs() - 0.5 * kappa)
    )
    
    # Quantile weighting
    quantile_weight = torch.abs(
        quantile_tau - (td_errors < 0).float()
    )
    
    # Combine
    loss = (quantile_weight * huber_loss).mean()
    return loss
```

### 3.3 Update PPO to Use Distributional Critic

**File**: `npp_rl/agents/distributional_ppo.py`

**Key Changes**:

```python
class DistributionalPPO(PPO):
    """PPO with distributional value function."""
    
    def train(self):
        # ... standard PPO policy updates ...
        
        # Modified value loss: use quantile Huber loss
        for quantile_idx in range(num_quantiles):
            quantile_target = self._compute_quantile_target(
                rollout_data, quantile_idx
            )
            quantile_pred = values[:, quantile_idx]
            value_loss += quantile_huber_loss(
                quantile_pred, quantile_target, tau[quantile_idx]
            )
        
        value_loss = value_loss / num_quantiles
```

## Phase 4: Auxiliary Prediction Tasks

### 4.1 Add Multi-Task Prediction Heads

**File**: `npp_rl/models/auxiliary_tasks.py`

**Tasks**:

1. **Death Prediction** (binary classification)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Input: 256-dim policy features
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Output: Probability of death in next 10 steps
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Loss: Binary cross-entropy
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Label: 1 if died within 10 steps, 0 otherwise

2. **Time-to-Goal Prediction** (regression)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Input: 256-dim policy features
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Output: Estimated steps to reach current objective
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Loss: Smooth L1 loss
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Label: Actual steps taken to reach objective (from hindsight)

3. **Next Subgoal Classification** (multi-class)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Input: 256-dim policy features
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Output: Which objective to pursue next (exit_switch, exit_door, locked_door_N, locked_switch_N)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Loss: Cross-entropy
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Label: Optimal next objective from expert trajectories or A* planning

**Architecture**:

```python
class AuxiliaryTaskHeads(nn.Module):
    """Multi-task prediction heads for auxiliary learning."""
    
    def __init__(self, feature_dim: int = 256, max_objectives: int = 34):
        # Death prediction head
        self.death_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        # Time-to-goal prediction head
        self.time_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Softplus(),  # Ensure positive output
        )
        
        # Next subgoal classification head
        self.subgoal_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, max_objectives),  # One logit per possible objective
        )
    
    def forward(self, features):
        return {
            "death_prob": self.death_head(features),
            "time_to_goal": self.time_head(features),
            "next_subgoal_logits": self.subgoal_head(features),
        }
```

### 4.2 Update Training Loop for Multi-Task Learning

**File**: `npp_rl/training/multi_task_trainer.py`

**Training procedure**:

```python
# Total loss = policy_loss + value_loss + auxiliary_losses
total_loss = (
    policy_loss 
    + vf_coef * value_loss 
    + 0.1 * death_loss  # Weight auxiliary tasks lower
    + 0.1 * time_loss 
    + 0.1 * subgoal_loss
)

# Auxiliary task labels (computed during rollout)
death_labels = (episode_death_frame - current_frame <= 10).float()
time_labels = (episode_goal_frame - current_frame).clamp(min=0, max=1000)
subgoal_labels = get_optimal_next_objective_from_hindsight(trajectory)
```

## Phase 5: Advanced Reward Shaping and Time Sensitivity

### 5.1 Add Time-Conditional Policy Gating

**File**: `npp_rl/models/time_conditional_policy.py`

**Purpose**: Make policy urgency-aware based on time remaining

**Architecture**:

```python
class TimeConditionalGating(nn.Module):
    """
    Gate policy outputs based on time urgency.
    
    As time runs out, increases exploration and risk-taking.
    Early in episode: conservative, prefer safe paths
    Late in episode: aggressive, take risks to complete quickly
    """
    
    def __init__(self, feature_dim: int = 256):
        # Urgency encoder: time_remaining → urgency_embedding
        self.urgency_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, feature_dim),
            nn.Tanh(),  # Output in [-1, 1]
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid(),  # Gate values in [0, 1]
        )
    
    def forward(self, policy_features, time_remaining_normalized):
        # Encode urgency
        urgency = self.urgency_encoder(time_remaining_normalized)
        
        # Gate policy features based on urgency
        gate_values = self.gate(torch.cat([policy_features, urgency], dim=-1))
        gated_features = policy_features * gate_values
        
        return gated_features + urgency * (1 - gate_values)  # Residual
```

### 5.2 Update Reward Structure for Better Time Incentives

**File**: `nclone/nclone/gym_environment/reward_calculation/reward_constants.py`

**Proposed changes**:

```python
# Current
LEVEL_COMPLETION_REWARD = 20.0
TIME_PENALTY_PER_STEP = -0.00001

# Recommended: Add temporal decay to completion reward
def get_completion_reward(steps_taken, max_steps):
    """
    Decaying completion reward encourages speed.
    Fast completion (10% of time): 20.0
    Medium completion (50% of time): 15.0
    Slow completion (90% of time): 10.0
    """
    time_ratio = steps_taken / max_steps
    decay_factor = 1.0 - 0.5 * time_ratio  # Linear decay to 50%
    return LEVEL_COMPLETION_REWARD * decay_factor

# Add efficiency bonus for completing objectives quickly
OBJECTIVE_EFFICIENCY_BONUS = {
    "fast_switch": 2.0,    # Activated switch in <30% of time
    "fast_completion": 5.0,  # Completed level in <30% of time
}
```

## Phase 6: Integration and Testing

### 6.1 Create Unified Architecture Configuration

**File**: `npp_rl/training/architecture_configs.py`

**Add new configuration**:

```python
def create_deep_attention_config() -> ArchitectureConfig:
    """
    Deep attention-based architecture with all improvements.
    
    - Separate feature extractors (gradient isolation)
    - Deep ResNet policy (5 layers with residuals)
    - Dueling value function
    - Distributional critic (51 quantiles)
    - Attention over objectives
    - Auxiliary prediction tasks
    - Time-conditional gating
    
    Total parameters: ~15-20M
    Target hardware: H100 80GB
    Expected performance: 2-3x sample efficiency vs baseline
    """
    return ArchitectureConfig(
        name="deep_attention",
        description="Deep ResNet with attention, dueling, and auxiliary tasks",
        modalities=ModalityConfig(
            use_player_frame=True,
            use_global_view=True,
            use_graph=True,
            use_game_state=True,
            use_reachability=True,
        ),
        graph=GraphConfig(
            architecture=GraphArchitectureType.GAT,
            hidden_dim=128,
            output_dim=256,
            num_layers=3,
            num_heads=4,
        ),
        visual=VisualConfig(
            player_frame_output_dim=256,
            global_output_dim=128,
        ),
        state=StateConfig(
            game_state_dim=58,
            reachability_dim=8,
            hidden_dim=128,
            output_dim=128,
            use_attentive_state_mlp=True,
        ),
        fusion=FusionConfig(
            fusion_type=FusionType.MULTI_HEAD_ATTENTION,
            num_attention_heads=8,
        ),
        features_dim=512,
    )
```

### 6.2 Update Training Script

**File**: `scripts/train_deep_attention_agent.py`

**Training configuration**:

```python
# Architecture
architecture = "deep_attention"

# PPO hyperparameters (adjusted for deeper network)
hyperparameters = {
    "learning_rate": 1e-4,  # Lower for stability with deeper network
    "n_steps": 4096,  # Larger rollouts with H100 memory
    "batch_size": 512,  # Larger batches for stable gradients
    "n_epochs": 10,  # More epochs per update
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": 2.0,  # Higher for distributional values
    "ent_coef": 0.01,  # Moderate entropy for exploration
    "vf_coef": 0.5,
    "max_grad_norm": 1.0,  # Tighter gradient clipping
    "auxiliary_coef": 0.1,  # Weight for auxiliary tasks
}

# Training setup
num_envs = 128  # High parallelism with H100
total_timesteps = 10_000_000
eval_freq = 50_000
```

### 6.3 Comprehensive Testing Suite

**File**: `tests/test_deep_attention_policy.py`

**Test cases**:

1. Forward pass test (ensure shapes correct)
2. Backward pass test (ensure gradients flow)
3. Residual connection test (verify skip connections)
4. Attention mechanism test (verify masking for variable doors)
5. Dueling architecture test (verify V(s) + A(s,a) decomposition)
6. Distributional value test (verify quantile outputs)
7. Auxiliary task test (verify prediction heads)
8. Time gating test (verify urgency modulation)

## Implementation Timeline

### Week 1: Foundation

- [ ] Implement ResidualBlock and DeepResNetMLPExtractor
- [ ] Create DeepResNetActorCriticPolicy with dueling
- [ ] Update architecture_trainer.py for separate feature extractors
- [ ] Test forward/backward passes
- [ ] Benchmark parameter count and memory usage

### Week 2: Attention Mechanism

- [ ] Implement ObjectiveAttentionPolicy
- [ ] Update observation space for objective features
- [ ] Create attention mask handling for variable doors
- [ ] Test attention weights visualization
- [ ] Verify permutation invariance over locked doors

### Week 3: Distributional Value Function

- [ ] Implement QuantileValueHead
- [ ] Implement quantile_huber_loss
- [ ] Create DistributionalPPO class
- [ ] Test distributional value estimates
- [ ] Verify quantile target computation

### Week 4: Auxiliary Tasks

- [ ] Implement AuxiliaryTaskHeads
- [ ] Create multi-task training loop
- [ ] Generate auxiliary labels from trajectories
- [ ] Test auxiliary prediction accuracy
- [ ] Monitor auxiliary task gradients

### Week 5: Time Conditioning & Integration

- [ ] Implement TimeConditionalGating
- [ ] Update reward structure with temporal decay
- [ ] Create unified deep_attention config
- [ ] Integration testing of all components
- [ ] Performance benchmarking vs baseline

### Week 6: Evaluation & Tuning

- [ ] Train on full dataset (1-16 doors, 0-256 mines)
- [ ] Ablation studies (test each component individually)
- [ ] Generalization tests (train on subset, test on full range)
- [ ] Hyperparameter tuning (learning rate, auxiliary weights)
- [ ] Final evaluation and comparison

## Expected Improvements

### Performance Metrics

- **Sample Efficiency**: 2-3x fewer timesteps to reach same success rate
- **Success Rate**: 80%+ on test levels (vs 44% baseline)
- **Generalization**: <10% performance drop from 4 to 16 doors (vs 30%+ baseline)
- **Time Efficiency**: 20% faster completion times (temporal decay incentive)
- **Death Rate**: 30% reduction (death prediction auxiliary task)

### Computational Requirements

- **Parameters**: 15-20M (vs 724K baseline) - 20-30x larger
- **Memory**: ~40GB VRAM (policy + value + replay buffer) - well within H100 capacity
- **Training Time**: ~48 hours for 10M steps on H100 (vs 24 hours baseline)
- **Inference**: ~5ms per action (vs 2ms baseline) - still real-time capable

## Risk Mitigation

### Potential Issues & Solutions

1. **Overfitting**: Use dropout (0.1), weight decay (1e-4), and data augmentation
2. **Training instability**: Lower learning rate, gradient clipping, LayerNorm
3. **Slow convergence**: Learning rate warmup, auxiliary task annealing
4. **Memory issues**: Gradient accumulation if needed (shouldn't be with H100)
5. **Auxiliary task interference**: Carefully tune auxiliary loss weights (0.05-0.2 range)

## Success Criteria

The implementation is successful if:

1. ✓ All components integrate without errors
2. ✓ Training is stable (no NaN losses, exploding gradients)
3. ✓ Success rate improves by >50% over baseline
4. ✓ Generalization degradation <15% (4 doors → 16 doors)
5. ✓ Average completion time reduces by >15%
6. ✓ Death rate reduces by >25%
7. ✓ Auxiliary tasks achieve >70% accuracy

## Monitoring & Logging

**Key metrics to track**:

- Policy loss, value loss, auxiliary losses
- Gradient norms (policy, value, auxiliary)
- Policy entropy (should decay but not collapse)
- Attention weights distribution (should focus on relevant objectives)
- Quantile spread (should increase with uncertainty)
- Auxiliary task accuracies
- Episode rewards, length, success rate
- Death causes breakdown (mines vs impact vs truncation)
- Objective completion order (should learn optimal sequences)

### To-dos

- [ ] Create ResidualBlock and DeepResNetMLPExtractor with 5-layer policy network, dueling value streams, SiLU activation, and LayerNorm
- [ ] Create DeepResNetActorCriticPolicy class that inherits from MaskedActorCriticPolicy and uses separate feature extractors
- [ ] Update architecture_trainer.py to support share_features_extractor=False and new policy class
- [ ] Write tests for forward/backward passes, residual connections, and memory usage of foundation architecture
- [ ] Create ObjectiveAttentionPolicy with multi-head attention over variable objectives (locked doors, switches, exit)
- [ ] Add structured objective features to observations (positions, states, masks for up to 16 locked doors)
- [ ] Test attention mechanism with variable door counts, verify masking and permutation invariance
- [ ] Create QuantileValueHead with 51 quantiles and quantile_huber_loss function
- [ ] Create DistributionalPPO class that uses quantile regression for value function
- [ ] Test distributional value estimates, quantile target computation, and loss calculation
- [ ] Create AuxiliaryTaskHeads for death prediction, time-to-goal, and next-subgoal classification
- [ ] Update training loop to compute auxiliary labels and multi-task loss
- [ ] Test auxiliary prediction accuracy and gradient flow for auxiliary tasks
- [ ] Create TimeConditionalGating module to modulate policy based on time urgency
- [ ] Add temporal decay to completion reward and efficiency bonuses for fast objective completion
- [ ] Create deep_attention architecture config and integrate all components
- [ ] Perform end-to-end testing of complete architecture with all components
- [ ] Train deep attention agent and compare performance metrics vs baseline
- [ ] Run ablation studies to measure contribution of each architectural component
- [ ] Test generalization across variable door counts and mine configurations
- [ ] Fine-tune learning rate, auxiliary weights, and other hyperparameters