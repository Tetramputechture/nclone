---
name: Architecture Simplification Analysis
overview: Code changes for simplified ML architecture, rule-based controller, and MPC alternatives. Training/testing done manually.
todos:
  - id: minimal-observation-mode
    content: Add minimal 40-dim observation mode to nclone environment config
    status: completed
  - id: minimal-mlp-architecture
    content: Create minimal_mlp architecture config and feature extractor in npp-rl
    status: completed
  - id: rule-based-controller
    content: Implement rule-based controller script using A* next_hop_direction
    status: completed
  - id: mpc-controller
    content: Implement MPC controller with simulation rollouts
    status: completed
  - id: simplified-hyperparams
    content: Create simplified hyperparameter config without state stacking
    status: completed
---

# NPP-RL Architecture Simplification - Code Changes

## Background

Current setup is potentially over-engineered for a fully Markov problem:

- 4-frame state stacking adds 123 dims but optimal action is time-independent
- Agent learns to follow A* path guidance instead of using it directly
- ~200K network parameters for a 6-action discrete problem

---

## Code Change 1: Minimal Observation Mode (nclone)

### File: `nclone/nclone/gym_environment/config.py`

Add `ObservationMode` enum and config option:

```python
class ObservationMode(Enum):
    FULL = "full"           # Current: 41 game_state + 38 reach + 96 spatial + 3 sdf
    MINIMAL = "minimal"     # New: 40 dims total (physics + path + mines + buffers)

@dataclass
class EnvironmentConfig:
    # ... existing fields ...
    observation_mode: ObservationMode = ObservationMode.FULL
```

### File: `nclone/nclone/gym_environment/observation_processor.py`

Add method to compute minimal observation:

```python
def compute_minimal_observation(self, ninja, reachability_features, mine_overlay) -> np.ndarray:
    """40-dim minimal observation for simplified training."""
    obs = np.zeros(40, dtype=np.float32)
    
    # Core physics (12 dims): velocity, state one-hot, airborne, walled
    obs[0] = ninja.xspeed / MAX_HOR_SPEED
    obs[1] = ninja.yspeed / MAX_HOR_SPEED
    obs[2:7] = one_hot_state(ninja.state, 5)  # States 0-4
    obs[7] = 1.0 if ninja.airborn else -1.0
    obs[8] = 1.0 if ninja.walled else -1.0
    obs[9] = ninja.wall_normal if ninja.walled else 0.0
    obs[10] = ninja.floor_normalized_x
    obs[11] = ninja.floor_normalized_y
    
    # Path guidance (8 dims): from reachability_features indices
    obs[12:14] = reachability_features[13:15]  # next_hop_dir x,y
    obs[14:16] = reachability_features[15:17]  # waypoint_dir x,y
    obs[16:18] = reachability_features[8:10]   # exit_dir x,y
    obs[18] = reachability_features[12]        # phase (switch_activated)
    obs[19] = reachability_features[24]        # path_curvature
    
    # Mine context (16 dims): 4 nearest mines × 4 features
    obs[20:36] = mine_overlay[:16]  # First 4 mines from spatial_context
    
    # Buffers (4 dims): normalized to [-1, 1]
    obs[36] = ninja.jump_buffer / 5.0 if ninja.jump_buffer >= 0 else -1.0
    obs[37] = ninja.floor_buffer / 5.0 if ninja.floor_buffer >= 0 else -1.0
    obs[38] = ninja.wall_buffer / 5.0 if ninja.wall_buffer >= 0 else -1.0
    obs[39] = ninja.launch_pad_buffer / 4.0 if ninja.launch_pad_buffer >= 0 else -1.0
    
    return obs
```

### File: `nclone/nclone/gym_environment/constants.py`

Add constant:

```python
MINIMAL_OBSERVATION_DIM = 40
```

---

## Code Change 2: Minimal MLP Architecture (npp-rl)

### File: `npp-rl/npp_rl/training/architecture_configs.py`

Add new architecture config:

```python
def create_minimal_mlp_config() -> ArchitectureConfig:
    """Minimal MLP for testing time-independence hypothesis.
    
    40-dim input -> 64 -> 64 -> 6 actions
    ~6K parameters (30x reduction from graph_free)
    """
    return ArchitectureConfig(
        name="minimal_mlp",
        description="Minimal 40-dim observation with tiny MLP",
        modalities=ModalityConfig(
            use_player_frame=False,
            use_global_view=False,
            use_graph=False,
            use_game_state=False,      # Using minimal_observation instead
            use_reachability=False,
            use_spatial_context=False,
            use_minimal_observation=True,  # NEW
        ),
        state=StateConfig(
            hidden_dim=64,
            output_dim=64,
        ),
        fusion=FusionConfig(fusion_type=FusionType.CONCAT),
        features_dim=64,
    )

# Add to registry
ARCHITECTURE_REGISTRY["minimal_mlp"] = create_minimal_mlp_config()
```

### File: `npp-rl/npp_rl/feature_extractors/minimal_extractor.py` (NEW)

```python
"""Minimal feature extractor for 40-dim observation."""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

MINIMAL_OBS_DIM = 40

class MinimalMLPExtractor(BaseFeaturesExtractor):
    """Tiny MLP: 40 -> 64 -> 64 for minimal observation."""
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(MINIMAL_OBS_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations):
        return self.mlp(observations["minimal_observation"].float())
```

---

## Code Change 3: Rule-Based Controller (npp-rl)

### File: `npp-rl/scripts/rule_based_controller.py` (NEW)

```python
"""Rule-based controller using A* path guidance directly."""

import numpy as np
from typing import Tuple

def rule_based_policy(obs: dict) -> int:
    """Convert A* path guidance to action.
    
    Actions: 0=NOOP, 1=LEFT, 2=RIGHT, 3=JUMP, 4=JUMP+LEFT, 5=JUMP+RIGHT
    """
    # Extract from reachability_features
    reach = obs.get('reachability_features', obs.get('minimal_observation'))
    
    # Path direction (indices 13-14 in reachability, 12-13 in minimal)
    dir_x = reach[13] if len(reach) > 30 else reach[12]
    dir_y = reach[14] if len(reach) > 30 else reach[13]
    
    # Physics state from game_state or minimal
    game_state = obs.get('game_state', obs.get('minimal_observation'))
    airborne = game_state[7] > 0  # airborne flag
    walled = game_state[8] > 0 if len(game_state) > 8 else False
    wall_normal = game_state[9] if len(game_state) > 9 else 0
    velocity_y = game_state[1] if len(game_state) > 1 else 0
    
    # Buffer states (for jump timing)
    floor_buffer_active = game_state[37] > -0.5 if len(game_state) > 37 else not airborne
    wall_buffer_active = game_state[38] > -0.5 if len(game_state) > 38 else walled
    
    # Determine horizontal input
    hor_input = 0
    if dir_x > 0.3:
        hor_input = 1   # RIGHT
    elif dir_x < -0.3:
        hor_input = -1  # LEFT
    
    # Determine jump input
    jump_input = 0
    
    # Jump when path goes up
    if dir_y < -0.2:
        if floor_buffer_active:
            jump_input = 1  # Floor jump
        elif wall_buffer_active:
            # Wall jump: jump away from wall
            jump_input = 1
    
    # Hold jump while ascending (extend jump duration)
    if airborne and velocity_y < 0 and dir_y < 0:
        jump_input = 1
    
    # Convert to action
    return inputs_to_action(hor_input, jump_input)

def inputs_to_action(hor: int, jump: int) -> int:
    """Map (horizontal, jump) inputs to discrete action."""
    if jump:
        if hor == -1: return 4   # JUMP+LEFT
        if hor == 1: return 5    # JUMP+RIGHT
        return 3                  # JUMP
    else:
        if hor == -1: return 1   # LEFT
        if hor == 1: return 2    # RIGHT
        return 0                  # NOOP

def run_episode(env, max_steps=2000):
    """Run one episode with rule-based controller."""
    obs, _ = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        action = rule_based_policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    return total_reward, info.get('is_success', False), step + 1
```

---

## Code Change 4: MPC Controller (npp-rl)

### File: `npp-rl/scripts/mpc_controller.py` (NEW)

```python
"""Model Predictive Control using simulation rollouts."""

import numpy as np
from typing import List, Tuple, Optional
from copy import deepcopy

class MPCController:
    """MPC controller with beam search over action sequences."""
    
    def __init__(
        self,
        horizon: int = 8,           # Lookahead frames
        beam_width: int = 50,       # Top-k sequences to keep
        frame_skip: int = 4,        # Actions per decision
    ):
        self.horizon = horizon
        self.beam_width = beam_width
        self.frame_skip = frame_skip
    
    def select_action(self, sim_state, evaluate_fn) -> int:
        """Select best action via beam search.
        
        Args:
            sim_state: Clonable simulation state
            evaluate_fn: Function(state) -> float (lower = better, e.g., distance to goal)
        
        Returns:
            Best action (0-5)
        """
        # Initialize beam with all 6 actions
        beam = []
        for action in range(6):
            state_copy = deepcopy(sim_state)
            self._apply_action(state_copy, action, self.frame_skip)
            
            if self._is_dead(state_copy):
                continue  # Prune death states
            
            score = evaluate_fn(state_copy)
            beam.append((score, [action], state_copy))
        
        # Expand beam for remaining horizon
        for depth in range(1, self.horizon):
            new_beam = []
            
            for score, actions, state in beam:
                for action in range(6):
                    state_copy = deepcopy(state)
                    self._apply_action(state_copy, action, self.frame_skip)
                    
                    if self._is_dead(state_copy):
                        continue
                    
                    new_score = evaluate_fn(state_copy)
                    new_beam.append((new_score, actions + [action], state_copy))
            
            # Keep top-k
            new_beam.sort(key=lambda x: x[0])
            beam = new_beam[:self.beam_width]
            
            if not beam:
                break  # All paths lead to death
        
        if not beam:
            return 0  # Default to NOOP
        
        # Return first action of best sequence
        return beam[0][1][0]
    
    def _apply_action(self, sim, action: int, frames: int):
        """Apply action to simulation for N frames."""
        hor_input, jump_input = self._action_to_inputs(action)
        for _ in range(frames):
            sim.tick(hor_input, jump_input)
    
    def _action_to_inputs(self, action: int) -> Tuple[int, int]:
        """Convert action to (horizontal, jump) inputs."""
        mapping = {
            0: (0, 0),   # NOOP
            1: (-1, 0),  # LEFT
            2: (1, 0),   # RIGHT
            3: (0, 1),   # JUMP
            4: (-1, 1),  # JUMP+LEFT
            5: (1, 1),   # JUMP+RIGHT
        }
        return mapping[action]
    
    def _is_dead(self, sim) -> bool:
        """Check if ninja is dead."""
        return sim.ninja.state in (6, 7)  # Dead or awaiting death

def create_evaluator(level_cache, goal_id: str):
    """Create evaluation function using cached A* distances."""
    def evaluate(sim) -> float:
        ninja_pos = (int(sim.ninja.xpos), int(sim.ninja.ypos))
        # Snap to nearest graph node
        node = level_cache.get_nearest_node(ninja_pos)
        if node is None:
            return float('inf')
        return level_cache.get_geometric_distance(node, goal_id)
    return evaluate
```

---

## Code Change 5: Simplified Hyperparameter Config

### File: `npp-rl/configs/hyperparameters/minimal_baseline.json` (NEW)

```json
{
  "description": "Minimal baseline config: no state stacking, no RND, simple curriculum",
  "architecture": "minimal_mlp",
  "hardware": "gh200",
  "num_envs": 64,
  "ppo_hyperparameters": {
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 4096,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5
  },
  "architecture_hyperparameters": {
    "net_arch_depth": 2,
    "net_arch_width": 64,
    "features_dim": 64
  },
  "lstm_hyperparameters": {
    "recurrent_type": "none"
  },
  "exploration_features": {
    "enable_rnd": false,
    "enable_go_explore": false
  },
  "reward_config_overrides": {
    "enable_path_waypoints": false
  },
  "action_guidance": {
    "enable_action_guidance": false,
    "enable_straightness_masking": false,
    "enable_safety_critic": false
  },
  "goal_curriculum_config": {
    "enabled": false
  }
}
```

---

## Summary of Files to Create/Modify

| File | Action | Description |

|------|--------|-------------|

| `nclone/gym_environment/config.py` | Modify | Add `ObservationMode` enum |

| `nclone/gym_environment/observation_processor.py` | Modify | Add `compute_minimal_observation()` |

| `nclone/gym_environment/constants.py` | Modify | Add `MINIMAL_OBSERVATION_DIM` |

| `npp-rl/training/architecture_configs.py` | Modify | Add `minimal_mlp` config |

| `npp-rl/feature_extractors/minimal_extractor.py` | Create | `MinimalMLPExtractor` class |

| `npp-rl/scripts/rule_based_controller.py` | Create | Rule-based policy script |

| `npp-rl/scripts/mpc_controller.py` | Create | MPC controller script |

| `npp-rl/configs/hyperparameters/minimal_baseline.json` | Create | Simplified config |