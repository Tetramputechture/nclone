<!-- becd5651-522c-4975-b784-075b63228678 50de7e77-0ad6-4a34-a9dd-d0b5de751c4d -->
# Phase 3: MAML System for Fixed-Size Level Topology Adaptation

## Strategic Foundation

**Key Insight**: Since physics constants remain consistent and all levels are exactly **44x25 tiles (1056x600 pixels)**, our MAML system should focus purely on **tile configuration pattern adaptation**. This creates an optimal MAML scenario - consistent physics and spatial scale with variable topology patterns.

**Architecture Strategy**:

- **Fixed Spatial Scale**: All levels are exactly 44x25 tiles (1056x600 pixels) - no size adaptation needed
- **Topology Pattern Adaptation**: Focus on tile configuration patterns, connectivity, and reachable area utilization  
- **Preserve Physics Knowledge**: Keep AttentiveStateMLP physics processing frozen - physics constants never change
- **Few-Shot Deployment**: Target 1-3 gradient steps for rapid adaptation to new tile layouts

## 1. MAML-Enhanced Architecture Design

### 1.1 Topology Adaptation Modules

Create dedicated adaptation components for fixed-size spatial pattern recognition:

```python
class TopologyAdaptationModule(nn.Module):
    """Learns tile configuration patterns for MAML adaptation on fixed 44x25 grids."""
    # Fast adaptation parameters for:
    # - Connectivity patterns (narrow corridors vs open areas)
    # - Reachable area density (compact vs distributed layouts)
    # - Mine placement strategies (blocking choke points vs scattered)
    # - Path complexity (linear routes vs complex branching)
```

### 1.2 Meta-Policy Architecture

Extend ObjectiveAttentionActorCriticPolicy with topology-specific MAML components:

```python
class MAMLObjectiveAttentionPolicy(ObjectiveAttentionActorCriticPolicy):
    """MAML-enhanced policy with topology adaptation capabilities."""
    # Adaptation targets:
    # - Graph attention weights (connectivity pattern understanding)
    # - Objective attention parameters (route prioritization for specific layouts)
    # - Spatial feature projections (tile pattern-specific features)
```

### 1.3 Hierarchical Adaptation Strategy

**Freeze Physics Components**: AttentiveStateMLP physics processing (velocity, forces, contact) remains frozen

**Adapt Topology Components**: Graph neural networks, objective attention, and spatial reasoning modules

**Meta-Parameters**: ~80K parameters (1-3% of total network) for efficient topology adaptation

## 2. MAML Training Infrastructure

### 2.1 Episode Sampling Strategy

Design task distribution covering tile configuration variations on fixed 44x25 grids:

- **Mine Density Classes**: Sparse (0-50), Medium (51-150), Dense (151-256)
- **Connectivity Patterns**: Narrow corridors, open chambers, mixed layouts
- **Reachable Area Distribution**: Compact central areas, distributed zones, linear paths
- **Mine Strategic Placement**: Choke point blocking, scattered placement, cluster formations
- **Path Complexity**: Direct routes, maze-like branching, multi-level traversal

### 2.2 Meta-Learning Algorithm

Implement first-order MAML optimized for topology patterns:

- **Inner Loop**: 1-3 gradient steps on 15-25 episodes from target topology type
- **Outer Loop**: Meta-gradient update across diverse tile configurations  
- **Support Set**: 15-25 episodes per topology pattern for adaptation
- **Query Set**: 40-60 episodes for meta-gradient computation

### 2.3 Task Construction Pipeline

```python
class TopologyMetaTaskSampler:
    """Generates meta-learning tasks from tile configuration variations."""
    # Creates task batches with:
    # - Fixed 44x25 dimensions
    # - Variable tile patterns and connectivity
    # - Balanced mine densities and placement strategies
```

## 3. Spatial Feature Enhancement

### 3.1 Fixed-Scale Graph Processing

Enhance existing graph neural networks with topology adaptation:

- **Adaptive Edge Weights**: Meta-learned edge importance based on connectivity patterns
- **Topology Context Encoding**: Mine density, connectivity metrics, reachable area patterns
- **Multi-Resolution Reasoning**: Local tile patterns + global connectivity structure

### 3.2 Mine Pattern Recognition

Develop specialized modules for mine configuration understanding:

- **Choke Point Detection**: Identify mines blocking critical navigation paths
- **Cluster Analysis**: Recognize grouped vs scattered mine patterns  
- **Strategic Placement Recognition**: Detect mines requiring specific toggle sequences

## 4. Implementation Architecture

### 4.1 Core MAML Components

```
npp_rl/meta_learning/
├── maml_trainer.py              # Meta-training orchestration
├── topology_adaptation.py      # Topology pattern adaptation modules  
├── topology_sampler.py         # Tile configuration task generation
├── meta_policy.py              # MAML-enhanced policy architecture
└── adaptation_metrics.py       # Topology adaptation measurement
```

### 4.2 Training Pipeline Integration

#### 4.2.1 MAML Trainer Extension

Extend existing `ArchitectureTrainer` to support meta-learning:

```python
class MAMLArchitectureTrainer(ArchitectureTrainer):
    """MAML-enhanced trainer extending existing training pipeline."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_sampler = None
        self.meta_batch_size = 8  # Number of tasks per meta-batch
        self.inner_lr = 1e-3      # Inner loop learning rate
        self.adaptation_steps = 3  # Inner loop gradient steps
        
    def setup_meta_learning(self, meta_config: Dict[str, Any]):
        """Initialize MAML-specific components."""
        # Task sampler for generating topology variations
        self.task_sampler = TopologyMetaTaskSampler(
            train_dataset_path=self.train_dataset_path,
            meta_batch_size=self.meta_batch_size
        )
        
        # Replace standard policy with MAML-enhanced version
        self.policy_class = MAMLObjectiveAttentionPolicy
```

#### 4.2.2 Policy Architecture Integration

Extend existing `ObjectiveAttentionActorCriticPolicy` with MAML capabilities:

```python
class MAMLObjectiveAttentionPolicy(ObjectiveAttentionActorCriticPolicy):
    """MAML-enhanced policy maintaining existing architecture."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add topology adaptation modules
        self.topology_adapter = TopologyAdaptationModule()
        # Mark which parameters are adaptation targets
        self._setup_adaptation_parameters()
        
    def fast_adapt(self, support_episodes: List, num_steps: int = 3):
        """Perform inner loop adaptation on support episodes."""
        # Clone adaptation parameters
        adapted_params = self._clone_adaptation_params()
        # Gradient steps on support set
        for step in range(num_steps):
            loss = self._compute_adaptation_loss(support_episodes, adapted_params)
            adapted_params = self._gradient_update(adapted_params, loss)
        return adapted_params
```

#### 4.2.3 Environment Factory Extension

Extend existing `EnvironmentFactory` for task-specific environments:

```python
class MAMLEnvironmentFactory(EnvironmentFactory):
    """Task-aware environment factory for MAML training."""
    
    def create_task_env(self, task_config: Dict, num_envs: int = 16):
        """Create environment for specific topology task."""
        # Override level selection with task-specific levels
        return self._create_vectorized_env(
            num_envs=num_envs,
            level_filter=task_config['level_filter'],
            topology_pattern=task_config['topology_pattern']
        )
```

#### 4.2.4 Training Loop Integration

Modify existing training loop to support meta-learning:

```python
def meta_train(self, total_meta_episodes: int):
    """Main meta-training loop integrated with existing pipeline."""
    
    for meta_episode in range(total_meta_episodes):
        # Sample meta-batch of topology tasks
        task_batch = self.task_sampler.sample_meta_batch()
        
        # Inner loop: adapt to each task
        adapted_policies = []
        for task in task_batch:
            # Create task-specific environment
            task_env = self.environment_factory.create_task_env(task)
            
            # Collect support episodes
            support_episodes = self._collect_episodes(task_env, num_episodes=15)
            
            # Fast adaptation (1-3 gradient steps)
            adapted_policy = self.model.policy.fast_adapt(
                support_episodes, num_steps=self.adaptation_steps
            )
            adapted_policies.append(adapted_policy)
            
        # Outer loop: meta-update across all tasks
        self._meta_update(task_batch, adapted_policies)
        
        # Existing evaluation and checkpointing infrastructure
        if meta_episode % self.eval_freq == 0:
            self._evaluate_meta_performance()
```

### 4.3 Evaluation Integration

Extend existing `ComprehensiveEvaluator` for meta-learning evaluation:

```python
class MAMLEvaluator(ComprehensiveEvaluator):
    """MAML-aware evaluator extending existing evaluation pipeline."""
    
    def evaluate_few_shot_adaptation(self, model, adaptation_steps: List[int] = [1, 3, 5]):
        """Evaluate few-shot adaptation performance."""
        results = {}
        
        for num_steps in adaptation_steps:
            # Test adaptation to novel topology patterns
            adaptation_results = self._test_adaptation(model, num_steps)
            results[f"adaptation_{num_steps}_steps"] = adaptation_results
            
        return results
```

### 4.4 Training Phase Integration

Seamless integration with existing training phases:

1. **Phase 1**: Use existing `ArchitectureTrainer` for base policy pre-training
2. **Phase 2**: Switch to `MAMLArchitectureTrainer` for meta-learning
3. **Phase 3**: Use existing evaluation infrastructure with MAML extensions
```python
# Training pipeline integration example
def run_maml_training():
    # Phase 1: Base policy training (existing pipeline)
    base_trainer = ArchitectureTrainer(...)
    base_trainer.setup_model()
    base_trainer.setup_environments(num_envs=64)
    base_trainer.train(total_timesteps=1_000_000)
    
    # Phase 2: Meta-learning (new MAML pipeline)
    maml_trainer = MAMLArchitectureTrainer(...)
    maml_trainer.load_base_policy(base_trainer.model)
    maml_trainer.setup_meta_learning(meta_config)
    maml_trainer.meta_train(total_meta_episodes=10_000)
    
    # Phase 3: Evaluation (extended existing pipeline)
    evaluator = MAMLEvaluator(...)
    results = evaluator.evaluate_few_shot_adaptation(maml_trainer.model)
```


## 5. MAML Deployment & Inference Integration

### 5.1 Meta-Knowledge Storage & Loading

The MAML system produces a **meta-model** that stores fast adaptation capabilities:

```python
class MAMLDeploymentModel(nn.Module):
    """Production model with embedded MAML adaptation capabilities."""
    
    def __init__(self, base_policy: ObjectiveAttentionActorCriticPolicy):
        super().__init__()
        self.base_policy = base_policy
        
        # Store meta-learned initialization parameters
        self.meta_parameters = self._extract_meta_params(base_policy)
        
        # Fast adaptation components
        self.adaptation_optimizer = torch.optim.SGD(
            self.meta_parameters, lr=1e-3
        )
```

### 5.2 Runtime Fast Adaptation Process

When the agent encounters a new level, it performs fast adaptation before play:

```python
class AdaptiveAgent:
    """Production agent with MAML fast adaptation."""
    
    def adapt_to_level(self, level_data: Dict, num_adaptation_steps: int = 3):
        """Fast adaptation to new level topology."""
        
        # 1. Generate support episodes through quick exploration
        support_episodes = self._collect_support_episodes(
            level_data, num_episodes=15, exploration_mode=True
        )
        
        # 2. Fast gradient updates on support episodes
        adapted_params = self.meta_model.fast_adapt(
            support_episodes, num_steps=num_adaptation_steps
        )
        
        # 3. Update policy with adapted parameters
        self._apply_adapted_parameters(adapted_params)
        
        return adapted_params
```

### 5.3 Parameter Integration Architecture

The adapted parameters directly modify specific components of the policy:

```python
class MAMLObjectiveAttentionPolicy(ObjectiveAttentionActorCriticPolicy):
    """Policy with embedded adaptation capabilities."""
    
    def _setup_adaptation_parameters(self):
        """Mark parameters that can be adapted."""
        self.adaptable_params = {
            # Graph attention weights (topology understanding)
            'graph_attention_weights': self.graph_processor.attention_weights,
            
            # Objective attention parameters (route prioritization)  
            'objective_attention_keys': self.action_net.attention.key_proj,
            
            # Spatial feature projections (layout-specific features)
            'spatial_projections': self.features_extractor.spatial_projection_layers
        }
        
        # Physics processing remains FROZEN
        self.frozen_params = {
            'physics_state_mlp': self.features_extractor.state_mlp,
            'physics_components': self.features_extractor.attentive_state_mlp
        }
    
    def apply_adapted_parameters(self, adapted_params: Dict[str, torch.Tensor]):
        """Apply fast-adapted parameters to policy."""
        for param_name, new_values in adapted_params.items():
            if param_name in self.adaptable_params:
                # Update adaptable parameters
                self.adaptable_params[param_name].data = new_values
            # Physics parameters remain unchanged
```

### 5.4 Production Deployment Workflow

Complete workflow for deploying MAML-trained agents:

```python
# 1. Load meta-trained model
meta_model = MAMLDeploymentModel.load("maml_trained_model.pt")

# 2. Create adaptive agent
agent = AdaptiveAgent(meta_model)

# 3. For each new level:
def play_level(level_data: Dict) -> float:
    # Fast adaptation (1-3 seconds)
    agent.adapt_to_level(level_data, num_adaptation_steps=3)
    
    # Play level with adapted policy
    success = agent.play_level(level_data)
    return success
```

## 6. Evaluation Framework

### 5.1 Adaptation Efficiency Metrics

- **Few-Shot Success Rate**: Performance after 1, 2, 3 adaptation steps
- **Topology Generalization**: Success on novel tile configuration patterns  
- **Mine Pattern Transfer**: Adaptation to unseen mine placement strategies
- **Path Discovery Efficiency**: Route planning speed on new connectivity patterns

### 5.2 Robustness Testing

- **Extreme Connectivity**: Minimal paths, highly branched layouts
- **Mine Saturation**: Near-maximum mine counts (200-256 mines) in strategic positions
- **Pathological Patterns**: Forced mine interaction sequences, trap-like configurations

## 6. Computational Optimization

### 6.1 Efficient Meta-Updates

- **First-Order MAML**: Ignore second-order derivatives for 3x speedup
- **Selective Adaptation**: Only adapt topology components (~80K params)
- **Batch Meta-Training**: Parallel task sampling and gradient computation

### 6.2 Memory Management

- **Gradient Checkpointing**: Reduce memory usage during meta-gradient computation
- **Topology Cache**: Cache tile configuration analyses for consistent sampling
- **Pattern-Based Batching**: Group similar topology patterns for efficient processing

## Expected Outcomes

### Performance Targets

- **Rapid Adaptation**: 85%+ success rate within 3 gradient steps on new topology patterns
- **Pattern Generalization**: Robust performance on novel tile configurations with same mine density
- **Strategic Transfer**: Effective mine interaction strategies across different placement patterns

### System Benefits

- **Instant Deployment**: Ultra-fast adaptation to new level packs with same dimensions
- **Consistent Physics Exploitation**: Preserved low-level movement skills across all adaptations
- **Pattern Recognition**: Efficient identification and exploitation of topology-specific strategies

## Success Criteria

- Few-shot adaptation (≤3 steps) achieves 85% of full-training performance
- Successful generalization to completely novel tile configuration patterns
- Effective strategic mine interaction on configurations with 200+ mines
- Maintained physics exploitation performance after topology adaptation

### To-dos

- [ ] Create PhysicsContextProvider class in nclone/nclone/graph/physics_context.py
- [ ] Add physics context to NodeFeatureBuilder in feature_builder.py
- [ ] Add physics context to EdgeFeatureBuilder in feature_builder.py
- [ ] Enhance MultiHeadFusion with physics-aware attention in configurable_extractor.py
- [ ] Create PhysicsContextInjector module in configurable_extractor.py
- [ ] Integrate physics context into graph building pipeline in graph_mixin.py
- [ ] Test physics-enhanced features across all architecture variants