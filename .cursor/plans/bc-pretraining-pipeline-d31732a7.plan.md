<!-- d31732a7-db7e-4e1d-ac5f-7ae677cef372 01319439-7f04-4e34-b984-585adde1a7c4 -->
# BC Pretraining Pipeline Implementation

## Overview

Implement complete behavioral cloning pretraining system that:

- Loads compact replay files from `nclone/bc_replays` 
- Regenerates observations deterministically using `ReplayExecutor`
- Trains policy networks via behavioral cloning
- Saves checkpoints compatible with RL fine-tuning
- Integrates with `train_and_compare.py` workflow

## Critical Issues Identified

### 1. Missing BC Dataset (HIGH PRIORITY)

**File**: `npp_rl/training/bc_dataset.py` (doesn't exist)

- No PyTorch Dataset class for BC training
- Need to load `.replay` files using `CompactReplay.from_binary()`
- Must use `ReplayExecutor` to regenerate observations on-demand
- Should support caching processed observations for efficiency

**Implementation needed**:

```python
class BCReplayDataset(Dataset):
    - Load all .replay files from directory
    - Parse with CompactReplay.from_binary()
    - Use ReplayExecutor to generate (obs, action) pairs
    - Cache processed data as NPZ files
    - Handle multimodal observations (vision, graph, state)
```

### 2. Stub Pretraining Implementation (HIGH PRIORITY)

**File**: `npp_rl/training/pretraining_pipeline.py`

- Lines 104-163: `run_pretraining()` is a stub that logs warnings
- Lines 65-102: `prepare_bc_data()` doesn't process replay files
- Missing actual BC training loop

**Required changes**:

- `prepare_bc_data()`: Scan for `.replay` files, create BCReplayDataset
- `run_pretraining()`: Implement full BC training loop with loss calculation

### 3. Missing BC Trainer Module (HIGH PRIORITY)

**File**: `npp_rl/training/bc_trainer.py` (doesn't exist, referenced in README)

- Need standalone BC trainer for flexibility
- Should work independently and be callable from pipeline
- Must create policy from ArchitectureConfig
- Implement BC loss (negative log-likelihood)
- Save checkpoints in format compatible with RL loading

### 4. Checkpoint Format Mismatch (MEDIUM PRIORITY)

**File**: `npp_rl/training/architecture_trainer.py:389-409`

- Expects checkpoint with key `"policy_state_dict"`
- Need to ensure BC trainer saves in this exact format
- Validation in `pretraining_pipeline.py:188` checks for this key

### 5. Policy Creation for BC (MEDIUM PRIORITY)

- BC trainer needs to instantiate policy without full PPO model
- Must use `ConfigurableMultimodalExtractor` with ArchitectureConfig
- Should match exact architecture that RL training will use
- Need utility to create standalone policy network

## Implementation Plan

### Step 1: Create BC Dataset Class

**New file**: `npp_rl/training/bc_dataset.py` (~200 lines)

Core functionality:

- Load `.replay` files from directory
- Parse using `CompactReplay.from_binary()` 
- Execute replays with `ReplayExecutor` to get observations
- Cache processed data to disk as NPZ files
- Return (observation, action) tuples for training
- Support data augmentation and filtering

Key classes:

```python
class BCReplayDataset(Dataset):
    def __init__(self, replay_dir, cache_dir, architecture_config)
    def _load_replay_files(self) -> List[Path]
    def _process_replay_file(self, replay_path) -> List[Tuple]
    def _cache_processed_data(self, data, cache_path)
    def __getitem__(self, idx) -> Tuple[Dict, int]
    def __len__(self) -> int
```

### Step 2: Create BC Trainer Module  

**New file**: `npp_rl/training/bc_trainer.py` (~350 lines)

Core functionality:

- Create policy network from ArchitectureConfig
- Load BCReplayDataset with DataLoader
- Training loop with BC loss (negative log-likelihood)
- Validation loop and metrics tracking
- Checkpoint saving with correct format
- TensorBoard logging integration
- Early stopping support

Key components:

```python
class BCTrainer:
    def __init__(self, architecture_config, dataset, output_dir, device)
    def _create_policy(self) -> nn.Module
    def train_epoch(self, epoch) -> Dict[str, float]
    def validate(self) -> Dict[str, float]
    def save_checkpoint(self, path, metrics)
    def train(self, epochs, batch_size, lr) -> str
```

BC Loss implementation:

```python
def compute_bc_loss(policy, observations, actions):
    # Forward through policy to get action distribution
    # Compute negative log-likelihood of expert actions
    # Return loss + metrics
```

### Step 3: Update Pretraining Pipeline

**File**: `npp_rl/training/pretraining_pipeline.py`

Changes needed:

- `prepare_bc_data()` (lines 65-102): Create BCReplayDataset instance
- `run_pretraining()` (lines 104-163): Call BCTrainer with proper config
- Add method to validate BC checkpoint can be loaded by RL trainer
- Improve error handling and logging

### Step 4: Create Policy Utilities

**File**: `npp_rl/training/policy_utils.py` (new, ~150 lines)

Utilities for policy creation outside full PPO:

```python
def create_policy_from_config(
    observation_space,
    action_space, 
    architecture_config
) -> nn.Module

def create_action_distribution_head(
    features_dim,
    action_space
) -> nn.Module

def save_policy_checkpoint(
    policy,
    path,
    metadata=None
)

def load_policy_checkpoint(
    policy,
    path
) -> Dict
```

### Step 5: Integration Testing

- Test loading `.replay` files from `nclone/bc_replays`
- Verify observations match environment format
- Test BC training produces decreasing loss
- Validate checkpoint format matches RL expectations
- Test end-to-end: BC pretrain → load in RL → fine-tune

### Step 6: Documentation Updates

**File**: `npp_rl/training/README.md` or `docs/BC_PRETRAINING.md`

Document:

- How to prepare replay data
- Running BC pretraining standalone
- Running with `train_and_compare.py`
- Troubleshooting checkpoint loading issues
- Performance considerations and caching

## Files to Create

1. `npp_rl/training/bc_dataset.py` - BC dataset loader
2. `npp_rl/training/bc_trainer.py` - BC training module  
3. `npp_rl/training/policy_utils.py` - Policy creation utilities

## Files to Modify

1. `npp_rl/training/pretraining_pipeline.py` - Replace stubs with real implementation
2. `npp_rl/training/__init__.py` - Export new classes

## Testing Strategy

1. Unit test: `BCReplayDataset` loads and processes replay files correctly
2. Unit test: `BCTrainer` trains and saves checkpoints properly
3. Integration test: BC checkpoint loads into PPO model
4. End-to-end test: BC pretrain → RL fine-tune pipeline
5. Validation: Compare BC-pretrained vs random initialization performance

## Key Design Decisions

- Use compact replay format only (as requested)
- Cache processed observations to avoid repeated execution
- Support all architectures via ArchitectureConfig
- Checkpoint format compatible with SB3 PPO loading
- Separate BC trainer allows standalone use

### To-dos

- [ ] Create BCReplayDataset class in npp_rl/training/bc_dataset.py to load compact replay files and generate training data
- [ ] Create BCTrainer class in npp_rl/training/bc_trainer.py with training loop, loss calculation, and checkpoint saving
- [ ] Create policy_utils.py with utilities for creating standalone policy networks from architecture configs
- [ ] Update pretraining_pipeline.py to replace stub implementations with actual BC training integration
- [ ] Test end-to-end BC pretraining pipeline with nclone/bc_replays data and verify checkpoint loading in RL training