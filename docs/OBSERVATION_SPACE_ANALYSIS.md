# N++ Observation Space Analysis and Optimization

## Overview

This document analyzes the current 30-feature ninja state observation space for redundancy and opportunities for optimization. While all features are currently used, some contain information that can be derived from other features, making them redundant from a Markov Decision Process perspective.

## Current Feature Space (30 Features)

### Core Movement State (Features 0-7)
| Index | Feature | Markov Essential | Notes |
|-------|---------|------------------|-------|
| 0 | velocity_magnitude | ✅ Yes | Affects next state transition |
| 1-2 | velocity_direction (x, y) | ✅ Yes | Direction of movement |
| 3 | ground_movement | ⚠️ Partial | Redundant with airborne_status |
| 4 | air_movement | ⚠️ Partial | Redundant with airborne_status |
| 5 | wall_interaction | ✅ Yes | Distinct state affecting physics |
| 6 | special_states | ✅ Yes | Death/victory terminal states |
| 7 | airborne_status | ✅ Yes | Critical for physics state |

**Analysis**: Features 3-4 are categorical encodings that partially overlap with feature 7. The categorical encoding may help neural network learning, so keeping them provides a tradeoff between redundancy and learning efficiency.

### Input and Buffer State (Features 8-12)
| Index | Feature | Markov Essential | Notes |
|-------|---------|------------------|-------|
| 8 | horizontal_input | ✅ Yes | Current action effect |
| 9 | jump_input | ✅ Yes | Current action effect |
| 10 | jump_buffer | ✅ Yes | N++ temporal mechanic |
| 11 | floor_buffer | ✅ Yes | N++ temporal mechanic |
| 12 | wall_buffer | ✅ Yes | N++ temporal mechanic |

**Analysis**: All features are essential. Buffer mechanics are unique to N++ and cannot be inferred from other features.

### Surface Contact Information (Features 13-18)
| Index | Feature | Markov Essential | Notes |
|-------|---------|------------------|-------|
| 13 | floor_contact | ✅ Yes | Affects physics calculations |
| 14 | wall_contact | ✅ Yes | Affects physics calculations |
| 15 | ceiling_contact | ✅ Yes | Affects physics calculations |
| 16 | floor_normal_strength | ⚠️ Partial | Magnitude; somewhat redundant with floor_contact |
| 17 | wall_direction | ✅ Yes | Left vs right wall distinction |
| 18 | surface_slope | ✅ Yes | Affects movement physics |

**Analysis**: Feature 16 provides magnitude information beyond feature 13's binary contact. Useful for slopes and edges, so keep despite partial redundancy.

### Momentum and Physics (Features 19-22)
| Index | Feature | Markov Essential | Notes |
|-------|---------|------------------|-------|
| 19-20 | acceleration (x, y) | ✅ Yes | Change in velocity, useful for learning |
| 21 | momentum_preservation | ❌ No | Derived: dot product of current/previous velocity |
| 22 | impact_risk | ❌ No | Derived: f(velocity, surface_contact) |

**Analysis**: 
- Features 21-22 are **explicitly computed** from other features in the state
- `momentum_preservation = dot(velocity, velocity_old) / (|v| * |v_old|)` - derivable
- `impact_risk = velocity_mag if (floor or ceiling contact) else 0` - derivable
- **Recommendation**: HIGH PRIORITY for removal. Pure redundancy.

### Entity Proximity and Hazards (Features 23-26)
| Index | Feature | Markov Essential | Notes |
|-------|---------|------------------|-------|
| 23 | nearest_hazard_distance | ✅ Yes | Critical for mine avoidance |
| 24 | nearest_collectible_distance | ⚠️ Partial | May be visible in player_frame |
| 25 | hazard_threat_level | ❌ No | Derived: exp(-distance / decay_factor) |
| 26 | interaction_cooldown | ⚠️ Wrong Data | Currently uses jump_duration as proxy |

**Analysis**:
- Feature 23: Essential - mines are critical hazards
- Feature 24: Possibly redundant with visual observations, but useful for attention
- Feature 25: **Explicitly derived** from feature 23 via exponential decay
- Feature 26: **Currently incorrect** - uses jump_duration instead of actual entity interaction tracking
- **Recommendation**: Remove feature 25 (derived). Fix or remove feature 26.

### Level Progress and Objectives (Features 27-29)
| Index | Feature | Markov Essential | Notes |
|-------|---------|------------------|-------|
| 27 | switch_progress | ⚠️ Partial | Derived from distance to switch |
| 28 | exit_accessibility | ✅ Yes | Binary state: switch activated or not |
| 29 | completion_progress | ❌ No | Derived: (switch_progress + exit_progress) / 2 |

**Analysis**:
- Feature 27: Computed as `1 - (switch_dist / screen_diagonal)` - derivable from positions
- Feature 28: **Essential Markov state** - boolean whether switch is activated
- Feature 29: **Explicitly computed** as `(features[27] + exit_progress) / 2`
- **Recommendation**: Keep only feature 28. Remove 27 and 29.

## Redundancy Summary

### High Priority Removals (Explicitly Derived)
These features are **computed directly from other features** with no new information:

1. **Feature 21 (momentum_preservation)**: `dot(v, v_old) / (|v| * |v_old|)`
2. **Feature 22 (impact_risk)**: `velocity_mag if contact else 0`
3. **Feature 25 (hazard_threat_level)**: `exp(-nearest_hazard_dist / 100)`
4. **Feature 29 (completion_progress)**: `(switch_progress + exit_progress) / 2`

**Impact**: Remove 4 features → 30 to 26 features (-13% dimensionality)

### Medium Priority Optimizations

1. **Feature 26 (interaction_cooldown)**: Currently uses jump_duration as proxy instead of tracking actual entity interactions. Either fix to track real interactions or remove.

2. **Feature 27 (switch_progress)**: Derived from distance calculation `1 - (switch_dist / diagonal)`. Consider removal if position information is sufficient.

### Low Priority (Keep Despite Partial Redundancy)

1. **Features 3-4 (ground_movement, air_movement)**: Categorical encoding helps learning despite overlap with airborne_status
2. **Feature 16 (floor_normal_strength)**: Provides magnitude beyond binary contact
3. **Feature 24 (nearest_collectible_distance)**: Useful attention signal despite visual redundancy

## Implementation Plan

### Phase 1: Remove Explicit Derivations (Safe)
Remove features 21, 22, 25, 29 as they are pure mathematical combinations of other features:

**Files to modify**:
- `/workspace/nclone/nclone/nplay_headless.py::get_ninja_state()` - Remove computations
- `/workspace/nclone/nclone/gym_environment/observation_processor.py::process_game_state()` - Update feature indexing
- `/workspace/nclone/nclone/replay/replay_executor.py` - Update padding
- `/workspace/nclone/nclone/replay/unified_observation_extractor.py` - Update documentation

**Testing**: 
- Run full test suite to ensure observation space compatibility
- Verify neural network input shape compatibility
- Retrain or fine-tune models with new 26-feature state

### Phase 2: Fix Incorrect Feature (Medium Priority)
Fix feature 26 (interaction_cooldown) to track actual entity interactions instead of using jump_duration as proxy, or remove if not needed.

### Phase 3: Consider Further Optimization (Optional)
After validating Phase 1 performance, consider removing feature 27 (switch_progress) if distance-based features prove sufficient.

## Benefits of Optimization

1. **Reduced dimensionality**: 26 features instead of 30 (-13%)
2. **Cleaner state space**: Remove mathematical redundancy
3. **Faster training**: Fewer parameters in policy/value networks
4. **Better generalization**: Less opportunity for overfitting on derived features
5. **Clearer Markov properties**: State space contains only essential information

## Risks and Mitigations

**Risk**: Removing features may hurt learning if network was relying on precomputed features
**Mitigation**: The removed features are simple mathematical combinations that modern neural networks can easily learn internally

**Risk**: Changes affect existing trained models
**Mitigation**: Treat as architecture change, requiring retraining. Document as v2 observation space.

**Risk**: Test suite failures due to hardcoded feature counts
**Mitigation**: Update all references to 30-feature space systematically

## References

- Feature definitions: `/workspace/nclone/nclone/nplay_headless.py::get_ninja_state()`
- Feature processing: `/workspace/nclone/nclone/gym_environment/observation_processor.py`
- Usage in replay system: `/workspace/nclone/nclone/replay/unified_observation_extractor.py`

## Decision

**Current Status**: DOCUMENTED - Redundancies identified and analyzed

**Recommendation**: Implement Phase 1 (remove features 21, 22, 25, 29) in next major version update after validating current architecture performance. This provides 13% dimensionality reduction with minimal risk.

**Version Planning**:
- v1.0 (current): 30-feature state space
- v2.0 (proposed): 26-feature state space with redundancy removal
