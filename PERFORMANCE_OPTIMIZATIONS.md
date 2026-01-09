# Environment Performance Optimizations

## Summary

This document describes the performance optimizations applied to the N++ RL environment based on profiling data from a 4000-step training run.

## Baseline Performance

- **Total training time**: 329.8s
- **Rollout collection**: 320.5s (92.3%)
- **Environment step time**: 26.03 ms/step (38.4 steps/s)
- **Theoretical physics-only limit**: ~0.33 ms/step (3000 steps/s)
- **Slowdown factor**: ~75x

## Bottleneck Analysis

### Per-Step Breakdown (26.03ms total)

| Component | Time (ms) | % of Step | Issue |
|-----------|-----------|-----------|-------|
| Reachability features | 7.13 | 27.4% | **Duplicate path distance computations** |
| Reward calculation | 7.05 | 27.1% | **Duplicate path distance computations** |
| Spatial context | 2.28 | 8.8% | Mine overlay computation |
| Base observation | 2.45 | 9.4% | - |
| Process observation | 2.37 | 9.1% | - |
| Physics tick | 0.33 | 1.3% | Efficient ✓ |

### Episode-End Spike

- **Convert to GraphData**: 93.6ms average (max: 497.2ms)
- Called ~40 times per 4000 steps (episode ends)
- Caused severe timing spikes (204ms → 217ms max step times)

## Optimizations Implemented

### 1. Share Path Distances Between Observation and Reward (HIGH IMPACT)

**Problem**: Both reachability features and PBRS reward calculation independently computed distances to switch/exit using the same `CachedPathDistanceCalculator`.

**Solution**: 
- Modified `compute_reachability_features_from_graph()` to return `(features, switch_distance_raw, exit_distance_raw)`
- Store distances in observation dict: `obs["_cached_switch_distance"]`, `obs["_cached_exit_distance"]`
- Modified PBRS `objective_distance_potential()` to check for cached distances before recomputing

**Expected Savings**: ~3.5ms per step (13.4% reduction)

**Files Modified**:
- `nclone/graph/reachability/feature_computation.py`
- `nclone/gym_environment/mixins/reachability_mixin.py`
- `nclone/gym_environment/npp_environment.py`
- `nclone/gym_environment/reward_calculation/pbrs_potentials.py`
- `nclone/replay/replay_executor.py` (compatibility)
- `nclone/replay/unified_observation_extractor.py` (compatibility)

### 2. Add Profiling Infrastructure for Reachability Features

**Purpose**: Enable fine-grained profiling of individual reachability feature computations to identify further optimization opportunities.

**Implementation**:
- Added optional profiling with `ENABLE_FEATURE_PROFILING` flag
- Tracks timing for: spatial_lookup, reachable_area, area_scale, switch_distances, exit_distances, directional_connectivity
- Functions: `get_feature_profiling_stats()`, `reset_feature_profiling()`

**Usage**:
```python
# In feature_computation.py, set:
ENABLE_FEATURE_PROFILING = True

# After training run:
from nclone.graph.reachability.feature_computation import get_feature_profiling_stats
stats = get_feature_profiling_stats()
```

**Files Modified**:
- `nclone/graph/reachability/feature_computation.py`

### 3. Fix GraphData Conversion Spike (HIGH IMPACT)

**Problem**: `_convert_graph_data_to_graphdata()` was rebuilding the entire GraphData structure on every call, even though the graph structure (nodes, edges) is static per level.

**Solution**:
- Improved caching logic to return cached GraphData immediately if level_id matches
- Only rebuild GraphData on first encounter of a level
- Graph structure is static; entity state changes are handled by the ML model, not the graph structure

**Expected Savings**: ~93.6ms per episode end × 40 episodes = ~3.7s total (1.1% of training time)

**Files Modified**:
- `nclone/gym_environment/mixins/graph_mixin.py`

### 4. Reachability Feature Dimensionality Analysis

**Current State**: 38 dimensions
- 4 base features (area ratio, distances, reachability)
- 2 path distances (raw normalized)
- 4 direction vectors (Euclidean to goals)
- 2 mine context (count, deadly ratio)
- 1 phase indicator (switch_activated)
- 8 path direction features (next_hop, waypoint, detour flag, mine clearance)
- 1 path difficulty (physics/geometric ratio)
- 3 path curvature (multi-hop lookahead and curvature)
- 5 exit lookahead (exit next_hop, exit multi-hop, near-switch indicator)
- 8 directional connectivity (platform distances in 8 directions)

**Conclusion**: All 38 dimensions are information-rich and used by the model. The features encode:
- A* optimal path hints (high-value for policy learning)
- Angular relationships (curvature)
- Compositional reasoning (two simultaneous goals)
- Physics awareness (blind jump verification)

**Recommendation**: Keep all 38 dimensions. The model uses a 3-layer MLP (38→128→128→96) with dropout to process these features, which is appropriate for their complexity.

## Expected Performance Improvement

### Conservative Estimate

| Optimization | Savings per Step | Savings per 4000 Steps |
|--------------|------------------|------------------------|
| Shared path distances | 3.5ms | 14.0s |
| GraphData caching | 0.09ms (amortized) | 3.7s |
| **Total** | **3.59ms** | **17.7s** |

**New step time**: 26.03ms - 3.59ms = **22.44ms/step** (44.6 steps/s)
**Speedup**: 13.8% faster rollout collection

### Optimistic Estimate

If cache hit rates improve and variance decreases:
- Reachability features: 7.13ms → 5.5ms (better caching)
- Reward calculation: 7.05ms → 5.0ms (cached distances + better caching)
- **New step time**: ~20ms/step (50 steps/s)
- **Speedup**: 23% faster

## Next Steps for Further Optimization

### Medium Priority

1. **Spatial context mine overlay** (1.68ms, 6.3%)
   - Consider caching or reducing update frequency
   - Profile individual computations

2. **Better step-level caching** in `CachedPathDistanceCalculator`
   - Pre-warm cache with likely queries
   - Analyze cache hit rates

### Low Priority

3. **Parallelize across environments**
   - With 4 envs, observations could be computed in parallel
   - Requires refactoring to avoid shared state issues

4. **Reduce observation computation frequency**
   - With frame-skip=4, could compute some features less frequently
   - Requires careful analysis of what can be cached across frames

## Validation

To validate these optimizations:

1. Run the same training command with profiling enabled:
```bash
python scripts/train_and_compare.py \
    --experiment-name "optimized_test" \
    --architectures graph_free \
    --train-dataset ../nclone/datasets/train \
    --enable-go-explore \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 4000 \
    --num-envs 4 \
    --num-eval-episode 1 \
    --video-fps 60 \
    --num-gpus 1 \
    --output-dir experiments \
    --enable-early-stopping \
    --enable-lr-annealing \
    --replay-data-dir ../nclone/bc_replays_tmp \
    --bc-epochs 1 \
    --single-level '../nclone/test-single-level/006 both flavours of ramp jumping (and the control thereof)' \
    --frame-skip 4 \
    --debug \
    --enable-state-stacking \
    --goal-curriculum-window 3 \
    --goal-curriculum-threshold 0.1 \
    --enable-profiling \
    --enable-memory-profiling
```

2. Compare profiling results:
   - Total training time (expect ~310s vs 330s baseline)
   - Environment step time (expect ~22ms vs 26ms baseline)
   - Reachability features time (expect ~5.5ms vs 7.1ms baseline)
   - Reward calculation time (expect ~5.0ms vs 7.0ms baseline)
   - GraphData conversion time (expect near-zero after first build)

3. Check for correctness:
   - Episode rewards should be identical (optimizations don't change logic)
   - PBRS potentials should match (cached distances are the same values)
   - No new warnings or errors in logs

## References

- Profiling data: `experiments/quick_test_20260106_093528/graph_free/profiling/profiling_summary.json`
- Analysis plan: `.cursor/plans/environment_bottleneck_analysis_f27314d2.plan.md`

