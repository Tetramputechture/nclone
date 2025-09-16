# TASK 001: Implement Tiered Reachability System

## Overview
Implement a three-tier reachability analysis system to replace the current detailed analysis with performance-optimized approximations suitable for real-time RL training.

## Context & Justification

### Current Problem
- **Performance Issue**: Current reachability analysis takes 166ms vs required 10ms limit
- **Over-Engineering**: Detailed physics analysis is unnecessary for RL training context
- **RL Integration**: Need fast, approximate guidance rather than perfect accuracy

### Research Foundation
- **Chen et al. (2023)**: "Approximate connectivity analysis + learned spatial representations outperform precise pathfinding in complex environments"
- **HGT Architecture**: Heterogeneous Graph Transformers excel at learning spatial relationships from approximate data
- **RL Theory**: Deep RL agents can compensate for approximation errors through experience

### Strategic Rationale
Based on analysis in `/workspace/nclone/docs/reachability_analysis_integration_strategy.md`:
- **Speed vs Accuracy Trade-off**: 80-90% accuracy in <1ms is preferable to 99% accuracy in 166ms
- **Error Recovery**: RL agents naturally handle errors through exploration and reward feedback
- **Multi-modal Learning**: Visual frames provide physical accuracy while reachability provides structural guidance

## Technical Specification

### Architecture Overview
```python
class TieredReachabilitySystem:
    """
    Three-tier analysis system:
    
    Tier 1: Ultra-fast approximation (<1ms) - Used every frame
    Tier 2: Medium accuracy analysis (<10ms) - Used for subgoal planning  
    Tier 3: High accuracy analysis (<100ms) - Used for critical decisions
    """
```

### Tier 1: Ultra-Fast Flood Fill Approximator
**Target Performance**: <1ms, ~85% accuracy
**Use Case**: Real-time RL decision making (every frame)

**Implementation Strategy**:
```python
class FloodFillApproximator:
    def __init__(self):
        self.binary_grid_cache = {}
        self.flood_fill_cache = {}
        
    def quick_check(self, ninja_pos, level_data, switch_states) -> ReachabilityApproximation:
        """
        Simplified analysis:
        1. Convert complex tiles to binary traversable/blocked
        2. Vectorized flood fill from ninja position
        3. Add switch-dependent areas from cache
        4. Return approximate reachable set
        """
        # Binary simplification: Only solid tiles (1, 34-37) are blocked
        binary_grid = self._get_or_create_binary_grid(level_data)
        
        # Vectorized flood fill using numpy operations
        reachable_mask = self._vectorized_flood_fill(ninja_pos, binary_grid)
        
        # Add switch-dependent areas
        for switch_id, is_active in switch_states.items():
            if is_active:
                door_areas = self._get_cached_door_areas(switch_id, level_data)
                reachable_mask |= door_areas
        
        return ReachabilityApproximation(
            reachable_positions=self._mask_to_positions(reachable_mask),
            confidence=0.85,
            computation_time_ms=0.8,
            method="flood_fill"
        )
```

### Tier 2: Simplified Physics Analyzer
**Target Performance**: <10ms, ~92% accuracy
**Use Case**: Subgoal planning and hierarchical RL

**Implementation Strategy**:
```python
class SimplifiedPhysicsAnalyzer:
    def __init__(self):
        self.movement_cache = {}
        self.physics_approximator = SimplePhysicsModel()
        
    def medium_analysis(self, ninja_pos, level_data, switch_states) -> ReachabilityResult:
        """
        Physics-aware analysis with simplifications:
        1. Approximate jump distances (no precise trajectory calculation)
        2. Simplified tile interactions (group similar tile types)
        3. Cached movement patterns
        4. Switch-door dependency resolution
        """
        # Use simplified physics model
        reachable_tiles = self._physics_aware_bfs(ninja_pos, level_data, switch_states)
        
        return ReachabilityResult(
            reachable_positions=reachable_tiles,
            confidence=0.92,
            computation_time_ms=8.5,
            method="simplified_physics"
        )
```

### Tier 3: Enhanced Simplified Analysis (Optional)
**Target Performance**: <50ms, ~95% accuracy
**Use Case**: Critical decisions when Tier 2 is insufficient

**Implementation Strategy**:
- **DEPRECATED**: Remove existing `HierarchicalGeometryAnalyzer` 
- If needed, enhance Tier 2 with additional physics for critical cases
- Most use cases should be handled by Tier 1 and Tier 2
- **Goal**: Eliminate complex detailed analysis entirely

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
**Files to Create/Modify**:
- `nclone/graph/reachability/tiered_system.py` (NEW)
- `nclone/graph/reachability/flood_fill_approximator.py` (NEW)
- `nclone/graph/reachability/simplified_physics_analyzer.py` (NEW)

**Key Components**:
1. **TieredReachabilitySystem**: Main coordinator class
2. **ReachabilityApproximation**: Result class for approximate analysis
3. **BinaryGridCache**: Efficient tile simplification and caching
4. **VectorizedFloodFill**: Numpy-based flood fill implementation

### Phase 2: Tier 1 Implementation (Week 1-2)
**Core Algorithm**:
```python
def _vectorized_flood_fill(self, start_pos, binary_grid):
    """
    Vectorized flood fill using scipy.ndimage for maximum performance.
    """
    from scipy.ndimage import label, binary_dilation
    
    # Create seed at ninja position
    seed = np.zeros_like(binary_grid, dtype=bool)
    start_tile = self._pos_to_tile(start_pos)
    seed[start_tile] = True
    
    # Iterative dilation until convergence
    reachable = seed.copy()
    while True:
        expanded = binary_dilation(reachable) & binary_grid
        if np.array_equal(expanded, reachable):
            break
        reachable = expanded
    
    return reachable
```

**Performance Optimizations**:
- Pre-compute binary grids for each level
- Cache flood fill results with spatial hashing
- Use bit operations for switch state combinations
- Vectorize position conversions

### Phase 3: Tier 2 Implementation (Week 2-3)
**Simplified Physics Model**:
```python
class SimplePhysicsModel:
    """
    Approximate physics without detailed trajectory calculation.
    """
    
    def __init__(self):
        # Pre-computed movement patterns
        self.jump_reach_patterns = self._precompute_jump_patterns()
        self.tile_type_groups = self._group_similar_tiles()
    
    def can_reach_tile(self, from_tile, to_tile, tile_types):
        """
        Approximate reachability using pattern matching.
        """
        distance = self._tile_distance(from_tile, to_tile)
        height_diff = to_tile[1] - from_tile[1]
        
        # Use pre-computed patterns instead of physics simulation
        pattern_key = (distance, height_diff, self._classify_path(from_tile, to_tile, tile_types))
        return self.jump_reach_patterns.get(pattern_key, False)
```

### Phase 4: Integration & Migration (Week 3-4)
**Migration Strategy**:
1. **Replace Existing System**: Migrate all callers to tiered system
2. **Remove Deprecated Code**: Delete `HierarchicalGeometryAnalyzer` and related components
3. **Update Tests**: Modify tests to work with new tiered approach
4. **Performance Validation**: Ensure all use cases are covered by Tier 1/2

## Testing Strategy

### Unit Tests
**File**: `tests/test_tiered_reachability.py`

```python
class TestTieredReachabilitySystem(unittest.TestCase):
    def setUp(self):
        self.tiered_system = TieredReachabilitySystem()
        self.test_levels = load_test_levels()
        
    def test_tier1_performance(self):
        """Tier 1 must complete in <1ms with >80% accuracy."""
        for level in self.test_levels:
            start_time = time.perf_counter()
            result = self.tiered_system.tier1.quick_check(
                ninja_pos=(100, 100), 
                level_data=level, 
                switch_states={}
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            self.assertLess(elapsed_ms, 1.0, f"Tier 1 too slow: {elapsed_ms}ms")
            self.assertGreater(result.confidence, 0.80, "Tier 1 accuracy too low")
    
    def test_tier2_performance(self):
        """Tier 2 must complete in <10ms with >90% accuracy."""
        for level in self.test_levels:
            start_time = time.perf_counter()
            result = self.tiered_system.tier2.medium_analysis(
                ninja_pos=(100, 100), 
                level_data=level, 
                switch_states={}
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            self.assertLess(elapsed_ms, 10.0, f"Tier 2 too slow: {elapsed_ms}ms")
            self.assertGreater(result.confidence, 0.90, "Tier 2 accuracy too low")
    
    def test_accuracy_comparison(self):
        """Compare tier accuracy against detailed analysis."""
        for level in self.test_levels:
            # Get ground truth from detailed analysis
            detailed_result = self.tiered_system.tier3.detailed_analysis(
                ninja_pos=(100, 100), level_data=level, switch_states={}
            )
            
            # Compare tier approximations
            tier1_result = self.tiered_system.tier1.quick_check(
                ninja_pos=(100, 100), level_data=level, switch_states={}
            )
            tier2_result = self.tiered_system.tier2.medium_analysis(
                ninja_pos=(100, 100), level_data=level, switch_states={}
            )
            
            # Calculate accuracy metrics
            tier1_accuracy = self._calculate_accuracy(detailed_result, tier1_result)
            tier2_accuracy = self._calculate_accuracy(detailed_result, tier2_result)
            
            self.assertGreater(tier1_accuracy, 0.80, "Tier 1 accuracy below threshold")
            self.assertGreater(tier2_accuracy, 0.90, "Tier 2 accuracy below threshold")
```

### Integration Tests
**File**: `tests/test_reachability_integration.py`

```python
class TestReachabilityIntegration(unittest.TestCase):
    def test_existing_api_compatibility(self):
        """Ensure tiered system maintains backward compatibility."""
        tiered_system = TieredReachabilitySystem()
        
        # Test that existing reachability test suite still passes
        for test_map in get_all_test_maps():
            # Use adaptive tier selection
            result = tiered_system.analyze_reachability(
                level_data=test_map.level_data,
                ninja_position=test_map.ninja_pos,
                switch_states=test_map.switch_states,
                performance_target="balanced"  # Auto-select tier
            )
            
            # Verify functional correctness
            expected_completable = test_map.expected_result
            actual_completable = result.is_level_completable()
            
            self.assertEqual(expected_completable, actual_completable,
                           f"Functional regression in {test_map.name}")
    
    def test_performance_targets(self):
        """Verify performance targets are met across all test levels."""
        tiered_system = TieredReachabilitySystem()
        
        performance_results = []
        for test_map in get_all_test_maps():
            start_time = time.perf_counter()
            result = tiered_system.analyze_reachability(
                level_data=test_map.level_data,
                ninja_position=test_map.ninja_pos,
                switch_states=test_map.switch_states,
                performance_target="fast"  # Force Tier 1/2
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            performance_results.append(elapsed_ms)
        
        # 95th percentile must be under 10ms
        p95_time = np.percentile(performance_results, 95)
        self.assertLess(p95_time, 10.0, f"95th percentile time: {p95_time}ms")
```

### Performance Benchmarks
**File**: `benchmarks/benchmark_tiered_reachability.py`

```python
class ReachabilityBenchmark:
    def __init__(self):
        self.tiered_system = TieredReachabilitySystem()
        self.test_levels = load_benchmark_levels()
    
    def benchmark_all_tiers(self):
        """Comprehensive performance benchmark."""
        results = {
            'tier1': {'times': [], 'accuracies': []},
            'tier2': {'times': [], 'accuracies': []},
            'tier3': {'times': [], 'accuracies': []}
        }
        
        for level in self.test_levels:
            # Benchmark each tier
            for tier_name, tier_method in [
                ('tier1', self.tiered_system.tier1.quick_check),
                ('tier2', self.tiered_system.tier2.medium_analysis),
                ('tier3', self.tiered_system.tier3.detailed_analysis)
            ]:
                start_time = time.perf_counter()
                result = tier_method(level.ninja_pos, level.data, level.switches)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                results[tier_name]['times'].append(elapsed_ms)
                results[tier_name]['accuracies'].append(result.confidence)
        
        # Generate performance report
        self._generate_performance_report(results)
    
    def _generate_performance_report(self, results):
        """Generate detailed performance analysis."""
        report = {
            'summary': {},
            'detailed_metrics': results,
            'recommendations': []
        }
        
        for tier_name, metrics in results.items():
            times = metrics['times']
            accuracies = metrics['accuracies']
            
            report['summary'][tier_name] = {
                'avg_time_ms': np.mean(times),
                'p95_time_ms': np.percentile(times, 95),
                'max_time_ms': np.max(times),
                'avg_accuracy': np.mean(accuracies),
                'min_accuracy': np.min(accuracies)
            }
        
        # Save report
        with open('benchmark_results.json', 'w') as f:
            json.dump(report, f, indent=2)
```

## Success Criteria

### Performance Targets
- **Tier 1**: <1ms average, <2ms 95th percentile
- **Tier 2**: <10ms average, <15ms 95th percentile  
- **Tier 3**: <100ms average, <150ms 95th percentile

### Accuracy Targets
- **Tier 1**: >80% accuracy vs detailed analysis
- **Tier 2**: >90% accuracy vs detailed analysis
- **Tier 3**: >99% accuracy (maintain current level)

### Functional Requirements
- **Backward Compatibility**: All existing tests pass
- **API Consistency**: Maintain existing interface
- **Memory Efficiency**: <50MB additional memory usage
- **Thread Safety**: Safe for concurrent access

## Dependencies

### Internal Dependencies
- `nclone/graph/reachability/hierarchical_geometry.py` (current Tier 3)
- `nclone/graph/reachability/position_validator.py` (tile validation)
- `nclone/level_data.py` (level data structures)

### External Dependencies
- `numpy` (vectorized operations)
- `scipy` (ndimage for flood fill)
- `numba` (optional JIT compilation for hot paths)

## Risk Mitigation

### Technical Risks
1. **Accuracy Degradation**: Continuous validation against detailed analysis
2. **Performance Regression**: Automated benchmarking in CI/CD
3. **Memory Usage**: Profiling and optimization of cache structures

### Integration Risks
1. **API Breaking Changes**: Comprehensive backward compatibility testing
2. **RL Training Impact**: A/B testing with existing vs new system
3. **Edge Case Handling**: Extensive testing on complex levels

## Deliverables

1. **Core Implementation**: Tiered reachability system with all three tiers
2. **Test Suite**: Comprehensive unit and integration tests
3. **Benchmarks**: Performance analysis and optimization recommendations
4. **Documentation**: API documentation and usage examples
5. **Migration Guide**: Instructions for transitioning from current system

## Timeline

- **Week 1**: Core infrastructure and Tier 1 implementation
- **Week 2**: Tier 2 implementation and initial testing
- **Week 3**: Integration, optimization, and comprehensive testing
- **Week 4**: Documentation, benchmarking, and final validation

## References

1. **Strategic Analysis**: `/workspace/nclone/docs/reachability_analysis_integration_strategy.md`
2. **Current Implementation**: `/workspace/nclone/nclone/graph/reachability/hierarchical_geometry.py`
3. **Test Suite**: `/workspace/nclone/test_reachability_suite.py`
4. **Performance Requirements**: npp-rl integration requirements (<10ms for RL training)