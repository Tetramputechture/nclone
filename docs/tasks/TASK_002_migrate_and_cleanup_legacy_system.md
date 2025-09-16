# TASK 002: Migrate and Clean Up Legacy Reachability System

## Overview
Migrate all existing code from the legacy detailed reachability analysis system to the new tiered system, and remove deprecated components to maintain a clean, efficient codebase.

## Context & Justification

### Why Remove Instead of Optimize
Based on strategic analysis in `/workspace/nclone/docs/reachability_analysis_integration_strategy.md`:
- **Over-Engineering**: Current detailed analysis (166ms) is 16x slower than needed
- **Unnecessary Complexity**: RL training doesn't need 99% accuracy when 85-90% suffices
- **Maintenance Burden**: Keeping both systems increases complexity and technical debt
- **Clean Architecture**: Single tiered system is easier to understand and maintain

### Migration Strategy
- **Replace, Don't Duplicate**: Migrate all callers to tiered system
- **Validate Functionality**: Ensure no regression in test pass rates
- **Remove Deprecated Code**: Delete legacy components after migration
- **Update Documentation**: Reflect new architecture in all docs

## Technical Specification

### Migration Mapping
```python
# OLD: Legacy detailed analysis
legacy_analyzer = HierarchicalGeometryAnalyzer()
result = legacy_analyzer.analyze_reachability(level_data, ninja_pos, switch_states)

# NEW: Tiered system with appropriate tier selection
tiered_system = TieredReachabilitySystem()
result = tiered_system.analyze_reachability(
    level_data, ninja_pos, switch_states,
    performance_target="balanced"  # Auto-selects Tier 1 or 2
)
```

### Components to Remove
```
DEPRECATED COMPONENTS TO DELETE:
├── nclone/graph/reachability/hierarchical_geometry.py
├── nclone/graph/reachability/position_validator.py (if unused)
├── nclone/graph/reachability/subcell_analyzer.py (if unused)
├── nclone/graph/reachability/tile_connectivity.py (replace with simplified version)
└── Related test files and documentation
```

### Components to Migrate
```
MIGRATION TARGETS:
├── test_reachability_suite.py → Update to use tiered system
├── maps.json validation → Use Tier 2 for completable checks
├── Any external callers → Replace with tiered system calls
└── Documentation and examples → Update to new API
```

## Implementation Plan

### Phase 1: Analysis and Mapping (Week 1)
**Deliverables**:
1. **Dependency Analysis**: Identify all code that uses legacy system
2. **Migration Plan**: Detailed mapping of old → new API calls
3. **Test Impact Assessment**: Identify tests that need updates

**Key Activities**:
```bash
# Find all usages of legacy system
grep -r "HierarchicalGeometryAnalyzer" --include="*.py" .
grep -r "hierarchical_geometry" --include="*.py" .
grep -r "analyze_reachability" --include="*.py" .

# Analyze test dependencies
grep -r "from.*hierarchical_geometry" tests/
grep -r "import.*hierarchical_geometry" tests/
```

### Phase 2: Create Migration Interface (Week 1-2)
**Deliverables**:
1. **Compatibility Layer**: Temporary wrapper for smooth migration
2. **Migration Utilities**: Tools to help with code migration
3. **Validation Framework**: Ensure migrated code works correctly

**Implementation**:
```python
class LegacyCompatibilityWrapper:
    """
    Temporary wrapper to ease migration from legacy system.
    
    This class provides the old API while using the new tiered system internally.
    Should be removed after migration is complete.
    """
    
    def __init__(self):
        self.tiered_system = TieredReachabilitySystem()
        self._deprecation_warnings_shown = set()
    
    def analyze_reachability(self, level_data, ninja_position, switch_states=None):
        """
        Legacy API compatibility method.
        
        DEPRECATED: Use TieredReachabilitySystem directly.
        """
        caller_info = self._get_caller_info()
        if caller_info not in self._deprecation_warnings_shown:
            warnings.warn(
                f"HierarchicalGeometryAnalyzer is deprecated. "
                f"Use TieredReachabilitySystem instead. Called from: {caller_info}",
                DeprecationWarning,
                stacklevel=2
            )
            self._deprecation_warnings_shown.add(caller_info)
        
        # Use Tier 2 (balanced performance/accuracy) for legacy compatibility
        return self.tiered_system.analyze_reachability(
            level_data, ninja_position, switch_states,
            performance_target="balanced"
        )
    
    def _get_caller_info(self):
        """Get information about the calling code for deprecation warnings."""
        import inspect
        frame = inspect.currentframe().f_back.f_back
        return f"{frame.f_code.co_filename}:{frame.f_lineno}"

# Temporary alias for migration period
HierarchicalGeometryAnalyzer = LegacyCompatibilityWrapper
```

### Phase 3: Migrate Core Components (Week 2-3)
**Deliverables**:
1. **Test Suite Migration**: Update all reachability tests
2. **Maps Validation**: Migrate level completability checks
3. **External API Updates**: Update any public interfaces

**Test Migration Strategy**:
```python
class TestReachabilitySuiteMigration:
    """
    Migration strategy for test suite.
    """
    
    def migrate_test_case(self, test_case):
        """
        Migrate individual test case to use tiered system.
        """
        # OLD
        # analyzer = HierarchicalGeometryAnalyzer()
        # result = analyzer.analyze_reachability(...)
        
        # NEW
        tiered_system = TieredReachabilitySystem()
        result = tiered_system.analyze_reachability(
            test_case.level_data,
            test_case.ninja_position,
            test_case.switch_states,
            performance_target="accurate"  # Use Tier 2 for test accuracy
        )
        
        # Validate that result format is compatible
        self._validate_result_compatibility(result, test_case.expected_result)
    
    def _validate_result_compatibility(self, new_result, expected_result):
        """
        Ensure new result format matches expected format.
        """
        # Check that all expected fields are present
        assert hasattr(new_result, 'reachable_positions')
        assert hasattr(new_result, 'is_level_completable')
        
        # Check that completability matches (most important)
        assert new_result.is_level_completable() == expected_result.is_level_completable()
```

### Phase 4: Remove Legacy Code (Week 3-4)
**Deliverables**:
1. **Code Deletion**: Remove all deprecated components
2. **Documentation Updates**: Update all references to new system
3. **Final Validation**: Ensure no regressions after cleanup

**Removal Checklist**:
```python
REMOVAL_CHECKLIST = [
    # Core legacy files
    "nclone/graph/reachability/hierarchical_geometry.py",
    "nclone/graph/reachability/position_validator.py",  # If not used by new system
    "nclone/graph/reachability/subcell_analyzer.py",   # If not used by new system
    
    # Legacy test files
    "tests/test_hierarchical_geometry.py",
    "tests/test_position_validator.py",  # If not needed
    
    # Compatibility wrapper (after migration complete)
    "nclone/graph/reachability/legacy_compatibility.py",
    
    # Update imports in remaining files
    # Remove any remaining references to deleted components
]
```

## Testing Strategy

### Migration Validation
```python
class MigrationValidationTest(unittest.TestCase):
    """
    Validate that migration doesn't break existing functionality.
    """
    
    def setUp(self):
        self.legacy_system = LegacyCompatibilityWrapper()
        self.tiered_system = TieredReachabilitySystem()
        self.test_cases = load_all_test_cases()
    
    def test_functional_equivalence(self):
        """
        Test that new system produces functionally equivalent results.
        """
        for test_case in self.test_cases:
            # Get results from both systems
            legacy_result = self.legacy_system.analyze_reachability(
                test_case.level_data, test_case.ninja_pos, test_case.switch_states
            )
            
            tiered_result = self.tiered_system.analyze_reachability(
                test_case.level_data, test_case.ninja_pos, test_case.switch_states,
                performance_target="balanced"
            )
            
            # Most important: level completability should match
            self.assertEqual(
                legacy_result.is_level_completable(),
                tiered_result.is_level_completable(),
                f"Completability mismatch in {test_case.name}"
            )
            
            # Reachable positions should have high overlap (>90%)
            overlap = self._calculate_position_overlap(
                legacy_result.reachable_positions,
                tiered_result.reachable_positions
            )
            self.assertGreater(overlap, 0.90, 
                             f"Low position overlap in {test_case.name}: {overlap:.2f}")
    
    def test_performance_improvement(self):
        """
        Validate that new system is significantly faster.
        """
        performance_results = []
        
        for test_case in self.test_cases:
            # Benchmark legacy system (via compatibility wrapper)
            start_time = time.perf_counter()
            legacy_result = self.legacy_system.analyze_reachability(
                test_case.level_data, test_case.ninja_pos, test_case.switch_states
            )
            legacy_time = (time.perf_counter() - start_time) * 1000
            
            # Benchmark tiered system
            start_time = time.perf_counter()
            tiered_result = self.tiered_system.analyze_reachability(
                test_case.level_data, test_case.ninja_pos, test_case.switch_states,
                performance_target="balanced"
            )
            tiered_time = (time.perf_counter() - start_time) * 1000
            
            improvement_ratio = legacy_time / tiered_time
            performance_results.append(improvement_ratio)
        
        # Should see significant performance improvement
        avg_improvement = np.mean(performance_results)
        self.assertGreater(avg_improvement, 5.0, 
                          f"Insufficient performance improvement: {avg_improvement:.1f}x")
    
    def test_no_missing_dependencies(self):
        """
        Test that no code still depends on removed components.
        """
        # This would be run after code removal
        try:
            # Try to import removed modules - should fail
            with self.assertRaises(ImportError):
                from nclone.graph.reachability.hierarchical_geometry import HierarchicalGeometryAnalyzer
        except ImportError:
            pass  # Expected after removal
```

### Regression Testing
```python
class RegressionTest(unittest.TestCase):
    """
    Ensure no regressions after migration.
    """
    
    def test_all_existing_tests_pass(self):
        """
        Run entire existing test suite with migrated system.
        """
        # This should pass all existing reachability tests
        # using the new tiered system
        test_suite = load_existing_reachability_tests()
        
        for test in test_suite:
            with self.subTest(test=test.name):
                result = test.run_with_tiered_system()
                self.assertTrue(result.passed, f"Test {test.name} failed after migration")
    
    def test_maps_json_validation(self):
        """
        Ensure all levels in maps.json still validate correctly.
        """
        maps_data = load_maps_json()
        tiered_system = TieredReachabilitySystem()
        
        for level_name, level_info in maps_data.items():
            if level_info.get('completable', False):
                # Test that completable levels are still detected as completable
                result = tiered_system.analyze_reachability(
                    level_info['level_data'],
                    level_info['ninja_start_pos'],
                    level_info.get('initial_switch_states', {}),
                    performance_target="accurate"
                )
                
                self.assertTrue(result.is_level_completable(),
                              f"Level {level_name} no longer detected as completable")
```

## Success Criteria

### Functional Requirements
- **Zero Regressions**: All existing tests pass with new system
- **API Compatibility**: Smooth migration path for all existing code
- **Performance Improvement**: >5x speed improvement on average
- **Clean Codebase**: No deprecated code remains after migration

### Quality Requirements
- **Test Coverage**: >95% coverage for migration utilities
- **Documentation**: All docs updated to reflect new architecture
- **Code Quality**: No technical debt from legacy system remains
- **Maintainability**: Simplified codebase easier to understand and modify

### Performance Requirements
- **Migration Speed**: Complete migration in 4 weeks
- **Zero Downtime**: No interruption to development during migration
- **Validation**: Comprehensive testing ensures no functionality lost

## Risk Mitigation

### Technical Risks
1. **Functionality Loss**: Comprehensive testing and validation
2. **Performance Regression**: Continuous benchmarking during migration
3. **Integration Issues**: Gradual migration with compatibility layer

### Project Risks
1. **Timeline Pressure**: Prioritize critical functionality first
2. **Scope Creep**: Focus on migration, not new features
3. **Team Coordination**: Clear communication about deprecated components

### Mitigation Strategies
- **Compatibility Layer**: Temporary wrapper eases migration pressure
- **Incremental Migration**: Migrate components one at a time
- **Rollback Plan**: Keep legacy code until migration fully validated
- **Extensive Testing**: Multiple validation layers ensure correctness

## Deliverables

1. **Migration Analysis**: Complete dependency analysis and migration plan
2. **Compatibility Layer**: Temporary wrapper for smooth transition
3. **Migrated Codebase**: All code updated to use tiered system
4. **Clean Architecture**: All deprecated components removed
5. **Updated Documentation**: Reflects new tiered architecture
6. **Validation Report**: Comprehensive testing results and performance analysis

## Timeline

- **Week 1**: Analysis, mapping, and compatibility layer creation
- **Week 2**: Core component migration and initial testing
- **Week 3**: Complete migration and validation
- **Week 4**: Legacy code removal and final cleanup

## Dependencies

### Internal Dependencies
- **TASK 001**: Tiered Reachability System must be complete and tested
- **TASK 003**: Compact Features implementation (can run in parallel)

### External Dependencies
- **Testing Infrastructure**: Comprehensive test suite for validation
- **Documentation System**: Update all references to new architecture

## References

1. **Strategic Analysis**: `/workspace/nclone/docs/reachability_analysis_integration_strategy.md`
2. **Current Implementation**: `/workspace/nclone/nclone/graph/reachability/hierarchical_geometry.py`
3. **Test Suite**: `/workspace/nclone/test_reachability_suite.py`
4. **Tiered System**: TASK 001 - Implement Tiered Reachability System