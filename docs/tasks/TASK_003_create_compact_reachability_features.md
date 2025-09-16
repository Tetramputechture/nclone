# TASK 003: Create Compact Reachability Features for RL Integration

## Overview
Design and implement a compact, efficient encoding of reachability information specifically optimized for integration with the HGT-based RL architecture in npp-rl.

## Context & Justification

### Integration Requirements
Based on analysis of npp-rl architecture:
- **HGT Multimodal Extractor**: Primary feature extraction using Heterogeneous Graph Transformers
- **Real-time Constraints**: 60 FPS decision making (<16ms per action)
- **Multi-modal Fusion**: Integration with visual frames, game state vectors, and graph representations
- **Memory Efficiency**: Minimize additional memory overhead for RL training

### Strategic Rationale
From `/workspace/nclone/docs/reachability_analysis_integration_strategy.md`:
- **Compact Encoding**: 32-dimensional features vs detailed position lists
- **Guidance Over Ground Truth**: Approximate features for learned spatial reasoning
- **Performance Priority**: Speed and consistency over perfect accuracy
- **HGT Integration**: Leverage graph transformer's ability to learn from approximate data

### Research Foundation
- **HGT Architecture**: Heterogeneous Graph Transformers excel at learning spatial relationships from compact representations
- **Multi-modal RL**: Visual frames provide physical accuracy while reachability provides structural guidance
- **Feature Learning**: Deep RL can learn complex spatial reasoning from simple numerical features

## Technical Specification

### Compact Feature Design
**Target**: 64-dimensional feature vector encoding all reachability information

```python
class CompactReachabilityFeatures:
    """
    Compact encoding of reachability analysis for RL integration.
    
    Feature Vector Layout (64 dimensions):
    [0-7]:    Objective distances (8 closest objectives)
    [8-23]:   Switch states and dependencies (16 switches max)
    [24-39]:  Hazard proximities and threat levels (16 hazards max)
    [40-47]:  Area connectivity metrics (8 directional areas)
    [48-55]:  Movement capability indicators (8 movement types)
    [56-63]:  Meta-features (confidence, timing, complexity)
    """
    
    def __init__(self):
        self.feature_dim = 64
        self.objective_slots = 8
        self.switch_slots = 16
        self.hazard_slots = 16
        self.area_slots = 8
        self.movement_slots = 8
        self.meta_slots = 8
    
    def encode_reachability(self, reachability_result, level_data, ninja_pos) -> torch.Tensor:
        """
        Encode reachability analysis into compact feature vector.
        """
        features = torch.zeros(self.feature_dim, dtype=torch.float32)
        
        # [0-7] Objective distances
        objective_distances = self._encode_objective_distances(
            reachability_result, level_data, ninja_pos
        )
        features[0:8] = objective_distances
        
        # [8-23] Switch states and dependencies
        switch_features = self._encode_switch_states(
            reachability_result, level_data
        )
        features[8:24] = switch_features
        
        # [24-39] Hazard proximities
        hazard_features = self._encode_hazard_proximities(
            reachability_result, level_data, ninja_pos
        )
        features[24:40] = hazard_features
        
        # [40-47] Area connectivity
        area_features = self._encode_area_connectivity(
            reachability_result, level_data, ninja_pos
        )
        features[40:48] = area_features
        
        # [48-55] Movement capabilities
        movement_features = self._encode_movement_capabilities(
            reachability_result, level_data, ninja_pos
        )
        features[48:56] = movement_features
        
        # [56-63] Meta-features
        meta_features = self._encode_meta_features(
            reachability_result, level_data
        )
        features[56:64] = meta_features
        
        return features
```

### Feature Encoding Strategies

#### 1. Objective Distance Encoding
```python
def _encode_objective_distances(self, reachability_result, level_data, ninja_pos) -> torch.Tensor:
    """
    Encode distances to key objectives (switches, doors, gold, exit).
    
    Encoding Strategy:
    - Normalize distances by level size
    - Use log scaling for better gradient flow
    - Unreachable objectives encoded as 1.0 (maximum distance)
    """
    objectives = self._identify_key_objectives(level_data)
    distances = torch.ones(self.objective_slots)  # Default: unreachable
    
    level_diagonal = math.sqrt(level_data.width**2 + level_data.height**2)
    
    for i, objective in enumerate(objectives[:self.objective_slots]):
        if self._is_objective_reachable(objective, reachability_result):
            raw_distance = self._calculate_distance(ninja_pos, objective.position)
            normalized_distance = raw_distance / level_diagonal
            # Log scaling: log(1 + x) for better gradient properties
            distances[i] = math.log(1 + normalized_distance) / math.log(2)
        # else: keep default 1.0 (unreachable)
    
    return distances

def _identify_key_objectives(self, level_data) -> List[Objective]:
    """
    Identify and prioritize key objectives for distance encoding.
    
    Priority Order:
    1. Exit door (highest priority)
    2. Exit switch
    3. Locked door switches (by proximity)
    4. Gold pieces (by value/proximity)
    5. Trap door switches (negative priority - avoid)
    """
    objectives = []
    
    # Exit door and switch (highest priority)
    exit_door = level_data.get_exit_door()
    if exit_door:
        objectives.append(Objective(exit_door.position, 'exit_door', priority=1.0))
    
    exit_switch = level_data.get_exit_switch()
    if exit_switch:
        objectives.append(Objective(exit_switch.position, 'exit_switch', priority=0.9))
    
    # Locked door switches
    for door_switch in level_data.get_locked_door_switches():
        priority = 0.8 - 0.1 * len(objectives)  # Decreasing priority
        objectives.append(Objective(door_switch.position, 'door_switch', priority=priority))
    
    # Gold pieces
    for gold in level_data.get_gold_pieces():
        priority = 0.5 - 0.05 * len(objectives)
        objectives.append(Objective(gold.position, 'gold', priority=priority))
    
    # Sort by priority and return top objectives
    objectives.sort(key=lambda x: x.priority, reverse=True)
    return objectives
```

#### 2. Switch State Encoding
```python
def _encode_switch_states(self, reachability_result, level_data) -> torch.Tensor:
    """
    Encode switch states and dependencies.
    
    Encoding Strategy:
    - Binary encoding for switch states (0=inactive, 1=active)
    - Reachability encoding (0=unreachable, 0.5=reachable, 1=activated)
    - Dependency encoding for switch-door relationships
    """
    switch_features = torch.zeros(self.switch_slots)
    
    switches = level_data.get_all_switches()[:self.switch_slots]
    
    for i, switch in enumerate(switches):
        # Base encoding: switch state
        if switch.is_active:
            switch_features[i] = 1.0  # Activated
        elif self._is_switch_reachable(switch, reachability_result):
            switch_features[i] = 0.5  # Reachable but not activated
        else:
            switch_features[i] = 0.0  # Unreachable
        
        # Add dependency information (affects encoding slightly)
        if switch.has_dependent_doors:
            dependency_bonus = 0.1 * len(switch.dependent_doors)
            switch_features[i] += min(dependency_bonus, 0.4)
    
    return switch_features
```

#### 3. Hazard Proximity Encoding
```python
def _encode_hazard_proximities(self, reachability_result, level_data, ninja_pos) -> torch.Tensor:
    """
    Encode proximity and threat level of hazards.
    
    Encoding Strategy:
    - Distance-based threat encoding (closer = higher threat)
    - Hazard type weighting (drones > mines > static hazards)
    - Reachability-aware encoding (unreachable hazards have lower threat)
    """
    hazard_features = torch.zeros(self.hazard_slots)
    
    hazards = self._get_prioritized_hazards(level_data, ninja_pos)
    
    for i, hazard in enumerate(hazards[:self.hazard_slots]):
        # Calculate base threat based on distance and type
        distance = self._calculate_distance(ninja_pos, hazard.position)
        threat_radius = hazard.get_threat_radius()
        
        if distance <= threat_radius:
            # Immediate threat
            base_threat = 1.0 - (distance / threat_radius)
        else:
            # Distant threat (exponential decay)
            base_threat = math.exp(-(distance - threat_radius) / threat_radius)
        
        # Weight by hazard type
        type_weight = {
            'drone': 1.0,      # Highest threat (moving)
            'mine': 0.8,       # High threat (explosive)
            'thwump': 0.6,     # Medium threat (predictable)
            'static': 0.4      # Low threat (avoidable)
        }.get(hazard.type, 0.5)
        
        # Adjust for reachability (unreachable hazards less threatening)
        if not self._is_position_reachable(hazard.position, reachability_result):
            type_weight *= 0.3
        
        hazard_features[i] = base_threat * type_weight
    
    return hazard_features
```

#### 4. Area Connectivity Encoding
```python
def _encode_area_connectivity(self, reachability_result, level_data, ninja_pos) -> torch.Tensor:
    """
    Encode connectivity to different areas of the level.
    
    Encoding Strategy:
    - Divide level into 8 directional sectors (N, NE, E, SE, S, SW, W, NW)
    - Encode reachable area percentage in each sector
    - Weight by objective density in each sector
    """
    area_features = torch.zeros(self.area_slots)
    
    # Define 8 directional sectors
    sectors = self._define_level_sectors(level_data, ninja_pos)
    
    for i, sector in enumerate(sectors):
        # Calculate reachable area in this sector
        sector_positions = self._get_positions_in_sector(sector, level_data)
        reachable_positions = set(reachability_result.reachable_positions)
        
        sector_reachable = len(sector_positions & reachable_positions)
        sector_total = len(sector_positions)
        
        if sector_total > 0:
            reachability_ratio = sector_reachable / sector_total
            
            # Weight by objective density
            objectives_in_sector = self._count_objectives_in_sector(sector, level_data)
            objective_weight = 1.0 + 0.5 * objectives_in_sector
            
            area_features[i] = reachability_ratio * objective_weight
        else:
            area_features[i] = 0.0
    
    return area_features
```

#### 5. Movement Capability Encoding
```python
def _encode_movement_capabilities(self, reachability_result, level_data, ninja_pos) -> torch.Tensor:
    """
    Encode available movement capabilities from current position.
    
    Encoding Strategy:
    - Test each movement type availability (walk, jump, wall_jump, etc.)
    - Encode as capability strength (0=impossible, 1=fully available)
    - Consider local tile types and physics constraints
    """
    movement_features = torch.zeros(self.movement_slots)
    
    movement_types = [
        'walk_left', 'walk_right', 'jump_up', 'jump_left', 
        'jump_right', 'wall_jump', 'fall', 'special'
    ]
    
    for i, movement_type in enumerate(movement_types):
        capability = self._assess_movement_capability(
            ninja_pos, movement_type, level_data, reachability_result
        )
        movement_features[i] = capability
    
    return movement_features

def _assess_movement_capability(self, ninja_pos, movement_type, level_data, reachability_result) -> float:
    """
    Assess capability for specific movement type.
    """
    # Get positions reachable via this movement type
    reachable_via_movement = self._get_positions_reachable_via_movement(
        ninja_pos, movement_type, level_data
    )
    
    if not reachable_via_movement:
        return 0.0  # Movement type not available
    
    # Check how many of these positions are in reachable set
    reachable_positions = set(reachability_result.reachable_positions)
    valid_movements = len(reachable_via_movement & reachable_positions)
    total_movements = len(reachable_via_movement)
    
    return valid_movements / total_movements if total_movements > 0 else 0.0
```

#### 6. Meta-Feature Encoding
```python
def _encode_meta_features(self, reachability_result, level_data) -> torch.Tensor:
    """
    Encode meta-information about the reachability analysis.
    
    Features:
    [0] Analysis confidence (0-1)
    [1] Computation time (normalized)
    [2] Level complexity estimate (0-1)
    [3] Reachable area ratio (0-1)
    [4] Switch dependency complexity (0-1)
    [5] Hazard density (0-1)
    [6] Analysis method indicator (tier 1/2/3)
    [7] Cache hit indicator (0=miss, 1=hit)
    """
    meta_features = torch.zeros(self.meta_slots)
    
    # Analysis confidence
    meta_features[0] = getattr(reachability_result, 'confidence', 1.0)
    
    # Computation time (log-normalized)
    computation_time = getattr(reachability_result, 'computation_time_ms', 1.0)
    meta_features[1] = min(math.log(1 + computation_time) / math.log(100), 1.0)
    
    # Level complexity
    complexity = self._estimate_level_complexity(level_data)
    meta_features[2] = complexity
    
    # Reachable area ratio
    total_positions = level_data.width * level_data.height
    reachable_count = len(reachability_result.reachable_positions)
    meta_features[3] = reachable_count / total_positions
    
    # Switch dependency complexity
    switch_complexity = self._calculate_switch_complexity(level_data)
    meta_features[4] = switch_complexity
    
    # Hazard density
    hazard_density = self._calculate_hazard_density(level_data)
    meta_features[5] = hazard_density
    
    # Analysis method
    method = getattr(reachability_result, 'method', 'detailed')
    method_encoding = {'flood_fill': 0.2, 'simplified_physics': 0.6, 'detailed': 1.0}
    meta_features[6] = method_encoding.get(method, 0.5)
    
    # Cache hit indicator
    cache_hit = getattr(reachability_result, 'from_cache', False)
    meta_features[7] = 1.0 if cache_hit else 0.0
    
    return meta_features
```

## Implementation Plan

### Phase 1: Core Feature Encoder (Week 1)
**Deliverables**:
1. **CompactReachabilityFeatures**: Main encoding class
2. **Feature Validation**: Ensure encoding consistency and bounds
3. **Unit Tests**: Test each encoding component

**Key Files**:
- `nclone/graph/reachability/compact_features.py` (NEW)
- `nclone/graph/reachability/feature_encoders.py` (NEW)
- `tests/test_compact_features.py` (NEW)

### Phase 2: Integration Interface (Week 2)
**Deliverables**:
1. **ReachabilityFeatureExtractor**: Interface for RL integration
2. **Caching System**: Efficient feature caching
3. **Performance Optimization**: Sub-millisecond feature extraction

**Implementation**:
```python
class ReachabilityFeatureExtractor:
    """
    High-level interface for extracting compact reachability features.
    """
    
    def __init__(self, tiered_system):
        self.tiered_system = tiered_system
        self.feature_encoder = CompactReachabilityFeatures()
        self.feature_cache = {}
        self.cache_ttl = 100  # milliseconds
    
    def extract_features(self, ninja_pos, level_data, switch_states, 
                        performance_target="fast") -> torch.Tensor:
        """
        Extract compact reachability features for RL integration.
        
        Args:
            ninja_pos: Current ninja position
            level_data: Level data structure
            switch_states: Current switch states
            performance_target: "fast" (Tier 1), "balanced" (Tier 2), "accurate" (Tier 3)
        
        Returns:
            64-dimensional feature tensor
        """
        # Check cache first
        cache_key = self._generate_cache_key(ninja_pos, switch_states, level_data)
        if self._is_cache_valid(cache_key):
            return self.feature_cache[cache_key]['features']
        
        # Get reachability analysis
        if performance_target == "fast":
            reachability_result = self.tiered_system.tier1.quick_check(
                ninja_pos, level_data, switch_states
            )
        elif performance_target == "balanced":
            reachability_result = self.tiered_system.tier2.medium_analysis(
                ninja_pos, level_data, switch_states
            )
        else:  # "accurate"
            reachability_result = self.tiered_system.tier3.detailed_analysis(
                ninja_pos, level_data, switch_states
            )
        
        # Encode features
        features = self.feature_encoder.encode_reachability(
            reachability_result, level_data, ninja_pos
        )
        
        # Cache result
        self._cache_features(cache_key, features, reachability_result)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get human-readable names for each feature dimension.
        """
        names = []
        
        # Objective distances [0-7]
        for i in range(8):
            names.append(f"objective_distance_{i}")
        
        # Switch states [8-23]
        for i in range(16):
            names.append(f"switch_state_{i}")
        
        # Hazard proximities [24-39]
        for i in range(16):
            names.append(f"hazard_proximity_{i}")
        
        # Area connectivity [40-47]
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        for direction in directions:
            names.append(f"area_connectivity_{direction}")
        
        # Movement capabilities [48-55]
        movements = ['walk_left', 'walk_right', 'jump_up', 'jump_left', 
                    'jump_right', 'wall_jump', 'fall', 'special']
        for movement in movements:
            names.append(f"movement_{movement}")
        
        # Meta-features [56-63]
        meta_names = ['confidence', 'computation_time', 'level_complexity', 
                     'reachable_ratio', 'switch_complexity', 'hazard_density',
                     'analysis_method', 'cache_hit']
        for meta_name in meta_names:
            names.append(f"meta_{meta_name}")
        
        return names
```

### Phase 3: Validation and Testing (Week 3)
**Deliverables**:
1. **Feature Validation Suite**: Comprehensive testing of feature encoding
2. **Performance Benchmarks**: Feature extraction performance analysis
3. **Interpretability Tools**: Feature visualization and analysis

### Phase 4: Documentation and Integration (Week 4)
**Deliverables**:
1. **API Documentation**: Complete documentation of feature encoding
2. **Integration Examples**: Sample code for npp-rl integration
3. **Feature Analysis**: Statistical analysis of feature distributions

## Testing Strategy

### Unit Tests
```python
class TestCompactReachabilityFeatures(unittest.TestCase):
    def setUp(self):
        self.feature_encoder = CompactReachabilityFeatures()
        self.test_levels = load_test_levels()
    
    def test_feature_dimensions(self):
        """Test that features always have correct dimensions."""
        for level in self.test_levels:
            reachability_result = create_mock_reachability_result()
            features = self.feature_encoder.encode_reachability(
                reachability_result, level.data, level.ninja_pos
            )
            
            self.assertEqual(features.shape, (64,), "Incorrect feature dimensions")
            self.assertTrue(torch.all(torch.isfinite(features)), "Non-finite features")
    
    def test_feature_bounds(self):
        """Test that features are within expected bounds."""
        for level in self.test_levels:
            reachability_result = create_mock_reachability_result()
            features = self.feature_encoder.encode_reachability(
                reachability_result, level.data, level.ninja_pos
            )
            
            # Most features should be in [0, 1] range
            self.assertTrue(torch.all(features >= 0.0), "Negative features found")
            self.assertTrue(torch.all(features <= 2.0), "Features exceed reasonable bounds")
    
    def test_feature_consistency(self):
        """Test that identical inputs produce identical features."""
        level = self.test_levels[0]
        reachability_result = create_mock_reachability_result()
        
        features1 = self.feature_encoder.encode_reachability(
            reachability_result, level.data, level.ninja_pos
        )
        features2 = self.feature_encoder.encode_reachability(
            reachability_result, level.data, level.ninja_pos
        )
        
        self.assertTrue(torch.allclose(features1, features2), 
                       "Inconsistent feature encoding")
    
    def test_feature_sensitivity(self):
        """Test that features change appropriately with input changes."""
        level = self.test_levels[0]
        
        # Test switch state sensitivity
        result1 = create_mock_reachability_result(switch_states={'switch_1': False})
        result2 = create_mock_reachability_result(switch_states={'switch_1': True})
        
        features1 = self.feature_encoder.encode_reachability(result1, level.data, level.ninja_pos)
        features2 = self.feature_encoder.encode_reachability(result2, level.data, level.ninja_pos)
        
        # Switch state features should be different
        switch_features1 = features1[8:24]
        switch_features2 = features2[8:24]
        
        self.assertFalse(torch.allclose(switch_features1, switch_features2),
                        "Features not sensitive to switch state changes")
```

### Integration Tests
```python
class TestFeatureExtractorIntegration(unittest.TestCase):
    def setUp(self):
        self.tiered_system = TieredReachabilitySystem()
        self.feature_extractor = ReachabilityFeatureExtractor(self.tiered_system)
    
    def test_performance_targets(self):
        """Test that feature extraction meets performance targets."""
        for level in load_performance_test_levels():
            # Test fast extraction (Tier 1)
            start_time = time.perf_counter()
            features = self.feature_extractor.extract_features(
                level.ninja_pos, level.data, level.switch_states, "fast"
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            self.assertLess(elapsed_ms, 2.0, f"Fast extraction too slow: {elapsed_ms}ms")
            self.assertEqual(features.shape, (64,), "Incorrect feature dimensions")
    
    def test_cache_effectiveness(self):
        """Test that caching improves performance."""
        level = load_test_level("complex-path-switch-required")
        
        # First extraction (cache miss)
        start_time = time.perf_counter()
        features1 = self.feature_extractor.extract_features(
            level.ninja_pos, level.data, level.switch_states, "balanced"
        )
        first_time = (time.perf_counter() - start_time) * 1000
        
        # Second extraction (cache hit)
        start_time = time.perf_counter()
        features2 = self.feature_extractor.extract_features(
            level.ninja_pos, level.data, level.switch_states, "balanced"
        )
        second_time = (time.perf_counter() - start_time) * 1000
        
        # Cache hit should be much faster
        self.assertLess(second_time, first_time * 0.1, "Cache not effective")
        self.assertTrue(torch.allclose(features1, features2), "Cache inconsistency")
```

### Feature Analysis Tests
```python
class TestFeatureAnalysis(unittest.TestCase):
    def test_feature_distributions(self):
        """Analyze feature distributions across test levels."""
        feature_extractor = ReachabilityFeatureExtractor(TieredReachabilitySystem())
        all_features = []
        
        for level in load_all_test_levels():
            features = feature_extractor.extract_features(
                level.ninja_pos, level.data, level.switch_states, "balanced"
            )
            all_features.append(features.numpy())
        
        feature_matrix = np.array(all_features)
        
        # Analyze each feature dimension
        for i in range(64):
            feature_values = feature_matrix[:, i]
            
            # Check for reasonable variance
            variance = np.var(feature_values)
            self.assertGreater(variance, 0.001, f"Feature {i} has too low variance")
            
            # Check for reasonable range
            min_val, max_val = np.min(feature_values), np.max(feature_values)
            range_val = max_val - min_val
            self.assertGreater(range_val, 0.1, f"Feature {i} has too small range")
    
    def test_feature_correlations(self):
        """Test that features are not overly correlated."""
        feature_extractor = ReachabilityFeatureExtractor(TieredReachabilitySystem())
        all_features = []
        
        for level in load_all_test_levels():
            features = feature_extractor.extract_features(
                level.ninja_pos, level.data, level.switch_states, "balanced"
            )
            all_features.append(features.numpy())
        
        feature_matrix = np.array(all_features)
        correlation_matrix = np.corrcoef(feature_matrix.T)
        
        # Check for excessive correlations
        high_correlations = np.where(np.abs(correlation_matrix) > 0.9)
        high_corr_pairs = [(i, j) for i, j in zip(high_correlations[0], high_correlations[1]) if i != j]
        
        # Allow some high correlations but not too many
        self.assertLess(len(high_corr_pairs), 10, 
                       f"Too many highly correlated features: {len(high_corr_pairs)}")
```

## Success Criteria

### Performance Requirements
- **Feature Extraction Time**: <2ms for Tier 1, <5ms for Tier 2
- **Memory Usage**: <10MB additional memory for caching
- **Cache Hit Rate**: >80% during typical RL training

### Quality Requirements
- **Feature Stability**: Consistent encoding for identical inputs
- **Feature Sensitivity**: Appropriate response to input changes
- **Feature Bounds**: All features within reasonable ranges [0, 2]
- **Feature Variance**: Sufficient variance across test levels

### Integration Requirements
- **API Compatibility**: Clean interface for npp-rl integration
- **Documentation**: Complete API documentation and examples
- **Testing Coverage**: >95% code coverage for feature encoding

## Risk Mitigation

### Technical Risks
1. **Feature Informativeness**: Validate that compact features retain essential information
2. **Encoding Stability**: Ensure consistent encoding across different scenarios
3. **Performance Regression**: Monitor feature extraction performance

### Integration Risks
1. **RL Training Impact**: A/B test compact features vs detailed analysis
2. **Feature Learning**: Validate that RL agent can learn from compact features
3. **Gradient Flow**: Ensure features have good gradient properties

## Deliverables

1. **CompactReachabilityFeatures**: Core feature encoding implementation
2. **ReachabilityFeatureExtractor**: High-level interface for RL integration
3. **Feature Analysis Tools**: Visualization and analysis utilities
4. **Integration Documentation**: Complete guide for npp-rl integration
5. **Performance Benchmarks**: Comprehensive performance analysis

## Timeline

- **Week 1**: Core feature encoder implementation and unit testing
- **Week 2**: Integration interface and caching system
- **Week 3**: Validation, testing, and performance optimization
- **Week 4**: Documentation, analysis tools, and integration examples

## References

1. **Strategic Analysis**: `/workspace/nclone/docs/reachability_analysis_integration_strategy.md`
2. **HGT Architecture**: `/workspace/npp-rl/npp_rl/feature_extractors/hgt_multimodal.py`
3. **Tiered System**: Task 001 - Implement Tiered Reachability System
4. **Integration Requirements**: `/workspace/npp-rl/tasks/TASK_002_integrate_reachability_system.md`