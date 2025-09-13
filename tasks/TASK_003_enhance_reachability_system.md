# Task 003: Enhance Reachability System for RL Integration

## Overview
Enhance the existing reachability analysis system in nclone to better support the RL agent's needs, including performance optimizations, caching improvements, and additional analysis capabilities required for hierarchical RL and curiosity-driven exploration.

## Context Reference
See [npp-rl comprehensive technical roadmap](../../../npp-rl/docs/comprehensive_technical_roadmap.md) Section 1.2: "Physics-Aware Reachability Analysis Strategy" and Section 1.3: "Integration with RL Architecture"

## Requirements

### Primary Objectives
1. **Optimize reachability analysis performance** for real-time RL decision making
2. **Enhance caching system** for efficient repeated queries
3. **Add subgoal identification** for hierarchical RL integration
4. **Implement frontier detection** for curiosity-driven exploration
5. **Add strategic planning support** for level completion heuristics
6. **Enhance entity handling** for comprehensive collision and hazard analysis

### Current System Analysis
The existing reachability system is implemented in the `nclone/graph/reachability/` package:

**Core Components:**
- `ReachabilityAnalyzer` class in `nclone/graph/reachability/reachability_analyzer.py`
- `PositionValidator` class in `nclone/graph/reachability/position_validator.py`
- `CollisionChecker` class in `nclone/graph/reachability/collision_checker.py`
- `PhysicsMovement` class in `nclone/graph/reachability/physics_movement.py`
- `GameMechanics` class in `nclone/graph/reachability/game_mechanics.py`

**Current Functionality:**
- Physics-based BFS reachability analysis
- Switch-door dependency handling
- Dynamic entity integration
- Basic subgoal identification
- Modular component architecture

### Enhancements Required

#### Entity Integration
1. **Hazard Entity Handling**
   - Toggled-on toggle mines (`EntityToggleMine` from `nclone/entity_classes/entity_toggle_mine.py`) as blocking hazards
   - Drone movement patterns and collision zones:
     - `EntityDroneZap` from `nclone/entity_classes/entity_drone_zap.py`
     - `EntityMiniDrone` from `nclone/entity_classes/entity_mini_drone.py`
   - Thwump crush zones and safe positioning (`EntityThwump` from `nclone/entity_classes/entity_thwump.py`)
   - Shwump active/inactive state collision properties (`EntityShoveThwump` from `nclone/entity_classes/entity_shove_thwump.py`)

2. **Collision Entity Handling**
   - One way platform directional traversal rules (`EntityOneWayPlatform` from `nclone/entity_classes/entity_one_way_platform.py`)
   - Safe side positioning for thwumps (using `EntityThwump` safe zones)
   - Inactive shwump collision detection on all sides (using `EntityShoveThwump` state management)
   - Bounce block physics integration for trajectory calculation (`EntityBounceBlock` from `nclone/entity_classes/entity_bounce_block.py`)

3. **Movement Type Simplifications**
   - Wall slide movement type excluded from reachability analysis (reserved for precise subcell-level agent actions)
   - Focus on primary movement types: walk, jump, fall, wall jump
   - Wall jump analysis critical for reaching positions otherwise inaccessible via standard jumping

#### Performance Optimizations
1. **Incremental Updates**
   - Cache reachability results between similar queries
   - Only recompute when ninja position or switch states change significantly
   - Implement dirty region tracking for partial updates

2. **Multi-Resolution Analysis**
   - Coarse-grained analysis for distant areas
   - Fine-grained analysis for nearby areas
   - Adaptive resolution based on distance from ninja

3. **Parallel Processing**
   - Parallelize BFS exploration where possible
   - Async entity state updates
   - Background precomputation for likely scenarios

#### Enhanced Caching System
1. **Intelligent Cache Management**
   - LRU cache with configurable size limits
   - Cache invalidation based on game state changes
   - Persistent caching across episodes

2. **Cache Key Optimization**
   - Efficient hashing of game states
   - Approximate matching for similar states
   - Cache warming for common scenarios

#### Subgoal Identification Enhancements
1. **Strategic Subgoals**
   - Exit switch and door identification
   - Door switch prioritization
   - Gold collection opportunities
   - Hazard avoidance waypoints

2. **Hierarchical Subgoals**
   - High-level strategic goals (complete level)
   - Mid-level tactical goals (activate switch)
   - Low-level movement goals (reach position)

#### Frontier Detection for Curiosity
1. **Exploration Frontiers**
   - Identify boundaries between reachable and unreachable areas
   - Detect areas that might become reachable with switch activation
   - Classify exploration value of different areas

2. **Curiosity Metrics**
   - Distance-based exploration bonuses
   - Novelty detection for unexplored areas
   - Risk-reward analysis for dangerous areas

## Acceptance Criteria

### Performance Requirements
1. **Real-time Analysis**: Reachability analysis completes in <10ms for typical levels
2. **Large Level Support**: Full-size levels (42x23) analyze in <100ms
3. **Cache Efficiency**: >80% cache hit rate for repeated queries
4. **Memory Usage**: Peak memory usage <50MB for reachability data

### Functional Requirements
1. **Backward Compatibility**: Existing reachability API continues to work
2. **Enhanced API**: New methods for RL-specific functionality
3. **Robust Caching**: Cache survives switch state changes and ninja movement
4. **Accurate Subgoals**: Subgoal identification matches expected strategic goals
5. **Entity Awareness**: Proper handling of all hazards and collision entities
6. **Movement Type Coverage**: Core movement types (walk, jump, fall, wall jump) fully supported

### Quality Requirements
1. **Thread Safety**: Safe for concurrent access from RL training
2. **Error Handling**: Graceful degradation when analysis fails
3. **Debugging Support**: Detailed logging and visualization options
4. **Documentation**: Comprehensive API documentation

## Test Scenarios

### Performance Testing
```python
# Test real-time performance requirements
def test_real_time_performance():
    analyzer = EnhancedReachabilityAnalyzer()
    level_data = load_map('test_maps/large_level.nmap')
    
    # Test multiple queries in sequence (simulating RL training)
    ninja_positions = generate_ninja_trajectory(1000)  # 1000 positions
    
    start_time = time.time()
    for pos in ninja_positions:
        result = analyzer.analyze_reachability(level_data, pos, {})
    total_time = time.time() - start_time
    
    avg_time = total_time / len(ninja_positions)
    assert avg_time < 0.01, f"Average analysis time {avg_time:.3f}s exceeds 10ms limit"

# Test cache efficiency
def test_cache_efficiency():
    analyzer = EnhancedReachabilityAnalyzer()
    level_data = load_map('test_maps/switch_puzzle_maze.nmap')
    
    # Perform repeated queries with slight variations
    base_pos = (100, 400)
    positions = [(base_pos[0] + i, base_pos[1] + j) 
                for i in range(-5, 6) for j in range(-5, 6)]
    
    # First pass - populate cache
    for pos in positions:
        analyzer.analyze_reachability(level_data, pos, {})
    
    # Second pass - should hit cache
    start_time = time.time()
    for pos in positions:
        analyzer.analyze_reachability(level_data, pos, {})
    cached_time = time.time() - start_time
    
    cache_hit_rate = analyzer.get_cache_hit_rate()
    assert cache_hit_rate > 0.8, f"Cache hit rate {cache_hit_rate:.2f} below 80% target"
```

### Functional Testing
```python
# Test subgoal identification
def test_subgoal_identification():
    analyzer = EnhancedReachabilityAnalyzer()
    level_data = load_map('test_maps/complex_multi_switch_level.nmap')
    ninja_pos = (50, 400)
    
    result = analyzer.analyze_reachability(level_data, ninja_pos, {})
    
    # Should identify strategic subgoals
    subgoals = result.subgoals
    subgoal_types = [sg[2] for sg in subgoals]  # Extract goal types
    
    assert 'exit_switch' in subgoal_types, "Should identify exit switch as subgoal"
    assert 'door_switch' in subgoal_types, "Should identify door switches as subgoals"
    assert len([sg for sg in subgoals if sg[2] == 'door_switch']) >= 2, "Should find multiple door switches"

# Test wall jump reachability
def test_wall_jump_reachability():
    analyzer = EnhancedReachabilityAnalyzer()
    level_data = load_map('test_maps/wall_jump_test_level.nmap')
    ninja_pos = (50, 400)
    
    result = analyzer.analyze_reachability(level_data, ninja_pos, {})
    
    # Should identify positions reachable only via wall jump
    wall_jump_positions = [pos for pos in result.reachable_positions 
                          if result.get_movement_type_to_position(pos) == 'wall_jump']
    
    assert len(wall_jump_positions) > 0, "Should find wall jump accessible positions"
    
    # Verify specific wall jump scenarios
    high_ledge_pos = (200, 200)  # Position accessible only via wall jump
    assert high_ledge_pos in result.reachable_positions, \
           "Should reach high ledge via wall jump"

# Test frontier detection
def test_frontier_detection():
    analyzer = EnhancedReachabilityAnalyzer()
    level_data = load_map('test_maps/maze_with_unreachable_areas.nmap')
    ninja_pos = (50, 400)
    
    result = analyzer.analyze_reachability(level_data, ninja_pos, {})
    
    # Should identify exploration frontiers
    frontiers = analyzer.get_exploration_frontiers(result)
    
    assert len(frontiers) > 0, "Should identify exploration frontiers"
    
    # Test frontier classification
    for frontier_pos in frontiers:
        classification = analyzer.classify_frontier(frontier_pos, level_data, result)
        assert classification in ['unreachable', 'locked', 'dangerous'], \
               f"Invalid frontier classification: {classification}"
```

### Integration Testing
```python
# Test RL integration compatibility
def test_rl_integration():
    analyzer = EnhancedReachabilityAnalyzer()
    level_data = load_map('test_maps/exploration_test.nmap')
    
    # Simulate RL training scenario
    ninja_pos = (50, 400)
    switch_states = {}
    
    # Get reachable subgoals (for HRL)
    reachable_subgoals = analyzer.get_reachable_subgoals(
        ninja_pos, level_data, switch_states
    )
    
    assert isinstance(reachable_subgoals, list), "Should return list of subgoals"
    assert len(reachable_subgoals) > 0, "Should find some reachable subgoals"
    
    # Get curiosity bonuses (for exploration)
    test_targets = [(100, 400), (200, 300), (500, 100)]
    for target in test_targets:
        bonus = analyzer.compute_curiosity_bonus(ninja_pos, target, level_data)
        assert 0.0 <= bonus <= 1.0, f"Curiosity bonus {bonus} outside valid range"
```

## Implementation Steps

### Phase 1: Performance Optimization
1. **Profile Current Performance**
   ```python
   # Add performance profiling to existing system
   import cProfile
   import pstats
   
   def profile_reachability_analysis():
       profiler = cProfile.Profile()
       profiler.enable()
       
       # Run reachability analysis
       analyzer = ReachabilityAnalyzer(TrajectoryCalculator())
       level_data = load_map('test_maps/large_level.nmap')
       result = analyzer.analyze_reachability(level_data, (50, 400), {})
       
       profiler.disable()
       stats = pstats.Stats(profiler)
       stats.sort_stats('cumulative')
       stats.print_stats(20)  # Top 20 time consumers
   ```

2. **Implement Incremental Updates**
   ```python
   class EnhancedReachabilityAnalyzer(ReachabilityAnalyzer):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.cache = {}
           self.last_analysis = None
           self.dirty_regions = set()
       
       def analyze_reachability_incremental(self, level_data, ninja_pos, switch_states):
           # Check if we can use cached results
           cache_key = self._generate_cache_key(ninja_pos, switch_states)
           
           if cache_key in self.cache and not self._has_significant_changes(ninja_pos):
               return self.cache[cache_key]
           
           # Perform full or partial analysis
           if self._should_do_full_analysis(ninja_pos, switch_states):
               result = self._full_analysis(level_data, ninja_pos, switch_states)
           else:
               result = self._incremental_analysis(level_data, ninja_pos, switch_states)
           
           self.cache[cache_key] = result
           return result
   ```

### Phase 2: Enhanced Caching
1. **Implement Intelligent Cache Management**
   ```python
   from collections import OrderedDict
   import hashlib
   
   class ReachabilityCache:
       def __init__(self, max_size=1000):
           self.cache = OrderedDict()
           self.max_size = max_size
           self.hit_count = 0
           self.miss_count = 0
       
       def get(self, key):
           if key in self.cache:
               # Move to end (most recently used)
               self.cache.move_to_end(key)
               self.hit_count += 1
               return self.cache[key]
           
           self.miss_count += 1
           return None
       
       def put(self, key, value):
           if key in self.cache:
               self.cache.move_to_end(key)
           else:
               if len(self.cache) >= self.max_size:
                   # Remove least recently used
                   self.cache.popitem(last=False)
           
           self.cache[key] = value
       
       def get_hit_rate(self):
           total = self.hit_count + self.miss_count
           return self.hit_count / total if total > 0 else 0.0
   ```

### Phase 3: Entity Integration and Enhancement
1. **Entity State Management**
   ```python
   # Add to nclone/graph/reachability/reachability_analyzer.py
   from ...entity_classes.entity_toggle_mine import EntityToggleMine
   from ...entity_classes.entity_drone_zap import EntityDroneZap
   from ...entity_classes.entity_mini_drone import EntityMiniDrone
   from ...entity_classes.entity_thwump import EntityThwump
   from ...entity_classes.entity_shove_thwump import EntityShoveThwump
   from ...entity_classes.entity_one_way_platform import EntityOneWayPlatform
   from ...entity_classes.entity_bounce_block import EntityBounceBlock
   from ...constants.entity_types import EntityType
   
   def analyze_entity_interactions(self, level_data, ninja_pos, switch_states):
       entity_constraints = []
       
       # Handle hazard entities
       for toggle_mine in level_data.toggle_mines:
           if self._is_toggled_on(toggle_mine, switch_states):
               # Add blocking constraint for active toggle mine
               entity_constraints.append({
                   'type': 'blocking_hazard',
                   'position': toggle_mine.position,
                   'entity': toggle_mine
               })
       
       # Handle different drone types
       for drone in level_data.drones:
           if isinstance(drone, EntityDroneZap):
               # Add movement zone constraints for zap drone paths
               drone_path = self._calculate_drone_zap_path(drone)
           elif isinstance(drone, EntityMiniDrone):
               # Add movement zone constraints for mini drone paths
               drone_path = self._calculate_mini_drone_path(drone)
           else:
               drone_path = self._calculate_generic_drone_path(drone)
           
           entity_constraints.append({
               'type': 'moving_hazard',
               'positions': drone_path,
               'entity': drone
           })
       
       # Handle thwump entities (both types)
       for thwump in level_data.thwumps:
           if isinstance(thwump, EntityThwump):
               # Regular thwump - vertical crushing
               crush_zone = self._calculate_thwump_crush_zone(thwump)
               safe_positions = self._calculate_thwump_safe_positions(thwump)
           elif isinstance(thwump, EntityShoveThwump):
               # Shove thwump (shwump) - horizontal pushing
               crush_zone = self._calculate_shwump_crush_zone(thwump)
               safe_positions = self._calculate_shwump_safe_positions(thwump)
           
           entity_constraints.append({
               'type': 'conditional_hazard',
               'crush_zone': crush_zone,
               'safe_positions': safe_positions,
               'entity': thwump
           })
       
       # Handle shwumps (EntityShoveThwump) separately for state checking
       for shwump in level_data.shwumps:
           if isinstance(shwump, EntityShoveThwump) and not shwump.is_active:
               # Inactive shwump acts as collision block on all sides
               entity_constraints.append({
                   'type': 'collision_block',
                   'position': shwump.position,
                   'entity': shwump
               })
       
       # Handle collision entities
       for platform in level_data.one_way_platforms:
           if isinstance(platform, EntityOneWayPlatform):
               entity_constraints.append({
                   'type': 'directional_platform',
                   'position': platform.position,
                   'direction': platform.direction,
                   'entity': platform
               })
       
       for bounce_block in level_data.bounce_blocks:
           if isinstance(bounce_block, EntityBounceBlock):
               entity_constraints.append({
                   'type': 'bounce_surface',
                   'position': bounce_block.position,
                   'bounce_vector': bounce_block.bounce_direction,
                   'entity': bounce_block
               })
       
       return entity_constraints
   ```

2. **Wall Jump Analysis Integration**
   ```python
   # Add to nclone/graph/reachability/physics_movement.py
   def analyze_wall_jump_opportunities(self, src_pos, level_data, entity_constraints):
       """Identify wall jump trajectories for enhanced reachability."""
       wall_jump_targets = []
       
       # Find nearby walls suitable for wall jumping
       nearby_walls = self._find_walls_in_radius(src_pos, wall_jump_detection_radius=100)
       
       for wall in nearby_walls:
           # Check if wall can support wall jump
           if self._can_wall_jump_from(wall, src_pos, level_data):
               # Calculate possible wall jump trajectories
               jump_targets = self._calculate_wall_jump_trajectories(
                   src_pos, wall, level_data, entity_constraints
               )
               
               for target_pos, trajectory in jump_targets:
                   # Validate target position is safe and reachable
                   if self._is_valid_wall_jump_target(target_pos, level_data, entity_constraints):
                       wall_jump_targets.append({
                           'target': target_pos,
                           'trajectory': trajectory,
                           'wall_position': wall.position,
                           'movement_type': 'wall_jump'
                       })
       
       return wall_jump_targets
   
   def _calculate_wall_jump_trajectories(self, src_pos, wall, level_data, entity_constraints):
       """Calculate physics-accurate wall jump trajectories."""
       trajectories = []
       
       # Use existing trajectory calculator with wall jump physics
       wall_approach_trajectories = self.trajectory_calculator.calculate_wall_approach(
           src_pos, wall.position
       )
       
       for approach in wall_approach_trajectories:
           if approach.is_feasible:
               # Calculate wall jump launch physics
               wall_jump_velocity = self._calculate_wall_jump_velocity(
                   approach.final_velocity, wall.normal_vector
               )
               
               # Calculate trajectory from wall jump launch
               jump_trajectory = self.trajectory_calculator.calculate_trajectory(
                   wall.position, wall_jump_velocity, level_data
               )
               
               if jump_trajectory.is_feasible:
                   trajectories.append((jump_trajectory.final_position, jump_trajectory))
       
       return trajectories
   ```

3. **Enhanced Subgoal Identification**
   ```python
   def identify_strategic_subgoals(self, level_data, reachability_state):
       subgoals = []
       
       # Find exit-related subgoals
       exit_switch = self._find_exit_switch(level_data)
       if exit_switch and self._is_reachable(exit_switch.position, reachability_state):
           subgoals.append((exit_switch.position[0], exit_switch.position[1], 'exit_switch'))
       
       exit_door = self._find_exit_door(level_data)
       if exit_door and self._is_reachable(exit_door.position, reachability_state):
           subgoals.append((exit_door.position[0], exit_door.position[1], 'exit_door'))
       
       # Find door switch subgoals
       for door_switch in self._find_door_switches(level_data):
           if self._is_reachable(door_switch.position, reachability_state):
               subgoals.append((door_switch.position[0], door_switch.position[1], 'door_switch'))
       
       # Find gold collection subgoals
       for gold in self._find_gold(level_data):
           if self._is_reachable(gold.position, reachability_state):
               subgoals.append((gold.position[0], gold.position[1], 'gold'))
       
       return subgoals
   ```

2. **Frontier Detection Implementation**
   ```python
   def get_exploration_frontiers(self, reachability_state):
       frontiers = []
       
       # Find boundaries between reachable and unreachable areas
       for reachable_pos in reachability_state.reachable_positions:
           neighbors = self._get_neighboring_positions(reachable_pos)
           for neighbor in neighbors:
               if neighbor not in reachability_state.reachable_positions:
                   # This is a frontier position
                   frontiers.append(neighbor)
       
       return list(set(frontiers))  # Remove duplicates
   
   def classify_frontier(self, frontier_pos, level_data, reachability_state):
       # Check if frontier is behind a locked door
       if self._is_behind_locked_door(frontier_pos, level_data, reachability_state):
           return 'locked'
       
       # Check if frontier is dangerous (near hazards)
       if self._is_near_hazards(frontier_pos, level_data):
           return 'dangerous'
       
       # Otherwise, it's truly unreachable
       return 'unreachable'
   ```

### Phase 4: RL Integration API
1. **Create RL-Specific Interface**
   ```python
   class RLReachabilityInterface:
       def __init__(self, analyzer):
           self.analyzer = analyzer
       
       def get_reachable_subgoals(self, ninja_pos, level_data, switch_states):
           """Get list of currently reachable subgoals for HRL."""
           result = self.analyzer.analyze_reachability(level_data, ninja_pos, switch_states)
           return [f"{sg[2]}_{sg[0]}_{sg[1]}" for sg in result.subgoals]
       
       def compute_curiosity_bonus(self, ninja_pos, target_pos, level_data):
           """Compute exploration bonus for curiosity-driven RL."""
           result = self.analyzer.analyze_reachability(level_data, ninja_pos, {})
           
           if self._is_reachable(target_pos, result):
               return 1.0  # High curiosity for reachable areas
           
           frontiers = self.analyzer.get_exploration_frontiers(result)
           if target_pos in frontiers:
               return 0.5  # Medium curiosity for frontier areas
           
           return 0.0  # No curiosity for unreachable areas
       
       def plan_level_completion(self, ninja_pos, level_data, switch_states):
           """Generate strategic plan for level completion."""
           result = self.analyzer.analyze_reachability(level_data, ninja_pos, switch_states)
           
           # Implement level completion heuristic
           plan = []
           
           # Step 1: Can we reach exit switch?
           exit_switch_pos = self._find_exit_switch_position(level_data)
           if self._is_reachable(exit_switch_pos, result):
               plan.append(f"navigate_to_exit_switch")
               plan.append(f"activate_exit_switch")
           else:
               # Find required door switches
               blocking_doors = self._find_blocking_doors(ninja_pos, exit_switch_pos, level_data, result)
               for door in blocking_doors:
                   switch_pos = self._find_controlling_switch(door, level_data)
                   plan.append(f"navigate_to_door_switch_{door.id}")
                   plan.append(f"activate_door_switch_{door.id}")
               
               plan.append(f"navigate_to_exit_switch")
               plan.append(f"activate_exit_switch")
           
           # Step 2: Navigate to exit door
           plan.append(f"navigate_to_exit_door")
           
           return plan
   ```

## Success Metrics
- **Performance**: <10ms analysis time for typical levels, <100ms for large levels
- **Cache Efficiency**: >80% cache hit rate during RL training
- **Memory Usage**: <50MB peak memory usage
- **API Completeness**: All RL integration methods implemented and tested
- **Backward Compatibility**: Existing code continues to work unchanged

## Dependencies
- Existing reachability analysis system
- Test maps from Task 002
- Performance profiling tools

## Estimated Effort
- **Time**: 1-2 weeks
- **Complexity**: Medium-High (performance optimization)
- **Risk**: Medium (changes to core system)

## Notes
- Maintain backward compatibility with existing reachability API
- Focus on performance optimizations that don't compromise accuracy
- Consider future extensibility for additional RL algorithms
- Coordinate with npp-rl team for API requirements