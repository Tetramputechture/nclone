# Graph Module Analysis for Production Cleanup

## Current Structure (43 files total)

### Core Production Components (KEEP - 15 files)

#### Essential Graph Infrastructure
1. **`graph/__init__.py`** - Package exports
2. **`graph/common.py`** - Core data structures (GraphData, EdgeType, NodeType)
3. **`graph/hierarchical_builder.py`** - Main graph builder (used by gym env, test env)
4. **`graph/graph_construction.py`** - Graph construction logic
5. **`graph/edge_building.py`** - Edge creation logic
6. **`graph/level_data.py`** - Level data structures
7. **`graph/subgoal_planner.py`** - Enhanced with hierarchical completion algorithm

#### Production Reachability System
8. **`graph/reachability/__init__.py`** - Package exports
9. **`graph/reachability/tiered_system.py`** - Main production reachability system
10. **`graph/reachability/opencv_flood_fill.py`** - Production OpenCV backend
11. **`graph/reachability/reachability_types.py`** - Shared data types
12. **`graph/reachability/reachability_state.py`** - State management
13. **`graph/reachability/reachability_analyzer.py`** - Main analyzer (used by graph construction)

#### Supporting Infrastructure
14. **`graph/precise_collision.py`** - Collision detection (used by multiple components)
15. **`graph/hazard_system.py`** - Hazard classification (used by trajectory calculator)

### Potentially Unused Components (CANDIDATES FOR REMOVAL - 28 files)

#### Duplicate/Legacy Reachability Components
1. **`graph/reachability/subgoal_planner.py`** - DUPLICATE of main SubgoalPlanner
2. **`graph/reachability/simplified_physics_analyzer.py`** - Legacy physics analyzer
3. **`graph/reachability/flood_fill_approximator.py`** - Legacy flood fill
4. **`graph/reachability/enhanced_flood_fill.py`** - Legacy enhanced flood fill
5. **`graph/reachability/entity_aware_flood_fill.py`** - Legacy entity-aware flood fill
6. **`graph/reachability/wall_jump_analyzer.py`** - Legacy wall jump analyzer
7. **`graph/reachability/physics_movement.py`** - Legacy physics movement

#### Multiple Collision Systems (Consolidate?)
8. **`graph/reachability/collision_checker.py`** - Reachability-specific collision
9. **`graph/optimized_collision.py`** - Optimized collision variant

#### Multiple Validation Systems
10. **`graph/reachability/position_validator.py`** - Base position validator
11. **`graph/reachability/entity_aware_validator.py`** - Entity-aware validator

#### Integration Layers (May be obsolete)
12. **`graph/reachability/subgoal_integration.py`** - Integration layer for old system
13. **`graph/reachability/rl_integration.py`** - RL integration layer
14. **`graph/reachability/game_mechanics.py`** - Game mechanics integration
15. **`graph/reachability/hazard_integration.py`** - Hazard integration

#### Caching Systems (May be unused)
16. **`graph/reachability/reachability_cache.py`** - Reachability caching
17. **`graph/performance_cache.py`** - Performance caching

#### Multiple Visualization Systems
18. **`graph/visualization.py`** - Base visualization
19. **`graph/physics_accurate_visualization.py`** - Physics-accurate visualization
20. **`graph/enhanced_debug_overlay.py`** - Enhanced debug overlay
21. **`graph/reachability/visualization_system.py`** - Reachability visualization

#### Analysis Components (May be unused)
22. **`graph/reachability/frontier_detector.py`** - Frontier detection
23. **`graph/reachability/level_completion_analyzer.py`** - Level completion analysis
24. **`graph/feature_extraction.py`** - Feature extraction
25. **`graph/movement_classifier.py`** - Movement classification
26. **`graph/trajectory_calculator.py`** - Trajectory calculation
27. **`graph/navigation.py`** - Navigation/pathfinding
28. **`graph/segment_consolidator.py`** - Segment consolidation

## Detailed Usage Analysis

### Files Actually Imported by External Code:
- `hierarchical_builder.py` - Used by gym env, test env, debug overlay
- `common.py` - Used by debug overlay, gym env
- `level_data.py` - Used by gym env
- `subgoal_planner.py` - Used by test env
- `reachability_analyzer.py` - Used by test env, graph construction

### Files Only Used Internally:
- Most reachability/* files are only used within the reachability package
- Many visualization and analysis components have no external usage

### Confirmed Duplicates:
- `graph/reachability/subgoal_planner.py` vs `graph/subgoal_planner.py`
- Multiple collision detection systems
- Multiple flood fill implementations (we use OpenCV now)

## Recommendations

### Phase 1: Remove Confirmed Duplicates (Safe)
1. Remove `graph/reachability/subgoal_planner.py` (duplicate)
2. Remove legacy flood fill implementations
3. Remove legacy physics analyzers

### Phase 2: Consolidate Similar Systems (Needs Review)
1. Consolidate collision detection systems
2. Consolidate visualization systems
3. Review integration layers

### Phase 3: Remove Unused Analysis Components (Needs Confirmation)
1. Remove unused trajectory/movement analysis
2. Remove unused navigation components
3. Remove unused caching systems

## Questions for Confirmation

1. **Visualization Systems**: Do we need multiple visualization systems, or can we keep just one?
2. **Collision Detection**: Can we consolidate to just `precise_collision.py`?
3. **Navigation**: Is the `navigation.py` pathfinding system used, or do we rely on the hierarchical graph?
4. **Analysis Components**: Are `movement_classifier.py`, `trajectory_calculator.py`, `feature_extraction.py` used in production?
5. **Integration Layers**: Are the various integration components in reachability/ still needed after our consolidation?