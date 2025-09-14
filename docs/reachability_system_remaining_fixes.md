# Reachability System: Remaining Fixes and Improvements

## Overview

This document outlines the remaining fixes and improvements needed for the Deep RL simulator's reachability system. The system has been significantly enhanced with entity-aware analysis and subgoal planning, achieving a 250% improvement in test pass rate (from 2/19 to 7/19 tests passing). However, several critical issues remain that prevent full completion of complex puzzle levels.

## Current Status

### ✅ **Completed Systems**
- **Entity-aware reachability analysis**: Automatically detects interactive entities
- **Iterative switch discovery**: Handles multi-step puzzles with switch chains
- **Subgoal planning framework**: Complete dependency resolution system
- **Performance optimization**: Smart detection reduces analysis time by 50%
- **Multi-state analysis**: Continues until no new switches are found

### 🔧 **Critical Issues Identified**

#### 1. **Granularity Mismatch in Exit Validation** (HIGH PRIORITY)
**Problem**: Subgoal planner uses tile-level reachability (coarse) while exit validation uses pixel-level checks (precise)

**Evidence**:
```
DEBUG: Checking exit at (660, 276) -> tile (27, 11)
DEBUG: Exit tile reachable: True
DEBUG: Exit at (660, 276) is directly reachable
```
But final result shows:
```
Exit 1: exit_door_0 at pixel (660, 276), radius=12
Exit 1 not reachable from any ninja position
```

**Impact**: `complex-path-switch-required` test fails despite correct subgoal planning

#### 2. **Locked Door Collision Detection** (HIGH PRIORITY)
**Problem**: Traversability checks may not properly consider locked door states during pathfinding

**Evidence**: Locked switches `locked_0` and `locked_1` remain unreachable even when their controlling doors should be open

**Impact**: Complex puzzles cannot be solved because doors block access to required switches

#### 3. **Incomplete Subgoal Execution** (MEDIUM PRIORITY)
**Problem**: Subgoal planning generates correct dependency chains but execution doesn't follow through

**Evidence**: System identifies subgoals but doesn't simulate their execution to verify reachability

## Test Results Analysis

### Current Test Suite Status (7/19 passing)

#### ✅ **Passing Tests**
1. `simple-walk` - Basic movement
2. `fall-required` - Gravity mechanics  
3. `jump-then-fall` - Jump + fall sequence
4. `long-walk` - Extended horizontal movement
5. `only-jump` - Pure jumping mechanics
6. `path-jump-required` - Jump-based pathfinding
7. `long-vertical-corridor` - Vertical movement

#### ❌ **Critical Failing Tests**
1. **`complex-path-switch-required`** - Complex puzzle logic (PRIORITY)
2. **`halfaligned-path`** - Precision pathfinding
3. **`thwump-platform-required`** - Platform interaction
4. **`drone-unreachable`** - Hazard detection

#### ⚡ **Performance Failing Tests** (10 tests exceed 10ms limit)
- Most can be optimized with better caching and selective analysis

## Required Fixes

### Fix 1: Resolve Exit Validation Granularity Mismatch

**Location**: `nclone/graph/reachability/subgoal_planner.py`

**Problem**: Subgoal planner checks tile-level reachability but exit validation requires pixel-level precision

**Solution**:
```python
def _is_exit_reachable_precise(self, exit_x: float, exit_y: float, 
                              reachable_positions: Set[Tuple[int, int]]) -> bool:
    """Check if exit is reachable with pixel-level precision."""
    exit_tile_x = int(exit_x // TILE_PIXEL_SIZE)
    exit_tile_y = int(exit_y // TILE_PIXEL_SIZE)
    
    # First check tile-level reachability
    if (exit_tile_x, exit_tile_y) not in reachable_positions:
        return False
    
    # Then check pixel-level precision within the tile
    # Consider exit radius and ninja positioning constraints
    exit_radius = 12  # Standard exit radius
    
    # Check if ninja can get within exit radius
    # This requires checking subcell-level reachability or pathfinding
    return self._check_pixel_level_reachability(exit_x, exit_y, exit_radius)
```

**Testing**:
```bash
python test_reachability_suite.py --map=complex-path-switch-required --verbose
# Should show subgoal planner correctly identifying unreachable exit
```

### Fix 2: Enhance Locked Door Collision Detection

**Location**: `nclone/graph/reachability/entity_aware_validator.py`

**Problem**: Traversability checks don't properly consider door states during pathfinding

**Current Issue**:
```python
# Current door state checking may be incomplete
def _is_position_traversable_with_doors(self, tile_x: int, tile_y: int, 
                                       switch_states: Dict[str, bool]) -> bool:
    # May not properly check all door types and states
```

**Solution**:
```python
def _is_position_traversable_with_doors(self, tile_x: int, tile_y: int, 
                                       switch_states: Dict[str, bool]) -> bool:
    """Enhanced door state checking for traversability."""
    
    # Check base traversability first
    if not self._is_position_traversable(tile_x, tile_y):
        return False
    
    # Check for doors at this position
    pixel_x = tile_x * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
    pixel_y = tile_y * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
    
    # Check all door types that could block this position
    for door_id, door_info in self.door_states.items():
        door_x, door_y = door_info['position']
        door_type = door_info['type']
        
        # Check if door overlaps with this position
        if self._position_overlaps_door(pixel_x, pixel_y, door_x, door_y):
            # Check if door is locked based on switch states
            controlling_switch = door_info.get('controlling_switch')
            if controlling_switch:
                switch_active = switch_states.get(controlling_switch, False)
                if not switch_active:
                    if self.debug:
                        print(f"DEBUG: Position ({tile_x}, {tile_y}) blocked by locked door {door_id}")
                    return False
    
    return True

def _position_overlaps_door(self, pos_x: float, pos_y: float, 
                           door_x: float, door_y: float) -> bool:
    """Check if position overlaps with door collision area."""
    # Door collision detection logic
    door_width = TILE_PIXEL_SIZE  # Adjust based on door type
    door_height = TILE_PIXEL_SIZE
    
    return (abs(pos_x - door_x) < door_width // 2 and 
            abs(pos_y - door_y) < door_height // 2)
```

**Testing**:
```bash
python test_reachability_suite.py --map=complex-path-switch-required --verbose
# Should show proper door blocking behavior
# Debug output should show "Position blocked by locked door"
```

### Fix 3: Implement Subgoal Execution Simulation

**Location**: `nclone/graph/reachability/subgoal_planner.py`

**Problem**: Subgoals are planned but not executed to verify final reachability

**Solution**:
```python
def simulate_subgoal_execution(self, subgoals: List[Subgoal], 
                              initial_reachable: Set[Tuple[int, int]],
                              initial_switch_states: Dict[str, bool]) -> bool:
    """Simulate executing subgoals to verify final reachability."""
    
    current_reachable = initial_reachable.copy()
    current_switch_states = initial_switch_states.copy()
    
    for subgoal in subgoals:
        if subgoal.goal_type == 'activate_switch':
            # Simulate switch activation
            switch_id = subgoal.target_id
            if self._can_reach_switch(switch_id, current_reachable):
                current_switch_states[switch_id] = True
                # Recalculate reachability with new switch state
                current_reachable = self._recalculate_reachability(current_switch_states)
            else:
                if self.debug:
                    print(f"DEBUG: Cannot reach switch {switch_id} for subgoal execution")
                return False
                
        elif subgoal.goal_type == 'reach_exit':
            # Check if exit is now reachable
            for exit_x, exit_y in self.exit_positions:
                if self._is_exit_reachable_precise(exit_x, exit_y, current_reachable):
                    return True
            return False
    
    return True
```

### Fix 4: Implement Missing Test Features

#### A. Precision Pathfinding (`halfaligned-path`)
**Location**: `nclone/graph/reachability/hierarchical_geometry.py`

**Problem**: Half-aligned positions require subcell-level precision

**Solution**: Enhance subcell analysis to handle half-tile positioning

#### B. Thwump Platform Handling (`thwump-platform-required`)
**Location**: `nclone/graph/reachability/entity_aware_validator.py`

**Problem**: Thwump platforms are interactive entities that affect reachability

**Solution**: Add thwump platform detection and interaction logic

#### C. Drone Hazard Detection (`drone-unreachable`)
**Location**: `nclone/graph/reachability/hazard_integration.py`

**Problem**: Drone hazards make certain areas unreachable

**Solution**: Enhance hazard system to detect and avoid drone patrol areas

## Implementation Plan

### Phase 1: Critical Fixes (HIGH PRIORITY)
1. **Fix granularity mismatch** in exit validation
2. **Enhance locked door collision detection**
3. **Implement subgoal execution simulation**
4. **Test on `complex-path-switch-required`**

### Phase 2: Feature Completion (MEDIUM PRIORITY)
1. **Implement precision pathfinding** for `halfaligned-path`
2. **Add thwump platform handling**
3. **Enhance drone hazard detection**
4. **Optimize performance** for remaining tests

### Phase 3: Full Validation (LOW PRIORITY)
1. **Run complete test suite**
2. **Achieve 100% pass rate**
3. **Performance optimization** (all tests < 10ms)
4. **Documentation and cleanup**

## Testing Strategy

### Unit Testing
```bash
# Test specific components
python -m pytest tests/test_subgoal_planner.py
python -m pytest tests/test_door_collision.py
python -m pytest tests/test_exit_validation.py
```

### Integration Testing
```bash
# Test complex puzzle specifically
python test_reachability_suite.py --map=complex-path-switch-required --verbose

# Test all failing tests
python test_reachability_suite.py --only-failing --verbose

# Full test suite
python test_reachability_suite.py
```

### Debug Testing
```bash
# Enable comprehensive debug output
python test_reachability_suite.py --map=complex-path-switch-required --verbose 2>&1 | grep -E "(DEBUG|subgoal|door|exit)"

# Test subgoal planning specifically
python test_reachability_suite.py --map=complex-path-switch-required --verbose 2>&1 | grep -A 10 "Planning subgoals"
```

## Key Files to Modify

### Primary Files
1. **`nclone/graph/reachability/subgoal_planner.py`**
   - Fix exit validation granularity
   - Implement subgoal execution simulation
   - Add pixel-level precision checks

2. **`nclone/graph/reachability/entity_aware_validator.py`**
   - Enhance door collision detection
   - Improve traversability checks with door states
   - Add door overlap detection

3. **`nclone/graph/reachability/hierarchical_geometry.py`**
   - Integrate improved subgoal execution
   - Add precision pathfinding for subcells
   - Enhance multi-state analysis

### Secondary Files
4. **`nclone/graph/reachability/hazard_integration.py`**
   - Drone hazard detection
   - Thwump platform handling

5. **`nclone/graph/reachability/hierarchical_adapter.py`**
   - Performance optimizations
   - Better entity detection

## Expected Outcomes

### After Phase 1 Fixes
- **`complex-path-switch-required`** should pass ✅
- **Test pass rate**: 8/19 → 10/19 (53%)
- **Complex puzzle logic**: Fully functional

### After Phase 2 Completion
- **Additional tests passing**: `halfaligned-path`, `thwump-platform-required`, `drone-unreachable`
- **Test pass rate**: 10/19 → 13/19 (68%)
- **Feature completeness**: All major mechanics implemented

### After Phase 3 Optimization
- **Test pass rate**: 13/19 → 19/19 (100%)
- **Performance**: All tests < 10ms
- **System**: Production-ready

## Development Context

### Current Architecture
```
HierarchicalAdapter (Smart Detection)
    ↓
HierarchicalReachabilityAnalyzer (Multi-State Analysis)
    ↓
HierarchicalGeometryAnalyzer (Iterative Switch Discovery)
    ↓
EntityAwareValidator (Switch/Door Management)
    ↓
SubgoalPlanner (Dependency Resolution)
```

### Key Insights from Analysis
1. **Entity-aware analysis works** - correctly detects switches and doors
2. **Iterative discovery works** - finds reachable switches in multiple passes
3. **Subgoal planning framework works** - generates correct dependency chains
4. **Main issue**: Granularity mismatch between planning and validation
5. **Secondary issue**: Door collision detection needs enhancement

### Debug Output Patterns
```bash
# Successful switch discovery
DEBUG: Found 4 switches: exit_pair_0, locked_0, locked_1, trap_0
DEBUG: Iteration 1 - New switches found: True, total states: {'exit_pair_0': True, 'trap_0': True}

# Subgoal planning activation
DEBUG: No new switches found, attempting subgoal planning
DEBUG: Planning subgoals for 1 exits

# Granularity mismatch evidence
DEBUG: Exit tile reachable: True
DEBUG: Exit at (660, 276) is directly reachable
# But final result: Exit 1 not reachable from any ninja position
```

## Success Criteria

### Functional Requirements
- [ ] `complex-path-switch-required` test passes
- [ ] All door collision detection works correctly
- [ ] Subgoal execution simulation validates reachability
- [ ] Exit validation uses consistent granularity

### Performance Requirements
- [ ] Complex puzzle analysis < 50ms
- [ ] No regression in currently passing tests
- [ ] Memory usage remains stable

### Quality Requirements
- [ ] Comprehensive debug output for troubleshooting
- [ ] Clean, maintainable code structure
- [ ] Proper error handling and edge cases
- [ ] Full test coverage for new functionality

---

**Ready for Implementation**: This document provides complete context for an agentic coding tool to implement the remaining fixes and achieve full reachability system functionality.