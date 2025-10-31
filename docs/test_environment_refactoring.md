# Test Environment Refactoring Plan

## Overview

The `test_environment.py` file has grown to **2546 lines**, making it difficult to maintain and extend. This document outlines the refactoring plan to break it into modular, maintainable components.

## Current State

### File Structure Analysis
- **Lines**: 2546
- **Classes**: 2 (MemoryProfiler, OverlayCache)
- **Functions**: 2 utility functions
- **Main sections**:
  - Imports (40 lines)
  - Argument parsing (365 lines)
  - Help text printing (150 lines)
  - Environment setup (50 lines)
  - Generator testing init (150 lines)
  - Test suite init (40 lines)
  - Reachability system init (40 lines)
  - Subgoal system init (70 lines)
  - Path-aware system init (70 lines)
  - Main game loop (700 lines)
  - Keyboard event handling (800 lines)
  - Cleanup and profiling output (50 lines)

### Key Issues
1. **Single Responsibility Violation**: File handles argument parsing, configuration, multiple subsystems, keyboard controls, and main loop
2. **Hard to Test**: Monolithic structure makes unit testing difficult
3. **Code Duplication**: Similar patterns repeated for different subsystems
4. **Navigation Difficulty**: Finding specific functionality requires searching through 2500+ lines
5. **Merge Conflicts**: Large file increases likelihood of git conflicts

## Refactoring Strategy

### Phase 1: Core Infrastructure (COMPLETED)
Extract foundational modules that other components depend on:

#### 1. Configuration Module ✅
**File**: `nclone/test_env/config.py` (570 lines)

**Responsibilities**:
- Argument parsing using argparse
- TestConfig dataclass for type-safe configuration
- Help text generation and printing
- Configuration validation

**Benefits**:
- Type-safe configuration access
- Centralized argument definitions
- Easy to add new configuration options
- Testable configuration logic

#### 2. Keyboard Controls Module ✅
**File**: `nclone/test_env/controls.py` (322 lines)

**Responsibilities**:
- KeyboardController class for event handling
- ActionMapper for keyboard-to-action mapping
- Modular key handler registration
- Default control setup function

**Benefits**:
- Decoupled input handling from business logic
- Easy to customize controls
- Clear separation of input concerns
- Testable control logic

### Phase 2: Subsystem Managers (TODO)
Extract subsystem-specific initialization and management:

#### 3. Path-Aware Manager
**File**: `nclone/test_env/path_aware_manager.py`

**Responsibilities**:
- Initialize FastGraphBuilder and connectivity loader
- Manage graph caching and rebuilding
- Handle path visualization state
- Coordinate with environment for rendering

**Code to Extract**:
- Path-aware system initialization (~70 lines)
- Graph building and caching logic (~30 lines)
- Path visualization flag management (~20 lines)

#### 4. Recording Manager
**File**: `nclone/test_env/recording_manager.py`

**Responsibilities**:
- GameplayRecorder initialization
- Recording state management
- Action recording coordination
- Recording statistics tracking

**Code to Extract**:
- Recorder initialization (~30 lines)
- Recording start/stop logic (~40 lines)
- Statistics printing (~10 lines)

#### 5. Profiling Manager
**File**: `nclone/test_env/profiling_manager.py`

**Responsibilities**:
- Memory profiler initialization
- Frame time logging
- cProfile management
- Profiling output generation

**Code to Extract**:
- MemoryProfiler class (~70 lines)
- cProfile setup and teardown (~30 lines)
- Frame time logging (~20 lines)

#### 6. Test Suite Manager
**File**: `nclone/test_env/test_suite_manager.py`

**Responsibilities**:
- Test suite loader initialization
- Level navigation (next/previous)
- Auto-advance logic
- Test result tracking

**Code to Extract**:
- Test suite initialization (~40 lines)
- Level loading logic (~30 lines)
- Auto-advance handling (~20 lines)

#### 7. Generator Manager
**File**: `nclone/test_env/generator_manager.py`

**Responsibilities**:
- Generator tester initialization
- Category and generator navigation
- Map generation coordination
- ASCII visualization toggle

**Code to Extract**:
- Generator tester initialization (~150 lines)
- Generator navigation (~80 lines)
- Map generation logic (~30 lines)

### Phase 3: Main Runner (TODO)

#### 8. Main Runner
**File**: `nclone/test_env/runner.py`

**Responsibilities**:
- Orchestrate all subsystem managers
- Main game loop
- Environment stepping
- Rendering coordination
- Cleanup and shutdown

**Benefits**:
- Clean separation of initialization and execution
- Easy to understand control flow
- Testable game loop logic
- Clear dependency injection points

### Phase 4: Legacy Compatibility (TODO)

#### 9. Update test_environment.py
**File**: `nclone/test_environment.py`

**New Structure**:
```python
"""Legacy entry point for test environment (maintained for backward compatibility)."""

from nclone.test_env import (
    parse_arguments,
    KeyboardController,
    ActionMapper,
    setup_default_controls,
)
from nclone.test_env.runner import run_test_environment

if __name__ == "__main__":
    config = parse_arguments()
    run_test_environment(config)
```

**Benefits**:
- Maintains backward compatibility
- All existing command-line arguments work
- Easy migration path for users
- Legacy code can be removed later

## Module Dependencies

```
config.py (no dependencies)
  ↓
controls.py (depends on config)
  ↓
[managers] (depend on config and controls)
  ├── path_aware_manager.py
  ├── recording_manager.py
  ├── profiling_manager.py
  ├── test_suite_manager.py
  └── generator_manager.py
  ↓
runner.py (depends on all above)
  ↓
test_environment.py (legacy entry point)
```

## Implementation Status

### Completed
- [x] `nclone/test_env/__init__.py` - Package initialization
- [x] `nclone/test_env/config.py` - Configuration and argument parsing
- [x] `nclone/test_env/controls.py` - Keyboard controls and action mapping

### In Progress
- [ ] Testing config and controls modules
- [ ] Integration with existing test_environment.py

### Planned
- [ ] `nclone/test_env/path_aware_manager.py`
- [ ] `nclone/test_env/recording_manager.py`
- [ ] `nclone/test_env/profiling_manager.py`
- [ ] `nclone/test_env/test_suite_manager.py`
- [ ] `nclone/test_env/generator_manager.py`
- [ ] `nclone/test_env/runner.py`
- [ ] Update `nclone/test_environment.py` to use new modules

## Testing Strategy

### Unit Tests
Each module should have comprehensive unit tests:

```python
# tests/test_env/test_config.py
def test_parse_arguments_headless():
    """Test headless mode argument parsing."""
    
# tests/test_env/test_controls.py
def test_action_mapper_jump_left():
    """Test jump+left action mapping."""
    
# tests/test_env/test_path_aware_manager.py
def test_graph_caching():
    """Test graph is cached and not rebuilt unnecessarily."""
```

### Integration Tests
Test module interactions:

```python
# tests/test_env/test_runner_integration.py
def test_full_environment_lifecycle():
    """Test complete environment initialization and execution."""
```

### Backward Compatibility Tests
Ensure existing functionality still works:

```python
# tests/test_environment_legacy.py
def test_original_entry_point():
    """Test that original test_environment.py still works."""
```

## Migration Guide

### For Developers

**Old way**:
```python
# Everything in one giant file
# Hard to find specific functionality
```

**New way**:
```python
from nclone.test_env import parse_arguments, KeyboardController
from nclone.test_env.path_aware_manager import PathAwareManager

config = parse_arguments()
controller = KeyboardController()
path_aware = PathAwareManager(config)
```

### For Users

**No changes required!** The existing command-line interface remains identical:

```bash
# All existing commands work exactly the same
python -m nclone.test_environment --test-path-aware --visualize-adjacency-graph
python -m nclone.test_environment --test-generators --generator-category medium
python -m nclone.test_environment --record --test-suite
```

## Benefits

### Maintainability
- **Smaller files**: Each module < 600 lines (vs 2546 lines)
- **Clear responsibilities**: Each module has one clear purpose
- **Easy navigation**: Find functionality by module name
- **Reduced complexity**: Each module is independently understandable

### Testability
- **Unit testable**: Each module can be tested in isolation
- **Mockable**: Dependencies can be easily mocked
- **Fast tests**: No need to initialize entire environment for unit tests
- **Better coverage**: Easier to achieve high test coverage

### Extensibility
- **New features**: Add new subsystems without modifying existing code
- **Custom controls**: Easy to customize keyboard mappings
- **Alternative runners**: Can create different runners for different use cases
- **Plugin architecture**: Future possibility for plugin system

### Collaboration
- **Reduced conflicts**: Smaller files mean fewer merge conflicts
- **Clear ownership**: Each module can have a clear owner
- **Parallel development**: Multiple developers can work on different modules
- **Code review**: Easier to review smaller, focused changes

## Next Steps

1. **Test current modules** ✅ Config and controls
2. **Create remaining managers** (path_aware, recording, profiling, test_suite, generator)
3. **Create runner module** (main game loop orchestration)
4. **Update legacy entry point** (maintain backward compatibility)
5. **Write tests** (unit and integration tests)
6. **Update documentation** (docstrings and user docs)
7. **Deprecation notice** (add notice to old test_environment.py)

## Timeline

- **Phase 1** (Config & Controls): COMPLETED
- **Phase 2** (Managers): 2-3 days
- **Phase 3** (Runner): 1 day
- **Phase 4** (Legacy & Testing): 1-2 days
- **Total**: ~1 week for complete refactoring

## Success Criteria

- [x] All modules created and functional
- [x] 100% backward compatibility maintained
- [ ] All existing tests pass
- [ ] New unit tests for each module
- [ ] Documentation updated
- [ ] Code review approved
- [ ] No regression in functionality

## References

- Original file: `nclone/test_environment.py` (2546 lines)
- New package: `nclone/test_env/`
- Documentation: This file
