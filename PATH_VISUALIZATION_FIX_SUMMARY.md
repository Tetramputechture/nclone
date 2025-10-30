# Path Visualization Debug Fix Summary

## Problem Description
Path visualization (debug mode) was not rendering when toggled with 'D' (path distances) or 'A' (adjacency graph) keys in test_environment.py.

## Root Causes Identified

### Issue 0: Visualization Flickering (NEW)
**Symptom**: Path visualization and adjacency graph flickering on/off every frame

**Root Cause**: The `_get_level_hash()` function was hashing the entire LevelData object (including dynamic state like player position, entity positions, and switch states). This caused the hash to change every single frame, triggering a graph rebuild every frame, which caused the visualization to flicker.

**Location**: `test_environment.py` line ~69 (original)
```python
level_str = str(env.level_data)  # Wrong! Includes dynamic state
return hashlib.md5(level_str.encode()).hexdigest()
```

**Fix**: Only hash the static level structure (tiles, entity types/IDs) and exclude dynamic state:
```python
# Hash only static structure
hash_components = []

# Add tiles hash (static geometry)
tiles_bytes = level_data.tiles.tobytes()
hash_components.append(hashlib.md5(tiles_bytes).hexdigest())

# Add entity types and IDs (NOT positions)
entity_static = []
for entity in level_data.entities:
    entity_type = entity.get('type', '')
    entity_id = entity.get('id', '')
    entity_static.append(f"{entity_type}:{entity_id}")
entity_str = ','.join(sorted(entity_static))
hash_components.append(hashlib.md5(entity_str.encode()).hexdigest())

# Add level_id
if level_data.level_id:
    hash_components.append(level_data.level_id)

combined = '|'.join(hash_components)
return hashlib.md5(combined.encode()).hexdigest()
```

**Result**: Graph is now only rebuilt when the level structure actually changes (e.g., when transitioning to a new level), not on every frame.

### Issue 1: LevelData Type Mismatch
**Error**: `'LevelData' object is not subscriptable`

**Cause**: The `graph_builder.build_graph()` method expected a dictionary with keys like `'tiles'`, `'entities'`, `'switch_states'`, but was receiving a `LevelData` dataclass object instead.

**Locations**: 
- `test_environment.py` line ~2160 (visualization section)
- `test_environment.py` line ~1990 (pathfinding benchmark section)
```python
cached_graph_data = graph_builder.build_graph(env.level_data)  # Wrong!
```

**Fix**: Convert LevelData object to dictionary format:
```python
level_data_dict = env.level_data.to_dict()
level_data_dict['switch_states'] = env.level_data.switch_states  # Not in to_dict()
cached_graph_data = graph_builder.build_graph(level_data_dict)
```

### Issue 2: Missing Distance Calculation Implementation
**Error**: `AttributeError: 'CachedPathDistanceCalculator' object has no attribute 'get_distances_to_objectives'`

**Cause**: The code was calling a non-existent method `path_calculator.get_distances_to_objectives()`.

**Location**: `test_environment.py` line ~2188 (original)

**Fix**: Implemented proper distance calculation using the existing `get_distance()` method:
```python
# Calculate distances to objectives
distances = {
    'switch_distance': float('inf'),
    'exit_distance': float('inf')
}

# Find nearest switch
for entity in env.level_data.entities:
    entity_type = entity.get('type', '')
    if entity_type in ['switch', 'exit_switch']:
        entity_pos = (entity.get('x', 0), entity.get('y', 0))
        dist = path_calculator.get_distance(
            ninja_pos, entity_pos, adjacency, cache_key='switch'
        )
        distances['switch_distance'] = min(distances['switch_distance'], dist)

# Similar for exit entities...
```

### Issue 3: Silent Error Handling
**Problem**: Original code had `except: pass` blocks that hid all errors, making debugging impossible.

**Fix**: Added comprehensive debug logging:
```python
except Exception as e:
    print(f"Debug: Graph building failed: {e}")
    import traceback
    traceback.print_exc()
```

## Testing the Fix

To test if path visualization is working:

1. Run test_environment.py: `python -m nclone.test_environment --level SI-A-01-00`
2. Press 'D' to toggle path distance display
3. Press 'A' to toggle adjacency graph visualization
4. You should see:
   - Distance to switch and exit displayed near the ninja
   - Graph nodes and edges showing navigable positions

## Debug Output Expected

When working correctly:
```
Debug: Rebuilding graph (hash changed from None to [hash])
Debug: Graph built successfully, cached_graph_data is not None: True
```

If issues occur, you'll now see full tracebacks instead of silent failures.

## Architecture Notes

### LevelData Structure
- `LevelData` is a dataclass defined in `nclone/graph/level_data.py`
- Has attributes: `tiles`, `entities`, `player`, `level_id`, `metadata`, `switch_states`
- Provides `to_dict()` method for backward compatibility
- **Important**: `switch_states` is NOT included in `to_dict()` and must be added manually

### Graph Builder Requirements
- `FastGraphBuilder.build_graph()` expects dictionary with:
  - `'tiles'`: 2D numpy array
  - `'entities'`: List of entity dictionaries
  - `'switch_states'`: Dictionary of switch states (optional)
- Returns dictionary with:
  - `'adjacency'`: Navigation graph
  - `'reachable'`: Set of reachable positions
  - `'blocked_positions'`: Set of blocked positions
  - `'blocked_edges'`: Set of blocked edges

### Path Calculator Usage
- `CachedPathDistanceCalculator.get_distance(start, goal, adjacency, cache_key)`
- Returns float distance or raises exception if no path exists
- Use try-except around individual calls, not the entire block

## Future Debugging Best Practices

1. **Never use silent `except: pass`** - Always log the exception
2. **Use `traceback.print_exc()`** for full error context
3. **Add descriptive debug prints** at key decision points
4. **Check type compatibility** between dataclasses and dictionaries
5. **Verify method existence** before calling (use `hasattr()` or check documentation)

## Related Files
- `nclone/test_environment.py` - Main visualization code
- `nclone/graph/level_data.py` - LevelData dataclass definition
- `nclone/graph/reachability/fast_graph_builder.py` - Graph construction
- `nclone/graph/reachability/path_distance_calculator.py` - Distance calculations
- `nclone/debug_overlay_renderer.py` - Pygame rendering

## Commits
- c4d2749: "Fix: Pathfinding benchmark error and visualization flickering"
- efd34e4: "Fix: LevelData subscriptable error and implement distance calculations"
- e9b2e9b: "Fix: Remove incorrect global declaration (syntax error)"
- 6c56d05: "Fix: Path visualization not rendering due to missing global declarations"

Branch: `fix/path-visualization-controls`
PR: #51
