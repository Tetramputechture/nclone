# Graph Traversability Improvements

This document summarizes the major improvements made to the graph builder to address traversability issues with half-width tiles, one-way platforms, and diagonal slopes.

## Issues Addressed

### 1. Half-Width Tile Traversability
**Problem**: The original graph builder used coarse grid resolution (24x24 pixels) and simple edge-blocking logic that failed to detect traversable passages formed by combining half-tiles at offset positions.

**Solution**: 
- Implemented **sub-grid resolution** (2x2 subdivision, 12x12 pixel sub-cells)
- Added accurate **ninja collision detection** that checks if a ninja (radius=10px) can occupy specific positions
- Improved **half-tile geometry detection** that correctly identifies solid vs. traversable halves of tiles

### 2. One-Way Platform Detection
**Problem**: The original one-way platform blocking logic used incorrect constants and overly restrictive proximity detection.

**Solution**:
- Fixed the **platform radius** to use the actual `EntityOneWayPlatform.SEMI_SIDE = 12` constant
- Improved **path-based blocking detection** that checks if platforms intersect the movement path
- Enhanced **directional blocking** that correctly applies platform normals

### 3. Diagonal Traversability
**Problem**: The original graph only supported 4-connected movement, missing valid diagonal paths through slope combinations and corner navigation.

**Solution**:
- Added **8-connected graph topology** (4 cardinal + 4 diagonal directions)
- Implemented **diagonal movement validation** that prevents corner-cutting through solid geometry
- Added **slope-aware traversability** for complex tile combinations

## Technical Implementation

### Sub-Grid Architecture
```python
# New constants for improved spatial resolution
SUB_GRID_RESOLUTION = 2  # 2x2 subdivision per tile
SUB_CELL_SIZE = TILE_PIXEL_SIZE // SUB_GRID_RESOLUTION  # 12 pixels per sub-cell
SUB_GRID_WIDTH = MAP_TILE_WIDTH * SUB_GRID_RESOLUTION   # 84 sub-cells wide  
SUB_GRID_HEIGHT = MAP_TILE_HEIGHT * SUB_GRID_RESOLUTION # 46 sub-cells tall
```

### Key Methods Added

#### `_is_sub_cell_traversable(level_data, sub_row, sub_col)`
Determines if a 12x12 pixel sub-cell is traversable by checking:
1. Tile type at the sub-cell location
2. Ninja collision detection with accurate geometry
3. Half-tile and slope-specific collision rules

#### `_determine_sub_cell_traversability(src, tgt, ...)`
Enhanced traversability logic that:
1. Validates both source and target sub-cells
2. Performs swept collision detection along movement paths
3. Checks one-way platform and door blocking
4. Supports both cardinal and diagonal movement

#### `_ninja_intersects_tile_geometry(ninja_pos, tile_type)`
Accurate collision detection for half-tiles and slopes:
- **Half-tiles**: Checks which half of the tile contains the ninja
- **Slopes**: Uses edge-based collision detection with tile geometry definitions

### Graph Structure Changes

**Before**: 
- ~966 nodes (42×23 grid)
- ~3,864 edges (4-connected)

**After**:
- ~3,864 nodes (84×46 sub-grid) 
- ~30,912 edges (8-connected with validation)

## Performance Characteristics

### Memory Usage
- **Nodes**: ~4x increase (sub-grid resolution)
- **Edges**: ~8x increase (diagonal connectivity + higher resolution)
- **Total**: ~32x increase in graph complexity

### Accuracy Improvements
- **Spatial Resolution**: 4x improvement (24px → 12px cells)
- **Movement Types**: 2x improvement (4-connected → 8-connected)
- **Collision Accuracy**: Ninja-radius aware (10px precision)

## Compatibility

### Backward Compatibility
- Maintains the same `GraphBuilder.build_graph()` interface
- Preserves existing observation space dimensions via padding
- Legacy `_determine_traversability()` method kept for compatibility

### Integration
- Works with existing `GraphObservationMixin`
- Compatible with current entity extraction logic
- No changes required to environment classes

## Validation Results

All improvements were validated with comprehensive tests:

✅ **Sub-grid Resolution**: Verified 4x node increase with proper spatial mapping  
✅ **Half-tile Traversability**: Confirmed accurate detection of traversable/blocked sub-cells  
✅ **One-way Platform Blocking**: Validated directional blocking with correct platform geometry  
✅ **Diagonal Traversability**: Verified diagonal edges with corner-cutting prevention  
✅ **Ninja Collision Detection**: Confirmed accurate collision detection for various tile types  

## Usage

The improved graph builder is a drop-in replacement:

```python
from nclone.graph.graph_builder import GraphBuilder

builder = GraphBuilder()
graph = builder.build_graph(level_data, ninja_position, entities)

# Graph now has:
# - Higher spatial resolution (12px sub-cells)  
# - Accurate half-tile traversability
# - Proper one-way platform blocking
# - Diagonal movement support
# - Ninja-radius aware collision detection
```

The improvements provide significantly better accuracy for pathfinding, navigation, and spatial reasoning tasks while maintaining full backward compatibility.
