# Graph Traversability Fix - Implementation Summary

## Problem

The graph building and reachability system had critical bugs where:
1. Nodes were rendered in areas disconnected from the player position
2. Sub-nodes within the same tile were incorrectly assumed to all be reachable from each other
3. This caused BFS flood fill to traverse through solid geometry

## Root Cause

In `fast_graph_builder.py`, line 791-793 incorrectly assumed:
```python
# If in same tile, both nodes are in dark grey areas so traversable
if tile_dx == 0 and tile_dy == 0:
    return True
```

This was wrong for tiles with internal solid geometry (slopes, circles, etc.). For example, in a diagonal slope tile (type 6), sub-nodes on opposite sides of the slope cannot reach each other without crossing solid geometry.

## Solution

### 1. Within-Tile Connectivity Precomputation (O(1))

Created a precomputed lookup table `WITHIN_TILE_CONNECTIVITY` that determines which sub-nodes within a tile can reach each other:

**New Functions:**
- `_line_crosses_diagonal()`: Checks if line segment crosses diagonal slope (O(1))
- `_line_crosses_circle()`: Checks if line segment crosses circle boundary (O(1))  
- `_can_traverse_within_tile()`: Determines if two sub-nodes can reach each other (O(1))
- `_precompute_within_tile_connectivity()`: Precomputes connectivity for all 38 tile types

**Example for Tile 6 (Slope \):**
```
Valid sub-nodes: (1,0), (0,1), (1,1)
Connectivity:
      (1,0)  (0,1)  (1,1)
(1,0)   -      ✓      ✓
(0,1)   ✓      -      ✓
(1,1)   ✓      ✓      -
```

All three valid sub-nodes can reach each other because the line between them is parallel to the slope and stays in non-solid area.

### 2. Updated Traversability Check

Modified `_is_sub_node_traversable()` to use the precomputed table:

```python
# If in same tile, check within-tile connectivity (O(1) lookup)
if tile_dx == 0 and tile_dy == 0:
    # Get sub-node indices
    src_sub_x = 0 if src_rel_x < 12 else 1
    src_sub_y = 0 if src_rel_y < 12 else 1
    dst_sub_x = 0 if dst_rel_x < 12 else 1
    dst_sub_y = 0 if dst_rel_y < 12 else 1
    
    # O(1) lookup in precomputed within-tile connectivity table
    reachable = WITHIN_TILE_CONNECTIVITY.get(src_tile_type, {}).get(
        (src_sub_x, src_sub_y), set()
    )
    return (dst_sub_x, dst_sub_y) in reachable
```

### 3. Debug Output System

Added comprehensive debug tracking to `FastGraphBuilder`:

**Debug Statistics:**
- Total tiles and sub-nodes generated
- Sub-nodes per tile type
- Traversability check counts
- Blocked reasons (geometry, connectivity, diagonal)

**Usage:**
```python
builder = FastGraphBuilder(debug=True)
result = builder.build_graph(level_data, ninja_pos)
# Automatically prints detailed statistics
```

### 4. Visualization Tools

#### ASCII Tile Renderer (`nclone/tools/ascii_tile_renderer.py`)

Text-based rendering for immediate debugging:

```bash
# Render specific tile
python -m nclone.tools.ascii_tile_renderer --tile 6

# Show connectivity matrix
python -c "from nclone.tools.ascii_tile_renderer import print_tile_connectivity_matrix; print_tile_connectivity_matrix(6)"

# Render all tiles
python -m nclone.tools.ascii_tile_renderer --all
```

**Features:**
- Shows solid areas (█) vs non-solid areas (·)
- Marks valid sub-nodes (1-4) and invalid ones (X)
- Displays within-tile connectivity matrix
- Immediate feedback without image generation

#### Image Visualizer (`nclone/tools/visualize_tile_types.py`)

Creates PNG images for detailed analysis:

```bash
# Generate all tile images
python -m nclone.tools.visualize_tile_types --all

# Generate specific tile
python -m nclone.tools.visualize_tile_types --tile 6 --scale 20
```

**Output:**
- Individual PNG for each tile type (0-33)
- Grid visualization showing all tiles
- Green circles for valid sub-nodes
- Red X for invalid sub-nodes
- Red lines showing diagonal/circular geometry
- Saved to `debug_output/` directory

## Implementation Details

### Geometric Checks

For each tile type, the system checks sub-node connectivity by testing if the line between them crosses solid geometry:

**Half Tiles (2-5):** Check if both sub-nodes are on the same (non-solid) side
```python
if tile_type == 2:  # Top half solid
    return src_offset[1] >= 12 and dst_offset[1] >= 12
```

**Diagonal Slopes (6-9):** Use line intersection test
```python
if tile_type == 6:  # Slope from (0,24) to (24,0)
    return not _line_crosses_diagonal(src_offset, dst_offset, (0, 24), (24, 0))
```

**Quarter Circles (10-13):** Check if line crosses circle boundary
```python
if tile_type == 10:  # Bottom-right quarter circle
    return not _line_crosses_circle(src_offset, dst_offset, (24, 24), 24)
```

### Performance

All checks are **O(1)** using precomputed lookup tables:
- `SUBNODE_VALIDITY_TABLE`: Which sub-nodes exist in each tile
- `WITHIN_TILE_CONNECTIVITY`: Which sub-nodes can reach each other
- No expensive geometric calculations at runtime

## Files Modified

1. **`nclone/graph/reachability/fast_graph_builder.py`**
   - Added within-tile connectivity precomputation
   - Updated `_is_sub_node_traversable()` to use lookup table
   - Added debug output system
   - Added helper functions for geometric checks

2. **`nclone/debug_overlay_renderer.py`**
   - Modified to filter nodes/edges by reachable set
   - Previously completed in earlier fixes

## Files Created

1. **`nclone/tools/__init__.py`**
   - Tools package initialization

2. **`nclone/tools/ascii_tile_renderer.py`**
   - Text-based tile visualization
   - Connectivity matrix display
   - Command-line interface

3. **`nclone/tools/visualize_tile_types.py`**
   - Image-based tile visualization
   - PNG generation for all tile types
   - Grid view of all tiles

## Testing

### Manual Verification

```python
from nclone.tools.ascii_tile_renderer import print_tile_connectivity_matrix

# Test slope tile
print_tile_connectivity_matrix(6)

# Test half tile
print_tile_connectivity_matrix(2)

# Test circular tile
print_tile_connectivity_matrix(10)
```

### Integration Test

```python
from nclone.graph.reachability.fast_graph_builder import FastGraphBuilder
import numpy as np

tiles = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)
level_data = {"width": 3, "height": 3, "tiles": tiles}

builder = FastGraphBuilder(debug=True)
result = builder.build_graph(level_data, ninja_pos=(36, 36), level_id="test")
```

### Visual Testing

Run test environment with graph visualization:
```bash
python nclone/test_environment.py --visualize-adjacency-graph
```

Nodes should now only appear in areas reachable from the player position.

## Next Steps (If Issues Persist)

1. **Verify precomputed connectivity table**: Check `tile_connectivity_precomputer.py` for errors
2. **Test specific tile combinations**: Use `ascii_tile_renderer.py --pair` to test problematic pairs
3. **Enable debug mode**: Use `FastGraphBuilder(debug=True)` to see detailed traversability decisions
4. **Check entity masking**: Verify doors/mines are properly updating the graph

## Summary

The fix addresses the root cause by properly handling within-tile connectivity for tiles with internal solid geometry. All geometric checks are O(1) using precomputed lookup tables, maintaining the performance requirements. Comprehensive visualization and debug tools are now available for ongoing verification and troubleshooting.

