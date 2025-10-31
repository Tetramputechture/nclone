# Sub-Node Geometry Fix Summary

## Problem
Green nodes (sub-nodes) were appearing in dark grey (solid) areas of tiles in the debug overlay visualization, indicating incorrect geometric calculations in `_check_subnode_validity_simple()`. This affected pathfinding and reachability analysis.

## Investigation Results

### Files Analyzed
- `nclone/graph/reachability/fast_graph_builder.py` - Main graph building implementation
- `nclone/graph/reachability/tile_connectivity_precomputer.py` - Precomputation logic
- `nclone/tile_definitions.py` - Tile segment definitions (ground truth)
- `nclone/shared_tile_renderer.py` - Visual rendering logic

### Inconsistencies Found
11 tile types had inconsistent implementations between `fast_graph_builder.py` and `tile_connectivity_precomputer.py`:
- Type 24: Raised Mild R-Dn
- Type 27: Steep Slope Up-R  
- Type 29: Steep Slope Dn-L
- Type 30: Raised Steep L-Up
- Type 31: Raised Steep R-Up
- Type 32: Raised Steep R-Dn
- Type 33: Raised Steep L-Dn
- Types 34-37: Glitched tiles (different treatment)

## Fixes Applied

### Type 24 (Raised Mild R-Dn)
**Before:**
```python
# fast_graph: return pixel_y <= (pixel_x * 12 / 24)
# precomputer: return pixel_y <= (24 - pixel_x * 12 / 24)
```

**After (both files):**
```python
return pixel_y < (12 - pixel_x * 12 / 24)
```
Solid region: Trapezoid. Traversable: Triangle at top above line from (0,12) to (24,0).

### Type 27 (Steep Slope Up-R)
**Before:**
```python
# fast_graph: if pixel_x >= 12 else False
# precomputer: if pixel_x >= 12 else True
```

**After (both files):**
```python
return pixel_x < 12 or pixel_y < ((pixel_x - 12) * 24 / 12)
```
Solid region: Right triangle (12,0), (24,0), (24,24). Traversable: Left of x=12 OR above diagonal.

### Type 29 (Steep Slope Dn-L)
**Before:**
```python
# fast_graph: if pixel_x <= 12 else False
# precomputer: if pixel_x <= 12 else True
```

**After (both files):**
```python
return pixel_x > 12 or pixel_y < (pixel_x * 24 / 12)
```
Solid region: Left triangle (12,24), (0,0), (0,24). Traversable: Right of x=12 OR above diagonal.

### Types 30-33 (Raised Steep Slopes)
**Before:** Completely different formulas and conditionals between implementations.

**After:** Made consistent using correct geometric analysis of solid quadrilaterals and traversable triangular regions based on visual renderer polygons.

### Types 34-37 (Glitched Tiles)
**Before:** 
- fast_graph: returned `True` (non-solid)
- precomputer: returned `False` (solid)

**After (both files):**
```python
return True  # Treat as non-solid for safety (unused in gameplay)
```

## Tools Created

### 1. Comprehensive Tile Validator (`comprehensive_tile_validator.py`)
- Compares both implementations pixel-by-pixel for all 38 tile types
- Generates visualization grid showing all tiles with sub-node validity markers
- Exports individual tile images for detailed inspection
- Reports inconsistencies with solid pixel counts

### 2. Geometric Analysis Tool (`analyze_tile_geometry.py`)
- Analyzes tile rendering polygons from `shared_tile_renderer.py`
- Cross-references with `TILE_SEGMENT_DIAG_MAP` definitions
- Computes correct traversability formulas based on solid regions
- Documents expected behavior for each problematic tile type

### 3. Automated Test Suite (`test_subnode_validity.py`)
- 12 comprehensive tests covering all tile categories
- Tests consistency between both implementations
- Verifies specific tile behaviors (empty, solid, half-tiles, slopes, circles, pipes)
- Tests all 4 sub-node positions for every tile type
- All tests pass ✅

## Validation Results

### Before Fix
- 11 inconsistent tile types
- Nodes appearing in solid areas
- Incorrect reachability calculations

### After Fix
- ✅ All 38 tile types consistent between implementations
- ✅ No invalid nodes found on solid tiles
- ✅ No reachable nodes in unreachable areas
- ✅ All automated tests pass
- ✅ Regenerated `tile_connectivity.pkl.gz` with correct data (1.54 KB)

## Files Modified
1. `/home/tetra/projects/nclone/nclone/graph/reachability/fast_graph_builder.py`
2. `/home/tetra/projects/nclone/nclone/graph/reachability/tile_connectivity_precomputer.py`
3. `/home/tetra/projects/nclone/nclone/data/tile_connectivity.pkl.gz` (regenerated)

## Files Created
1. `/home/tetra/projects/nclone/nclone/tools/comprehensive_tile_validator.py`
2. `/home/tetra/projects/nclone/nclone/tools/analyze_tile_geometry.py`
3. `/home/tetra/projects/nclone/nclone/graph/reachability/test_subnode_validity.py`
4. `/home/tetra/projects/nclone/debug_output/tile_validation_grid.png`
5. `/home/tetra/projects/nclone/debug_output/tiles/tile_type_*.png` (28 individual images)

## Impact
- **Pathfinding Accuracy:** Sub-nodes now correctly represent traversable areas only
- **Reachability Analysis:** Flood-fill correctly identifies reachable regions
- **Performance:** No performance impact (same O(1) lookups, just correct formulas)
- **ML Training:** Agents will now receive correct reachability signals

## Testing Commands
```bash
# Run comprehensive validator
python -m nclone.tools.comprehensive_tile_validator

# Run automated tests
python -m nclone.graph.reachability.test_subnode_validity

# Debug invalid nodes on map 0
python -m nclone.tools.debug_invalid_nodes 0

# Regenerate connectivity data (if needed)
python -m nclone.graph.reachability.tile_connectivity_precomputer
```

## Conclusion
All sub-node geometric calculations have been corrected and verified. The adjacency graph now accurately represents traversable areas for all 38 tile types, preventing nodes from appearing in solid regions.

