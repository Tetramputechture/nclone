# Diagonal and Raised Steep Slope Geometry Fixes

## Summary
Fixed critical geometry bugs in diagonal slopes (types 6-9) and raised steep slopes (types 32-33) that were causing green nodes to appear in solid areas. The root cause was incorrect line equations and Y-axis coordinate system confusion.

## Root Cause: Y-Axis Coordinate System
The N++ rendering system uses **Y-down coordinates** (0 at top, 24 at bottom), which is opposite to mathematical conventions. The original implementations had:
1. Incorrect line equations
2. Wrong solid/traversable region determinations
3. Inverted comparisons

## Fixed Tile Types

### Diagonal Slopes (Types 6-9)
All 4 diagonal tile types had completely wrong geometry:

**Type 6 (Slope `\`):**
- Triangle vertices: (0,24), (24,0), (0,0) - fills top-right
- Old: `pixel_y >= (24 - pixel_x)` ❌
- New: `pixel_y > (24 - pixel_x)` ✓
- Fix: Changed `>=` to `>` for correct boundary handling

**Type 7 (Slope `/`):**
- Triangle vertices: (0,0), (24,24), (0,24) - fills left side  
- Old: `pixel_y >= pixel_x` ❌
- New: `pixel_x > pixel_y` ✓
- Fix: Inverted comparison - solid is LEFT (x <= y), traversable is RIGHT (x > y)

**Type 8 (Slope `/` inverted):**
- Triangle vertices: (24,0), (0,24), (24,24) - fills bottom-right
- Old: `pixel_y <= pixel_x` ❌
- New: `pixel_y < (24 - pixel_x)` ✓
- Fix: Used correct line equation y = 24 - x instead of y = x

**Type 9 (Slope `\` inverted):**
- Triangle vertices: (24,24), (0,0), (0,24) - fills left side
- Old: `pixel_y <= (24 - pixel_x)` ❌
- New: `pixel_x > pixel_y` ✓
- Fix: Used correct comparison for line y = x

### Raised Steep Slopes (Types 32-33)

**Type 32 (Platform on right with sharp drop):**
- Quad vertices: (12,0), (0,24), (24,24), (24,0)
- Diagonal line from (12,0) to (0,24): `y = 24 - 2x`
- Old: `pixel_y <= ((pixel_x - 12) * 24 / 12) if pixel_x >= 12 else pixel_y <= 0` ❌
- New: `pixel_x < 12 and pixel_y < (24 - 2 * pixel_x)` ✓
- Fix: Correct line equation and region logic

**Type 33 (Platform on left with sharp drop):**
- Quad vertices: (12,0), (24,24), (0,24), (0,0)
- Diagonal line from (12,0) to (24,24): `y = 2x - 24`
- Old: `pixel_y <= (24 - pixel_x * 24 / 12) if pixel_x <= 12 else pixel_y <= 0` ❌
- New: `pixel_x > 12 and pixel_y < (2 * pixel_x - 24)` ✓
- Fix: Correct line equation and region logic

## Verification Method

### PIL Ground Truth
For each tile type, we rendered the actual solid region using PIL's polygon drawing and checked pixel colors:
- White (255,255,255) = Traversable
- Grey (128,128,128) = Solid

### Comparison
Compared our `_check_subnode_validity_simple()` function output against PIL rendering for all 4 sub-node positions: (6,6), (18,6), (6,18), (18,18)

## Code Centralization

As requested by the user, we **centralized the geometry logic**:

1. **Single Source of Truth**: `_check_subnode_validity_simple()` now exists ONLY in `/home/tetra/projects/nclone/nclone/graph/reachability/fast_graph_builder.py`

2. **Tile Connectivity Precomputer**: Uses the original simpler tile-level approach (3D array: [tile_a, tile_b, direction]). Sub-node validity is handled by the fast_graph_builder importing the centralized function.

3. **No Duplication**: Removed the duplicate 180-line `_check_subnode_validity_simple()` function that was in `tile_connectivity_precomputer.py`

## Files Modified

1. `nclone/graph/reachability/fast_graph_builder.py`
   - Fixed diagonal slopes (6-9) and raised steep slopes (32-33)
   - Centralized `_check_subnode_validity_simple()` function

2. `nclone/graph/reachability/tile_connectivity_precomputer.py`
   - Restored to original tile-level approach
   - Removed duplicate geometry function

3. `nclone/graph/reachability/tile_connectivity_loader.py`
   - Updated to work with 3D precomputed array
   - Added note about sub-node checks being in fast_graph_builder

4. `nclone/graph/reachability/test_subnode_validity.py`
   - Updated diagonal slope tests to match PIL ground truth
   - Simplified to use centralized function

5. `nclone/data/tile_connectivity.pkl.gz`
   - Regenerated with correct tile-level connectivity
   - 4,600/9,248 traversable combinations (49.7%)

6. `nclone/tools/visualize_tile_types.py`
   - Previously fixed quarter circle/pipe rendering

## Test Results
✅ All 12 sub-node validity tests pass  
✅ Diagonal slopes match PIL ground truth  
✅ Raised steep slopes match PIL ground truth  
✅ Quarter circles correct  
✅ No more green nodes in solid areas  

## Impact
These were **critical bugs** affecting:
- Pathfinding through diagonal slopes
- Movement near raised platforms
- Adjacency graph accuracy
- Reachability analysis

All fixed and verified with automated tests and visual inspection!

