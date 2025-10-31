# Graph Debug - Next Steps for User

## Summary

I've implemented pixel-perfect edge sampling with comprehensive debug tools, but you're still reporting connectivity issues. Here's what we've verified and what needs investigation:

## What's Verified ✓

1. **Tile-level connectivity is correct**:
   - No edges between solid tiles
   - No edges from/to solid tiles
   - All cross-tile edges are Type 0 → Type 0 (empty to empty)

2. **Sampling is now pixel-perfect**:
   - 11 samples per cardinal direction (every 2 pixels across 20-pixel clearance)
   - ALL points must be clear (zero tolerance for solid pixels)
   - Edge case handling for insufficient samples near tile edges

3. **Debug tools created**:
   - `debug_tile_connectivity.py` - Analyzes precomputed table
   - `detailed_graph_debug.py` - Analyzes actual graph building
   - `visualize_tile_edges.py` - ASCII tile-level visualization

## What Needs Investigation ⚠

Based on your images showing nodes in unreachable areas, the issue is likely one of:

### Hypothesis 1: Sub-node placement near tile boundaries
**Problem**: Sub-nodes at positions (6, 6), (18, 6), (6, 18), (18, 18) might be too close to tile edges, causing the ninja (10px radius) to overlap into adjacent tiles' solid geometry.

**Test**: Check if nodes near tile boundaries with adjacent solid tiles are incorrectly marked as traversable.

**Fix**: Adjust sub-node positions away from edges, or add stricter boundary checks.

### Hypothesis 2: Complex tile geometry edge cases
**Problem**: Half-tiles, slopes, and circular segments have complex boundaries. The sample points might miss thin solid regions or corners.

**Test**: Identify which tile types appear in the problematic areas from your images.

**Fix**: Add geometry-specific sampling patterns or increase sample density for complex tiles.

### Hypothesis 3: Diagonal intermediate tile checks
**Problem**: Even though we check both intermediate tiles, the specific sub-node positions checked might not align with the actual solid geometry.

**Test**: Look for diagonal connections across corners in your images.

**Fix**: Sample multiple positions in intermediate tiles, not just the sub-node positions.

## Immediate Action Required from You

To pinpoint the exact issue, please provide:

1. **Specific tile coordinates** where impossible connections occur
   - From your images, identify a few (x, y) tile positions where nodes shouldn't exist or connect
   
2. **Tile types involved**
   - Run `detailed_graph_debug.py` and note which tile types are adjacent to the problematic areas
   
3. **Specific impossible edge**
   - Identify one specific node-to-node edge that shouldn't exist
   - Provide: source position, destination position, tile types

## Debug Commands to Run

```bash
# 1. Analyze the graph for your problematic map
python nclone/tools/detailed_graph_debug.py 0

# 2. Visualize tile-level edges
python nclone/tools/visualize_tile_edges.py 0

# 3. Check tile connectivity table
python nclone/tools/debug_tile_connectivity.py

# 4. Run test environment to see the issue
python nclone/test_environment.py --map 0 --visualize-adjacency-graph
```

## Possible Immediate Fixes to Try

###  Fix Option 1: Stricter sub-node validity (most likely)
If sub-nodes near tile boundaries are the issue:

```python
# In _generate_sub_nodes, add boundary distance check
SAFE_DISTANCE_FROM_EDGE = 2  # pixels

for (offset_x, offset_y), (sub_x, sub_y) in zip(SUB_NODE_OFFSETS, SUB_NODE_COORDS):
    pixel_x = tile_x * CELL_SIZE + offset_x
    pixel_y = tile_y * CELL_SIZE + offset_y
    
    # Check if sub-node is too close to a solid tile edge
    if _is_near_solid_boundary(pixel_x, pixel_y, tile_x, tile_y, tiles, SAFE_DISTANCE_FROM_EDGE):
        continue
    
    # ... rest of code
```

### Fix Option 2: Increase sampling density (if gaps remain)
If 11 samples aren't enough:

```python
# In tile_connectivity_precomputer.py
for x_offset in range(-10, 11, 1):  # Every pixel instead of every 2 pixels
    # ... sampling code
```

### Fix Option 3: Add ninja-radius-aware geometry checks
If the issue is ninja overlapping into adjacent solid geometry:

```python
# Check ninja (10px radius) clearance around sub-node
def _check_ninja_clearance(pixel_x, pixel_y, tile_x, tile_y, tiles):
    for dx in range(-10, 11):
        for dy in range(-10, 11):
            if dx*dx + dy*dy <= 100:  # Within 10px radius
                check_x = pixel_x + dx
                check_y = pixel_y + dy
                if _is_in_solid_area(check_x, check_y, tiles):
                    return False
    return True
```

## Next Steps

1. **User provides specific problematic coordinates** from images
2. **We trace exact decision path** for that specific connection
3. **Identify root cause** (sub-node placement, sampling gap, geometry, etc.)
4. **Implement targeted fix**
5. **Verify with regression test**

Please run the debug tools and provide the specific details so we can fix this holistically!

