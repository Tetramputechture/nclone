# Quarter Pipe Tiles (14-17) Geometry Fix

## Problem

Tile types 14-17 (quarter pipes) were incorrectly marking some sub-nodes as invalid (solid). However, quarter pipes should be fully traversable for navigation purposes, even though they render with a visual solid corner region.

## Root Cause

The geometric check in `_check_subnode_validity_simple()` was treating the visual corner region as solid for collision/pathfinding purposes. However, in N++, quarter pipes are fully traversable - the solid corner is visual only and doesn't block navigation.

## Solution

Quarter pipes are **fully traversable** for navigation and pathfinding purposes. The visual solid corner region is for rendering only and does not affect collision or movement.

**Correct logic (after fix):**
```python
# Quarter pipes (14-17): All positions are traversable
if tile_type in [14, 15, 16, 17]:
    return True
```

This marks all pixels in quarter pipe tiles as non-solid/traversable, allowing free movement through the entire tile.

## Tile-by-Tile Fix

All quarter pipe tiles (14-17) now use the same logic:

### Tile 14: Top-Left Quarter Pipe
- **Visual appearance:** Shows solid corner in top-left
- **Navigation:** Fully traversable (all positions valid)
- **Logic:** `return True`

### Tile 15: Top-Right Quarter Pipe
- **Visual appearance:** Shows solid corner in top-right
- **Navigation:** Fully traversable (all positions valid)
- **Logic:** `return True`

### Tile 16: Bottom-Right Quarter Pipe
- **Visual appearance:** Shows solid corner in bottom-right
- **Navigation:** Fully traversable (all positions valid)
- **Logic:** `return True`

### Tile 17: Bottom-Left Quarter Pipe
- **Visual appearance:** Shows solid corner in bottom-left
- **Navigation:** Fully traversable (all positions valid)
- **Logic:** `return True`

## Verification

### Before Fix
Tiles 14-17 showed some sub-nodes as invalid based on geometric calculations:
```
Tile Type 14: Quarter Pipe TL
Sub-nodes:
  1: (0,0)@(6,6) = X    <- Incorrectly marked as solid
  2: (1,0)@(18,6) = O
  3: (0,1)@(6,18) = O
  4: (1,1)@(18,18) = O
```

### After Fix
All quarter pipe tiles now correctly show all 4 valid sub-nodes:

**All Tiles 14-17:**
```
Sub-nodes:
  1: (0,0)@(6,6) = O    <- Now valid
  2: (1,0)@(18,6) = O   <- Valid
  3: (0,1)@(6,18) = O   <- Valid
  4: (1,1)@(18,18) = O  <- Valid
```

The visual rendering may still show a solid corner region, but this is purely visual and doesn't affect navigation or pathfinding.

## Within-Tile Connectivity

All 4 sub-nodes within each quarter pipe can reach each other since the entire tile is traversable:

```
Tile 14 connectivity matrix:
       (0, 0)   (1, 0)   (0, 1)   (1, 1)  
(0, 0)   -       ✓        ✓        ✓
(1, 0)   ✓       -        ✓        ✓
(0, 1)   ✓       ✓        -        ✓
(1, 1)   ✓       ✓        ✓        -
```

## Impact

This fix makes quarter pipe tiles fully traversable for pathfinding and reachability analysis, matching the actual N++ game behavior. The visual solid corner is for rendering purposes only and doesn't affect collision detection or navigation. All sub-nodes are now valid, allowing the ninja to move freely through these tiles.

**Important Note:** While all positions within quarter pipe tiles are traversable, movement FROM adjacent tiles INTO quarter pipe tiles still depends on the precomputed 8-direction connectivity table, which considers tile edge geometry and collision rules.

## Files Modified

- `nclone/graph/reachability/fast_graph_builder.py` - Fixed `_check_subnode_validity_simple()` for tiles 14-17
- `debug_output/*.png` - Regenerated all tile visualizations

## Testing

Run the ASCII tile renderer to verify:
```bash
python -m nclone.tools.ascii_tile_renderer --tile 14
python -m nclone.tools.ascii_tile_renderer --tile 15
python -m nclone.tools.ascii_tile_renderer --tile 16
python -m nclone.tools.ascii_tile_renderer --tile 17
```

Or regenerate PNG visualizations:
```bash
python -m nclone.tools.visualize_tile_types --all
```

