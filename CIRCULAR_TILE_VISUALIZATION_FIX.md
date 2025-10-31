# Quarter Circle and Quarter Pipe Visualization Fix

## Problem
In `visualize_tile_types.py`, quarter circles (types 10-13) and quarter pipes (types 14-17) were only showing a small dot at the center instead of the actual circular arc geometry.

## Root Cause
The original code had a placeholder comment "simplified - just draw circle center marker" and only drew a 10-pixel dot:

```python
# Old code (WRONG):
draw.ellipse(
    [cx - 5, cy - 5, cx + 5, cy + 5],
    fill=(255, 0, 0),
    outline=(255, 0, 0)
)
```

This didn't show the actual circular boundary that defines the solid vs traversable regions for these tiles.

## Solution
Implemented proper circular arc rendering matching the Cairo renderer logic from `shared_tile_renderer.py`:

### Quarter Circles (Types 10-13)
- **Type 10**: Bottom-right quarter circle - solid in BR corner, center at TL (0,0)
- **Type 11**: Bottom-left quarter circle - solid in BL corner, center at TR (24,0)
- **Type 12**: Top-left quarter circle - solid in TL corner, center at BR (24,24)
- **Type 13**: Top-right quarter circle - solid in TR corner, center at BL (0,24)

Angles: `a1 = (π/2) × (tile_type - 10)`, `a2 = (π/2) × (tile_type - 9)`

### Quarter Pipes (Types 14-17)
- **Type 14**: Top-left quarter pipe - hollow in TL, solid elsewhere
- **Type 15**: Top-right quarter pipe - hollow in TR, solid elsewhere
- **Type 16**: Bottom-right quarter pipe - hollow in BR, solid elsewhere
- **Type 17**: Bottom-left quarter pipe - hollow in BL, solid elsewhere

Angles: `a1 = π + (π/2) × (tile_type - 10)`, `a2 = π + (π/2) × (tile_type - 9)`

## Implementation Details

### New Rendering Code
```python
# Draw the full circle outline for reference
draw.ellipse(
    [cx - radius, cy - radius, cx + radius, cy + radius],
    outline=(255, 0, 255),  # Magenta outline
    width=2
)

# Calculate angles matching Cairo renderer
if tile_type < 14:
    # Quarter circles (10-13)
    a1_rad = (math.pi / 2) * (tile_type - 10)
    a2_rad = (math.pi / 2) * (tile_type - 9)
else:
    # Quarter pipes (14-17)
    a1_rad = math.pi + (math.pi / 2) * (tile_type - 10)
    a2_rad = math.pi + (math.pi / 2) * (tile_type - 9)

# Convert to PIL degrees (starts at 3 o'clock, goes counterclockwise)
a1_deg = -math.degrees(a1_rad)
a2_deg = -math.degrees(a2_rad)

# Draw the arc
draw.arc(
    [cx - radius, cy - radius, cx + radius, cy + radius],
    start=a1_deg,
    end=a2_deg,
    fill=(255, 0, 0),  # Red arc
    width=4
)

# Draw center marker
draw.ellipse(
    [cx - 3, cy - 3, cx + 3, cy + 3],
    fill=(255, 0, 0)
)
```

## Visual Legend

The updated visualization now shows:
- **Light grey (220,220,220)**: Traversable area
- **Dark grey (100,100,100)**: Solid area
- **Magenta circle outline**: Full circle boundary (24-pixel radius)
- **Red arc**: The actual arc segment defining the tile geometry
- **Red dot**: Center point of the circle
- **Red lines**: Diagonal segments (for slope tiles)
- **Green circles**: Valid sub-nodes (traversable)
- **Red X**: Invalid sub-nodes (in solid areas)
- **Blue grid lines**: 12-pixel sub-node boundaries

## Testing

Tested on:
- Type 10 (quarter circle BR)
- Type 12 (quarter circle TL)
- Type 14 (quarter pipe TL)
- Type 16 (quarter pipe BR)

All circular geometries now render correctly, matching the actual collision boundaries used in the game physics.

## Files Modified
- `/home/tetra/projects/nclone/nclone/tools/visualize_tile_types.py`

## Regenerated Outputs
- `debug_output/tile_visualization_grid.png` - Grid showing all 34 tile types
- `debug_output/tile_type_10.png` through `tile_type_17.png` - Individual circular tiles

