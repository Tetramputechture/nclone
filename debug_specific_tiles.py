#!/usr/bin/env python3
"""
Debug specific tile types by creating individual tile visualizations.

This script focuses on the complex tile types that are likely causing rendering issues:
- Quarter circles (10-13)
- Quarter pipes (14-17) 
- Slopes (18-33)

It will create individual tile visualizations to compare with actual game rendering.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, Polygon, Wedge

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.constants.physics_constants import TILE_PIXEL_SIZE
from nclone.tile_definitions import (
    TILE_SEGMENT_DIAG_MAP, 
    TILE_SEGMENT_CIRCULAR_MAP,
    TILE_GRID_EDGE_MAP,
    TILE_SEGMENT_ORTHO_MAP
)

def create_individual_tile_visualization(tile_value, title_suffix=""):
    """Create a detailed visualization of a single tile type."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Set up coordinate system for a single tile
    ax.set_xlim(-2, TILE_PIXEL_SIZE + 2)
    ax.set_ylim(-2, TILE_PIXEL_SIZE + 2)
    ax.invert_yaxis()  # Match game coordinate system
    
    # Draw tile at (0, 0)
    tile_x, tile_y = 0, 0
    
    print(f"\n🔍 ANALYZING TILE TYPE {tile_value}:")
    
    # Show all definitions for this tile
    if tile_value in TILE_GRID_EDGE_MAP:
        grid_edges = TILE_GRID_EDGE_MAP[tile_value]
        print(f"   Grid edges: {grid_edges}")
    
    if tile_value in TILE_SEGMENT_ORTHO_MAP:
        ortho_segments = TILE_SEGMENT_ORTHO_MAP[tile_value]
        print(f"   Ortho segments: {ortho_segments}")
    
    if tile_value in TILE_SEGMENT_DIAG_MAP:
        diag_segment = TILE_SEGMENT_DIAG_MAP[tile_value]
        print(f"   Diagonal segment: {diag_segment}")
        
        # Draw the diagonal line for reference
        (x1, y1), (x2, y2) = diag_segment
        ax.plot([tile_x + x1, tile_x + x2], [tile_y + y1, tile_y + y2], 
               'r-', linewidth=3, alpha=0.7, label=f'Diagonal: ({x1},{y1}) to ({x2},{y2})')
    
    if tile_value in TILE_SEGMENT_CIRCULAR_MAP:
        circular_segment = TILE_SEGMENT_CIRCULAR_MAP[tile_value]
        print(f"   Circular segment: {circular_segment}")
        
        # Draw the center point for reference
        (cx, cy), (qx, qy), is_convex = circular_segment
        center_x = tile_x + cx
        center_y = tile_y + cy
        ax.plot(center_x, center_y, 'ro', markersize=8, label=f'Center: ({cx},{cy})')
        
        # Draw quadrant direction
        ax.arrow(center_x, center_y, qx * 8, qy * 8, head_width=2, head_length=2, 
                fc='blue', ec='blue', alpha=0.7, label=f'Quadrant: ({qx},{qy})')
    
    # Create the actual tile rendering
    patches_list = create_accurate_tile_patch_debug(tile_value, tile_x, tile_y)
    for patch in patches_list:
        ax.add_patch(patch)
    
    # Add grid lines for reference
    for x in range(0, TILE_PIXEL_SIZE + 1, TILE_PIXEL_SIZE // 4):
        ax.axvline(x, color='gray', alpha=0.3, linewidth=0.5)
    for y in range(0, TILE_PIXEL_SIZE + 1, TILE_PIXEL_SIZE // 4):
        ax.axhline(y, color='gray', alpha=0.3, linewidth=0.5)
    
    # Add coordinate labels
    ax.text(0, -1, '(0,0)', ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(TILE_PIXEL_SIZE, -1, f'({TILE_PIXEL_SIZE},0)', ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(0, TILE_PIXEL_SIZE + 1, f'(0,{TILE_PIXEL_SIZE})', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(TILE_PIXEL_SIZE, TILE_PIXEL_SIZE + 1, f'({TILE_PIXEL_SIZE},{TILE_PIXEL_SIZE})', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add title and labels
    ax.set_title(f'Tile Type {tile_value} - Detailed Analysis{title_suffix}\n'
                f'Size: {TILE_PIXEL_SIZE}x{TILE_PIXEL_SIZE} pixels',
                fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xlabel('X Position (pixels)', fontsize=12)
    ax.set_ylabel('Y Position (pixels)', fontsize=12)
    
    # Add legend if there are reference elements
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    plt.tight_layout()
    
    # Save individual tile visualization
    output_path = f"/workspace/nclone/tile_{tile_value}_debug.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved to: {output_path}")
    
    plt.close()

def create_accurate_tile_patch_debug(tile_value, tile_x, tile_y):
    """
    Create accurate tile patch with detailed debugging information.
    """
    patches_list = []
    
    if tile_value == 0:
        # Empty tile - no visual representation
        return patches_list
    elif tile_value == 1:
        # Full solid tile
        rect = Rectangle((tile_x, tile_y), TILE_PIXEL_SIZE, TILE_PIXEL_SIZE, 
                        facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)
        patches_list.append(rect)
    elif tile_value in [10, 11, 12, 13]:
        # Quarter circles - EXACT implementation
        if tile_value in TILE_SEGMENT_CIRCULAR_MAP:
            (cx, cy), (qx, qy), is_convex = TILE_SEGMENT_CIRCULAR_MAP[tile_value]
            center_x = tile_x + cx
            center_y = tile_y + cy
            
            print(f"   Quarter circle: center=({cx},{cy}), quadrant=({qx},{qy}), convex={is_convex}")
            
            if is_convex:  # Solid quarter circles
                if tile_value == 10:  # ((0, 0), (1, 1), True) - center at top-left, quadrant bottom-right
                    # This should be a quarter circle in the bottom-right quadrant
                    wedge = Wedge((center_x, center_y), TILE_PIXEL_SIZE, 0, 90, 
                                        facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)
                    print(f"   Creating wedge: center=({center_x},{center_y}), radius={TILE_PIXEL_SIZE}, angles=0-90")
                elif tile_value == 11:  # ((24, 0), (-1, 1), True) - center at top-right, quadrant bottom-left
                    wedge = Wedge((center_x, center_y), TILE_PIXEL_SIZE, 90, 180, 
                                        facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)
                    print(f"   Creating wedge: center=({center_x},{center_y}), radius={TILE_PIXEL_SIZE}, angles=90-180")
                elif tile_value == 12:  # ((24, 24), (-1, -1), True) - center at bottom-right, quadrant top-left
                    wedge = Wedge((center_x, center_y), TILE_PIXEL_SIZE, 180, 270, 
                                        facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)
                    print(f"   Creating wedge: center=({center_x},{center_y}), radius={TILE_PIXEL_SIZE}, angles=180-270")
                elif tile_value == 13:  # ((0, 24), (1, -1), True) - center at bottom-left, quadrant top-right
                    wedge = Wedge((center_x, center_y), TILE_PIXEL_SIZE, 270, 360, 
                                        facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)
                    print(f"   Creating wedge: center=({center_x},{center_y}), radius={TILE_PIXEL_SIZE}, angles=270-360")
                patches_list.append(wedge)
    elif tile_value in [14, 15, 16, 17]:
        # Quarter pipes - EXACT implementation
        if tile_value in TILE_SEGMENT_CIRCULAR_MAP:
            (cx, cy), (qx, qy), is_convex = TILE_SEGMENT_CIRCULAR_MAP[tile_value]
            
            print(f"   Quarter pipe: center=({cx},{cy}), quadrant=({qx},{qy}), convex={is_convex}")
            
            if not is_convex:  # Hollow quarter pipes - create L-shapes
                if tile_value == 14:  # ((24, 24), (-1, -1), False) - hollow top-left
                    print(f"   Creating L-shape: hollow top-left quarter")
                    # L-shape: solid everywhere except top-left quarter
                    rect1 = Rectangle((tile_x, tile_y + TILE_PIXEL_SIZE/2), TILE_PIXEL_SIZE, TILE_PIXEL_SIZE/2, 
                                    facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)  # Bottom part
                    rect2 = Rectangle((tile_x + TILE_PIXEL_SIZE/2, tile_y), TILE_PIXEL_SIZE/2, TILE_PIXEL_SIZE/2, 
                                    facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)  # Top-right part
                    patches_list.extend([rect1, rect2])
                elif tile_value == 15:  # ((0, 24), (1, -1), False) - hollow top-right
                    print(f"   Creating L-shape: hollow top-right quarter")
                    rect1 = Rectangle((tile_x, tile_y + TILE_PIXEL_SIZE/2), TILE_PIXEL_SIZE, TILE_PIXEL_SIZE/2, 
                                    facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)  # Bottom part
                    rect2 = Rectangle((tile_x, tile_y), TILE_PIXEL_SIZE/2, TILE_PIXEL_SIZE/2, 
                                    facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)  # Top-left part
                    patches_list.extend([rect1, rect2])
                elif tile_value == 16:  # ((0, 0), (1, 1), False) - hollow bottom-right
                    print(f"   Creating L-shape: hollow bottom-right quarter")
                    rect1 = Rectangle((tile_x, tile_y), TILE_PIXEL_SIZE, TILE_PIXEL_SIZE/2, 
                                    facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)  # Top part
                    rect2 = Rectangle((tile_x, tile_y + TILE_PIXEL_SIZE/2), TILE_PIXEL_SIZE/2, TILE_PIXEL_SIZE/2, 
                                    facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)  # Bottom-left part
                    patches_list.extend([rect1, rect2])
                elif tile_value == 17:  # ((24, 0), (-1, 1), False) - hollow bottom-left
                    print(f"   Creating L-shape: hollow bottom-left quarter")
                    rect1 = Rectangle((tile_x, tile_y), TILE_PIXEL_SIZE, TILE_PIXEL_SIZE/2, 
                                    facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)  # Top part
                    rect2 = Rectangle((tile_x + TILE_PIXEL_SIZE/2, tile_y + TILE_PIXEL_SIZE/2), TILE_PIXEL_SIZE/2, TILE_PIXEL_SIZE/2, 
                                    facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)  # Bottom-right part
                    patches_list.extend([rect1, rect2])
    elif tile_value in TILE_SEGMENT_DIAG_MAP:
        # Slopes with diagonal segments
        (x1, y1), (x2, y2) = TILE_SEGMENT_DIAG_MAP[tile_value]
        print(f"   Diagonal slope: from ({x1},{y1}) to ({x2},{y2})")
        
        # Create polygon based on the diagonal segment and tile boundaries
        if tile_value == 18:  # ((0, 12), (24, 0)) - mild slope up-left
            polygon = Polygon([(tile_x, tile_y + TILE_PIXEL_SIZE), 
                             (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE),
                             (tile_x + x2, tile_y + y2),
                             (tile_x + x1, tile_y + y1)], 
                            facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)
        elif tile_value == 19:  # ((0, 0), (24, 12)) - mild slope up-right
            polygon = Polygon([(tile_x, tile_y + TILE_PIXEL_SIZE), 
                             (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE),
                             (tile_x + x2, tile_y + y2),
                             (tile_x + x1, tile_y + y1)], 
                            facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)
        elif tile_value == 20:  # ((24, 12), (0, 24)) - mild slope down-right
            polygon = Polygon([(tile_x + x1, tile_y + y1), 
                             (tile_x + x2, tile_y + y2),
                             (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE),
                             (tile_x, tile_y + TILE_PIXEL_SIZE)], 
                            facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)
        elif tile_value in [22, 23, 24, 25]:  # Raised mild slopes
            if tile_value == 22:  # ((0, 24), (24, 12))
                polygon = Polygon([(tile_x + x1, tile_y + y1), 
                                 (tile_x + x2, tile_y + y2),
                                 (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE),
                                 (tile_x, tile_y + TILE_PIXEL_SIZE)], 
                                facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)
            elif tile_value == 23:  # ((0, 12), (24, 24))
                polygon = Polygon([(tile_x + x1, tile_y + y1), 
                                 (tile_x + x2, tile_y + y2),
                                 (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE),
                                 (tile_x, tile_y + TILE_PIXEL_SIZE)], 
                                facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)
            elif tile_value == 24:  # ((24, 0), (0, 12))
                polygon = Polygon([(tile_x, tile_y + TILE_PIXEL_SIZE), 
                                 (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE),
                                 (tile_x + x1, tile_y + y1),
                                 (tile_x + x2, tile_y + y2)], 
                                facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)
        elif tile_value in [30, 31, 32, 33]:  # Raised steep slopes
            if tile_value == 30:  # ((12, 24), (24, 0))
                polygon = Polygon([(tile_x + x1, tile_y + y1), 
                                 (tile_x + x2, tile_y + y2),
                                 (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE),
                                 (tile_x, tile_y + TILE_PIXEL_SIZE)], 
                                facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)
            elif tile_value == 31:  # ((0, 0), (12, 24))
                polygon = Polygon([(tile_x + x1, tile_y + y1), 
                                 (tile_x + x2, tile_y + y2),
                                 (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE),
                                 (tile_x, tile_y + TILE_PIXEL_SIZE)], 
                                facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)
            elif tile_value == 32:  # ((12, 0), (0, 24))
                polygon = Polygon([(tile_x + x1, tile_y + y1), 
                                 (tile_x + x2, tile_y + y2),
                                 (tile_x, tile_y + TILE_PIXEL_SIZE),
                                 (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE)], 
                                facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)
            elif tile_value == 33:  # ((24, 24), (12, 0))
                polygon = Polygon([(tile_x + x1, tile_y + y1), 
                                 (tile_x + x2, tile_y + y2),
                                 (tile_x, tile_y + TILE_PIXEL_SIZE),
                                 (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE)], 
                                facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)
        else:
            # Generic slope polygon
            polygon = Polygon([(tile_x, tile_y + TILE_PIXEL_SIZE), 
                             (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE),
                             (tile_x + x2, tile_y + y2),
                             (tile_x + x1, tile_y + y1)], 
                            facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=2)
        
        patches_list.append(polygon)
    else:
        # Other tile types - draw as partial solid for debugging
        rect = Rectangle((tile_x + TILE_PIXEL_SIZE/4, tile_y + TILE_PIXEL_SIZE/4), 
                        TILE_PIXEL_SIZE/2, TILE_PIXEL_SIZE/2, 
                        facecolor='#FF0000', edgecolor='#000000', linewidth=2)  # Red for debugging
        patches_list.append(rect)
    
    return patches_list

def debug_specific_tiles():
    """Debug specific problematic tile types."""
    print("=" * 80)
    print("🔍 DEBUGGING SPECIFIC TILE TYPES")
    print("=" * 80)
    
    # Focus on the complex tile types found in doortest map
    complex_tiles = [13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 30, 31, 32, 33]
    
    print(f"🎯 Analyzing {len(complex_tiles)} complex tile types...")
    
    for tile_value in complex_tiles:
        create_individual_tile_visualization(tile_value)
    
    print(f"\n✅ Created individual visualizations for {len(complex_tiles)} tile types")
    
    return True

def main():
    """Main function."""
    print("🔍 SPECIFIC TILE TYPE DEBUG ANALYSIS")
    print("=" * 80)
    
    success = debug_specific_tiles()
    
    if success:
        print("\n" + "=" * 80)
        print("🎉 SUCCESS! Specific tile debugging completed!")
        print("✅ Created individual tile visualizations")
        print("✅ Analyzed complex tile definitions")
        print("✅ Generated detailed debugging output")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ FAILED! Could not complete tile debugging")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(main())