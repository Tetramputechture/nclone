#!/usr/bin/env python3
"""
Debug tile rendering by examining actual tiles in the doortest map and their definitions.

This script will:
1. Load the doortest map
2. Identify all unique tile types present
3. Show the exact definitions for each tile type
4. Create a visual comparison between expected and actual rendering
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, Polygon, Wedge

# Add nclone to path
sys.path.insert(0, '/workspace/nclone')

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.constants.physics_constants import TILE_PIXEL_SIZE
from nclone.tile_definitions import (
    TILE_SEGMENT_DIAG_MAP, 
    TILE_SEGMENT_CIRCULAR_MAP,
    TILE_GRID_EDGE_MAP,
    TILE_SEGMENT_ORTHO_MAP
)

def analyze_doortest_tiles():
    """Analyze all tiles present in the doortest map."""
    print("=" * 80)
    print("🔍 ANALYZING DOORTEST MAP TILES")
    print("=" * 80)
    
    # Load environment
    env = BasicLevelNoGold(
        render_mode="rgb_array",
        enable_frame_stack=False,
        enable_debug_overlay=False,
        eval_mode=False,
        seed=42
    )
    env.reset()
    level_data = env.level_data
    
    print(f"📁 Map size: {level_data.width}x{level_data.height} tiles")
    
    # Find all unique tile types
    unique_tiles = set()
    tile_positions = {}
    
    for tile_y in range(level_data.height):
        for tile_x in range(level_data.width):
            tile_value = level_data.tiles[tile_y, tile_x]
            unique_tiles.add(tile_value)
            
            if tile_value not in tile_positions:
                tile_positions[tile_value] = []
            tile_positions[tile_value].append((tile_x, tile_y))
    
    unique_tiles = sorted(unique_tiles)
    print(f"🎯 Found {len(unique_tiles)} unique tile types: {unique_tiles}")
    
    # Analyze each tile type
    for tile_value in unique_tiles:
        positions = tile_positions[tile_value]
        print(f"\n📋 TILE TYPE {tile_value}:")
        print(f"   Count: {len(positions)} tiles")
        print(f"   First position: {positions[0] if positions else 'None'}")
        
        # Show tile definitions
        if tile_value in TILE_GRID_EDGE_MAP:
            grid_edges = TILE_GRID_EDGE_MAP[tile_value]
            print(f"   Grid edges: {grid_edges}")
        
        if tile_value in TILE_SEGMENT_ORTHO_MAP:
            ortho_segments = TILE_SEGMENT_ORTHO_MAP[tile_value]
            print(f"   Ortho segments: {ortho_segments}")
        
        if tile_value in TILE_SEGMENT_DIAG_MAP:
            diag_segment = TILE_SEGMENT_DIAG_MAP[tile_value]
            print(f"   Diagonal segment: {diag_segment}")
        
        if tile_value in TILE_SEGMENT_CIRCULAR_MAP:
            circular_segment = TILE_SEGMENT_CIRCULAR_MAP[tile_value]
            print(f"   Circular segment: {circular_segment}")
    
    return level_data, unique_tiles, tile_positions

def create_accurate_tile_patch(tile_value, tile_x, tile_y):
    """
    Create the most accurate possible tile patch based on actual definitions.
    
    This function uses the EXACT tile definitions from tile_definitions.py
    """
    patches_list = []
    
    if tile_value == 0:
        # Empty tile - no visual representation
        return patches_list
    elif tile_value == 1:
        # Full solid tile
        rect = Rectangle((tile_x, tile_y), TILE_PIXEL_SIZE, TILE_PIXEL_SIZE, 
                        facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
        patches_list.append(rect)
    elif tile_value == 2:
        # Top half solid - EXACT: from tile definitions, this has grid edges [1,1,1,1,0,0,1,0,0,0,1,0]
        # This means top and middle horizontal edges are solid
        rect = Rectangle((tile_x, tile_y), TILE_PIXEL_SIZE, TILE_PIXEL_SIZE/2, 
                       facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
        patches_list.append(rect)
    elif tile_value == 3:
        # Right half solid - EXACT: grid edges [0,1,0,0,0,1,0,0,1,1,1,1]
        rect = Rectangle((tile_x + TILE_PIXEL_SIZE/2, tile_y), TILE_PIXEL_SIZE/2, TILE_PIXEL_SIZE, 
                       facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
        patches_list.append(rect)
    elif tile_value == 4:
        # Bottom half solid - EXACT: grid edges [0,0,1,1,1,1,0,1,0,0,0,1]
        rect = Rectangle((tile_x, tile_y + TILE_PIXEL_SIZE/2), TILE_PIXEL_SIZE, TILE_PIXEL_SIZE/2, 
                       facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
        patches_list.append(rect)
    elif tile_value == 5:
        # Left half solid - EXACT: grid edges [1,0,0,0,1,0,1,1,1,1,0,0]
        rect = Rectangle((tile_x, tile_y), TILE_PIXEL_SIZE/2, TILE_PIXEL_SIZE, 
                       facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
        patches_list.append(rect)
    elif tile_value in [6, 7, 8, 9]:
        # 45-degree slopes - use EXACT diagonal segment definitions
        if tile_value in TILE_SEGMENT_DIAG_MAP:
            (x1, y1), (x2, y2) = TILE_SEGMENT_DIAG_MAP[tile_value]
            
            # Create triangle based on exact diagonal segment
            if tile_value == 6:  # ((0, 24), (24, 0)) - top-left to bottom-right
                triangle = Polygon([(tile_x, tile_y), (tile_x + TILE_PIXEL_SIZE, tile_y), 
                                  (tile_x, tile_y + TILE_PIXEL_SIZE)], 
                                 facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
            elif tile_value == 7:  # ((0, 0), (24, 24)) - top-left to bottom-right
                triangle = Polygon([(tile_x, tile_y), (tile_x + TILE_PIXEL_SIZE, tile_y), 
                                  (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE)], 
                                 facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
            elif tile_value == 8:  # ((24, 0), (0, 24)) - top-right to bottom-left
                triangle = Polygon([(tile_x, tile_y + TILE_PIXEL_SIZE), (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE), 
                                  (tile_x + TILE_PIXEL_SIZE, tile_y)], 
                                 facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
            elif tile_value == 9:  # ((24, 24), (0, 0)) - bottom-right to top-left
                triangle = Polygon([(tile_x, tile_y + TILE_PIXEL_SIZE), (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE), 
                                  (tile_x, tile_y)], 
                                 facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
            patches_list.append(triangle)
    elif tile_value in [10, 11, 12, 13]:
        # Quarter circles - use EXACT circular segment definitions
        if tile_value in TILE_SEGMENT_CIRCULAR_MAP:
            (cx, cy), (qx, qy), is_convex = TILE_SEGMENT_CIRCULAR_MAP[tile_value]
            center_x = tile_x + cx
            center_y = tile_y + cy
            
            if is_convex:  # Solid quarter circles
                if tile_value == 10:  # ((0, 0), (1, 1), True) - bottom-right quarter
                    wedge = Wedge((center_x, center_y), TILE_PIXEL_SIZE, 0, 90, 
                                        facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
                elif tile_value == 11:  # ((24, 0), (-1, 1), True) - bottom-left quarter
                    wedge = Wedge((center_x, center_y), TILE_PIXEL_SIZE, 90, 180, 
                                        facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
                elif tile_value == 12:  # ((24, 24), (-1, -1), True) - top-left quarter
                    wedge = Wedge((center_x, center_y), TILE_PIXEL_SIZE, 180, 270, 
                                        facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
                elif tile_value == 13:  # ((0, 24), (1, -1), True) - top-right quarter
                    wedge = Wedge((center_x, center_y), TILE_PIXEL_SIZE, 270, 360, 
                                        facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
                patches_list.append(wedge)
    elif tile_value in [14, 15, 16, 17]:
        # Quarter pipes - use EXACT circular segment definitions
        if tile_value in TILE_SEGMENT_CIRCULAR_MAP:
            (cx, cy), (qx, qy), is_convex = TILE_SEGMENT_CIRCULAR_MAP[tile_value]
            
            if not is_convex:  # Hollow quarter pipes - create L-shapes
                if tile_value == 14:  # ((24, 24), (-1, -1), False) - hollow top-left
                    # L-shape: solid everywhere except top-left quarter
                    rect1 = Rectangle((tile_x, tile_y + TILE_PIXEL_SIZE/2), TILE_PIXEL_SIZE, TILE_PIXEL_SIZE/2, 
                                    facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)  # Bottom part
                    rect2 = Rectangle((tile_x + TILE_PIXEL_SIZE/2, tile_y), TILE_PIXEL_SIZE/2, TILE_PIXEL_SIZE/2, 
                                    facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)  # Top-right part
                    patches_list.extend([rect1, rect2])
                elif tile_value == 15:  # ((0, 24), (1, -1), False) - hollow top-right
                    rect1 = Rectangle((tile_x, tile_y + TILE_PIXEL_SIZE/2), TILE_PIXEL_SIZE, TILE_PIXEL_SIZE/2, 
                                    facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)  # Bottom part
                    rect2 = Rectangle((tile_x, tile_y), TILE_PIXEL_SIZE/2, TILE_PIXEL_SIZE/2, 
                                    facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)  # Top-left part
                    patches_list.extend([rect1, rect2])
                elif tile_value == 16:  # ((0, 0), (1, 1), False) - hollow bottom-right
                    rect1 = Rectangle((tile_x, tile_y), TILE_PIXEL_SIZE, TILE_PIXEL_SIZE/2, 
                                    facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)  # Top part
                    rect2 = Rectangle((tile_x, tile_y + TILE_PIXEL_SIZE/2), TILE_PIXEL_SIZE/2, TILE_PIXEL_SIZE/2, 
                                    facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)  # Bottom-left part
                    patches_list.extend([rect1, rect2])
                elif tile_value == 17:  # ((24, 0), (-1, 1), False) - hollow bottom-left
                    rect1 = Rectangle((tile_x, tile_y), TILE_PIXEL_SIZE, TILE_PIXEL_SIZE/2, 
                                    facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)  # Top part
                    rect2 = Rectangle((tile_x + TILE_PIXEL_SIZE/2, tile_y + TILE_PIXEL_SIZE/2), TILE_PIXEL_SIZE/2, TILE_PIXEL_SIZE/2, 
                                    facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)  # Bottom-right part
                    patches_list.extend([rect1, rect2])
    elif tile_value in [18, 19, 20, 21]:
        # Short mild slopes - use EXACT diagonal segment definitions
        if tile_value in TILE_SEGMENT_DIAG_MAP:
            (x1, y1), (x2, y2) = TILE_SEGMENT_DIAG_MAP[tile_value]
            
            # Create polygon based on exact diagonal segment endpoints
            if tile_value == 18:  # ((0, 12), (24, 0)) - mild slope up-left
                polygon = Polygon([(tile_x, tile_y + TILE_PIXEL_SIZE), 
                                 (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE),
                                 (tile_x + x2, tile_y + y2),
                                 (tile_x + x1, tile_y + y1)], 
                                facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
            elif tile_value == 19:  # ((0, 0), (24, 12)) - mild slope up-right
                polygon = Polygon([(tile_x, tile_y + TILE_PIXEL_SIZE), 
                                 (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE),
                                 (tile_x + x2, tile_y + y2),
                                 (tile_x + x1, tile_y + y1)], 
                                facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
            elif tile_value == 20:  # ((24, 12), (0, 24)) - mild slope down-right
                polygon = Polygon([(tile_x + x1, tile_y + y1), 
                                 (tile_x + x2, tile_y + y2),
                                 (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE),
                                 (tile_x, tile_y + TILE_PIXEL_SIZE)], 
                                facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
            elif tile_value == 21:  # ((24, 24), (0, 12)) - mild slope down-left
                polygon = Polygon([(tile_x + x2, tile_y + y2), 
                                 (tile_x + x1, tile_y + y1),
                                 (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE),
                                 (tile_x, tile_y + TILE_PIXEL_SIZE)], 
                                facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
            patches_list.append(polygon)
    elif tile_value in [22, 23, 24, 25]:
        # Raised mild slopes - use EXACT diagonal segment definitions
        if tile_value in TILE_SEGMENT_DIAG_MAP:
            (x1, y1), (x2, y2) = TILE_SEGMENT_DIAG_MAP[tile_value]
            
            # Create polygon based on exact diagonal segment endpoints
            if tile_value == 22:  # ((0, 24), (24, 12)) - raised mild slope
                polygon = Polygon([(tile_x + x1, tile_y + y1), 
                                 (tile_x + x2, tile_y + y2),
                                 (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE),
                                 (tile_x, tile_y + TILE_PIXEL_SIZE)], 
                                facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
            elif tile_value == 23:  # ((0, 12), (24, 24)) - raised mild slope
                polygon = Polygon([(tile_x + x1, tile_y + y1), 
                                 (tile_x + x2, tile_y + y2),
                                 (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE),
                                 (tile_x, tile_y + TILE_PIXEL_SIZE)], 
                                facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
            elif tile_value == 24:  # ((24, 0), (0, 12)) - raised mild slope
                polygon = Polygon([(tile_x, tile_y + TILE_PIXEL_SIZE), 
                                 (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE),
                                 (tile_x + x1, tile_y + y1),
                                 (tile_x + x2, tile_y + y2)], 
                                facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
            elif tile_value == 25:  # ((24, 12), (0, 0)) - raised mild slope
                polygon = Polygon([(tile_x, tile_y + TILE_PIXEL_SIZE), 
                                 (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE),
                                 (tile_x + x1, tile_y + y1),
                                 (tile_x + x2, tile_y + y2)], 
                                facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
            patches_list.append(polygon)
    elif tile_value in [26, 27, 28, 29]:
        # Short steep slopes - use EXACT diagonal segment definitions
        if tile_value in TILE_SEGMENT_DIAG_MAP:
            (x1, y1), (x2, y2) = TILE_SEGMENT_DIAG_MAP[tile_value]
            
            # Create polygon based on exact diagonal segment endpoints
            if tile_value == 26:  # ((0, 24), (12, 0)) - steep slope up-left
                polygon = Polygon([(tile_x + x1, tile_y + y1), 
                                 (tile_x + x2, tile_y + y2),
                                 (tile_x + TILE_PIXEL_SIZE/2, tile_y + TILE_PIXEL_SIZE),
                                 (tile_x, tile_y + TILE_PIXEL_SIZE)], 
                                facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
            elif tile_value == 27:  # ((12, 0), (24, 24)) - steep slope up-right
                polygon = Polygon([(tile_x + TILE_PIXEL_SIZE/2, tile_y + TILE_PIXEL_SIZE), 
                                 (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE),
                                 (tile_x + x2, tile_y + y2),
                                 (tile_x + x1, tile_y + y1)], 
                                facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
            elif tile_value == 28:  # ((24, 0), (12, 24)) - steep slope down-right
                polygon = Polygon([(tile_x + x1, tile_y + y1), 
                                 (tile_x + x2, tile_y + y2),
                                 (tile_x + TILE_PIXEL_SIZE/2, tile_y + TILE_PIXEL_SIZE),
                                 (tile_x + TILE_PIXEL_SIZE, tile_y + TILE_PIXEL_SIZE)], 
                                facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
            elif tile_value == 29:  # ((12, 24), (0, 0)) - steep slope down-left
                polygon = Polygon([(tile_x + x2, tile_y + y2), 
                                 (tile_x + x1, tile_y + y1),
                                 (tile_x, tile_y + TILE_PIXEL_SIZE),
                                 (tile_x + TILE_PIXEL_SIZE/2, tile_y + TILE_PIXEL_SIZE)], 
                                facecolor='#C0C0C0', edgecolor='#A0A0A0', linewidth=1)
            patches_list.append(polygon)
    else:
        # Other tile types - draw as partial solid for debugging
        rect = Rectangle((tile_x + TILE_PIXEL_SIZE/4, tile_y + TILE_PIXEL_SIZE/4), 
                        TILE_PIXEL_SIZE/2, TILE_PIXEL_SIZE/2, 
                        facecolor='#FF0000', edgecolor='#000000', linewidth=2)  # Red for debugging
        patches_list.append(rect)
    
    return patches_list

def create_tile_debugging_visualization():
    """Create a detailed tile debugging visualization."""
    print("\n" + "=" * 80)
    print("🎨 CREATING TILE DEBUGGING VISUALIZATION")
    print("=" * 80)
    
    # Analyze tiles
    level_data, unique_tiles, tile_positions = analyze_doortest_tiles()
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    
    # Set up coordinate system
    map_width = level_data.width * TILE_PIXEL_SIZE
    map_height = level_data.height * TILE_PIXEL_SIZE
    
    ax.set_xlim(0, map_width)
    ax.set_ylim(0, map_height)
    ax.invert_yaxis()  # Match game coordinate system
    
    # Draw each tile with accurate rendering
    print("🏗️ Drawing tiles with accurate definitions...")
    tile_count = 0
    for tile_y in range(level_data.height):
        for tile_x in range(level_data.width):
            tile_value = level_data.tiles[tile_y, tile_x]
            if tile_value != 0:  # Non-empty tile
                pixel_x = tile_x * TILE_PIXEL_SIZE
                pixel_y = tile_y * TILE_PIXEL_SIZE
                
                # Create accurate tile patches
                tile_patches = create_accurate_tile_patch(tile_value, pixel_x, pixel_y)
                for patch in tile_patches:
                    ax.add_patch(patch)
                
                # Add tile type label for debugging
                if tile_value != 1:  # Don't label every solid tile
                    ax.text(pixel_x + TILE_PIXEL_SIZE/2, pixel_y + TILE_PIXEL_SIZE/2, 
                           str(tile_value), fontsize=8, ha='center', va='center',
                           color='red', fontweight='bold')
                
                tile_count += 1
    
    print(f"✅ Drew {tile_count} non-empty tiles")
    
    # Add grid for reference
    for x in range(0, map_width + 1, TILE_PIXEL_SIZE):
        ax.axvline(x, color='gray', alpha=0.3, linewidth=0.5)
    for y in range(0, map_height + 1, TILE_PIXEL_SIZE):
        ax.axhline(y, color='gray', alpha=0.3, linewidth=0.5)
    
    # Add title and labels
    ax.set_title(f'Tile Debugging Visualization - Doortest Map\n'
                f'Unique tile types: {unique_tiles}\n'
                f'Using EXACT tile definitions from tile_definitions.py',
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlabel('X Position (pixels)', fontsize=14)
    ax.set_ylabel('Y Position (pixels)', fontsize=14)
    
    # Add legend for tile types
    legend_text = "Tile Types Found:\n"
    for tile_value in unique_tiles:
        count = len(tile_positions[tile_value])
        legend_text += f"  {tile_value}: {count} tiles\n"
    
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    
    # Save the debugging visualization
    output_path = "/workspace/nclone/tile_debugging_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved tile debugging visualization to: {output_path}")
    
    plt.close()
    
    return True

def main():
    """Main function."""
    print("🔍 TILE RENDERING DEBUG ANALYSIS")
    print("=" * 80)
    
    success = create_tile_debugging_visualization()
    
    if success:
        print("\n" + "=" * 80)
        print("🎉 SUCCESS! Tile debugging visualization created!")
        print("✅ Analyzed all tiles in doortest map")
        print("✅ Used exact tile definitions from tile_definitions.py")
        print("✅ Created visual debugging output")
        print("✅ Visualization saved as tile_debugging_visualization.png")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ FAILED! Could not create tile debugging visualization")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(main())