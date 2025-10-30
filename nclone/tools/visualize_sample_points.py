"""
Visualize exact sample points used for tile connectivity checks.

This tool shows pixel-by-pixel what points are sampled when checking
if two tiles can be connected, helping identify sampling gaps.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from PIL import Image, ImageDraw
from nclone.tile_renderer import render_tile_to_array


def visualize_sample_points_for_connection(
    tile_a: int,
    tile_b: int,
    direction: str,
    src_sub_x: int,
    src_sub_y: int,
    dst_sub_x: int,
    dst_sub_y: int,
    output_file: str = "sample_points.png"
):
    """
    Visualize the exact sample points used for a specific tile-to-tile connection.
    
    Args:
        tile_a: Source tile type
        tile_b: Destination tile type
        direction: Direction name (N, NE, E, SE, S, SW, W, NW)
        src_sub_x: Source sub-node X (0 or 1)
        src_sub_y: Source sub-node Y (0 or 1)
        dst_sub_x: Destination sub-node X (0 or 1)
        dst_sub_y: Destination sub-node Y (0 or 1)
        output_file: Output PNG filename
    """
    
    # Render both tiles
    tile_a_img = render_tile_to_array(tile_a)
    tile_b_img = render_tile_to_array(tile_b)
    
    # Direction to delta mapping
    dir_map = {
        'N': (0, -1), 'NE': (1, -1), 'E': (1, 0), 'SE': (1, 1),
        'S': (0, 1), 'SW': (-1, 1), 'W': (-1, 0), 'NW': (-1, -1)
    }
    
    dx, dy = dir_map[direction]
    
    # Calculate sub-node pixel positions
    src_x = 6 if src_sub_x == 0 else 18
    src_y = 6 if src_sub_y == 0 else 18
    dst_x = 6 if dst_sub_x == 0 else 18
    dst_y = 6 if dst_sub_y == 0 else 18
    
    # Get sample points (replicate logic from tile_connectivity_precomputer.py)
    points_a = []
    points_b = []
    
    if dx == 0 and dy == -1:  # North
        for x_offset in range(-10, 11, 2):
            x_pos = src_x + x_offset
            if 0 <= x_pos < 24:
                points_a.append((x_pos, 23))
        for x_offset in range(-10, 11, 2):
            x_pos = dst_x + x_offset
            if 0 <= x_pos < 24:
                points_b.append((x_pos, 0))
                
    elif dx == 0 and dy == 1:  # South
        for x_offset in range(-10, 11, 2):
            x_pos = src_x + x_offset
            if 0 <= x_pos < 24:
                points_a.append((x_pos, 0))
        for x_offset in range(-10, 11, 2):
            x_pos = dst_x + x_offset
            if 0 <= x_pos < 24:
                points_b.append((x_pos, 23))
                
    elif dx == 1 and dy == 0:  # East
        for y_offset in range(-10, 11, 2):
            y_pos = src_y + y_offset
            if 0 <= y_pos < 24:
                points_a.append((23, y_pos))
        for y_offset in range(-10, 11, 2):
            y_pos = dst_y + y_offset
            if 0 <= y_pos < 24:
                points_b.append((0, y_pos))
                
    elif dx == -1 and dy == 0:  # West
        for y_offset in range(-10, 11, 2):
            y_pos = src_y + y_offset
            if 0 <= y_pos < 24:
                points_a.append((0, y_pos))
        for y_offset in range(-10, 11, 2):
            y_pos = dst_y + y_offset
            if 0 <= y_pos < 24:
                points_b.append((23, y_pos))
    
    # For diagonals, use L-shape pattern
    elif dx == 1 and dy == -1:  # Northeast
        for offset in range(-10, 11, 2):
            y_pos = src_y + offset
            if 0 <= y_pos < 24:
                points_a.append((23, y_pos))
            x_pos = src_x + offset
            if 0 <= x_pos < 24:
                points_a.append((x_pos, 23))
        for offset in range(-10, 11, 2):
            y_pos = dst_y + offset
            if 0 <= y_pos < 24:
                points_b.append((0, y_pos))
            x_pos = dst_x + offset
            if 0 <= x_pos < 24:
                points_b.append((x_pos, 0))
    
    # (Similar for other diagonals...)
    
    # Create composite image: tile_a on left, tile_b on right
    width = 24 * 2 + 10  # 2 tiles + spacing
    height = 24
    img = Image.new('RGB', (width * 4, height * 4), (40, 40, 40))  # Scale 4x for visibility
    
    # Paste tiles
    tile_a_pil = Image.fromarray(tile_a_img)
    tile_b_pil = Image.fromarray(tile_b_img)
    
    # Scale up 4x
    tile_a_pil = tile_a_pil.resize((24 * 4, 24 * 4), Image.NEAREST)
    tile_b_pil = tile_b_pil.resize((24 * 4, 24 * 4), Image.NEAREST)
    
    img.paste(tile_a_pil, (0, 0))
    img.paste(tile_b_pil, ((24 + 5) * 4, 0))
    
    # Draw sample points
    draw = ImageDraw.Draw(img)
    
    # Draw sub-node centers
    draw.ellipse([src_x * 4 - 4, src_y * 4 - 4, src_x * 4 + 4, src_y * 4 + 4], 
                 fill=(0, 255, 0), outline=(255, 255, 255))
    draw.ellipse([(24 + 5 + dst_x) * 4 - 4, dst_y * 4 - 4, 
                  (24 + 5 + dst_x) * 4 + 4, dst_y * 4 + 4], 
                 fill=(0, 255, 0), outline=(255, 255, 255))
    
    # Draw sample points as red dots
    for px, py in points_a:
        draw.rectangle([px * 4 - 2, py * 4 - 2, px * 4 + 2, py * 4 + 2], fill=(255, 0, 0))
    
    for px, py in points_b:
        draw.rectangle([(24 + 5 + px) * 4 - 2, py * 4 - 2, 
                       (24 + 5 + px) * 4 + 2, py * 4 + 2], fill=(255, 0, 0))
    
    img.save(output_file)
    
    print(f"\n{'='*70}")
    print("SAMPLE POINT VISUALIZATION")
    print(f"{'='*70}")
    print(f"Tile A: {tile_a}")
    print(f"Tile B: {tile_b}")
    print(f"Direction: {direction}")
    print(f"Source sub-node: ({src_sub_x}, {src_sub_y}) → pixel ({src_x}, {src_y})")
    print(f"Dest sub-node: ({dst_sub_x}, {dst_sub_y}) → pixel ({dst_x}, {dst_y})")
    print(f"\nSample points in Tile A: {len(points_a)}")
    for i, (px, py) in enumerate(points_a):
        print(f"  {i+1}. ({px}, {py})")
    print(f"\nSample points in Tile B: {len(points_b)}")
    for i, (px, py) in enumerate(points_b):
        print(f"  {i+1}. ({px}, {py})")
    print(f"\nVisualization saved to: {output_file}")
    print("  Green circles = sub-node centers")
    print("  Red squares = sample points")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Example: visualize empty→empty East connection, top-left sub-nodes
    if len(sys.argv) >= 6:
        tile_a = int(sys.argv[1])
        tile_b = int(sys.argv[2])
        direction = sys.argv[3]
        src_sub_x = int(sys.argv[4])
        src_sub_y = int(sys.argv[5])
        dst_sub_x = int(sys.argv[6]) if len(sys.argv) > 6 else 0
        dst_sub_y = int(sys.argv[7]) if len(sys.argv) > 7 else 0
        
        visualize_sample_points_for_connection(
            tile_a, tile_b, direction,
            src_sub_x, src_sub_y,
            dst_sub_x, dst_sub_y
        )
    else:
        print("Usage: python visualize_sample_points.py TILE_A TILE_B DIRECTION SRC_SUB_X SRC_SUB_Y [DST_SUB_X] [DST_SUB_Y]")
        print("\nExample: python visualize_sample_points.py 0 0 E 1 0 0 0")
        print("  (visualize empty→empty East, from right-top to left-top sub-nodes)")
        
        # Run a default example
        print("\nRunning default example: Tile 0→0, East direction, TR→TL sub-nodes\n")
        visualize_sample_points_for_connection(0, 0, 'E', 1, 0, 0, 0, "sample_points_example.png")

