"""
Comprehensive tile type validation and visualization tool.

This script:
1. Renders all 38 tile types showing solid vs traversable areas
2. Overlays sub-node positions with validity markers
3. Shows diagonal/circular segments from tile definitions
4. Compares fast_graph_builder vs tile_connectivity_precomputer implementations
5. Exports detailed PNG visualization for manual inspection

Usage:
    python -m nclone.tools.comprehensive_tile_validator
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Tuple, List

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import both implementations
from nclone.graph.reachability.fast_graph_builder import (
    _check_subnode_validity_simple as fast_graph_check,
    SUB_NODE_OFFSETS,
    SUB_NODE_COORDS,
)
from nclone.graph.reachability.tile_connectivity_precomputer import (
    _check_subnode_validity_simple as precomputer_check,
)
from nclone.tile_definitions import TILE_SEGMENT_DIAG_MAP, TILE_SEGMENT_CIRCULAR_MAP


# Tile type names for labels
TILE_NAMES = {
    0: "Empty", 1: "Solid",
    2: "Half Top", 3: "Half Right", 4: "Half Bottom", 5: "Half Left",
    6: "Slope \\", 7: "Slope /", 8: "Slope / (inv)", 9: "Slope \\ (inv)",
    10: "Quarter Circle BR", 11: "Quarter Circle BL",
    12: "Quarter Circle TL", 13: "Quarter Circle TR",
    14: "Quarter Pipe TL", 15: "Quarter Pipe TR",
    16: "Quarter Pipe BR", 17: "Quarter Pipe BL",
    18: "Mild Slope Up-L", 19: "Mild Slope Up-R",
    20: "Mild Slope Dn-R", 21: "Mild Slope Dn-L",
    22: "Raised Mild L-Up", 23: "Raised Mild R-Up",
    24: "Raised Mild R-Dn", 25: "Raised Mild L-Dn",
    26: "Steep Slope Up-L", 27: "Steep Slope Up-R",
    28: "Steep Slope Dn-R", 29: "Steep Slope Dn-L",
    30: "Raised Steep L-Up", 31: "Raised Steep R-Up",
    32: "Raised Steep R-Dn", 33: "Raised Steep L-Dn",
    34: "Glitched 34", 35: "Glitched 35",
    36: "Glitched 36", 37: "Glitched 37",
}


def analyze_tile_implementation(tile_type: int) -> Dict:
    """
    Analyze a tile type's geometry implementation.
    
    Returns dict with:
    - sub_node_validity_fast: list of 4 bools from fast_graph_builder
    - sub_node_validity_precomp: list of 4 bools from precomputer
    - consistency: bool - whether implementations agree
    - solid_pixel_count_fast: int - number of solid pixels per fast_graph
    - solid_pixel_count_precomp: int - number of solid pixels per precomputer
    """
    sub_node_validity_fast = []
    sub_node_validity_precomp = []
    
    # Check all 4 sub-nodes
    for (offset_x, offset_y) in SUB_NODE_OFFSETS:
        fast_valid = fast_graph_check(tile_type, offset_x, offset_y)
        precomp_valid = precomputer_check(tile_type, offset_x, offset_y)
        sub_node_validity_fast.append(fast_valid)
        sub_node_validity_precomp.append(precomp_valid)
    
    # Count solid pixels for each implementation
    solid_fast = 0
    solid_precomp = 0
    for y in range(24):
        for x in range(24):
            if not fast_graph_check(tile_type, x, y):
                solid_fast += 1
            if not precomputer_check(tile_type, x, y):
                solid_precomp += 1
    
    consistency = (sub_node_validity_fast == sub_node_validity_precomp)
    
    return {
        'sub_node_validity_fast': sub_node_validity_fast,
        'sub_node_validity_precomp': sub_node_validity_precomp,
        'consistency': consistency,
        'solid_pixel_count_fast': solid_fast,
        'solid_pixel_count_precomp': solid_precomp,
        'geometry_consistent': solid_fast == solid_precomp,
    }


def render_tile_comparison(tile_type: int, scale: int = 20) -> Image.Image:
    """
    Render a tile type with both implementations side-by-side.
    
    Shows:
    - Left: fast_graph_builder implementation
    - Right: tile_connectivity_precomputer implementation
    - Sub-node positions with validity markers
    - Diagonal/circular segments from tile definitions
    """
    size = 24
    img_width = size * scale * 2 + scale * 2  # Two tiles + gap
    img_height = size * scale + scale * 6  # Tile + header + footer
    
    # Create image
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(scale * 0.7))
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(scale * 0.5))
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Header
    header_y = scale
    title = f"Type {tile_type}: {TILE_NAMES.get(tile_type, 'Unknown')}"
    draw.text((img_width // 2, header_y), title, fill='black', font=font, anchor='mm')
    
    # Sub-headers
    subheader_y = scale * 2.5
    draw.text((size * scale // 2, subheader_y), "fast_graph_builder", fill='blue', font=font_small, anchor='mm')
    draw.text((size * scale * 1.5 + scale * 2, subheader_y), "tile_connectivity_precomputer", fill='blue', font=font_small, anchor='mm')
    
    # Tile rendering offset
    tile_y_offset = scale * 3
    
    # Render both implementations
    for impl_idx, check_func in enumerate([fast_graph_check, precomputer_check]):
        x_offset = impl_idx * (size * scale + scale * 2)
        
        # Draw tile pixels
        for y in range(size):
            for x in range(size):
                is_solid = not check_func(tile_type, x, y)
                color = (80, 80, 80) if is_solid else (220, 220, 220)
                draw.rectangle(
                    [
                        x_offset + x * scale,
                        tile_y_offset + y * scale,
                        x_offset + (x + 1) * scale - 1,
                        tile_y_offset + (y + 1) * scale - 1
                    ],
                    fill=color,
                    outline=color
                )
        
        # Draw grid lines
        for i in range(size + 1):
            # Horizontal
            draw.line(
                [
                    (x_offset, tile_y_offset + i * scale),
                    (x_offset + size * scale, tile_y_offset + i * scale)
                ],
                fill=(150, 150, 150),
                width=1
            )
            # Vertical
            draw.line(
                [
                    (x_offset + i * scale, tile_y_offset),
                    (x_offset + i * scale, tile_y_offset + size * scale)
                ],
                fill=(150, 150, 150),
                width=1
            )
        
        # Draw 12-pixel boundaries (sub-node boundaries) thicker
        for i in [0, 12, 24]:
            # Horizontal
            draw.line(
                [
                    (x_offset, tile_y_offset + i * scale),
                    (x_offset + size * scale, tile_y_offset + i * scale)
                ],
                fill=(100, 100, 255),
                width=2
            )
            # Vertical
            draw.line(
                [
                    (x_offset + i * scale, tile_y_offset),
                    (x_offset + i * scale, tile_y_offset + size * scale)
                ],
                fill=(100, 100, 255),
                width=2
            )
        
        # Draw diagonal/circular segments
        if tile_type in TILE_SEGMENT_DIAG_MAP:
            p1, p2 = TILE_SEGMENT_DIAG_MAP[tile_type]
            draw.line(
                [
                    x_offset + p1[0] * scale,
                    tile_y_offset + p1[1] * scale,
                    x_offset + p2[0] * scale,
                    tile_y_offset + p2[1] * scale
                ],
                fill=(255, 0, 0),
                width=3
            )
        
        if tile_type in TILE_SEGMENT_CIRCULAR_MAP:
            center, quadrant, convex = TILE_SEGMENT_CIRCULAR_MAP[tile_type]
            # Draw circle arc
            circle_x = x_offset + center[0] * scale
            circle_y = tile_y_offset + center[1] * scale
            radius = 24 * scale
            draw.ellipse(
                [
                    circle_x - radius,
                    circle_y - radius,
                    circle_x + radius,
                    circle_y + radius
                ],
                outline=(255, 0, 255),
                width=2
            )
        
        # Draw sub-node positions
        for (offset_x, offset_y), (sub_x, sub_y) in zip(SUB_NODE_OFFSETS, SUB_NODE_COORDS):
            is_valid = check_func(tile_type, offset_x, offset_y)
            node_color = (0, 255, 0) if is_valid else (255, 0, 0)
            node_x = x_offset + offset_x * scale
            node_y = tile_y_offset + offset_y * scale
            
            # Draw circle
            radius = scale // 2
            draw.ellipse(
                [node_x - radius, node_y - radius, node_x + radius, node_y + radius],
                fill=node_color,
                outline='black',
                width=2
            )
            
            # Draw sub-node label
            label = f"({sub_x},{sub_y})"
            draw.text(
                (node_x, node_y),
                label,
                fill='white',
                font=font_small,
                anchor='mm'
            )
    
    # Footer with analysis
    analysis = analyze_tile_implementation(tile_type)
    footer_y = tile_y_offset + size * scale + scale
    
    if analysis['consistency']:
        footer_text = "✓ Implementations CONSISTENT"
        footer_color = (0, 150, 0)
    else:
        footer_text = "✗ Implementations DIFFER!"
        footer_color = (200, 0, 0)
    
    draw.text((img_width // 2, footer_y), footer_text, fill=footer_color, font=font, anchor='mm')
    
    # Detailed stats
    stats_y = footer_y + scale * 1.5
    stats_text = f"Solid pixels: Fast={analysis['solid_pixel_count_fast']}, Precomp={analysis['solid_pixel_count_precomp']}"
    draw.text((img_width // 2, stats_y), stats_text, fill='black', font=font_small, anchor='mm')
    
    return img


def create_comprehensive_grid(output_path: str = "debug_output/tile_validation_grid.png"):
    """Create a comprehensive grid showing all tile types."""
    print("=" * 70)
    print("COMPREHENSIVE TILE VALIDATION")
    print("=" * 70)
    print()
    
    # Analyze all tiles
    inconsistencies = []
    for tile_type in range(38):
        analysis = analyze_tile_implementation(tile_type)
        if not analysis['consistency']:
            inconsistencies.append((tile_type, analysis))
            print(f"⚠️  Type {tile_type:2d} ({TILE_NAMES.get(tile_type, 'Unknown')}): INCONSISTENT")
            print(f"    Fast sub-nodes:     {analysis['sub_node_validity_fast']}")
            print(f"    Precomp sub-nodes:  {analysis['sub_node_validity_precomp']}")
            print(f"    Solid pixels:       Fast={analysis['solid_pixel_count_fast']}, Precomp={analysis['solid_pixel_count_precomp']}")
            print()
    
    if not inconsistencies:
        print("✅ All tile types have consistent implementations!")
    else:
        print(f"\n❌ Found {len(inconsistencies)} inconsistent tile types")
    
    print()
    print("=" * 70)
    print("GENERATING VISUALIZATION")
    print("=" * 70)
    print()
    
    # Create grid: 6 tiles per row, 7 rows (38 tiles + 4 empty)
    cols = 6
    rows = 7
    scale = 15
    
    # Calculate grid dimensions
    tile_img_width = 24 * scale * 2 + scale * 2
    tile_img_height = 24 * scale + scale * 6
    
    grid_width = tile_img_width * cols
    grid_height = tile_img_height * rows
    
    # Create grid image
    grid = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Render each tile
    for tile_type in range(38):
        print(f"  Rendering tile type {tile_type}...", end='\r')
        
        row = tile_type // cols
        col = tile_type % cols
        
        tile_img = render_tile_comparison(tile_type, scale)
        
        # Paste into grid
        x_pos = col * tile_img_width
        y_pos = row * tile_img_height
        grid.paste(tile_img, (x_pos, y_pos))
    
    print(f"  Rendered all 38 tile types" + " " * 20)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grid.save(output_path)
    
    print(f"\n✅ Saved comprehensive grid to: {output_path}")
    print(f"   Image size: {grid_width}x{grid_height} pixels")
    print()
    
    return inconsistencies


def create_individual_tiles(output_dir: str = "debug_output/tiles"):
    """Create individual tile images for problematic types."""
    print("=" * 70)
    print("CREATING INDIVIDUAL TILE IMAGES")
    print("=" * 70)
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create images for all slope types (18-33) and problematic ones
    problem_types = list(range(18, 34)) + [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    
    for tile_type in problem_types:
        analysis = analyze_tile_implementation(tile_type)
        status = "INCONSISTENT" if not analysis['consistency'] else "consistent"
        
        print(f"  Type {tile_type:2d} ({TILE_NAMES.get(tile_type, 'Unknown')}): {status}")
        
        img = render_tile_comparison(tile_type, scale=20)
        output_path = os.path.join(output_dir, f"tile_type_{tile_type:02d}.png")
        img.save(output_path)
    
    print(f"\n✅ Saved {len(problem_types)} individual tile images to: {output_dir}")
    print()


def main():
    """Run comprehensive tile validation."""
    print("\n" + "=" * 70)
    print("TILE TYPE GEOMETRIC VALIDATION TOOL")
    print("=" * 70)
    print()
    print("This tool compares the two implementations of _check_subnode_validity_simple()")
    print("and identifies geometric calculation bugs.")
    print()
    
    # Create comprehensive grid
    inconsistencies = create_comprehensive_grid()
    
    # Create individual images for problematic types
    create_individual_tiles()
    
    # Summary
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print()
    
    if inconsistencies:
        print(f"❌ Found {len(inconsistencies)} tile types with inconsistent implementations:")
        for tile_type, analysis in inconsistencies:
            print(f"   - Type {tile_type:2d}: {TILE_NAMES.get(tile_type, 'Unknown')}")
        print()
        print("These tile types need to be fixed!")
        return 1
    else:
        print("✅ All tile types have consistent implementations!")
        return 0


if __name__ == "__main__":
    sys.exit(main())

