"""
Image-based tile visualization tool for comprehensive analysis.

Creates PNG images showing all tile types with sub-node validity and geometry.
"""

import sys
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nclone.graph.reachability.fast_graph_builder import (
    _check_subnode_validity_simple,
    SUB_NODE_OFFSETS,
    SUB_NODE_COORDS,
    SUBNODE_VALIDITY_TABLE,
    WITHIN_TILE_CONNECTIVITY,
)
from nclone.tile_definitions import TILE_SEGMENT_DIAG_MAP, TILE_SEGMENT_CIRCULAR_MAP


# Tile type names
TILE_NAMES = {
    0: "Empty",
    1: "Solid",
    2: "Half Top",
    3: "Half Right",
    4: "Half Bottom",
    5: "Half Left",
    6: "Slope \\",
    7: "Slope /",
    8: "Slope / (inv)",
    9: "Slope \\ (inv)",
    10: "Quarter Circle BR",
    11: "Quarter Circle BL",
    12: "Quarter Circle TL",
    13: "Quarter Circle TR",
    14: "Quarter Pipe TL",
    15: "Quarter Pipe TR",
    16: "Quarter Pipe BR",
    17: "Quarter Pipe BL",
    18: "Mild Slope Up-L",
    19: "Mild Slope Up-R",
    20: "Mild Slope Dn-R",
    21: "Mild Slope Dn-L",
    22: "Raised Mild L-Up",
    23: "Raised Mild R-Up",
    24: "Raised Mild R-Dn",
    25: "Raised Mild L-Dn",
    26: "Steep Slope Up-L",
    27: "Steep Slope Up-R",
    28: "Steep Slope Dn-R",
    29: "Steep Slope Dn-L",
    30: "Raised Steep L-Up",
    31: "Raised Steep R-Up",
    32: "Raised Steep R-Dn",
    33: "Raised Steep L-Dn",
    34: "Glitched 34",
    35: "Glitched 35",
    36: "Glitched 36",
    37: "Glitched 37",
}


def render_tile_image(tile_type: int, scale: int = 10) -> Image.Image:
    """
    Render a tile type as a PIL Image for detailed visual analysis.

    Args:
        tile_type: Tile type ID (0-37)
        scale: Scaling factor for visualization (default 10 = 240x240 image)

    Returns:
        PIL Image showing the tile with sub-node positions and validity
    """
    size = 24
    img_size = size * scale

    # Create base image
    img = Image.new("RGB", (img_size, img_size), color="white")
    draw = ImageDraw.Draw(img)

    # Draw tile pixels
    for y in range(size):
        for x in range(size):
            is_solid = not _check_subnode_validity_simple(tile_type, x, y)
            color = (100, 100, 100) if is_solid else (220, 220, 220)
            draw.rectangle(
                [x * scale, y * scale, (x + 1) * scale - 1, (y + 1) * scale - 1],
                fill=color,
                outline=color,
            )

    # Draw grid lines
    for i in range(size + 1):
        draw.line(
            [(0, i * scale), (img_size, i * scale)], fill=(180, 180, 180), width=1
        )
        draw.line(
            [(i * scale, 0), (i * scale, img_size)], fill=(180, 180, 180), width=1
        )

    # Draw tile boundaries thicker
    draw.line([(0, 0), (img_size, 0)], fill=(0, 0, 0), width=2)
    draw.line([(0, 0), (0, img_size)], fill=(0, 0, 0), width=2)
    draw.line([(img_size - 1, 0), (img_size - 1, img_size)], fill=(0, 0, 0), width=2)
    draw.line([(0, img_size - 1), (img_size, img_size - 1)], fill=(0, 0, 0), width=2)

    # Draw diagonal/circular segments if present
    if tile_type in TILE_SEGMENT_DIAG_MAP:
        p1, p2 = TILE_SEGMENT_DIAG_MAP[tile_type]
        draw.line(
            [p1[0] * scale, p1[1] * scale, p2[0] * scale, p2[1] * scale],
            fill=(255, 0, 0),
            width=3,
        )

    if tile_type in TILE_SEGMENT_CIRCULAR_MAP:
        center, quadrant, is_convex = TILE_SEGMENT_CIRCULAR_MAP[tile_type]
        cx, cy = center[0] * scale, center[1] * scale
        radius = 24 * scale

        # Draw the full circle outline for reference
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline=(255, 0, 255),  # Magenta for circle outline
            width=2,
        )

        # Draw the actual arc segment
        # Calculate angles based on tile type (matching Cairo renderer logic)
        if tile_type < 14:
            # Quarter circles (10-13)
            a1_rad = (math.pi / 2) * (tile_type - 10)
            a2_rad = (math.pi / 2) * (tile_type - 9)
        else:
            # Quarter pipes (14-17)
            a1_rad = math.pi + (math.pi / 2) * (tile_type - 10)
            a2_rad = math.pi + (math.pi / 2) * (tile_type - 9)

        # Convert to degrees for PIL (PIL uses degrees, starting from 3 o'clock, counterclockwise)
        a1_deg = -math.degrees(a1_rad)  # Negative because PIL goes counterclockwise
        a2_deg = -math.degrees(a2_rad)

        # Draw the arc
        draw.arc(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            start=a1_deg,
            end=a2_deg,
            fill=(255, 0, 0),  # Red for the arc
            width=4,
        )

        # Draw center marker
        draw.ellipse(
            [cx - 3, cy - 3, cx + 3, cy + 3], fill=(255, 0, 0), outline=(255, 0, 0)
        )

    # Draw sub-node markers
    marker_size = scale // 2
    for (offset_x, offset_y), (sub_x, sub_y) in zip(SUB_NODE_OFFSETS, SUB_NODE_COORDS):
        is_valid = SUBNODE_VALIDITY_TABLE.get(tile_type, {}).get((sub_x, sub_y), False)

        px = offset_x * scale
        py = offset_y * scale

        # Draw marker
        if is_valid:
            # Green circle for valid sub-node
            draw.ellipse(
                [
                    px - marker_size,
                    py - marker_size,
                    px + marker_size,
                    py + marker_size,
                ],
                fill=(0, 255, 0),
                outline=(0, 100, 0),
                width=2,
            )
        else:
            # Red X for invalid sub-node
            draw.line(
                [
                    px - marker_size,
                    py - marker_size,
                    px + marker_size,
                    py + marker_size,
                ],
                fill=(255, 0, 0),
                width=3,
            )
            draw.line(
                [
                    px - marker_size,
                    py + marker_size,
                    px + marker_size,
                    py - marker_size,
                ],
                fill=(255, 0, 0),
                width=3,
            )

        # Label with sub-node number
        node_num = sub_x + sub_y * 2 + 1
        text = str(node_num)
        # Simple text positioning (PIL default font)
        draw.text((px + marker_size + 2, py - marker_size), text, fill=(0, 0, 255))

    return img


def create_tile_visualization_grid(
    tile_range=(0, 34), output_path="tile_visualization_grid.png"
):
    """
    Create a grid image showing all tile types.

    Args:
        tile_range: Tuple of (start, end) tile types
        output_path: Path to save output image
    """
    start, end = tile_range
    tiles_per_row = 6
    tile_scale = 8
    tile_size = 24 * tile_scale
    label_height = 30

    num_tiles = end - start
    num_rows = (num_tiles + tiles_per_row - 1) // tiles_per_row

    grid_width = tiles_per_row * tile_size
    grid_height = num_rows * (tile_size + label_height)

    grid_img = Image.new("RGB", (grid_width, grid_height), color="white")

    for i, tile_type in enumerate(range(start, end)):
        row = i // tiles_per_row
        col = i % tiles_per_row

        # Render tile
        tile_img = render_tile_image(tile_type, scale=tile_scale)

        # Paste into grid
        x = col * tile_size
        y = row * (tile_size + label_height)
        grid_img.paste(tile_img, (x, y))

        # Add label
        draw = ImageDraw.Draw(grid_img)
        label_text = f"{tile_type}: {TILE_NAMES.get(tile_type, 'Unknown')}"
        draw.text((x + 5, y + tile_size + 5), label_text, fill=(0, 0, 0))

    grid_img.save(output_path)
    print(f"Saved tile visualization grid to: {output_path}")
    return grid_img


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Image-based Tile Visualizer")
    parser.add_argument("--tile", type=int, help="Render specific tile type")
    parser.add_argument("--all", action="store_true", help="Create grid of all tiles")
    parser.add_argument(
        "--output-dir", default="./debug_output", help="Output directory"
    )
    parser.add_argument("--scale", type=int, default=10, help="Scale factor for images")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.tile is not None:
        # Render single tile
        img = render_tile_image(args.tile, scale=args.scale)
        output_path = os.path.join(args.output_dir, f"tile_type_{args.tile:02d}.png")
        img.save(output_path)
        print(f"Saved tile {args.tile} to: {output_path}")
    elif args.all:
        # Create grid of all tiles
        output_path = os.path.join(args.output_dir, "tile_visualization_grid.png")
        create_tile_visualization_grid(output_path=output_path)

        # Also save individual tiles
        for tile_type in range(34):
            img = render_tile_image(tile_type, scale=args.scale)
            output_path = os.path.join(
                args.output_dir, f"tile_type_{tile_type:02d}.png"
            )
            img.save(output_path)
        print(f"Saved all {34} individual tile images to: {args.output_dir}")
    else:
        # Default: create grid
        output_path = os.path.join(args.output_dir, "tile_visualization_grid.png")
        create_tile_visualization_grid(output_path=output_path)


if __name__ == "__main__":
    main()
