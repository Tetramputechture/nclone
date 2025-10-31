"""
ASCII-based tile visualization tool for debugging graph reachability.

Provides text-based rendering of tile types to quickly verify geometric checks
and sub-node positioning without requiring image generation.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import Tuple, Set, Dict
from nclone.graph.reachability.fast_graph_builder import (
    _check_subnode_validity_simple,
    SUB_NODE_OFFSETS,
    SUB_NODE_COORDS,
    SUBNODE_VALIDITY_TABLE,
    WITHIN_TILE_CONNECTIVITY,
)
from nclone.tile_definitions import (
    TILE_SEGMENT_DIAG_MAP,
    TILE_SEGMENT_CIRCULAR_MAP,
)


# Tile type names for reference
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


def render_tile_ascii(tile_type: int, width: int = 24, height: int = 24, 
                     show_legend: bool = True) -> str:
    """
    Render a tile type as ASCII art for debugging.
    
    Uses characters:
    - '█' for solid areas (dark grey in rendering)
    - '·' for non-solid areas (light grey/empty)
    - '1', '2', '3', '4' for the 4 sub-node positions
    - 'X' for sub-nodes that fail validity check (in solid)
    - 'O' for sub-nodes that pass validity check (in non-solid)
    
    Args:
        tile_type: Tile type ID (0-37)
        width: Width in pixels (default 24)
        height: Height in pixels (default 24)
        show_legend: Whether to show legend explaining characters
    
    Returns:
        Multi-line string representing the tile in ASCII art
    """
    # Create 2D grid
    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            # Check if this position is solid
            is_solid = not _check_subnode_validity_simple(tile_type, x, y)
            row.append('█' if is_solid else '·')
        grid.append(row)
    
    # Mark sub-node positions
    sub_node_markers = []
    for (offset_x, offset_y), (sub_x, sub_y) in zip(SUB_NODE_OFFSETS, SUB_NODE_COORDS):
        is_valid = SUBNODE_VALIDITY_TABLE.get(tile_type, {}).get((sub_x, sub_y), False)
        marker = 'O' if is_valid else 'X'
        
        # Find which sub-node number this is (1-4)
        node_num = sub_x + sub_y * 2 + 1
        
        if 0 <= offset_y < height and 0 <= offset_x < width:
            grid[offset_y][offset_x] = str(node_num) if is_valid else 'X'
            sub_node_markers.append(f"  {node_num}: ({sub_x},{sub_y})@({offset_x},{offset_y}) = {marker}")
    
    # Convert grid to string
    lines = []
    tile_name = TILE_NAMES.get(tile_type, f"Unknown {tile_type}")
    lines.append(f"Tile Type {tile_type}: {tile_name}")
    lines.append("┌" + "─" * width + "┐")
    for row in grid:
        lines.append("│" + "".join(row) + "│")
    lines.append("└" + "─" * width + "┘")
    
    # Add sub-node legend
    lines.append("Sub-nodes:")
    lines.extend(sub_node_markers)
    
    if show_legend:
        lines.append("")
        lines.append("Legend: █=solid  ·=empty  1-4=valid sub-nodes  X=invalid sub-node")
    
    return "\n".join(lines)


def print_tile_connectivity_matrix(tile_type: int) -> None:
    """
    Print which sub-nodes can reach each other within a tile.
    
    Shows ASCII rendering and a connectivity matrix indicating which
    sub-nodes can traverse to each other within the same tile.
    """
    print(render_tile_ascii(tile_type, show_legend=False))
    print()
    
    # Get valid sub-nodes
    valid_subs = []
    for (offset_x, offset_y), (sub_x, sub_y) in zip(SUB_NODE_OFFSETS, SUB_NODE_COORDS):
        is_valid = SUBNODE_VALIDITY_TABLE.get(tile_type, {}).get((sub_x, sub_y), False)
        if is_valid:
            valid_subs.append((sub_x, sub_y))
    
    if not valid_subs:
        print("No valid sub-nodes in this tile (all solid)")
        return
    
    print("Within-tile connectivity:")
    print("(Using precomputed WITHIN_TILE_CONNECTIVITY lookup table)")
    print()
    
    # Get connectivity for this tile
    connectivity = WITHIN_TILE_CONNECTIVITY.get(tile_type, {})
    
    # Print header
    print("      ", end="")
    for sub in valid_subs:
        print(f" {sub}  ", end="")
    print()
    
    # Print matrix
    for src_sub in valid_subs:
        print(f"{src_sub} ", end="")
        reachable = connectivity.get(src_sub, set())
        for dst_sub in valid_subs:
            if src_sub == dst_sub:
                print("  -   ", end="")
            elif dst_sub in reachable:
                print("  ✓   ", end="")
            else:
                print("  ✗   ", end="")
        print()
    print()
    print("✓ = Can reach   ✗ = Cannot reach (blocked by solid geometry)")


def print_all_tiles_summary(tile_range: Tuple[int, int] = (0, 34)) -> None:
    """
    Print ASCII rendering of all tile types in a compact grid.
    
    Args:
        tile_range: Tuple of (start, end) tile types to render (default 0-33)
    """
    start, end = tile_range
    
    print(f"=== Tile Types {start} to {end-1} ===")
    print()
    
    for tile_type in range(start, end):
        print(render_tile_ascii(tile_type, show_legend=False))
        print()


def test_tile_pair_ascii(tile_a: int, tile_b: int, direction: str) -> None:
    """
    Test and print traversability between two tile types in ASCII.
    
    Shows side-by-side ASCII rendering with connectivity information.
    
    Args:
        tile_a: Source tile type
        tile_b: Destination tile type
        direction: Direction ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')
    """
    from nclone.graph.reachability.tile_connectivity_loader import TileConnectivityLoader
    
    loader = TileConnectivityLoader()
    
    print(f"=== Testing Tile Pair: {tile_a} → {tile_b} ({direction}) ===")
    print()
    
    # Render both tiles
    tile_a_lines = render_tile_ascii(tile_a, show_legend=False).split('\n')
    tile_b_lines = render_tile_ascii(tile_b, show_legend=False).split('\n')
    
    # Print side by side with arrow
    max_lines = max(len(tile_a_lines), len(tile_b_lines))
    arrow = f"  {direction}→  "
    
    for i in range(max_lines):
        line_a = tile_a_lines[i] if i < len(tile_a_lines) else " " * 28
        line_b = tile_b_lines[i] if i < len(tile_b_lines) else " " * 28
        
        if i == max_lines // 2:
            print(f"{line_a}{arrow}{line_b}")
        else:
            print(f"{line_a}      {line_b}")
    
    print()
    
    # Check precomputed connectivity
    try:
        is_traversable = loader.is_traversable(tile_a, tile_b, direction)
        result = "✓ TRAVERSABLE" if is_traversable else "✗ BLOCKED"
        print(f"Precomputed connectivity result: {result}")
    except Exception as e:
        print(f"Error checking connectivity: {e}")
    
    print()


def main():
    """Main function for interactive testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ASCII Tile Renderer for Debugging')
    parser.add_argument('--tile', type=int, help='Render specific tile type')
    parser.add_argument('--all', action='store_true', help='Render all tile types')
    parser.add_argument('--pair', nargs=3, metavar=('TILE_A', 'TILE_B', 'DIR'),
                       help='Test tile pair traversability')
    parser.add_argument('--matrix', type=int, help='Show connectivity matrix for tile type')
    
    args = parser.parse_args()
    
    if args.tile is not None:
        print(render_tile_ascii(args.tile))
    elif args.all:
        print_all_tiles_summary()
    elif args.pair:
        tile_a, tile_b, direction = int(args.pair[0]), int(args.pair[1]), args.pair[2]
        test_tile_pair_ascii(tile_a, tile_b, direction)
    elif args.matrix is not None:
        print_tile_connectivity_matrix(args.matrix)
    else:
        # Default: show a few example tiles
        print("ASCII Tile Renderer - Examples")
        print("=" * 60)
        print()
        for tile_type in [0, 1, 2, 6, 10, 14]:
            print(render_tile_ascii(tile_type, show_legend=False))
            print()
        print("Use --help for more options")


if __name__ == "__main__":
    main()

