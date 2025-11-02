"""
Detailed geometric analysis of tile types with rendering verification.

This script analyzes each problematic tile type by:
1. Computing what the visual renderer draws (solid polygon)
2. Determining correct traversability logic
3. Comparing against both implementations
4. Generating the correct fix
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nclone.tile_definitions import TILE_SEGMENT_DIAG_MAP


def get_rendered_polygon(tile_type: int):
    """
    Get the solid polygon points for a tile type based on shared_tile_renderer.py logic.

    Returns list of (x, y) points defining the solid region.
    """
    tile_size = 24

    if tile_type == 24:
        # From renderer: dx1=0, dy1=12, dx2=24, dy2=0, dx3=24, dy3=24, dx4=0, dy4=24
        return [(0, 12), (24, 0), (24, 24), (0, 24)]

    elif tile_type == 25:
        # dx1=0, dy1=0, dx2=24, dy2=12, dx3=24, dy3=0, dx4=0, dy4=24
        # Wait, need to recalculate...
        # For tile_type == 25:
        # dx1 = 0
        # dy1 = 0  # tile_size/2 if (23 or 24), else 0 -> 0
        # dx2 = tile_size = 24  # 0 if type 23, else tile_size -> 24
        # dy2 = tile_size/2 = 12  # tile_size/2 if tile_type == 25 -> 12
        # dx3 = tile_size = 24
        # dy3 = 0 if tile_type < 24 else tile_size  # tile_type=25, so tile_size = 24
        # dx4 = 0  # tile_size if type 23, else 0 -> 0
        # dy4 = tile_size = 24
        return [(0, 0), (24, 12), (24, 24), (0, 24)]

    elif tile_type == 27:
        # Triangular tiles with midpoint (types 26-29)
        # dx1 = tile_size/2 = 12
        # dy1 = 0  # tile_size if (28 or 29), else 0 -> 0
        # dx2 = tile_size = 24  # tile_size if (27 or 28) -> 24
        # dy2 = 0
        # dx3 = tile_size = 24  # tile_size if (27 or 28) -> 24
        # dy3 = tile_size = 24
        return [(12, 0), (24, 0), (24, 24)]

    elif tile_type == 29:
        # dx1 = tile_size/2 = 12
        # dy1 = tile_size = 24  # tile_size if (28 or 29) -> 24
        # dx2 = 0  # tile_size if (27 or 28), else 0 -> 0
        # dy2 = 0
        # dx3 = 0  # tile_size if (27 or 28), else 0 -> 0
        # dy3 = tile_size = 24
        return [(12, 24), (0, 0), (0, 24)]

    elif tile_type == 30:
        # Complex quadrilateral tiles (types 30-33)
        # dx1 = tile_size/2 = 12
        # dy1 = tile_size = 24  # tile_size if (30 or 31) -> 24
        # dx2 = 0  # tile_size if (31 or 33), else 0 -> 0
        # dy2 = tile_size = 24
        # dx3 = tile_size = 24  # tile_size if (31 or 32) -> 24
        # dy3 = 0  # tile_size if (32 or 33), else 0 -> 0
        # dx4 = tile_size = 24  # tile_size if (30 or 32) -> 24
        # dy4 = 0
        return [
            (12, 24),
            (0, 24),
            (24, 0),
            (24, 0),
        ]  # Note: duplicate point, should be: [(12,24), (0,24), (24,0)]

    elif tile_type == 31:
        # dx1 = tile_size/2 = 12
        # dy1 = tile_size = 24  # tile_size if (30 or 31) -> 24
        # dx2 = tile_size = 24  # tile_size if (31 or 33) -> 24
        # dy2 = tile_size = 24
        # dx3 = 0  # tile_size if (31 or 32), else 0 -> 0
        # dy3 = 0  # tile_size if (32 or 33), else 0 -> 0
        # dx4 = 0  # tile_size if (30 or 32), else 0 -> 0
        # dy4 = 0
        return [(12, 24), (24, 24), (0, 0), (0, 0)]  # Note: duplicate

    elif tile_type == 32:
        # dx1 = 12, dy1 = 0  # not (30 or 31) -> 0
        # dx2 = 0  # not (31 or 33) -> 0
        # dy2 = 24
        # dx3 = tile_size = 24  # tile_type 31 or 32 -> 24
        # dy3 = tile_size = 24  # tile_type 32 or 33 -> 24
        # dx4 = tile_size = 24  # tile_type 30 or 32 -> 24
        # dy4 = 0
        return [(12, 0), (0, 24), (24, 24), (24, 0)]

    elif tile_type == 33:
        # dx1 = 12, dy1 = 0
        # dx2 = tile_size = 24  # tile_type 31 or 33 -> 24
        # dy2 = 24
        # dx3 = 0  # not (31 or 32) -> 0
        # dy3 = tile_size = 24  # tile_type 32 or 33 -> 24
        # dx4 = 0  # not (30 or 32) -> 0
        # dy4 = 0
        return [(12, 0), (24, 24), (0, 24), (0, 0)]

    return []


def compute_correct_logic(tile_type: int):
    """
    Compute the correct traversability logic based on the rendered polygon.

    Returns a description of the correct formula.
    """
    polygon = get_rendered_polygon(tile_type)
    diag = TILE_SEGMENT_DIAG_MAP.get(tile_type)

    print(f"\n{'=' * 70}")
    print(f"TILE TYPE {tile_type}")
    print(f"{'=' * 70}")
    print(f"Rendered polygon (solid region): {polygon}")
    print(f"Diagonal from TILE_SEGMENT_DIAG_MAP: {diag}")
    print()

    # For each tile type, determine the traversable region
    if tile_type == 24:
        # Polygon: [(0,12), (24,0), (24,24), (0,24)]
        # Solid quad, traversable is the triangle above the line from (0,12) to (24,0)
        # Line: y = 12 - x*12/24
        # Traversable: y < 12 - x*12/24, or y <= 12 - x*12/24 (approximately)
        print("Analysis:")
        print("  Solid region: Quadrilateral with top edge from (0,12) to (24,0)")
        print("  Traversable region: Triangle above that line (smaller y values)")
        print("  Line equation: y = 12 - x*12/24")
        print("  Correct formula: pixel_y < 12 - pixel_x*12/24")
        print("  Or with <=: pixel_y <= (12 - pixel_x * 12 / 24) - epsilon")
        print()
        print("CORRECT IMPLEMENTATION:")
        print("  return pixel_y < (12 - pixel_x * 12 / 24)")
        print("  OR: return pixel_y <= (12 - pixel_x * 12 / 24 - 1)  # conservative")
        return "pixel_y < (12 - pixel_x * 12 / 24)"

    elif tile_type == 27:
        # Polygon: [(12,0), (24,0), (24,24)] - triangle
        # Solid is this right triangle
        # Traversable is left side: x < 12 OR (x >= 12 AND above/left of diagonal)
        # Diagonal from TILE_SEGMENT_DIAG_MAP: ((12,0), (24,24))
        # Line from (12,0) to (24,24): y = (x-12)*24/12 = (x-12)*2
        # Traversable if: x < 12 OR y < (x-12)*2
        print("Analysis:")
        print("  Solid region: Right triangle with vertices (12,0), (24,0), (24,24)")
        print("  Traversable region: Left half (x<12) OR above diagonal")
        print("  Diagonal line: from (12,0) to (24,24), equation: y = (x-12)*2")
        print("  Correct formula: x < 12 OR y < (x-12)*2")
        print()
        print("CORRECT IMPLEMENTATION:")
        print("  return pixel_x < 12 or pixel_y < ((pixel_x - 12) * 24 / 12)")
        print("  Simplified: return pixel_x < 12 or pixel_y < ((pixel_x - 12) * 2)")
        return "pixel_x < 12 or pixel_y < ((pixel_x - 12) * 24 / 12)"

    elif tile_type == 29:
        # Polygon: [(12,24), (0,0), (0,24)] - triangle
        # Solid is this left triangle
        # Traversable is right side: x > 12 OR (x <= 12 AND above/right of diagonal)
        # Diagonal from TILE_SEGMENT_DIAG_MAP: ((12,24), (0,0))
        # Line from (12,24) to (0,0): slope = (0-24)/(0-12) = 24/12 = 2
        # Using point-slope from (12,24): y - 24 = 2(x - 12) => y = 2x
        # Traversable if: x > 12 OR y < 2x
        print("Analysis:")
        print("  Solid region: Left triangle with vertices (12,24), (0,0), (0,24)")
        print("  Traversable region: Right half (x>12) OR above diagonal")
        print("  Diagonal line: from (12,24) to (0,0), equation: y = 2x")
        print("  Correct formula: x > 12 OR y < 2x")
        print()
        print("CORRECT IMPLEMENTATION:")
        print("  return pixel_x > 12 or pixel_y < (pixel_x * 24 / 12)")
        print("  Simplified: return pixel_x > 12 or pixel_y < (pixel_x * 2)")
        return "pixel_x > 12 or pixel_y < (pixel_x * 24 / 12)"

    elif tile_type == 30:
        # Need to recalculate polygon more carefully from renderer
        # Diagonal from TILE_SEGMENT_DIAG_MAP: ((12,24), (24,0))
        print("Analysis:")
        print("  From TILE_SEGMENT_DIAG_MAP: diagonal from (12,24) to (24,0)")
        print("  This is a raised steep slope")
        print("  Need to determine solid/traversable regions from visual rendering")
        return "TBD - need visual verification"

    elif tile_type == 31:
        # Diagonal from TILE_SEGMENT_DIAG_MAP: ((0,0), (12,24))
        print("Analysis:")
        print("  From TILE_SEGMENT_DIAG_MAP: diagonal from (0,0) to (12,24)")
        print("  This is a raised steep slope")
        return "TBD - need visual verification"

    elif tile_type == 32:
        # Polygon: [(12,0), (0,24), (24,24), (24,0)]
        # Diagonal from TILE_SEGMENT_DIAG_MAP: ((12,0), (0,24))
        print("Analysis:")
        print("  Solid region: Quadrilateral")
        print("  From TILE_SEGMENT_DIAG_MAP: diagonal from (12,0) to (0,24)")
        return "TBD - need visual verification"

    elif tile_type == 33:
        # Polygon: [(12,0), (24,24), (0,24), (0,0)]
        # Diagonal from TILE_SEGMENT_DIAG_MAP: ((24,24), (12,0))
        print("Analysis:")
        print("  Solid region: Quadrilateral")
        print("  From TILE_SEGMENT_DIAG_MAP: diagonal from (24,24) to (12,0)")
        return "TBD - need visual verification"

    return "Unknown"


def main():
    """Analyze all problematic tile types."""
    print("\n" + "=" * 70)
    print("GEOMETRIC ANALYSIS OF PROBLEMATIC TILE TYPES")
    print("=" * 70)

    problematic_types = [24, 27, 29, 30, 31, 32, 33]

    for tile_type in problematic_types:
        compute_correct_logic(tile_type)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Verify these formulas against the visual renderer output")
    print("2. Fix both graph_builder.py and tile_connectivity_precomputer.py")
    print("3. Regenerate tile_connectivity.pkl.gz")
    print()


if __name__ == "__main__":
    sys.exit(main())
