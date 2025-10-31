"""
Precomputes tile-to-tile traversability for ninja-sized circles.

This module generates the tile_connectivity.pkl.gz file that contains:
- For each (src_tile_type, dst_tile_type) pair
- For each of the 4 sub-nodes in src_tile  
- Which of the 4 sub-nodes in dst_tile are reachable

This precomputation allows fast graph building without repeatedly checking
if ninja-sized circles can traverse between specific tile positions.

NOTE: This module now imports the centralized _check_subnode_validity_simple()
function from fast_graph_builder.py to ensure 100% consistency between
precomputation and runtime graph building. There is no duplicate logic.
"""

import gzip
import pickle
from pathlib import Path
from typing import Dict, Tuple, Set
import numpy as np

# Import centralized sub-node validity checker and constants
# This ensures consistency between precomputation and runtime graph building
from .fast_graph_builder import (
    _check_subnode_validity_simple,
    SUB_NODE_OFFSETS,
    NINJA_RADIUS
)

# Hardcoded cell size as per N++ constants
CELL_SIZE = 24


class TileConnectivityPrecomputer:
    """
    Precompute tile-to-tile traversability for all tile type combinations.

    This offline tool generates a comprehensive lookup table that encodes
    whether a ninja-sized circle can traverse between any two adjacent tiles
    in each of the 8 cardinal/diagonal directions.

    Key optimizations:
    - Only tiles 0-33 are computed (34-37 are glitched and unused)
    - Uses actual N++ physics for collision detection
    - Results cached to ~1.5-3 KB compressed file
    """

    # Only valid tiles, not glitched (34-37)
    VALID_TILES = 34

    def __init__(self):
        """Initialize precomputer with N++ physics constants."""
        self.ninja_radius = NINJA_RADIUS

        # 8-connectivity directions
        self.directions = {
            "N": (0, -1),
            "NE": (1, -1),
            "E": (1, 0),
            "SE": (1, 1),
            "S": (0, 1),
            "SW": (-1, 1),
            "W": (-1, 0),
            "NW": (-1, -1),
        }

    def can_ninja_traverse(
        self,
        src_tile_type: int,
        src_sub_x: int,
        src_sub_y: int,
        dst_tile_type: int,
        dst_sub_x: int,
        dst_sub_y: int,
        direction: str,
    ) -> bool:
        """
        Check if ninja can traverse from src sub-node to dst sub-node.

        Uses circle-based collision detection matching N++ physics.

        Args:
            src_tile_type: Source tile type (0-33)
            src_sub_x: Source sub-node x index (0 or 1)
            src_sub_y: Source sub-node y index (0 or 1)
            dst_tile_type: Destination tile type (0-33)
            dst_sub_x: Destination sub-node x index (0 or 1)
            dst_sub_y: Destination sub-node y index (0 or 1)
            direction: Movement direction ("N", "NE", etc.)

        Returns:
            True if ninja can traverse this path, False otherwise
        """
        # First check: both sub-nodes must be in non-solid areas
        src_pixel_x, src_pixel_y = SUB_NODE_OFFSETS[src_sub_y * 2 + src_sub_x]
        dst_pixel_x, dst_pixel_y = SUB_NODE_OFFSETS[dst_sub_y * 2 + dst_sub_x]

        src_valid = _check_subnode_validity_simple(
            src_tile_type, src_pixel_x, src_pixel_y
        )
        dst_valid = _check_subnode_validity_simple(
            dst_tile_type, dst_pixel_x, dst_pixel_y
        )

        if not (src_valid and dst_valid):
            return False

        # Second check: path between centers must be clear for ninja radius
        # Get absolute pixel positions
        dx, dy = self.directions[direction]
        src_abs_x = src_pixel_x
        src_abs_y = src_pixel_y
        dst_abs_x = dst_pixel_x + dx * CELL_SIZE
        dst_abs_y = dst_pixel_y + dy * CELL_SIZE

        # Check if circle can move along path
        # For simplicity, we sample points along the path
        num_samples = max(abs(dst_abs_x - src_abs_x), abs(dst_abs_y - src_abs_y)) // 2
        num_samples = max(num_samples, 5)  # At least 5 samples

        for i in range(num_samples + 1):
            t = i / num_samples
            check_x = src_abs_x + t * (dst_abs_x - src_abs_x)
            check_y = src_abs_y + t * (dst_abs_y - src_abs_y)

            # Check circle at this position against both tiles
            if not self._check_circle_clear(
                check_x, check_y, src_tile_type, dst_tile_type, dx, dy
            ):
                return False

        return True

    def _check_circle_clear(
        self,
        center_x: float,
        center_y: float,
        src_tile_type: int,
        dst_tile_type: int,
        dx: int,
        dy: int,
    ) -> bool:
        """
        Check if ninja circle at position is clear of solid tile areas.

        Samples points around circle perimeter and checks against tile geometry.

        Args:
            center_x: Circle center X in absolute pixels
            center_y: Circle center Y in absolute pixels
            src_tile_type: Source tile type
            dst_tile_type: Destination tile type
            dx: Tile offset X (-1, 0, or 1)
            dy: Tile offset Y (-1, 0, or 1)

        Returns:
            True if circle is clear, False if it intersects solid areas
        """
        # Sample 8 points around circle perimeter
        angles = [i * np.pi / 4 for i in range(8)]

        for angle in angles:
            # Point on circle perimeter
            px = center_x + self.ninja_radius * np.cos(angle)
            py = center_y + self.ninja_radius * np.sin(angle)

            # Check which tile this point is in
            if px < 0 or py < 0:
                # Point in source tile (negative means went backward)
                tile_x = int(px // CELL_SIZE)
                tile_y = int(py // CELL_SIZE)
                local_x = int(px - tile_x * CELL_SIZE)
                local_y = int(py - tile_y * CELL_SIZE)

                # Clamp to valid range
                local_x = max(0, min(CELL_SIZE - 1, local_x))
                local_y = max(0, min(CELL_SIZE - 1, local_y))

                if tile_x == 0 and tile_y == 0:
                    tile_type = src_tile_type
                elif tile_x == dx and tile_y == dy:
                    tile_type = dst_tile_type
                else:
                    # Point is in intermediate tile, assume solid for safety
                    tile_type = 1

            elif px >= CELL_SIZE or py >= CELL_SIZE:
                # Point in destination tile
                tile_x = int(px // CELL_SIZE)
                tile_y = int(py // CELL_SIZE)
                local_x = int(px - tile_x * CELL_SIZE)
                local_y = int(py - tile_y * CELL_SIZE)

                # Clamp to valid range
                local_x = max(0, min(CELL_SIZE - 1, local_x))
                local_y = max(0, min(CELL_SIZE - 1, local_y))

                if tile_x == 0 and tile_y == 0:
                    tile_type = src_tile_type
                elif tile_x == dx and tile_y == dy:
                    tile_type = dst_tile_type
                else:
                    # Point is in intermediate tile, assume solid for safety
                    tile_type = 1

            else:
                # Point still in source tile
                local_x = int(px)
                local_y = int(py)
                local_x = max(0, min(CELL_SIZE - 1, local_x))
                local_y = max(0, min(CELL_SIZE - 1, local_y))
                tile_type = src_tile_type

            # Check if this point is in a solid area
            is_traversable = _check_subnode_validity_simple(
                tile_type, local_x, local_y
            )
            if not is_traversable:
                return False

        return True

    def precompute_all_connections(self) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
        """
        Precompute tile-to-tile traversability for all tile type pairs.

        Returns:
            Dictionary mapping (src_tile_type, dst_tile_type) to set of
            traversable (src_subnode_idx, dst_subnode_idx) pairs
        """
        connections = {}
        total_combinations = self.VALID_TILES * self.VALID_TILES * 8
        processed = 0

        print(f"Precomputing tile connectivity for {self.VALID_TILES} tile types...")
        print(f"Total combinations to check: {total_combinations}")

        for src_tile_type in range(self.VALID_TILES):
            for dst_tile_type in range(self.VALID_TILES):
                for direction in self.directions.keys():
                    key = (src_tile_type, dst_tile_type, direction)
                    traversable_pairs = set()

                    # Check all 4x4 = 16 sub-node combinations
                    for src_sub_idx in range(4):
                        src_sub_x = src_sub_idx % 2
                        src_sub_y = src_sub_idx // 2

                        for dst_sub_idx in range(4):
                            dst_sub_x = dst_sub_idx % 2
                            dst_sub_y = dst_sub_idx // 2

                            if self.can_ninja_traverse(
                                src_tile_type,
                                src_sub_x,
                                src_sub_y,
                                dst_tile_type,
                                dst_sub_x,
                                dst_sub_y,
                                direction,
                            ):
                                traversable_pairs.add((src_sub_idx, dst_sub_idx))

                    connections[key] = traversable_pairs
                    processed += 1

                    if processed % 1000 == 0:
                        progress = 100 * processed / total_combinations
                        print(f"  Progress: {progress:.1f}% ({processed}/{total_combinations})")

        print("Precomputation complete!")
        return connections

    def save_to_file(self, connections: Dict, output_path: Path):
        """
        Save precomputed connections to compressed pickle file.

        Args:
            connections: Precomputed connectivity data
            output_path: Path to output file
        """
        with gzip.open(output_path, "wb") as f:
            pickle.dump(connections, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Print statistics
        file_size = output_path.stat().st_size
        num_entries = len(connections)
        traversable_count = sum(len(v) for v in connections.values())

        print(f"\nSaved to: {output_path}")
        print(f"File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
        print(f"Number of (src, dst, direction) combinations: {num_entries:,}")
        print(f"Total traversable sub-node pairs: {traversable_count:,}")
        print(f"Average traversable pairs per combination: {traversable_count/num_entries:.1f}")


def main():
    """Run the precomputation and save results."""
    precomputer = TileConnectivityPrecomputer()
    connections = precomputer.precompute_all_connections()

    # Save to data directory
    script_dir = Path(__file__).parent
    output_path = script_dir / "../../data/tile_connectivity.pkl.gz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    precomputer.save_to_file(connections, output_path)
    print("\n✅ Tile connectivity precomputation complete!")


if __name__ == "__main__":
    main()

