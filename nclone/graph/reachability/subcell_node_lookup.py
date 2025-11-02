"""
Precomputed subcell-to-node lookup table for O(1) node queries.

Precomputes which node position is closest to each 12px×12px subcell,
enabling true O(1) direct array access instead of spatial hash queries.

Since node positions are fixed on a 12px grid (tile*24 + {6, 18}),
we can precompute this once offline and reuse for all levels.
"""

import os
import pickle
import gzip
import numpy as np
from typing import Dict, Tuple, Optional

from ...constants.physics_constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT

# Hardcoded constants matching graph_builder.py
SUB_NODE_SIZE = 12  # 12px subcells
CELL_SIZE = 24  # 24px tiles

# Map dimensions in tile data space (playable area)
MAP_WIDTH_PX = MAP_TILE_WIDTH * CELL_SIZE  # 42 * 24 = 1008 pixels
MAP_HEIGHT_PX = MAP_TILE_HEIGHT * CELL_SIZE  # 23 * 24 = 552 pixels

# Subcell grid dimensions
SUBCELL_WIDTH = MAP_WIDTH_PX // SUB_NODE_SIZE  # 1008 / 12 = 84
SUBCELL_HEIGHT = MAP_HEIGHT_PX // SUB_NODE_SIZE  # 552 / 12 = 46


class SubcellNodeLookupPrecomputer:
    """
    Precompute closest node position for each 12px×12px subcell.

    Uses the fast grid snapping algorithm from _find_closest_node() to
    determine which node position (on the fixed 12px grid) is closest
    to each subcell center.

    Node positions are fixed at: tile*24 + {6, 18} for each dimension.
    """

    def __init__(self):
        """Initialize precomputer with map dimensions."""
        self.subcell_width = SUBCELL_WIDTH
        self.subcell_height = SUBCELL_HEIGHT
        self.subcell_size = SUB_NODE_SIZE

    def precompute_all(self, verbose: bool = True) -> np.ndarray:
        """
        Precompute closest node position for all subcells.

        Args:
            verbose: Print progress updates

        Returns:
            2D numpy array of shape (subcell_width, subcell_height, 2)
            Each entry is (node_x, node_y) or (-1, -1) for invalid positions
        """
        if verbose:
            total_subcells = self.subcell_width * self.subcell_height
            print(
                f"Precomputing closest node positions for {total_subcells} subcells..."
            )
            print(f"  Subcell grid: {self.subcell_width}×{self.subcell_height}")
            print(f"  Map dimensions: {MAP_WIDTH_PX}×{MAP_HEIGHT_PX} pixels")
            print()

        # Initialize lookup table: (subcell_x, subcell_y) -> (node_x, node_y)
        # Shape: (subcell_width, subcell_height, 2) for x,y coordinates
        lookup = np.zeros((self.subcell_width, self.subcell_height, 2), dtype=np.int32)

        # Fill with invalid marker initially
        lookup.fill(-1)

        for i in range(self.subcell_width):
            for j in range(self.subcell_height):
                # Subcell center at (6,6) alignment
                subcell_center_x = 6 + i * self.subcell_size
                subcell_center_y = 6 + j * self.subcell_size

                # Use fast grid snapping (same as _find_closest_node in graph_builder.py)
                # Nodes are at positions: ..., -18, -6, 6, 18, 30, 42, ...
                snap_x = round((subcell_center_x - 6) / 12) * 12 + 6
                snap_y = round((subcell_center_y - 6) / 12) * 12 + 6

                # Store node position
                lookup[i, j, 0] = snap_x
                lookup[i, j, 1] = snap_y

        if verbose:
            print(
                f"Precomputation complete: {self.subcell_width}×{self.subcell_height} subcells"
            )
            print("  Node positions computed using grid snapping algorithm")

        return lookup

    def save_to_file(self, lookup: np.ndarray, filepath: str, verbose: bool = True):
        """
        Save precomputed lookup table to compressed file.

        Args:
            lookup: Precomputed lookup array from precompute_all()
            filepath: Output file path
            verbose: Print save statistics
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save with compression
        with gzip.open(filepath, "wb") as f:
            pickle.dump(lookup, f, protocol=pickle.HIGHEST_PROTOCOL)

        if verbose:
            file_size = os.path.getsize(filepath)
            print()
            print(f"Saved subcell node lookup table to {filepath}")
            print(f"  Array shape: {lookup.shape}")
            print(f"  Compressed size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")

            # Statistics
            valid_count = np.sum(lookup[:, :, 0] >= 0)
            total = lookup.shape[0] * lookup.shape[1]
            print(
                f"  Valid entries: {valid_count:,}/{total:,} ({100 * valid_count / total:.1f}%)"
            )


class SubcellNodeLookupLoader:
    """
    Loads and provides fast access to precomputed subcell-to-node lookup table.

    Singleton pattern ensures only one copy of the lookup table is loaded
    in memory, shared across all environments.

    Provides O(1) direct array access to find closest node position for any
    query coordinate, with runtime masking using actual graph nodes.
    """

    _instance = None
    _lookup_table = None

    def __new__(cls):
        """Singleton pattern for shared lookup table."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize loader (only once due to singleton)."""
        if self._lookup_table is None:
            self._load_lookup_table()

    def _load_lookup_table(self):
        """Load precomputed lookup table from file."""
        # Find data file
        data_path = os.path.join(
            os.path.dirname(__file__), "../../data/subcell_node_lookup.pkl.gz"
        )

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Precomputed subcell node lookup not found at {data_path}. "
                f"Please run subcell_node_lookup.py first to generate it."
            )

        # Load compressed data
        with gzip.open(data_path, "rb") as f:
            self._lookup_table = pickle.load(f)

        print(
            "[SubcellNodeLookupLoader] Loaded lookup table: "
            f"shape {self._lookup_table.shape}, "
            f"size {self._lookup_table.nbytes / 1024:.2f} KB"
        )

    def find_closest_node_position(
        self,
        query_x: float,
        query_y: float,
        adjacency: Dict[Tuple[int, int], list],
        max_radius: float = 12.0,
    ) -> Optional[Tuple[int, int]]:
        """
        Find closest node position using precomputed lookup table.

        Uses adjacency keys as a runtime mask to filter precomputed positions.
        Checks neighboring subcells if primary candidate doesn't exist.

        Args:
            query_x: Query X coordinate in pixels (tile data space)
            query_y: Query Y coordinate in pixels (tile data space)
            adjacency: Graph adjacency structure (keys are node positions)
            max_radius: Maximum search radius in pixels (default 12px for entity radius)

        Returns:
            Closest node position (x, y) if found, None otherwise

        Raises:
            IndexError: If query position is out of bounds
        """
        if self._lookup_table is None:
            raise RuntimeError("Lookup table not loaded")

        # Convert to subcell index (accounting for (6,6) center alignment)
        subcell_x = int((query_x - 6) // SUB_NODE_SIZE)
        subcell_y = int((query_y - 6) // SUB_NODE_SIZE)

        # Bounds check - throw exception if out of bounds
        if not (
            0 <= subcell_x < self._lookup_table.shape[0]
            and 0 <= subcell_y < self._lookup_table.shape[1]
        ):
            raise IndexError(
                f"Query position ({query_x}, {query_y}) out of bounds. "
                f"Valid range: x=[0, {MAP_WIDTH_PX}), y=[0, {MAP_HEIGHT_PX})"
            )

        # Get primary candidate from lookup table
        candidate_x = int(self._lookup_table[subcell_x, subcell_y, 0])
        candidate_y = int(self._lookup_table[subcell_x, subcell_y, 1])

        # Check if candidate exists in adjacency (runtime mask)
        if candidate_x >= 0 and candidate_y >= 0:
            candidate = (candidate_x, candidate_y)
            if candidate in adjacency:
                return candidate

        # Check neighbors (up to max_radius away)
        # Convert max_radius to subcells: ceil(max_radius / 12)
        max_subcell_radius = int(np.ceil(max_radius / SUB_NODE_SIZE))

        # Check all subcells within radius (not just perimeter)
        # This ensures we find nodes even for entities at subcell boundaries
        for radius in range(1, max_subcell_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # Skip center (already checked)
                    if dx == 0 and dy == 0:
                        continue

                    neighbor_x = subcell_x + dx
                    neighbor_y = subcell_y + dy

                    # Bounds check
                    if not (
                        0 <= neighbor_x < self._lookup_table.shape[0]
                        and 0 <= neighbor_y < self._lookup_table.shape[1]
                    ):
                        continue

                    # Get candidate from neighbor subcell
                    candidate_x = int(self._lookup_table[neighbor_x, neighbor_y, 0])
                    candidate_y = int(self._lookup_table[neighbor_x, neighbor_y, 1])

                    if candidate_x >= 0 and candidate_y >= 0:
                        candidate = (candidate_x, candidate_y)
                        if candidate in adjacency:
                            # Verify distance is within threshold
                            dist_sq = (candidate_x - query_x) ** 2 + (
                                candidate_y - query_y
                            ) ** 2
                            if dist_sq <= max_radius * max_radius:
                                return candidate

        return None  # No node found within search radius

    @property
    def table_shape(self) -> tuple:
        """Get shape of lookup table."""
        return self._lookup_table.shape if self._lookup_table is not None else None

    @property
    def table_size_kb(self) -> float:
        """Get size of lookup table in KB."""
        if self._lookup_table is not None:
            return self._lookup_table.nbytes / 1024
        return 0.0


def main():
    """Run precomputation and save results."""
    print("=" * 70)
    print("N++ Subcell Node Lookup Precomputation")
    print("=" * 70)
    print()
    print("This offline tool precomputes closest node positions for all subcells.")
    print("This only needs to be run once to generate the lookup table.")
    print()

    precomputer = SubcellNodeLookupPrecomputer()

    # Run precomputation
    lookup = precomputer.precompute_all(verbose=True)

    # Save to data file
    output_path = os.path.join(
        os.path.dirname(__file__), "../../data/subcell_node_lookup.pkl.gz"
    )
    precomputer.save_to_file(lookup, output_path, verbose=True)

    print()
    print("=" * 70)
    print("Precomputation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
