"""
Runtime loader for precomputed tile connectivity data.

Provides singleton access to the precomputed tile traversability lookup table
with O(1) query performance (~10-20 nanoseconds per lookup).
"""

import os
import pickle
import gzip
from typing import Dict, List


class TileConnectivityLoader:
    """
    Loads and provides fast access to precomputed tile connectivity data.

    This class manages the precomputed lookup table and provides
    O(1) traversability queries for tile pairs.

    Singleton pattern ensures only one copy of the connectivity table
    is loaded in memory, shared across all environments.
    """

    _instance = None
    _connectivity_table = None

    # Direction name to index mapping
    DIR_TO_IDX = {"N": 0, "NE": 1, "E": 2, "SE": 3, "S": 4, "SW": 5, "W": 6, "NW": 7}

    # Direction index to vector mapping
    IDX_TO_VEC = {
        0: (0, -1),
        1: (1, -1),
        2: (1, 0),
        3: (1, 1),
        4: (0, 1),
        5: (-1, 1),
        6: (-1, 0),
        7: (-1, -1),
    }

    def __new__(cls):
        """Singleton pattern for shared connectivity table."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize loader (only once due to singleton)."""
        if self._connectivity_table is None:
            self._load_connectivity_table()

    def _load_connectivity_table(self):
        """Load precomputed connectivity table from file."""
        # Find data file
        data_path = os.path.join(
            os.path.dirname(__file__), "../../data/tile_connectivity.pkl.gz"
        )

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Precomputed tile connectivity not found at {data_path}. "
                f"Please run tile_connectivity_precomputer.py first to generate it."
            )

        # Load compressed data
        with gzip.open(data_path, "rb") as f:
            self._connectivity_table = pickle.load(f)

        print(
            f"[TileConnectivityLoader] Loaded connectivity table: "
            f"shape {self._connectivity_table.shape}, "
            f"size {self._connectivity_table.nbytes / 1024:.2f} KB"
        )

    def is_traversable(
        self,
        tile_a: int,
        tile_b: int,
        direction: str,
        src_sub_x: int = 0,
        src_sub_y: int = 0,
        dst_sub_x: int = 0,
        dst_sub_y: int = 0,
    ) -> bool:
        """
        Check if ninja can traverse from specific sub-node in tile_a to specific sub-node in tile_b.

        Args:
            tile_a: Source tile type (0-33)
            tile_b: Target tile type (0-33)
            direction: Direction name ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')
            src_sub_x: Source sub-node X index (0 or 1), defaults to 0
            src_sub_y: Source sub-node Y index (0 or 1), defaults to 0
            dst_sub_x: Dest sub-node X index (0 or 1), defaults to 0
            dst_sub_y: Dest sub-node Y index (0 or 1), defaults to 0

        Returns:
            True if traversable, False otherwise

        Performance: O(1) lookup, ~10-20 nanoseconds
        """
        if direction not in self.DIR_TO_IDX:
            raise ValueError(
                f"Invalid direction: {direction}. "
                f"Must be one of {list(self.DIR_TO_IDX.keys())}"
            )

        # Boundary check
        max_tile = self._connectivity_table.shape[0]
        if not (0 <= tile_a < max_tile and 0 <= tile_b < max_tile):
            return False  # Out of range tile types

        dir_idx = self.DIR_TO_IDX[direction]

        # NOTE: The precomputed table is 3D (tile-level only): [tile_a, tile_b, dir_idx]
        # Sub-node parameters are ignored since fast_graph_builder handles sub-node
        # validity checks directly using _check_subnode_validity_simple()
        return bool(self._connectivity_table[tile_a, tile_b, dir_idx])

    def get_traversable_neighbors(
        self, tile_type: int, neighbor_tiles: Dict[str, int]
    ) -> List[str]:
        """
        Get list of traversable directions from tile to its neighbors.

        Args:
            tile_type: Center tile type (0-33)
            neighbor_tiles: Dict mapping direction -> tile type
                Example: {'N': 0, 'E': 1, 'S': 0, 'W': 0}

        Returns:
            List of traversable direction names
        """
        traversable = []

        for direction, neighbor_type in neighbor_tiles.items():
            if self.is_traversable(tile_type, neighbor_type, direction):
                traversable.append(direction)

        return traversable

    def get_direction_vector(self, direction: str) -> tuple:
        """
        Get (dx, dy) vector for direction name.

        Args:
            direction: Direction name ('N', 'NE', 'E', etc.)

        Returns:
            (dx, dy) tuple
        """
        if direction not in self.DIR_TO_IDX:
            raise ValueError(f"Invalid direction: {direction}")

        dir_idx = self.DIR_TO_IDX[direction]
        return self.IDX_TO_VEC[dir_idx]

    @property
    def table_shape(self) -> tuple:
        """Get shape of connectivity table."""
        return (
            self._connectivity_table.shape
            if self._connectivity_table is not None
            else None
        )

    @property
    def table_size_kb(self) -> float:
        """Get size of connectivity table in KB."""
        if self._connectivity_table is not None:
            return self._connectivity_table.nbytes / 1024
        return 0.0
