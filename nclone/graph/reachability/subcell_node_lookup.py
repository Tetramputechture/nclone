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

# Border extension for out-of-bounds entity queries
# Entities can be positioned up to ±2 tiles (48px) outside normal bounds
LOOKUP_BORDER_TILES = 2  # 2 tiles = 48px border on all sides
LOOKUP_BORDER_PX = LOOKUP_BORDER_TILES * CELL_SIZE  # 48 pixels

# Subcell grid dimensions (original, for reference)
SUBCELL_WIDTH = MAP_WIDTH_PX // SUB_NODE_SIZE  # 1008 / 12 = 84
SUBCELL_HEIGHT = MAP_HEIGHT_PX // SUB_NODE_SIZE  # 552 / 12 = 46

# Extended subcell grid dimensions (including border)
# Extended bounds: [-48, 1056) x [-48, 600) in pixel coordinates
EXTENDED_MAP_WIDTH_PX = MAP_WIDTH_PX + 2 * LOOKUP_BORDER_PX  # 1008 + 96 = 1104 pixels
EXTENDED_MAP_HEIGHT_PX = MAP_HEIGHT_PX + 2 * LOOKUP_BORDER_PX  # 552 + 96 = 648 pixels
SUBCELL_WIDTH_EXTENDED = EXTENDED_MAP_WIDTH_PX // SUB_NODE_SIZE  # 1104 / 12 = 92
SUBCELL_HEIGHT_EXTENDED = EXTENDED_MAP_HEIGHT_PX // SUB_NODE_SIZE  # 648 / 12 = 54

# Coordinate offset for converting query positions to extended grid indices
# Query position (x, y) maps to subcell index: ((x - LOOKUP_X_OFFSET - 6) // 12, (y - LOOKUP_Y_OFFSET - 6) // 12)
LOOKUP_X_OFFSET = -LOOKUP_BORDER_PX  # -48 pixels
LOOKUP_Y_OFFSET = -LOOKUP_BORDER_PX  # -48 pixels


class SubcellNodeLookupPrecomputer:
    """
    Precompute closest node position for each 12px×12px subcell.

    Uses the fast grid snapping algorithm from _find_closest_node() to
    determine which node position (on the fixed 12px grid) is closest
    to each subcell center.

    Node positions are fixed at: tile*24 + {6, 18} for each dimension.
    """

    def __init__(self):
        """Initialize precomputer with extended map dimensions."""
        self.subcell_width = SUBCELL_WIDTH_EXTENDED
        self.subcell_height = SUBCELL_HEIGHT_EXTENDED
        self.subcell_size = SUB_NODE_SIZE
        self.x_offset = LOOKUP_X_OFFSET
        self.y_offset = LOOKUP_Y_OFFSET

    def precompute_all(self, verbose: bool = False) -> np.ndarray:
        """
        Precompute closest node position for all subcells in extended region.

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
            print(
                f"  Extended map dimensions: {EXTENDED_MAP_WIDTH_PX}×{EXTENDED_MAP_HEIGHT_PX} pixels"
            )
            print(f"  Original map dimensions: {MAP_WIDTH_PX}×{MAP_HEIGHT_PX} pixels")
            print(
                f"  Border extension: ±{LOOKUP_BORDER_PX}px ({LOOKUP_BORDER_TILES} tiles) on all sides"
            )
            print()

        # Initialize lookup table: (subcell_x, subcell_y) -> (node_x, node_y)
        # Shape: (subcell_width, subcell_height, 2) for x,y coordinates
        lookup = np.zeros((self.subcell_width, self.subcell_height, 2), dtype=np.int32)

        # Fill with invalid marker initially
        lookup.fill(-1)

        for i in range(self.subcell_width):
            for j in range(self.subcell_height):
                # Convert subcell index to pixel coordinates in extended space
                # Subcell index i=0 corresponds to x = LOOKUP_X_OFFSET + 6
                # Subcell center at (6,6) alignment relative to extended grid origin
                subcell_center_x = self.x_offset + 6 + i * self.subcell_size
                subcell_center_y = self.y_offset + 6 + j * self.subcell_size

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
            print(
                f"  Extended bounds cover: x=[{LOOKUP_X_OFFSET}, {LOOKUP_X_OFFSET + EXTENDED_MAP_WIDTH_PX}), "
                f"y=[{LOOKUP_Y_OFFSET}, {LOOKUP_Y_OFFSET + EXTENDED_MAP_HEIGHT_PX})"
            )

        return lookup

    def save_to_file(self, lookup: np.ndarray, filepath: str, verbose: bool = False):
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

    def _is_node_grounded(
        self,
        node_pos: Tuple[int, int],
        physics_cache: Optional[Dict[Tuple[int, int], Dict[str, bool]]] = None,
        base_adjacency: Optional[Dict[Tuple[int, int], list]] = None,
    ) -> bool:
        """
        Check if a node is grounded (has solid surface directly below).

        IMPORTANT: Uses physics_cache or base_adjacency (tile geometry only)
        to ensure grounding is determined by level tiles, NOT entity positions.

        Args:
            node_pos: Node position (x, y) in pixels
            physics_cache: Optional pre-computed physics properties for O(1) lookup
            base_adjacency: Optional base adjacency (pre-entity-mask) for grounding checks

        Returns:
            True if node is grounded, False otherwise

        Raises:
            ValueError: If neither physics_cache nor base_adjacency is provided
        """
        # Prefer physics_cache for O(1) lookup
        if physics_cache is not None and node_pos in physics_cache:
            return physics_cache[node_pos]["grounded"]

        # Fallback to base_adjacency traversal
        if base_adjacency is not None:
            x, y = node_pos
            below_pos = (x, y + 12)  # 12px down

            # If node directly below doesn't exist, this node is on solid surface
            if below_pos not in base_adjacency:
                return True

            # Check if there's a direct downward edge
            if node_pos in base_adjacency:
                neighbors = base_adjacency[node_pos]
                for neighbor_info in neighbors:
                    # Handle both tuple format (neighbor_pos, cost) and just neighbor_pos
                    if isinstance(neighbor_info, tuple) and len(neighbor_info) >= 2:
                        neighbor_pos = neighbor_info[0]
                    else:
                        neighbor_pos = neighbor_info

                    if neighbor_pos == below_pos:
                        return False  # Can fall down, not grounded

            return True  # No downward edge, grounded

        # Error: neither source provided
        raise ValueError(
            "_is_node_grounded requires either physics_cache or base_adjacency. "
            "This ensures grounding is determined by tile geometry only, not mines/doors."
        )

    def find_closest_node_position(
        self,
        query_x: float,
        query_y: float,
        adjacency: Dict[Tuple[int, int], list],
        max_radius: float = 12.0,
        prefer_grounded: bool = False,
        physics_cache: Optional[Dict[Tuple[int, int], Dict[str, bool]]] = None,
        base_adjacency: Optional[Dict[Tuple[int, int], list]] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Find closest node position using precomputed lookup table.

        Uses adjacency keys as a runtime mask to filter precomputed positions.
        Checks neighboring subcells if primary candidate doesn't exist.
        Handles out-of-bounds queries by clamping to extended lookup table bounds.

        IMPORTANT: When prefer_grounded=True, uses physics_cache or base_adjacency
        to ensure grounding is determined by tile geometry only, not entity positions.

        Args:
            query_x: Query X coordinate in pixels (tile data space)
            query_y: Query Y coordinate in pixels (tile data space)
            adjacency: Graph adjacency structure (keys are node positions, for candidate filtering)
            max_radius: Maximum search radius in pixels (default 12px for entity radius)
            prefer_grounded: If True, prioritize grounded nodes over air nodes
            physics_cache: Optional pre-computed physics properties for grounding checks
            base_adjacency: Optional base adjacency (pre-entity-mask) for grounding checks

        Returns:
            Closest node position (x, y) if found, None otherwise
        """
        if self._lookup_table is None:
            raise RuntimeError("Lookup table not loaded")

        # Convert to subcell index (accounting for offset and (6,6) center alignment)
        # Extended grid: subcell index = ((query_x - LOOKUP_X_OFFSET - 6) // 12)
        subcell_x = int((query_x - LOOKUP_X_OFFSET - 6) // SUB_NODE_SIZE)
        subcell_y = int((query_y - LOOKUP_Y_OFFSET - 6) // SUB_NODE_SIZE)

        # Clamp to valid extended bounds instead of raising exception
        # This handles entities positioned up to ±2 tiles outside normal map bounds
        subcell_x = max(0, min(subcell_x, self._lookup_table.shape[0] - 1))
        subcell_y = max(0, min(subcell_y, self._lookup_table.shape[1] - 1))

        # Collect candidates for grounded filtering
        candidates = []

        # Get primary candidate from lookup table
        candidate_x = int(self._lookup_table[subcell_x, subcell_y, 0])
        candidate_y = int(self._lookup_table[subcell_x, subcell_y, 1])

        # Check if candidate exists in adjacency (runtime mask)
        if candidate_x >= 0 and candidate_y >= 0:
            candidate = (candidate_x, candidate_y)
            if candidate in adjacency:
                if not prefer_grounded:
                    return candidate  # Return immediately if not preferring grounded
                candidates.append(candidate)

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

                    # Bounds check (clamp to extended bounds)
                    neighbor_x = max(
                        0, min(neighbor_x, self._lookup_table.shape[0] - 1)
                    )
                    neighbor_y = max(
                        0, min(neighbor_y, self._lookup_table.shape[1] - 1)
                    )

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
                                if not prefer_grounded:
                                    return candidate  # Return first found if not preferring
                                candidates.append(candidate)

        # If preferring grounded, filter candidates
        if prefer_grounded and candidates:
            # Separate grounded and air candidates
            grounded = []
            air_nodes = []

            for candidate in candidates:
                if self._is_node_grounded(candidate, physics_cache, base_adjacency):
                    grounded.append(candidate)
                else:
                    air_nodes.append(candidate)

            # Prefer grounded, fall back to air
            if grounded:
                # Return closest grounded node
                best = min(
                    grounded,
                    key=lambda c: (c[0] - query_x) ** 2 + (c[1] - query_y) ** 2,
                )
                return best
            elif air_nodes:
                # Return closest air node
                best = min(
                    air_nodes,
                    key=lambda c: (c[0] - query_x) ** 2 + (c[1] - query_y) ** 2,
                )
                return best

        # No candidates found
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
    lookup = precomputer.precompute_all(verbose=False)

    # Save to data file
    output_path = os.path.join(
        os.path.dirname(__file__), "../../data/subcell_node_lookup.pkl.gz"
    )
    precomputer.save_to_file(lookup, output_path, verbose=False)

    print()
    print("=" * 70)
    print("Precomputation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
