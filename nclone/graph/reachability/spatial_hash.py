"""
Spatial hash grid for fast O(1) node lookup.

Provides efficient spatial queries for finding closest nodes to positions,
replacing O(N) linear search with O(1) grid-based lookup.
"""

from typing import Dict, List, Tuple, Optional, Set
import math


class SpatialHash:
    """
    2D spatial hash grid for fast nearest-neighbor queries.

    Nodes are hashed into grid cells for O(1) lookup. Finding the closest
    node requires checking only the local cell and immediate neighbors (9 cells).

    Performance:
    - Build: O(N) where N is number of nodes
    - Query: O(1) in practice (checks at most 9 cells with few nodes each)
    - Memory: O(N) for node storage + O(N) for grid cells
    """

    def __init__(self, cell_size: float = 24.0):
        """
        Initialize spatial hash grid.

        Args:
            cell_size: Size of grid cells in pixels (default 24px = 1 tile)
        """
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self.nodes: Set[Tuple[int, int]] = set()

    def _hash_position(self, x: float, y: float) -> Tuple[int, int]:
        """
        Hash a position to a grid cell.

        Args:
            x: X coordinate in pixels
            y: Y coordinate in pixels

        Returns:
            Grid cell coordinates (cell_x, cell_y)
        """
        cell_x = int(math.floor(x / self.cell_size))
        cell_y = int(math.floor(y / self.cell_size))
        return (cell_x, cell_y)

    def build(self, node_positions: List[Tuple[int, int]]):
        """
        Build spatial hash from list of node positions.

        Args:
            node_positions: List of (x, y) node positions in pixels
        """
        self.grid.clear()
        self.nodes = set(node_positions)

        for pos in node_positions:
            cell = self._hash_position(pos[0], pos[1])
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append(pos)

    def find_closest(
        self, query_x: float, query_y: float, max_distance: float = 50.0
    ) -> Optional[Tuple[int, int]]:
        """
        Find closest node to query position within max_distance.

        Checks the query cell and all 8 neighboring cells (9 cells total).
        This guarantees finding the closest node if one exists within range.

        Args:
            query_x: Query X coordinate in pixels
            query_y: Query Y coordinate in pixels
            max_distance: Maximum search distance in pixels

        Returns:
            Closest node position, or None if no node within max_distance
        """
        if not self.grid:
            return None

        # Get query cell
        query_cell = self._hash_position(query_x, query_y)
        cell_x, cell_y = query_cell

        # Check query cell and all 8 neighbors (3x3 grid)
        best_node = None
        best_dist_sq = max_distance * max_distance

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_cell = (cell_x + dx, cell_y + dy)
                nodes_in_cell = self.grid.get(check_cell, [])

                for node_pos in nodes_in_cell:
                    # Calculate squared distance (avoid sqrt until needed)
                    dist_sq = (node_pos[0] - query_x) ** 2 + (
                        node_pos[1] - query_y
                    ) ** 2

                    if dist_sq < best_dist_sq:
                        best_dist_sq = dist_sq
                        best_node = node_pos

        return best_node

    def find_all_within_radius(
        self, query_x: float, query_y: float, radius: float
    ) -> List[Tuple[Tuple[int, int], float]]:
        """
        Find all nodes within radius of query position.

        Returns nodes sorted by distance (closest first).

        Args:
            query_x: Query X coordinate in pixels
            query_y: Query Y coordinate in pixels
            radius: Search radius in pixels

        Returns:
            List of (node_pos, distance) tuples sorted by distance
        """
        if not self.grid:
            return []

        query_cell = self._hash_position(query_x, query_y)
        cell_x, cell_y = query_cell

        # Determine how many cells to check based on radius
        cells_to_check = int(math.ceil(radius / self.cell_size)) + 1

        nodes_within_radius = []
        radius_sq = radius * radius

        for dx in range(-cells_to_check, cells_to_check + 1):
            for dy in range(-cells_to_check, cells_to_check + 1):
                check_cell = (cell_x + dx, cell_y + dy)
                nodes_in_cell = self.grid.get(check_cell, [])

                for node_pos in nodes_in_cell:
                    dist_sq = (node_pos[0] - query_x) ** 2 + (
                        node_pos[1] - query_y
                    ) ** 2

                    if dist_sq <= radius_sq:
                        distance = math.sqrt(dist_sq)
                        nodes_within_radius.append((node_pos, distance))

        # Sort by distance (closest first)
        nodes_within_radius.sort(key=lambda x: x[1])
        return nodes_within_radius

    def clear(self):
        """Clear the spatial hash."""
        self.grid.clear()
        self.nodes.clear()

    def get_statistics(self) -> Dict[str, float]:
        """
        Get spatial hash statistics for debugging/profiling.

        Returns:
            Dictionary with statistics about the spatial hash
        """
        if not self.grid:
            return {
                "total_nodes": 0,
                "total_cells": 0,
                "avg_nodes_per_cell": 0.0,
                "max_nodes_per_cell": 0,
                "empty_cells": 0,
            }

        nodes_per_cell = [len(nodes) for nodes in self.grid.values()]

        return {
            "total_nodes": len(self.nodes),
            "total_cells": len(self.grid),
            "avg_nodes_per_cell": sum(nodes_per_cell) / len(nodes_per_cell),
            "max_nodes_per_cell": max(nodes_per_cell),
            "min_nodes_per_cell": min(nodes_per_cell),
        }
