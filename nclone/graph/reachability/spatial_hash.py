"""
Spatial hash grid for fast O(1) node lookup.

Provides efficient spatial queries for finding closest nodes to positions,
replacing O(N) linear search with O(1) grid-based lookup.
"""

from operator import itemgetter
from typing import Dict, List, Tuple, Optional, Set
import math

# Module-level sort keys: `operator.itemgetter` runs in C, where the equivalent
# lambda costs a Python frame per element on a per-step hot path.
_first = itemgetter(0)
_second = itemgetter(1)


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

    def query(
        self, query_x: float, query_y: float, radius: float
    ) -> List[Tuple[int, int]]:
        """Return node positions within radius, closest first.

        Same result as ``[pos for pos, _ in find_all_within_radius(...)]`` but it
        never takes a square root: ordering by squared distance is the same
        ordering (sqrt is strictly increasing) and both sorts are stable, so ties
        resolve identically. This is the per-step raycast path
        (``find_platform_in_direction`` calls it once per 6px step), so the sqrt
        and the (pos, distance) tuples it built were pure waste.
        """
        grid = self.grid
        if not grid:
            return []

        cell_size = self.cell_size
        cell_x = int(math.floor(query_x / cell_size))
        cell_y = int(math.floor(query_y / cell_size))
        # `ceil(radius / cell_size)` rings is EXACT, not an approximation: two
        # points at most `radius` apart differ by at most that many cells on each
        # axis, so a further ring can hold nothing the radius test would keep.
        # The old `+ 1` scanned 25 cells where 9 suffice, and this is the inner
        # loop of the 500px/6px platform raycast.
        cells_to_check = int(math.ceil(radius / cell_size))
        radius_sq = radius * radius

        found = []
        append = found.append
        get = grid.get
        for cx in range(cell_x - cells_to_check, cell_x + cells_to_check + 1):
            for cy in range(cell_y - cells_to_check, cell_y + cells_to_check + 1):
                for node_pos in get((cx, cy), ()):
                    ddx = node_pos[0] - query_x
                    ddy = node_pos[1] - query_y
                    dist_sq = ddx * ddx + ddy * ddy
                    if dist_sq <= radius_sq:
                        append((dist_sq, node_pos))

        found.sort(key=_first)
        return [pos for _, pos in found]

    def nearest_member_within(
        self, query_x: float, query_y: float, radius: float, membership
    ) -> Optional[Tuple[int, int]]:
        """Nearest node STRICTLY within ``radius`` that is also in ``membership``.

        Equivalent to scanning ``query(...)`` and keeping the first candidate that
        is in ``membership`` and strictly closer than ``radius``, which is what
        the platform raycast does -- but without materialising or sorting the
        candidate list. The results match exactly: ``query``'s sort is stable over
        this same cell walk, so among equal distances the first element of the
        sorted list is the first one this walk sees.

        ``membership`` is anything supporting ``in`` (the caller passes the graph
        adjacency dict).
        """
        grid = self.grid
        if not grid:
            return None

        cell_size = self.cell_size
        cell_x = int(math.floor(query_x / cell_size))
        cell_y = int(math.floor(query_y / cell_size))
        cells_to_check = int(math.ceil(radius / cell_size))

        best = None
        best_dist_sq = radius * radius
        get = grid.get
        for cx in range(cell_x - cells_to_check, cell_x + cells_to_check + 1):
            for cy in range(cell_y - cells_to_check, cell_y + cells_to_check + 1):
                for node_pos in get((cx, cy), ()):
                    ddx = node_pos[0] - query_x
                    ddy = node_pos[1] - query_y
                    dist_sq = ddx * ddx + ddy * ddy
                    if dist_sq < best_dist_sq and node_pos in membership:
                        best_dist_sq = dist_sq
                        best = node_pos
        return best

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
        cells_to_check = int(math.ceil(radius / self.cell_size))  # exact; see query()

        nodes_within_radius = []
        append = nodes_within_radius.append
        radius_sq = radius * radius
        get = self.grid.get
        sqrt = math.sqrt

        for dx in range(-cells_to_check, cells_to_check + 1):
            for dy in range(-cells_to_check, cells_to_check + 1):
                for node_pos in get((cell_x + dx, cell_y + dy), ()):
                    ddx = node_pos[0] - query_x
                    ddy = node_pos[1] - query_y
                    dist_sq = ddx * ddx + ddy * ddy

                    if dist_sq <= radius_sq:
                        append((node_pos, sqrt(dist_sq)))

        # Sort by distance (closest first)
        nodes_within_radius.sort(key=_second)
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
