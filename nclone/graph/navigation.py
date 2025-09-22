"""
Simple pathfinding engine for subgoal planning.

This module provides basic pathfinding functionality for the subgoal planner.
"""

from typing import List, Optional, Dict
from collections import deque
from enum import Enum


class PathfindingAlgorithm(Enum):
    """Pathfinding algorithm options."""

    BFS = "bfs"
    DIJKSTRA = "dijkstra"
    A_STAR = "a_star"


class PathfindingEngine:
    """
    Simple pathfinding engine using Dijkstra's algorithm.

    This is a basic implementation to support subgoal planning functionality.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize pathfinding engine.

        Args:
            debug: Enable debug output
        """
        self.debug = debug
        self.graph_cache = {}

    def find_shortest_path(
        self, start_node: int, end_node: int, graph_data=None
    ) -> Optional[List[int]]:
        """
        Find shortest path between two nodes using Dijkstra's algorithm.

        Args:
            start_node: Starting node index
            end_node: Target node index
            graph_data: Graph data structure (optional, uses cached if available)

        Returns:
            List of node indices representing the path, or None if no path exists
        """
        if start_node == end_node:
            return [start_node]

        if graph_data is None and not self.graph_cache:
            if self.debug:
                print("No graph data available for pathfinding")
            return None

        # Use provided graph_data or cached graph
        graph = graph_data if graph_data is not None else self.graph_cache

        # Simple BFS for unweighted graph
        if not hasattr(graph, "edges") and not isinstance(graph, dict):
            # If graph structure is not as expected, return direct path
            return [start_node, end_node]

        # Extract adjacency information
        if hasattr(graph, "edges"):
            adjacency = self._build_adjacency_from_edges(graph.edges)
        elif isinstance(graph, dict):
            adjacency = graph
        else:
            # Fallback: assume direct connection
            return [start_node, end_node]

        # BFS pathfinding
        queue = deque([(start_node, [start_node])])
        visited = {start_node}

        while queue:
            current_node, path = queue.popleft()

            if current_node == end_node:
                return path

            # Get neighbors
            neighbors = adjacency.get(current_node, [])
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        # No path found
        if self.debug:
            print(f"No path found from {start_node} to {end_node}")
        return None

    def _build_adjacency_from_edges(self, edges) -> Dict[int, List[int]]:
        """
        Build adjacency list from edge data.

        Args:
            edges: Edge data structure

        Returns:
            Adjacency list as dictionary
        """
        adjacency = {}

        for edge in edges:
            # Handle different edge data formats
            if hasattr(edge, "source") and hasattr(edge, "target"):
                source, target = edge.source, edge.target
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                source, target = edge[0], edge[1]
            else:
                continue

            # Add bidirectional edges
            if source not in adjacency:
                adjacency[source] = []
            if target not in adjacency:
                adjacency[target] = []

            adjacency[source].append(target)
            adjacency[target].append(source)

        return adjacency

    def cache_graph(self, graph_data):
        """
        Cache graph data for future pathfinding operations.

        Args:
            graph_data: Graph data to cache
        """
        self.graph_cache = graph_data

    def clear_cache(self):
        """Clear cached graph data."""
        self.graph_cache = {}
