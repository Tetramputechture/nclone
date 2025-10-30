"""
Path distance calculation using BFS and A* algorithms.

Calculates shortest navigable path distances on precomputed traversability graphs,
providing accurate distance metrics for reward shaping.
"""

import heapq
import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import deque

# Hardcoded cell size as per N++ constants
CELL_SIZE = 24


class PathDistanceCalculator:
    """
    Calculate shortest navigable path distances using BFS or A*.
    
    Operates on precomputed traversability graph for maximum performance.
    
    BFS: Guaranteed shortest path, explores uniformly
    A*: Faster with Manhattan heuristic, still optimal
    """
    
    def __init__(self, use_astar: bool = True):
        """
        Initialize path distance calculator.
        
        Args:
            use_astar: Use A* (True) or BFS (False) for pathfinding
        """
        self.use_astar = use_astar
    
    def calculate_distance(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]
    ) -> float:
        """
        Calculate shortest navigable path distance.
        
        Args:
            start: Starting position (x, y) in pixels
            goal: Goal position (x, y) in pixels  
            adjacency: Graph adjacency structure
        
        Returns:
            Shortest path distance in pixels, or float('inf') if unreachable
        """
        # Quick checks
        if start not in adjacency or goal not in adjacency:
            return float('inf')
        if start == goal:
            return 0.0
        
        # Choose algorithm
        if self.use_astar:
            return self._astar_distance(start, goal, adjacency)
        else:
            return self._bfs_distance(start, goal, adjacency)
    
    def _bfs_distance(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        adjacency: Dict
    ) -> float:
        """BFS pathfinding - guaranteed shortest path."""
        queue = deque([(start, 0.0)])
        visited = {start}
        
        while queue:
            current, dist = queue.popleft()
            
            if current == goal:
                return dist
            
            # Explore neighbors from adjacency graph
            for neighbor, edge_cost in adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + edge_cost))
        
        return float('inf')
    
    def _astar_distance(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        adjacency: Dict
    ) -> float:
        """A* pathfinding - faster than BFS with heuristic."""
        def manhattan_heuristic(pos: Tuple[int, int]) -> float:
            """Manhattan distance heuristic (admissible)."""
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        # Priority queue: (f_score, g_score, position)
        open_set = [(manhattan_heuristic(start), 0.0, start)]
        g_score = {start: 0.0}
        visited = set()
        
        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == goal:
                return current_g
            
            # Explore neighbors
            for neighbor, edge_cost in adjacency.get(current, []):
                if neighbor in visited:
                    continue
                
                tentative_g = current_g + edge_cost
                
                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + manhattan_heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        
        return float('inf')


class CachedPathDistanceCalculator:
    """Path distance calculator with caching for static goals."""
    
    def __init__(self, max_cache_size: int = 200, use_astar: bool = True):
        """
        Initialize cached path distance calculator.
        
        Args:
            max_cache_size: Maximum number of cached distance queries
            use_astar: Use A* (True) or BFS (False) for pathfinding
        """
        self.calculator = PathDistanceCalculator(use_astar=use_astar)
        self.cache: Dict[Tuple, float] = {}
        self.max_cache_size = max_cache_size
        self.hits = 0
        self.misses = 0
    
    def get_distance(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        adjacency: Dict,
        cache_key: Optional[str] = None
    ) -> float:
        """
        Get path distance with caching.
        
        Args:
            start: Start position
            goal: Goal position
            adjacency: Graph adjacency
            cache_key: Optional key for cache invalidation (e.g., entity type)
        
        Returns:
            Shortest path distance in pixels
        """
        # Snap to grid for cache consistency (24 pixel tiles)
        start_grid = (start[0] // 24, start[1] // 24)
        goal_grid = (goal[0] // 24, goal[1] // 24)
        key = (start_grid, goal_grid, cache_key)
        
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        
        # Cache miss - compute
        self.misses += 1
        distance = self.calculator.calculate_distance(start, goal, adjacency)
        
        # Store in cache (FIFO eviction)
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        
        self.cache[key] = distance
        return distance
    
    def clear_cache(self):
        """Clear cache (call on level change)."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_statistics(self) -> Dict[str, float]:
        """Get cache performance statistics."""
        total_queries = self.hits + self.misses
        hit_rate = self.hits / total_queries if total_queries > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_queries': total_queries,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }
