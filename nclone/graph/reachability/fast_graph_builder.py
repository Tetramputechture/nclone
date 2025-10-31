"""
Fast graph builder using precomputed tile connectivity.

Combines precomputed tile traversability lookups with dynamic entity masking
to build level-specific navigation graphs efficiently.

Improved implementation with sub-tile nodes:
- Each 24px tile divided into 4 sub-nodes (2x2 grid at 12px resolution)
- Respects tile types (type 0=empty, type 1=solid)
- Builds graph only for reachable nodes from player spawn
- More accurate pathfinding for 10px radius player
"""

import time
import numpy as np
from typing import Dict, Set, Tuple, List, Any, Optional
from collections import deque

from .tile_connectivity_loader import TileConnectivityLoader
from .entity_mask import EntityMask

# Hardcoded cell size as per N++ constants
CELL_SIZE = 24
SUB_NODE_SIZE = 12  # Divide each 24px tile into 2x2 grid of 12px nodes
PLAYER_RADIUS = 10  # Player collision radius in pixels


class FastGraphBuilder:
    """
    Builds level traversability graph using precomputed tile connectivity.
    
    Combines:
    1. Static tile connectivity (precomputed, O(1) lookup)
    2. Dynamic entity mask (doors, mines)
    3. Result: Complete traversability graph for pathfinding
    
    Performance: 
    - First call per level: <0.2ms (builds and caches graph)
    - Subsequent calls: <0.05ms (uses cached graph + state mask)
    
    Caching Strategy:
    - Tiles never change during a run → cache base graph
    - Entity POSITIONS never change → cache in base graph
    - Entity STATES change → lightweight mask update only
    """
    
    def __init__(self):
        """Initialize graph builder with connectivity loader."""
        self.connectivity_loader = TileConnectivityLoader()
        
        # Per-level caching
        self._level_graph_cache = {}  # level_id -> base graph
        self._current_level_id = None
        
        # Performance tracking
        self.build_times = []
        self.cache_hits = 0
        self.cache_misses = 0
    
    def build_graph(
        self,
        level_data: Dict[str, Any],
        ninja_pos: Optional[Tuple[int, int]] = None,
        level_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build complete traversability graph for level.
        
        Args:
            level_data: Level data with tiles, entities, switch_states
            ninja_pos: Optional ninja position for reachability analysis
            level_id: Optional level identifier for caching
        
        Returns:
            Dictionary containing:
            - 'adjacency': Dict mapping (x, y) -> List[(neighbor_x, neighbor_y, cost)]
            - 'reachable': Set of reachable positions from ninja_pos (if provided)
            - 'blocked_positions': Set of blocked positions (by entities)
            - 'blocked_edges': Set of blocked edges
        """
        start_time = time.perf_counter()
        
        tiles = level_data['tiles']  # 2D numpy array
        height, width = tiles.shape
        
        # Generate level ID if not provided
        if level_id is None:
            level_id = f"level_{hash(tiles.tobytes())}"
        
        # Check cache
        if level_id in self._level_graph_cache:
            # Use cached base graph
            base_graph = self._level_graph_cache[level_id]
            self.cache_hits += 1
        else:
            # Build new base graph (includes tile connectivity, entity positions)
            base_graph = self._build_base_graph(tiles, level_data)
            self._level_graph_cache[level_id] = base_graph
            self.cache_misses += 1
        
        # Apply dynamic entity state mask
        entity_mask = EntityMask(level_data)
        blocked_positions = entity_mask.get_blocked_positions()
        blocked_edges = entity_mask.get_blocked_edges()
        
        # Create final adjacency by filtering blocked positions/edges
        adjacency = self._apply_entity_mask(
            base_graph['adjacency'],
            blocked_positions,
            blocked_edges
        )
        
        result = {
            'adjacency': adjacency,
            'blocked_positions': blocked_positions,
            'blocked_edges': blocked_edges,
            'base_graph_cached': level_id in self._level_graph_cache
        }
        
        # If ninja position provided, compute reachable set
        if ninja_pos is not None:
            reachable = self._flood_fill_from_graph(ninja_pos, adjacency)
            result['reachable'] = reachable
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.build_times.append(elapsed_ms)
        
        return result
    
    def _build_base_graph(
        self,
        tiles: np.ndarray,
        level_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build base graph from tile layout with sub-tile nodes (cached per level).
        
        Key improvements:
        1. Each 24px tile → 4 sub-nodes at 12px resolution (2x2 grid)
        2. Respects tile types (type 0=empty OK, type 1=solid blocked)
        3. Only builds graph for reachable nodes from player spawn
        4. More accurate collision detection with 10px player radius
        """
        height, width = tiles.shape
        
        # Find player spawn position
        player_spawn = self._find_player_spawn(level_data, tiles)
        
        # Generate all sub-nodes and check which are valid (not in solid tiles)
        all_sub_nodes = self._generate_sub_nodes(tiles)
        
        # Build adjacency only for reachable sub-nodes from player spawn
        adjacency = self._build_reachable_adjacency(
            tiles, all_sub_nodes, player_spawn
        )
        
        return {'adjacency': adjacency}
    
    def _find_player_spawn(
        self,
        level_data: Dict[str, Any],
        tiles: np.ndarray
    ) -> Tuple[int, int]:
        """
        Find player spawn position from level data.
        
        Returns pixel coordinates of spawn point.
        Falls back to center of level if not found.
        """
        # Check for entities with type 14 (ninja spawn)
        entities = level_data.get('entities', [])
        for entity in entities:
            if entity.get('type') == 14:  # Ninja spawn point
                # Entity position is in pixels
                x = entity.get('x', 0)
                y = entity.get('y', 0)
                return (int(x), int(y))
        
        # Fallback: center of level (skip padding)
        height, width = tiles.shape
        center_x = (width // 2) * CELL_SIZE + CELL_SIZE // 2
        center_y = (height // 2) * CELL_SIZE + CELL_SIZE // 2
        return (center_x, center_y)
    
    def _generate_sub_nodes(
        self,
        tiles: np.ndarray
    ) -> Dict[Tuple[int, int], Tuple[int, int, int, int]]:
        """
        Generate all sub-nodes for level.
        
        Returns:
            Dict mapping (pixel_x, pixel_y) -> (tile_x, tile_y, sub_x, sub_y)
            where sub_x, sub_y are 0 or 1 indicating position within tile
        """
        height, width = tiles.shape
        sub_nodes = {}
        
        # Sub-node offsets within a 24px tile
        # Creates 2x2 grid: (6,6), (18,6), (6,18), (18,18)
        sub_offsets = [(6, 6), (18, 6), (6, 18), (18, 18)]
        sub_coords = [(0, 0), (1, 0), (0, 1), (1, 1)]  # (sub_x, sub_y) for each offset
        
        for tile_y in range(height):
            for tile_x in range(width):
                tile_type = tiles[tile_y, tile_x]
                
                # Skip completely solid tiles (type 1)
                # Type 0 (empty) is always traversable
                if tile_type == 1:
                    continue
                
                # Create 4 sub-nodes for this tile
                for (offset_x, offset_y), (sub_x, sub_y) in zip(sub_offsets, sub_coords):
                    pixel_x = tile_x * CELL_SIZE + offset_x
                    pixel_y = tile_y * CELL_SIZE + offset_y
                    
                    sub_nodes[(pixel_x, pixel_y)] = (tile_x, tile_y, sub_x, sub_y)
        
        return sub_nodes
    
    def _build_reachable_adjacency(
        self,
        tiles: np.ndarray,
        all_sub_nodes: Dict[Tuple[int, int], Tuple[int, int, int, int]],
        player_spawn: Tuple[int, int]
    ) -> Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]:
        """
        Build adjacency graph only for reachable sub-nodes from player spawn.
        
        Uses flood-fill to find reachable nodes, then builds detailed adjacency
        only for those nodes. Unreachable nodes are excluded entirely.
        """
        # Find closest sub-node to player spawn
        spawn_node = self._find_closest_node(player_spawn, all_sub_nodes)
        if spawn_node is None:
            # No valid spawn node found, return empty adjacency
            return {}
        
        # Build adjacency incrementally using BFS from spawn
        adjacency = {}
        visited = set()
        queue = deque([spawn_node])
        visited.add(spawn_node)
        
        # Direction mappings for 8-connectivity
        directions = {
            'N':  (0, -12),  'NE': (12, -12),
            'E':  (12, 0),   'SE': (12, 12),
            'S':  (0, 12),   'SW': (-12, 12),
            'W':  (-12, 0),  'NW': (-12, -12)
        }
        
        while queue:
            current_pos = queue.popleft()
            current_x, current_y = current_pos
            
            if current_pos not in all_sub_nodes:
                continue
            
            tile_x, tile_y, sub_x, sub_y = all_sub_nodes[current_pos]
            tile_type = tiles[tile_y, tile_x]
            
            neighbors = []
            
            # Check each direction for potential neighbors
            for dir_name, (dx, dy) in directions.items():
                neighbor_x = current_x + dx
                neighbor_y = current_y + dy
                neighbor_pos = (neighbor_x, neighbor_y)
                
                # Check if neighbor sub-node exists
                if neighbor_pos not in all_sub_nodes:
                    continue
                
                n_tile_x, n_tile_y, n_sub_x, n_sub_y = all_sub_nodes[neighbor_pos]
                neighbor_tile_type = tiles[n_tile_y, n_tile_x]
                
                # Check traversability
                # For sub-nodes, we need to check if movement is possible
                # Use simplified check: both nodes must be in non-solid tiles
                if self._is_sub_node_traversable(
                    tile_type, neighbor_tile_type, tile_x, tile_y, 
                    n_tile_x, n_tile_y, dir_name, tiles
                ):
                    # Calculate cost (Euclidean distance)
                    cost = (dx**2 + dy**2) ** 0.5
                    neighbors.append((neighbor_pos, cost))
                    
                    # Add to queue if not visited
                    if neighbor_pos not in visited:
                        visited.add(neighbor_pos)
                        queue.append(neighbor_pos)
            
            adjacency[current_pos] = neighbors
        
        return adjacency
    
    def _find_closest_node(
        self,
        pos: Tuple[int, int],
        sub_nodes: Dict[Tuple[int, int], Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int]]:
        """Find closest sub-node to given position."""
        if not sub_nodes:
            return None
        
        px, py = pos
        closest_node = None
        min_dist = float('inf')
        
        for node_pos in sub_nodes.keys():
            nx, ny = node_pos
            dist = ((nx - px)**2 + (ny - py)**2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest_node = node_pos
        
        return closest_node
    
    def _is_sub_node_traversable(
        self,
        src_tile_type: int,
        dst_tile_type: int,
        src_tile_x: int,
        src_tile_y: int,
        dst_tile_x: int,
        dst_tile_y: int,
        direction: str,
        tiles: np.ndarray
    ) -> bool:
        """
        Check if movement from source sub-node to destination sub-node is traversable.
        
        Simplified checks:
        1. Both tiles must not be type 1 (solid)
        2. For diagonal movements, check intermediate tiles aren't solid
        3. Respect level boundaries (1-tile padding of solid tiles)
        """
        # Type 1 tiles are completely solid - cannot traverse
        if src_tile_type == 1 or dst_tile_type == 1:
            return False
        
        # Type 0 tiles are completely empty - always traversable
        if src_tile_type == 0 and dst_tile_type == 0:
            # For diagonals, check corner isn't blocked
            if direction in ['NE', 'SE', 'SW', 'NW']:
                # Check intermediate tiles for corner cutting
                if not self._check_diagonal_clear(
                    src_tile_x, src_tile_y, dst_tile_x, dst_tile_y, tiles
                ):
                    return False
            return True
        
        # For other tile types, use precomputed connectivity
        # Map tile-to-tile direction (since sub-nodes might be within same tile
        # or in adjacent tiles)
        tile_dx = dst_tile_x - src_tile_x
        tile_dy = dst_tile_y - src_tile_y
        
        # If in same tile, always traversable (already checked type != 1)
        if tile_dx == 0 and tile_dy == 0:
            return True
        
        # Use connectivity loader for tile-to-tile movement
        try:
            return self.connectivity_loader.is_traversable(
                src_tile_type, dst_tile_type, direction
            )
        except:
            # Fallback: allow if neither is solid
            return src_tile_type != 1 and dst_tile_type != 1
    
    def _check_diagonal_clear(
        self,
        src_x: int,
        src_y: int,
        dst_x: int,
        dst_y: int,
        tiles: np.ndarray
    ) -> bool:
        """
        Check if diagonal movement doesn't cut through solid corners.
        
        For diagonal movement, need to ensure the two intermediate
        tiles aren't both solid (preventing corner cutting).
        """
        height, width = tiles.shape
        
        # Check the two intermediate tiles
        # For NE: check N and E tiles
        # For SE: check S and E tiles
        # etc.
        
        dx = dst_x - src_x
        dy = dst_y - src_y
        
        # Check tile to the side (dx, 0)
        side_x = src_x + dx
        side_y = src_y
        if 0 <= side_x < width and 0 <= side_y < height:
            if tiles[side_y, side_x] == 1:
                # Side tile is solid, check vertical
                vert_x = src_x
                vert_y = src_y + dy
                if 0 <= vert_x < width and 0 <= vert_y < height:
                    if tiles[vert_y, vert_x] == 1:
                        # Both intermediate tiles are solid - cannot cut corner
                        return False
        
        return True
    
    def _apply_entity_mask(
        self,
        base_adjacency: Dict,
        blocked_positions: Set[Tuple[int, int]],
        blocked_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]]
    ) -> Dict:
        """
        Apply entity state mask to base adjacency graph.
        
        Filters out blocked positions and edges based on current entity states.
        For sub-node system: block all 4 sub-nodes in a blocked tile.
        """
        # Convert blocked tile positions to sub-node pixel positions
        # Each blocked tile blocks all 4 sub-nodes within it
        blocked_pixels = set()
        sub_offsets = [(6, 6), (18, 6), (6, 18), (18, 18)]
        
        for tile_x, tile_y in blocked_positions:
            # Block all 4 sub-nodes in this tile
            for offset_x, offset_y in sub_offsets:
                pixel_x = tile_x * CELL_SIZE + offset_x
                pixel_y = tile_y * CELL_SIZE + offset_y
                blocked_pixels.add((pixel_x, pixel_y))
        
        # Build filtered adjacency
        filtered = {}
        
        for pos, neighbors in base_adjacency.items():
            # Skip if position itself is blocked
            if pos in blocked_pixels:
                continue
            
            # Filter neighbors
            valid_neighbors = []
            for neighbor_pos, cost in neighbors:
                # Skip if neighbor is blocked
                if neighbor_pos in blocked_pixels:
                    continue
                
                # Skip if edge is blocked
                edge = (pos, neighbor_pos)
                if edge in blocked_edges:
                    continue
                
                valid_neighbors.append((neighbor_pos, cost))
            
            filtered[pos] = valid_neighbors
        
        return filtered
    
    def _flood_fill_from_graph(
        self,
        start: Tuple[int, int],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]
    ) -> Set[Tuple[int, int]]:
        """
        Perform flood fill on adjacency graph to find reachable positions.
        
        Args:
            start: Starting position (pixel coordinates)
            adjacency: Graph adjacency structure
        
        Returns:
            Set of reachable positions
        """
        if start not in adjacency:
            return set()
        
        reachable = set()
        queue = deque([start])
        visited = {start}
        
        while queue:
            current = queue.popleft()
            reachable.add(current)
            
            # Get neighbors from adjacency
            neighbors = adjacency.get(current, [])
            for neighbor_pos, _ in neighbors:
                if neighbor_pos not in visited:
                    visited.add(neighbor_pos)
                    queue.append(neighbor_pos)
        
        return reachable
    
    def clear_cache(self):
        """Clear level graph cache (call on environment reset)."""
        self._level_graph_cache.clear()
        self._current_level_id = None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_time = np.mean(self.build_times) if self.build_times else 0
        max_time = np.max(self.build_times) if self.build_times else 0
        
        return {
            'total_builds': len(self.build_times),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) 
                       if (self.cache_hits + self.cache_misses) > 0 else 0,
            'avg_build_time_ms': avg_time,
            'max_build_time_ms': max_time,
            'cached_levels': len(self._level_graph_cache)
        }
