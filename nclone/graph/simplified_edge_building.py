"""
Simplified edge building for strategic RL representation.

This module creates graph edges based on connectivity and strategic relationships
rather than detailed physics calculations. It focuses on information that helps
RL agents make strategic decisions while letting them learn movement through experience.
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass

from .common import EdgeType, NodeType, GraphData
from .level_data import LevelData
from ..constants.physics_constants import TILE_PIXEL_SIZE


@dataclass
class SimplifiedEdge:
    """Simplified edge representation for strategic RL."""
    source: Tuple[int, int]
    target: Tuple[int, int]
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Optional[Dict] = None


class SimplifiedEdgeBuilder:
    """
    Builds graph edges using simplified connectivity rules.
    
    This approach focuses on strategic information rather than detailed physics:
    - Adjacent edges for direct movement possibilities
    - Reachable edges based on flood fill analysis
    - Functional edges for entity interactions
    - Blocked edges for conditional access
    """
    
    def __init__(self, debug: bool = False):
        """Initialize simplified edge builder."""
        self.debug = debug
        self.reachability_system = None  # Will be injected
        
    def set_reachability_system(self, reachability_system):
        """Inject reachability system for connectivity analysis."""
        self.reachability_system = reachability_system
    
    def build_edges(self, level_data: LevelData, entities: List, ninja_pos: Tuple[int, int]) -> List[SimplifiedEdge]:
        """
        Build simplified edges for strategic RL representation.
        
        Args:
            level_data: Level tile data
            entities: List of game entities
            ninja_pos: Current ninja position
            
        Returns:
            List of simplified edges
        """
        edges = []
        
        # Get traversable positions from level data
        traversable_positions = self._get_traversable_positions(level_data)
        
        # 1. Create adjacent edges (basic connectivity)
        adjacent_edges = self._create_adjacent_edges(traversable_positions, level_data)
        edges.extend(adjacent_edges)
        
        # 2. Create reachable edges (using flood fill results)
        if self.reachability_system:
            reachable_edges = self._create_reachable_edges(ninja_pos, level_data)
            edges.extend(reachable_edges)
        
        # 3. Create functional edges (entity interactions)
        functional_edges = self._create_functional_edges(entities, traversable_positions)
        edges.extend(functional_edges)
        
        # 4. Create blocked edges (conditional access)
        blocked_edges = self._create_blocked_edges(entities, level_data)
        edges.extend(blocked_edges)
        
        if self.debug:
            print(f"Created {len(edges)} simplified edges:")
            print(f"  Adjacent: {len(adjacent_edges)}")
            print(f"  Reachable: {len(reachable_edges) if self.reachability_system else 0}")
            print(f"  Functional: {len(functional_edges)}")
            print(f"  Blocked: {len(blocked_edges)}")
        
        return edges
    
    def _get_traversable_positions(self, level_data: LevelData) -> Set[Tuple[int, int]]:
        """Get all positions that are not solid walls."""
        traversable = set()
        
        for row in range(level_data.height):
            for col in range(level_data.width):
                tile_id = level_data.tiles[row, col]
                # Assume tile_id 0 is empty space (traversable)
                # This is a simplification - in reality we'd check tile properties
                if tile_id == 0:
                    # Convert to pixel coordinates
                    pixel_x = col * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
                    pixel_y = row * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
                    traversable.add((pixel_x, pixel_y))
        
        return traversable
    
    def _create_adjacent_edges(self, traversable_positions: Set[Tuple[int, int]], 
                             level_data: LevelData) -> List[SimplifiedEdge]:
        """Create edges between adjacent traversable positions."""
        edges = []
        
        # Use 4-connectivity (up, down, left, right)
        directions = [(0, TILE_PIXEL_SIZE), (0, -TILE_PIXEL_SIZE), 
                     (TILE_PIXEL_SIZE, 0), (-TILE_PIXEL_SIZE, 0)]
        
        for pos in traversable_positions:
            x, y = pos
            for dx, dy in directions:
                neighbor = (x + dx, y + dy)
                if neighbor in traversable_positions:
                    edges.append(SimplifiedEdge(
                        source=pos,
                        target=neighbor,
                        edge_type=EdgeType.ADJACENT,
                        weight=1.0
                    ))
        
        return edges
    
    def _create_reachable_edges(self, ninja_pos: Tuple[int, int], 
                              level_data: LevelData) -> List[SimplifiedEdge]:
        """Create edges based on reachability analysis."""
        edges = []
        
        if not self.reachability_system:
            return edges
        
        try:
            # Get reachable positions from flood fill
            result = self.reachability_system.quick_check(level_data, [], ninja_pos)
            
            if hasattr(result, 'reachable_positions'):
                reachable_positions = result.reachable_positions
                
                # Create reachable edges between positions that are connected
                # but not necessarily adjacent
                for pos1 in reachable_positions:
                    for pos2 in reachable_positions:
                        if pos1 != pos2:
                            distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                            # Connect positions that are reachable but not immediately adjacent
                            if TILE_PIXEL_SIZE < distance < TILE_PIXEL_SIZE * 3:
                                edges.append(SimplifiedEdge(
                                    source=pos1,
                                    target=pos2,
                                    edge_type=EdgeType.REACHABLE,
                                    weight=distance / TILE_PIXEL_SIZE
                                ))
        
        except Exception as e:
            if self.debug:
                print(f"Warning: Could not create reachable edges: {e}")
        
        return edges
    
    def _create_functional_edges(self, entities: List, 
                               traversable_positions: Set[Tuple[int, int]]) -> List[SimplifiedEdge]:
        """Create edges for entity interactions."""
        edges = []
        
        for entity in entities:
            entity_pos = (entity.x, entity.y)
            
            # Find nearby traversable positions
            nearby_positions = self._get_positions_near_entity(entity_pos, traversable_positions)
            
            for pos in nearby_positions:
                edges.append(SimplifiedEdge(
                    source=pos,
                    target=entity_pos,
                    edge_type=EdgeType.FUNCTIONAL,
                    weight=1.0,
                    metadata={'entity_type': type(entity).__name__}
                ))
        
        return edges
    
    def _create_blocked_edges(self, entities: List, level_data: LevelData) -> List[SimplifiedEdge]:
        """Create edges for conditionally blocked access."""
        edges = []
        
        # Find door-switch relationships
        doors = [e for e in entities if 'door' in type(e).__name__.lower()]
        switches = [e for e in entities if 'switch' in type(e).__name__.lower()]
        
        for door in doors:
            door_pos = (door.x, door.y)
            
            # Create blocked edges around doors
            # This is a simplification - in reality we'd check door state
            nearby_positions = self._get_positions_near_entity(door_pos, set())
            
            for i, pos1 in enumerate(nearby_positions):
                for pos2 in nearby_positions[i+1:]:
                    edges.append(SimplifiedEdge(
                        source=pos1,
                        target=pos2,
                        edge_type=EdgeType.BLOCKED,
                        weight=float('inf'),  # Infinite weight when blocked
                        metadata={'door_id': getattr(door, 'id', 0)}
                    ))
        
        return edges
    
    def _get_positions_near_entity(self, entity_pos: Tuple[int, int], 
                                 traversable_positions: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Get traversable positions near an entity."""
        nearby = []
        x, y = entity_pos
        
        # Check positions in a small radius around the entity
        for dx in [-TILE_PIXEL_SIZE, 0, TILE_PIXEL_SIZE]:
            for dy in [-TILE_PIXEL_SIZE, 0, TILE_PIXEL_SIZE]:
                if dx == 0 and dy == 0:
                    continue
                pos = (x + dx, y + dy)
                if pos in traversable_positions:
                    nearby.append(pos)
        
        return nearby


def create_simplified_graph_data(edges: List[SimplifiedEdge], 
                               level_data: LevelData,
                               entities: List) -> GraphData:
    """
    Create GraphData from simplified edges.
    
    This function converts the simplified edge representation into the
    standard GraphData format expected by the rest of the system.
    """
    from .common import N_MAX_NODES, E_MAX_EDGES
    
    # Extract unique positions from edges
    positions = set()
    for edge in edges:
        positions.add(edge.source)
        positions.add(edge.target)
    
    # If no positions, create at least one node
    if not positions:
        positions.add((0, 0))
    
    # Create position to index mapping
    pos_to_idx = {pos: idx for idx, pos in enumerate(sorted(positions))}
    
    num_nodes = len(positions)
    num_edges = len(edges)
    
    # Initialize fixed-size arrays
    node_features = np.zeros((N_MAX_NODES, 3), dtype=np.float32)  # [x, y, node_type]
    node_types = np.zeros(N_MAX_NODES, dtype=np.int32)
    node_mask = np.zeros(N_MAX_NODES, dtype=np.int32)
    
    edge_index = np.zeros((2, E_MAX_EDGES), dtype=np.int32)
    edge_features = np.zeros((E_MAX_EDGES, 1), dtype=np.float32)  # Just weight for now
    edge_types = np.zeros(E_MAX_EDGES, dtype=np.int32)
    edge_mask = np.zeros(E_MAX_EDGES, dtype=np.int32)
    
    # Fill node data
    for i, pos in enumerate(sorted(positions)):
        if i >= N_MAX_NODES:
            break
            
        # Determine node type based on position and entities
        node_type = NodeType.EMPTY  # Default
        
        # Check if position corresponds to an entity
        for entity in entities:
            if (entity.x, entity.y) == pos:
                if 'hazard' in type(entity).__name__.lower() or 'mine' in type(entity).__name__.lower():
                    node_type = NodeType.HAZARD
                elif 'exit' in type(entity).__name__.lower():
                    node_type = NodeType.EXIT
                else:
                    node_type = NodeType.ENTITY
                break
        
        node_features[i] = [pos[0], pos[1], float(node_type)]
        node_types[i] = node_type
        node_mask[i] = 1
    
    # Fill edge data
    for i, edge in enumerate(edges):
        if i >= E_MAX_EDGES:
            break
            
        source_idx = pos_to_idx[edge.source]
        target_idx = pos_to_idx[edge.target]
        
        edge_index[0, i] = source_idx
        edge_index[1, i] = target_idx
        edge_features[i, 0] = edge.weight
        edge_types[i] = edge.edge_type
        edge_mask[i] = 1
    
    return GraphData(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        node_mask=node_mask,
        edge_mask=edge_mask,
        node_types=node_types,
        edge_types=edge_types,
        num_nodes=num_nodes,
        num_edges=num_edges
    )