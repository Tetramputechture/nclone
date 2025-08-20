"""
Graph builder for creating structural representations of N++ levels.

This module constructs graph representations of N++ levels where:
- Nodes represent grid cells and entities
- Edges represent traversability and functional relationships
- Features encode tile types, entity states, and physics properties
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import IntEnum

# Graph configuration constants
N_MAX_NODES = 1200  # 42×23 grid + entities buffer
E_MAX_EDGES = 4800  # Approximate max edges (4 directions × nodes)
GRID_WIDTH = 42
GRID_HEIGHT = 23
CELL_SIZE = 24


class NodeType(IntEnum):
    """Types of nodes in the graph."""
    GRID_CELL = 0
    ENTITY = 1
    NINJA = 2


class EdgeType(IntEnum):
    """Types of edges in the graph."""
    WALK = 0
    JUMP = 1
    WALL_SLIDE = 2
    FALL = 3
    ONE_WAY = 4
    FUNCTIONAL = 5  # switch->door, launchpad->target, etc.


@dataclass
class GraphData:
    """Container for graph data with fixed-size arrays for Gym compatibility."""
    node_features: np.ndarray  # [N_MAX_NODES, F_node]
    edge_index: np.ndarray     # [2, E_MAX_EDGES] 
    edge_features: np.ndarray  # [E_MAX_EDGES, F_edge]
    node_mask: np.ndarray      # [N_MAX_NODES] - 1 for valid nodes, 0 for padding
    edge_mask: np.ndarray      # [E_MAX_EDGES] - 1 for valid edges, 0 for padding
    num_nodes: int
    num_edges: int


class GraphBuilder:
    """
    Builds graph representations of N++ levels.
    
    The graph construction follows a canonical ordering:
    1. Grid cells: iterate rows then columns (top-left to bottom-right)
    2. Entities: sorted by type then by position
    3. Edges: sorted by (source, target, edge_type)
    """
    
    def __init__(self):
        """Initialize graph builder."""
        # Node feature dimensions
        self.tile_type_dim = 38  # Number of tile types in N++
        self.entity_type_dim = 20  # Approximate number of entity types
        self.node_feature_dim = (
            self.tile_type_dim +  # One-hot tile type
            4 +  # Solidity flags (solid, half, slope, hazard)
            self.entity_type_dim +  # One-hot entity type
            4 +  # Entity state (active, position_x, position_y, custom_state)
            1    # Ninja position flag
        )
        
        # Edge feature dimensions  
        self.edge_feature_dim = (
            len(EdgeType) +  # One-hot edge type
            2 +  # Direction (dx, dy normalized)
            1    # Traversability cost
        )
    
    def build_graph(
        self,
        level_data: Dict[str, Any],
        ninja_position: Tuple[float, float],
        entities: List[Dict[str, Any]]
    ) -> GraphData:
        """
        Build graph representation of a level.
        
        Args:
            level_data: Level tile data and structure
            ninja_position: Current ninja position (x, y)
            entities: List of entity dictionaries with type, position, state
            
        Returns:
            GraphData with padded arrays for Gym compatibility
        """
        # Initialize arrays
        node_features = np.zeros((N_MAX_NODES, self.node_feature_dim), dtype=np.float32)
        edge_index = np.zeros((2, E_MAX_EDGES), dtype=np.int32)
        edge_features = np.zeros((E_MAX_EDGES, self.edge_feature_dim), dtype=np.float32)
        node_mask = np.zeros(N_MAX_NODES, dtype=np.float32)
        edge_mask = np.zeros(E_MAX_EDGES, dtype=np.float32)
        
        node_count = 0
        edge_count = 0
        
        # Build grid nodes (canonical ordering: row by row)
        grid_node_map = {}  # (row, col) -> node_index
        
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                if node_count >= N_MAX_NODES:
                    break
                    
                node_idx = node_count
                grid_node_map[(row, col)] = node_idx
                
                # Extract tile features
                tile_features = self._extract_tile_features(
                    level_data, row, col, ninja_position
                )
                
                node_features[node_idx] = tile_features
                node_mask[node_idx] = 1.0
                node_count += 1
        
        # Build entity nodes (sorted by type then position for determinism)
        entity_nodes = []
        sorted_entities = sorted(entities, key=lambda e: (e.get('type', ''), e.get('x', 0), e.get('y', 0)))
        
        for entity in sorted_entities:
            if node_count >= N_MAX_NODES:
                break
                
            node_idx = node_count
            entity_nodes.append((node_idx, entity))
            
            # Extract entity features
            entity_features = self._extract_entity_features(entity, ninja_position)
            
            node_features[node_idx] = entity_features
            node_mask[node_idx] = 1.0
            node_count += 1
        
        # Build edges (canonical ordering: by source, target, type)
        edges_to_add = []
        
        # Grid traversability edges
        for (row, col), src_idx in grid_node_map.items():
            # Check 4-connected neighbors
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_row, new_col = row + dr, col + dc
                
                if (new_row, new_col) in grid_node_map:
                    tgt_idx = grid_node_map[(new_row, new_col)]
                    
                    # Determine edge type based on tile properties
                    edge_type, cost = self._determine_traversability(
                        level_data, row, col, new_row, new_col
                    )
                    
                    if edge_type is not None:
                        edges_to_add.append((src_idx, tgt_idx, edge_type, cost, dr, dc))
        
        # Functional relationship edges (switch->door, etc.)
        functional_edges = self._build_functional_edges(entity_nodes, grid_node_map)
        edges_to_add.extend(functional_edges)
        
        # Sort edges canonically and add to arrays
        edges_to_add.sort(key=lambda e: (e[0], e[1], e[2]))  # Sort by src, tgt, type
        
        for src, tgt, edge_type, cost, dx, dy in edges_to_add:
            if edge_count >= E_MAX_EDGES:
                break
                
            edge_index[0, edge_count] = src
            edge_index[1, edge_count] = tgt
            
            # Build edge features
            edge_feat = np.zeros(self.edge_feature_dim, dtype=np.float32)
            edge_feat[edge_type] = 1.0  # One-hot edge type
            edge_feat[len(EdgeType)] = dx  # Direction x
            edge_feat[len(EdgeType) + 1] = dy  # Direction y  
            edge_feat[len(EdgeType) + 2] = cost  # Traversability cost
            
            edge_features[edge_count] = edge_feat
            edge_mask[edge_count] = 1.0
            edge_count += 1
        
        return GraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_mask=node_mask,
            edge_mask=edge_mask,
            num_nodes=node_count,
            num_edges=edge_count
        )
    
    def _extract_tile_features(
        self,
        level_data: Dict[str, Any],
        row: int,
        col: int,
        ninja_position: Tuple[float, float]
    ) -> np.ndarray:
        """
        Extract features for a grid cell node.
        
        Args:
            level_data: Level data containing tile information
            row: Grid row
            col: Grid column
            ninja_position: Current ninja position
            
        Returns:
            Feature vector for the grid cell
        """
        features = np.zeros(self.node_feature_dim, dtype=np.float32)
        
        # Get tile type (mock implementation - would use actual level data)
        tile_type = self._get_tile_type(level_data, row, col)
        if 0 <= tile_type < self.tile_type_dim:
            features[tile_type] = 1.0
        
        # Solidity flags
        solidity_offset = self.tile_type_dim
        features[solidity_offset] = self._is_solid(tile_type)
        features[solidity_offset + 1] = self._is_half_tile(tile_type)
        features[solidity_offset + 2] = self._is_slope(tile_type)
        features[solidity_offset + 3] = self._is_hazard(tile_type)
        
        # Entity features (zero for grid cells)
        entity_offset = solidity_offset + 4
        # Entity type one-hot: all zeros
        # Entity state: all zeros
        
        # Ninja position flag
        ninja_offset = entity_offset + self.entity_type_dim + 4
        cell_x = col * CELL_SIZE + CELL_SIZE // 2
        cell_y = row * CELL_SIZE + CELL_SIZE // 2
        
        # Check if ninja is in this cell (within cell bounds)
        if (abs(ninja_position[0] - cell_x) < CELL_SIZE // 2 and
            abs(ninja_position[1] - cell_y) < CELL_SIZE // 2):
            features[ninja_offset] = 1.0
        
        return features
    
    def _extract_entity_features(
        self,
        entity: Dict[str, Any],
        ninja_position: Tuple[float, float]
    ) -> np.ndarray:
        """
        Extract features for an entity node.
        
        Args:
            entity: Entity data dictionary
            ninja_position: Current ninja position
            
        Returns:
            Feature vector for the entity
        """
        features = np.zeros(self.node_feature_dim, dtype=np.float32)
        
        # Tile features (zero for entities)
        tile_offset = self.tile_type_dim + 4
        
        # Entity type one-hot
        entity_offset = tile_offset
        entity_type_id = self._get_entity_type_id(entity.get('type', 'unknown'))
        if 0 <= entity_type_id < self.entity_type_dim:
            features[entity_offset + entity_type_id] = 1.0
        
        # Entity state
        state_offset = entity_offset + self.entity_type_dim
        features[state_offset] = float(entity.get('active', True))
        features[state_offset + 1] = entity.get('x', 0) / (GRID_WIDTH * CELL_SIZE)  # Normalized x
        features[state_offset + 2] = entity.get('y', 0) / (GRID_HEIGHT * CELL_SIZE)  # Normalized y
        features[state_offset + 3] = entity.get('state', 0)  # Custom state
        
        # Ninja position flag (always 0 for entity nodes)
        ninja_offset = state_offset + 4
        features[ninja_offset] = 0.0
        
        return features
    
    def _determine_traversability(
        self,
        level_data: Dict[str, Any],
        src_row: int,
        src_col: int,
        tgt_row: int,
        tgt_col: int
    ) -> Tuple[Optional[EdgeType], float]:
        """
        Determine if two grid cells are connected and how.
        
        Args:
            level_data: Level data
            src_row, src_col: Source cell coordinates
            tgt_row, tgt_col: Target cell coordinates
            
        Returns:
            Tuple of (edge_type, traversability_cost) or (None, 0) if not traversable
        """
        src_tile = self._get_tile_type(level_data, src_row, src_col)
        tgt_tile = self._get_tile_type(level_data, tgt_row, tgt_col)
        
        # Simple traversability rules (would be more complex in real implementation)
        if self._is_solid(tgt_tile):
            return None, 0.0  # Can't move into solid tiles
        
        # Determine movement type
        dr = tgt_row - src_row
        dc = tgt_col - src_col
        
        if dr == 0:  # Horizontal movement
            if self._is_solid(src_tile):
                return EdgeType.WALL_SLIDE, 1.5  # Wall sliding
            else:
                return EdgeType.WALK, 1.0  # Normal walking
        elif dr > 0:  # Downward movement
            return EdgeType.FALL, 0.8  # Falling is fast
        else:  # Upward movement
            return EdgeType.JUMP, 2.0  # Jumping is costly
    
    def _build_functional_edges(
        self,
        entity_nodes: List[Tuple[int, Dict[str, Any]]],
        grid_node_map: Dict[Tuple[int, int], int]
    ) -> List[Tuple[int, int, EdgeType, float, float, float]]:
        """
        Build functional relationship edges between entities.
        
        Args:
            entity_nodes: List of (node_index, entity_data) tuples
            grid_node_map: Mapping from grid coordinates to node indices
            
        Returns:
            List of edge tuples (src, tgt, type, cost, dx, dy)
        """
        edges = []
        
        # Find switches and doors for functional relationships
        switches = []
        doors = []
        
        for node_idx, entity in entity_nodes:
            entity_type = entity.get('type', '')
            if 'switch' in entity_type.lower():
                switches.append((node_idx, entity))
            elif 'door' in entity_type.lower():
                doors.append((node_idx, entity))
        
        # Connect switches to doors (simplified - would use actual game logic)
        for switch_idx, switch_entity in switches:
            for door_idx, door_entity in doors:
                # Add functional edge from switch to door
                dx = door_entity.get('x', 0) - switch_entity.get('x', 0)
                dy = door_entity.get('y', 0) - switch_entity.get('y', 0)
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance > 0:
                    dx /= distance
                    dy /= distance
                
                edges.append((switch_idx, door_idx, EdgeType.FUNCTIONAL, 1.0, dx, dy))
        
        return edges
    
    def _get_tile_type(self, level_data: Dict[str, Any], row: int, col: int) -> int:
        """Get tile type at grid position (mock implementation)."""
        # Mock implementation - would use actual level data
        if row == 0 or row == GRID_HEIGHT - 1 or col == 0 or col == GRID_WIDTH - 1:
            return 1  # Wall tile
        return 0  # Empty tile
    
    def _is_solid(self, tile_type: int) -> float:
        """Check if tile is solid."""
        return 1.0 if tile_type == 1 else 0.0
    
    def _is_half_tile(self, tile_type: int) -> float:
        """Check if tile is a half tile."""
        return 0.0  # Mock implementation
    
    def _is_slope(self, tile_type: int) -> float:
        """Check if tile is a slope."""
        return 0.0  # Mock implementation
    
    def _is_hazard(self, tile_type: int) -> float:
        """Check if tile is hazardous."""
        return 0.0  # Mock implementation
    
    def _get_entity_type_id(self, entity_type: str) -> int:
        """Map entity type string to integer ID."""
        entity_type_map = {
            'exit': 0,
            'switch': 1,
            'door': 2,
            'gold': 3,
            'drone': 4,
            'mine': 5,
            'thwump': 6,
            'launchpad': 7,
            'unknown': 19
        }
        
        for key, value in entity_type_map.items():
            if key in entity_type.lower():
                return value
        
        return entity_type_map['unknown']