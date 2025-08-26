"""
Graph builder for creating structural representations of N++ levels.

This module constructs graph representations of N++ levels where:
- Nodes represent grid cells and entities
- Edges represent traversability and functional relationships
- Features encode tile types, entity states, and physics properties
"""

import numpy as np
import logging
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import IntEnum

# Use shared constants and tile definitions used by the simulator itself
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT, TILE_PIXEL_SIZE
from ..tile_definitions import TILE_GRID_EDGE_MAP

# Graph configuration constants
# Sub-grid resolution for better spatial accuracy
SUB_GRID_RESOLUTION = 4  # Divide each tile into 4x4 sub-cells (6x6 pixels each)
SUB_CELL_SIZE = TILE_PIXEL_SIZE // SUB_GRID_RESOLUTION  # 6 pixels per sub-cell

# Calculate sub-grid dimensions
SUB_GRID_WIDTH = MAP_TILE_WIDTH * SUB_GRID_RESOLUTION  # 168 sub-cells wide
SUB_GRID_HEIGHT = MAP_TILE_HEIGHT * SUB_GRID_RESOLUTION  # 92 sub-cells tall

# Keep generous upper bounds to allow entity padding in observations
N_MAX_NODES = SUB_GRID_WIDTH * SUB_GRID_HEIGHT + 400  # Sub-grid + entities buffer (~15856)
E_MAX_EDGES = N_MAX_NODES * 8  # Up to 8 directions per node (4 cardinal + 4 diagonal)

# Mirror simulator map constants to avoid divergence
GRID_WIDTH = MAP_TILE_WIDTH
GRID_HEIGHT = MAP_TILE_HEIGHT
CELL_SIZE = TILE_PIXEL_SIZE


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
            self.tile_type_dim +  # One-hot tile type (38)
            4 +  # Solidity flags (solid, half, slope, hazard)
            self.entity_type_dim +  # One-hot entity type (20)
            4 +  # Entity state (active, position_x, position_y, custom_state)
            1 +  # Ninja position flag
            # Physics state features (18 additional)
            2 +  # Ninja velocity (vx, vy) normalized by MAX_HOR_SPEED
            1 +  # Velocity magnitude
            1 +  # Movement state (0-9 from sim_mechanics_doc.md)
            3 +  # Contact flags (ground_contact, wall_contact, airborne)
            2 +  # Momentum direction (normalized)
            1 +  # Kinetic energy (0.5 * m * v²)
            1 +  # Potential energy (relative to level bottom)
            5 +  # Input buffers (jump_buffer, floor_buffer, wall_buffer, launch_pad_buffer, input_state)
            2    # Physics capabilities (can_jump, can_wall_jump)
        )
        
        # Edge feature dimensions  
        self.edge_feature_dim = (
            len(EdgeType) +  # One-hot edge type (6)
            2 +  # Direction (dx, dy normalized) 
            1 +  # Traversability cost
            3 +  # NEW: Trajectory parameters (time_of_flight, energy_cost, success_probability)
            2 +  # NEW: Physics constraints (min_velocity, max_velocity)
            2    # NEW: Movement requirements (requires_jump, requires_wall_contact)
        )
        
        # Initialize trajectory calculator (lazy loading to avoid import issues)
        self.trajectory_calc = None
        self.movement_classifier = None
        self.physics_extractor = None
    
    def build_graph(
        self,
        level_data: Dict[str, Any],
        ninja_position: Tuple[float, float],
        entities: List[Dict[str, Any]],
        ninja_velocity: Optional[Tuple[float, float]] = None,
        ninja_state: Optional[int] = None
    ) -> GraphData:
        """
        Build graph representation of a level.
        
        Args:
            level_data: Level tile data and structure
            ninja_position: Current ninja position (x, y)
            entities: List of entity dictionaries with type, position, state
            ninja_velocity: Current ninja velocity (vx, vy) for physics calculations
            ninja_state: Current ninja movement state (0-8)
            
        Returns:
            GraphData with padded arrays for Gym compatibility
        """
        # Initialize arrays
        node_features = np.zeros((N_MAX_NODES, self.node_feature_dim), dtype=np.float32)
        edge_index = np.zeros((2, E_MAX_EDGES), dtype=np.int32)
        edge_features = np.zeros((E_MAX_EDGES, self.edge_feature_dim), dtype=np.float32)
        node_mask = np.zeros(N_MAX_NODES, dtype=np.float32)
        edge_mask = np.zeros(E_MAX_EDGES, dtype=np.float32)
        
        # Initialize trajectory calculator if needed
        self._ensure_trajectory_calculator()
        
        node_count = 0
        edge_count = 0
        
        # Build sub-grid nodes (canonical ordering: sub-row by sub-row)
        sub_grid_node_map = {}  # (sub_row, sub_col) -> node_index
        
        for sub_row in range(SUB_GRID_HEIGHT):
            for sub_col in range(SUB_GRID_WIDTH):
                if node_count >= N_MAX_NODES:
                    break
                    
                node_idx = node_count
                sub_grid_node_map[(sub_row, sub_col)] = node_idx
                
                # Extract sub-cell features
                sub_cell_features = self._extract_sub_cell_features(
                    level_data, sub_row, sub_col, ninja_position, ninja_velocity, ninja_state
                )
                
                node_features[node_idx] = sub_cell_features
                node_mask[node_idx] = 1.0
                node_count += 1
        
        # Build entity nodes (sorted by type then position for determinism)
        entity_nodes = []
        # Normalize sort key to avoid mixing str/int types which cause TypeError in Python 3
        sorted_entities = sorted(entities, key=self._entity_sort_key)
        
        for entity in sorted_entities:
            if node_count >= N_MAX_NODES:
                break
                
            node_idx = node_count
            entity_nodes.append((node_idx, entity))
            
            # Extract entity features
            entity_features = self._extract_entity_features(entity)
            
            node_features[node_idx] = entity_features
            node_mask[node_idx] = 1.0
            node_count += 1
        
        # Build edges (canonical ordering: by source, target, type)
        edges_to_add = []
        # Pre-collect one-way platforms for traversability checks
        one_way_index = self._collect_one_way_platforms(entities)
        
        # Sub-grid traversability edges
        door_blockers_index = self._collect_door_blockers(entities)
        for (sub_row, sub_col), src_idx in sub_grid_node_map.items():
            # Check 8-connected neighbors (4 cardinal + 4 diagonal)
            directions = [
                (0, 1), (1, 0), (0, -1), (-1, 0),  # Cardinal
                (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal
            ]
            
            for dr, dc in directions:
                new_sub_row, new_sub_col = sub_row + dr, sub_col + dc
                
                if (new_sub_row, new_sub_col) in sub_grid_node_map:
                    tgt_idx = sub_grid_node_map[(new_sub_row, new_sub_col)]
                    # Determine edge type based on sub-cell traversability
                    edge_type, cost, trajectory_info = self._determine_sub_cell_traversability(
                        level_data, sub_row, sub_col, new_sub_row, new_sub_col, 
                        one_way_index, door_blockers_index, dr, dc, ninja_velocity, ninja_state
                    )
                    if edge_type is not None:
                        # Normalize direction vector
                        norm = np.sqrt(dr*dr + dc*dc)
                        dx, dy = dc / norm, dr / norm
                        edges_to_add.append((src_idx, tgt_idx, edge_type, cost, dx, dy, trajectory_info))
        
        # Functional relationship edges (switch->door, etc.)
        functional_edges = self._build_functional_edges(entity_nodes, sub_grid_node_map)
        edges_to_add.extend(functional_edges)
        
        # Sort edges canonically and add to arrays
        edges_to_add.sort(key=lambda e: (e[0], e[1], e[2]))  # Sort by src, tgt, type
        
        for src, tgt, edge_type, cost, dx, dy, trajectory_info in edges_to_add:
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
            
            # Add trajectory features if available
            if trajectory_info is not None:
                base_idx = len(EdgeType) + 3  # After edge type, direction, and cost
                edge_feat[base_idx] = trajectory_info.get('time_of_flight', 0.0)
                edge_feat[base_idx + 1] = trajectory_info.get('energy_cost', 0.0)
                edge_feat[base_idx + 2] = trajectory_info.get('success_probability', 1.0)
                edge_feat[base_idx + 3] = trajectory_info.get('min_velocity', 0.0)
                edge_feat[base_idx + 4] = trajectory_info.get('max_velocity', 0.0)
                edge_feat[base_idx + 5] = 1.0 if trajectory_info.get('requires_jump', False) else 0.0
                edge_feat[base_idx + 6] = 1.0 if trajectory_info.get('requires_wall_contact', False) else 0.0
            
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
    
    def _ensure_trajectory_calculator(self):
        """Initialize trajectory calculator, movement classifier, and physics extractor if needed."""
        if self.trajectory_calc is None or self.physics_extractor is None:
            try:
                # Import here to avoid circular imports
                import sys
                import os
                
                # Add npp_rl to path if not already there
                npp_rl_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'npp-rl')
                if os.path.exists(npp_rl_path) and npp_rl_path not in sys.path:
                    sys.path.insert(0, npp_rl_path)
                
                from npp_rl.models.trajectory_calculator import TrajectoryCalculator
                from npp_rl.models.movement_classifier import MovementClassifier
                from npp_rl.models.physics_state_extractor import PhysicsStateExtractor
                
                if self.trajectory_calc is None:
                    self.trajectory_calc = TrajectoryCalculator()
                if self.movement_classifier is None:
                    self.movement_classifier = MovementClassifier()
                if self.physics_extractor is None:
                    self.physics_extractor = PhysicsStateExtractor()
            except ImportError as e:
                logging.warning(f"Could not import physics modules: {e}")
                if self.trajectory_calc is None:
                    self.trajectory_calc = None
                if self.movement_classifier is None:
                    self.movement_classifier = None
                if self.physics_extractor is None:
                    self.physics_extractor = None
    
    def _extract_sub_cell_features(
        self,
        level_data: Dict[str, Any],
        sub_row: int,
        sub_col: int,
        ninja_position: Tuple[float, float],
        ninja_velocity: Optional[Tuple[float, float]] = None,
        ninja_state: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Extract features for a sub-grid cell node with physics state.
        
        Args:
            level_data: Level data containing tile information
            sub_row: Sub-grid row (0 to SUB_GRID_HEIGHT-1)
            sub_col: Sub-grid column (0 to SUB_GRID_WIDTH-1)
            ninja_position: Current ninja position
            ninja_velocity: Current ninja velocity (vx, vy) for physics calculations
            ninja_state: Current ninja state dictionary with movement_state and buffers
            
        Returns:
            Feature vector for the sub-cell
        """
        features = np.zeros(self.node_feature_dim, dtype=np.float32)
        
        # Convert sub-grid coordinates to tile coordinates
        tile_row = sub_row // SUB_GRID_RESOLUTION
        tile_col = sub_col // SUB_GRID_RESOLUTION
        
        # Get tile type from level data (0-37, see tile_definitions)
        tile_type = self._get_tile_type(level_data, tile_row, tile_col)
        if 0 <= tile_type < self.tile_type_dim:
            features[tile_type] = 1.0
        
        # Enhanced solidity flags with sub-cell spatial awareness
        solidity_offset = self.tile_type_dim
        
        # Check if this specific sub-cell is traversable
        is_traversable = self._is_sub_cell_traversable(level_data, sub_row, sub_col)
        features[solidity_offset] = 0.0 if is_traversable else 1.0  # Inverse of traversability
        features[solidity_offset + 1] = 1.0 if self._is_half_tile(tile_type) else 0.0
        features[solidity_offset + 2] = 1.0 if self._is_slope(tile_type) else 0.0
        # Tiles are not hazards in the simulator; hazards are entities (mines, drones, thwumps...)
        features[solidity_offset + 3] = 0.0
        
        # Entity features (zero for grid cells)
        entity_offset = solidity_offset + 4
        # Entity type one-hot: all zeros
        # Entity state: all zeros
        
        # Ninja position flag
        ninja_offset = entity_offset + self.entity_type_dim + 4
        sub_cell_x = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        sub_cell_y = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        
        # Check if ninja is in this sub-cell (within sub-cell bounds)
        ninja_in_cell = (abs(ninja_position[0] - sub_cell_x) < SUB_CELL_SIZE // 2 and
                        abs(ninja_position[1] - sub_cell_y) < SUB_CELL_SIZE // 2)
        features[ninja_offset] = 1.0 if ninja_in_cell else 0.0
        
        # Physics state features (only for ninja's current cell)
        physics_offset = ninja_offset + 1
        if ninja_in_cell and ninja_velocity is not None and ninja_state is not None and self.physics_extractor is not None:
            try:
                physics_features = self.physics_extractor.extract_ninja_physics_state(
                    ninja_position, ninja_velocity, ninja_state, level_data
                )
                features[physics_offset:physics_offset + len(physics_features)] = physics_features
            except Exception as e:
                logging.warning(f"Failed to extract physics features: {e}")
                # Fill with zeros if extraction fails
                features[physics_offset:physics_offset + 18] = 0.0
        
        return features
    
    def _extract_entity_features(
        self,
        entity: Dict[str, Any]
    ) -> np.ndarray:
        """
        Extract features for an entity node.
        
        Args:
            entity: Entity data dictionary
            
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
        # Coerce to float defensively; upstream callers sometimes provide strings
        try:
            ex = float(entity.get('x', 0.0))
        except Exception as ex_err:
            logging.warning("GraphBuilder: invalid entity x value %r (%s); defaulting to 0.0", entity.get('x', None), ex_err)
            ex = 0.0
        try:
            ey = float(entity.get('y', 0.0))
        except Exception as ey_err:
            logging.warning("GraphBuilder: invalid entity y value %r (%s); defaulting to 0.0", entity.get('y', None), ey_err)
            ey = 0.0
        features[state_offset + 1] = ex / (GRID_WIDTH * CELL_SIZE)  # Normalized x
        features[state_offset + 2] = ey / (GRID_HEIGHT * CELL_SIZE)  # Normalized y
        # Custom state encodes hazard flag for entities
        features[state_offset + 3] = 1.0 if self._is_hazard_entity(entity) else 0.0
        
        # Ninja position flag (always 0 for entity nodes)
        ninja_offset = state_offset + 4
        features[ninja_offset] = 0.0
        
        return features
    
    def _is_sub_cell_traversable(
        self,
        level_data: Dict[str, Any],
        sub_row: int,
        sub_col: int
    ) -> bool:
        """
        Determine if a sub-cell is traversable by the ninja.
        
        Uses a more permissive approach that focuses on the sub-cell center
        and basic collision checks rather than full ninja clearance.
        
        Args:
            level_data: Level data containing tile information
            sub_row: Sub-grid row
            sub_col: Sub-grid column
            
        Returns:
            True if the sub-cell is traversable, False otherwise
        """
        # Convert sub-grid coordinates to world position (center of sub-cell)
        world_x = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        world_y = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        
        # Get the tile that contains this sub-cell
        tile_row = sub_row // SUB_GRID_RESOLUTION
        tile_col = sub_col // SUB_GRID_RESOLUTION
        tile_type = self._get_tile_type(level_data, tile_row, tile_col)
        
        # Solid tiles are never traversable
        if self._is_solid(tile_type):
            return False
        
        # For empty tiles, be much more permissive - just check center point
        if tile_type == 0:
            return not self._point_collides_with_geometry(level_data, world_x, world_y)
        
        # For half-tiles and slopes, check if center point is in open space
        if self._is_half_tile(tile_type):
            return not self._point_in_solid_half(world_x, world_y, tile_row, tile_col, tile_type)
        
        # For slopes, check point collision with slope geometry
        if self._is_slope(tile_type):
            return not self._point_collides_with_slope(world_x, world_y, tile_row, tile_col, tile_type)
        
        # Default to traversable for other tile types
        return True
    
    def _has_ninja_clearance(
        self,
        level_data: Dict[str, Any],
        world_x: float,
        world_y: float,
        ninja_radius: float
    ) -> bool:
        """
        Check if there's enough clearance around a position for the ninja.
        
        This is more permissive than exact collision detection - it checks if there's
        a ninja-diameter (20px) wide area of non-solid space around the position.
        
        Args:
            level_data: Level data
            world_x, world_y: Position to check
            ninja_radius: Ninja radius
            
        Returns:
            True if there's sufficient clearance, False otherwise
        """
        # Use a more game-appropriate clearance check
        # Check if ninja center is safe and there's reasonable clearance
        if self._point_collides_with_geometry(level_data, world_x, world_y):
            return False
        
        # Check 4 cardinal directions at reduced radius for more permissive traversal
        clearance_radius = ninja_radius * 0.5  # More permissive clearance
        test_points = [
            (world_x - clearance_radius, world_y),  # Left
            (world_x + clearance_radius, world_y),  # Right
            (world_x, world_y - clearance_radius),  # Up
            (world_x, world_y + clearance_radius),  # Down
        ]
        
        # Allow traversal if at least 3 out of 4 directions are clear
        clear_directions = 0
        for test_x, test_y in test_points:
            if not self._point_collides_with_geometry(level_data, test_x, test_y):
                clear_directions += 1
        
        return clear_directions >= 3
    
    def _check_ninja_collision_at_position(
        self,
        level_data: Dict[str, Any],
        world_x: float,
        world_y: float,
        ninja_radius: float
    ) -> bool:
        """
        Check if a ninja at the given position would collide with level geometry.
        
        Uses raycasting approach: cast rays from ninja center to points around the
        ninja's circumference to detect collisions with solid geometry.
        
        Args:
            level_data: Level data containing tile information
            world_x: World X coordinate
            world_y: World Y coordinate
            ninja_radius: Ninja collision radius
            
        Returns:
            True if position is safe (no collision), False if collision detected
        """
        # Bounds check - be more lenient for sub-grid positions
        if (world_x < 0 or world_x >= GRID_WIDTH * CELL_SIZE or
            world_y < 0 or world_y >= GRID_HEIGHT * CELL_SIZE):
            return False  # Out of level bounds
        
        # First check ninja center - if it's in a solid tile, definitely collision
        if self._point_collides_with_geometry(level_data, world_x, world_y):
            return False
        
        # For positions clearly in empty tiles, skip expensive raycasting
        center_tile_col = int(world_x // CELL_SIZE)
        center_tile_row = int(world_y // CELL_SIZE)
        center_tile_type = self._get_tile_type(level_data, center_tile_row, center_tile_col)
        
        if center_tile_type == 0:
            # In empty tile - only need to check if ninja extends into adjacent solid tiles
            return self._check_ninja_extends_into_solid_tiles(
                level_data, world_x, world_y, ninja_radius
            )
        
        # For complex tiles (half-tiles, slopes), use raycasting
        num_rays = 8  # Reduced from 16 for performance
        for i in range(num_rays):
            angle = (2 * np.pi * i) / num_rays
            # Point on ninja circumference
            ray_x = world_x + ninja_radius * np.cos(angle)
            ray_y = world_y + ninja_radius * np.sin(angle)
            
            # Check if this point collides with solid geometry
            if self._point_collides_with_geometry(level_data, ray_x, ray_y):
                return False
        
        return True
    
    def _check_ninja_extends_into_solid_tiles(
        self,
        level_data: Dict[str, Any],
        world_x: float,
        world_y: float,
        ninja_radius: float
    ) -> bool:
        """
        Optimized check for ninja in empty tiles - only check adjacent tiles.
        
        Args:
            level_data: Level data
            world_x, world_y: Ninja center position
            ninja_radius: Ninja radius
            
        Returns:
            True if safe (no collision), False if ninja extends into solid tiles
        """
        center_tile_col = int(world_x // CELL_SIZE)
        center_tile_row = int(world_y // CELL_SIZE)
        
        # Check the 8 adjacent tiles to see if ninja extends into any solid ones
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # Skip center tile (already checked)
                
                adj_row = center_tile_row + dr
                adj_col = center_tile_col + dc
                
                # Skip out of bounds
                if adj_row < 0 or adj_row >= GRID_HEIGHT or adj_col < 0 or adj_col >= GRID_WIDTH:
                    continue
                
                adj_tile_type = self._get_tile_type(level_data, adj_row, adj_col)
                
                # Skip empty tiles
                if adj_tile_type == 0:
                    continue
                
                # Check if ninja circle extends into this tile
                adj_tile_x = adj_col * CELL_SIZE
                adj_tile_y = adj_row * CELL_SIZE
                
                # Distance from ninja center to tile bounds
                closest_x = max(adj_tile_x, min(world_x, adj_tile_x + CELL_SIZE))
                closest_y = max(adj_tile_y, min(world_y, adj_tile_y + CELL_SIZE))
                
                dist_sq = (world_x - closest_x) ** 2 + (world_y - closest_y) ** 2
                
                if dist_sq <= ninja_radius ** 2:
                    # Ninja extends into this tile - check if it's solid
                    if self._is_solid(adj_tile_type):
                        # For solid tiles, use full radius check to be strict about collisions
                        return False
                    elif self._is_half_tile(adj_tile_type) or self._is_slope(adj_tile_type):
                        # For complex tiles, do detailed check
                        if self._ninja_intersects_tile_geometry(
                            world_x, world_y, ninja_radius, adj_row, adj_col, adj_tile_type
                        ):
                            return False
        
        return True
    
    def _point_collides_with_geometry(
        self,
        level_data: Dict[str, Any],
        world_x: float,
        world_y: float
    ) -> bool:
        """
        Check if a single point collides with solid level geometry.
        
        Args:
            level_data: Level data containing tile information
            world_x: World X coordinate
            world_y: World Y coordinate
            
        Returns:
            True if point collides with solid geometry, False otherwise
        """
        # Get the tile containing this point
        tile_col = int(world_x // CELL_SIZE)
        tile_row = int(world_y // CELL_SIZE)
        
        # Bounds check
        if (tile_row < 0 or tile_row >= GRID_HEIGHT or 
            tile_col < 0 or tile_col >= GRID_WIDTH):
            return True  # Out of bounds = collision
        
        tile_type = self._get_tile_type(level_data, tile_row, tile_col)
        
        # Empty tiles never collide
        if tile_type == 0:
            return False
            
        # Solid tiles always collide
        if self._is_solid(tile_type):
            return True
        
        # For half-tiles, check which half the point is in
        if self._is_half_tile(tile_type):
            return self._point_in_solid_half(world_x, world_y, tile_row, tile_col, tile_type)
        
        # For slopes, use more detailed collision detection
        if self._is_slope(tile_type):
            return self._point_collides_with_slope(world_x, world_y, tile_row, tile_col, tile_type)
        
        # Default: no collision
        return False
    
    def _point_in_solid_half(
        self,
        world_x: float,
        world_y: float,
        tile_row: int,
        tile_col: int,
        tile_type: int
    ) -> bool:
        """
        Check if a point is in the solid half of a half-tile.
        
        Args:
            world_x, world_y: Point coordinates
            tile_row, tile_col: Tile coordinates
            tile_type: Half-tile type (2-5)
            
        Returns:
            True if point is in solid half, False otherwise
        """
        # Get tile world position
        tile_x = tile_col * CELL_SIZE
        tile_y = tile_row * CELL_SIZE
        
        # Point position relative to tile center
        rel_x = world_x - (tile_x + CELL_SIZE / 2)
        rel_y = world_y - (tile_y + CELL_SIZE / 2)
        
        # Half-tile collision based on tile type:
        if tile_type == 2:  # Top half solid
            return rel_y <= 0
        elif tile_type == 3:  # Right half solid
            return rel_x >= 0
        elif tile_type == 4:  # Bottom half solid
            return rel_y >= 0
        elif tile_type == 5:  # Left half solid
            return rel_x <= 0
        
        return False
    
    def _point_collides_with_slope(
        self,
        world_x: float,
        world_y: float,
        tile_row: int,
        tile_col: int,
        tile_type: int
    ) -> bool:
        """
        Check if a point collides with a slope tile.
        
        Uses the tile's diagonal segment definitions for accurate collision.
        
        Args:
            world_x, world_y: Point coordinates
            tile_row, tile_col: Tile coordinates
            tile_type: Slope tile type
            
        Returns:
            True if point collides with solid part of slope, False otherwise
        """
        from ..tile_definitions import TILE_SEGMENT_DIAG_MAP
        
        # Get tile world position
        tile_x = tile_col * CELL_SIZE
        tile_y = tile_row * CELL_SIZE
        
        # Point position relative to tile origin
        rel_x = world_x - tile_x
        rel_y = world_y - tile_y
        
        # Check if this tile type has diagonal segments
        if tile_type in TILE_SEGMENT_DIAG_MAP:
            # Get diagonal segment definition
            (x1, y1), (x2, y2) = TILE_SEGMENT_DIAG_MAP[tile_type]
            
            # Determine which side of the diagonal line the point is on
            # Line equation: (y2-y1)*(x-x1) - (x2-x1)*(y-y1) = 0
            # Positive/negative indicates which side
            line_val = (y2 - y1) * (rel_x - x1) - (x2 - x1) * (rel_y - y1)
            
            # For 45-degree slopes (6-9), the solid region is typically below/right of the line
            if 6 <= tile_type <= 9:
                if tile_type == 6:  # Top-left to bottom-right diagonal
                    return line_val <= 0  # Below the line
                elif tile_type == 7:  # Top-left to bottom-right diagonal (different orientation)
                    return line_val >= 0  # Above the line
                elif tile_type == 8:  # Bottom-left to top-right diagonal
                    return line_val >= 0  # Above the line
                elif tile_type == 9:  # Bottom-left to top-right diagonal (different orientation)
                    return line_val <= 0  # Below the line
        
        # For other slopes, use edge-based detection as fallback
        return self._ninja_intersects_tile_geometry(
            world_x, world_y, 1.0, tile_row, tile_col, tile_type
        )
    
    def _ninja_intersects_tile_geometry(
        self,
        ninja_x: float,
        ninja_y: float,
        ninja_radius: float,
        tile_row: int,
        tile_col: int,
        tile_type: int
    ) -> bool:
        """
        Check if ninja circle intersects with specific tile geometry.
        
        This uses a simplified approach: for half-tiles, check if the ninja
        is in the solid half of the tile based on the tile type.
        
        Args:
            ninja_x, ninja_y: Ninja center position
            ninja_radius: Ninja collision radius
            tile_row, tile_col: Tile grid coordinates
            tile_type: Tile type ID
            
        Returns:
            True if collision detected, False otherwise
        """
        # Get tile world position
        tile_x = tile_col * CELL_SIZE
        tile_y = tile_row * CELL_SIZE
        
        # For half-tiles, use simplified geometry check
        if self._is_half_tile(tile_type):
            # Get ninja position relative to tile center
            rel_x = ninja_x - (tile_x + CELL_SIZE / 2)
            rel_y = ninja_y - (tile_y + CELL_SIZE / 2)
            
            # Half-tile collision based on tile type:
            # Type 2: Top half solid
            # Type 3: Right half solid  
            # Type 4: Bottom half solid
            # Type 5: Left half solid
            
            if tile_type == 2:  # Top half solid
                return rel_y <= 0  # Ninja in upper half
            elif tile_type == 3:  # Right half solid
                return rel_x >= 0  # Ninja in right half
            elif tile_type == 4:  # Bottom half solid
                return rel_y >= 0  # Ninja in lower half
            elif tile_type == 5:  # Left half solid
                return rel_x <= 0  # Ninja in left half
        
        # For slopes, use edge-based detection
        elif self._is_slope(tile_type):
            # Get tile edge map
            edges = TILE_GRID_EDGE_MAP.get(tile_type, [0] * 12)
            
            # Check collision with each edge segment that exists
            edge_segments = []
            
            # Horizontal edges: [top-L, top-R, mid-L, mid-R, bot-L, bot-R]
            if edges[0]: edge_segments.append((tile_x, tile_y, tile_x + CELL_SIZE//2, tile_y))
            if edges[1]: edge_segments.append((tile_x + CELL_SIZE//2, tile_y, tile_x + CELL_SIZE, tile_y))
            if edges[2]: edge_segments.append((tile_x, tile_y + CELL_SIZE//2, tile_x + CELL_SIZE//2, tile_y + CELL_SIZE//2))
            if edges[3]: edge_segments.append((tile_x + CELL_SIZE//2, tile_y + CELL_SIZE//2, tile_x + CELL_SIZE, tile_y + CELL_SIZE//2))
            if edges[4]: edge_segments.append((tile_x, tile_y + CELL_SIZE, tile_x + CELL_SIZE//2, tile_y + CELL_SIZE))
            if edges[5]: edge_segments.append((tile_x + CELL_SIZE//2, tile_y + CELL_SIZE, tile_x + CELL_SIZE, tile_y + CELL_SIZE))
            
            # Vertical edges: [left-T, left-B, mid-T, mid-B, right-T, right-B]
            if edges[6]: edge_segments.append((tile_x, tile_y, tile_x, tile_y + CELL_SIZE//2))
            if edges[7]: edge_segments.append((tile_x, tile_y + CELL_SIZE//2, tile_x, tile_y + CELL_SIZE))
            if edges[8]: edge_segments.append((tile_x + CELL_SIZE//2, tile_y, tile_x + CELL_SIZE//2, tile_y + CELL_SIZE//2))
            if edges[9]: edge_segments.append((tile_x + CELL_SIZE//2, tile_y + CELL_SIZE//2, tile_x + CELL_SIZE//2, tile_y + CELL_SIZE))
            if edges[10]: edge_segments.append((tile_x + CELL_SIZE, tile_y, tile_x + CELL_SIZE, tile_y + CELL_SIZE//2))
            if edges[11]: edge_segments.append((tile_x + CELL_SIZE, tile_y + CELL_SIZE//2, tile_x + CELL_SIZE, tile_y + CELL_SIZE))
            
            # Check collision with each edge segment
            for x1, y1, x2, y2 in edge_segments:
                if self._circle_intersects_line_segment(ninja_x, ninja_y, ninja_radius, x1, y1, x2, y2):
                    return True
                
        return False
    
    def _circle_intersects_line_segment(
        self,
        cx: float, cy: float, r: float,
        x1: float, y1: float, x2: float, y2: float
    ) -> bool:
        """
        Check if a circle intersects with a line segment.
        
        Args:
            cx, cy: Circle center
            r: Circle radius  
            x1, y1, x2, y2: Line segment endpoints
            
        Returns:
            True if intersection exists, False otherwise
        """
        # Vector from line start to circle center
        dx = cx - x1
        dy = cy - y1
        
        # Line segment vector
        lx = x2 - x1
        ly = y2 - y1
        
        # Line segment length squared
        len_sq = lx * lx + ly * ly
        
        if len_sq == 0:
            # Degenerate segment - check distance to point
            return dx * dx + dy * dy <= r * r
        
        # Project circle center onto line segment
        t = max(0, min(1, (dx * lx + dy * ly) / len_sq))
        
        # Closest point on segment to circle center
        closest_x = x1 + t * lx
        closest_y = y1 + t * ly
        
        # Distance squared from circle center to closest point
        dist_sq = (cx - closest_x) ** 2 + (cy - closest_y) ** 2
        
        return dist_sq <= r * r
    
    def _determine_sub_cell_traversability(
        self,
        level_data: Dict[str, Any],
        src_sub_row: int,
        src_sub_col: int,
        tgt_sub_row: int,
        tgt_sub_col: int,
        one_way_index: List[Dict[str, Any]],
        door_blockers: List[Dict[str, Any]],
        dr: int,
        dc: int,
        ninja_velocity: Optional[Tuple[float, float]] = None,
        ninja_state: Optional[int] = None
    ) -> Tuple[Optional[EdgeType], float, Optional[dict]]:
        """
        Determine if two sub-grid cells are connected and how.
        
        This improved method uses sub-grid resolution and proper collision detection
        to determine traversability between adjacent sub-cells, with physics-based
        trajectory validation.
        
        Args:
            level_data: Level data containing tile information
            src_sub_row, src_sub_col: Source sub-cell coordinates
            tgt_sub_row, tgt_sub_col: Target sub-cell coordinates  
            one_way_index: List of one-way platform data
            door_blockers: List of door blocker data
            dr, dc: Direction vector (sub-cell delta)
            ninja_velocity: Current ninja velocity for physics calculations
            ninja_state: Current ninja movement state
            
        Returns:
            (EdgeType, cost, trajectory_info) if traversable, (None, 0.0, None) if blocked
        """
        from ..constants import NINJA_RADIUS
        
        # First check if target sub-cell is traversable
        if not self._is_sub_cell_traversable(level_data, tgt_sub_row, tgt_sub_col):
            return None, 0.0, None
        
        # Check if source sub-cell is traversable (should be, but be safe)
        if not self._is_sub_cell_traversable(level_data, src_sub_row, src_sub_col):
            return None, 0.0, None
            
        # Convert sub-cell coordinates to world positions
        src_world_x = src_sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        src_world_y = src_sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        tgt_world_x = tgt_sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        tgt_world_y = tgt_sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        
        # Check for obstacles along the path using swept collision
        if not self._check_path_clear(
            level_data, src_world_x, src_world_y, tgt_world_x, tgt_world_y, NINJA_RADIUS
        ):
            return None, 0.0, None
        
        # One-way platform directional blocking
        if self._is_sub_cell_blocked_by_one_way(
            one_way_index, src_sub_row, src_sub_col, tgt_sub_row, tgt_sub_col
        ):
            return None, 0.0, None

        # Door blockers crossing the path
        if self._is_sub_cell_blocked_by_door(
            door_blockers, src_sub_row, src_sub_col, tgt_sub_row, tgt_sub_col
        ):
            return None, 0.0, None

        # Determine movement type and cost based on direction
        is_diagonal = (abs(dr) + abs(dc)) > 1
        
        # Calculate trajectory information if trajectory calculator is available
        trajectory_info = None
        if self.trajectory_calc is not None and ninja_velocity is not None:
            try:
                # Calculate trajectory parameters
                dx_world = tgt_world_x - src_world_x
                dy_world = tgt_world_y - src_world_y
                
                # Classify movement type using movement classifier
                movement_type = None
                if self.movement_classifier is not None:
                    movement_type = self.movement_classifier.classify_movement(
                        ninja_velocity, ninja_state, dx_world, dy_world
                    )
                
                # Calculate trajectory for jump/fall movements
                if movement_type in ['JUMP', 'FALL', 'WALL_JUMP']:
                    trajectory = self.trajectory_calc.calculate_jump_trajectory(
                        src_world_x, src_world_y, ninja_velocity[0], ninja_velocity[1]
                    )
                    
                    # Validate trajectory clearance
                    success_prob = 1.0
                    if trajectory:
                        success_prob = self.trajectory_calc.validate_trajectory_clearance(
                            trajectory, level_data, tgt_world_x, tgt_world_y
                        )
                    
                    # Calculate physics parameters
                    distance = math.sqrt(dx_world**2 + dy_world**2)
                    time_of_flight = distance / max(abs(ninja_velocity[0]), 1.0) if ninja_velocity[0] != 0 else 1.0
                    energy_cost = abs(dy_world) * 0.1 + distance * 0.05  # Simplified energy model
                    
                    trajectory_info = {
                        'time_of_flight': time_of_flight,
                        'energy_cost': energy_cost,
                        'success_probability': success_prob,
                        'min_velocity': max(abs(dx_world) / time_of_flight, 1.0),
                        'max_velocity': max(abs(dx_world) / time_of_flight * 2, 10.0),
                        'requires_jump': movement_type in ['JUMP', 'WALL_JUMP'],
                        'requires_wall_contact': movement_type in ['WALL_SLIDE', 'WALL_JUMP']
                    }
                else:
                    # Simple movement (walk, etc.)
                    distance = math.sqrt(dx_world**2 + dy_world**2)
                    trajectory_info = {
                        'time_of_flight': distance / 100.0,  # Assume 100 pixels/time unit for walking
                        'energy_cost': distance * 0.01,
                        'success_probability': 0.95,  # High success for simple movements
                        'min_velocity': 50.0,
                        'max_velocity': 150.0,
                        'requires_jump': False,
                        'requires_wall_contact': False
                    }
            except Exception as e:
                logging.debug(f"Trajectory calculation failed: {e}")
                trajectory_info = None
        
        if is_diagonal:
            # Diagonal movement - check if it's a valid corner cut
            if not self._is_diagonal_traversable(
                level_data, src_sub_row, src_sub_col, tgt_sub_row, tgt_sub_col
            ):
                return None, 0.0, None
            
            # Diagonal movement cost is higher
            if dr > 0:  # Moving down diagonally
                return EdgeType.FALL, 1.2, trajectory_info
            elif dr < 0:  # Moving up diagonally  
                return EdgeType.JUMP, 2.5, trajectory_info
            else:  # Horizontal diagonal (shouldn't happen with our directions)
                return EdgeType.WALK, 1.4, trajectory_info
        else:
            # Cardinal movement
            if dr == 0:  # Horizontal
                return EdgeType.WALK, 1.0, trajectory_info
            elif dr > 0:  # Moving down
                return EdgeType.FALL, 0.8, trajectory_info
            else:  # Moving up
                return EdgeType.JUMP, 2.0, trajectory_info
    
    def _check_path_clear(
        self,
        level_data: Dict[str, Any],
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        ninja_radius: float
    ) -> bool:
        """
        Check if the path between two points is clear of obstacles.
        
        Uses a much more permissive approach for sub-cell connections,
        focusing on point-based collision rather than full ninja collision.
        
        Args:
            level_data: Level data
            src_x, src_y: Source world coordinates
            tgt_x, tgt_y: Target world coordinates
            ninja_radius: Ninja collision radius (ignored for simplicity)
            
        Returns:
            True if path is clear, False if blocked
        """
        # Calculate path distance
        path_distance = np.hypot(tgt_x - src_x, tgt_y - src_y)
        
        # For adjacent sub-cells, use very simple check
        if path_distance <= SUB_CELL_SIZE * 1.5:  # Adjacent or near-adjacent
            # Just check that both endpoints are clear and no solid wall between
            if (self._point_collides_with_geometry(level_data, src_x, src_y) or
                self._point_collides_with_geometry(level_data, tgt_x, tgt_y)):
                return False
            
            # Check midpoint for any obvious solid barriers
            midpoint_x = (src_x + tgt_x) / 2
            midpoint_y = (src_y + tgt_y) / 2
            return not self._point_collides_with_geometry(level_data, midpoint_x, midpoint_y)
        
        # For longer paths, sample more conservatively but still use point-based checks
        num_samples = max(3, min(8, int(path_distance / SUB_CELL_SIZE)))
        
        for i in range(num_samples + 1):
            t = i / num_samples if num_samples > 0 else 0
            sample_x = src_x + t * (tgt_x - src_x)
            sample_y = src_y + t * (tgt_y - src_y)
            
            if self._point_collides_with_geometry(level_data, sample_x, sample_y):
                return False
                
        return True
    
    def _is_diagonal_traversable(
        self,
        level_data: Dict[str, Any],
        src_sub_row: int,
        src_sub_col: int,
        tgt_sub_row: int,
        tgt_sub_col: int
    ) -> bool:
        """
        Check if diagonal movement between sub-cells is valid.
        
        With higher resolution sub-cells, we can be more permissive about diagonal movement.
        We only require that at least one of the adjacent cardinal directions is traversable.
        
        Args:
            level_data: Level data
            src_sub_row, src_sub_col: Source sub-cell
            tgt_sub_row, tgt_sub_col: Target sub-cell
            
        Returns:
            True if diagonal movement is valid, False otherwise
        """
        # For diagonal movement, check the two adjacent cardinal directions
        mid1_row = src_sub_row
        mid1_col = tgt_sub_col
        mid2_row = tgt_sub_row  
        mid2_col = src_sub_col
        
        # At least one intermediate position must be traversable (more permissive)
        return (self._is_sub_cell_traversable(level_data, mid1_row, mid1_col) or
                self._is_sub_cell_traversable(level_data, mid2_row, mid2_col))
    
    def _is_sub_cell_blocked_by_one_way(
        self,
        one_way_index: List[Dict[str, Any]],
        src_sub_row: int,
        src_sub_col: int,
        tgt_sub_row: int,
        tgt_sub_col: int,
    ) -> bool:
        """
        Check if one-way platform blocks movement between sub-cells.
        
        Args:
            one_way_index: List of one-way platform data
            src_sub_row, src_sub_col: Source sub-cell
            tgt_sub_row, tgt_sub_col: Target sub-cell
            
        Returns:
            True if blocked by one-way platform, False otherwise
        """
        if not one_way_index:
            return False

        # Convert to world coordinates
        src_cx = src_sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5
        src_cy = src_sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5
        tgt_cx = tgt_sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5
        tgt_cy = tgt_sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5

        # Movement direction (normalized)
        mvx = tgt_cx - src_cx
        mvy = tgt_cy - src_cy
        norm = np.hypot(mvx, mvy)
        if norm == 0:
            return False
        mvx /= norm
        mvy /= norm

        # Use the actual EntityOneWayPlatform.SEMI_SIDE value
        from ..entity_classes.entity_one_way_platform import EntityOneWayPlatform
        semi_side = EntityOneWayPlatform.SEMI_SIDE  # Should be 12

        # Check path against one-way platforms
        for ow in one_way_index:
            px, py = ow['x'], ow['y']
            nx, ny = ow['nx'], ow['ny']
            
            # Check if platform is near the movement path
            # Distance from platform center to line segment (src -> tgt)
            dx = px - src_cx
            dy = py - src_cy
            
            # Project onto movement vector
            proj = dx * mvx + dy * mvy
            
            # Closest point on path to platform
            if proj < 0:
                closest_x, closest_y = src_cx, src_cy
            elif proj > norm:
                closest_x, closest_y = tgt_cx, tgt_cy
            else:
                closest_x = src_cx + proj * mvx
                closest_y = src_cy + proj * mvy
            
            # Distance from platform to closest point on path
            dist_sq = (px - closest_x) ** 2 + (py - closest_y) ** 2
            
            if dist_sq <= semi_side ** 2:
                # Platform is close to path - check direction
                if mvx * nx + mvy * ny < 0.0:  # Moving against platform normal
                    return True
                    
        return False
    
    def _is_sub_cell_blocked_by_door(
        self,
        door_blockers: List[Dict[str, Any]],
        src_sub_row: int,
        src_sub_col: int,
        tgt_sub_row: int,
        tgt_sub_col: int,
    ) -> bool:
        """
        Check if door blocks movement between sub-cells.
        
        Args:
            door_blockers: List of door blocker data
            src_sub_row, src_sub_col: Source sub-cell  
            tgt_sub_row, tgt_sub_col: Target sub-cell
            
        Returns:
            True if blocked by door, False otherwise
        """
        if not door_blockers:
            return False

        # Convert to world coordinates
        src_x = src_sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5
        src_y = src_sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5
        tgt_x = tgt_sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5
        tgt_y = tgt_sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5

        # Check each door blocker
        from ..constants import NINJA_RADIUS
        MARGIN = NINJA_RADIUS // 2  # Smaller margin for sub-cells
        
        for d in door_blockers:
            px, py, r = d['x'], d['y'], d['r']
            expanded_r_sq = (r + MARGIN) ** 2
            
            # Distance from door center to line segment (src -> tgt)
            if self._point_to_line_segment_distance_sq(px, py, src_x, src_y, tgt_x, tgt_y) <= expanded_r_sq:
                return True
                
        return False
    
    def _point_to_line_segment_distance_sq(
        self,
        px: float, py: float,
        x1: float, y1: float, x2: float, y2: float
    ) -> float:
        """
        Calculate squared distance from point to line segment.
        
        Args:
            px, py: Point coordinates
            x1, y1, x2, y2: Line segment endpoints
            
        Returns:
            Squared distance from point to line segment
        """
        # Vector from segment start to point
        dx = px - x1
        dy = py - y1
        
        # Segment vector
        sx = x2 - x1
        sy = y2 - y1
        
        # Segment length squared
        seg_len_sq = sx * sx + sy * sy
        
        if seg_len_sq == 0:
            # Degenerate segment
            return dx * dx + dy * dy
        
        # Project point onto segment
        t = max(0, min(1, (dx * sx + dy * sy) / seg_len_sq))
        
        # Closest point on segment
        closest_x = x1 + t * sx
        closest_y = y1 + t * sy
        
        # Distance squared
        dist_x = px - closest_x
        dist_y = py - closest_y
        return dist_x * dist_x + dist_y * dist_y

    def _determine_traversability(
        self,
        level_data: Dict[str, Any],
        src_row: int,
        src_col: int,
        tgt_row: int,
        tgt_col: int,
        one_way_index: List[Dict[str, Any]],
        door_blockers: List[Dict[str, Any]]
    ) -> Tuple[Optional[EdgeType], float]:
        """
        Legacy traversability method - kept for compatibility.
        
        This method is now deprecated in favor of _determine_sub_cell_traversability.
        """
        # Simple fallback implementation
        src_tile = self._get_tile_type(level_data, src_row, src_col)
        tgt_tile = self._get_tile_type(level_data, tgt_row, tgt_col)

        # Disallow entering entirely solid tiles (tile id 1)
        if self._is_solid(tgt_tile):
            return None, 0.0

        dr = tgt_row - src_row
        dc = tgt_col - src_col
        
        # Simple movement semantics
        if dr == 0:  # Horizontal
            return EdgeType.WALK, 1.0
        elif dr > 0:  # Moving down
            return EdgeType.FALL, 0.8
        else:  # Moving up
            return EdgeType.JUMP, 2.0

    def _collect_door_blockers(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collect circular blockers representing doors that can block traversal.

        We treat:
        - Locked doors (type 6 or 'locked_door'): block while active is True (closed)
        - Trap doors (type 8 or 'trap_door'): block when closed; in simulator, this
          corresponds to active == False after being triggered. We also accept an
          explicit 'closed' flag if present.
        """
        blockers: List[Dict[str, Any]] = []
        for e in entities:
            et = e.get('type', '')
            et_num = None
            if isinstance(et, int):
                et_num = et
            et_str = str(et).lower()

            # Default door radius from entity definitions (5 pixels)
            radius = float(e.get('radius', 5.0))
            try:
                x = float(e.get('x', 0.0))
                y = float(e.get('y', 0.0))
            except Exception:
                x, y = 0.0, 0.0

            # Locked door
            if (et_num == 6) or ('locked_door' in et_str):
                # The door entity itself has its center at segment midpoint; switch at x/y
                cx = float(e.get('door_x', e.get('x', 0.0)))
                cy = float(e.get('door_y', e.get('y', 0.0)))
                # Consider both 'closed' flag and 'active' (true when still closed)
                is_blocking = bool(e.get('closed', e.get('active', True)))
                if is_blocking:
                    blockers.append({'x': cx, 'y': cy, 'r': radius, 'kind': 'locked'})
                continue

            # Trap door
            if (et_num == 8) or ('trap_door' in et_str):
                cx = float(e.get('door_x', e.get('x', 0.0)))
                cy = float(e.get('door_y', e.get('y', 0.0)))
                closed_flag = bool(e.get('closed', False))
                # In our sim: after closing, active becomes False
                is_blocking = closed_flag or (e.get('active', True) is False)
                if is_blocking:
                    blockers.append({'x': cx, 'y': cy, 'r': radius, 'kind': 'trap'})
                continue
        return blockers

    def _is_blocked_by_door(
        self,
        door_blockers: List[Dict[str, Any]],
        src_row: int,
        src_col: int,
        tgt_row: int,
        tgt_col: int,
    ) -> bool:
        """Return True if a door circle blocks movement from source cell to target cell.

        Previous implementation only checked if the door overlapped the shared border, which
        misses the common case where the door is centered inside a narrow corridor cell.
        We now test the circle against the segment that connects the two cell centers, which
        approximates the path used by our coarse grid edges.
        """
        if not door_blockers:
            return False

        # Segment endpoints: centers of source and target cells
        x0 = src_col * CELL_SIZE + CELL_SIZE * 0.5
        y0 = src_row * CELL_SIZE + CELL_SIZE * 0.5
        x1 = tgt_col * CELL_SIZE + CELL_SIZE * 0.5
        y1 = tgt_row * CELL_SIZE + CELL_SIZE * 0.5

        # Helper: squared distance from point P to segment AB
        def dist2_point_to_seg(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
            abx = bx - ax
            aby = by - ay
            apx = px - ax
            apy = py - ay
            denom = abx * abx + aby * aby
            if denom == 0.0:
                # Degenerate; treat as point distance
                return apx * apx + apy * apy
            t = (apx * abx + apy * aby) / denom
            if t < 0.0:
                cx, cy = ax, ay
            elif t > 1.0:
                cx, cy = bx, by
            else:
                cx = ax + t * abx
                cy = ay + t * aby
            dx = px - cx
            dy = py - cy
            return dx * dx + dy * dy

        # Small margin to approximate ninja + clearance through corridors
        MARGIN = 6.0
        for d in door_blockers:
            px, py, r = d['x'], d['y'], d['r']
            rr = (r + MARGIN) * (r + MARGIN)
            if dist2_point_to_seg(px, py, x0, y0, x1, y1) <= rr:
                return True
        return False

    def _collect_one_way_platforms(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract one-way platform descriptors with world position and normal.

        Each entry contains: {'x': float, 'y': float, 'nx': float, 'ny': float}
        """
        collected: List[Dict[str, Any]] = []
        for e in entities:
            if self._is_one_way_entity(e):
                try:
                    x = float(e.get('x', 0.0))
                    y = float(e.get('y', 0.0))
                except Exception:
                    x, y = 0.0, 0.0
                orientation = e.get('orientation', None)
                nx, ny = self._orientation_to_vector(orientation)
                collected.append({'x': x, 'y': y, 'nx': nx, 'ny': ny})
        return collected

    def _is_one_way_entity(self, entity: Dict[str, Any]) -> bool:
        et = entity.get('type', '')
        if isinstance(et, int):
            return et == 11
        return 'one_way' in str(et).lower()

    def _orientation_to_vector(self, orientation: Optional[int]) -> Tuple[float, float]:
        """Map orientation (0-7) to normalized vector. Fallback to (1,0)."""
        # sqrt(2)/2 for diagonals
        diag = np.sqrt(2.0) / 2.0
        mapping = {
            0: (1.0, 0.0),
            1: (diag, diag),
            2: (0.0, 1.0),
            3: (-diag, diag),
            4: (-1.0, 0.0),
            5: (-diag, -diag),
            6: (0.0, -1.0),
            7: (diag, -diag),
        }
        try:
            key = int(orientation) if orientation is not None else 0
        except Exception:
            key = 0
        return mapping.get(key, (1.0, 0.0))

    def _is_blocked_by_one_way(
        self,
        one_way_index: List[Dict[str, Any]],
        src_row: int,
        src_col: int,
        tgt_row: int,
        tgt_col: int,
    ) -> bool:
        """Return True if a one-way platform blocks movement across the shared border.

        Heuristic: A one-way platform near the shared border blocks crossing if
        the movement direction points against the platform normal (dot < 0).
        """
        if not one_way_index:
            return False

        # Centers of source and target cells
        src_cx = src_col * CELL_SIZE + CELL_SIZE * 0.5
        src_cy = src_row * CELL_SIZE + CELL_SIZE * 0.5
        tgt_cx = tgt_col * CELL_SIZE + CELL_SIZE * 0.5
        tgt_cy = tgt_row * CELL_SIZE + CELL_SIZE * 0.5

        # Movement direction (normalized): axis-aligned for neighbors
        mvx = tgt_cx - src_cx
        mvy = tgt_cy - src_cy
        norm = np.hypot(mvx, mvy)
        if norm == 0:
            return False
        mvx /= norm
        mvy /= norm

        # Border line coordinate and proximity check
        min_col = min(src_col, tgt_col)
        min_row = min(src_row, tgt_row)
        semi_side = CELL_SIZE * 0.5  # Same as EntityOneWayPlatform.SEMI_SIDE

        if src_row == tgt_row:
            # Vertical border at x = (min_col + 1) * CELL_SIZE
            border_x = (min_col + 1) * CELL_SIZE
            y0 = min_row * CELL_SIZE - semi_side
            y1 = (min_row + 1) * CELL_SIZE + semi_side
            for ow in one_way_index:
                px, py = ow['x'], ow['y']
                if abs(px - border_x) <= semi_side and y0 <= py <= y1:
                    # Directional check
                    nx, ny = ow['nx'], ow['ny']
                    if mvx * nx + mvy * ny < 0.0:
                        return True
            return False
        else:
            # Horizontal border at y = (min_row + 1) * CELL_SIZE
            border_y = (min_row + 1) * CELL_SIZE
            x0 = min_col * CELL_SIZE - semi_side
            x1 = (min_col + 1) * CELL_SIZE + semi_side
            for ow in one_way_index:
                px, py = ow['x'], ow['y']
                if abs(py - border_y) <= semi_side and x0 <= px <= x1:
                    nx, ny = ow['nx'], ow['ny']
                    if mvx * nx + mvy * ny < 0.0:
                        return True
            return False
    
    def _build_functional_edges(
        self,
        entity_nodes: List[Tuple[int, Dict[str, Any]]],
        sub_grid_node_map: Dict[Tuple[int, int], int]
    ) -> List[Tuple[int, int, EdgeType, float, float, float, Optional[dict]]]:
        """
        Build functional relationship edges between entities.
        
        Args:
            entity_nodes: List of (node_index, entity_data) tuples
            sub_grid_node_map: Mapping from sub-grid coordinates to node indices
            
        Returns:
            List of edge tuples (src, tgt, type, cost, dx, dy)
        """
        edges = []
        
        # Utility to find sub-grid node at world position
        def sub_grid_node_for_xy(x: float, y: float) -> Optional[int]:
            sub_col = int(x // SUB_CELL_SIZE)
            sub_row = int(y // SUB_CELL_SIZE)
            return sub_grid_node_map.get((sub_row, sub_col))

        # Buckets
        exit_switches: List[Tuple[int, Dict[str, Any]]] = []
        exit_doors: List[Tuple[int, Dict[str, Any]]] = []
        regular_doors: List[Tuple[int, Dict[str, Any]]] = []
        locked_doors: List[Tuple[int, Dict[str, Any]]] = []
        trap_doors: List[Tuple[int, Dict[str, Any]]] = []
        door_segments: List[Tuple[int, Dict[str, Any]]] = []

        def is_type(e: Dict[str, Any], kinds: Tuple) -> bool:
            t = e.get('type', '')
            if isinstance(t, int):
                return t in kinds
            lt = str(t).lower()
            # Only perform substring checks for string kind entries; ignore numeric kinds here
            return any((isinstance(k, str) and k in lt) for k in kinds)

        for node_idx, entity in entity_nodes:
            if is_type(entity, (4, 'exit_switch')):
                exit_switches.append((node_idx, entity))
            elif is_type(entity, (3, 'exit_door')):
                exit_doors.append((node_idx, entity))
            elif is_type(entity, (5, 'regular_door')):
                regular_doors.append((node_idx, entity))
            elif is_type(entity, (6, 'locked_door')):
                locked_doors.append((node_idx, entity))
            elif is_type(entity, (8, 'trap_door')):
                trap_doors.append((node_idx, entity))
            elif is_type(entity, ('door_segment_locked', 'door_segment_trap')):
                door_segments.append((node_idx, entity))

        # 1) Exit: connect each exit switch to its nearest exit door
        for sw_idx, sw in exit_switches:
            sx, sy = sw.get('x', 0.0), sw.get('y', 0.0)
            if not exit_doors:
                continue
            # Choose nearest exit door spatially
            best = min(exit_doors, key=lambda it: (it[1].get('x', 0.0) - sx) ** 2 + (it[1].get('y', 0.0) - sy) ** 2)
            door_idx, door = best
            dx = door.get('x', 0.0) - sx
            dy = door.get('y', 0.0) - sy
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > 0:
                dx /= dist
                dy /= dist
            edges.append((sw_idx, door_idx, EdgeType.FUNCTIONAL, 1.0, dx, dy, None))

        # 2) Locked/Trap doors: link from switch entities -> door segment entities
        for switch_idx, switch_entity in locked_doors + trap_doors:
            sx, sy = switch_entity.get('x', 0.0), switch_entity.get('y', 0.0)
            # Find matching door segment
            for segment_idx, segment_entity in door_segments:
                parent_x = segment_entity.get('parent_switch_x', 0.0)
                parent_y = segment_entity.get('parent_switch_y', 0.0)
                # Check if this segment belongs to this switch
                if abs(parent_x - sx) < 1.0 and abs(parent_y - sy) < 1.0:
                    dx = segment_entity.get('x', 0.0) - sx
                    dy = segment_entity.get('y', 0.0) - sy
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist > 0:
                        dx /= dist
                        dy /= dist
                    edges.append((switch_idx, segment_idx, EdgeType.FUNCTIONAL, 1.0, dx, dy, None))
                    break

        # 3) Regular doors: add proximity activation edges from nearby sub-grid cells
        # Use activation radius consistent with EntityDoorRegular.RADIUS (10)
        activation_radius = 10.0
        for door_idx, door in regular_doors:
            cx = float(door.get('x', 0.0))
            cy = float(door.get('y', 0.0))
            # Determine affected sub-cell range
            min_sc = int(max(0, (cx - activation_radius) // SUB_CELL_SIZE))
            max_sc = int(min(SUB_GRID_WIDTH - 1, (cx + activation_radius) // SUB_CELL_SIZE))
            min_sr = int(max(0, (cy - activation_radius) // SUB_CELL_SIZE))
            max_sr = int(min(SUB_GRID_HEIGHT - 1, (cy + activation_radius) // SUB_CELL_SIZE))
            for sub_row in range(min_sr, max_sr + 1):
                for sub_col in range(min_sc, max_sc + 1):
                    node = sub_grid_node_map.get((sub_row, sub_col))
                    if node is None:
                        continue
                    # Use sub-cell center for direction
                    px = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5
                    py = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE * 0.5
                    dx = cx - px
                    dy = cy - py
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist <= activation_radius and dist > 0:
                        edges.append((node, door_idx, EdgeType.FUNCTIONAL, 1.0, dx / dist, dy / dist, None))

        return edges

    def _get_tile_type(self, level_data: Dict[str, Any], row: int, col: int) -> int:
        """Get tile id at grid position from provided level_data.

        level_data is expected to contain either:
        - 'tiles' as a 2D array with shape [GRID_HEIGHT, GRID_WIDTH]
        - or 'tile_data' as a flat list of length GRID_WIDTH*GRID_HEIGHT in row-major order
        """
        tiles = level_data.get('tiles')
        if tiles is not None:
            # Clamp to bounds defensively
            if 0 <= row < tiles.shape[0] and 0 <= col < tiles.shape[1]:
                return int(tiles[row, col])
            return 1  # Treat out of bounds as solid walls
        flat = level_data.get('tile_data')
        if flat is not None:
            if 0 <= row < GRID_HEIGHT and 0 <= col < GRID_WIDTH:
                return int(flat[row * GRID_WIDTH + col])
            return 1
        # As a last resort assume empty inside bounds, solid outside
        if row == 0 or row == GRID_HEIGHT - 1 or col == 0 or col == GRID_WIDTH - 1:
            return 1
        return 0

    def _is_solid(self, tile_type: int) -> bool:
        """Check if tile is an entirely solid block (tile id 1)."""
        return tile_type == 1

    def _is_half_tile(self, tile_type: int) -> bool:
        """Check if tile is one of the four half blocks (2-5)."""
        return 2 <= tile_type <= 5

    def _is_slope(self, tile_type: int) -> bool:
        """Check if tile is a slope/curved surface.

        This includes 45° slopes (6-9), quarter moons (10-13), quarter pipes (14-17),
        and both mild/steep variants (18-33).
        """
        return 6 <= tile_type <= 33

    def _is_hazard_entity(self, entity: Dict[str, Any]) -> bool:
        """Determine if an entity is hazardous.

        Hazard entity is:
        - A mine / toggle mine that has been toggled
        - Any drone (including mini drone)
        - Death ball
        - Thwump or shove thwump (dangerous face)
        """
        et = entity.get('type', 'unknown')
        state = entity.get('state', None)

        # Support both numeric ids and semantic strings coming from different callers
        if isinstance(et, int):
            # Known hazardous numeric ids from simulator
            return et in {12, 14, 20, 21, 25, 26, 28}  # death ball(12/25), drones(14/26), thwumps(20/28), toggled mine(21)

        et_l = str(et).lower()
        if 'drone' in et_l:
            return True
        if 'death' in et_l and 'ball' in et_l:
            return True
        if 'thwump' in et_l:
            return True
        if 'mine' in et_l:
            # default to hazardous only if explicitly toggled
            return state in (0, 'toggled', True)
        return False

    def _get_entity_type_id(self, entity_type: Any) -> int:
        """Map entity type (string or numeric id) to a compact type id space for nodes."""
        # A small, stable taxonomy for graph observations (<= entity_type_dim)
        TYPE_OTHER = 19
        type_map_strings = {
            'exit': 0,
            'switch': 1,
            'door': 2,
            'gold': 3,
            'drone': 4,
            'mine': 5,
            'thwump': 6,
            'launchpad': 7,
            'boost': 7,
            'one_way': 8,
            'death_ball': 9,
            'bounce_block': 10,
            'shove_thwump': 11,
        }

        if isinstance(entity_type, int):
            # Map simulator numeric ids into categories above
            numeric_map = {
                2: 3,     # Gold
                3: 2,     # Exit door
                4: 1,     # Exit switch
                5: 2, 6: 2, 8: 2,  # Doors
                10: 7, 24: 7,      # Launch/boost pad variants
                11: 8,             # One way
                14: 4, 26: 4,      # Drones
                17: 10,            # Bounce block
                20: 6, 28: 11,     # Thwumps
                21: 5, 1: 5,       # Mines / toggle mines
                25: 9, 12: 9,      # Death ball (keep both aliases safe)
            }
            return numeric_map.get(entity_type, TYPE_OTHER)

        # String fuzzy mapping
        et_l = str(entity_type).lower()
        # Handle door segment types
        if 'door_segment' in et_l:
            return 2  # Door category
        for key, val in type_map_strings.items():
            if key in et_l:
                return val
        return TYPE_OTHER

    def _entity_sort_key(self, entity: Dict[str, Any]) -> Tuple[int, float, float]:
        """Stable sort key for entities: (normalized_type, x, y).

        Normalizes potentially mixed representations (e.g., numeric ids vs strings)
        and coerces coordinates to floats to prevent TypeError during sorting.
        """
        # Normalize type to compact id space
        try:
            type_key = int(self._get_entity_type_id(entity.get('type', 'unknown')))
        except Exception as type_err:
            logging.warning("GraphBuilder: invalid entity type %r (%s); defaulting type key",
                            entity.get('type', None), type_err)
            type_key = 999
        # Normalize positions
        try:
            x = float(entity.get('x', 0.0))
        except Exception as x_err:
            logging.warning("GraphBuilder: invalid entity x in sort %r (%s); using 0.0",
                            entity.get('x', None), x_err)
            x = 0.0
        try:
            y = float(entity.get('y', 0.0))
        except Exception as y_err:
            logging.warning("GraphBuilder: invalid entity y in sort %r (%s); using 0.0",
                            entity.get('y', None), y_err)
            y = 0.0
        return (type_key, x, y)