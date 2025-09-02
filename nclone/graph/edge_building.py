"""
Edge building utilities for graph construction.

This module provides specialized edge building methods with comprehensive
bounce block awareness and traversability checking.

TODO: HAZARD-AWARE EDGE TRAVERSABILITY SYSTEM
=============================================

PRIORITY: HIGH - Player movement blocked by hazards affects core gameplay mechanics

OVERVIEW:
The current edge traversability system does not account for hazards that can block
player movement. This leads to invalid paths being generated, causing the AI to
attempt movements through dangerous or impossible areas.

REQUIREMENTS:
1. Players cannot move across or through hazards (collision detection)
2. Static hazards (toggle mines, thwumps, shove thwumps) create permanent path blocks
3. Dynamic hazards (moving drones, charging thwumps) create temporary path blocks 
   that change position over time
4. Directional hazards (one-way platforms) block movement from specific directions only
5. Path invalidation must be computationally efficient for real-time gameplay

IMPLEMENTATION STRATEGY:

A. Hazard Classification System:
   - Static hazards: Fixed position, permanent traversability impact
     * Toggle mines (type 5): Block movement when active
     * Thwumps (type 20): Block movement in charging direction when activated
       - Deadly face during charge and stationary state
       - Safe from sides and behind
       - Line-of-sight activation (38-pixel range)
     * Shove Thwumps (type 28): Block movement based on contact direction
       - Deadly inner core (8-pixel radius) 
       - Contact-triggered activation. Before triggering, any side can be touched safely (landed on or wall collision).
        - After triggering (after the player breaks contact), the inner core is deadly.
       - Launch direction affects traversability
   - Directional traversability constraints:
     * One-Way Platforms (type 11): Block movement from specific direction only
       - Orientation-based collision (8 directions: 0-7)
       - Adaptive collision width based on approach velocity
       - Allow passage from non-blocking directions
   - Dynamic hazards: Moving position, time-varying traversability impact
     * Drones (moving enemies): Block current and predicted positions
       - Track movement patterns and patrol routes
       - Consider acceleration and turning radius

B. Computational Efficiency Optimizations:
   - Static Hazard Cache: Pre-compute static hazard positions at level load
     * Create spatial hash map of permanently blocked grid cells
     * Update only when level state changes (switches activated, doors opened)
   - Dynamic Hazard Radius System: Only update edges for dynamic hazards within range
     * HAZARD_UPDATE_RADIUS = 150 pixels from ninja position
     * Outside radius: Use cached traversability from last update
     * Inside radius: Real-time hazard position tracking and path blocking
   - Incremental Edge Updates: Instead of rebuilding entire graph
     * Mark affected edges as "hazard_dirty" when dynamic hazards move
     * Batch update dirty edges every N frames or when hazard exits radius

C. Integration Points:

   1. is_traversable_with_bounce_blocks() method enhancement:
      - Add hazard_entities parameter for dynamic hazards in range
      - Check static hazard cache for permanent blocks
      - Perform dynamic hazard intersection testing for nearby threats

   2. New methods to implement:
      - build_static_hazard_cache(entities, level_data) -> Dict[Tuple[int,int], bool]
        * Index thwumps by charge direction and activation zones
        * Cache shove thwump deadly cores and launch paths
        * Map one-way platform blocked directions
      - get_dynamic_hazards_in_range(entities, ninja_pos, radius) -> List[Entity]
        * Filter for drones and moving entities within update radius
        * Include thwumps in charging state (dynamic behavior)
      - check_path_hazard_intersection(src, tgt, hazard_entity) -> bool
        * Thwump: Check if path crosses charging lane
        * Shove Thwump: Check core radius and potential launch trajectory
        * One-Way Platform: Check approach direction vs blocked direction
        * Drone: Check current position and predicted movement
      - update_dynamic_hazard_edges(hazards, affected_edges) -> None
        * Batch update edges affected by moving drones
        * Update thwump state transitions (immobile -> charging -> retreating)

   3. Edge feature enhancement:
      - Add hazard_risk field to edge features (0.0 = safe, 1.0 = blocked)
      - Add hazard_type field for different hazard response strategies
        * Static blocking, directional blocking, moving threat, activation trigger
      - Add time_to_hazard field for dynamic hazards (when will path be blocked)
      - Add directional_safety field for one-way platforms (safe directions bitmask)

D. Data Structures:
   - Static hazard cache: Dict[(sub_row, sub_col), HazardInfo]
     * HazardInfo: {blocked: bool, hazard_type: int, orientation: int, state: int}
     * Toggle mines: {blocked: bool based on active state}
     * Thwumps: {charge_direction: int, activation_zone: List[Tuple[int,int]]}
     * Shove Thwumps: {core_position: Tuple[int,int], launch_states: Dict[int, bool]}
     * One-Way Platforms: {blocked_direction: int, safe_directions: List[int]}
   - Dynamic hazard tracker: List[DynamicHazard]
     * DynamicHazard: {entity_id: int, entity_type: int, position: Tuple[float,float], 
                       velocity: Tuple[float,float], danger_radius: float, 
                       state: int, predicted_path: List[Tuple[float,float]]}
     * Drones: Track patrol routes and turning points
     * Active Thwumps: Track charging/retreating state and movement direction
   - Edge hazard metadata: Array[edge_idx] -> EdgeHazardMeta
     * EdgeHazardMeta: {last_update_frame: int, hazard_risk: float, 
                        affecting_hazards: List[int], directional_safety: int}

E. Performance Considerations:
   - Limit hazard intersection checks to edges within hazard influence radius
   - Use spatial partitioning for large numbers of dynamic hazards
   - Cache line-segment intersection calculations for repeated path checks
   - Consider hazard update frequency (every frame vs every N frames based on movement speed)

TESTING STRATEGY:
- Static hazard tests:
  * Toggle mine activation/deactivation blocking
  * Thwump charge direction and activation zone detection
  * Shove thwump core collision and launch trajectory blocking
  * One-way platform directional collision (all 8 orientations)
- Dynamic hazard tests:
  * Drone movement prediction and path intersection
  * Thwump state transitions (immobile -> charging -> retreating)
  * Radius-based update system with moving drones
- Performance tests:
  * < 1ms update time for typical entity counts (100+ entities)
  * Efficient spatial partitioning for large levels
- Edge case tests:
  * Hazards at sub-cell grid boundaries
  * Multiple overlapping hazard influence zones
  * Thwump activation during ninja movement
  * One-way platform edge cases (velocity-dependent collision width)

TODO: PRECISE TILE COLLISION SYSTEM
===================================

PRIORITY: HIGH - Current boolean solid/empty tile checking is insufficient for accurate pathfinding

OVERVIEW:
The current tile traversability system in is_basic_traversable() only performs boolean 
solid/empty checks (tile == 0 vs tile != 0). This oversimplified approach doesn't 
account for the detailed tile geometry definitions available in tile_definitions.py, 
leading to inaccurate path planning around complex tile shapes.

CURRENT LIMITATIONS:
- Boolean tile checking treats all non-zero tiles as completely impassable
- Ignores precise collision geometry from TILE_GRID_EDGE_MAP, TILE_SEGMENT_ORTHO_MAP, 
  TILE_SEGMENT_DIAG_MAP, and TILE_SEGMENT_CIRCULAR_MAP
- Cannot handle partial tile traversability (slopes, quarter-pipes, etc.)
- Misses opportunities for movement through tile gaps and along tile edges
- Inconsistent with physics.py collision detection which uses precise segment geometry

PLANNED ENHANCEMENT:
Replace boolean tile checking with precise tile shape collision detection:

A. Tile Geometry Integration:
   - Import and utilize tile_definitions.py mapping tables
   - Implement segment-based path intersection testing
   - Support for orthogonal, diagonal, and circular tile segments
   - Handle edge cases for tiles with multiple segment types

B. Path-Segment Intersection Algorithm:
   - Replace simple tile boundary checks with segment intersection testing
   - Use existing physics.py collision functions as reference:
     * get_time_of_intersection_circle_vs_lineseg() for linear segments
     * get_time_of_intersection_circle_vs_arc() for circular segments
   - Account for ninja radius (NINJA_RADIUS) in collision calculations
   - Test movement ray against all tile segments in path

C. Enhanced Traversability Logic:
   - is_basic_traversable() -> is_precise_traversable()
   - Check path intersection with tile segments instead of tile solidity
   - Support partial traversability for slopes and complex tile shapes
   - Maintain compatibility with existing bounce block logic

D. Implementation Methods:
   - get_tile_segments_for_path(src_x, src_y, tgt_x, tgt_y, level_data) -> List[Segment]
     * Gather all tile segments that could intersect movement path
     * Convert tile definitions to world-space segment coordinates
     * Filter segments based on movement direction and tile position
   - test_path_vs_tile_segments(src_x, src_y, tgt_x, tgt_y, segments, ninja_radius) -> bool
     * Use swept-circle collision detection against tile segments
     * Return True if path is clear, False if blocked by tile geometry
   - convert_tile_definition_to_segments(tile_id, tile_x, tile_y) -> List[Segment]
     * Convert tile definition data to world-space collision segments
     * Handle coordinate transformation from tile-local to world coordinates
     * Create appropriate segment objects (linear, circular) for collision testing

CODEBASE PROPAGATION:
This enhancement will require updates across multiple areas:

1. graph/feature_extraction.py:
   - Update edge feature extraction to use precise geometry
   - Enhance movement cost calculations based on tile complexity
   - Add tile shape complexity metrics to edge features

2. physics.py integration:
   - Leverage existing segment collision functions
   - Ensure consistent collision detection between pathfinding and physics simulation
   - May need additional helper functions for graph-specific collision queries

3. environments/dynamic_graph_*.py:
   - Update graph construction to use precise tile collision
   - Modify edge invalidation logic to account for detailed tile geometry
   - Update pathfinding validation with segment-based collision checking

4. models/trajectory_calculator.py:
   - Enhance trajectory feasibility checking with precise tile collision
   - Update movement classification to consider tile shape complexity
   - Improve path success probability estimation based on tile geometry

5. Testing framework:
   - Add comprehensive tests for all tile types (0-37)
   - Test edge cases: slopes, quarter-pipes, circular segments
   - Validate consistency between pathfinding and physics collision
   - Performance testing for real-time path updates

PERFORMANCE CONSIDERATIONS:
- Cache tile segment conversions to avoid repeated calculations
- Spatial partitioning for efficient segment queries
- Optimize segment intersection testing for common movement patterns
- Balance precision with computational overhead for real-time usage

INTEGRATION ORDER:
1. Implement precise tile collision system and integrate with existing traversability
2. Implement static hazard cache system and integrate with existing traversability
3. Add dynamic hazard radius system with incremental edge updates
4. Enhance edge features with hazard risk information
5. Optimize performance and add comprehensive testing
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from ..constants.physics_constants import FULL_MAP_WIDTH, FULL_MAP_HEIGHT, TILE_PIXEL_SIZE
from .common import SUB_CELL_SIZE, EdgeType, E_MAX_EDGES
from .feature_extraction import FeatureExtractor
from .precise_collision import PreciseTileCollision
from .hazard_system import HazardClassificationSystem, HazardInfo, EdgeHazardMeta


class EdgeBuilder:
    """
    Handles edge construction with precise collision detection and hazard awareness.
    
    This enhanced edge builder uses segment-based tile collision detection and
    comprehensive hazard classification to create accurate traversability graphs.
    """
    
    def __init__(self, feature_extractor: FeatureExtractor):
        """
        Initialize edge builder with precise collision and hazard systems.
        
        Args:
            feature_extractor: Feature extraction utility instance
        """
        self.feature_extractor = feature_extractor
        self.precise_collision = PreciseTileCollision()
        self.hazard_system = HazardClassificationSystem(self.precise_collision)
        
        # Cache for static hazards (rebuilt when level changes)
        self._static_hazard_cache = {}
        self._current_level_id = None
    
    def build_edges(
        self,
        sub_grid_node_map: Dict[Tuple[int, int], int],
        entity_nodes: List[Tuple[int, Dict[str, Any]]],
        entities: List[Dict[str, Any]],
        level_data: Dict[str, Any],
        ninja_position: Tuple[float, float],
        ninja_velocity: Optional[Tuple[float, float]],
        ninja_state: Optional[int],
        edge_index: np.ndarray,
        edge_features: np.ndarray,
        edge_mask: np.ndarray,
        edge_types: np.ndarray,
        edge_count: int,
        edge_feature_dim: int
    ) -> int:
        """
        Build edges with bounce block awareness.
        
        Args:
            sub_grid_node_map: Mapping from (row, col) to node indices
            entity_nodes: List of (node_idx, entity) tuples
            entities: List of entity dictionaries
            level_data: Level tile data and structure
            ninja_position: Current ninja position (x, y)
            ninja_velocity: Current ninja velocity (vx, vy)
            ninja_state: Current ninja movement state (0-8)
            edge_index: Edge connectivity array
            edge_features: Edge feature array
            edge_mask: Edge mask array
            edge_types: Edge type array
            edge_count: Current edge count
            edge_feature_dim: Edge feature dimension
            
        Returns:
            Updated edge count
        """
        # Set level data for hazard system collision checking
        self.hazard_system.set_level_data(level_data)
        
        # Build sub-grid traversability edges
        for (sub_row, sub_col), src_idx in sub_grid_node_map.items():
            if edge_count >= E_MAX_EDGES - 8:  # Leave room for 8 directions
                break
                
            # Check 8-connected neighbors
            directions = [
                (0, 1), (1, 0), (0, -1), (-1, 0),  # Cardinal
                (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal
            ]
            
            for dr, dc in directions:
                tgt_row = sub_row + dr
                tgt_col = sub_col + dc
                
                if (tgt_row, tgt_col) in sub_grid_node_map:
                    tgt_idx = sub_grid_node_map[(tgt_row, tgt_col)]
                    
                    # Check traversability with precise collision and hazard awareness
                    if self.is_traversable_with_hazards(
                        sub_row, sub_col, tgt_row, tgt_col, entities, level_data, ninja_position
                    ):
                        edge_index[0, edge_count] = src_idx
                        edge_index[1, edge_count] = tgt_idx
                        
                        # Extract edge features
                        edge_features[edge_count] = self.feature_extractor.extract_edge_features(
                            sub_row, sub_col, tgt_row, tgt_col, entities, ninja_position, ninja_velocity, edge_feature_dim
                        )
                        
                        edge_mask[edge_count] = 1.0
                        edge_types[edge_count] = EdgeType.WALK
                        edge_count += 1
        
        # Build entity interaction edges
        edge_count = self.build_entity_edges(
            entity_nodes, entities, ninja_position, ninja_velocity,
            edge_index, edge_features, edge_mask, edge_types, edge_count, edge_feature_dim
        )
        
        return edge_count
    
    def build_entity_edges(
        self,
        entity_nodes: List[Tuple[int, Dict[str, Any]]],
        entities: List[Dict[str, Any]],
        ninja_position: Tuple[float, float],
        ninja_velocity: Optional[Tuple[float, float]],
        edge_index: np.ndarray,
        edge_features: np.ndarray,
        edge_mask: np.ndarray,
        edge_types: np.ndarray,
        edge_count: int,
        edge_feature_dim: int
    ) -> int:
        """
        Build entity interaction edges with bounce block chaining support.
        
        Args:
            entity_nodes: List of (node_idx, entity) tuples
            entities: List of entity dictionaries
            ninja_position: Current ninja position (x, y)
            ninja_velocity: Current ninja velocity (vx, vy)
            edge_index: Edge connectivity array
            edge_features: Edge feature array
            edge_mask: Edge mask array
            edge_types: Edge type array
            edge_count: Current edge count
            edge_feature_dim: Edge feature dimension
            
        Returns:
            Updated edge count
        """
        # Build bounce block interaction edges
        for i, (node_idx, entity) in enumerate(entity_nodes):
            if edge_count >= E_MAX_EDGES - 10:  # Leave room for more edges
                break
                
            if entity.get('type') == 17:  # Bounce block
                # Create edges to nearby bounce blocks for chaining
                for j, (other_idx, other_entity) in enumerate(entity_nodes):
                    if i != j and other_entity.get('type') == 17:
                        entity_x = entity.get('x', 0.0)
                        entity_y = entity.get('y', 0.0)
                        other_x = other_entity.get('x', 0.0)
                        other_y = other_entity.get('y', 0.0)
                        
                        distance = math.sqrt((entity_x - other_x)**2 + (entity_y - other_y)**2)
                        
                        # Connect nearby bounce blocks for chaining mechanics
                        if distance < 50.0:  # Within chaining range
                            edge_index[0, edge_count] = node_idx
                            edge_index[1, edge_count] = other_idx
                            
                            # Create bounce chain edge features
                            chain_features = np.zeros(edge_feature_dim, dtype=np.float32)
                            chain_features[EdgeType.FUNCTIONAL] = 1.0  # Functional edge type
                            
                            # Direction
                            if distance > 0:
                                chain_features[len(EdgeType)] = (other_x - entity_x) / distance
                                chain_features[len(EdgeType) + 1] = (other_y - entity_y) / distance
                            
                            # Chain interaction parameters
                            chain_features[len(EdgeType) + 2] = 0.5  # Low traversability cost
                            chain_features[len(EdgeType) + 3] = 0.3  # Fast time of flight
                            chain_features[len(EdgeType) + 4] = 0.6  # Moderate energy cost
                            chain_features[len(EdgeType) + 5] = 0.85  # High success probability
                            chain_features[len(EdgeType) + 8] = 1.0  # Bounce boost available
                            
                            edge_features[edge_count] = chain_features
                            edge_mask[edge_count] = 1.0
                            edge_types[edge_count] = EdgeType.FUNCTIONAL
                            edge_count += 1
        
        return edge_count
    
    def is_traversable_with_hazards(
        self,
        src_row: int,
        src_col: int,
        tgt_row: int,
        tgt_col: int,
        entities: List[Dict[str, Any]],
        level_data: Dict[str, Any],
        ninja_position: Tuple[float, float]
    ) -> bool:
        """
        Check if movement between sub-cells is traversable with hazard awareness.
        
        This method uses precise tile collision detection and comprehensive hazard
        classification to determine path traversability.
        
        Args:
            src_row: Source sub-cell row
            src_col: Source sub-cell column
            tgt_row: Target sub-cell row
            tgt_col: Target sub-cell column
            entities: List of entity dictionaries
            level_data: Level tile data and structure
            ninja_position: Current ninja position for dynamic hazard range
            
        Returns:
            True if movement is traversable
        """
        # Convert to pixel coordinates
        src_x = src_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        src_y = src_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        tgt_x = tgt_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        tgt_y = tgt_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
        
        # Check precise tile collision first
        if not self.is_precise_traversable(src_x, src_y, tgt_x, tgt_y, level_data):
            return False
        
        # Check static hazards
        if not self._check_static_hazards(src_x, src_y, tgt_x, tgt_y, entities, level_data):
            return False
        
        # Check dynamic hazards within range
        if not self._check_dynamic_hazards(src_x, src_y, tgt_x, tgt_y, entities, ninja_position):
            return False
        
        # Check for bounce blocks that may block traversal in narrow passages
        if not self._check_bounce_block_traversal(src_x, src_y, tgt_x, tgt_y, entities):
            return False
        
        return True
    
    def is_precise_traversable(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        level_data: Dict[str, Any]
    ) -> bool:
        """
        Check precise tile traversability using segment-based collision detection.
        
        This method replaces the simplified boolean tile checking with accurate
        physics-based collision testing against tile geometry segments.
        
        Args:
            src_x: Source x coordinate
            src_y: Source y coordinate
            tgt_x: Target x coordinate
            tgt_y: Target y coordinate
            level_data: Level tile data and structure
            
        Returns:
            True if path is traversable through tile geometry
        """
        return self.precise_collision.is_path_traversable(
            src_x, src_y, tgt_x, tgt_y, level_data
        )
    
    def _check_static_hazards(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        entities: List[Dict[str, Any]],
        level_data: Dict[str, Any]
    ) -> bool:
        """
        Check if path is blocked by static hazards.
        
        Args:
            src_x: Source x coordinate
            src_y: Source y coordinate
            tgt_x: Target x coordinate
            tgt_y: Target y coordinate
            entities: List of entity dictionaries
            level_data: Level data and structure
            
        Returns:
            True if path is safe from static hazards
        """
        # Build/update static hazard cache
        level_id = level_data.get('level_id', id(level_data))
        if self._current_level_id != level_id:
            self._static_hazard_cache = self.hazard_system.build_static_hazard_cache(
                entities, level_data
            )
            self._current_level_id = level_id
        
        # Check path against cached static hazards
        src_sub_x = int(src_x // 12)  # Sub-cell coordinates
        src_sub_y = int(src_y // 12)
        tgt_sub_x = int(tgt_x // 12)
        tgt_sub_y = int(tgt_y // 12)
        
        # Check all sub-cells along the path
        steps = max(abs(tgt_sub_x - src_sub_x), abs(tgt_sub_y - src_sub_y))
        if steps == 0:
            steps = 1
        
        for i in range(steps + 1):
            t = i / steps
            check_sub_x = int(src_sub_x + t * (tgt_sub_x - src_sub_x))
            check_sub_y = int(src_sub_y + t * (tgt_sub_y - src_sub_y))
            
            if (check_sub_x, check_sub_y) in self._static_hazard_cache:
                hazard_info = self._static_hazard_cache[(check_sub_x, check_sub_y)]
                if self.hazard_system.check_path_hazard_intersection(
                    src_x, src_y, tgt_x, tgt_y, hazard_info
                ):
                    return False  # Path blocked by static hazard
        
        return True  # Path is safe from static hazards
    
    def _check_dynamic_hazards(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        entities: List[Dict[str, Any]],
        ninja_position: Tuple[float, float]
    ) -> bool:
        """
        Check if path is blocked by dynamic hazards within range.
        
        Args:
            src_x: Source x coordinate
            src_y: Source y coordinate
            tgt_x: Target x coordinate
            tgt_y: Target y coordinate
            entities: List of entity dictionaries
            ninja_position: Current ninja position for range calculation
            
        Returns:
            True if path is safe from dynamic hazards
        """
        # Get dynamic hazards within range
        dynamic_hazards = self.hazard_system.get_dynamic_hazards_in_range(
            entities, ninja_position
        )
        
        # Check path against each dynamic hazard
        for hazard_info in dynamic_hazards:
            if self.hazard_system.check_path_hazard_intersection(
                src_x, src_y, tgt_x, tgt_y, hazard_info
            ):
                return False  # Path blocked by dynamic hazard
        
        return True  # Path is safe from dynamic hazards
    
    def _check_bounce_block_traversal(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        entities: List[Dict[str, Any]]
    ) -> bool:
        """
        Check if bounce blocks block traversal in narrow passages.
        
        Bounce blocks can block traversal if they're positioned in the center
        of a one-tile (24px) path and the ninja cannot displace them enough
        horizontally or vertically to get clearance.
        
        Args:
            src_x: Source x coordinate
            src_y: Source y coordinate
            tgt_x: Target x coordinate
            tgt_y: Target y coordinate
            entities: List of entity dictionaries
            
        Returns:
            True if path is traversable (not blocked by bounce blocks)
        """
        for entity in entities:
            if entity.get('type') == EntityType.BOUNCE_BLOCK:
                # Use hazard system's bounce block traversal analysis
                if self.hazard_system.analyze_bounce_block_traversal_blocking(
                    entity, entities, (src_x, src_y), (tgt_x, tgt_y)
                ):
                    return False  # Bounce block blocks this path
        
        return True  # No bounce blocks block the path
    
    def _check_bounce_block_interactions(
        self,
        src_x: float,
        src_y: float,
        tgt_x: float,
        tgt_y: float,
        entities: List[Dict[str, Any]]
    ) -> bool:
        """
        Check for bounce block interactions along the path for feature extraction.
        
        This method identifies potential bounce interactions that affect movement
        but don't necessarily block traversal.
        
        Args:
            src_x: Source x coordinate
            src_y: Source y coordinate
            tgt_x: Target x coordinate
            tgt_y: Target y coordinate
            entities: List of entity dictionaries
            
        Returns:
            True if bounce blocks detected (for feature extraction)
        """
        bounce_detected = False
        
        for entity in entities:
            if entity.get('type') == EntityType.BOUNCE_BLOCK:
                entity_x = entity.get('x', 0.0)
                entity_y = entity.get('y', 0.0)
                
                # Check if path intersects with bounce block
                if self.feature_extractor._path_intersects_bounce_block(
                    src_x, src_y, tgt_x, tgt_y, entity_x, entity_y
                ):
                    bounce_detected = True
                    # Note: This is for feature extraction, not traversal blocking
        
        return bounce_detected