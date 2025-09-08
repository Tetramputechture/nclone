#!/usr/bin/env python3
"""
Physics-enhanced edge builder with comprehensive waypoint pathfinding.

This module extends the standard edge builder with a physics-accurate waypoint
system that respects all N++ game mechanics and tile definitions.
"""

import math
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

from .edge_building import EdgeBuilder
from .physics_waypoint_pathfinder import PhysicsWaypointPathfinder
from .level_data import LevelData
from .common import SUB_CELL_SIZE, EdgeType
from ..constants.physics_constants import NINJA_RADIUS, MAX_JUMP_DISTANCE, MAX_FALL_DISTANCE


class PhysicsEnhancedEdgeBuilder(EdgeBuilder):
    """
    Physics-enhanced edge builder with comprehensive waypoint pathfinding.
    
    Extends the standard edge builder to create physics-accurate intermediate
    waypoints for long-distance navigation that exceeds game physics constraints.
    
    Features:
    - Respects all N++ physics constraints (jump/fall distances, ninja radius)
    - Handles all 34 tile types with proper collision detection
    - Creates reliable multi-hop paths for complex navigation
    - Maintains compatibility with existing graph system
    """
    
    def __init__(self, feature_extractor):
        """Initialize physics-enhanced edge builder."""
        super().__init__(feature_extractor)
        self.waypoint_pathfinder = PhysicsWaypointPathfinder()
        self._waypoint_nodes = {}  # Maps waypoint positions to node indices
        
    def build_edges(
        self,
        sub_grid_node_map: Dict[Tuple[int, int], int],
        entity_nodes: List[Tuple[int, Dict[str, Any]]],
        level_data: LevelData,
        ninja_position: Tuple[float, float],
        ninja_velocity: Optional[Tuple[float, float]],
        ninja_state: Optional[int],
        edge_index: np.ndarray,
        edge_features: np.ndarray,
        edge_mask: np.ndarray,
        edge_types: np.ndarray,
        edge_count: int,
        edge_feature_dim: int,
    ) -> int:
        """
        Build edges with physics-enhanced waypoint pathfinding.
        
        This method:
        1. Builds all standard edges using the parent class
        2. Identifies long-distance paths that exceed physics constraints
        3. Creates physics-accurate waypoints for multi-hop navigation
        4. Builds edges connecting waypoints with proper physics validation
        """
        print("🚀 Building physics-enhanced edges with comprehensive waypoint pathfinding...")
        
        # Step 1: Build standard edges using parent class
        edge_count = super().build_edges(
            sub_grid_node_map, entity_nodes, level_data, ninja_position, ninja_velocity,
            ninja_state, edge_index, edge_features, edge_mask, edge_types,
            edge_count, edge_feature_dim
        )
        
        print(f"   Standard edges built: {edge_count}")
        
        # Step 2: Identify target positions that require long-distance navigation
        target_positions = self._identify_navigation_targets(entity_nodes, ninja_position)
        if not target_positions:
            print("   No long-distance navigation targets found")
            return edge_count
        
        print(f"   Found {len(target_positions)} navigation targets requiring waypoints")
        
        # Step 3: Create physics-accurate waypoints and edges
        waypoint_edge_count = self._build_physics_waypoint_edges(
            sub_grid_node_map, level_data, ninja_position, target_positions,
            edge_index, edge_features, edge_mask, edge_types,
            edge_count, edge_feature_dim
        )
        
        edge_count += waypoint_edge_count
        print(f"   Physics waypoint edges built: {waypoint_edge_count}")
        print(f"   Total edges with physics enhancement: {edge_count}")
        
        return edge_count
    
    def _identify_navigation_targets(self, entity_nodes: List[Tuple[int, Dict[str, Any]]], 
                                   ninja_position: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Identify target positions that require long-distance navigation.
        
        Focuses on switches and doors that are beyond reliable jump distance
        from the ninja's current position.
        """
        targets = []
        
        print(f"   Analyzing {len(entity_nodes)} entities for navigation targets...")
        
        for node_idx, entity in entity_nodes:
            entity_type = entity.get('type', -1)
            x = float(entity.get('x', 0))
            y = float(entity.get('y', 0))
            
            # Calculate distance from ninja to entity
            distance = math.sqrt((x - ninja_position[0])**2 + (y - ninja_position[1])**2)
            
            print(f"     Entity type {entity_type} at ({x:.1f}, {y:.1f}) - {distance:.1f}px")
            
            # Focus on switches and doors - primary navigation targets
            # Entity types: 2=switch, 3=door, 4=locked_door_switch (based on debug output)
            if entity_type in [2, 3, 4]:
                # Only create waypoints for targets beyond reliable jump distance
                if distance > MAX_JUMP_DISTANCE:
                    targets.append((x, y))
                    entity_name = "switch" if entity_type in [2, 4] else "door"
                    print(f"       ✅ Long-distance target: {entity_name} at ({x:.1f}, {y:.1f}) - {distance:.1f}px")
                else:
                    entity_name = "switch" if entity_type in [2, 4] else "door"
                    print(f"       ⚠️  Short-distance target: {entity_name} at ({x:.1f}, {y:.1f}) - {distance:.1f}px (within jump range)")
        
        return targets
    
    def _build_physics_waypoint_edges(
        self,
        sub_grid_node_map: Dict[Tuple[int, int], int],
        level_data: LevelData,
        ninja_position: Tuple[float, float],
        target_positions: List[Tuple[float, float]],
        edge_index: np.ndarray,
        edge_features: np.ndarray,
        edge_mask: np.ndarray,
        edge_types: np.ndarray,
        edge_count: int,
        edge_feature_dim: int,
    ) -> int:
        """
        Build physics-accurate waypoint edges for long-distance navigation.
        
        Creates waypoints that respect all game physics and builds edges
        connecting them with proper collision detection and distance validation.
        """
        print("🎯 Creating physics-accurate waypoints for long-distance navigation...")
        
        total_waypoint_edges = 0
        
        # Process each long-distance target
        for i, target_pos in enumerate(target_positions):
            distance = math.sqrt((target_pos[0] - ninja_position[0])**2 + (target_pos[1] - ninja_position[1])**2)
            print(f"\n   Target {i+1}/{len(target_positions)}: ({target_pos[0]:.1f}, {target_pos[1]:.1f}) - {distance:.1f}px")
            
            # Create physics-accurate waypoints for this path
            waypoints = self.waypoint_pathfinder.create_physics_accurate_waypoints(
                ninja_position, target_pos, level_data
            )
            
            if waypoints:
                # Add waypoint nodes to the graph
                waypoint_node_indices = self._add_physics_waypoint_nodes(
                    waypoints, sub_grid_node_map
                )
                
                # Build physics-validated edges connecting the waypoints
                edges_added = self._connect_physics_waypoint_nodes(
                    waypoint_node_indices, ninja_position, target_pos,
                    sub_grid_node_map, level_data,
                    edge_index, edge_features, edge_mask, edge_types,
                    edge_count + total_waypoint_edges, edge_feature_dim
                )
                
                total_waypoint_edges += edges_added
                print(f"     Added {edges_added} physics-validated edges for this target")
            else:
                print(f"     No physics-accurate waypoints could be created for this target")
        
        # Display comprehensive statistics
        stats = self.waypoint_pathfinder.get_physics_statistics()
        print(f"\n📊 Physics waypoint statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.1f}")
            else:
                print(f"   {key}: {value}")
        
        return total_waypoint_edges
    
    def _add_physics_waypoint_nodes(
        self,
        waypoints: List,
        sub_grid_node_map: Dict[Tuple[int, int], int]
    ) -> List[int]:
        """
        Add physics-validated waypoint nodes to the graph.
        
        Returns list of node indices for the waypoints.
        """
        waypoint_node_indices = []
        
        for waypoint in waypoints:
            # Convert waypoint position to sub-grid coordinates
            sub_col = int(waypoint.x / SUB_CELL_SIZE)
            sub_row = int(waypoint.y / SUB_CELL_SIZE)
            
            # Find or create node for this position
            if (sub_row, sub_col) in sub_grid_node_map:
                node_idx = sub_grid_node_map[(sub_row, sub_col)]
                print(f"     Using existing node {node_idx} for waypoint at ({waypoint.x:.1f}, {waypoint.y:.1f})")
            else:
                # Create new node index
                node_idx = len(sub_grid_node_map)
                sub_grid_node_map[(sub_row, sub_col)] = node_idx
                print(f"     Created new node {node_idx} for waypoint at ({waypoint.x:.1f}, {waypoint.y:.1f})")
            
            waypoint_node_indices.append(node_idx)
            self._waypoint_nodes[(waypoint.x, waypoint.y)] = node_idx
        
        return waypoint_node_indices
    
    def _connect_physics_waypoint_nodes(
        self,
        waypoint_node_indices: List[int],
        ninja_position: Tuple[float, float],
        target_position: Tuple[float, float],
        sub_grid_node_map: Dict[Tuple[int, int], int],
        level_data: LevelData,
        edge_index: np.ndarray,
        edge_features: np.ndarray,
        edge_mask: np.ndarray,
        edge_types: np.ndarray,
        edge_count: int,
        edge_feature_dim: int,
    ) -> int:
        """
        Connect waypoint nodes with physics-validated edges.
        
        Creates edges: ninja -> waypoint1 -> waypoint2 -> ... -> target
        All edges are validated for physics accuracy.
        """
        edges_added = 0
        
        if not waypoint_node_indices:
            return 0
        
        # Find ninja and target nodes
        ninja_sub_col = int(ninja_position[0] / SUB_CELL_SIZE)
        ninja_sub_row = int(ninja_position[1] / SUB_CELL_SIZE)
        ninja_node = sub_grid_node_map.get((ninja_sub_row, ninja_sub_col))
        
        # If ninja node not found, try nearby positions
        if ninja_node is None:
            print(f"     Searching for ninja node near ({ninja_sub_row}, {ninja_sub_col})...")
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    check_pos = (ninja_sub_row + dr, ninja_sub_col + dc)
                    if check_pos in sub_grid_node_map:
                        ninja_node = sub_grid_node_map[check_pos]
                        print(f"     Found ninja node {ninja_node} at offset ({dr}, {dc})")
                        break
                if ninja_node is not None:
                    break
        
        target_sub_col = int(target_position[0] / SUB_CELL_SIZE)
        target_sub_row = int(target_position[1] / SUB_CELL_SIZE)
        target_node = sub_grid_node_map.get((target_sub_row, target_sub_col))
        
        # If target node not found, try nearby positions
        if target_node is None:
            print(f"     Searching for target node near ({target_sub_row}, {target_sub_col})...")
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    check_pos = (target_sub_row + dr, target_sub_col + dc)
                    if check_pos in sub_grid_node_map:
                        target_node = sub_grid_node_map[check_pos]
                        print(f"     Found target node {target_node} at offset ({dr}, {dc})")
                        break
                if target_node is not None:
                    break
        
        if ninja_node is None:
            print(f"     Warning: Could not find ninja node at ({ninja_position[0]:.1f}, {ninja_position[1]:.1f})")
            return 0
        
        if target_node is None:
            print(f"     Warning: Could not find target node at ({target_position[0]:.1f}, {target_position[1]:.1f})")
            return 0
        
        # Connect ninja to first waypoint
        if len(waypoint_node_indices) > 0:
            first_waypoint_node = waypoint_node_indices[0]
            if self._add_physics_validated_edge(ninja_node, first_waypoint_node, ninja_position, 
                                              self._get_node_position(first_waypoint_node, sub_grid_node_map),
                                              edge_index, edge_features, edge_mask, edge_types, 
                                              edge_count + edges_added, edge_feature_dim):
                edges_added += 1
                print(f"     Connected ninja node {ninja_node} to first waypoint {first_waypoint_node}")
        
        # Connect waypoints to each other
        for i in range(len(waypoint_node_indices) - 1):
            current_node = waypoint_node_indices[i]
            next_node = waypoint_node_indices[i + 1]
            
            current_pos = self._get_node_position(current_node, sub_grid_node_map)
            next_pos = self._get_node_position(next_node, sub_grid_node_map)
            
            if self._add_physics_validated_edge(current_node, next_node, current_pos, next_pos,
                                              edge_index, edge_features, edge_mask, edge_types,
                                              edge_count + edges_added, edge_feature_dim):
                edges_added += 1
                print(f"     Connected waypoint {current_node} to waypoint {next_node}")
        
        # Connect last waypoint to target
        if len(waypoint_node_indices) > 0:
            last_waypoint_node = waypoint_node_indices[-1]
            last_waypoint_pos = self._get_node_position(last_waypoint_node, sub_grid_node_map)
            
            if self._add_physics_validated_edge(last_waypoint_node, target_node, last_waypoint_pos, target_position,
                                              edge_index, edge_features, edge_mask, edge_types,
                                              edge_count + edges_added, edge_feature_dim):
                edges_added += 1
                print(f"     Connected last waypoint {last_waypoint_node} to target {target_node}")
        
        return edges_added
    
    def _get_node_position(self, node_idx: int, sub_grid_node_map: Dict[Tuple[int, int], int]) -> Tuple[float, float]:
        """Get the position of a node from its index."""
        for (sub_row, sub_col), idx in sub_grid_node_map.items():
            if idx == node_idx:
                x = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
                y = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
                return (float(x), float(y))
        return (0.0, 0.0)  # Fallback
    
    def _add_physics_validated_edge(
        self,
        source_node: int,
        target_node: int,
        source_pos: Tuple[float, float],
        target_pos: Tuple[float, float],
        edge_index: np.ndarray,
        edge_features: np.ndarray,
        edge_mask: np.ndarray,
        edge_types: np.ndarray,
        edge_idx: int,
        edge_feature_dim: int,
    ) -> bool:
        """
        Add a physics-validated edge to the graph.
        
        Validates physics constraints before adding the edge.
        """
        if edge_idx >= edge_index.shape[1]:
            print(f"     Warning: Edge index {edge_idx} exceeds array size {edge_index.shape[1]}")
            return False
        
        # Validate physics constraints
        dx = target_pos[0] - source_pos[0]
        dy = target_pos[1] - source_pos[1]
        distance = math.sqrt(dx * dx + dy * dy)
        
        # Determine edge type based on movement direction and validate distance
        if dy <= 0:  # Upward or horizontal movement (jump)
            if distance > MAX_JUMP_DISTANCE:
                print(f"     Physics violation: Jump distance {distance:.1f}px exceeds limit {MAX_JUMP_DISTANCE}px")
                return False
            edge_type = EdgeType.JUMP
        else:  # Downward movement (fall)
            if distance > MAX_FALL_DISTANCE:
                print(f"     Physics violation: Fall distance {distance:.1f}px exceeds limit {MAX_FALL_DISTANCE}px")
                return False
            edge_type = EdgeType.FALL
        
        # Add edge to graph
        edge_index[0, edge_idx] = source_node
        edge_index[1, edge_idx] = target_node
        edge_types[edge_idx] = edge_type
        edge_mask[edge_idx] = True
        
        # Set physics-based edge features
        if edge_feature_dim > 0:
            edge_features[edge_idx, 0] = distance / 100.0  # Normalized distance cost
        
        return True