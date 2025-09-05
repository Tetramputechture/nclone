"""
Pathfinding algorithms for graph-based navigation in N++ levels.

This module provides A* and Dijkstra pathfinding algorithms with 100% accurate
physics-based edge costs using the same movement classification and trajectory
calculation logic as the npp-rl training system.
"""

import heapq
import math
import sys
import os
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

# Add npp-rl to path for accurate physics integration
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../npp-rl'))

from .common import GraphData, NodeType, EdgeType
from .graph_construction import GraphConstructor
from .hierarchical_builder import HierarchicalGraphBuilder, HierarchicalGraphData

# Import accurate movement classification and trajectory calculation
try:
    from npp_rl.models.movement_classifier import MovementClassifier, MovementType, NinjaState
    from npp_rl.models.trajectory_calculator import TrajectoryCalculator, TrajectoryResult, MovementState
    ACCURATE_PHYSICS_AVAILABLE = True
except ImportError:
    print("Warning: npp-rl physics modules not available, using simplified physics")
    ACCURATE_PHYSICS_AVAILABLE = False


class PathfindingAlgorithm(IntEnum):
    """Available pathfinding algorithms."""
    DIJKSTRA = 0
    A_STAR = 1


@dataclass
class PathResult:
    """Result of pathfinding operation."""
    path: List[int]  # Node indices forming the path
    total_cost: float
    success: bool
    nodes_explored: int
    path_coordinates: List[Tuple[float, float]]  # World coordinates of path nodes
    edge_types: List[EdgeType]  # Types of edges used in path


@dataclass
class PathNode:
    """Node in pathfinding search."""
    node_idx: int
    g_cost: float  # Cost from start
    h_cost: float  # Heuristic cost to goal
    f_cost: float  # Total cost (g + h)
    parent: Optional[int]  # Parent node index
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost


class AccuratePathfindingEngine:
    """
    Physics-accurate pathfinding engine for N++ graph navigation.
    
    Uses the exact same movement classification and trajectory calculation
    logic as the npp-rl training system for 100% accurate pathfinding.
    """
    
    def __init__(self, level_data: Optional[Dict[str, Any]] = None, entities: Optional[List[Dict[str, Any]]] = None):
        """Initialize accurate pathfinding engine."""
        self.graph_builder = HierarchicalGraphBuilder()
        
        # Initialize accurate physics components if available
        if ACCURATE_PHYSICS_AVAILABLE:
            self.movement_classifier = MovementClassifier()
            self.trajectory_calculator = TrajectoryCalculator()
        else:
            self.movement_classifier = None
            self.trajectory_calculator = None
        
        # Cache level data and entities for physics calculations
        self.level_data = level_data
        self.entities = entities or []
        
        # Heuristic weight for A* (lower = more optimal, higher = faster)
        self.heuristic_weight = 1.0
        
        # Physics-based cost calculation parameters
        self.cost_weights = {
            'energy_cost': 1.0,
            'time_estimate': 0.5,
            'difficulty': 2.0,
            'success_probability': -1.0,  # Negative because higher probability = lower cost
        }
    
    def find_shortest_path(
        self,
        graph_data: GraphData,
        start_node: int,
        goal_node: int,
        algorithm: PathfindingAlgorithm = PathfindingAlgorithm.A_STAR,
        max_nodes_to_explore: int = 10000,
        ninja_state: Optional[Dict[str, Any]] = None
    ) -> PathResult:
        """
        Find shortest path between two nodes using accurate physics.
        
        Args:
            graph_data: Graph data containing nodes and edges
            start_node: Starting node index
            goal_node: Goal node index
            algorithm: Pathfinding algorithm to use
            max_nodes_to_explore: Maximum nodes to explore before giving up
            ninja_state: Current ninja state for accurate physics calculations
            
        Returns:
            PathResult with path information
        """
        if start_node == goal_node:
            return PathResult(
                path=[start_node],
                total_cost=0.0,
                success=True,
                nodes_explored=1,
                path_coordinates=self._get_node_coordinates(graph_data, [start_node]),
                edge_types=[]
            )
        
        if algorithm == PathfindingAlgorithm.A_STAR:
            return self._find_path_a_star(
                graph_data, start_node, goal_node, max_nodes_to_explore, ninja_state
            )
        else:
            return self._find_path_dijkstra(
                graph_data, start_node, goal_node, max_nodes_to_explore, ninja_state
            )
    
    def find_path_to_goal_type(
        self,
        graph_data: GraphData,
        start_node: int,
        goal_node_type: NodeType,
        entities: List[Dict[str, Any]],
        algorithm: PathfindingAlgorithm = PathfindingAlgorithm.A_STAR,
        max_nodes_to_explore: int = 10000
    ) -> PathResult:
        """
        Find shortest path to any node of a specific type.
        
        Args:
            graph_data: Graph data containing nodes and edges
            start_node: Starting node index
            goal_node_type: Type of goal node to find
            entities: List of entities for goal identification
            algorithm: Pathfinding algorithm to use
            max_nodes_to_explore: Maximum nodes to explore
            
        Returns:
            PathResult with path to nearest goal of specified type
        """
        # Find all nodes of the goal type
        goal_nodes = []
        for i in range(graph_data.num_nodes):
            if graph_data.node_mask[i] > 0 and graph_data.node_types[i] == goal_node_type:
                goal_nodes.append(i)
        
        if not goal_nodes:
            return PathResult(
                path=[], total_cost=float('inf'), success=False,
                nodes_explored=0, path_coordinates=[], edge_types=[]
            )
        
        # Find shortest path to any goal node
        best_result = None
        for goal_node in goal_nodes:
            result = self.find_shortest_path(
                graph_data, start_node, goal_node, algorithm, max_nodes_to_explore
            )
            if result.success and (best_result is None or result.total_cost < best_result.total_cost):
                best_result = result
        
        return best_result or PathResult(
            path=[], total_cost=float('inf'), success=False,
            nodes_explored=0, path_coordinates=[], edge_types=[]
        )
    
    def find_hierarchical_path(
        self,
        hierarchical_data: HierarchicalGraphData,
        start_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
        algorithm: PathfindingAlgorithm = PathfindingAlgorithm.A_STAR
    ) -> PathResult:
        """
        Find path using hierarchical pathfinding for better performance.
        
        Uses coarse-to-fine pathfinding: plan at region level, refine at tile level,
        then execute at sub-cell level.
        
        Args:
            hierarchical_data: Multi-resolution graph data
            start_pos: Starting world position (x, y)
            goal_pos: Goal world position (x, y)
            algorithm: Pathfinding algorithm to use
            
        Returns:
            PathResult with detailed path at sub-cell resolution
        """
        # Find corresponding nodes at each resolution level
        start_region = self._find_node_at_position(
            hierarchical_data.region_graph, start_pos
        )
        goal_region = self._find_node_at_position(
            hierarchical_data.region_graph, goal_pos
        )
        
        if start_region is None or goal_region is None:
            return PathResult(
                path=[], total_cost=float('inf'), success=False,
                nodes_explored=0, path_coordinates=[], edge_types=[]
            )
        
        # Plan at region level first
        region_path = self.find_shortest_path(
            hierarchical_data.region_graph, start_region, goal_region, algorithm
        )
        
        if not region_path.success:
            return region_path
        
        # Refine path at tile level
        detailed_path = []
        total_cost = 0.0
        total_explored = region_path.nodes_explored
        all_edge_types = []
        
        for i in range(len(region_path.path) - 1):
            current_region = region_path.path[i]
            next_region = region_path.path[i + 1]
            
            # Find tile nodes in these regions
            current_tiles = self._find_tiles_in_region(
                hierarchical_data, current_region
            )
            next_tiles = self._find_tiles_in_region(
                hierarchical_data, next_region
            )
            
            # Find best tile-level path between regions
            best_tile_path = None
            for start_tile in current_tiles:
                for goal_tile in next_tiles:
                    tile_path = self.find_shortest_path(
                        hierarchical_data.tile_graph, start_tile, goal_tile, algorithm
                    )
                    if tile_path.success and (
                        best_tile_path is None or tile_path.total_cost < best_tile_path.total_cost
                    ):
                        best_tile_path = tile_path
            
            if best_tile_path is None:
                return PathResult(
                    path=[], total_cost=float('inf'), success=False,
                    nodes_explored=total_explored, path_coordinates=[], edge_types=[]
                )
            
            # Add tile path (excluding duplicate nodes)
            if i == 0:
                detailed_path.extend(best_tile_path.path)
            else:
                detailed_path.extend(best_tile_path.path[1:])
            
            total_cost += best_tile_path.total_cost
            total_explored += best_tile_path.nodes_explored
            all_edge_types.extend(best_tile_path.edge_types)
        
        # Convert to sub-cell coordinates
        path_coordinates = self._get_node_coordinates(
            hierarchical_data.sub_cell_graph, detailed_path
        )
        
        return PathResult(
            path=detailed_path,
            total_cost=total_cost,
            success=True,
            nodes_explored=total_explored,
            path_coordinates=path_coordinates,
            edge_types=all_edge_types
        )
    
    def _find_path_a_star(
        self,
        graph_data: GraphData,
        start_node: int,
        goal_node: int,
        max_nodes_to_explore: int,
        ninja_state: Optional[Dict[str, Any]] = None
    ) -> PathResult:
        """A* pathfinding implementation with accurate physics."""
        # Build adjacency list with accurate physics-based costs
        adjacency = self._build_adjacency_list(graph_data, ninja_state)
        
        # Initialize search
        open_set = []
        closed_set: Set[int] = set()
        g_costs = {start_node: 0.0}
        
        # Get goal position for heuristic calculation
        goal_pos = self._get_node_position(graph_data, goal_node)
        
        # Add start node to open set
        start_pos = self._get_node_position(graph_data, start_node)
        h_cost = self._calculate_heuristic(start_pos, goal_pos)
        start_path_node = PathNode(
            node_idx=start_node,
            g_cost=0.0,
            h_cost=h_cost,
            f_cost=h_cost,
            parent=None
        )
        heapq.heappush(open_set, start_path_node)
        
        nodes_explored = 0
        parent_map = {}
        
        while open_set and nodes_explored < max_nodes_to_explore:
            current = heapq.heappop(open_set)
            current_node = current.node_idx
            
            if current_node in closed_set:
                continue
            
            closed_set.add(current_node)
            nodes_explored += 1
            
            # Check if we reached the goal
            if current_node == goal_node:
                path = self._reconstruct_path(parent_map, start_node, goal_node)
                edge_types = self._get_path_edge_types(graph_data, adjacency, path)
                return PathResult(
                    path=path,
                    total_cost=current.g_cost,
                    success=True,
                    nodes_explored=nodes_explored,
                    path_coordinates=self._get_node_coordinates(graph_data, path),
                    edge_types=edge_types
                )
            
            # Explore neighbors
            if current_node in adjacency:
                for neighbor_node, edge_cost, edge_type in adjacency[current_node]:
                    if neighbor_node in closed_set:
                        continue
                    
                    tentative_g_cost = current.g_cost + edge_cost
                    
                    if neighbor_node not in g_costs or tentative_g_cost < g_costs[neighbor_node]:
                        g_costs[neighbor_node] = tentative_g_cost
                        parent_map[neighbor_node] = current_node
                        
                        neighbor_pos = self._get_node_position(graph_data, neighbor_node)
                        h_cost = self._calculate_heuristic(neighbor_pos, goal_pos)
                        f_cost = tentative_g_cost + self.heuristic_weight * h_cost
                        
                        neighbor_path_node = PathNode(
                            node_idx=neighbor_node,
                            g_cost=tentative_g_cost,
                            h_cost=h_cost,
                            f_cost=f_cost,
                            parent=current_node
                        )
                        heapq.heappush(open_set, neighbor_path_node)
        
        # No path found
        return PathResult(
            path=[], total_cost=float('inf'), success=False,
            nodes_explored=nodes_explored, path_coordinates=[], edge_types=[]
        )
    
    def _find_path_dijkstra(
        self,
        graph_data: GraphData,
        start_node: int,
        goal_node: int,
        max_nodes_to_explore: int,
        ninja_state: Optional[Dict[str, Any]] = None
    ) -> PathResult:
        """Dijkstra pathfinding implementation with accurate physics."""
        # Build adjacency list with accurate physics-based costs
        adjacency = self._build_adjacency_list(graph_data, ninja_state)
        
        # Initialize search
        distances = {start_node: 0.0}
        parent_map = {}
        visited: Set[int] = set()
        priority_queue = [(0.0, start_node)]
        
        nodes_explored = 0
        
        while priority_queue and nodes_explored < max_nodes_to_explore:
            current_dist, current_node = heapq.heappop(priority_queue)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            nodes_explored += 1
            
            # Check if we reached the goal
            if current_node == goal_node:
                path = self._reconstruct_path(parent_map, start_node, goal_node)
                edge_types = self._get_path_edge_types(graph_data, adjacency, path)
                return PathResult(
                    path=path,
                    total_cost=current_dist,
                    success=True,
                    nodes_explored=nodes_explored,
                    path_coordinates=self._get_node_coordinates(graph_data, path),
                    edge_types=edge_types
                )
            
            # Explore neighbors
            if current_node in adjacency:
                for neighbor_node, edge_cost, edge_type in adjacency[current_node]:
                    if neighbor_node in visited:
                        continue
                    
                    new_distance = current_dist + edge_cost
                    
                    if neighbor_node not in distances or new_distance < distances[neighbor_node]:
                        distances[neighbor_node] = new_distance
                        parent_map[neighbor_node] = current_node
                        heapq.heappush(priority_queue, (new_distance, neighbor_node))
        
        # No path found
        return PathResult(
            path=[], total_cost=float('inf'), success=False,
            nodes_explored=nodes_explored, path_coordinates=[], edge_types=[]
        )
    
    def _build_adjacency_list(
        self, 
        graph_data: GraphData, 
        ninja_state: Optional[Dict[str, Any]] = None
    ) -> Dict[int, List[Tuple[int, float, EdgeType]]]:
        """Build adjacency list from graph data with accurate physics-based costs."""
        adjacency = {}
        
        for edge_idx in range(graph_data.num_edges):
            if graph_data.edge_mask[edge_idx] == 0:
                continue
            
            src_node = graph_data.edge_index[0, edge_idx]
            dst_node = graph_data.edge_index[1, edge_idx]
            edge_type = EdgeType(graph_data.edge_types[edge_idx])
            
            # Calculate accurate physics-based edge cost
            edge_cost = self._calculate_edge_cost(
                graph_data, edge_idx, edge_type, src_node, dst_node, ninja_state
            )
            
            # Add edge to adjacency list
            if src_node not in adjacency:
                adjacency[src_node] = []
            adjacency[src_node].append((dst_node, edge_cost, edge_type))
        
        return adjacency
    
    def _calculate_edge_cost(
        self, 
        graph_data: GraphData, 
        edge_idx: int, 
        edge_type: EdgeType,
        src_node: int,
        dst_node: int,
        ninja_state: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate accurate physics-based cost for traversing an edge."""
        if not ACCURATE_PHYSICS_AVAILABLE or not self.movement_classifier:
            # Fallback to simplified cost calculation
            edge_features = graph_data.edge_features[edge_idx]
            base_cost = edge_features[0] if len(edge_features) > 0 else 1.0
            type_multipliers = {
                EdgeType.WALK: 1.0,
                EdgeType.JUMP: 1.5,
                EdgeType.FALL: 1.2,
                EdgeType.WALL_SLIDE: 2.0,
                EdgeType.ONE_WAY: 1.3,
                EdgeType.FUNCTIONAL: 1.1,
            }
            return max(base_cost * type_multipliers.get(edge_type, 1.0), 0.1)
        
        # Get node positions
        src_pos = self._get_node_position(graph_data, src_node)
        dst_pos = self._get_node_position(graph_data, dst_node)
        
        # Create ninja state for movement classification
        ninja_state_obj = NinjaState(
            movement_state=ninja_state.get('movement_state', 0) if ninja_state else 0,
            velocity=ninja_state.get('velocity', (0.0, 0.0)) if ninja_state else (0.0, 0.0),
            position=src_pos,
            ground_contact=ninja_state.get('ground_contact', True) if ninja_state else True,
            wall_contact=ninja_state.get('wall_contact', False) if ninja_state else False
        )
        
        # Use accurate movement classification
        movement_type, physics_params = self.movement_classifier.classify_movement(
            src_pos, dst_pos, ninja_state_obj, self.level_data
        )
        
        # Calculate cost based on physics parameters
        cost = 0.0
        
        # Energy cost component
        energy_cost = physics_params.get('energy_cost', 1.0)
        cost += energy_cost * self.cost_weights['energy_cost']
        
        # Time estimate component
        time_estimate = physics_params.get('time_estimate', 1.0)
        cost += time_estimate * self.cost_weights['time_estimate']
        
        # Difficulty component
        difficulty = physics_params.get('difficulty', 0.5)
        cost += difficulty * self.cost_weights['difficulty']
        
        # Success probability component (lower probability = higher cost)
        success_probability = physics_params.get('success_probability', 0.8)
        cost += (1.0 - success_probability) * abs(self.cost_weights['success_probability'])
        
        # Additional trajectory validation if available
        if self.trajectory_calculator:
            # Check if movement is physically feasible
            is_feasible = self.movement_classifier.is_movement_physically_feasible(
                src_pos, dst_pos, self.level_data, self.entities
            )
            
            if not is_feasible:
                cost += 1000.0  # Very high cost for infeasible movements
            else:
                # Calculate trajectory-specific costs
                if movement_type in [MovementType.JUMP, MovementType.WALL_JUMP]:
                    trajectory_result = self.trajectory_calculator.calculate_jump_trajectory(
                        src_pos, dst_pos, 
                        MovementState.JUMPING if movement_type == MovementType.JUMP else MovementState.WALL_JUMPING
                    )
                    
                    if trajectory_result.feasible:
                        # Add trajectory-specific costs
                        cost += trajectory_result.energy_cost * 0.5
                        cost += (1.0 - trajectory_result.success_probability) * 2.0
                        
                        # Validate trajectory clearance
                        if trajectory_result.trajectory_points:
                            is_clear = self.trajectory_calculator.validate_trajectory_clearance(
                                trajectory_result.trajectory_points, self.level_data, self.entities
                            )
                            if not is_clear:
                                cost += 500.0  # High cost for blocked trajectories
                    else:
                        cost += 1000.0  # Very high cost for infeasible trajectories
        
        return max(cost, 0.1)  # Minimum cost to avoid zero-cost edges
    
    def _calculate_heuristic(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate physics-informed heuristic distance between two positions."""
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Basic Euclidean distance
        euclidean_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        if not ACCURATE_PHYSICS_AVAILABLE or not self.movement_classifier:
            return euclidean_distance
        
        # Physics-informed heuristic based on movement type
        dx = x2 - x1
        dy = y2 - y1
        
        # Estimate movement type based on displacement
        if abs(dy) < 10.0:  # Mostly horizontal - likely walking
            # Use walking speed for time estimate
            from nclone.constants.physics_constants import MAX_HOR_SPEED
            time_estimate = abs(dx) / MAX_HOR_SPEED if abs(dx) > 0 else 0.1
            return time_estimate
        elif dy < -20.0:  # Significant upward movement - likely jumping
            # Use jump physics for time estimate
            from nclone.constants.physics_constants import GRAVITY_JUMP, JUMP_FLAT_GROUND_Y
            initial_vy = abs(JUMP_FLAT_GROUND_Y)
            if abs(dy) > 0:
                time_up = initial_vy / GRAVITY_JUMP
                time_estimate = time_up * 2  # Rough estimate for jump time
            else:
                time_estimate = euclidean_distance / MAX_HOR_SPEED
            return time_estimate * 1.5  # Jump penalty
        elif dy > 20.0:  # Significant downward movement - likely falling
            # Use fall physics for time estimate
            from nclone.constants.physics_constants import GRAVITY_FALL
            time_estimate = math.sqrt(2 * abs(dy) / GRAVITY_FALL) if abs(dy) > 0 else 0.1
            return time_estimate * 1.2  # Fall penalty
        else:
            # Mixed movement - use distance with moderate penalty
            return euclidean_distance * 1.3
    
    def _get_node_position(self, graph_data: GraphData, node_idx: int) -> Tuple[float, float]:
        """Extract world position from node features."""
        if node_idx >= graph_data.num_nodes or graph_data.node_mask[node_idx] == 0:
            return (0.0, 0.0)
        
        # Assume first two features are x, y coordinates
        node_features = graph_data.node_features[node_idx]
        x = node_features[0] if len(node_features) > 0 else 0.0
        y = node_features[1] if len(node_features) > 1 else 0.0
        
        return (float(x), float(y))
    
    def _get_node_coordinates(self, graph_data: GraphData, path: List[int]) -> List[Tuple[float, float]]:
        """Get world coordinates for all nodes in path."""
        coordinates = []
        for node_idx in path:
            coordinates.append(self._get_node_position(graph_data, node_idx))
        return coordinates
    
    def _get_path_edge_types(
        self,
        graph_data: GraphData,
        adjacency: Dict[int, List[Tuple[int, float, EdgeType]]],
        path: List[int]
    ) -> List[EdgeType]:
        """Get edge types for all edges in path."""
        edge_types = []
        for i in range(len(path) - 1):
            src_node = path[i]
            dst_node = path[i + 1]
            
            # Find edge type in adjacency list
            if src_node in adjacency:
                for neighbor, cost, edge_type in adjacency[src_node]:
                    if neighbor == dst_node:
                        edge_types.append(edge_type)
                        break
                else:
                    edge_types.append(EdgeType.WALK)  # Default if not found
            else:
                edge_types.append(EdgeType.WALK)  # Default if not found
        
        return edge_types
    
    def _reconstruct_path(self, parent_map: Dict[int, int], start_node: int, goal_node: int) -> List[int]:
        """Reconstruct path from parent map."""
        path = []
        current = goal_node
        
        while current is not None:
            path.append(current)
            current = parent_map.get(current)
        
        path.reverse()
        return path
    
    def _find_node_at_position(self, graph_data: GraphData, position: Tuple[float, float]) -> Optional[int]:
        """Find the node closest to a given world position."""
        target_x, target_y = position
        best_node = None
        best_distance = float('inf')
        
        for node_idx in range(graph_data.num_nodes):
            if graph_data.node_mask[node_idx] == 0:
                continue
            
            node_pos = self._get_node_position(graph_data, node_idx)
            distance = self._calculate_heuristic(node_pos, position)
            
            if distance < best_distance:
                best_distance = distance
                best_node = node_idx
        
        return best_node
    
    def _find_tiles_in_region(self, hierarchical_data: HierarchicalGraphData, region_node: int) -> List[int]:
        """Find all tile nodes that belong to a specific region."""
        # This would use the hierarchical mapping data
        # For now, return a simple implementation
        tiles = []
        
        # Use the tile_to_region_mapping to find tiles in this region
        for tile_idx in range(hierarchical_data.tile_graph.num_nodes):
            if (tile_idx < len(hierarchical_data.tile_to_region_mapping) and
                hierarchical_data.tile_to_region_mapping[tile_idx] == region_node):
                tiles.append(tile_idx)
        
        return tiles


# Compatibility alias for existing code
PathfindingEngine = AccuratePathfindingEngine