import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional, Set
import heapq
from dataclasses import dataclass
from enum import Enum

from .navigation_graph import JumpCalculator
from .surface_parser import SurfaceType
from ..constants import (
    NINJA_RADIUS, GRAVITY_FALL, GRAVITY_JUMP, MAX_HOR_SPEED,
    MAX_SURVIVABLE_IMPACT
)

class MovementType(Enum):
    """Different types of movement with associated costs and constraints"""
    WALK = "walk"
    JUMP = "jump"
    WALL_JUMP = "wall_jump"
    FALL = "fall"
    SLIDE = "slide"

@dataclass
class PathNode:
    """Node representation for platformer pathfinding with physics state"""
    spatial_id: int
    position: Tuple[float, float]
    movement_state: int  # Ninja movement state (0-9)
    jump_height: float   # Current jump height for gravity constraints
    velocity: Tuple[float, float]  # Estimated velocity at this node
    
    def __hash__(self):
        return hash((self.spatial_id, self.movement_state, int(self.jump_height * 10)))
    
    def __eq__(self, other):
        if not isinstance(other, PathNode):
            return False
        return (self.spatial_id == other.spatial_id and 
                self.movement_state == other.movement_state and
                abs(self.jump_height - other.jump_height) < 0.1)

class PlatformerAStar:
    """A* pathfinding specifically designed for N++ platformer mechanics"""
    
    def __init__(self, nav_graph: nx.DiGraph, jump_calculator: Optional[JumpCalculator] = None):
        self.graph = nav_graph
        self.jump_calc = jump_calculator
        self.path_cache: Dict[Tuple, List[int]] = {}
        
        # N++ physics constants from constants.py
        self.MAX_JUMP_HEIGHT = 96.0  # ~4 tiles in pixels (game-specific derived value)
        self.MAX_HORIZONTAL_SPEED = MAX_HOR_SPEED
        self.GRAVITY_FALL = GRAVITY_FALL
        self.GRAVITY_JUMP = GRAVITY_JUMP
        self.NINJA_RADIUS = NINJA_RADIUS
        self.MAX_SURVIVABLE_IMPACT = MAX_SURVIVABLE_IMPACT
        
        # Movement type costs (relative to base cost)
        self.MOVEMENT_COSTS = {
            MovementType.WALK: 1.0,
            MovementType.JUMP: 1.8,      # Jumping is expensive
            MovementType.WALL_JUMP: 2.5,  # Wall jumping is very expensive
            MovementType.FALL: 0.6,      # Falling is cheap if controlled
            MovementType.SLIDE: 1.2      # Sliding has moderate cost
        }
        
    def find_path(self, start_node_id: int, goal_node_id: int,
                  ninja_state: int = 0, ninja_velocity: Tuple[float, float] = (0, 0),
                  avoid_hazards: bool = True, collect_gold: bool = False) -> Optional[List[int]]:
        """
        Find optimal path using A* with physics-aware heuristics
        
        Args:
            start_node_id: Starting navigation node ID
            goal_node_id: Target navigation node ID
            ninja_state: Current ninja movement state (0-9)
            ninja_velocity: Current ninja velocity (xspeed, yspeed)
            avoid_hazards: Whether to avoid hazardous areas
            collect_gold: Whether to prioritize gold collection
            
        Returns:
            List of node IDs representing the optimal path, or None if no path exists
        """
        cache_key = (start_node_id, goal_node_id, ninja_state, ninja_velocity, avoid_hazards, collect_gold)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        if start_node_id not in self.graph or goal_node_id not in self.graph:
            return None

        # Create start node with physics state
        start_node = PathNode(
            spatial_id=start_node_id,
            position=self.graph.nodes[start_node_id]['position'],
            movement_state=ninja_state,
            jump_height=0.0,
            velocity=ninja_velocity
        )
        
        # Priority queue: (f_score, PathNode)
        open_set: List[Tuple[float, PathNode]] = []
        came_from: Dict[PathNode, PathNode] = {}
        
        g_score: Dict[PathNode, float] = {start_node: 0}
        f_score: Dict[PathNode, float] = {start_node: self._physics_heuristic(start_node, goal_node_id)}
        
        heapq.heappush(open_set, (f_score[start_node], start_node))
        open_set_hash: Set[PathNode] = {start_node}
        
        iterations = 0
        max_iterations = 10000  # Prevent infinite loops
        
        while open_set and iterations < max_iterations:
            iterations += 1
            _, current_node = heapq.heappop(open_set)
            open_set_hash.remove(current_node)
            
            # Goal reached
            if current_node.spatial_id == goal_node_id:
                path = self._reconstruct_path(came_from, current_node)
                self.path_cache[cache_key] = path
                return path
            
            # Explore neighbors with physics constraints
            for neighbor_id in self.graph.neighbors(current_node.spatial_id):
                edge_data = self.graph.get_edge_data(current_node.spatial_id, neighbor_id)
                if not edge_data:
                    continue
                
                # Calculate possible movement states for this transition
                possible_transitions = self._get_possible_transitions(current_node, neighbor_id, edge_data)
                
                for neighbor_node, transition_cost in possible_transitions:
                    # Skip if this transition violates physics constraints
                    if not self._is_physics_valid(current_node, neighbor_node, edge_data):
                        continue
                    
                    # Calculate total cost including hazard penalties
                    total_cost = self._calculate_total_cost(
                        current_node, neighbor_node, edge_data, transition_cost, 
                        avoid_hazards, collect_gold
                    )
                    
                    if total_cost == float('inf'):
                        continue
                    
                    tentative_g_score = g_score[current_node] + total_cost
                    
                    if tentative_g_score < g_score.get(neighbor_node, float('inf')):
                        came_from[neighbor_node] = current_node
                        g_score[neighbor_node] = tentative_g_score
                        f_score[neighbor_node] = tentative_g_score + self._physics_heuristic(neighbor_node, goal_node_id)
                        
                        if neighbor_node not in open_set_hash:
                            heapq.heappush(open_set, (f_score[neighbor_node], neighbor_node))
                            open_set_hash.add(neighbor_node)
        
        return None  # No path found
    
    def _physics_heuristic(self, node: PathNode, goal_id: int) -> float:
        """
        Physics-aware heuristic for platformer movement
        
        Considers:
        - Euclidean distance (base cost)
        - Vertical movement penalties based on gravity
        - Surface type preferences
        - Movement state efficiency
        """
        goal_data = self.graph.nodes[goal_id]
        goal_pos = goal_data['position']
        
        dx = goal_pos[0] - node.position[0]
        dy = goal_pos[1] - node.position[1]
        
        # Base distance
        euclidean_dist = np.sqrt(dx*dx + dy*dy)
        
        # Vertical movement penalty (upward movement is expensive)
        vertical_penalty = 0.0
        if dy < 0:  # Goal is above current position
            height_diff = abs(dy)
            # Exponential penalty for height - jumping gets progressively harder
            vertical_penalty = height_diff * 2.0 * (1 + height_diff / self.MAX_JUMP_HEIGHT)
        else:
            # Downward movement is cheaper but still costs something
            vertical_penalty = abs(dy) * 0.3
        
        # Surface type penalty
        surface_penalty = self._get_surface_preference_penalty(node)
        
        # Movement state efficiency
        state_penalty = self._get_movement_state_penalty(node, dx, dy)
        
        # Jump height penalty (being mid-jump limits options)
        jump_penalty = node.jump_height * 0.5
        
        # Velocity consideration (maintaining momentum is good)
        velocity_bonus = self._get_velocity_bonus(node, dx, dy)
        
        total_heuristic = (euclidean_dist + vertical_penalty + surface_penalty + 
                          state_penalty + jump_penalty - velocity_bonus)
        
        return max(0.0, total_heuristic)
    
    def _get_surface_preference_penalty(self, node: PathNode) -> float:
        """Penalty based on surface type preferences"""
        node_data = self.graph.nodes[node.spatial_id]
        surface_type = node_data.get('surface_type', SurfaceType.FLOOR)
        
        # Prefer floors over walls/ceilings
        if surface_type == SurfaceType.FLOOR:
            return 0.0
        elif surface_type in [SurfaceType.WALL_LEFT, SurfaceType.WALL_RIGHT]:
            return 15.0  # Wall positions are less preferred
        elif surface_type == SurfaceType.CEILING:
            return 25.0  # Ceiling positions are least preferred
        else:
            return 5.0   # Other surface types
    
    def _get_movement_state_penalty(self, node: PathNode, dx: float, dy: float) -> float:
        """Penalty based on movement state efficiency for the required direction"""
        # Movement states: 0=immobile, 1=running, 2=sliding, 3=jumping, 4=falling, 5=wall_sliding
        if node.movement_state == 0:  # Immobile
            return 10.0 if abs(dx) > 0 else 0.0
        elif node.movement_state == 1:  # Running
            return 0.0 if abs(dx) > abs(dy) else 5.0
        elif node.movement_state == 3:  # Jumping
            return 0.0 if dy < 0 else 8.0  # Good for upward movement
        elif node.movement_state == 4:  # Falling
            return 0.0 if dy > 0 else 12.0  # Good for downward movement
        elif node.movement_state == 5:  # Wall sliding
            return 0.0 if abs(dx) > 0 else 8.0  # Good for horizontal movement from walls
        else:
            return 3.0  # Default small penalty for other states
    
    def _get_velocity_bonus(self, node: PathNode, dx: float, dy: float) -> float:
        """Bonus for maintaining beneficial velocity"""
        vx, vy = node.velocity
        
        # Bonus if velocity is aligned with desired direction
        if abs(dx) > 0.1:
            x_alignment = (vx * dx) / (abs(vx) + 0.1) / (abs(dx) + 0.1)
            x_bonus = max(0, x_alignment) * 5.0
        else:
            x_bonus = 0.0
        
        if abs(dy) > 0.1:
            y_alignment = (vy * dy) / (abs(vy) + 0.1) / (abs(dy) + 0.1)
            y_bonus = max(0, y_alignment) * 3.0
        else:
            y_bonus = 0.0
        
        return x_bonus + y_bonus
    
    def _get_possible_transitions(self, current_node: PathNode, neighbor_id: int, 
                                 edge_data: dict) -> List[Tuple[PathNode, float]]:
        """
        Generate possible state transitions for moving to neighbor
        
        Returns list of (neighbor_node, transition_cost) tuples
        """
        neighbor_pos = self.graph.nodes[neighbor_id]['position']
        move_type = edge_data.get('move_type', MovementType.WALK.value)
        
        transitions = []
        
        # Calculate movement vector
        dx = neighbor_pos[0] - current_node.position[0]
        dy = neighbor_pos[1] - current_node.position[1]
        
        # Determine possible movement states after transition
        if move_type == MovementType.WALK.value:
            # Walking transition
            new_state = 1 if abs(dx) > 0.1 else 0
            new_velocity = (np.sign(dx) * min(abs(dx) * 0.1, self.MAX_HORIZONTAL_SPEED), 0)
            cost = self.MOVEMENT_COSTS[MovementType.WALK]
            
        elif move_type == MovementType.JUMP.value:
            # Jumping transition
            new_state = 3
            jump_height = current_node.jump_height + abs(dy) if dy < 0 else max(0, current_node.jump_height - abs(dy))
            new_velocity = (np.sign(dx) * min(abs(dx) * 0.08, self.MAX_HORIZONTAL_SPEED), 
                          -2.0 if dy < 0 else min(0, current_node.velocity[1]))
            cost = self.MOVEMENT_COSTS[MovementType.JUMP]
            
        elif move_type == MovementType.WALL_JUMP.value:
            # Wall jumping transition
            new_state = 3
            jump_height = abs(dy) if dy < 0 else 0
            new_velocity = (np.sign(dx) * 2.0, -1.4 if dy < 0 else 0)
            cost = self.MOVEMENT_COSTS[MovementType.WALL_JUMP]
            
        elif move_type == MovementType.FALL.value:
            # Falling transition
            new_state = 4
            jump_height = 0
            new_velocity = (current_node.velocity[0] * 0.9, max(current_node.velocity[1] + self.GRAVITY_FALL, -6.0))
            cost = self.MOVEMENT_COSTS[MovementType.FALL]
            
        else:
            # Default movement
            new_state = 1
            new_velocity = (0, 0)
            cost = 1.0
            jump_height = 0
        
        # Create neighbor node
        neighbor_node = PathNode(
            spatial_id=neighbor_id,
            position=neighbor_pos,
            movement_state=new_state,
            jump_height=jump_height,
            velocity=new_velocity
        )
        
        transitions.append((neighbor_node, cost))
        return transitions
    
    def _is_physics_valid(self, current_node: PathNode, neighbor_node: PathNode, 
                         edge_data: dict) -> bool:
        """
        Check if transition is valid according to N++ physics
        
        Validates:
        - Jump height constraints
        - Velocity limits
        - Movement state transitions
        """
        # Check jump height constraint
        if neighbor_node.jump_height > self.MAX_JUMP_HEIGHT:
            return False
        
        # Check velocity limits
        vx, vy = neighbor_node.velocity
        if abs(vx) > self.MAX_HORIZONTAL_SPEED + 0.1:  # Small tolerance
            return False
        if abs(vy) > self.MAX_SURVIVABLE_IMPACT:  # Terminal velocity check
            return False
        
        # Check movement state transitions
        current_state = current_node.movement_state
        new_state = neighbor_node.movement_state
        
        # Validate state transition logic
        if current_state == 6:  # Dead state
            return False
        
        # Don't allow impossible state transitions
        if current_state == 0 and new_state == 5:  # Can't go from immobile to wall sliding
            return False
        
        return True
    
    def _calculate_total_cost(self, current_node: PathNode, neighbor_node: PathNode,
                             edge_data: dict, base_cost: float, avoid_hazards: bool,
                             collect_gold: bool) -> float:
        """Calculate total movement cost including penalties and bonuses"""
        
        # Base movement cost
        total_cost = base_cost * edge_data.get('weight', 1.0)
        
        # Hazard avoidance penalty
        if avoid_hazards:
            hazard_penalty = self._calculate_hazard_penalty(neighbor_node, edge_data)
            if hazard_penalty == float('inf'):
                return float('inf')
            total_cost += hazard_penalty
        
        # Gold collection bonus
        if collect_gold:
            gold_bonus = self._calculate_gold_bonus(neighbor_node)
            total_cost -= gold_bonus
        
        # Physics efficiency bonus/penalty
        physics_modifier = self._calculate_physics_modifier(current_node, neighbor_node)
        total_cost *= physics_modifier
        
        return max(0.1, total_cost)  # Ensure positive cost
    
    def _calculate_hazard_penalty(self, node: PathNode, edge_data: dict) -> float:
        """Calculate penalty for hazard proximity"""
        hazard_proximity = edge_data.get('hazard_proximity', 0)
        
        if hazard_proximity > 0.9:
            return float('inf')  # Impassable
        elif hazard_proximity > 0.7:
            return 100.0  # Very dangerous
        elif hazard_proximity > 0.4:
            return 50.0   # Dangerous
        elif hazard_proximity > 0.1:
            return 10.0   # Risky
        else:
            return 0.0    # Safe
    
    def _calculate_gold_bonus(self, node: PathNode) -> float:
        """Calculate bonus for gold collection opportunities"""
        node_data = self.graph.nodes[node.spatial_id]
        has_gold = node_data.get('has_gold', False)
        return 20.0 if has_gold else 0.0
    
    def _calculate_physics_modifier(self, current_node: PathNode, neighbor_node: PathNode) -> float:
        """Calculate cost modifier based on physics efficiency"""
        # Bonus for maintaining momentum
        momentum_bonus = 1.0
        
        # Check if movement maintains or builds momentum efficiently
        current_vel = np.array(current_node.velocity)
        neighbor_vel = np.array(neighbor_node.velocity)
        
        # Reward smooth velocity changes
        vel_change = np.linalg.norm(neighbor_vel - current_vel)
        if vel_change < 1.0:
            momentum_bonus = 0.9  # Small bonus for smooth movement
        elif vel_change > 3.0:
            momentum_bonus = 1.2  # Penalty for abrupt changes
        
        # Efficiency bonus for appropriate movement states
        state_efficiency = 1.0
        if neighbor_node.movement_state == 1:  # Running
            state_efficiency = 0.95
        elif neighbor_node.movement_state == 3:  # Jumping
            state_efficiency = 1.1
        elif neighbor_node.movement_state == 4:  # Falling
            state_efficiency = 0.85
        
        return momentum_bonus * state_efficiency
    
    def _reconstruct_path(self, came_from: Dict[PathNode, PathNode], 
                         current_node: PathNode) -> List[int]:
        """Reconstruct path from goal to start"""
        path = [current_node.spatial_id]
        
        while current_node in came_from:
            current_node = came_from[current_node]
            path.append(current_node.spatial_id)
        
        return path[::-1]

class MultiObjectivePathfinder:
    """Multi-objective pathfinding with physics awareness"""
    
    def __init__(self, pathfinder: PlatformerAStar):
        self.pathfinder = pathfinder
        
    def find_complete_level_path(self, start_node_id: int, switch_node_id: Optional[int],
                                exit_node_id: int, gold_nodes: Optional[List[int]] = None,
                                ninja_state: int = 0, ninja_velocity: Tuple[float, float] = (0, 0)) -> Optional[List[int]]:
        """
        Find optimal path through level objectives with physics awareness
        
        Args:
            start_node_id: Starting position
            switch_node_id: Exit switch position (None if no switch)
            exit_node_id: Level exit position
            gold_nodes: Optional gold collection targets
            ninja_state: Current ninja movement state
            ninja_velocity: Current ninja velocity
            
        Returns:
            Complete path through all objectives, or None if impossible
        """
        current_location = start_node_id
        full_path = [start_node_id]
        current_state = ninja_state
        current_velocity = ninja_velocity
        
        # Path to switch (if required)
        if switch_node_id is not None:
            switch_path = self.pathfinder.find_path(
                current_location, switch_node_id, 
                ninja_state=current_state, ninja_velocity=current_velocity
            )
            if not switch_path:
                return None
            
            full_path.extend(switch_path[1:])
            current_location = switch_node_id
            # Update estimated state after reaching switch
            current_state = 0  # Assume stationary after reaching switch
            current_velocity = (0, 0)
        
        # Path to exit
        exit_path = self.pathfinder.find_path(
            current_location, exit_node_id,
            ninja_state=current_state, ninja_velocity=current_velocity
        )
        if not exit_path:
            return None
        
        full_path.extend(exit_path[1:])
        
        # Optional gold optimization (simplified greedy approach)
        if gold_nodes:
            optimized_path = self._optimize_gold_collection(full_path, gold_nodes)
            return optimized_path if optimized_path else full_path
        
        return full_path
    
    def _optimize_gold_collection(self, base_path: List[int], 
                                 gold_nodes: List[int]) -> Optional[List[int]]:
        """
        Optimize path for gold collection using greedy insertion
        
        This is a simplified approach - a full implementation would use
        more sophisticated algorithms like dynamic programming or TSP solvers
        """
        if not gold_nodes or not base_path:
            return base_path
        
        # For now, return the base path without gold optimization
        # A full implementation would insert gold collection detours
        # where they don't significantly increase path cost
        return base_path
