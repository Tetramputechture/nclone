import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional

from .utils import Enemy # Assuming Enemy class is in utils.py

class TemporalNode:
    """Node with time component for dynamic obstacle avoidance"""
    def __init__(self, spatial_node_id: int, time: int):
        self.spatial_node_id = spatial_node_id
        self.time = time
    
    # For hashing and equality in sets/dictionary keys for NetworkX
    def __hash__(self):
        return hash((self.spatial_node_id, self.time))

    def __eq__(self, other):
        if not isinstance(other, TemporalNode):
            return NotImplemented
        return (self.spatial_node_id, self.time) == (other.spatial_node_id, other.time)
    
    def __lt__(self, other): # For heapq if not using tuples directly
        if not isinstance(other, TemporalNode):
            return NotImplemented
        return (self.spatial_node_id, self.time) < (other.spatial_node_id, other.time)

    def __repr__(self):
        return f"TNode({self.spatial_node_id}, t={self.time})"

class EnemyPredictor:
    """Predicts future positions of enemies"""
    
    def __init__(self, enemies: List[Enemy]):
        self.enemies = enemies
        # Predictions: Dict[enemy_id, List[Tuple[time, position_tuple, radius_float]]]
        self.predictions: Dict[int, List[Tuple[int, Tuple[float,float], float]]] = {}
        if enemies:
            self._precompute_predictions()
        else:
            print("Warning: EnemyPredictor initialized with no enemies.")
    
    def _precompute_predictions(self, max_time_prediction: int = 2000):
        """Pre-calculate enemy positions for efficiency up to max_time_prediction frames."""
        for enemy in self.enemies:
            if enemy.type == 'thwump':
                self._predict_thwump_movement(enemy, max_time_prediction)
            elif enemy.type == 'drone':
                self._predict_drone_movement(enemy, max_time_prediction)
            elif enemy.type == 'death_ball':
                self._predict_death_ball_movement(enemy, max_time_prediction)
            else:
                print(f"Warning: Unknown enemy type {enemy.type} for prediction.")
    
    def _predict_thwump_movement(self, thwump: Enemy, max_time: int):
        """Predict thwump movement pattern. Simplified: assumes periodic patrol if not player-tracking."""
        # This is a highly simplified Thwump prediction. Real Thwumps react to player.
        # For pathfinding, often a worst-case or predictable patrol is assumed.
        # Assuming a simple back-and-forth patrol for this stub.
        predictions_for_thwump: List[Tuple[int, Tuple[float,float], float]] = []
        patrol_distance = 100 # pixels
        speed = 20/7 # pixels/frame (from prompt, forward speed)
        # Using a simpler cycle: move patrol_distance then return.
        # Time to move one way = patrol_distance / speed
        one_way_frames = int(patrol_distance / speed) if speed > 0 else 100 # Default if speed is 0
        cycle_duration_frames = one_way_frames * 2 # Move out and back

        if cycle_duration_frames == 0: # Avoid division by zero if speed is very high or distance 0
            # Thwump is stationary
            for t in range(0, max_time, 10): # Sample every 10 frames
                 predictions_for_thwump.append((t, thwump.origin, thwump.radius))
            self.predictions[thwump.id] = predictions_for_thwump
            return

        for t in range(0, max_time, 10): # Sample every 10 frames
            time_in_cycle = t % cycle_duration_frames
            current_pos = list(thwump.origin)
            direction_vector = thwump.direction if thwump.direction != (0,0) else (1,0) # Default direction if not set

            if time_in_cycle < one_way_frames:
                # Moving along direction vector
                displacement = speed * time_in_cycle
            else:
                # Moving back
                displacement_out = speed * one_way_frames
                displacement_back = speed * (time_in_cycle - one_way_frames)
                displacement = displacement_out - displacement_back
            
            current_pos[0] += direction_vector[0] * displacement
            current_pos[1] += direction_vector[1] * displacement
            predictions_for_thwump.append((t, tuple(current_pos), thwump.radius))
        
        self.predictions[thwump.id] = predictions_for_thwump

    def _predict_drone_movement(self, drone: Enemy, max_time: int):
        """Predict drone movement pattern (stub)."""
        print(f"Warning: Drone movement prediction for enemy {drone.id} is a stub.")
        # Drones might follow fixed paths or patrol. Assume stationary for stub.
        self.predictions[drone.id] = [(t, drone.origin, drone.radius) for t in range(0, max_time, 10)]

    def _predict_death_ball_movement(self, death_ball: Enemy, max_time: int):
        """Predict death ball movement (stub)."""
        print(f"Warning: Death ball movement prediction for enemy {death_ball.id} is a stub.")
        # Death balls might roll or follow complex paths. Assume stationary for stub.
        self.predictions[death_ball.id] = [(t, death_ball.origin, death_ball.radius) for t in range(0, max_time, 10)]

    def get_enemy_positions_at_time(self, time: int) -> List[Tuple[Tuple[float, float], float]]:
        """Get all predicted enemy positions and radii at a specific time."""
        active_enemies_at_time: List[Tuple[Tuple[float, float], float]] = []
        for enemy_id in self.predictions:
            enemy_pred_list = self.predictions[enemy_id]
            if not enemy_pred_list: continue

            # Find the closest prediction time <= current time
            # This assumes predictions are sorted by time, which they should be from _precompute
            # A more robust way would be to interpolate between prediction points or find exact match.
            # For simplicity, find the prediction for the sampled time step closest to 'time'.
            # Example: if predictions are every 10 frames (0, 10, 20...) and time is 13, use prediction for t=10.
            best_match = None
            # Find the latest prediction sample that is <= time
            for pred_time, pos, radius in reversed(enemy_pred_list):
                if pred_time <= time:
                    best_match = (pos, radius)
                    break
            if best_match is None and enemy_pred_list: # If time is before first prediction, use first one
                best_match = (enemy_pred_list[0][1], enemy_pred_list[0][2])

            if best_match:
                active_enemies_at_time.append(best_match)
        return active_enemies_at_time

class DynamicPathfinder:
    """Pathfinding that considers time and moving obstacles using a time-expanded graph approach."""
    
    def __init__(self, static_graph: nx.DiGraph, 
                 enemy_predictor: EnemyPredictor):
        self.static_graph = static_graph
        self.enemy_predictor = enemy_predictor
        self.time_resolution = 30  # frames (e.g., 0.5 seconds at 60fps)
        self.max_temporal_depth = 1000 # Max frames into future to search
        
    def find_safe_path(self, start_node_id: int, goal_node_id: int,
                      start_time: int = 0) -> Optional[List[Tuple[int, int]]]: # Returns list of (node_id, time) tuples
        """Find path that avoids moving enemies using A* on a conceptual temporal graph."""

        # Temporal A* components
        # Priority queue stores: (f_score, spatial_node_id, time)
        # Using TemporalNode for state representation in sets and dicts
        start_tnode = TemporalNode(start_node_id, start_time)

        open_set: List[Tuple[float, TemporalNode]] = [] # (f_score, TemporalNode)
        came_from: Dict[TemporalNode, TemporalNode] = {}
        
        g_score: Dict[TemporalNode, float] = {start_tnode: 0}
        f_score: Dict[TemporalNode, float] = {start_tnode: self._temporal_heuristic(start_tnode, goal_node_id)}
        
        heapq.heappush(open_set, (f_score[start_tnode], start_tnode))
        open_set_hash: Set[TemporalNode] = {start_tnode}

        while open_set:
            _, current_tnode = heapq.heappop(open_set)
            open_set_hash.remove(current_tnode)

            if current_tnode.spatial_node_id == goal_node_id:
                # Goal reached. Reconstruct path.
                # We accept reaching the goal spatial node at any safe time.
                return self._reconstruct_temporal_path(came_from, current_tnode)
            
            if current_tnode.time > start_time + self.max_temporal_depth:
                continue # Exceeded max search time

            # Explore neighbors in space and time
            # Option 1: Wait at the current_tnode.spatial_node_id
            next_time_wait = current_tnode.time + self.time_resolution
            wait_tnode = TemporalNode(current_tnode.spatial_node_id, next_time_wait)
            if self._is_safe_at_time(wait_tnode.spatial_node_id, wait_tnode.time):
                cost_wait = float(self.time_resolution) # Cost of waiting
                tentative_g_wait = g_score[current_tnode] + cost_wait
                if tentative_g_wait < g_score.get(wait_tnode, float('inf')):
                    came_from[wait_tnode] = current_tnode
                    g_score[wait_tnode] = tentative_g_wait
                    f_score[wait_tnode] = tentative_g_wait + self._temporal_heuristic(wait_tnode, goal_node_id)
                    if wait_tnode not in open_set_hash:
                        heapq.heappush(open_set, (f_score[wait_tnode], wait_tnode))
                        open_set_hash.add(wait_tnode)
            
            # Option 2: Move to spatial neighbors
            if current_tnode.spatial_node_id not in self.static_graph.adj:
                continue

            for neighbor_spatial_id in self.static_graph.neighbors(current_tnode.spatial_node_id):
                edge_data = self.static_graph.get_edge_data(current_tnode.spatial_node_id, neighbor_spatial_id)
                if not edge_data: continue

                # Estimate travel time. 'frames' field from prompt, or use 'weight' as proxy.
                # 'weight' in static graph is often cost, not time. Need a 'time_cost' or 'frames' attribute.
                # Defaulting to a fixed time per edge if not specified, or using time_resolution.
                travel_time_frames = edge_data.get('frames', self.time_resolution) 
                # Ensure travel_time is somewhat realistic, e.g. at least 1 if not 0.
                travel_time_frames = max(1, int(travel_time_frames))

                arrival_time = current_tnode.time + travel_time_frames
                neighbor_tnode = TemporalNode(neighbor_spatial_id, arrival_time)

                if self._is_safe_at_time(neighbor_spatial_id, arrival_time):
                    # Cost of movement: use static graph edge weight, or travel_time itself if cost is time.
                    move_cost = edge_data.get('weight', float(travel_time_frames))
                    tentative_g_move = g_score[current_tnode] + move_cost

                    if tentative_g_move < g_score.get(neighbor_tnode, float('inf')):
                        came_from[neighbor_tnode] = current_tnode
                        g_score[neighbor_tnode] = tentative_g_move
                        f_score[neighbor_tnode] = tentative_g_move + self._temporal_heuristic(neighbor_tnode, goal_node_id)
                        if neighbor_tnode not in open_set_hash:
                            heapq.heappush(open_set, (f_score[neighbor_tnode], neighbor_tnode))
                            open_set_hash.add(neighbor_tnode)
        
        print(f"No safe temporal path found from {start_node_id} to {goal_node_id} starting at t={start_time}")
        return None

    def _temporal_heuristic(self, tnode: TemporalNode, goal_spatial_node_id: int) -> float:
        """Heuristic for temporal A*. Estimates cost from tnode to goal_spatial_node_id."""
        # Basic heuristic: Euclidean distance in space. Time is not directly part of heuristic here,
        # as we want the spatially shortest path that is also safe over time.
        # A more advanced heuristic could estimate minimum time to reach goal_spatial_node_id.
        current_pos = self.static_graph.nodes[tnode.spatial_node_id]['position']
        goal_pos = self.static_graph.nodes[goal_spatial_node_id]['position']
        
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        spatial_distance = np.sqrt(dx*dx + dy*dy)
        
        # Heuristic should ideally be in same units as g_score (cost).
        # If g_score is time, then this should be estimated time.
        # If g_score is abstract cost, this should be abstract cost.
        # Assuming cost is roughly proportional to distance / speed.
        # For simplicity, using spatial distance as a proxy for cost.
        # This is admissible if edge costs are >= 0.
        return spatial_distance / self.static_graph.graph.get('avg_speed', 3.0) # Normalize by an assumed avg speed

    def _reconstruct_temporal_path(self, came_from: Dict[TemporalNode, TemporalNode], 
                                   current_tnode: TemporalNode) -> List[Tuple[int, int]]:
        path = [(current_tnode.spatial_node_id, current_tnode.time)]
        temp_node = current_tnode
        while temp_node in came_from:
            temp_node = came_from[temp_node]
            path.append((temp_node.spatial_node_id, temp_node.time))
        return path[::-1]

    def _is_safe_at_time(self, node_id: int, time: int) -> bool:
        """Check if a spatial node_id is safe from enemies at a given time."""
        if node_id not in self.static_graph.nodes:
            return False # Node doesn't exist
            
        node_pos = self.static_graph.nodes[node_id]['position']
        agent_radius = 12 # Approximate radius of the player agent
        
        enemy_positions_and_radii = self.enemy_predictor.get_enemy_positions_at_time(time)
        
        for enemy_pos, enemy_radius in enemy_positions_and_radii:
            dist_sq = (node_pos[0] - enemy_pos[0])**2 + (node_pos[1] - enemy_pos[1])**2
            # Sum of radii for collision detection
            # Adding a small safety margin to enemy_radius or agent_radius
            # The prompt used enemy_radius + 20, which is a large margin.
            # Using sum of radii + small buffer.
            min_safe_dist_sq = (enemy_radius + agent_radius + 5)**2 # 5 pixel buffer
            if dist_sq < min_safe_dist_sq:
                # print(f"Node {node_id} at pos {node_pos} is NOT safe at time {time} due to enemy at {enemy_pos} (r={enemy_radius})")
                return False
        
        return True

    # _build_temporal_graph from prompt is not explicitly built if using A* directly on conceptual graph.
    # The A* search explores the temporal graph implicitly.
