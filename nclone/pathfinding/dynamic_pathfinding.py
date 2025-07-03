import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional
import heapq

from .utils import Enemy
from ..ninja import NINJA_RADIUS

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
        self.prediction_sample_rate = 5 # frames, for sampling moving enemies
        self.static_prediction_sample_rate = 30 # frames, for static enemies

        if enemies:
            self._precompute_predictions()
        else:
            print("Warning: EnemyPredictor initialized with no enemies.")
    
    def _precompute_predictions(self, max_time_prediction: int = 2000):
        """Pre-calculate enemy positions for efficiency up to max_time_prediction frames."""
        for enemy in self.enemies:
            if enemy.type == enemy.THWUMP:
                self._predict_thwump_movement(enemy, max_time_prediction)
            elif enemy.type in [enemy.DRONE_CLOCKWISE, enemy.DRONE_COUNTER_CLOCKWISE]:
                self._predict_drone_movement(enemy, max_time_prediction)
            elif enemy.type == enemy.DEATH_BALL:
                self._predict_death_ball_movement(enemy, max_time_prediction)
            elif enemy.type in [enemy.TOGGLE_MINE, enemy.TOGGLE_MINE_TOGGLED]:
                self._predict_static_enemy(enemy, max_time_prediction)
            else:
                # For other enemy types, assume static or simple linear movement
                self._predict_static_enemy(enemy, max_time_prediction)
    
    def _predict_thwump_movement(self, thwump: Enemy, max_time: int):
        """Predict thwump movement: rest, smash, wait, return cycle."""
        predictions_for_thwump: List[Tuple[int, Tuple[float,float], float]] = []
        
        # Thwump cycle parameters (example values, can be tuned or derive from Enemy object if available)
        rest_at_origin_frames = thwump.properties.get('rest_frames', 120) # 2 seconds
        smash_duration_frames = thwump.properties.get('smash_duration', 15) # 0.25 seconds
        wait_extended_frames = thwump.properties.get('wait_extended', 60)  # 1 second
        return_duration_frames = thwump.properties.get('return_duration', 30) # 0.5 seconds
        
        cycle_duration_frames = rest_at_origin_frames + smash_duration_frames + wait_extended_frames + return_duration_frames
        
        smash_vector = thwump.direction if thwump.direction != (0,0) else (0, 1) # Default smash down
        # Infer smash_distance from direction vector if it encodes magnitude, or use a default
        smash_distance = thwump.properties.get('smash_distance', 96) # e.g., 4 tiles
        if np.linalg.norm(smash_vector) > 1e-3: # Normalize direction if it's not zero vector
            smash_unit_vector = np.array(smash_vector) / np.linalg.norm(smash_vector)
        else: # Should not happen if direction is meaningful
            smash_unit_vector = np.array([0,1]) # Default down

        origin_pos = np.array(thwump.origin)
        extended_pos = origin_pos + smash_unit_vector * smash_distance

        if cycle_duration_frames == 0: # Should not happen with positive durations
            for t in range(0, max_time, self.prediction_sample_rate):
                 predictions_for_thwump.append((t, thwump.origin, thwump.radius))
            self.predictions[thwump.id] = predictions_for_thwump
            return

        for t in range(0, max_time, self.prediction_sample_rate):
            time_in_cycle = t % cycle_duration_frames
            current_pos_np = np.array(origin_pos) # Default to origin

            if time_in_cycle < rest_at_origin_frames:
                # Resting at origin
                current_pos_np = origin_pos
            elif time_in_cycle < rest_at_origin_frames + smash_duration_frames:
                # Smashing
                progress = (time_in_cycle - rest_at_origin_frames) / float(smash_duration_frames)
                current_pos_np = origin_pos + (extended_pos - origin_pos) * progress
            elif time_in_cycle < rest_at_origin_frames + smash_duration_frames + wait_extended_frames:
                # Waiting at extended position
                current_pos_np = extended_pos
            else:
                # Returning to origin
                progress = (time_in_cycle - (rest_at_origin_frames + smash_duration_frames + wait_extended_frames)) / float(return_duration_frames)
                current_pos_np = extended_pos + (origin_pos - extended_pos) * progress
            
            predictions_for_thwump.append((t, tuple(current_pos_np), thwump.radius))
        
        self.predictions[thwump.id] = predictions_for_thwump

    def _predict_drone_movement(self, drone: Enemy, max_time: int):
        """Predict drone movement pattern (stub). Assumes stationary if no patrol path."""
        # TODO: Implement path following if drone.patrol_path exists and is populated
        # For now, if drone.properties contains 'patrol_nodes' and 'patrol_speed' use that.
        patrol_nodes = drone.properties.get('patrol_nodes', [])
        patrol_speed = drone.properties.get('patrol_speed', 2.0) # pixels/frame

        if not patrol_nodes:
            print(f"Warning: Drone {drone.id} has no patrol_nodes. Predicting as static at origin {drone.origin}.")
            self.predictions[drone.id] = [(t, drone.origin, drone.radius) for t in range(0, max_time, self.prediction_sample_rate)]
            return
        
        # Ensure origin is part of patrol if not explicitly, or start from first patrol node
        all_patrol_points = [np.array(drone.origin)] + [np.array(pn) for pn in patrol_nodes]
        if drone.properties.get('patrol_is_loop', True):
            all_patrol_points.append(np.array(all_patrol_points[0])) # Close the loop

        predictions_for_drone: List[Tuple[int, Tuple[float,float], float]] = []
        current_patrol_idx = 0
        pos_on_patrol = np.array(all_patrol_points[0])
        time_elapsed_on_segment = 0

        for t in range(0, max_time, self.prediction_sample_rate):
            if current_patrol_idx >= len(all_patrol_points) -1: # Reached end of non-loop or finished one loop cycle
                if drone.properties.get('patrol_is_loop', True):
                    current_patrol_idx = 0 # Reset for loop
                    pos_on_patrol = np.array(all_patrol_points[0])
                    time_elapsed_on_segment = 0
                else: # End of non-looping patrol, stay at last point
                    predictions_for_drone.append((t, tuple(pos_on_patrol), drone.radius))
                    continue
            
            segment_start = np.array(all_patrol_points[current_patrol_idx])
            segment_end = np.array(all_patrol_points[current_patrol_idx+1])
            segment_vector = segment_end - segment_start
            segment_length = np.linalg.norm(segment_vector)

            if segment_length < 1e-3: # Segment is a point, move to next
                time_to_traverse_segment = 0
            else:
                time_to_traverse_segment = segment_length / patrol_speed
            
            # Advance drone simulation by self.prediction_sample_rate
            # This is a simplified lookahead for sampling, actual position is calculated per 't'
            # To get position at time 't': simulate from t=0 up to 't' along patrol path.
            # This can be slow if done per 't'. Instead, track progress along patrol.

            # This loop advances the drone state to be *at or after* time 't'
            # This is not quite right. We need to calculate pos for *each* 't' in the outer loop.
            # Let's recalculate drone's position along its full patrol for each time t.
            # This is less efficient but more accurate for sampling.

            sim_t = 0
            current_sim_pos = np.array(all_patrol_points[0])
            current_segment_idx_for_sim_t = 0
            dist_along_segment_for_sim_t = 0.0

            while sim_t < t and current_segment_idx_for_sim_t < len(all_patrol_points) - 1:
                _start = np.array(all_patrol_points[current_segment_idx_for_sim_t])
                _end = np.array(all_patrol_points[current_segment_idx_for_sim_t + 1])
                _vec = _end - _start
                _len = np.linalg.norm(_vec)
                _time_for_seg = (_len / patrol_speed) if patrol_speed > 0 else float('inf')

                if sim_t + _time_for_seg > t: # Target time 't' is within this segment
                    remaining_time_in_t = t - sim_t
                    dist_to_move = remaining_time_in_t * patrol_speed
                    current_sim_pos = _start + (_vec / _len) * dist_to_move if _len > 0 else _start
                    sim_t = t # Reached target time
                    break
                else: # Drone completes this segment before or at time 't'
                    current_sim_pos = _end
                    sim_t += _time_for_seg
                    current_segment_idx_for_sim_t += 1
                    if drone.properties.get('patrol_is_loop', True) and current_segment_idx_for_sim_t == len(all_patrol_points) -1:
                        current_segment_idx_for_sim_t = 0 # Loop back
            
            predictions_for_drone.append((t, tuple(current_sim_pos), drone.radius))

        self.predictions[drone.id] = predictions_for_drone

    def _predict_death_ball_movement(self, death_ball: Enemy, max_time: int):
        """Predict death ball movement using linear motion (stub, no wall bounce/seeking)."""
        # TODO: Implement wall bouncing using CollisionChecker.
        # TODO: Basic seeking if player path is known or target point given.
        predictions_for_death_ball = []
        
        # Death balls accelerate toward ninja and bounce off walls
        # For pathfinding, assume they continue current trajectory
        current_pos = [death_ball.xpos, death_ball.ypos]
        current_vel = [death_ball.xspeed, death_ball.yspeed]
        
        for t in range(0, max_time, 5):  # Sample every 5 frames
            # Simple physics simulation for death ball
            # In reality, death balls seek the ninja and bounce off walls
            current_pos[0] += current_vel[0] * 5
            current_pos[1] += current_vel[1] * 5
            
            predictions_for_death_ball.append((t, tuple(current_pos), death_ball.radius))
        
        self.predictions[death_ball.id] = predictions_for_death_ball
    
    def _predict_static_enemy(self, enemy: Enemy, max_time: int):
        """Predict static enemy positions (mines, etc.)."""
        # Static enemies don't move, so position is constant
        predictions_for_enemy = []
        
        for t in range(0, max_time, self.static_prediction_sample_rate):  # Use specific sample rate
            predictions_for_enemy.append((t, (enemy.xpos, enemy.ypos), enemy.radius))
        
        self.predictions[enemy.id] = predictions_for_enemy

    def get_enemy_positions_at_time(self, time: int) -> List[Tuple[Tuple[float, float], float]]:
        """Get all predicted enemy positions and radii at a specific time, with interpolation."""
        active_enemies_at_time: List[Tuple[Tuple[float, float], float]] = []
        for enemy_id in self.predictions:
            enemy_pred_list = self.predictions[enemy_id]
            if not enemy_pred_list: continue

            # Find prediction samples around 'time' for interpolation
            p_before: Optional[Tuple[int, Tuple[float,float], float]] = None
            p_after: Optional[Tuple[int, Tuple[float,float], float]] = None

            for pred_sample in enemy_pred_list:
                if pred_sample[0] <= time:
                    p_before = pred_sample
                if pred_sample[0] >= time and p_after is None: # First sample at or after time
                    p_after = pred_sample
            
            current_pos: Optional[Tuple[float,float]] = None
            current_radius: Optional[float] = None

            if p_before and p_after:
                if p_before[0] == p_after[0]: # Exact match or time beyond last sample
                    current_pos = p_before[1]
                    current_radius = p_before[2]
                elif time == p_before[0]: # Exact match on before
                    current_pos = p_before[1]
                    current_radius = p_before[2]
                elif time == p_after[0]: # Exact match on after
                    current_pos = p_after[1]
                    current_radius = p_after[2]
                else: # Interpolate
                    t_diff = float(p_after[0] - p_before[0])
                    if t_diff <= 0: # Should not happen if sorted and distinct
                        current_pos = p_before[1]
                        current_radius = p_before[2]
                    else:
                        alpha = (time - p_before[0]) / t_diff
                        pos_before_np = np.array(p_before[1])
                        pos_after_np = np.array(p_after[1])
                        interp_pos_np = pos_before_np + (pos_after_np - pos_before_np) * alpha
                        current_pos = tuple(interp_pos_np)
                        # Radius usually doesn't change, use p_before's radius
                        current_radius = p_before[2] 
            elif p_before: # time is after the last prediction sample
                current_pos = p_before[1]
                current_radius = p_before[2]
            elif p_after: # time is before the first prediction sample
                current_pos = p_after[1]
                current_radius = p_after[2]
            # else: no predictions for this enemy (should not happen if list not empty)
            
            if current_pos and current_radius is not None:
                active_enemies_at_time.append((current_pos, current_radius))
        return active_enemies_at_time

class DynamicPathfinder:
    """Pathfinding that considers time and moving obstacles using a time-expanded graph approach."""
    
    def __init__(self, static_graph: nx.DiGraph, 
                 enemy_predictor: EnemyPredictor):
        self.static_graph = static_graph
        self.enemy_predictor = enemy_predictor
        self.time_resolution = 10  # frames (e.g., ~0.16 seconds at 60fps)
        self.max_temporal_depth = 1500 # Max frames into future to search (e.g. 25 seconds)
        
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
        # Basic heuristic: Euclidean distance in space.
        current_pos_data = self.static_graph.nodes.get(tnode.spatial_node_id)
        goal_pos_data = self.static_graph.nodes.get(goal_spatial_node_id)

        if not current_pos_data or not goal_pos_data:
            return float('inf') # Should not happen if nodes are valid

        current_pos = current_pos_data['position']
        goal_pos = goal_pos_data['position']
        
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        spatial_distance = np.sqrt(dx*dx + dy*dy)
        
        # If g_score is time, then this should be estimated time.
        # If g_score is abstract cost, this should be abstract cost.
        # Assume edge weights ('frames') are time. So heuristic should be estimated time.
        avg_travel_speed = 2.5 # pixels per frame (estimated average speed)
        # Ensure avg_travel_speed is not zero to avoid division by zero
        if avg_travel_speed <= 1e-6:
            return spatial_distance # Fallback to distance if speed is zero/negligible
        return spatial_distance / avg_travel_speed

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
        node_data = self.static_graph.nodes.get(node_id)
        if not node_data:
            return False # Node doesn't exist
            
        node_pos = node_data['position']
        
        enemy_positions_and_radii = self.enemy_predictor.get_enemy_positions_at_time(time)
        
        for enemy_pos, enemy_radius in enemy_positions_and_radii:
            dist_sq = (node_pos[0] - enemy_pos[0])**2 + (node_pos[1] - enemy_pos[1])**2
            min_safe_dist_sq = (enemy_radius + NINJA_RADIUS + 2)**2 # 2 pixel buffer
            if dist_sq < min_safe_dist_sq:
                # print(f"Node {node_id} at pos {node_pos} is NOT safe at time {time} due to enemy at {enemy_pos} (r={enemy_radius})")
                return False
        
        return True

    # _build_temporal_graph from prompt is not explicitly built if using A* directly on conceptual graph.
    # The A* search explores the temporal graph implicitly.
