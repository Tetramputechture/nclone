import networkx as nx
from typing import List, Tuple, Dict, Optional
import math 

from .surface_parser import Surface, SurfaceType
from .utils import CollisionChecker, PathfindingUtils
from ..ninja import (
    NINJA_RADIUS,
    GRAVITY_FALL,
    GRAVITY_JUMP,
    AIR_ACCEL,
    DRAG_REGULAR,
    MAX_HOR_SPEED,
    MAX_JUMP_DURATION,
    MAX_SURVIVABLE_IMPACT,
    JUMP_FLAT_GROUND_Y,
    JUMP_WALL_REGULAR_Y,
    JUMP_WALL_REGULAR_X,
    JUMP_WALL_SLIDE_X
)
# Import existing physics functions
from ..physics import sweep_circle_vs_tiles

class NavigationNode:
    """Represents a position on a surface where movement decisions can be made"""
    def __init__(self, node_id: int, position: Tuple[float, float], 
                 surface: Surface, surface_offset: float):
        self.id = node_id
        self.position = position  # World coordinates
        self.surface = surface
        self.surface_offset = surface_offset  # Distance along surface from start
        self.node_type = self._determine_type()
        
    def _determine_type(self) -> str:
        """Determine if this is an edge, junction, or intermediate node"""
        # Check if surface length is valid before comparing
        if self.surface and self.surface.length > 0:
            if self.surface_offset == 0:
                return "edge_start"
            # Use a small epsilon for floating point comparison
            elif abs(self.surface_offset - self.surface.length) < 1e-5 :
                return "edge_end"
            else:
                return "intermediate"
        elif self.surface_offset == 0: # Fallback if surface length is 0 or surface is None
             return "edge_start" # Or perhaps "isolated" if surface.length is 0
        return "intermediate" # Default

class NavigationGraphBuilder:
    """Builds a navigation graph from parsed surfaces"""
    
    def __init__(self, surfaces: List[Surface], collision_checker: CollisionChecker):
        self.surfaces = surfaces
        self.collision_checker = collision_checker # Store if needed later
        self.graph = nx.DiGraph()  # Directed graph for one-way paths
        self.node_counter = 0
        self.nodes_by_position: Dict[Tuple[int,int], List[NavigationNode]] = {}  # Spatial index for fast lookup
        
    def build_graph(self) -> nx.DiGraph:
        """Construct the navigation graph"""
        self._create_surface_nodes()
        self._connect_surface_nodes()
        self._create_gap_crossing_edges()
        # Note: Jump/fall edges between surfaces are added later by PathfindingSystem
        # after the JumpCalculator is available and the initial graph structure is complete
        return self.graph
    
    def _create_surface_nodes(self):
        """Create nodes at important positions on each surface"""
        for surface in self.surfaces:
            if not surface.start_pos or not surface.end_pos:
                print("Warning: Surface has no start/end pos, skipping node creation for it.")
                continue

            # Always create nodes at surface edges
            start_node = self._create_node(surface.start_pos, surface, 0)
            # Ensure surface.length is non-zero before creating end_node with it as offset
            end_node_offset = surface.length if surface.length > 0 else 0
            end_node = self._create_node(surface.end_pos, surface, end_node_offset)
            
            # Add intermediate nodes for long surfaces
            if surface.length > 48:  # Two tile widths
                num_intermediate = int(surface.length / 48) # Ensure this is at least 1 if length > 48
                if num_intermediate > 0:
                    for i in range(1, num_intermediate + 1): # Iterate up to num_intermediate
                        # Corrected offset calculation
                        offset = (surface.length / (num_intermediate + 1)) * i 
                        pos = self._interpolate_position(surface, offset)
                        if pos:
                            self._create_node(pos, surface, offset)
    
    def _interpolate_position(self, surface: Surface, offset: float) -> Optional[Tuple[float,float]]:
        """Interpolate position along a surface given an offset from its start."""
        if not surface.start_pos or not surface.end_pos or surface.length == 0:
            return None
        
        # Direction vector of the surface
        direction_x = surface.end_pos[0] - surface.start_pos[0]
        direction_y = surface.end_pos[1] - surface.start_pos[1]
        
        # Normalized direction vector
        norm_direction_x = direction_x / surface.length
        norm_direction_y = direction_y / surface.length
        
        # Position is start_pos + offset * normalized_direction
        interp_x = surface.start_pos[0] + offset * norm_direction_x
        interp_y = surface.start_pos[1] + offset * norm_direction_y
        
        return (interp_x, interp_y)

    def _create_node(self, position: Tuple[float, float], 
                     surface: Surface, offset: float) -> NavigationNode:
        """Create and register a navigation node"""
        node = NavigationNode(self.node_counter, position, surface, offset)
        self.graph.add_node(node.id, 
                           position=position,
                           surface_id=id(surface), # Store surface ID for grouping
                           surface_type=surface.type,
                           node_type=node.node_type,
                           nav_node_object=node) # Store the node object itself for easier access
        
        # Add to spatial index (grid cells are 24x24 pixels)
        grid_pos = (int(position[0] / 24), int(position[1] / 24))
        if grid_pos not in self.nodes_by_position:
            self.nodes_by_position[grid_pos] = []
        self.nodes_by_position[grid_pos].append(node)
        
        self.node_counter += 1
        return node
    
    def _connect_surface_nodes(self):
        """Connect nodes on the same surface with walk/climb edges"""
        for surface in self.surfaces:
            # Get nodes belonging to this specific surface instance
            surface_nodes_ids = [n_id for n_id, data in self.graph.nodes(data=True) 
                               if data.get('surface_id') == id(surface)]
            
            if not surface_nodes_ids:
                continue

            # Retrieve the NavigationNode objects to sort by offset
            nodes_on_surface = [self.graph.nodes[n_id]['nav_node_object'] for n_id in surface_nodes_ids]
            
            # Sort nodes by their offset along the surface
            nodes_on_surface.sort(key=lambda node_obj: node_obj.surface_offset)
            
            # Connect adjacent nodes on the surface
            for i in range(len(nodes_on_surface) - 1):
                node_a = nodes_on_surface[i]
                node_b = nodes_on_surface[i+1]
                self._add_walk_edge(node_a.id, node_b.id, (node_b.surface_offset - node_a.surface_offset))

    def _add_walk_edge(self, node_id_a: int, node_id_b: int, weight: float):
        """Adds a walkable edge between two nodes on the same surface."""
        # Edges are typically bidirectional for walking, unless one-way platforms
        # Estimate time cost for walk edges
        avg_walk_speed = MAX_HOR_SPEED * 0.6 # Slower than max for planning
        time_cost_frames = int(weight / avg_walk_speed) if avg_walk_speed > 0 else int(weight) # Avoid div by zero, ensure int
        time_cost_frames = max(1, time_cost_frames) # Minimum 1 frame

        self.graph.add_edge(node_id_a, node_id_b, weight=weight, move_type='walk', frames=time_cost_frames)
        self.graph.add_edge(node_id_b, node_id_a, weight=weight, move_type='walk', frames=time_cost_frames) # Add reverse path

    def _create_gap_crossing_edges(self):
        """Bridge small horizontal gaps between co-linear floor or ceiling surfaces."""
        surface_types_to_bridge = [SurfaceType.FLOOR, SurfaceType.CEILING]
        MAX_GAP_SIZE = 8.1 

        for surface_type in surface_types_to_bridge:
            surfaces_of_type = [s for s in self.surfaces if s.type == surface_type]
            surfaces_of_type.sort(key=lambda s: (s.start_pos[1], s.start_pos[0]))

            for i in range(len(surfaces_of_type) - 1):
                s1 = surfaces_of_type[i]
                s2 = surfaces_of_type[i+1]

                if abs(s1.start_pos[1] - s2.start_pos[1]) < 1.0: 
                    gap = s2.start_pos[0] - s1.end_pos[0]
                    
                    if 0 < gap <= MAX_GAP_SIZE:
                        print(f"DEBUG: Potential gap: s1_end={s1.end_pos}, s2_start={s2.start_pos}, gap={gap:.2f}") 
                        s1_end_node_id = None
                        s2_start_node_id = None

                        for node_id, data in self.graph.nodes(data=True):
                            if data.get('surface_id') == id(s1) and \
                               abs(data['position'][0] - s1.end_pos[0]) < 1e-5 and \
                               abs(data['position'][1] - s1.end_pos[1]) < 1e-5:
                                s1_end_node_id = node_id
                                print(f"DEBUG: Found s1_end_node_id: {s1_end_node_id} for s1_end_pos {s1.end_pos}") 
                            
                            if data.get('surface_id') == id(s2) and \
                               abs(data['position'][0] - s2.start_pos[0]) < 1e-5 and \
                               abs(data['position'][1] - s2.start_pos[1]) < 1e-5:
                                s2_start_node_id = node_id
                                print(f"DEBUG: Found s2_start_node_id: {s2_start_node_id} for s2_start_pos {s2.start_pos}") 
                            
                            if s1_end_node_id and s2_start_node_id: 
                                break
                        
                        if s1_end_node_id is not None and s2_start_node_id is not None and s1_end_node_id != s2_start_node_id:
                            cost = gap 
                            # Estimate time for gap crossing (similar to walk)
                            avg_walk_speed = MAX_HOR_SPEED * 0.6
                            time_cost_frames_gap = int(cost / avg_walk_speed) if avg_walk_speed > 0 else int(cost)
                            time_cost_frames_gap = max(1, time_cost_frames_gap)

                            self.graph.add_edge(s1_end_node_id, s2_start_node_id, weight=cost, move_type='walk_gap', frames=time_cost_frames_gap)
                            self.graph.add_edge(s2_start_node_id, s1_end_node_id, weight=cost, move_type='walk_gap', frames=time_cost_frames_gap)
                            print(f"DEBUG: Added gap edge between node {s1_end_node_id} and node {s2_start_node_id} for gap {gap:.2f}") 
                        else:
                            print(f"DEBUG: Failed to find nodes or nodes are same for gap. s1_end_node={s1_end_node_id}, s2_start_node={s2_start_node_id}")


class JumpTrajectory:
    """Represents a calculated jump path between two nodes"""
    
    def __init__(self, start_node_id: int, end_node_id: int, 
                 initial_velocity: Tuple[float, float],
                 jump_type: str):
        self.start_node_id = start_node_id
        self.end_node_id = end_node_id
        self.initial_velocity = initial_velocity
        self.jump_type = jump_type  # "floor_jump", "wall_jump", etc.
        self.frames: List[Tuple[Tuple[float,float], Tuple[float,float]]] = []  # List of (position, velocity) tuples
        self.total_frames = 0
        self.max_height = 0 # Relative to start_pos y
        self.requires_held_jump = False
        
class JumpCalculator:
    """Calculates physically accurate jump trajectories using existing N++ physics"""
    
    def __init__(self, sim):
        """Initialize with simulator reference to use existing physics."""
        self.sim = sim
        self.pathfinding_utils = PathfindingUtils(sim)
        
    def calculate_jump(self, start_pos: Tuple[float, float], 
                      end_pos: Tuple[float, float],
                      start_surface_type: SurfaceType,
                      start_surface_normal: Optional[Tuple[float,float]] = None,
                      initial_run_velocities_x: Optional[List[float]] = None,
                      max_attempts: int = 10) -> Optional[JumpTrajectory]:
        """Calculate optimal jump trajectory between two positions"""
        
        trajectories: List[JumpTrajectory] = []
        attempts_made = 0
        
        if initial_run_velocities_x is None:
            # Default run velocities to test for floor jumps if not provided
            base_run_velocities = [0.0]
            if start_surface_type == SurfaceType.FLOOR or start_surface_type == SurfaceType.SLOPE:
                base_run_velocities.extend([MAX_HOR_SPEED, -MAX_HOR_SPEED, MAX_HOR_SPEED / 2, -MAX_HOR_SPEED / 2])
        else:
            base_run_velocities = initial_run_velocities_x

        # Try different hold durations
        hold_durations = [1, 5, 15, 30, MAX_JUMP_DURATION]

        for run_vx in base_run_velocities:
            # Skip non-zero run_vx for wall jumps as horizontal speed is dictated by the wall jump type
            if start_surface_type in [SurfaceType.WALL_LEFT, SurfaceType.WALL_RIGHT] and run_vx != 0.0:
                continue

            for hold in hold_durations:
                # Check if we've reached the maximum number of attempts
                if attempts_made >= max_attempts:
                    break
                    
                if start_surface_type == SurfaceType.FLOOR or start_surface_type == SurfaceType.SLOPE:
                    traj = self._try_jump_strategy(start_pos, end_pos, start_surface_type, 
                                                   hold_frames=hold, initial_vx_run=run_vx, 
                                                   surface_normal=start_surface_normal,
                                                   jump_type_prefix=f"run{run_vx:.1f}_hold{hold}")
                    attempts_made += 1
                    if traj: 
                        trajectories.append(traj)
                        # Early exit if we found a valid trajectory and want to limit attempts
                        if len(trajectories) >= 3:  # Stop after finding a few good options
                            break
                elif start_surface_type == SurfaceType.WALL_LEFT or start_surface_type == SurfaceType.WALL_RIGHT:
                    # Normal Wall Jump
                    if attempts_made < max_attempts:
                        traj_normal = self._try_jump_strategy(start_pos, end_pos, start_surface_type, 
                                                              hold_frames=hold, wall_jump_x_type="normal",
                                                              surface_normal=start_surface_normal,
                                                              jump_type_prefix=f"wall_norm_hold{hold}")
                        attempts_made += 1
                        if traj_normal: 
                            trajectories.append(traj_normal)
                    
                    # Slide Wall Jump
                    if attempts_made < max_attempts:
                        traj_slide = self._try_jump_strategy(start_pos, end_pos, start_surface_type, 
                                                             hold_frames=hold, wall_jump_x_type="slide",
                                                             surface_normal=start_surface_normal,
                                                             jump_type_prefix=f"wall_slide_hold{hold}")
                        attempts_made += 1
                        if traj_slide: 
                            trajectories.append(traj_slide)
            
            # Break outer loop if we've reached max attempts
            if attempts_made >= max_attempts:
                break
        
        if trajectories:
            return min(trajectories, key=lambda t: t.total_frames) # Prioritize faster jumps
        # print(f"Warning: JumpCalculator.calculate_jump from {start_pos} to {end_pos} found no valid trajectory.")
        return None

    def _try_jump_strategy(self, start_pos: Tuple[float, float],
                           target_pos: Tuple[float, float],
                           surface_type: SurfaceType,
                           hold_frames: int, 
                           jump_type_prefix: str,
                           initial_vx_run: float = 0.0, 
                           wall_jump_x_type: str = "normal",
                           surface_normal: Optional[Tuple[float,float]] = None
                           ) -> Optional[JumpTrajectory]:
        """Helper to attempt a jump with a specific strategy (e.g. hold duration, run speed, wall jump type)."""
        initial_vy = 0
        actual_initial_vx = initial_vx_run 
        jump_specific_vx_component = 0.0 
        jump_specific_vy_component = 0.0 # For slope jumps
        jump_type = ""

        if surface_type == SurfaceType.FLOOR:
            jump_specific_vy_component = JUMP_FLAT_GROUND_Y
            jump_type = f"{jump_type_prefix}_floor_jump"
        elif surface_type == SurfaceType.SLOPE:
            if surface_normal and (abs(surface_normal[0]) > 1e-6 or abs(surface_normal[1]) > 1e-6):
                # Jump perpendicular to slope normal
                jump_impulse_x = JUMP_FLAT_GROUND_Y * surface_normal[0]
                jump_impulse_y = JUMP_FLAT_GROUND_Y * surface_normal[1]
                jump_specific_vx_component = jump_impulse_x
                jump_specific_vy_component = jump_impulse_y
                jump_type = f"{jump_type_prefix}_slope_jump"
            else: # Fallback if normal is zero or not provided (treat as floor)
                jump_specific_vy_component = JUMP_FLAT_GROUND_Y
                jump_type = f"{jump_type_prefix}_slope_as_floor_jump"
        elif surface_type == SurfaceType.WALL_LEFT: 
            jump_specific_vy_component = JUMP_WALL_REGULAR_Y
            if wall_jump_x_type == "normal":
                jump_specific_vx_component = JUMP_WALL_REGULAR_X
                jump_type = f"{jump_type_prefix}_wall_jump_right_normal"
            elif wall_jump_x_type == "slide":
                jump_specific_vx_component = JUMP_WALL_SLIDE_X
                jump_type = f"{jump_type_prefix}_wall_jump_right_slide"
            else:
                return None 
        elif surface_type == SurfaceType.WALL_RIGHT: 
            jump_specific_vy_component = JUMP_WALL_REGULAR_Y
            if wall_jump_x_type == "normal":
                jump_specific_vx_component = -JUMP_WALL_REGULAR_X 
                jump_type = f"{jump_type_prefix}_wall_jump_left_normal"
            elif wall_jump_x_type == "slide":
                jump_specific_vx_component = -JUMP_WALL_SLIDE_X  
                jump_type = f"{jump_type_prefix}_wall_jump_left_slide"
            else:
                return None 
        else:
            return None

        actual_initial_vx += jump_specific_vx_component
        initial_vy = jump_specific_vy_component # This is the primary vertical impulse from jump
        
        # initial_vy should also consider any existing y-velocity from run_on_slope, but N++ resets y-vel on jump.
        # So, current_vel_sim[1] is overridden by jump_specific_vy_component.

        initial_vel = (actual_initial_vx, initial_vy) 
        
        simulated_traj = self._simulate_jump_with_physics(start_pos, initial_vel, target_pos, hold_frames, jump_type)
        if simulated_traj:
            # Assign IDs later when integrating with graph nodes
            simulated_traj.start_node_id = -1 # Placeholder
            simulated_traj.end_node_id = -1   # Placeholder
        return simulated_traj

    def _simulate_jump_with_physics(self, start_pos: Tuple[float, float],
                                   initial_velocity: Tuple[float, float],
                                   target_pos: Tuple[float, float],
                                   hold_frames: int,
                                   jump_type_str: str) -> Optional[JumpTrajectory]:
        """Simulate a jump using the existing physics system for collision detection."""
        
        # Node IDs are not known here, set to placeholder
        trajectory = JumpTrajectory(-1, -1, initial_velocity, jump_type_str)
        pos = list(start_pos)
        vel = list(initial_velocity)
        max_y_achieved = start_pos[1]
        
        # Track ninja state for accurate physics simulation
        jump_duration = 0
        
        for frame_num in range(1, 201):  # Max simulation frames (~3.3 seconds at 60fps)
            # Store old position for collision checking
            old_pos = pos.copy()
            
            # Apply drag first (like in ninja.integrate())
            vel[0] *= DRAG_REGULAR
            vel[1] *= DRAG_REGULAR
            
            # Apply gravity based on jump state (like in ninja.integrate())
            if frame_num <= hold_frames and vel[1] < 0 and jump_duration < MAX_JUMP_DURATION:
                vel[1] += GRAVITY_JUMP
                jump_duration += 1
            else:
                vel[1] += GRAVITY_FALL
            
            # Apply horizontal air control (more accurate to N++ air control)
            target_dx_component = target_pos[0] - pos[0]
            if abs(target_dx_component) > 1.0:  # Only apply control if significant distance
                if target_dx_component > 0:
                    # Moving right
                    if vel[0] < MAX_HOR_SPEED:
                        vel[0] += AIR_ACCEL
                        vel[0] = min(vel[0], MAX_HOR_SPEED)
                else:
                    # Moving left
                    if vel[0] > -MAX_HOR_SPEED:
                        vel[0] -= AIR_ACCEL
                        vel[0] = max(vel[0], -MAX_HOR_SPEED)
            
            # Update position (like in ninja.integrate())
            pos[0] += vel[0]
            pos[1] += vel[1]
            
            # Track maximum height achieved
            if pos[1] < max_y_achieved:
                max_y_achieved = pos[1]
            
            # Use existing physics system for collision detection
            dx = pos[0] - old_pos[0]
            dy = pos[1] - old_pos[1]
            collision_time = sweep_circle_vs_tiles(self.sim, old_pos[0], old_pos[1], dx, dy, NINJA_RADIUS)
            
            if collision_time < 1.0:
                return None  # Collision detected
            
            # Store frame data
            trajectory.frames.append(((pos[0], pos[1]), (vel[0], vel[1])))
            
            # Check if we've reached the target area
            dist_to_target = math.sqrt((pos[0] - target_pos[0])**2 + (pos[1] - target_pos[1])**2)
            if dist_to_target < NINJA_RADIUS * 2:  # Within reasonable range of target
                trajectory.total_frames = frame_num
                trajectory.max_height = start_pos[1] - max_y_achieved
                trajectory.requires_held_jump = (hold_frames > 5)
                return trajectory
            
            # Check if ninja has fallen too far below target (failed jump)
            if pos[1] > target_pos[1] + 100:  # 100 pixels below target
                return None
            
            # Check for lethal impact velocity if we hit ground
            if vel[1] > MAX_SURVIVABLE_IMPACT:
                return None  # Would die from impact
            
            # Check if we've gone too far horizontally or fallen too far
            if abs(pos[0] - start_pos[0]) > 700:  # Traveled too far horizontally
                return None
        
        # Simulation ended without reaching target
        return None
