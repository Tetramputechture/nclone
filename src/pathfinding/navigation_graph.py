import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional

from .surface_parser import Surface, SurfaceType
from .utils import CollisionChecker # Assuming CollisionChecker is in utils.py

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
    
    def __init__(self, surfaces: List[Surface]):
        self.surfaces = surfaces
        self.graph = nx.DiGraph()  # Directed graph for one-way paths
        self.node_counter = 0
        self.nodes_by_position: Dict[Tuple[int,int], List[NavigationNode]] = {}  # Spatial index for fast lookup
        
    def build_graph(self) -> nx.DiGraph:
        """Construct the navigation graph"""
        # Step 1: Create nodes at key positions
        self._create_surface_nodes()
        
        # Step 2: Connect nodes on same surface
        self._connect_surface_nodes()
        
        # Step 3: Add jump/fall connections between surfaces (Placeholder)
        self._create_inter_surface_edges()
        
        return self.graph
    
    def _create_surface_nodes(self):
        """Create nodes at important positions on each surface"""
        for surface in self.surfaces:
            if not surface.start_pos or not surface.end_pos:
                print(f"Warning: Surface has no start/end pos, skipping node creation for it.")
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
        self.graph.add_edge(node_id_a, node_id_b, weight=weight, move_type='walk')
        self.graph.add_edge(node_id_b, node_id_a, weight=weight, move_type='walk') # Add reverse path
        # print(f"Added walk edge between {node_id_a} and {node_id_b} with weight {weight}")

    def _create_inter_surface_edges(self):
        """Add jump/fall connections between surfaces. Placeholder."""
        # This is where JumpCalculator would be used to find valid jump/fall paths
        # between nodes on different surfaces.
        print("Warning: _create_inter_surface_edges is a stub and does not create jump/fall edges.")
        # Example (conceptual):
        # for node1_id in self.graph.nodes():
        #     for node2_id in self.graph.nodes():
        #         if node1_id == node2_id: continue
        #         node1 = self.graph.nodes[node1_id]['nav_node_object']
        #         node2 = self.graph.nodes[node2_id]['nav_node_object']
        #         if node1.surface == node2.surface: continue # Already handled by _connect_surface_nodes
        #         
        #         # Check for potential jump/fall
        #         # trajectory = jump_calculator.calculate_jump(node1.position, node2.position, node1.surface.type)
        #         # if trajectory:
        #         #     self.graph.add_edge(node1.id, node2.id, weight=trajectory.total_frames, move_type='jump', trajectory=trajectory)
        pass

class JumpTrajectory:
    """Represents a calculated jump path between two nodes"""
    
    def __init__(self, start_node_id: int, end_node_id: int, 
                 initial_velocity: Tuple[float, float],
                 jump_type: str):
        self.start_node_id = start_node_id # Changed from start_node to id
        self.end_node_id = end_node_id   # Changed from end_node to id
        self.initial_velocity = initial_velocity
        self.jump_type = jump_type  # "floor_jump", "wall_jump", etc.
        self.frames: List[Tuple[Tuple[float,float], Tuple[float,float]]] = []  # List of (position, velocity) tuples
        self.total_frames = 0
        self.max_height = 0 # Relative to start_pos y
        self.requires_held_jump = False
        
class JumpCalculator:
    """Calculates physically accurate jump trajectories"""
    
    # Physics constants from N++ simulation
    GRAVITY_JUMP = 0.0111
    GRAVITY_FALL = 0.0667
    MAX_JUMP_FRAMES_HOLD = 45 # Max frames jump button can be held for effect
    FLOOR_JUMP_VELOCITY_Y = -2.0 # Initial Y velocity for floor jump
    WALL_JUMP_VELOCITY_Y = -1.4  # Initial Y velocity for wall jump
    # Wall jump also has an X component away from the wall
    WALL_JUMP_VELOCITY_X = 1.5 # Example, depends on wall direction
    AIR_ACCELERATION = 0.0444
    MAX_HORIZONTAL_SPEED = 3.333
    REGULAR_DRAG = 0.9933
    
    def __init__(self, collision_checker: CollisionChecker):
        self.collision_checker = collision_checker
        
    def calculate_jump(self, start_pos: Tuple[float, float], 
                      end_pos: Tuple[float, float],
                      start_surface_type: SurfaceType,
                      max_attempts: int = 10) -> Optional[JumpTrajectory]:
        """Calculate optimal jump trajectory between two positions"""
        
        trajectories: List[JumpTrajectory] = []
        
        # Strategy 1: Minimum height jump (short hold)
        traj = self._try_jump_strategy(start_pos, end_pos, start_surface_type, hold_frames=5, jump_type_prefix="min_h")
        if traj: trajectories.append(traj)
        
        # Strategy 2: Maximum height jump (long hold)
        traj = self._try_jump_strategy(start_pos, end_pos, start_surface_type, hold_frames=self.MAX_JUMP_FRAMES_HOLD, jump_type_prefix="max_h")
        if traj: trajectories.append(traj)
        
        # Strategy 3: Variable height jumps (medium holds)
        for hold_frames in [15, 30]:
            traj = self._try_jump_strategy(start_pos, end_pos, start_surface_type, hold_frames=hold_frames, jump_type_prefix="var_h")
            if traj: trajectories.append(traj)
        
        if trajectories:
            return min(trajectories, key=lambda t: t.total_frames)
        print(f"Warning: JumpCalculator.calculate_jump from {start_pos} to {end_pos} found no valid trajectory.")
        return None

    def _try_jump_strategy(self, start_pos: Tuple[float, float],
                           target_pos: Tuple[float, float],
                           surface_type: SurfaceType,
                           hold_frames: int, 
                           jump_type_prefix: str) -> Optional[JumpTrajectory]:
        """Helper to attempt a jump with a specific strategy (e.g. hold duration)."""
        initial_vy = 0
        initial_vx_abs = 0 # Absolute horizontal speed component from jump itself
        jump_type = ""

        if surface_type == SurfaceType.FLOOR or surface_type == SurfaceType.SLOPE:
            initial_vy = self.FLOOR_JUMP_VELOCITY_Y
            jump_type = f"{jump_type_prefix}_floor_jump"
            # Horizontal velocity is mostly from running, but jump can have small x influence
        elif surface_type == SurfaceType.WALL_LEFT: # Jumping off a left wall (to the right)
            initial_vy = self.WALL_JUMP_VELOCITY_Y
            initial_vx_abs = self.WALL_JUMP_VELOCITY_X 
            jump_type = f"{jump_type_prefix}_wall_jump_right"
        elif surface_type == SurfaceType.WALL_RIGHT: # Jumping off a right wall (to the left)
            initial_vy = self.WALL_JUMP_VELOCITY_Y
            initial_vx_abs = -self.WALL_JUMP_VELOCITY_X
            jump_type = f"{jump_type_prefix}_wall_jump_left"
        else:
            # print(f"Cannot initiate jump from surface type: {surface_type}")
            return None

        # Simulate with and without initial horizontal boost from jump itself
        # This needs to be combined with player's running speed usually.
        # For pre-computation, we might test a few initial running speeds or assume a neutral one.
        
        # Simplified: Assume jump provides some base horizontal velocity, player can add to it via air control.
        # The _simulate_jump will handle air control towards target_pos[0]
        initial_vel = (initial_vx_abs, initial_vy) 
        
        simulated_traj = self._simulate_jump(start_pos, initial_vel, target_pos, hold_frames, jump_type)
        if simulated_traj:
            # Assign IDs later when integrating with graph nodes
            simulated_traj.start_node_id = -1 # Placeholder
            simulated_traj.end_node_id = -1   # Placeholder
        return simulated_traj

    # The _try_minimum_jump, _try_maximum_jump, _try_variable_jump methods from the prompt
    # are consolidated into _try_jump_strategy and _simulate_jump.

    def _simulate_jump(self, start_pos: Tuple[float, float],
                      initial_velocity: Tuple[float, float],
                      target_pos: Tuple[float, float],
                      hold_frames: int,
                      jump_type_str: str) -> Optional[JumpTrajectory]:
        """Simulate a jump with given parameters"""
        
        # Node IDs are not known here, set to placeholder
        trajectory = JumpTrajectory(-1, -1, initial_velocity, jump_type_str)
        pos = list(start_pos)
        vel = list(initial_velocity)
        max_y_achieved = start_pos[1]
        
        for frame_num in range(1, 201):  # Max simulation frames (e.g., ~3-4 seconds at 60fps)
            # Apply drag (applied to current velocity before other forces)
            vel[0] *= self.REGULAR_DRAG
            vel[1] *= self.REGULAR_DRAG # N++ drag is more complex, this is simplified
            
            # Apply gravity
            # Gravity is different if jump button is held AND ninja is moving up
            if frame_num <= hold_frames and vel[1] < 0:
                vel[1] += self.GRAVITY_JUMP
            else:
                vel[1] += self.GRAVITY_FALL
            
            # Apply horizontal air control (simplified)
            # Ninja tries to move towards the target_pos horizontally
            target_dx_component = target_pos[0] - pos[0]
            # Only apply air control if there's a significant horizontal distance to cover
            # or if current horizontal speed is not already maxed out in the desired direction.
            if abs(target_dx_component) > 1.0: # Threshold to apply control
                control_accel = self.AIR_ACCELERATION * np.sign(target_dx_component)
                vel[0] += control_accel
                # Clamp horizontal velocity to max speed
                vel[0] = np.clip(vel[0], -self.MAX_HORIZONTAL_SPEED, self.MAX_HORIZONTAL_SPEED)
            
            # Update position: new_pos = old_pos + velocity_after_forces * time_step (time_step=1 frame)
            new_pos = [pos[0] + vel[0], pos[1] + vel[1]]
            
            # Collision check for the segment from pos to new_pos
            if self.collision_checker.check_collision(tuple(pos), tuple(new_pos)):
                # print(f"Collision during jump simulation from {pos} to {new_pos}")
                return None # Collision detected
            
            pos = new_pos
            trajectory.frames.append((tuple(pos), tuple(vel)))
            if pos[1] < max_y_achieved: # Y is typically negative for up in game coords
                max_y_achieved = pos[1]
            
            # Check if we reached target (within a certain radius)
            # Target radius should be related to ninja size, e.g., half-width
            dist_to_target = np.sqrt((pos[0] - target_pos[0])**2 + (pos[1] - target_pos[1])**2)
            if dist_to_target < 12:  # Ninja half-width is ~12 pixels (tile is 24x24)
                trajectory.total_frames = frame_num
                trajectory.max_height = start_pos[1] - max_y_achieved # Max height relative to start
                trajectory.requires_held_jump = hold_frames > 5 # Arbitrary threshold for 'held'
                return trajectory
            
            # Check if we've gone too far, too low, or too high without hitting target
            if pos[1] > start_pos[1] + 200: # Fallen too far below start
                return None
            if abs(pos[0] - start_pos[0]) > 700: # Traveled too far horizontally
                 return None
        
        # print(f"Jump simulation timed out for target {target_pos}")
        return None # Simulation ended without reaching target
