import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional

from .utils import CollisionChecker # Assuming CollisionChecker is in utils.py

class PathOptimizer:
    """Optimizes and smooths paths for better execution"""
    
    def __init__(self, collision_checker: CollisionChecker):
        self.collision_checker = collision_checker
        
    def smooth_path(self, node_path_ids: List[int], 
                   nav_graph: nx.DiGraph) -> List[Tuple[float, float]]:
        """Convert node path (IDs) to a smoothed list of world coordinate waypoints."""
        if not node_path_ids or not nav_graph:
            return []

        # Get world positions from node IDs
        positions: List[Tuple[float, float]] = [] 
        for node_id in node_path_ids:
            if node_id in nav_graph.nodes:
                positions.append(nav_graph.nodes[node_id]['position'])
            else:
                print(f"Warning: Node ID {node_id} not found in nav_graph during path smoothing.")
                # Potentially skip or handle error
        
        if not positions:
            return []

        # Apply line-of-sight optimization (string pulling)
        los_optimized_path = self._line_of_sight_smooth(positions)
        
        # Apply curve smoothing (e.g., Bezier curves or Catmull-Rom splines)
        # For N++, sharp turns are common, so excessive smoothing might be bad.
        # A simple Bezier might be too much. Catmull-Rom could be better as it passes through control points.
        # For now, the prompt mentions Bezier, so we'll stub that.
        final_smoothed_path = self._bezier_smooth(los_optimized_path)
        
        return final_smoothed_path
    
    def _line_of_sight_smooth(self, positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Remove unnecessary waypoints using line-of-sight checks (String Pulling algorithm)."""
        if len(positions) <= 2:
            return positions
        
        smoothed_path = [positions[0]] # Start with the first point
        current_anchor_idx = 0
        
        while current_anchor_idx < len(positions) - 1:
            # Look ahead for the furthest point visible from current_anchor_idx
            furthest_visible_idx = current_anchor_idx + 1
            
            # Iterate from one after the next point up to the end of the path
            for test_idx in range(current_anchor_idx + 2, len(positions)):
                if self._has_line_of_sight(positions[current_anchor_idx], positions[test_idx]):
                    furthest_visible_idx = test_idx # This point is visible, try further
                else:
                    break # Obstruction found, previous point was the furthest visible
            
            # Add the furthest visible point and make it the new anchor
            smoothed_path.append(positions[furthest_visible_idx])
            current_anchor_idx = furthest_visible_idx
            
            # If the anchor became the last point, we are done
            if current_anchor_idx == len(positions) - 1 and positions[furthest_visible_idx] not in smoothed_path:
                 # This case should ideally not happen if logic is correct, but as a safeguard:
                 if smoothed_path[-1] != positions[-1]: smoothed_path.append(positions[-1])

        return smoothed_path
    
    def _has_line_of_sight(self, start_pos: Tuple[float, float], 
                          end_pos: Tuple[float, float]) -> bool:
        """Check if there's a clear path (no collisions) between two world positions."""
        # Uses the CollisionChecker. For a line-of-sight, we need to check points along the line.
        # Bresenham's line algorithm or sample points along the segment.
        
        num_steps = 20 # Number of intermediate points to check (adjust based on typical distances)
        if start_pos == end_pos: return True

        for i in range(num_steps + 1):
            t = i / float(num_steps)
            # Interpolate point
            check_x = start_pos[0] + t * (end_pos[0] - start_pos[0])
            check_y = start_pos[1] + t * (end_pos[1] - start_pos[1])
            
            # Check for collision at this intermediate point
            # The prompt's CollisionChecker has point_in_wall, which is suitable here.
            if self.collision_checker.point_in_wall((check_x, check_y)):
                # print(f"LoS obstructed between {start_pos} and {end_pos} at ({check_x}, {check_y})")
                return False # Collision detected
        
        return True # No collision found along the line

    def _bezier_smooth(self, positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Smooth the path using Bezier curves (stub)."""
        # This is a complex topic. A simple implementation might not be suitable for N++.
        # For N++, path often follows surfaces closely. Over-smoothing can be detrimental.
        # Catmull-Rom splines might be better as they pass through control points.
        # For now, returning the path as is, as Bezier smoothing needs careful implementation.
        print("Warning: _bezier_smooth is a stub and returns the path unsmoothed.")
        if len(positions) < 3: # Not enough points for meaningful Bezier on segments
            return positions
        
        # A proper Bezier implementation would generate new points along curves
        # between segments of the LoS path. This is non-trivial.
        # For a stub, we just return the LoS path.
        return positions

class MovementController:
    """Converts high-level paths to frame-by-frame N++ input commands using accurate physics."""
    
    # Exact physics constants from the actual N++ simulation
    GRAVITY_FALL = 0.06666666666666665
    GRAVITY_JUMP = 0.01111111111111111
    GROUND_ACCEL = 0.06666666666666665
    AIR_ACCEL = 0.04444444444444444
    DRAG_REGULAR = 0.9933221725495059
    FRICTION_GROUND = 0.9459290248857720
    FRICTION_WALL = 0.9113380468927672
    MAX_HOR_SPEED = 3.333333333333333
    MAX_JUMP_DURATION = 45
    NINJA_RADIUS = 10
    
    # Jump velocities from actual ninja mechanics
    FLOOR_JUMP_VELOCITY_Y = -2.0
    WALL_JUMP_VELOCITY_Y = -1.4
    WALL_JUMP_VELOCITY_X_NORMAL = 1.0
    WALL_JUMP_VELOCITY_X_SLIDE = 2.0/3.0
    
    def __init__(self, physics_params: Optional[dict] = None):
        self.physics = physics_params if physics_params else {}
        self.command_buffer: List[dict] = []
        
        # Allow override of physics constants if needed
        self.max_jump_hold_frames = self.physics.get('max_jump_hold_frames', self.MAX_JUMP_DURATION)
        self.ninja_max_speed = self.physics.get('ninja_speed', self.MAX_HOR_SPEED)
        self.jump_vel_y = self.physics.get('jump_vel_y', self.FLOOR_JUMP_VELOCITY_Y)
        self.wall_jump_vel_y = self.physics.get('wall_jump_vel_y', self.WALL_JUMP_VELOCITY_Y)
        self.wall_jump_vel_x = self.physics.get('wall_jump_vel_x', self.WALL_JUMP_VELOCITY_X_NORMAL)

    def generate_commands(self, current_pos: Tuple[float, float],
                         current_vel: Tuple[float, float],
                         waypoint_path: List[Tuple[float, float]], # Smoothed world coordinate path
                         edge_types: List[str] # Type of movement for each segment (walk, jump, etc.)
                         ) -> List[dict]:
        """Generate frame-by-frame movement commands to follow the waypoint_path."""
        
        self.command_buffer = []
        if not waypoint_path or len(waypoint_path) < 2:
            return []

        # Simulate frame by frame, trying to reach next waypoint
        # This is a simplified controller. A real one would be a PID or state machine.
        # For N++, precise inputs are key.

        # The prompt's example for _generate_jump_commands is very basic.
        # N++ movement is highly nuanced. This will be a high-level approximation.

        total_frames_simulated = 0

        for i in range(len(waypoint_path) - 1):
            segment_start_wp = waypoint_path[i]
            segment_end_wp = waypoint_path[i+1]
            # Edge type corresponds to the segment from waypoint_path[i] to waypoint_path[i+1]
            edge_type = edge_types[i] if i < len(edge_types) else 'walk' 
            
            # print(f"Segment {i}: {segment_start_wp} -> {segment_end_wp}, type: {edge_type}")

            # This is where a low-level controller would take over for this segment.
            # The provided structure implies generating all commands for the whole path at once.
            # This is more like a plan than reactive control.
            segment_commands: List[dict] = []
            if edge_type == 'walk' or edge_type == 'run':
                segment_commands = self._generate_walk_run_commands(segment_start_wp, segment_end_wp, current_vel, total_frames_simulated)
            elif edge_type == 'jump':
                # Need to know if it's a floor jump or part of an air trajectory
                # The JumpCalculator would have set jump_type on the trajectory object.
                # Here, we only have 'jump' as edge_type.
                # This needs more info (e.g. from surface or trajectory data associated with edge)
                segment_commands = self._generate_abstract_jump_commands(segment_start_wp, segment_end_wp, current_vel, total_frames_simulated)
            elif edge_type == 'wall_jump':
                # Needs to know which wall (left/right) to determine jump direction
                segment_commands = self._generate_abstract_wall_jump_commands(segment_start_wp, segment_end_wp, current_vel, total_frames_simulated)
            elif edge_type == 'fall':
                segment_commands = self._generate_fall_commands(segment_start_wp, segment_end_wp, current_vel, total_frames_simulated)
            else:
                print(f"Warning: Unknown edge type '{edge_type}' for command generation. Defaulting to walk.")
                segment_commands = self._generate_walk_run_commands(segment_start_wp, segment_end_wp, current_vel, total_frames_simulated)
            
            if segment_commands:
                self.command_buffer.extend(segment_commands)
                total_frames_simulated += len(segment_commands) # Rough estimate, commands are per frame
                # Update current_pos, current_vel based on simulated segment (idealized)
                # This is tricky without a full physics sim here.
                # Assume we magically reach segment_end_wp with some velocity.
                # For now, this controller is more of a command planner.
                current_pos = segment_end_wp 
                # current_vel would be complex to estimate. Zero it out or use last frame of jump sim.
                # This part is a major simplification.

        return self.command_buffer

    def _generate_walk_run_commands(self, start_wp: Tuple[float, float], end_wp: Tuple[float, float],
                                   current_vel: Tuple[float, float], base_frame: int) -> List[dict]:
        print(f"Warning: _generate_walk_run_commands from {start_wp} to {end_wp} is a stub.")
        # Determine direction and number of frames based on distance and speed.
        dx = end_wp[0] - start_wp[0]
        # dy should be small for walk/run on a surface
        num_frames = int(abs(dx) / self.ninja_max_speed) if self.ninja_max_speed > 0 else 10
        num_frames = max(1, num_frames) # At least one frame
        cmds = []
        for i in range(num_frames):
            cmds.append({
                'frame': base_frame + i,
                'jump': False, 'left': dx < 0, 'right': dx > 0,
                'action': 'walk/run'
            })
        return cmds

    def _generate_abstract_jump_commands(self, start_wp: Tuple[float, float], end_wp: Tuple[float, float],
                                 current_vel: Tuple[float, float], base_frame: int) -> List[dict]:
        """Generate commands for a generic jump segment based on prompt's example."""
        print(f"Warning: _generate_abstract_jump_commands from {start_wp} to {end_wp} is a stub (using prompt's logic)." )
        commands = []
        dx = end_wp[0] - start_wp[0]
        dy = end_wp[1] - start_wp[1] # Negative dy means jumping up

        # Determine jump hold duration based on dy (as per prompt example)
        if dy < -50:  # High jump target
            hold_frames = self.MAX_JUMP_HOLD_FRAMES
        elif dy < -30: # Medium jump target
            hold_frames = self.MAX_JUMP_HOLD_FRAMES // 2
        else:  # Low jump target
            hold_frames = max(1, self.MAX_JUMP_HOLD_FRAMES // 4)
        
        # Initial jump command (assumes player is on ground and can jump)
        commands.append({
            'frame': base_frame,
            'jump': True, 'left': False, 'right': False, # Initial jump is usually straight up or with run momentum
            'action': 'jump_initiate'
        })
        
        # Hold jump button and apply air control
        # This simplified version just holds jump and sets horizontal direction.
        # Real N++ jump trajectory is complex.
        simulated_frames_for_jump = hold_frames + 20 # Arbitrary extra frames for air time

        for frame_offset in range(1, simulated_frames_for_jump):
            cmd = {'frame': base_frame + frame_offset, 'jump': False, 'left': False, 'right': False}
            if frame_offset < hold_frames:
                cmd['jump'] = True # Hold jump key
            
            # Basic air control towards end_wp's x coordinate
            # This is a gross simplification. True air control depends on physics.
            if end_wp[0] < start_wp[0] - 5: # Target is to the left
                cmd['left'] = True
            elif end_wp[0] > start_wp[0] + 5: # Target is to the right
                cmd['right'] = True
            
            cmd['action'] = 'jump_air_control'
            commands.append(cmd)
        
        return commands

    def _generate_abstract_wall_jump_commands(self, start_wp: Tuple[float, float], end_wp: Tuple[float, float],
                                     current_vel: Tuple[float, float], base_frame: int) -> List[dict]:
        print(f"Warning: _generate_abstract_wall_jump_commands from {start_wp} to {end_wp} is a stub.")
        # Needs to know which wall (left/right) to determine jump direction (away from wall)
        # Assume start_wp is on a wall. If end_wp is to the right of start_wp, assume left wall jump.
        dx = end_wp[0] - start_wp[0]
        dy = end_wp[1] - start_wp[1]
        cmds = []

        # Wall jump action (one frame typically)
        # Direction of jump is away from wall and slightly up.
        # Horizontal input is usually *away* from the wall initially.
        is_left_wall_jump = dx > 0 # Jumping off left wall, moving right

        cmds.append({
            'frame': base_frame,
            'jump': True, 
            'left': not is_left_wall_jump, # Press into wall or away? N++ is nuanced.
            'right': is_left_wall_jump,    # Usually press away from wall for wall jump.
            'action': 'wall_jump_initiate'
        })

        # Air control after wall jump (similar to regular jump air control)
        # Simplified: hold jump for a bit (wall jumps are often shorter duration holds)
        wall_jump_hold = max(1, self.MAX_JUMP_HOLD_FRAMES // 3)
        simulated_frames_for_wall_jump = wall_jump_hold + 20

        for frame_offset in range(1, simulated_frames_for_wall_jump):
            cmd = {'frame': base_frame + frame_offset, 'jump': False, 'left': False, 'right': False}
            if frame_offset < wall_jump_hold:
                cmd['jump'] = True
            
            # Air control towards end_wp
            if end_wp[0] < start_wp[0] + dx * (frame_offset/simulated_frames_for_wall_jump) - 5 : # Simplified target tracking
                cmd['left'] = True
            elif end_wp[0] > start_wp[0] + dx * (frame_offset/simulated_frames_for_wall_jump) + 5:
                cmd['right'] = True
            cmd['action'] = 'wall_jump_air_control'
            cmds.append(cmd)
        return cmds

    def _generate_fall_commands(self, start_wp: Tuple[float, float], end_wp: Tuple[float, float],
                                current_vel: Tuple[float, float], base_frame: int) -> List[dict]:
        print(f"Warning: _generate_fall_commands from {start_wp} to {end_wp} is a stub.")
        # Control horizontal movement during fall to reach end_wp.x
        # Number of frames depends on vertical distance and gravity.
        # This is a simplification.
        dx = end_wp[0] - start_wp[0]
        dy = end_wp[1] - start_wp[1] # Positive dy means falling down
        # Estimate fall time (very rough, ignores drag, initial vel)
        # d = 0.5 * g * t^2 => t = sqrt(2d/g). Using GRAVITY_FALL from JumpCalculator (0.0667 pixels/frame^2)
        # This physics is simplified. N++ gravity is more complex.
        num_frames = int(np.sqrt(2 * dy / 0.0667)) if dy > 0 else 10 
        num_frames = max(1, num_frames)
        cmds = []
        for i in range(num_frames):
            cmds.append({
                'frame': base_frame + i,
                'jump': False, 
                'left': dx < 0, # Basic air control
                'right': dx > 0,
                'action': 'fall_control'
            })
        return cmds
