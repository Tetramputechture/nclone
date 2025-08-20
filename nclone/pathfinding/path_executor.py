import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional

from .utils import CollisionChecker # Assuming CollisionChecker is in utils.py
# Import constants from ninja.py
from ..ninja import (
    GRAVITY_FALL,
    GRAVITY_JUMP,
    GROUND_ACCEL,
    AIR_ACCEL,
    DRAG_REGULAR,
    FRICTION_GROUND,
    MAX_HOR_SPEED,
    MAX_JUMP_DURATION,
    NINJA_RADIUS,
    JUMP_FLAT_GROUND_Y,
    JUMP_WALL_REGULAR_Y,
    JUMP_WALL_REGULAR_X
)

class PathOptimizer:
    """Optimizes and smooths paths for better execution"""
    
    def __init__(self, collision_checker: CollisionChecker):
        self.collision_checker = collision_checker
        # Smoothing parameters
        self.bezier_segment_points = 10 # Number of points to generate for each Bezier segment
        
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

    def _evaluate_cubic_bezier(self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
        """Evaluate a point on a cubic Bezier curve defined by p0, p1, p2, p3 at parameter t."""
        if not (0 <= t <= 1):
            raise ValueError("t must be between 0 and 1")
        return (1-t)**3 * p0 + 3 * (1-t)**2 * t * p1 + 3 * (1-t) * t**2 * p2 + t**3 * p3

    def _calculate_bezier_control_points(self, points: List[Tuple[float, float]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Calculates control points for a sequence of cubic Bezier curves
        interpolating the given points, ensuring C2 continuity.
        Based on the method described by Omar Aflak: https://omaraflak.medium.com/b%C3%A9zier-interpolation-8033e9a262c2
        Args:
            points: A list of (x, y) tuples representing the points to interpolate.
        Returns:
            A tuple containing two lists: (first_control_points, second_control_points)
            where first_control_points[i] (a_i) and second_control_points[i] (b_i) are for the curve
            segment between points[i] and points[i+1].
        """
        n = len(points) - 1
        if n < 1:
            return [], []

        p = [np.array(pt) for pt in points]

        # Create the tridiagonal matrix for solving for 'a' control points
        A = np.zeros((n, n))
        # RHS vector
        B = np.zeros((n, 2)) # For (x,y) coordinates

        # Boundary conditions for a_0 (relates to P_0)
        A[0, 0] = 2
        if n > 1:
            A[0, 1] = 1
        B[0] = p[0] + 2 * p[1]

        # Equations for internal a_i
        for i in range(1, n - 1):
            A[i, i-1] = 1
            A[i, i] = 4
            A[i, i+1] = 1
            B[i] = 4 * p[i] + 2 * p[i+1]

        # Boundary conditions for a_{n-1} (relates to P_n)
        if n > 1: # Only if there's more than one segment
            A[n-1, n-2] = 2
            A[n-1, n-1] = 7 # Corrected based on typical formulation (or requires specific derivation)
                           # The article implies a slightly different system setup when solving for a_i.
                           # Let's stick to the article's equations for now if they are self-contained for 'a'.
                           # Re-checking the article's system for 'a':
                           # P_0 = 2a_0 - b_0
                           # P_n = 2b_{n-1} - a_{n-1}
                           # a_i + b_i = 2 * P_{i+1}
                           # a_{i+1} + b_i = 2 * P_{i+1} (smoothness of 1st derivative) -> a_i = a_{i+1} WRONG
                           # From article: b_i = 2*P_{i+1} - a_{i+1} (for i=0 to n-2)
                           # and b_{n-1} = (P_n + a_{n-1}) / 2
                           # Substitute b_i into: a_i + b_{i-1} = 2 * P_i (for i=1 to n-1)
                           # Leads to a system in terms of 'a's.
            
            # Let's use the direct system matrix from the article's Python code:
            # For a0:  A[0,0]=2; A[0,1]=1; B[0]=P[0]+2*P[1]
            # For a_i: A[i,i-1]=1; A[i,i]=4; A[i,i+1]=1; B[i]=4*P[i]+2*P[i+1]
            # For a_{n-1}: A[n-1,n-2]=2; A[n-1,n-1]=7; B[n-1]=8*P[n-1]+P[n]
            # This seems to be derived to solve for 'a' control points directly.
            A[n-1,n-1] = 7 # As per a common formulation for this specific system construction.
            if n > 1: A[n-1, n-2] = 2 # Check indexing carefully.
            
            B[n-1] = 8 * p[n-1] + p[n]
        elif n == 1: # Single segment, P0 to P1
            # a0 = (P0 + 2P1)/3 , b0 = (2P0 + P1)/3 could be a simple choice for one segment,
            # but the article's method is for chains. Let's ensure the single segment case is handled.
            # For n=1, the system is 1x1: A[0,0]a_0 = B_0.
            # The article's Python implies:
            # if n==1, a_0 = (P[0] + 2*P[1]) / 3, b_0 = (2*P[0] + P[1]) / 3 (This is not C2, just a segment)
            # The system solution works for n=1 if matrix/vector are built correctly.
            # A[0,0]=2, B[0]=P[0]+2*P[1] => a0 = (P[0]+2P[1])/2. This might be okay.
            pass # A[0,0]=2, B[0]=p[0]+2p[1] is already set.

        # Solve for first_control_points (a_i)
        # Using np.linalg.solve for robustness
        try:
            first_control_points_np = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            # Fallback or error handling if matrix is singular
            # For very short paths or co-linear points, this might happen.
            # A simple fallback: use midpoints or linear interpolation for control points
            print(f"Warning: Bezier control point calculation failed (singular matrix) for {n+1} points. Path may not be fully smoothed.")
            # Fallback: create less smooth control points. E.g., a_i = P_i, b_i = P_{i+1} (straight lines)
            # Or a_i = (2P_i + P_{i+1})/3, b_i = (P_i + 2P_{i+1})/3
            first_control_points = []
            second_control_points = []
            for i in range(n):
                pt_i = p[i]
                pt_i1 = p[i+1]
                a_i = (2*pt_i + pt_i1) / 3.0
                b_i = (pt_i + 2*pt_i1) / 3.0
                first_control_points.append(a_i)
                second_control_points.append(b_i)
            return first_control_points, second_control_points


        first_control_points = [val for val in first_control_points_np]

        # Calculate second_control_points (b_i) using a_i
        second_control_points = []
        for i in range(n):
            if i < n - 1:
                b_i = 2 * p[i+1] - first_control_points[i+1]
            else: # Last segment b_{n-1}
                b_i = (p[n] + first_control_points[n-1]) / 2
            second_control_points.append(b_i)
            
        return first_control_points, second_control_points


    def _bezier_smooth(self, positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Smooth the path using a sequence of cubic Bezier curves interpolating the points."""
        if len(positions) < 2: # Need at least two points to form a segment
            return positions
        if len(positions) < 3: # No "corners" to smooth with this method, effectively a straight line
             # For 2 points, can just return them, or generate points along the line.
             # The LoS path would already be [P0, P1].
             # If self.bezier_segment_points > 1, could interpolate if needed.
             # For now, return as is, consistent with needing 3+ points for the control point calculation.
            return positions

        num_segments = len(positions) - 1
        
        first_cps, second_cps = self._calculate_bezier_control_points(positions)

        if not first_cps: # Fallback from _calculate_bezier_control_points
            print("Warning: Bezier smoothing using fallback due to control point calculation issues.")
            # Simply return the LoS path if control points could not be determined well.
            return positions


        smoothed_path: List[Tuple[float, float]] = []
        path_points_np = [np.array(pt) for pt in positions]

        for i in range(num_segments):
            p0 = path_points_np[i]
            p3 = path_points_np[i+1]
            # Control points for segment P_i to P_{i+1} are first_cps[i] (a_i) and second_cps[i] (b_i)
            p1 = first_cps[i]
            p2 = second_cps[i]
            
            # Sample points along this Bezier segment
            # Include the start point (t=0) only for the first segment
            # For subsequent segments, t=0 point (which is p0 of this segment) was the t=1 point of previous.
            start_t_idx = 1 if i > 0 else 0
            
            for t_idx in range(start_t_idx, self.bezier_segment_points + 1):
                t = t_idx / float(self.bezier_segment_points)
                if t > 1.0: t = 1.0 # Clamp t to 1.0
                
                # For the very first point of the whole path (i=0, t_idx=0)
                if i == 0 and t_idx == 0:
                     point_np = self._evaluate_cubic_bezier(p0, p1, p2, p3, 0.0)
                     smoothed_path.append(tuple(point_np))
                     continue

                # For all other points, including the end of each segment
                if t_idx > 0 : # Avoid re-calculating t=0 if it's not the absolute start
                    point_np = self._evaluate_cubic_bezier(p0, p1, p2, p3, t)
                    # Add if not identical to previous point (can happen with low bezier_segment_points)
                    if not smoothed_path or np.any(point_np != smoothed_path[-1]):
                         smoothed_path.append(tuple(point_np))
        
        if not smoothed_path and positions: # Should not happen if positions is not empty
             return positions # Safety return
        
        # Ensure the very last point from the original path is included if not already by t=1.0
        # This can happen if self.bezier_segment_points doesn't perfectly land on it.
        # Or if rounding errors occur.
        last_original_point = positions[-1]
        if smoothed_path and smoothed_path[-1] != last_original_point:
            # Check if the last calculated point is very close to the last original point
            if np.linalg.norm(np.array(smoothed_path[-1]) - np.array(last_original_point)) > 1e-3: # Threshold
                smoothed_path.append(last_original_point)
            else: # If very close, replace with original to ensure exact match
                smoothed_path[-1] = last_original_point
                
        return smoothed_path


class MovementController:
    """Converts high-level paths to frame-by-frame N++ input commands using accurate physics."""
    
    def __init__(self, physics_params: Optional[dict] = None):
        self.physics = physics_params if physics_params else {}
        self.command_buffer: List[dict] = []
        
        # Allow override of physics constants if needed
        self.max_jump_hold_frames = self.physics.get('max_jump_hold_frames', MAX_JUMP_DURATION)
        self.ninja_max_speed = self.physics.get('ninja_speed', MAX_HOR_SPEED)
        self.jump_vel_y = self.physics.get('jump_vel_y', JUMP_FLAT_GROUND_Y)
        self.wall_jump_vel_y = self.physics.get('wall_jump_vel_y', JUMP_WALL_REGULAR_Y)
        self.wall_jump_vel_x = self.physics.get('wall_jump_vel_x', JUMP_WALL_REGULAR_X)

    def generate_commands(self, initial_pos: Tuple[float, float],
                         initial_vel: Tuple[float, float],
                         waypoint_path: List[Tuple[float, float]], # Smoothed world coordinate path
                         edge_types: List[str] # Type of movement for each segment (walk, jump, etc.)
                         ) -> List[dict]:
        """Generate frame-by-frame movement commands to follow the waypoint_path."""
        
        self.command_buffer = []
        if not waypoint_path or len(waypoint_path) < 1: # Path can be a single waypoint (target)
            return []
        
        if not waypoint_path: return [] # Should be caught by above, but defensive.

        # Simulate frame by frame, trying to reach next waypoint
        current_sim_pos = list(initial_pos)
        current_sim_vel = list(initial_vel)
        total_frames_offset = 0 # Keep track of global frame number for commands

        for i in range(len(waypoint_path)): # Iterate to each waypoint
            target_wp = waypoint_path[i]
            # For the last waypoint, edge_type might not be relevant if it's just a target point.
            # If waypoint_path has N points, edge_types has N-1 segments.
            edge_type = edge_types[i-1] if i > 0 and i-1 < len(edge_types) else 'walk' 
            # If it's the first waypoint, and it's the only one, treat as 'stop' or 'reach'.
            # If i=0, we are trying to reach the first waypoint from initial_pos.

            # Determine segment start. For i=0, it's initial_pos. For i>0, it's waypoint_path[i-1]
            segment_start_wp = initial_pos if i == 0 else waypoint_path[i-1]

            segment_commands: List[dict] = []
            sim_result: Dict[str, any] = {} # To store pos, vel, commands from segment generator

            # print(f"Segment {i}: {segment_start_wp} -> {target_wp}, type: {edge_type}, current_vel: {current_sim_vel}")

            if edge_type == 'walk' or edge_type == 'run':
                sim_result = self._generate_walk_run_commands(
                    current_sim_pos, target_wp, current_sim_vel, total_frames_offset
                )
            elif edge_type == 'jump':
                 # This needs the full trajectory data from the nav_graph edge
                sim_result = self._generate_abstract_jump_commands(
                    current_sim_pos, target_wp, current_sim_vel, total_frames_offset
                ) # Stub, needs trajectory
            elif edge_type == 'wall_jump':
                sim_result = self._generate_abstract_wall_jump_commands(
                    current_sim_pos, target_wp, current_sim_vel, total_frames_offset
                ) # Stub, needs trajectory
            elif edge_type == 'fall' or edge_type == 'walk_gap': # walk_gap is like a small fall/controlled step
                sim_result = self._generate_fall_commands(
                    current_sim_pos, target_wp, current_sim_vel, total_frames_offset
                ) # Stub
            else: # Includes specific jump types like 'min_h_floor_jump', etc.
                  # These should ideally be handled by a more generic jump executor
                  # that can follow a pre-calculated trajectory from nav_graph.
                  # For now, default to a simple 'reach' if specific handler not found.
                print(f"Warning: Unknown or unhandled edge type '{edge_type}' for command generation. Attempting generic reach.")
                sim_result = self._generate_walk_run_commands( # Fallback to walk/run
                    current_sim_pos, target_wp, current_sim_vel, total_frames_offset, is_generic_reach=True
                )
            
            segment_commands = sim_result.get('commands', [])
            if segment_commands:
                self.command_buffer.extend(segment_commands)
                total_frames_offset += len(segment_commands)
                current_sim_pos = list(sim_result.get('final_pos', target_wp))
                current_sim_vel = list(sim_result.get('final_vel', [0.0, 0.0]))
            else: # No commands generated, assume we are at target or stuck
                current_sim_pos = list(target_wp) # Magically teleport for next segment
                current_sim_vel = [0.0, 0.0]       # Reset velocity

        return self.command_buffer

    def _generate_walk_run_commands(self, start_pos_sim: List[float], 
                                   end_wp: Tuple[float, float],
                                   initial_vel_sim: List[float], 
                                   base_frame: int,
                                   max_frames: int = 300, # Max frames for this segment
                                   is_generic_reach: bool = False # If true, y-coord is also a target
                                   ) -> Dict[str, any]:
        """
        Generate commands to move horizontally from start_pos_sim to end_wp[0] on a surface.
        Uses N++ ground physics for acceleration and deceleration.
        Assumes entity is on a surface where walk/run is possible.
        Vertical movement is not directly controlled but simulated if is_generic_reach.
        Returns a dict: {'commands': [], 'final_pos': [], 'final_vel': []}
        """
        # print(f"Walk/Run from {start_pos_sim} (vel {initial_vel_sim}) to {end_wp}")
        commands = []
        pos = list(start_pos_sim)
        vel = list(initial_vel_sim)
        
        target_x = end_wp[0]
        target_y = end_wp[1] # Used if is_generic_reach or for y-pos updates

        # Tolerance for reaching target
        target_tolerance_x = 1.0 # pixels
        target_tolerance_y = 1.0 if is_generic_reach else float('inf') # Effectively ignore Y if not generic reach

        for frame_count in range(max_frames):
            current_frame_abs = base_frame + frame_count
            cmd = {'frame': current_frame_abs, 'jump': False, 'left': False, 'right': False, 'action': 'walk/run'}

            # Horizontal control
            dx_to_target = target_x - pos[0]
            
            # Basic deceleration logic:
            # Estimate stopping distance: v_current^2 / (2 * effective_decel)
            # Effective_decel is GROUND_ACCEL (if actively braking) or related to FRICTION_GROUND.
            # Friction: v_new = v_old * FRICTION_GROUND. Change = v_old * (1 - FRICTION_GROUND).
            # If passive deceleration: vel[0] *= self.FRICTION_GROUND
            # If active braking: vel[0] -= sign(vel[0]) * self.GROUND_ACCEL
            
            # Simplified PID-like control for now
            # (More advanced: calculate if we need to brake to stop at target_x)
            is_braking = False
            if abs(dx_to_target) < target_tolerance_x: # Close enough horizontally
                cmd['left'] = False
                cmd['right'] = False
                is_braking = True # Passively decelerate or actively brake
            elif dx_to_target > 0: # Target is to the right
                cmd['right'] = True
                if vel[0] < 0: is_braking = True # Counter-steering to brake
            else: # Target is to the left
                cmd['left'] = True
                if vel[0] > 0: is_braking = True # Counter-steering to brake

            # Apply ground friction (passive deceleration when no input or aligned input)
            # N++ physics: friction always applies, then acceleration.
            original_vx = vel[0]
            vel[0] *= FRICTION_GROUND
            
            # Apply acceleration from input
            if cmd['right'] and vel[0] < MAX_HOR_SPEED:
                vel[0] += GROUND_ACCEL
            elif cmd['left'] and vel[0] > -MAX_HOR_SPEED:
                vel[0] -= GROUND_ACCEL
            
            # Clamp to max speed
            vel[0] = max(-MAX_HOR_SPEED, min(MAX_HOR_SPEED, vel[0]))

            # If braking and input was applied, ensure velocity reduces towards zero if overshooting
            # This part is tricky. If trying to stop precisely, may need to oscillate or reduce input.
            # For now, if is_braking and no input, friction handles it.
            # If is_braking due to counter-steering, GROUND_ACCEL is already opposing.

            pos[0] += vel[0]
            
            # Vertical component (simplified for walk/run on flat surface, or simple fall for generic_reach)
            if is_generic_reach: # If trying to reach a y-coordinate too (e.g. falling to a waypoint)
                vel[1] += GRAVITY_FALL # Assume always falling if generic_reach controls Y
                vel[1] *= DRAG_REGULAR # Basic air drag
                pos[1] += vel[1]
            else: # Assume on a surface, y velocity is 0 unless path implies slope (not handled here)
                vel[1] = 0.0 
                pos[1] = start_pos_sim[1] # Keep Y constant if not generic_reach (walking on flat ground)

            commands.append(cmd)

            # Check for arrival (horizontal first, then vertical if applicable)
            if abs(pos[0] - target_x) < target_tolerance_x:
                if not is_generic_reach or abs(pos[1] - target_y) < target_tolerance_y:
                    # Arrived at target
                    # To stop precisely, we might need a few more frames of zero input / slight adjustment
                    # For now, we stop applying L/R input and let friction take over if needed.
                    # Final velocity will be what friction results in after this frame.
                    if abs(vel[0]) > 0.1: # If still moving significantly, add a frame of no input
                         final_cmd = {'frame': current_frame_abs + 1, 'jump': False, 'left': False, 'right': False, 'action': 'stop'}
                         commands.append(final_cmd)
                         vel[0] *= FRICTION_GROUND # One last friction application
                    # pos might be slightly off, but vel should be low.
                    break 
            
            # Failsafe if overshot significantly and oscillating (basic version)
            if frame_count > 10 and len(commands) > 2:
                if (commands[-1]['left'] and commands[-2]['right']) or \
                   (commands[-1]['right'] and commands[-2]['left']):
                   if abs(dx_to_target) < 5.0 : # If close and oscillating, stop
                        # print("Overshoot/oscillation detected, stopping.")
                        break


        final_pos_tuple = tuple(pos)
        final_vel_tuple = tuple(vel)
        # print(f"Walk/Run segment ended. Frames: {len(commands)}, Pos: {final_pos_tuple}, Vel: {final_vel_tuple}")
        return {'commands': commands, 'final_pos': final_pos_tuple, 'final_vel': final_vel_tuple}

    def _generate_abstract_jump_commands(self, start_wp: List[float], end_wp: Tuple[float, float],
                                 current_vel: List[float], base_frame: int) -> Dict[str, any]:
        """Generate commands for a generic jump segment. Needs actual trajectory data to be accurate."""
        print(f"Warning: _generate_abstract_jump_commands from {start_wp} to {end_wp} is a stub.")
        # This function should ideally take a JumpTrajectory object from the graph edge
        # and generate commands to follow its frame-by-frame positions/velocities,
        # or at least use its initial_velocity and jump_type.
        
        commands = []
        # Simplified: just a single jump input, then hope for the best.
        # This needs to be replaced by a controller that follows the pre-calculated jump trajectory.
        dx = end_wp[0] - start_wp[0]

        cmd_init = {
            'frame': base_frame, 'jump': True, 
            'left': dx < -5, 'right': dx > 5, # Basic directional aim
            'action': 'jump_initiate_abstract'
        }
        commands.append(cmd_init)

        # Simulate a few frames of holding jump and air control
        hold_frames = 10 # Arbitrary
        air_frames = 20  # Arbitrary
        
        sim_pos = list(start_wp)
        sim_vel = list(current_vel) # This should be reset by the jump impulse
        
        # Apply a mock jump impulse (very simplified)
        sim_vel[1] = JUMP_FLAT_GROUND_Y 
        if cmd_init['right']: sim_vel[0] = min(MAX_HOR_SPEED/2, sim_vel[0] + 1.0)
        if cmd_init['left']: sim_vel[0] = max(-MAX_HOR_SPEED/2, sim_vel[0] - 1.0)


        for i in range(1, hold_frames + air_frames):
            frame_abs = base_frame + i
            is_holding_jump = i < hold_frames
            
            # Simplified physics update
            sim_vel[1] += GRAVITY_JUMP if is_holding_jump and sim_vel[1] < 0 else GRAVITY_FALL
            sim_vel[0] *= DRAG_REGULAR # Simplified air drag on x
            sim_vel[1] *= DRAG_REGULAR # Simplified air drag on y

            # Air control
            current_dx_to_target = end_wp[0] - sim_pos[0]
            input_left = False
            input_right = False
            if current_dx_to_target < -NINJA_RADIUS: # Target to left
                input_left = True
                if sim_vel[0] > -MAX_HOR_SPEED: sim_vel[0] -= AIR_ACCEL
            elif current_dx_to_target > NINJA_RADIUS: # Target to right
                input_right = True
                if sim_vel[0] < MAX_HOR_SPEED: sim_vel[0] += AIR_ACCEL
            
            sim_pos[0] += sim_vel[0]
            sim_pos[1] += sim_vel[1]
            
            commands.append({
                'frame': frame_abs, 'jump': is_holding_jump,
                'left': input_left, 'right': input_right,
                'action': 'jump_air_control_abstract'
            })

            # Basic arrival check (very crude)
            if abs(sim_pos[0] - end_wp[0]) < NINJA_RADIUS and \
               abs(sim_pos[1] - end_wp[1]) < NINJA_RADIUS:
                break
        
        return {'commands': commands, 'final_pos': tuple(sim_pos), 'final_vel': tuple(sim_vel)}

    def _generate_abstract_wall_jump_commands(self, start_wp: List[float], end_wp: Tuple[float, float],
                                     current_vel: List[float], base_frame: int) -> Dict[str, any]:
        print("Warning: _generate_abstract_wall_jump_commands is largely a stub.")
        # Similar to jump, this needs trajectory data.
        # For now, very basic wall jump and air control.
        commands = []
        dx = end_wp[0] - start_wp[0]
        
        # Assume jumping off a left wall if target is to the right, etc.
        # This is a guess; true wall normal should come from surface or trajectory.
        is_left_wall_jump = dx > 0 

        cmd_init = {
            'frame': base_frame, 'jump': True,
            'left': not is_left_wall_jump, # Press away from wall
            'right': is_left_wall_jump,  # Press away from wall
            'action': 'wall_jump_initiate_abstract'
        }
        commands.append(cmd_init)
        
        sim_pos = list(start_wp)
        sim_vel = list(current_vel)

        # Mock wall jump impulse
        sim_vel[1] = JUMP_WALL_REGULAR_Y
        sim_vel[0] = JUMP_WALL_REGULAR_X if is_left_wall_jump else -JUMP_WALL_REGULAR_X
        
        hold_frames = 5  # Wall jump holds are often short
        air_frames = 25

        for i in range(1, hold_frames + air_frames):
            frame_abs = base_frame + i
            is_holding_jump = i < hold_frames

            sim_vel[1] += GRAVITY_JUMP if is_holding_jump and sim_vel[1] < 0 else GRAVITY_FALL
            sim_vel[0] *= DRAG_REGULAR
            sim_vel[1] *= DRAG_REGULAR

            current_dx_to_target = end_wp[0] - sim_pos[0]
            input_left = False
            input_right = False
            if current_dx_to_target < -NINJA_RADIUS:
                input_left = True
                if sim_vel[0] > -MAX_HOR_SPEED: sim_vel[0] -= AIR_ACCEL
            elif current_dx_to_target > NINJA_RADIUS:
                input_right = True
                if sim_vel[0] < MAX_HOR_SPEED: sim_vel[0] += AIR_ACCEL
            
            sim_pos[0] += sim_vel[0]
            sim_pos[1] += sim_vel[1]

            commands.append({
                'frame': frame_abs, 'jump': is_holding_jump,
                'left': input_left, 'right': input_right,
                'action': 'wall_jump_air_control_abstract'
            })
            if abs(sim_pos[0] - end_wp[0]) < NINJA_RADIUS and \
               abs(sim_pos[1] - end_wp[1]) < NINJA_RADIUS: # Basic arrival
                break

        return {'commands': commands, 'final_pos': tuple(sim_pos), 'final_vel': tuple(sim_vel)}


    def _generate_fall_commands(self, start_wp: List[float], end_wp: Tuple[float, float],
                                current_vel: List[float], base_frame: int) -> Dict[str, any]:
        print("Warning: _generate_fall_commands is a stub.")
        # Control horizontal movement during fall to reach end_wp[0].
        # Vertical movement is primarily due to gravity.
        commands = []
        sim_pos = list(start_wp)
        sim_vel = list(current_vel)
        max_fall_frames = 200 # Safety break

        for i in range(max_fall_frames):
            frame_abs = base_frame + i
            
            sim_vel[1] += GRAVITY_FALL
            sim_vel[0] *= DRAG_REGULAR 
            sim_vel[1] *= DRAG_REGULAR

            current_dx_to_target = end_wp[0] - sim_pos[0]
            input_left = False
            input_right = False
            if current_dx_to_target < -1.0: # Target to left
                input_left = True
                if sim_vel[0] > -MAX_HOR_SPEED: sim_vel[0] -= AIR_ACCEL
            elif current_dx_to_target > 1.0: # Target to right
                input_right = True
                if sim_vel[0] < MAX_HOR_SPEED: sim_vel[0] += AIR_ACCEL
            
            sim_pos[0] += sim_vel[0]
            sim_pos[1] += sim_vel[1]

            commands.append({
                'frame': frame_abs, 'jump': False, 
                'left': input_left, 'right': input_right,
                'action': 'fall_control'
            })

            # Arrival condition: pass below the y-target OR get close on x & y
            if sim_pos[1] > end_wp[1] - NINJA_RADIUS: # If fallen to target height
                 if abs(sim_pos[0] - end_wp[0]) < NINJA_RADIUS * 2: # And horizontally close
                    break
            if abs(sim_pos[0] - end_wp[0]) < NINJA_RADIUS and \
               abs(sim_pos[1] - end_wp[1]) < NINJA_RADIUS : # General proximity
                break
            if sim_pos[1] > start_wp[1] + 480: # Fallen too far (e.g. 20 tiles)
                break
        
        return {'commands': commands, 'final_pos': tuple(sim_pos), 'final_vel': tuple(sim_vel)}
