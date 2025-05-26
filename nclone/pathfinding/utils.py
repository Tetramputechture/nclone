import numpy as np
import math
from typing import Tuple, List, Optional, Dict, Any

# Import tile definitions from the actual simulation
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../nclone'))
    from tile_definitions import (
        TILE_GRID_EDGE_MAP, 
        TILE_SEGMENT_ORTHO_MAP, 
        TILE_SEGMENT_DIAG_MAP, 
        TILE_SEGMENT_CIRCULAR_MAP
    )
except ImportError:
    # Fallback definitions if import fails
    TILE_GRID_EDGE_MAP = {}
    TILE_SEGMENT_ORTHO_MAP = {}
    TILE_SEGMENT_DIAG_MAP = {}
    TILE_SEGMENT_CIRCULAR_MAP = {}

class CollisionChecker:
    """Checks for collisions with level geometry using actual N++ collision system."""
    
    # Ninja physics constants from the actual simulation
    NINJA_RADIUS = 10.0
    TILE_SIZE = 24
    
    def __init__(self, tile_map: np.ndarray):
        self.tile_map = tile_map
        self.grid_width = tile_map.shape[0]
        self.grid_height = tile_map.shape[1]
        
        # Pre-compute collision segments for efficiency
        self.segments = self._build_collision_segments()
        
    def _build_collision_segments(self) -> List[Dict[str, Any]]:
        """Build collision segments from tile map using actual tile definitions."""
        segments = []
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                tile_type = self.tile_map[x, y]
                if tile_type == 0:  # Empty tile
                    continue
                    
                tile_x = x * self.TILE_SIZE
                tile_y = y * self.TILE_SIZE
                
                # Add orthogonal segments
                if tile_type in TILE_SEGMENT_ORTHO_MAP:
                    ortho_data = TILE_SEGMENT_ORTHO_MAP[tile_type]
                    segments.extend(self._create_ortho_segments(tile_x, tile_y, ortho_data))
                
                # Add diagonal segments
                if tile_type in TILE_SEGMENT_DIAG_MAP:
                    diag_data = TILE_SEGMENT_DIAG_MAP[tile_type]
                    segments.append(self._create_diag_segment(tile_x, tile_y, diag_data))
                
                # Add circular segments
                if tile_type in TILE_SEGMENT_CIRCULAR_MAP:
                    circ_data = TILE_SEGMENT_CIRCULAR_MAP[tile_type]
                    segments.append(self._create_circular_segment(tile_x, tile_y, circ_data))
        
        return segments
    
    def _create_ortho_segments(self, tile_x: float, tile_y: float, ortho_data: List[int]) -> List[Dict[str, Any]]:
        """Create orthogonal line segments from tile data."""
        segments = []
        
        # Horizontal segments (top and bottom edges)
        for i in range(6):
            if ortho_data[i] != 0:
                if i < 3:  # Top edge segments
                    y_pos = tile_y
                    x_start = tile_x + (i * 8)
                    x_end = tile_x + ((i + 1) * 8)
                else:  # Bottom edge segments
                    y_pos = tile_y + self.TILE_SIZE
                    x_start = tile_x + ((i - 3) * 8)
                    x_end = tile_x + ((i - 2) * 8)
                
                segments.append({
                    'type': 'line',
                    'start': (x_start, y_pos),
                    'end': (x_end, y_pos),
                    'normal': (0, -ortho_data[i])
                })
        
        # Vertical segments (left and right edges)
        for i in range(6, 12):
            if ortho_data[i] != 0:
                if i < 9:  # Left edge segments
                    x_pos = tile_x
                    y_start = tile_y + ((i - 6) * 8)
                    y_end = tile_y + ((i - 5) * 8)
                else:  # Right edge segments
                    x_pos = tile_x + self.TILE_SIZE
                    y_start = tile_y + ((i - 9) * 8)
                    y_end = tile_y + ((i - 8) * 8)
                
                segments.append({
                    'type': 'line',
                    'start': (x_pos, y_start),
                    'end': (x_pos, y_end),
                    'normal': (-ortho_data[i], 0)
                })
        
        return segments
    
    def _create_diag_segment(self, tile_x: float, tile_y: float, diag_data: Tuple[Tuple[int, int], Tuple[int, int]]) -> Dict[str, Any]:
        """Create diagonal line segment from tile data."""
        start_offset, end_offset = diag_data
        start = (tile_x + start_offset[0], tile_y + start_offset[1])
        end = (tile_x + end_offset[0], tile_y + end_offset[1])
        
        # Calculate normal vector (perpendicular to segment, pointing outward)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            normal = (-dy/length, dx/length)  # Perpendicular vector
        else:
            normal = (0, 0)
        
        return {
            'type': 'line',
            'start': start,
            'end': end,
            'normal': normal
        }
    
    def _create_circular_segment(self, tile_x: float, tile_y: float, circ_data: Tuple[Tuple[int, int], Tuple[int, int], bool]) -> Dict[str, Any]:
        """Create circular arc segment from tile data."""
        center_offset, quadrant, is_concave = circ_data
        center = (tile_x + center_offset[0], tile_y + center_offset[1])
        
        return {
            'type': 'arc',
            'center': center,
            'radius': self.TILE_SIZE,
            'quadrant': quadrant,
            'concave': is_concave
        }

    def check_collision(self, pos1: Tuple[float, float], pos2: Tuple[float, float], radius: float = None) -> bool:
        """Check for collision along a movement path using swept circle collision."""
        if radius is None:
            radius = self.NINJA_RADIUS
            
        # Use simplified swept circle collision detection
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        
        # Check collision with relevant segments
        for segment in self._get_nearby_segments(pos1, pos2, radius):
            if self._segment_intersects_swept_circle(segment, pos1, dx, dy, radius):
                return True
        
        return False
    
    def _get_nearby_segments(self, pos1: Tuple[float, float], pos2: Tuple[float, float], radius: float) -> List[Dict[str, Any]]:
        """Get segments that could potentially collide with the movement path."""
        # Calculate bounding box of the swept area
        min_x = min(pos1[0], pos2[0]) - radius
        max_x = max(pos1[0], pos2[0]) + radius
        min_y = min(pos1[1], pos2[1]) - radius
        max_y = max(pos1[1], pos2[1]) + radius
        
        # Get grid cells that overlap with the bounding box
        start_grid_x = max(0, int(min_x // self.TILE_SIZE))
        end_grid_x = min(self.grid_width - 1, int(max_x // self.TILE_SIZE))
        start_grid_y = max(0, int(min_y // self.TILE_SIZE))
        end_grid_y = min(self.grid_height - 1, int(max_y // self.TILE_SIZE))
        
        nearby_segments = []
        for segment in self.segments:
            # Simple bounding box check for segment
            if segment['type'] == 'line':
                seg_min_x = min(segment['start'][0], segment['end'][0])
                seg_max_x = max(segment['start'][0], segment['end'][0])
                seg_min_y = min(segment['start'][1], segment['end'][1])
                seg_max_y = max(segment['start'][1], segment['end'][1])
            elif segment['type'] == 'arc':
                seg_min_x = segment['center'][0] - segment['radius']
                seg_max_x = segment['center'][0] + segment['radius']
                seg_min_y = segment['center'][1] - segment['radius']
                seg_max_y = segment['center'][1] + segment['radius']
            else:
                continue
                
            if (seg_max_x >= min_x and seg_min_x <= max_x and 
                seg_max_y >= min_y and seg_min_y <= max_y):
                nearby_segments.append(segment)
        
        return nearby_segments
    
    def _segment_intersects_swept_circle(self, segment: Dict[str, Any], start_pos: Tuple[float, float], 
                                       dx: float, dy: float, radius: float) -> bool:
        """Check if a segment intersects with a swept circle."""
        if segment['type'] == 'line':
            return self._line_intersects_swept_circle(segment, start_pos, dx, dy, radius)
        elif segment['type'] == 'arc':
            return self._arc_intersects_swept_circle(segment, start_pos, dx, dy, radius)
        return False
    
    def _line_intersects_swept_circle(self, segment: Dict[str, Any], start_pos: Tuple[float, float], 
                                    dx: float, dy: float, radius: float) -> bool:
        """Check line segment intersection with swept circle (simplified)."""
        # This is a simplified version of the actual N++ collision detection
        # For a more accurate implementation, we'd need the exact algorithm from physics.py
        
        x1, y1 = segment['start']
        x2, y2 = segment['end']
        px, py = start_pos
        
        # Vector from line start to circle start
        line_dx = x2 - x1
        line_dy = y2 - y1
        to_circle_dx = px - x1
        to_circle_dy = py - y1
        
        # Project circle center onto line
        line_length_sq = line_dx * line_dx + line_dy * line_dy
        if line_length_sq == 0:
            return False
            
        t = max(0, min(1, (to_circle_dx * line_dx + to_circle_dy * line_dy) / line_length_sq))
        
        # Closest point on line segment
        closest_x = x1 + t * line_dx
        closest_y = y1 + t * line_dy
        
        # Distance from circle center to closest point
        dist_x = px - closest_x
        dist_y = py - closest_y
        distance = math.sqrt(dist_x * dist_x + dist_y * dist_y)
        
        return distance < radius
    
    def _arc_intersects_swept_circle(self, segment: Dict[str, Any], start_pos: Tuple[float, float], 
                                   dx: float, dy: float, radius: float) -> bool:
        """Check arc segment intersection with swept circle (simplified)."""
        # Simplified arc collision - check distance to arc center
        center_x, center_y = segment['center']
        px, py = start_pos
        
        dist_to_center = math.sqrt((px - center_x)**2 + (py - center_y)**2)
        arc_radius = segment['radius']
        
        if segment['concave']:
            # For concave arcs (quarter moons), collision if inside the arc
            return dist_to_center < arc_radius + radius
        else:
            # For convex arcs (quarter pipes), collision if too close to outer edge
            return abs(dist_to_center - arc_radius) < radius

    def point_in_wall(self, point: Tuple[float, float]) -> bool:
        """Check if a point is inside solid geometry."""
        x, y = point
        grid_x = int(x // self.TILE_SIZE)
        grid_y = int(y // self.TILE_SIZE)
        
        if (grid_x < 0 or grid_x >= self.grid_width or 
            grid_y < 0 or grid_y >= self.grid_height):
            return True  # Outside map bounds
        
        tile_type = self.tile_map[grid_x, grid_y]
        
        # Simple check: solid tiles (type 1) are walls
        if tile_type == 1:
            return True
        
        # For more complex tiles, we'd need to check against the actual geometry
        # This is a simplified version
        return False
    
    def get_surface_normal_at_point(self, point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Get the surface normal at a given point."""
        # Find the closest segment to the point
        closest_segment = None
        min_distance = float('inf')
        
        for segment in self.segments:
            if segment['type'] == 'line':
                dist = self._point_to_line_distance(point, segment)
                if dist < min_distance:
                    min_distance = dist
                    closest_segment = segment
        
        if closest_segment and min_distance < self.NINJA_RADIUS:
            return closest_segment['normal']
        
        return None
    
    def _point_to_line_distance(self, point: Tuple[float, float], segment: Dict[str, Any]) -> float:
        """Calculate distance from point to line segment."""
        x1, y1 = segment['start']
        x2, y2 = segment['end']
        px, py = point
        
        line_dx = x2 - x1
        line_dy = y2 - y1
        line_length_sq = line_dx * line_dx + line_dy * line_dy
        
        if line_length_sq == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)
        
        t = max(0, min(1, ((px - x1) * line_dx + (py - y1) * line_dy) / line_length_sq))
        closest_x = x1 + t * line_dx
        closest_y = y1 + t * line_dy
        
        return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

class Enemy:
    """Represents an enemy in the game with actual N++ behavior patterns."""
    
    # Enemy type constants from the actual simulation
    TOGGLE_MINE = 1
    GOLD = 2
    EXIT_DOOR = 3
    EXIT_SWITCH = 4
    REGULAR_DOOR = 5
    LOCKED_DOOR = 6
    TRAP_DOOR = 8
    LAUNCH_PAD = 10
    ONE_WAY = 11
    DRONE_CLOCKWISE = 14
    BOUNCE_BLOCK = 17
    THWUMP = 20
    TOGGLE_MINE_TOGGLED = 21
    DEATH_BALL = 25
    DRONE_COUNTER_CLOCKWISE = 26
    SHWUMP = 28
    
    def __init__(self, entity_data: Dict[str, Any]):
        """Initialize enemy from entity data matching the simulation format."""
        self.id = entity_data.get('id', 0)
        self.type = entity_data.get('type', 0)
        self.xpos = entity_data.get('xpos', 0.0)
        self.ypos = entity_data.get('ypos', 0.0)
        self.xpos_old = entity_data.get('xpos_old', self.xpos)
        self.ypos_old = entity_data.get('ypos_old', self.ypos)
        self.xspeed = entity_data.get('xspeed', 0.0)
        self.yspeed = entity_data.get('yspeed', 0.0)
        self.orientation = entity_data.get('orientation', 0)
        self.active = entity_data.get('active', True)
        
        # Set type-specific properties
        self._set_type_properties()
        
    def _set_type_properties(self):
        """Set properties based on enemy type from actual simulation."""
        if self.type == self.THWUMP:
            self.radius = 9.0  # 9x9 pixel square
            self.forward_speed = 20.0 / 7.0  # ~2.86 pixels/frame
            self.backward_speed = 8.0 / 7.0  # ~1.14 pixels/frame
            self.detection_range = 240.0  # 10 tiles
            self.can_crush = True
            
        elif self.type in [self.DRONE_CLOCKWISE, self.DRONE_COUNTER_CLOCKWISE]:
            self.radius = 7.5
            self.speed = 1.0  # Variable based on specific drone type
            self.patrol_mode = 0  # 0-3 different patrol patterns
            self.can_kill = True
            
        elif self.type == self.DEATH_BALL:
            self.radius = 5.0  # Ninja collision radius
            self.env_radius = 8.0  # Environment collision radius
            self.acceleration = 0.04
            self.max_speed = 0.85
            self.can_kill = True
            
        elif self.type in [self.TOGGLE_MINE, self.TOGGLE_MINE_TOGGLED]:
            if self.type == self.TOGGLE_MINE:
                self.radius = 3.5  # Untoggled
            else:
                self.radius = 4.0  # Toggled
            self.can_kill = (self.type == self.TOGGLE_MINE_TOGGLED)
            
        elif self.type == self.BOUNCE_BLOCK:
            self.radius = 9.0  # 9x9 pixel square
            self.spring_constant = 0.0222
            self.damping = 0.98
            self.can_crush = False
            
        elif self.type == self.LAUNCH_PAD:
            self.radius = 6.0
            self.boost_strength = 36.0 / 7.0  # ~5.14 pixels/frame
            
        else:
            # Default properties for other entity types
            self.radius = 6.0
            self.can_kill = False
            self.can_crush = False
    
    def predict_position(self, frames_ahead: int) -> Tuple[float, float]:
        """Predict enemy position after given number of frames."""
        if not self.active:
            return (self.xpos, self.ypos)
            
        if self.type == self.THWUMP:
            return self._predict_thwump_position(frames_ahead)
        elif self.type in [self.DRONE_CLOCKWISE, self.DRONE_COUNTER_CLOCKWISE]:
            return self._predict_drone_position(frames_ahead)
        elif self.type == self.DEATH_BALL:
            return self._predict_death_ball_position(frames_ahead)
        else:
            # Static entities or simple linear movement
            return (self.xpos + self.xspeed * frames_ahead, 
                   self.ypos + self.yspeed * frames_ahead)
    
    def _predict_thwump_position(self, frames_ahead: int) -> Tuple[float, float]:
        """Predict thwump position based on its movement pattern."""
        # Simplified thwump prediction - actual behavior is more complex
        # Thwumps move toward ninja when in line of sight, return to origin otherwise
        
        # For pathfinding purposes, assume worst-case: thwump is moving
        predicted_x = self.xpos + self.xspeed * frames_ahead
        predicted_y = self.ypos + self.yspeed * frames_ahead
        
        return (predicted_x, predicted_y)
    
    def _predict_drone_position(self, frames_ahead: int) -> Tuple[float, float]:
        """Predict drone position based on patrol pattern."""
        # Simplified drone prediction - actual patrol patterns are complex
        # Drones follow walls or wander in grid-based patterns
        
        predicted_x = self.xpos + self.xspeed * frames_ahead
        predicted_y = self.ypos + self.yspeed * frames_ahead
        
        return (predicted_x, predicted_y)
    
    def _predict_death_ball_position(self, frames_ahead: int) -> Tuple[float, float]:
        """Predict death ball position with acceleration toward ninja."""
        # Death balls accelerate toward ninja position
        # For pathfinding, assume they continue current trajectory
        
        predicted_x = self.xpos + self.xspeed * frames_ahead
        predicted_y = self.ypos + self.yspeed * frames_ahead
        
        return (predicted_x, predicted_y)
    
    def get_danger_radius(self, time_frame: int = 60) -> float:
        """Get the radius of danger around this enemy for pathfinding."""
        base_radius = self.radius
        
        if self.type == self.THWUMP:
            # Thwumps can move quickly, expand danger zone
            return base_radius + self.forward_speed * time_frame
        elif self.type in [self.DRONE_CLOCKWISE, self.DRONE_COUNTER_CLOCKWISE]:
            # Drones patrol, moderate danger zone
            return base_radius + self.speed * time_frame * 0.5
        elif self.type == self.DEATH_BALL:
            # Death balls accelerate, large danger zone
            return base_radius + self.max_speed * time_frame
        else:
            return base_radius
    
    def is_deadly(self) -> bool:
        """Check if this enemy can kill the ninja."""
        return getattr(self, 'can_kill', False) or getattr(self, 'can_crush', False)
