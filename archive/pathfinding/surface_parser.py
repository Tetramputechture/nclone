import numpy as np
import math
from enum import Enum
from typing import List, Tuple, Dict, Optional

from ..tile_definitions import (
    TILE_SEGMENT_ORTHO_MAP,
    TILE_SEGMENT_DIAG_MAP,
    TILE_SEGMENT_CIRCULAR_MAP,
)
from ..constants import TILE_PIXEL_SIZE

class SurfaceType(Enum):
    FLOOR = 1
    WALL_LEFT = 2
    WALL_RIGHT = 3
    CEILING = 4
    SLOPE = 5
    CURVE_CONCAVE = 6  # Quarter moons
    CURVE_CONVEX = 7   # Quarter pipes

class Surface:
    """Represents a continuous traversable surface in the level"""
    def __init__(self, surface_type: SurfaceType, tiles: List[Tuple[int, int]]):
        self.type = surface_type
        self.tiles = tiles  # List of (x, y) grid positions
        self.start_pos = None  # World coordinates
        self.end_pos = None
        self.normal = None  # Surface normal vector
        self.length = 0
        self.angle = 0 # For slopes
        # self._calculate_properties() # Call explicitly after properties are set or after merge
    
    def _calculate_properties(self):
        """Calculate surface properties from tile positions"""
        # Convert grid coordinates to world coordinates (24x24 pixel tiles)
        world_tiles = [(x * 24, y * 24) for x, y in self.tiles]
        if not world_tiles:
            print("Warning: Surface created with no tiles.")
            return

        self.start_pos = world_tiles[0]
        self.end_pos = world_tiles[-1]
        
        # Calculate surface normal based on type
        if self.type == SurfaceType.FLOOR:
            self.normal = (0, -1)
        elif self.type == SurfaceType.CEILING:
            self.normal = (0, 1)
        elif self.type == SurfaceType.WALL_LEFT:
            self.normal = (1, 0)
        elif self.type == SurfaceType.WALL_RIGHT:
            self.normal = (-1, 0)
        # Slope normal is calculated in _parse_slope_surface
        
        # Calculate length for path cost estimation
        dx = self.end_pos[0] - self.start_pos[0]
        dy = self.end_pos[1] - self.start_pos[1]
        self.length = np.sqrt(dx*dx + dy*dy)

class SurfaceParser:
    """Parses level geometry into navigable surfaces using actual N++ tile definitions"""
    
    # Actual tile type definitions from the simulation
    TILE_EMPTY = 0
    TILE_SOLID = 1
    TILE_HALF_TOP = 2
    TILE_HALF_RIGHT = 3
    TILE_HALF_BOTTOM = 4
    TILE_HALF_LEFT = 5
    TILE_SLOPE_45_TYPES = [6, 7, 8, 9]  # 45-degree slopes
    TILE_QUARTER_MOONS = [10, 11, 12, 13]  # Concave curves
    TILE_QUARTER_PIPES = [14, 15, 16, 17]  # Convex curves
    TILE_MILD_SLOPES = list(range(18, 26))  # Mild slopes (various angles)
    TILE_STEEP_SLOPES = list(range(26, 34))  # Steep slopes (various angles)
    TILE_GLITCHED = [34, 35, 36, 37]  # Special glitched tiles
    
    def __init__(self, tile_map: np.ndarray):
        # Store map and dimensions (row-major: [y, x])
        self.tile_map = tile_map
        self.grid_height = tile_map.shape[0]
        self.grid_width = tile_map.shape[1]

        self.surfaces: List[Surface] = []
        
    def parse_surfaces(self) -> List[Surface]:
        """Extract all continuous surfaces from the tile map using actual tile definitions"""
        self.surfaces = []
        visited = np.zeros_like(self.tile_map, dtype=bool)
        
        # Parse surfaces based on actual tile geometry
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if visited[y, x]:
                    continue
                
                tile_type = int(self.tile_map[y, x])
                if tile_type == self.TILE_EMPTY:
                    continue
                
                # Parse different surface types based on tile geometry
                surfaces_from_tile = self._parse_tile_surfaces(x, y, tile_type)
                for surface in surfaces_from_tile:
                    if surface:
                        self.surfaces.append(surface)
                
                visited[y, x] = True
        
        # Group adjacent surfaces of the same type
        self._merge_adjacent_surfaces()
        
        return self.surfaces
    
    def _parse_tile_surfaces(self, x: int, y: int, tile_type: int) -> List[Surface]:
        """Parse all surfaces from a single tile using actual tile definitions."""
        surfaces = []
        
        # Parse orthogonal surfaces (floors, walls, ceilings)
        if tile_type in TILE_SEGMENT_ORTHO_MAP:
            ortho_surfaces = self._parse_orthogonal_surfaces(x, y, tile_type)
            surfaces.extend(ortho_surfaces)
        
        # Parse diagonal surfaces (slopes)
        if tile_type in TILE_SEGMENT_DIAG_MAP:
            slope_surface = self._parse_diagonal_surface(x, y, tile_type)
            if slope_surface:
                surfaces.append(slope_surface)
        
        # Parse circular surfaces (curves)
        if tile_type in TILE_SEGMENT_CIRCULAR_MAP:
            curve_surface = self._parse_circular_surface(x, y, tile_type)
            if curve_surface:
                surfaces.append(curve_surface)
        
        return surfaces
    
    def _parse_orthogonal_surfaces(self, x: int, y: int, tile_type: int) -> List[Surface]:
        """Parse orthogonal surfaces (floors, walls, ceilings) from tile."""
        surfaces = []
        # Ensure TILE_SEGMENT_ORTHO_MAP is accessed safely
        if tile_type not in TILE_SEGMENT_ORTHO_MAP:
            # print(f"Warning: tile_type {tile_type} not in TILE_SEGMENT_ORTHO_MAP. Skipping orthogonal parse for tile ({x},{y}).")
            return surfaces
        ortho_data = TILE_SEGMENT_ORTHO_MAP[tile_type]
        
        # Check for floor surfaces (top edges with downward normal)
        for i in range(3):  # Top edge segments
            if ortho_data[i] == -1:  # Downward normal (floor)
                surface = Surface(SurfaceType.FLOOR, [(x, y)])
                surface.start_pos = (x * TILE_PIXEL_SIZE + i * 8, y * TILE_PIXEL_SIZE)
                surface.end_pos = (x * TILE_PIXEL_SIZE + (i + 1) * 8, y * TILE_PIXEL_SIZE)
                surface.normal = (0, -1)
                dx = surface.end_pos[0] - surface.start_pos[0]
                dy = surface.end_pos[1] - surface.start_pos[1]
                surface.length = np.sqrt(dx*dx + dy*dy)
                surfaces.append(surface)
        
        # Check for ceiling surfaces (bottom edges with upward normal)
        for i in range(3, 6):  # Bottom edge segments
            if ortho_data[i] == 1:  # Upward normal (ceiling)
                surface = Surface(SurfaceType.CEILING, [(x, y)])
                surface.start_pos = (x * TILE_PIXEL_SIZE + (i - 3) * 8, y * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE)
                surface.end_pos = (x * TILE_PIXEL_SIZE + (i - 2) * 8, y * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE)
                surface.normal = (0, 1)
                dx = surface.end_pos[0] - surface.start_pos[0]
                dy = surface.end_pos[1] - surface.start_pos[1]
                surface.length = np.sqrt(dx*dx + dy*dy)
                surfaces.append(surface)
        
        # Check for wall surfaces
        for i in range(6, 9):  # Left edge segments
            if ortho_data[i] == 1:  # Rightward normal (left wall)
                surface = Surface(SurfaceType.WALL_LEFT, [(x, y)])
                surface.start_pos = (x * TILE_PIXEL_SIZE, y * TILE_PIXEL_SIZE + (i - 6) * 8)
                surface.end_pos = (x * TILE_PIXEL_SIZE, y * TILE_PIXEL_SIZE + (i - 5) * 8)
                surface.normal = (1, 0)
                dx = surface.end_pos[0] - surface.start_pos[0]
                dy = surface.end_pos[1] - surface.start_pos[1]
                surface.length = np.sqrt(dx*dx + dy*dy)
                surfaces.append(surface)
        
        for i in range(9, 12):  # Right edge segments
            if ortho_data[i] == -1:  # Leftward normal (right wall)
                surface = Surface(SurfaceType.WALL_RIGHT, [(x, y)])
                surface.start_pos = (x * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE, y * TILE_PIXEL_SIZE + (i - 9) * 8)
                surface.end_pos = (x * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE, y * TILE_PIXEL_SIZE + (i - 8) * 8)
                surface.normal = (-1, 0)
                dx = surface.end_pos[0] - surface.start_pos[0]
                dy = surface.end_pos[1] - surface.start_pos[1]
                surface.length = np.sqrt(dx*dx + dy*dy)
                surfaces.append(surface)
        
        return surfaces
    
    def _parse_diagonal_surface(self, x: int, y: int, tile_type: int) -> Optional[Surface]:
        """Parse diagonal surface (slope) from tile."""
        if tile_type not in TILE_SEGMENT_DIAG_MAP:
            return None
        diag_data = TILE_SEGMENT_DIAG_MAP[tile_type]
        start_offset, end_offset = diag_data
        
        surface = Surface(SurfaceType.SLOPE, [(x, y)])
        surface.start_pos = (x * TILE_PIXEL_SIZE + start_offset[0], y * TILE_PIXEL_SIZE + start_offset[1])
        surface.end_pos = (x * TILE_PIXEL_SIZE + end_offset[0], y * TILE_PIXEL_SIZE + end_offset[1])
        
        dx = surface.end_pos[0] - surface.start_pos[0]
        dy = surface.end_pos[1] - surface.start_pos[1]
        surface.length = math.sqrt(dx*dx + dy*dy) # Set length directly
        
        if surface.length > 0:
            surface.normal = (-dy/surface.length, dx/surface.length)
        else:
            surface.normal = (0, -1) # Default for zero-length slope
        
        if dx != 0:
            surface.angle = math.degrees(math.atan(dy / dx))
        else:
            surface.angle = 90.0 if dy > 0 else -90.0
            
        return surface
    
    def _parse_circular_surface(self, x: int, y: int, tile_type: int) -> Optional[Surface]:
        """Parse circular surface (curve) from tile."""
        if tile_type not in TILE_SEGMENT_CIRCULAR_MAP:
            return None
        circ_data = TILE_SEGMENT_CIRCULAR_MAP[tile_type]
        center_offset, quadrant, is_concave = circ_data # Assuming this structure
        
        surface_type = SurfaceType.CURVE_CONCAVE if is_concave else SurfaceType.CURVE_CONVEX
        
        surface = Surface(surface_type, [(x, y)])
        # For now, start/end are center, so length is 0. This might need refinement.
        surface.start_pos = (x * TILE_PIXEL_SIZE + center_offset[0], y * TILE_PIXEL_SIZE + center_offset[1])
        surface.end_pos = surface.start_pos 
        
        surface.normal = (quadrant[0], quadrant[1])  # Placeholder normal
        surface.radius = TILE_PIXEL_SIZE / 2 # Example, actual radius depends on curve type
        surface.quadrant = quadrant
        surface.concave = is_concave
        
        # Length of a point surface is 0
        surface.length = 0.0
            
        return surface
    
    def _merge_adjacent_surfaces(self):
        """Merge adjacent surfaces of the same type to create longer continuous surfaces."""
        if not self.surfaces:
            return

        new_surfaces = []
        # Group surfaces by type
        surface_groups: Dict[SurfaceType, List[Surface]] = {}
        for s in self.surfaces:
            surface_groups.setdefault(s.type, []).append(s)

        for surface_type, surfaces_of_type in surface_groups.items():
            if not surfaces_of_type:
                continue

            # Sort surfaces to make merging easier
            # For floors: sort by y, then x of start_pos
            # For walls: sort by x, then y of start_pos
            if surface_type == SurfaceType.FLOOR or surface_type == SurfaceType.CEILING:
                surfaces_of_type.sort(key=lambda s: (s.start_pos[1], s.start_pos[0]))
            elif surface_type == SurfaceType.WALL_LEFT or surface_type == SurfaceType.WALL_RIGHT:
                surfaces_of_type.sort(key=lambda s: (s.start_pos[0], s.start_pos[1]))
            elif surface_type == SurfaceType.SLOPE:
                # Sort slopes, e.g., primarily by y then x of start_pos, or by dominant axis of slope
                # For simplicity, using y, x as a general sort. Angle similarity will be key for merging.
                surfaces_of_type.sort(key=lambda s: (s.start_pos[1], s.start_pos[0]))
            else: # For curves, merging is more complex, skip for now
                new_surfaces.extend(surfaces_of_type)
                continue

            current_merged_surface: Optional[Surface] = None
            for s in surfaces_of_type:
                if current_merged_surface is None:
                    # Start a new merged surface, make a copy
                    current_merged_surface = Surface(s.type, list(s.tiles)) # Copy tiles list
                    current_merged_surface.start_pos = s.start_pos
                    current_merged_surface.end_pos = s.end_pos
                    current_merged_surface.normal = s.normal
                    current_merged_surface.length = s.length
                    # angle for slopes would need to be handled if they are merged
                else:
                    # Check for adjacency and co-linearity/compatibility
                    can_merge = False
                    epsilon = 1e-5
                    if surface_type == SurfaceType.FLOOR or surface_type == SurfaceType.CEILING:
                        # Horizontal merge: y must be same, x must be continuous
                        if abs(current_merged_surface.end_pos[1] - s.start_pos[1]) < epsilon and \
                           abs(current_merged_surface.end_pos[0] - s.start_pos[0]) < epsilon:
                            can_merge = True
                    elif surface_type == SurfaceType.WALL_LEFT or surface_type == SurfaceType.WALL_RIGHT:
                        # Vertical merge: x must be same, y must be continuous
                        if abs(current_merged_surface.end_pos[0] - s.start_pos[0]) < epsilon and \
                           abs(current_merged_surface.end_pos[1] - s.start_pos[1]) < epsilon:
                            can_merge = True
                    elif surface_type == SurfaceType.SLOPE:
                        angle_tolerance = 1.0 # degrees
                        if abs(current_merged_surface.angle - s.angle) < angle_tolerance and \
                           abs(current_merged_surface.end_pos[0] - s.start_pos[0]) < epsilon and \
                           abs(current_merged_surface.end_pos[1] - s.start_pos[1]) < epsilon:
                            # Check for co-linearity more robustly for slopes
                            # Vector of current merged surface
                            vec1_dx = current_merged_surface.end_pos[0] - current_merged_surface.start_pos[0]
                            vec1_dy = current_merged_surface.end_pos[1] - current_merged_surface.start_pos[1]
                            # Vector of new segment s
                            vec2_dx = s.end_pos[0] - s.start_pos[0]
                            vec2_dy = s.end_pos[1] - s.start_pos[1]
                            
                            # Normalize and check if parallel (dot product close to 1 or -1 if comparing unit vectors)
                            # Or, simpler: check if the point s.end_pos lies on the line defined by
                            # current_merged_surface.start_pos and s.start_pos (which is current_merged_surface.end_pos)
                            # A more direct co-linearity: (y1-y0)*(x2-x1) == (y2-y1)*(x1-x0)
                            x0, y0 = current_merged_surface.start_pos
                            x1, y1 = s.start_pos # current_merged_surface.end_pos
                            x2, y2 = s.end_pos
                            val = (y1 - y0) * (x2 - x1) - (y2 - y1) * (x1 - x0)
                            if abs(val) < epsilon * max(abs(x0),abs(y0),abs(x1),abs(y1),abs(x2),abs(y2),1.0): # Scaled epsilon
                                can_merge = True
                    
                    if can_merge:
                        # Extend current_merged_surface
                        current_merged_surface.end_pos = s.end_pos
                        # Update tiles: add unique parent tiles from s
                        for tile_coord in s.tiles:
                            if tile_coord not in current_merged_surface.tiles:
                                current_merged_surface.tiles.append(tile_coord)
                        # Recalculate length
                        dx = current_merged_surface.end_pos[0] - current_merged_surface.start_pos[0]
                        dy = current_merged_surface.end_pos[1] - current_merged_surface.start_pos[1]
                        current_merged_surface.length = np.sqrt(dx*dx + dy*dy)
                    else:
                        # Finalize current_merged_surface and add to new_surfaces
                        new_surfaces.append(current_merged_surface)
                        # Start a new one with s
                        current_merged_surface = Surface(s.type, list(s.tiles))
                        current_merged_surface.start_pos = s.start_pos
                        current_merged_surface.end_pos = s.end_pos
                        current_merged_surface.normal = s.normal
                        current_merged_surface.length = s.length
            
            # Add the last merged surface being built
            if current_merged_surface is not None:
                new_surfaces.append(current_merged_surface)

        self.surfaces = new_surfaces
    
    def _is_floor_tile(self, x: int, y: int) -> bool:
        """Check if a tile position represents a floor surface"""
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return False
            
        tile = int(self.tile_map[y, x])
        
        # Check if tile has a walkable top surface
        if tile in [self.TILE_SOLID, self.TILE_HALF_TOP]:
            # Also check if the tile above is empty
            if y > 0 and self.tile_map[y-1, x] == self.TILE_EMPTY:
                 return True
            elif y == 0: # Top edge of map
                 return True

        # Check slopes and curves (simplified: treat as floor if walkable from above)
        if tile in self.TILE_SLOPE_45_TYPES + self.TILE_MILD_SLOPES + self.TILE_STEEP_SLOPES:
            if y > 0 and self.tile_map[y-1, x] == self.TILE_EMPTY:
                return True
            elif y == 0:
                return True
            
        return False
    
    def _trace_floor_surface(self, start_x: int, start_y: int, 
                           visited: np.ndarray) -> Optional[Surface]:
        """Trace a continuous floor surface from starting position"""
        tiles = []
        x, y = start_x, start_y
        
        # Trace horizontally connected floor tiles
        while x < self.grid_width and self._is_floor_tile(x, y):
            if not visited[y, x]:
                tiles.append((x, y))
                visited[y, x] = True
                x += 1
            else: # Already visited (e.g. by another surface type or trace)
                break 
        
        if len(tiles) > 0:
            return Surface(SurfaceType.FLOOR, tiles)
        return None

    def _is_wall_tile(self, x: int, y: int) -> bool:
        """Check if a tile position could be part of a wall surface."""
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return False
        
        tile = int(self.tile_map[y, x])
        
        # Solid tiles are walls
        if tile == self.TILE_SOLID:
            # Check for adjacent empty space to confirm it's an actual wall edge
            # Check left
            if x > 0 and self.tile_map[y, x-1] == self.TILE_EMPTY:
                return True
            # Check right
            if x < self.grid_width - 1 and self.tile_map[y, x+1] == self.TILE_EMPTY:
                return True
        
        # Half tiles can also form walls
        if tile == self.TILE_HALF_LEFT and x < self.grid_width - 1 and self.tile_map[y, x+1] == self.TILE_EMPTY: # Exposed right face
            return True
        if tile == self.TILE_HALF_RIGHT and x > 0 and self.tile_map[y, x-1] == self.TILE_EMPTY: # Exposed left face
            return True

        print(f"Warning: _is_wall_tile({x},{y}) is a basic stub. Tile type: {tile}")
        return False # Fallback, needs more robust logic

    def _trace_wall_surface(self, start_x: int, start_y: int, 
                            visited: np.ndarray, wall_type: SurfaceType) -> Optional[Surface]:
        """Trace a continuous wall surface (vertical) from starting position."""
        tiles = []
        x, y = start_x, start_y
        
        # Determine if we are tracing a WALL_LEFT (empty space to its left) or WALL_RIGHT (empty space to its right)
        # This method needs to be called appropriately based on which side is exposed.
        
        # Trace vertically connected wall tiles
        while y < self.grid_height:
            is_valid_wall_segment = False
            if wall_type == SurfaceType.WALL_LEFT:
                # Tile (x,y) is solid, tile (x-1,y) is empty
                if self.tile_map[y, x] != self.TILE_EMPTY and x > 0 and self.tile_map[y, x-1] == self.TILE_EMPTY:
                    is_valid_wall_segment = True
            elif wall_type == SurfaceType.WALL_RIGHT:
                # Tile (x,y) is solid, tile (x+1,y) is empty
                if self.tile_map[y, x] != self.TILE_EMPTY and x < self.grid_width -1 and self.tile_map[y, x+1] == self.TILE_EMPTY:
                    is_valid_wall_segment = True
            
            if is_valid_wall_segment:
                if not visited[y, x]:
                    tiles.append((x, y))
                    visited[y, x] = True
                    y += 1
                else:
                    break # Already visited
            else:
                break # Not a wall segment of the specified type anymore

        if len(tiles) > 0:
            return Surface(wall_type, tiles)
        
        print(f"Warning: _trace_wall_surface({start_x},{start_y}, {wall_type}) is a stub.")
        return None


    def _parse_slope_surface(self, x: int, y: int, tile_type: int) -> Surface:
        """Parse slope tiles into surfaces with proper normals"""
        # Determine slope angle based on tile type
        if tile_type in self.TILE_SLOPE_45_TYPES:
            angle = 45  # degrees
            # Further refinement needed for direction of 45-degree slope based on specific tile_type
        elif tile_type in self.TILE_MILD_SLOPES:
            angle = 22.5  # approximate
        elif tile_type in self.TILE_STEEP_SLOPES:
            angle = 67.5  # steep slopes
        else: # Should not happen if called correctly
            angle = 0 
        
        # Calculate normal vector for slope (assuming upward slope to the right for now)
        # This needs to be adjusted based on the actual slope direction from tile_type
        # For a floor-like slope (walkable top), normal points upwards and left/right.
        # Example: slope rising to the right, normal is (-sin(rad), -cos(rad)) if angle is from horizontal
        # If angle is wall angle from vertical: normal = (-cos(rad), -sin(rad))
        # For now, let's assume a generic upward-pointing normal for a walkable slope.
        # A true N++ slope has a specific normal based on its geometry.
        # For a simple floor slope rising to the right:
        rad = np.radians(angle)
        # Normal for a surface whose "up" is perpendicular to the slope face.
        # If slope goes up and right, normal is up and left.
        normal_x = -np.sin(rad) 
        normal_y = -np.cos(rad)
        
        # Adjust normal based on actual slope orientation (TILE_SLOPE_45_TYPES has 4 types etc.)
        # This is a placeholder, real N++ has specific normals for each slope tile.
        # For example, tile type 6 (slope up-right) vs 7 (slope up-left)
        if tile_type == 7 or tile_type == self.TILE_MILD_SLOPES[1] or tile_type == self.TILE_STEEP_SLOPES[1]: # Slopes up-left
            normal_x = np.sin(rad)

        normal = (normal_x, normal_y)
        
        surface = Surface(SurfaceType.SLOPE, [(x, y)])
        surface.normal = normal
        surface.angle = angle # Store the angle too
        return surface
