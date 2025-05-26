import numpy as np
from enum import Enum
from typing import List, Tuple, Dict, Optional

class SurfaceType(Enum):
    FLOOR = 1
    WALL_LEFT = 2
    WALL_RIGHT = 3
    CEILING = 4
    SLOPE = 5

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
        self._calculate_properties()
    
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
    """Parses level geometry into navigable surfaces"""
    
    # Tile type definitions from the simulation
    TILE_EMPTY = 0
    TILE_SOLID = 1
    TILE_HALF_TOP = 2
    TILE_HALF_RIGHT = 3
    TILE_HALF_BOTTOM = 4
    TILE_HALF_LEFT = 5
    TILE_SLOPE_45_TYPES = [6, 7, 8, 9]
    TILE_CURVES = list(range(10, 18))
    TILE_MILD_SLOPES = list(range(18, 26))
    TILE_STEEP_SLOPES = list(range(26, 34))
    
    def __init__(self, tile_map: np.ndarray):
        self.tile_map = tile_map
        self.surfaces: List[Surface] = []
        self.grid_width = tile_map.shape[0] # Assuming tile_map is (width, height)
        self.grid_height = tile_map.shape[1]
        
    def parse_surfaces(self) -> List[Surface]:
        """Extract all continuous surfaces from the tile map"""
        # Mark visited tiles to avoid duplicates
        visited = np.zeros_like(self.tile_map, dtype=bool)
        
        # Scan for floor surfaces
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if not visited[x, y] and self._is_floor_tile(x, y):
                    surface = self._trace_floor_surface(x, y, visited)
                    if surface:
                        self.surfaces.append(surface)
        
        # Scan for wall surfaces
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if not visited[x, y] and self._is_wall_tile(x, y): # Checks for both left and right walls
                    # Trace left wall
                    surface_left = self._trace_wall_surface(x, y, visited, SurfaceType.WALL_LEFT)
                    if surface_left:
                        self.surfaces.append(surface_left)
                    
                    # Reset visited for this tile if it was part of a left wall, so right wall can also be traced if applicable
                    # This logic might need refinement based on how _is_wall_tile and _trace_wall_surface work
                    # For now, assume _trace_wall_surface marks tiles appropriately for its type
                    
                    # Trace right wall (if not already covered by left wall tracing from adjacent tile)
                    # This might require a more sophisticated check or a different tracing strategy
                    # to avoid double-counting or missing walls.
                    # A simple approach: if a tile is a wall, it could be a left or right wall depending on context.
                    # The current _is_wall_tile doesn't differentiate.
                    # Let's assume _trace_wall_surface handles the direction.
                    surface_right = self._trace_wall_surface(x, y, visited, SurfaceType.WALL_RIGHT)
                    if surface_right:
                         self.surfaces.append(surface_right)


        # Scan for slope surfaces (can be integrated with floor or separate)
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if not visited[x,y] and self.tile_map[x,y] in self.TILE_SLOPE_45_TYPES + self.TILE_MILD_SLOPES + self.TILE_STEEP_SLOPES:
                    tile_type = self.tile_map[x,y]
                    # Slopes are typically single tiles or short segments in N++
                    # For simplicity, let's treat each slope tile as a small surface.
                    # A more advanced parser might group connected slope tiles.
                    surface = self._parse_slope_surface(x,y, tile_type)
                    self.surfaces.append(surface)
                    visited[x,y] = True # Mark as visited since it's handled as a slope

        return self.surfaces
    
    def _is_floor_tile(self, x: int, y: int) -> bool:
        """Check if a tile position represents a floor surface"""
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return False
            
        tile = self.tile_map[x, y]
        
        # Check if tile has a walkable top surface
        if tile in [self.TILE_SOLID, self.TILE_HALF_TOP]:
            # Also check if the tile above is empty
            if y > 0 and self.tile_map[x, y-1] == self.TILE_EMPTY:
                 return True
            elif y == 0: # Top edge of map
                 return True

        # Check slopes and curves (simplified: treat as floor if walkable from above)
        if tile in self.TILE_SLOPE_45_TYPES + self.TILE_MILD_SLOPES + self.TILE_STEEP_SLOPES:
            if y > 0 and self.tile_map[x, y-1] == self.TILE_EMPTY:
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
            if not visited[x, y]:
                tiles.append((x, y))
                visited[x, y] = True
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
        
        tile = self.tile_map[x, y]
        
        # Solid tiles are walls
        if tile == self.TILE_SOLID:
            # Check for adjacent empty space to confirm it's an actual wall edge
            # Check left
            if x > 0 and self.tile_map[x-1, y] == self.TILE_EMPTY:
                return True
            # Check right
            if x < self.grid_width - 1 and self.tile_map[x+1, y] == self.TILE_EMPTY:
                return True
        
        # Half tiles can also form walls
        if tile == self.TILE_HALF_LEFT and x < self.grid_width - 1 and self.tile_map[x+1, y] == self.TILE_EMPTY: # Exposed right face
            return True
        if tile == self.TILE_HALF_RIGHT and x > 0 and self.tile_map[x-1, y] == self.TILE_EMPTY: # Exposed left face
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
                if self.tile_map[x,y] != self.TILE_EMPTY and x > 0 and self.tile_map[x-1,y] == self.TILE_EMPTY:
                    is_valid_wall_segment = True
            elif wall_type == SurfaceType.WALL_RIGHT:
                # Tile (x,y) is solid, tile (x+1,y) is empty
                if self.tile_map[x,y] != self.TILE_EMPTY and x < self.grid_width -1 and self.tile_map[x+1,y] == self.TILE_EMPTY:
                    is_valid_wall_segment = True
            
            if is_valid_wall_segment:
                if not visited[x, y]:
                    tiles.append((x, y))
                    visited[x, y] = True
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
