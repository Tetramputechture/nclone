import math

class Rectangle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.min_x = x
        self.min_y = y
        self.max_x = x + width
        self.max_y = y + height

    def intersects(self, other_bounds):
        # Optimized intersection check - inlined for performance
        return not (self.max_x < other_bounds[0] or
                    self.min_x > other_bounds[2] or
                    self.max_y < other_bounds[1] or
                    self.min_y > other_bounds[3])

    def contains_point(self, point_x, point_y):
        return (self.min_x <= point_x < self.max_x and
                self.min_y <= point_y < self.max_y)

    @property
    def center_x(self):
        return self.x + self.width / 2

    @property
    def center_y(self):
        return self.y + self.height / 2

class Quadtree:
    DEFAULT_MAX_OBJECTS = 10
    DEFAULT_MAX_LEVELS = 5

    def __init__(self, boundary_rect, level=0, max_objects=DEFAULT_MAX_OBJECTS, max_levels=DEFAULT_MAX_LEVELS,
                 map_tile_data=None, map_width_tiles=0, map_height_tiles=0,
                 solid_tile_types=None, tile_pixel_size=24):
        self.boundary = boundary_rect # Instance of Rectangle class
        self.level = level
        self.max_objects = max_objects
        self.max_levels = max_levels
        
        self.objects = [] # Objects stored in this quadtree node
        self.nodes = []   # Child quadtree nodes (NW, NE, SW, SE)

        # Map information for solidity check
        self.map_tile_data = map_tile_data
        self.map_width_tiles = map_width_tiles
        self.map_height_tiles = map_height_tiles
        self.solid_tile_types = solid_tile_types if solid_tile_types is not None else set()
        self.tile_pixel_size = tile_pixel_size

    def _is_boundary_entirely_solid(self):
        if not self.map_tile_data or not self.solid_tile_types or self.tile_pixel_size <= 0:
            return False # Cannot perform check or invalid parameters

        # Determine the range of tiles that overlap with this quadtree node's boundary
        # The boundary coordinates are [min_x, max_x) and [min_y, max_y)
        
        start_tile_x = math.floor(self.boundary.min_x / self.tile_pixel_size)
        start_tile_y = math.floor(self.boundary.min_y / self.tile_pixel_size)

        # self.boundary.max_x is exclusive. A tile tx is included if its range [tx*size, (tx+1)*size-1] overlaps.
        # The last tile index to check is floor((boundary_max_coord - 1) / tile_size)
        end_tile_x = math.floor((self.boundary.max_x - 1) / self.tile_pixel_size) if self.boundary.width > 0 else start_tile_x -1
        end_tile_y = math.floor((self.boundary.max_y - 1) / self.tile_pixel_size) if self.boundary.height > 0 else start_tile_y -1
        
        # Clamp to map boundaries (tile indices)
        start_tile_x = max(0, start_tile_x)
        start_tile_y = max(0, start_tile_y)
        end_tile_x = min(self.map_width_tiles - 1, end_tile_x)
        end_tile_y = min(self.map_height_tiles - 1, end_tile_y)

        if start_tile_x > end_tile_x or start_tile_y > end_tile_y:
            # This means the quadtree node does not meaningfully overlap any full tiles,
            # or is outside the map grid after clamping.
            # Consider it not "entirely solid map tiles".
            return False

        for ty_idx in range(start_tile_y, end_tile_y + 1):
            for tx_idx in range(start_tile_x, end_tile_x + 1):
                # This check is redundant if clamping and map_width/height are correct, but safe.
                if not (0 <= tx_idx < self.map_width_tiles and 0 <= ty_idx < self.map_height_tiles):
                    return False # Should be caught by clamping, but defensive.

                tile_map_index = ty_idx * self.map_width_tiles + tx_idx
                
                if not (0 <= tile_map_index < len(self.map_tile_data)):
                    # This implies an issue with map_width_tiles or map_height_tiles mismatch with map_tile_data length
                    return False 
                
                tile_type = self.map_tile_data[tile_map_index]
                if tile_type not in self.solid_tile_types:
                    return False # Found a non-solid tile that this quadtree node overlaps
        
        return True # All tiles within the boundary are solid

    def clear(self):
        self.objects = []
        self.nodes = []

    def _get_quadrant_for_bounds(self, item_bounds):
        # item_bounds is (min_x, min_y, max_x, max_y)
        # Determine which quadrant(s) the item's bounds belong to.
        # Returns an index 0-3, or -1 if it spans multiple quadrants or doesn't fit.
        
        mid_x = self.boundary.center_x
        mid_y = self.boundary.center_y

        # Calculate the center point of the item
        center_x = (item_bounds[0] + item_bounds[2]) / 2
        center_y = (item_bounds[1] + item_bounds[3]) / 2
        
        # Check if the item is too large for any single quadrant
        # If the item's size is greater than a significant portion of the quadrant, keep it in parent
        item_width = item_bounds[2] - item_bounds[0]
        item_height = item_bounds[3] - item_bounds[1]
        quad_width = self.boundary.width / 2
        quad_height = self.boundary.height / 2
        
        # Allow items to be up to 80% of a quadrant's size
        if item_width > quad_width * 0.8 or item_height > quad_height * 0.8:
            return -1  # Too large, keep in parent
            
        # Determine which quadrant the center falls into
        is_in_top = center_y < mid_y
        is_in_left = center_x < mid_x
        
        if is_in_top:
            if is_in_left:
                return 0  # Northwest
            else:
                return 1  # Northeast
        else:
            if is_in_left:
                return 2  # Southwest
            else:
                return 3  # Southeast

    def subdivide(self):
        # Create four sub-quadrants
        x = self.boundary.x
        y = self.boundary.y
        half_width = self.boundary.width / 2
        half_height = self.boundary.height / 2
        next_level = self.level + 1

        # Northwest
        self.nodes.append(Quadtree(Rectangle(x, y, half_width, half_height), 
                                   next_level, self.max_objects, self.max_levels,
                                   self.map_tile_data, self.map_width_tiles, self.map_height_tiles,
                                   self.solid_tile_types, self.tile_pixel_size))
        # Northeast
        self.nodes.append(Quadtree(Rectangle(x + half_width, y, half_width, half_height), 
                                   next_level, self.max_objects, self.max_levels,
                                   self.map_tile_data, self.map_width_tiles, self.map_height_tiles,
                                   self.solid_tile_types, self.tile_pixel_size))
        # Southwest
        self.nodes.append(Quadtree(Rectangle(x, y + half_height, half_width, half_height), 
                                   next_level, self.max_objects, self.max_levels,
                                   self.map_tile_data, self.map_width_tiles, self.map_height_tiles,
                                   self.solid_tile_types, self.tile_pixel_size))
        # Southeast
        self.nodes.append(Quadtree(Rectangle(x + half_width, y + half_height, half_width, half_height), 
                                   next_level, self.max_objects, self.max_levels,
                                   self.map_tile_data, self.map_width_tiles, self.map_height_tiles,
                                   self.solid_tile_types, self.tile_pixel_size))

    def insert(self, item):
        # item must have a get_bounds() method returning (min_x, min_y, max_x, max_y)
        item_bounds = item.get_bounds()

        # If this node has children, try to insert into them
        if self.nodes:
            quadrant_index = self._get_quadrant_for_bounds(item_bounds)
            if quadrant_index != -1:
                self.nodes[quadrant_index].insert(item)
                return
            # If it doesn't fit neatly into a child, store in this parent node
        
        self.objects.append(item)

        # If capacity is exceeded and we haven't reached max level, subdivide
        if len(self.objects) > self.max_objects and self.level < self.max_levels:
            # Optimization: Check if boundary is entirely solid before subdividing
            if self.map_tile_data and self._is_boundary_entirely_solid():
                return # Do not subdivide if the area is entirely composed of solid tiles

            if not self.nodes: # Only subdivide if we haven't already
                self.subdivide()
            
            # Try to move objects from this node to children
            i = 0
            while i < len(self.objects):
                obj_to_move = self.objects[i]
                obj_bounds = obj_to_move.get_bounds()
                quadrant_index = self._get_quadrant_for_bounds(obj_bounds)
                if quadrant_index != -1:
                    self.nodes[quadrant_index].insert(obj_to_move)
                    self.objects.pop(i) # Remove from parent
                else:
                    i += 1 # Keep in parent

    def query(self, query_bounds):
        # query_bounds is (min_x, min_y, max_x, max_y)
        # Optimized query method with better efficiency
        found_objects = []

        # Quick check if query doesn't intersect with this node at all
        if not self.boundary.intersects(query_bounds):
            return found_objects

        # Check objects in the current node - using direct checks to avoid function calls
        for obj in self.objects:
            obj_b = obj.get_bounds()
            if not (query_bounds[2] < obj_b[0] or  # query_max_x < obj_min_x
                    query_bounds[0] > obj_b[2] or  # query_min_x > obj_max_x
                    query_bounds[3] < obj_b[1] or  # query_max_y < obj_min_y
                    query_bounds[1] > obj_b[3]):   # query_min_y > obj_max_y
                found_objects.append(obj)
        
        # If this node has children, only query the ones that intersect with the query bounds
        if self.nodes:
            for node in self.nodes:
                if node.boundary.intersects(query_bounds):
                    found_objects.extend(node.query(query_bounds))
        
        return found_objects  # No need to remove duplicates - they shouldn't occur if insert works correctly

    def query_point(self, point_x, point_y):
        # Returns a list of objects that *could* contain the point
        # Optimized to reduce unnecessary checks
        found_objects = []

        # Quick check if point is not even in this quadrant
        if not self.boundary.contains_point(point_x, point_y):
            return found_objects

        # Check objects in the current node
        for obj in self.objects:
            obj_b = obj.get_bounds()
            if (obj_b[0] <= point_x < obj_b[2] and
                obj_b[1] <= point_y < obj_b[3]):
                 found_objects.append(obj)
        
        # If this node has children, only query the relevant one
        if self.nodes:
            mid_x = self.boundary.center_x
            mid_y = self.boundary.center_y

            # Determine which child quadrant contains this point
            index = 0
            if point_x >= mid_x: 
                index += 1  # NE or SE
            if point_y >= mid_y: 
                index += 2  # SW or SE
            
            # Only check that specific child
            child_results = self.nodes[index].query_point(point_x, point_y)
            if child_results:
                found_objects.extend(child_results)
            
        return found_objects

    def get_all_boundary_rects(self):
        """Recursively collects all boundary rectangles from this node and its children."""
        rects = [self.boundary]
        if self.nodes:
            for node in self.nodes:
                rects.extend(node.get_all_boundary_rects())
        return rects

# Example Usage (for testing, can be removed later)
if __name__ == '__main__':
    class TestItem:
        def __init__(self, id, min_x, min_y, max_x, max_y):
            self.id = id
            self._bounds = (min_x, min_y, max_x, max_y)
        
        def get_bounds(self):
            return self._bounds

        def __repr__(self):
            return f"Item({self.id}, {self._bounds})"

    # Define the overall boundary for the quadtree
    world_boundary = Rectangle(0, 0, 1000, 1000)
    qt = Quadtree(world_boundary, max_objects=5, max_levels=4)

    # Add some items
    items = [
        TestItem(1, 10, 10, 50, 50),      # NW
        TestItem(2, 600, 10, 650, 50),   # NE
        TestItem(3, 10, 600, 50, 650),   # SW
        TestItem(4, 600, 600, 650, 650), # SE
        TestItem(5, 480, 480, 520, 520), # Center, should stay in root or first level parent
        TestItem(6, 100, 100, 150, 150),  # NW
        TestItem(7, 110, 110, 160, 160),  # NW
        TestItem(8, 120, 120, 170, 170),  # NW
        TestItem(9, 130, 130, 180, 180),  # NW
        TestItem(10, 140, 140, 190, 190), # NW - should trigger subdivide for NW
        TestItem(11, 150, 150, 200, 200)  # NW
    ]

    for item in items:
        qt.insert(item)

    print("Quadtree constructed.")

    # Query a region
    query_area = (0, 0, 150, 150) # Should find items 1, 6, 7, 8, 9, 10, 11
    print(f"\nQuerying area: {query_area}")
    found = qt.query(query_area)
    print("Found items:", found)
    
    query_area_center = (400, 400, 600, 600) # Should find item 5
    print(f"\nQuerying area: {query_area_center}")
    found_center = qt.query(query_area_center)
    print("Found items:", found_center)

    query_area_ne = (500, 0, 1000, 500) # Should find item 2
    print(f"\nQuerying area: {query_area_ne}")
    found_ne = qt.query(query_area_ne)
    print("Found items:", found_ne)

    # Query a point
    point_query = (25, 25) # Should find item 1
    print(f"\nQuerying point: {point_query}")
    found_at_point = qt.query_point(point_query[0], point_query[1])
    print("Found items at point:", found_at_point)

    point_query_center = (500, 500) # Should find item 5
    print(f"\nQuerying point: {point_query_center}")
    found_at_point_center = qt.query_point(point_query_center[0], point_query_center[1])
    print("Found items at point:", found_at_point_center)

    point_query_empty = (300, 300) # Should find nothing from the initial small items
    print(f"\nQuerying point: {point_query_empty}")
    found_at_point_empty = qt.query_point(point_query_empty[0], point_query_empty[1])
    print("Found items at point:", found_at_point_empty) 