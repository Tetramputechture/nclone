import math
import array
import struct

from .physics import *


class GridSegmentLinear:
    """Contains all the linear segments of tiles and doors that the ninja can interract with"""

    def __init__(self, p1, p2, oriented=True):
        """Initiate an instance of a linear segment of a tile. 
        Each segment is defined by the coordinates of its two end points.
        Tile segments are oreinted which means they have an inner side and an outer side.
        Door segments are not oriented : Collision is the same regardless of the side.
        """
        self.x1, self.y1 = p1
        self.x2, self.y2 = p2
        self.oriented = oriented
        self.active = True
        self.type = "linear"
        # Pre-calculate segment properties
        self.px = self.x2 - self.x1
        self.py = self.y2 - self.y1
        # Avoid division by zero if p1 and p2 are the same
        self.seg_lensq = self.px**2 + self.py**2
        if self.seg_lensq == 0:
            # Handle degenerate segment (p1 and p2 are the same)
            # Default to a very small length to avoid division by zero,
            # or handle as a special case in get_closest_point.
            # For now, using a small epsilon.
            self.seg_lensq = 1e-9
            
        # Cache the bounding box for faster quadtree operations
        self._bounds_cache = (
            min(self.x1, self.x2),
            min(self.y1, self.y2),
            max(self.x1, self.x2),
            max(self.y1, self.y2)
        )

    def get_closest_point(self, xpos, ypos):
        """Find the closest point on the segment from the given position.
        is_back_facing is false if the position is facing the segment's outter edge.
        """
        dx = xpos - self.x1
        dy = ypos - self.y1
        # seg_lensq is now pre-calculated and accessed via self.seg_lensq
        u = (dx*self.px + dy*self.py)/self.seg_lensq
        u = max(u, 0)
        u = min(u, 1)
        # If u is between 0 and 1, position is closest to the line segment.
        # If u is exactly 0 or 1, position is closest to one of the two edges.
        a = self.x1 + u*self.px
        b = self.y1 + u*self.py
        # Note: can't be backfacing if segment belongs to a door.
        is_back_facing = dy*self.px - dx*self.py < 0 and self.oriented
        return is_back_facing, a, b

    def get_bounds(self):
        """Return the bounding box of the linear segment as (min_x, min_y, max_x, max_y)."""
        return self._bounds_cache

    def intersect_with_ray(self, xpos, ypos, dx, dy, radius):
        """Return the time of intersection (as a fraction of a frame) for the collision
        between the segment and a circle moving along a given direction. Return 0 if the circle 
        is already intersecting or 1 if it won't intersect within the frame.
        """
        time1 = get_time_of_intersection_circle_vs_circle(
            xpos, ypos, dx, dy, self.x1, self.y1, radius)
        time2 = get_time_of_intersection_circle_vs_circle(
            xpos, ypos, dx, dy, self.x2, self.y2, radius)
        time3 = get_time_of_intersection_circle_vs_lineseg(xpos, ypos, dx, dy, self.x1, self.y1,
                                                           self.x2, self.y2, radius)
        return min(time1, time2, time3)


class GridSegmentCircular:
    """Contains all the circular segments of tiles that the ninja can interract with"""

    def __init__(self, center, quadrant, convex, radius=24):
        """Initiate an instance of a circular segment of a tile. 
        Each segment is defined by the coordinates of its center, a vector indicating which
        quadrant contains the qurater-circle, a boolean indicating if the tile is convex or
        concave, and the radius of the quarter-circle."""
        self.xpos = center[0]
        self.ypos = center[1]
        self.hor = quadrant[0]
        self.ver = quadrant[1]
        self.radius = radius
        # The following two variables are the position of the two extremities of arc.
        self.p_hor = (self.xpos + self.radius*self.hor, self.ypos)
        self.p_ver = (self.xpos, self.ypos + self.radius*self.ver)
        self.active = True
        self.type = "circular"
        self.convex = convex
        
        # Cache the bounding box for faster quadtree operations
        # The relevant points for the bounding box are the center and the two arc ends
        min_x = min(self.xpos, self.p_hor[0])
        max_x = max(self.xpos, self.p_hor[0])
        min_y = min(self.ypos, self.p_ver[1])
        max_y = max(self.ypos, self.p_ver[1])
        self._bounds_cache = (min_x, min_y, max_x, max_y)

    def get_closest_point(self, xpos, ypos):
        """Find the closest point on the segment from the given position.
        is_back_facing is false if the position is facing the segment's outter edge.
        """
        dx = xpos - self.xpos
        dy = ypos - self.ypos
        is_back_facing = False
        # This is true if position is closer from arc than its edges.
        if dx * self.hor > 0 and dy * self.ver > 0:
            dist_sq = dx**2 + dy**2
            # Use cached sqrt if available, otherwise fall back to math.sqrt or ensure physics.get_cached_sqrt is used
            dist = get_cached_sqrt(dist_sq) # Assuming get_cached_sqrt is available from physics import
            if dist == 0: # Avoid division by zero if dist is zero
                # Handle this case: maybe point is exactly at the center.
                # For now, let's assume this means we use the edge points or a default.
                # This behavior might need more refinement based on game logic.
                if dx * self.hor > dy * self.ver:
                    a, b = self.p_hor
                else:
                    a, b = self.p_ver
                return is_back_facing, a, b

            a = self.xpos + self.radius*dx/dist
            b = self.ypos + self.radius*dy/dist
            is_back_facing = dist < self.radius if self.convex else dist > self.radius
        else:  # If closer to edges of arc, find position of closest point of the two.
            if dx * self.hor > dy * self.ver:
                a, b = self.p_hor
            else:
                a, b = self.p_ver
        return is_back_facing, a, b

    def get_bounds(self):
        """Return the bounding box of the circular segment as (min_x, min_y, max_x, max_y)."""
        return self._bounds_cache

    def intersect_with_ray(self, xpos, ypos, dx, dy, radius):
        """Return the time of intersection (as a fraction of a frame) for the collision
        between the segment and a circle moving along a given direction. Return 0 if the circle 
        is already intersecting or 1 if it won't intersect within the frame.
        """
        time1 = get_time_of_intersection_circle_vs_circle(
            xpos, ypos, dx, dy, self.p_hor[0], self.p_hor[1], radius)
        time2 = get_time_of_intersection_circle_vs_circle(
            xpos, ypos, dx, dy, self.p_ver[0], self.p_ver[1], radius)
        time3 = get_time_of_intersection_circle_vs_arc(xpos, ypos, dx, dy, self.xpos, self.ypos,
                                                       self.hor, self.ver, self.radius, radius)
        return min(time1, time2, time3)


class Entity:
    """Class that all entity types (gold, bounce blocks, thwumps, etc.) inherit from."""

    def __init__(self, entity_type, sim, xcoord, ycoord):
        """Inititate a member from map data"""
        self.type = entity_type
        # Initialize entity_counts for this simulator instance if not already present
        if not hasattr(sim, 'entity_counts'):
            sim.entity_counts = [0] * 40
        self.index = sim.entity_counts[self.type]
        sim.entity_counts[self.type] += 1
        self.sim = sim
        self.xpos = xcoord*6
        self.ypos = ycoord*6
        self.poslog = array.array('h')
        self.active = True
        self.is_logical_collidable = False
        self.is_physical_collidable = False
        self.is_movable = False
        self.is_thinkable = False
        self.log_positions = False
        self.log_collisions = True
        self.cell = clamp_cell(math.floor(self.xpos / 24),
                               math.floor(self.ypos / 24))
        self.last_exported_state = None
        self.last_exported_frame = None
        self.last_exported_coords = None
        self.exported_chunks = array.array('H')

    def get_state(self, minimal_state: bool = False):
        """Get the entity's state as a list of normalized float values between 0 and 1."""
        # Basic attributes that all entities have
        state = [
            float(self.active),  # Already 0 or 1
        ]

        if not minimal_state:
            # SRCWIDTH, clamped to [0,1]
            state.append(max(0.0, min(1.0, self.xpos / 1056)))
            # SRCHEIGHT, clamped to [0,1]
            state.append(max(0.0, min(1.0, self.ypos / 600)))
            # Add entity type
            # Normalize type by max type (28)
            state.append(max(0.0, min(1.0, float(self.type) / 28.0)))

        return state

    def grid_move(self):
        """As the entity is moving, if its center goes from one grid cell to another,
        remove it from the previous cell and insert it into the new cell.
        """
        cell_new = clamp_cell(math.floor(self.xpos / 24),
                              math.floor(self.ypos / 24))
        if cell_new != self.cell:
            self.sim.grid_entity[self.cell].remove(self)
            self.cell = cell_new
            self.sim.grid_entity[self.cell].append(self)

    def log_collision(self, state=1):
        """Log an interaction with this entity"""
        if self.log_collisions and self.sim.sim_config.log_data and self.sim.frame > 0 and state != self.last_exported_state:
            self.sim.collisionlog.append(struct.pack(
                '<HBHB', self.sim.frame, self.type, self.index, state))
            self.last_exported_state = state

    def log_position(self):
        """Log position of entity on current frame"""
        # Only export position if enabled and the entity has moved enough
        if not (self.active and self.sim.sim_config.log_data and self.log_positions):
            return
        last = self.last_exported_coords
        dist = abs(last[0] - self.xpos) + \
            abs(last[1] - self.ypos) if last else 0
        if last and dist < self.sim.sim_config.tolerance:
            return

        # Determine if a new chunk needs to be started or the last one extended
        if not self.last_exported_frame or self.sim.frame > self.last_exported_frame + 1:
            self.exported_chunks.extend((self.sim.frame, 1))
        else:
            self.exported_chunks[-1] += 1

        # Update logs
        self.poslog.extend((pack_coord(self.xpos), pack_coord(self.ypos)))
        self.last_exported_frame = self.sim.frame
        self.last_exported_coords = (self.xpos, self.ypos)


class EntityToggleMine(Entity):
    """This class handles both toggle mines (untoggled state) and regular mines (toggled state)."""
    ENTITY_TYPE = 1  # Also handles type 21 for toggled state
    RADII = {0: 4, 1: 3.5, 2: 4.5}  # 0:toggled, 1:untoggled, 2:toggling
    MAX_COUNT_PER_LEVEL = 8192

    def __init__(self, entity_type, sim, xcoord, ycoord, state):
        super().__init__(entity_type, sim, xcoord, ycoord)
        self.is_thinkable = True
        self.is_logical_collidable = True
        self.set_state(state)

    def think(self):
        """Handle interactions between the ninja and the untoggled mine"""
        ninja = self.sim.ninja
        if ninja.is_valid_target():
            if self.state == 1:  # Set state to toggling if ninja touches untoggled mine
                if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                            ninja.xpos, ninja.ypos, ninja.RADIUS):
                    self.set_state(2)
            elif self.state == 2:  # Set state to toggled if ninja stops touching toggling mine
                if not overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                                ninja.xpos, ninja.ypos, ninja.RADIUS):
                    self.set_state(0)
        else:  # Set state to untoggled if ninja dies while toggling a mine
            if self.state == 2 and ninja.state == 6:
                self.set_state(1)

    def logical_collision(self):
        """Kill the ninja if it touches a toggled mine"""
        ninja = self.sim.ninja
        if ninja.is_valid_target() and self.state == 0:
            if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                        ninja.xpos, ninja.ypos, ninja.RADIUS):
                self.set_state(1)
                ninja.kill(0, 0, 0, 0, 0)

    def set_state(self, state):
        """Set the state of the toggle. 0:toggled, 1:untoggled, 2:toggling."""
        if state in (0, 1, 2):
            self.state = state
            self.RADIUS = self.RADII[state]
            self.log_collision(state)

    def get_state(self):
        state = super().get_state()
        # Normalize state (0, 1, or 2)
        # state.append(max(0.0, min(1.0, float(self.state) / 2)))
        return state


class EntityGold(Entity):
    ENTITY_TYPE = 2
    RADIUS = 6
    MAX_COUNT_PER_LEVEL = 8192

    def __init__(self, type, sim, xcoord, ycoord):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_logical_collidable = True

    def logical_collision(self):
        """The gold is collected if touches by a ninja that is not in winning state."""
        ninja = self.sim.ninja
        if ninja.state != 8:
            if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                        ninja.xpos, ninja.ypos, ninja.RADIUS):
                ninja.gold_collected += 1
                self.active = False
                self.log_collision()


class EntityExit(Entity):
    ENTITY_TYPE = 3
    RADIUS = 12
    MAX_COUNT_PER_LEVEL = 16

    def __init__(self, type, sim, xcoord, ycoord):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_logical_collidable = True
        self.switch_hit = False

    def logical_collision(self):
        """The ninja wins if it touches the exit door. The door is not interactable from the entity
        grid before the exit switch is collected.
        """
        ninja = self.sim.ninja
        if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                    ninja.xpos, ninja.ypos, ninja.RADIUS):
            ninja.win()


class EntityExitSwitch(Entity):
    ENTITY_TYPE = 4
    RADIUS = 6

    def __init__(self, type, sim, xcoord, ycoord, parent):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_logical_collidable = True
        self.parent = parent

    def logical_collision(self):
        """If the ninja is colliding with the switch, open its associated door. This is done in practice
        by adding the parent door entity to the entity grid.
        """
        ninja = self.sim.ninja
        if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                    ninja.xpos, ninja.ypos, ninja.RADIUS):
            self.active = False
            # Add door to the entity grid so the ninja can touch it
            self.sim.grid_entity[self.parent.cell].append(self.parent)
            self.parent.switch_hit = True  # Mark the switch as hit on the parent Exit door
            self.log_collision()


class EntityDoorBase(Entity):
    """Parent class that all door type entities inherit from : regular doors, locked doors, trap doors."""

    def __init__(self, type, sim, xcoord, ycoord, orientation, sw_xcoord, sw_ycoord):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_logical_collidable = True
        self.closed = True
        self.orientation = orientation
        self.sw_xpos = 6 * sw_xcoord
        self.sw_ypos = 6 * sw_ycoord
        self.is_vertical = orientation in (0, 4)
        vec = map_orientation_to_vector(orientation)
        # Find the cell that the door is in for the grid segment.
        door_xcell = math.floor((self.xpos - 12*vec[0]) / 24)
        door_ycell = math.floor((self.ypos - 12*vec[1]) / 24)
        door_cell = clamp_cell(door_xcell, door_ycell)
        # Find the half cell of the door for the grid edges.
        door_half_xcell = 2*(door_cell[0] + 1)
        door_half_ycell = 2*(door_cell[1] + 1)
        # Create the grid segment and grid edges.
        self.grid_edges = []
        if self.is_vertical:
            self.segment = GridSegmentLinear((self.xpos, self.ypos-12), (self.xpos, self.ypos+12),
                                             oriented=False)
            self.grid_edges.append((door_half_xcell, door_half_ycell-2))
            self.grid_edges.append((door_half_xcell, door_half_ycell-1))
            for grid_edge in self.grid_edges:
                sim.ver_grid_edge_dic[grid_edge] += 1
        else:
            self.segment = GridSegmentLinear((self.xpos-12, self.ypos), (self.xpos+12, self.ypos),
                                             oriented=False)
            self.grid_edges.append((door_half_xcell-2, door_half_ycell))
            self.grid_edges.append((door_half_xcell-1, door_half_ycell))
            for grid_edge in self.grid_edges:
                sim.hor_grid_edge_dic[grid_edge] += 1
        sim.segment_dic[door_cell].append(self.segment)
        # Update position and cell so it corresponds to the switch and not the door.
        if hasattr(sim, 'collision_quadtree') and sim.collision_quadtree is not None:
            sim.collision_quadtree.insert(self.segment)
        self.xpos = self.sw_xpos
        self.ypos = self.sw_ypos
        self.cell = clamp_cell(math.floor(self.xpos / 24),
                               math.floor(self.ypos / 24))

    def change_state(self, closed):
        """Change the state of the door from closed to open or from open to closed."""
        self.closed = closed
        self.segment.active = closed
        self.log_collision(0 if closed else 1)
        for grid_edge in self.grid_edges:
            if self.is_vertical:
                self.sim.ver_grid_edge_dic[grid_edge] += 1 if closed else -1
            else:
                self.sim.hor_grid_edge_dic[grid_edge] += 1 if closed else -1

    def get_state(self):
        state = super().get_state()
        # state.append(float(self.closed))  # Already 0 or 1
        # state[7] = float(self.orientation) / 7  # Normalize orientation
        return state


class EntityDoorRegular(EntityDoorBase):
    ENTITY_TYPE = 5
    RADIUS = 10
    MAX_COUNT_PER_LEVEL = 256

    def __init__(self, type, sim, xcoord, ycoord, orientation, sw_xcoord, sw_ycoord):
        super().__init__(type, sim, xcoord, ycoord, orientation, sw_xcoord, sw_ycoord)
        self.is_thinkable = True
        self.open_timer = 0

    def think(self):
        """If the door has been opened for more than 5 frames without being touched by the ninja, 
        close it.
        """
        if not self.closed:
            self.open_timer += 1
            if self.open_timer > 5:
                self.change_state(closed=True)

    def logical_collision(self):
        """If the ninja touches the activation region of the door (circle with a radius of 10 at the
        door's center), open it."""
        ninja = self.sim.ninja
        if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                    ninja.xpos, ninja.ypos, ninja.RADIUS):
            self.change_state(closed=False)
            self.open_timer = 0


class EntityDoorLocked(EntityDoorBase):
    ENTITY_TYPE = 6
    RADIUS = 5
    MAX_COUNT_PER_LEVEL = 256

    def __init__(self, type, sim, xcoord, ycoord, orientation, sw_xcoord, sw_ycoord):
        super().__init__(type, sim, xcoord, ycoord, orientation, sw_xcoord, sw_ycoord)

    def logical_collision(self):
        """If the ninja collects the associated open switch, open the door."""
        ninja = self.sim.ninja
        if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                    ninja.xpos, ninja.ypos, ninja.RADIUS):
            ninja.doors_opened += 1
            self.change_state(closed=False)
            self.active = False


class EntityDoorTrap(EntityDoorBase):
    ENTITY_TYPE = 8
    RADIUS = 5
    MAX_COUNT_PER_LEVEL = 256

    def __init__(self, type, sim, xcoord, ycoord, orientation, sw_xcoord, sw_ycoord):
        super().__init__(type, sim, xcoord, ycoord, orientation, sw_xcoord, sw_ycoord)
        self.change_state(closed=False)

    def logical_collision(self):
        """If the ninja collects the associated close switch, close the door."""
        ninja = self.sim.ninja
        if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                    ninja.xpos, ninja.ypos, ninja.RADIUS):
            self.change_state(closed=True)
            self.active = False


class EntityLaunchPad(Entity):
    ENTITY_TYPE = 10
    RADIUS = 6
    BOOST = 36/7
    MAX_COUNT_PER_LEVEL = 256

    def __init__(self, type, sim, xcoord, ycoord, orientation):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_logical_collidable = True
        self.orientation = orientation
        self.normal_x, self.normal_y = map_orientation_to_vector(orientation)

    def logical_collision(self):
        """If the ninja is colliding with the launch pad (semi circle hitbox), return boost."""
        ninja = self.sim.ninja
        if ninja.is_valid_target():
            if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                        ninja.xpos, ninja.ypos, ninja.RADIUS):
                if ((self.xpos - (ninja.xpos - ninja.RADIUS*self.normal_x))*self.normal_x
                        + (self.ypos - (ninja.ypos - ninja.RADIUS*self.normal_y))*self.normal_y) >= -0.1:
                    yboost_scale = 1
                    if self.normal_y < 0:
                        yboost_scale = 1 - self.normal_y
                    xboost = self.normal_x * self.BOOST
                    yboost = self.normal_y * self.BOOST * yboost_scale
                    return (xboost, yboost)


class EntityOneWayPlatform(Entity):
    ENTITY_TYPE = 11
    SEMI_SIDE = 12
    MAX_COUNT_PER_LEVEL = 512

    def __init__(self, type, sim, xcoord, ycoord, orientation):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_logical_collidable = True
        self.is_physical_collidable = True
        self.orientation = orientation
        self.normal_x, self.normal_y = map_orientation_to_vector(orientation)

    def calculate_depenetration(self, ninja):
        """Return the depenetration vector between the ninja and the one way. Return nothing if no
        penetration.
        """
        dx = ninja.xpos - self.xpos
        dy = ninja.ypos - self.ypos
        lateral_dist = dy * self.normal_x - dx * self.normal_y
        direction = (ninja.yspeed * self.normal_x -
                     ninja.xspeed * self.normal_y) * lateral_dist
        # The platform has a bigger width if the ninja is moving towards its center.
        radius_scalar = 0.91 if direction < 0 else 0.51
        if abs(lateral_dist) < radius_scalar * ninja.RADIUS + self.SEMI_SIDE:
            normal_dist = dx * self.normal_x + dy * self.normal_y
            if 0 < normal_dist <= ninja.RADIUS:
                normal_proj = ninja.xspeed * self.normal_x + ninja.yspeed * self.normal_y
                if normal_proj <= 0:
                    dx_old = ninja.xpos_old - self.xpos
                    dy_old = ninja.ypos_old - self.ypos
                    normal_dist_old = dx_old * self.normal_x + dy_old * self.normal_y
                    if ninja.RADIUS - normal_dist_old <= 1.1:
                        return (self.normal_x, self.normal_y), (ninja.RADIUS - normal_dist, 0)

    def physical_collision(self):
        """Return depenetration between ninja and one way (None if no penetration)."""
        return self.calculate_depenetration(self.sim.ninja)

    def logical_collision(self):
        """Return wall normal if the ninja enters walled state from entity, else return None."""
        collision_result = self.calculate_depenetration(self.sim.ninja)
        if collision_result:
            if abs(self.normal_x) == 1:
                return self.normal_x


class EntityDroneBase(Entity):
    """Parent class that all drone type entities inherit from."""
    RADIUS = 7.5
    GRID_WIDTH = 24
    DIR_TO_VEC = {0: [1, 0], 1: [0, 1], 2: [-1, 0], 3: [0, -1]}
    # Dictionary to choose the next direction from the patrolling mode of the drone.
    # Patrolling modes : {0:follow wall CW, 1:follow wall CCW, 2:wander CW, 3:wander CCW}
    # Directions : {0:keep forward, 1:turn right, 2:go backward, 3:turn left}
    DIR_LIST = {0: [1, 0, 3, 2], 1: [3, 0, 1, 2],
                2: [0, 1, 3, 2], 3: [0, 3, 1, 2]}

    def __init__(self, type, sim, xcoord, ycoord, orientation, mode, speed):
        super().__init__(type, sim, xcoord, ycoord)
        self.log_positions = self.sim.sim_config.full_export
        self.is_movable = True
        self.speed = speed
        self.dir = None
        self.turn(orientation // 2)
        self.orientation = orientation
        self.mode = mode
        self.xtarget, self.ytarget = self.xpos, self.ypos
        self.xpos2, self.ypos2 = self.xpos, self.ypos

    def turn(self, dir):
        """Change the drone's direction and log it."""
        self.dir_old = self.dir or dir
        self.dir = dir
        if self.sim.sim_config.full_export:
            self.log_collision(dir)

    def move(self):
        """Make the drone move along the grid. The drone will try to move towards the center of an
        adjacent cell. When at the center of that cell, it will then choose the next cell to move
        towards.
        """
        xspeed = self.speed * self.DIR_TO_VEC[self.dir][0]
        yspeed = self.speed * self.DIR_TO_VEC[self.dir][1]
        dx = self.xtarget - self.xpos
        dy = self.ytarget - self.ypos
        dist = math.sqrt(dx**2 + dy**2)
        # If the drone has reached or passed the center of the cell, choose the next cell to go to.
        if dist < 0.000001 or (dx * (self.xtarget - (self.xpos + xspeed)) + dy * (self.ytarget - (self.ypos + yspeed))) < 0:
            self.xpos, self.ypos = self.xtarget, self.ytarget
            can_move = self.choose_next_direction_and_goal()
            if can_move:
                disp = self.speed - dist
                self.xpos += disp * self.DIR_TO_VEC[self.dir][0]
                self.ypos += disp * self.DIR_TO_VEC[self.dir][1]
        # Otherwise, make the drone keep moving along its current direction.
        else:
            xspeed = self.speed * self.DIR_TO_VEC[self.dir][0]
            yspeed = self.speed * self.DIR_TO_VEC[self.dir][1]
            self.xpos += xspeed
            self.ypos += yspeed
            self.grid_move()

    def choose_next_direction_and_goal(self):
        """Return true if the drone can move in at least one of four directions.
        The directions are tested in the order according to the drone's preference depending of its mode.
        """
        for i in range(4):
            new_dir = (self.dir + self.DIR_LIST[self.mode][i]) % 4
            valid_dir = self.test_next_direction_and_goal(new_dir)
            if valid_dir:
                self.turn(new_dir)
                return True
        return False

    def test_next_direction_and_goal(self, dir):
        """Return true if the drone can move to the adjacent cell along the given direction.
        This is true if there are no walls impeding the drone's movement.
        If true, set the center of the adjacent cell as the drone's next target."""
        xdir, ydir = self.DIR_TO_VEC[dir]
        xtarget = self.xpos + self.GRID_WIDTH*xdir
        ytarget = self.ypos + self.GRID_WIDTH*ydir
        if not ydir:
            cell_x = math.floor((self.xpos + xdir*self.RADIUS) / 12)
            cell_xtarget = math.floor((xtarget + xdir*self.RADIUS) / 12)
            cell_y1 = math.floor((self.ypos - self.RADIUS) / 12)
            cell_y2 = math.floor((self.ypos + self.RADIUS) / 12)
            while cell_x != cell_xtarget:
                if not is_empty_column(self.sim, cell_x, cell_y1, cell_y2, xdir):
                    return False
                cell_x += xdir
        else:
            cell_y = math.floor((self.ypos + ydir*self.RADIUS) / 12)
            cell_ytarget = math.floor((ytarget + ydir*self.RADIUS) / 12)
            cell_x1 = math.floor((self.xpos - self.RADIUS) / 12)
            cell_x2 = math.floor((self.xpos + self.RADIUS) / 12)
            while cell_y != cell_ytarget:
                if not is_empty_row(self.sim, cell_x1, cell_x2, cell_y, ydir):
                    return False
                cell_y += ydir
        self.xtarget, self.ytarget = xtarget, ytarget
        return True

    def get_state(self):
        state = super().get_state()
        # Normalize mode (0-3)
        # state.append(max(0.0, min(1.0, float(self.mode) / 3)))
        # state[6] = max(0.0, min(1.0, (float(self.dir) + 1) /
        #                2 if self.dir is not None else 0.5))  # Normalize direction
        # Normalize orientation
        # state[7] = max(0.0, min(1.0, float(self.orientation) / 7))
        return state


class EntityDroneZap(EntityDroneBase):
    ENTITY_TYPE = 14
    MAX_COUNT_PER_LEVEL = 256

    def __init__(self, type, sim, xcoord, ycoord, orientation, mode):
        super().__init__(type, sim, xcoord, ycoord, orientation, mode, 8/7)
        self.is_logical_collidable = True

    def logical_collision(self):
        """Kill the ninja if it touches the regular drone."""
        ninja = self.sim.ninja
        if ninja.is_valid_target():
            if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                        ninja.xpos, ninja.ypos, ninja.RADIUS):
                ninja.kill(0, 0, 0, 0, 0)


class EntityDroneChaser(EntityDroneZap):
    MAX_COUNT_PER_LEVEL = 256

    def __init__(self, type, sim, xcoord, ycoord, orientation, mode):
        super().__init__(type, sim, xcoord, ycoord, orientation, mode)
        self.is_thinkable = True
        self.speed_slow = self.speed
        self.speed_chase = 2 * self.speed
        self.chasing = False

    def think(self):
        # TODO
        if not self.chasing:
            ninja = self.sim.ninja
            if ninja.is_valid_target():
                for i in range(-1, 2):
                    dir = (self.dir + i) % 4
                    xdir, ydir = self.DIR_TO_VEC[dir]
                    if xdir*(ninja.xpos - self.xpos) + ydir*(ninja.ypos - self.ypos) > 0:
                        if abs(ydir*(ninja.xpos - self.xpos) - xdir*(ninja.ypos - self.ypos)) <= 12:
                            pass

    def choose_next_direction_and_goal(self):
        # TODO
        super().choose_next_direction_and_goal()


class EntityBounceBlock(Entity):
    ENTITY_TYPE = 17
    SEMI_SIDE = 9
    STIFFNESS = 0.02222222222222222
    DAMPENING = 0.98
    STRENGTH = 0.2
    MAX_COUNT_PER_LEVEL = 512

    def __init__(self, type, sim, xcoord, ycoord):
        super().__init__(type, sim, xcoord, ycoord)
        self.log_positions = self.sim.sim_config.full_export
        self.is_physical_collidable = True
        self.is_logical_collidable = True
        self.is_movable = True
        self.xspeed, self.yspeed = 0, 0
        self.xorigin, self.yorigin = self.xpos, self.ypos

    def move(self):
        """Update the position and speed of the bounce block by applying the spring force and dampening."""
        self.xspeed *= self.DAMPENING
        self.yspeed *= self.DAMPENING
        self.xpos += self.xspeed
        self.ypos += self.yspeed
        xforce = self.STIFFNESS * (self.xorigin - self.xpos)
        yforce = self.STIFFNESS * (self.yorigin - self.ypos)
        self.xpos += xforce
        self.ypos += yforce
        self.xspeed += xforce
        self.yspeed += yforce
        self.grid_move()

    def physical_collision(self):
        """Apply 80% of the depenetration to the bounce block and 20% to the ninja."""
        ninja = self.sim.ninja
        depen = penetration_square_vs_point(self.xpos, self.ypos, ninja.xpos, ninja.ypos,
                                            self.SEMI_SIDE + ninja.RADIUS)
        if depen:
            depen_x, depen_y = depen[0]
            depen_len = depen[1][0]
            self.xpos -= depen_x * depen_len * (1-self.STRENGTH)
            self.ypos -= depen_y * depen_len * (1-self.STRENGTH)
            self.xspeed -= depen_x * depen_len * (1-self.STRENGTH)
            self.yspeed -= depen_y * depen_len * (1-self.STRENGTH)
            return (depen_x, depen_y), (depen_len * self.STRENGTH, depen[1][1])

    def logical_collision(self):
        """Check if the ninja can interact with the wall of the bounce block"""
        ninja = self.sim.ninja
        depen = penetration_square_vs_point(self.xpos, self.ypos, ninja.xpos, ninja.ypos,
                                            self.SEMI_SIDE + ninja.RADIUS + 0.1)
        if depen:
            return depen[0][0]


class EntityThwump(Entity):
    ENTITY_TYPE = 20
    SEMI_SIDE = 9
    FORWARD_SPEED = 20/7
    BACKWARD_SPEED = 8/7
    MAX_COUNT_PER_LEVEL = 128

    def __init__(self, type, sim, xcoord, ycoord, orientation):
        super().__init__(type, sim, xcoord, ycoord)
        self.log_positions = self.sim.sim_config.full_export
        self.is_movable = True
        self.is_thinkable = True
        self.is_logical_collidable = True
        self.is_physical_collidable = True
        self.orientation = orientation
        self.is_horizontal = orientation in (0, 4)
        self.direction = 1 if orientation in (0, 2) else -1
        self.xorigin, self.yorigin = self.xpos, self.ypos
        self.set_state(0)  # 0:immobile, 1:forward, -1:backward

    def set_state(self, state):
        """Set the thwump's state and log it. 0:immobile, 1:forward, -1:backward"""
        self.state = state
        if self.sim.sim_config.full_export:
            self.log_collision(state % 3)  # The logged value goes from 0 to 2

    def move(self):
        """Update the position of the thwump only if it is already moving. If the thwump retracts past
        its origin, it will stop moving.
        """
        if self.state:  # If not immobile.
            speed = self.FORWARD_SPEED if self.state == 1 else self.BACKWARD_SPEED
            speed_dir = self.direction * self.state
            if not self.is_horizontal:
                ypos_new = self.ypos + speed * speed_dir
                # If the thwump as retreated past its starting point, set its position to the origin.
                if self.state == -1 and (ypos_new - self.yorigin) * (self.ypos - self.yorigin) < 0:
                    self.ypos = self.yorigin
                    self.set_state(0)
                    return
                cell_y = math.floor((self.ypos + speed_dir * 11) / 12)
                cell_y_new = math.floor((ypos_new + speed_dir * 11) / 12)
                if cell_y != cell_y_new:
                    cell_x1 = math.floor((self.xpos - 11) / 12)
                    cell_x2 = math.floor((self.xpos + 11) / 12)
                    if not is_empty_row(self.sim, cell_x1, cell_x2, cell_y, speed_dir):
                        self.set_state(-1)
                        return
                self.ypos = ypos_new
            else:
                xpos_new = self.xpos + speed * speed_dir
                # If the thwump as retreated past its starting point, set its position to the origin.
                if self.state == -1 and (xpos_new - self.xorigin) * (self.xpos - self.xorigin) < 0:
                    self.xpos = self.xorigin
                    self.set_state(0)
                    return
                cell_x = math.floor((self.xpos + speed_dir * 11) / 12)
                cell_x_new = math.floor((xpos_new + speed_dir * 11) / 12)
                if cell_x != cell_x_new:
                    cell_y1 = math.floor((self.ypos - 11) / 12)
                    cell_y2 = math.floor((self.ypos + 11) / 12)
                    if not is_empty_column(self.sim, cell_x, cell_y1, cell_y2, speed_dir):
                        self.set_state(-1)
                        return
                self.xpos = xpos_new
            self.grid_move()

    def think(self):
        """Make the thwump charge if it has sight of the ninja."""
        ninja = self.sim.ninja
        if not self.state and ninja.is_valid_target():
            activation_range = 2 * (self.SEMI_SIDE + ninja.RADIUS)
            if not self.is_horizontal:
                # If the ninja is in the activation range
                if abs(self.xpos - ninja.xpos) < activation_range:
                    ninja_ycell = math.floor(ninja.ypos / 12)
                    thwump_ycell = math.floor(
                        (self.ypos - self.direction * 11) / 12)
                    thwump_xcell1 = math.floor((self.xpos - 11) / 12)
                    thwump_xcell2 = math.floor((self.xpos + 11) / 12)
                    dy = ninja_ycell - thwump_ycell
                    if dy * self.direction >= 0:
                        for i in range(100):
                            if not is_empty_row(self.sim, thwump_xcell1, thwump_xcell2, thwump_ycell, self.direction):
                                dy = ninja_ycell - thwump_ycell
                                break
                            thwump_ycell += self.direction
                        if i > 0 and dy * self.direction <= 0:
                            self.set_state(1)
            else:
                # If the ninja is in the activation range
                if abs(self.ypos - ninja.ypos) < activation_range:
                    ninja_xcell = math.floor(ninja.xpos / 12)
                    thwump_xcell = math.floor(
                        (self.xpos - self.direction * 11) / 12)
                    thwump_ycell1 = math.floor((self.ypos - 11) / 12)
                    thwump_ycell2 = math.floor((self.ypos + 11) / 12)
                    dx = ninja_xcell - thwump_xcell
                    if dx * self.direction >= 0:
                        for i in range(100):
                            if not is_empty_column(self.sim, thwump_xcell, thwump_ycell1, thwump_ycell2, self.direction):
                                dx = ninja_xcell - thwump_xcell
                                break
                            thwump_xcell += self.direction
                        if i > 0 and dx * self.direction <= 0:
                            self.set_state(1)

    def physical_collision(self):
        """Return the depenetration vector for the ninja if it collides with the thwump."""
        ninja = self.sim.ninja
        return penetration_square_vs_point(self.xpos, self.ypos, ninja.xpos, ninja.ypos,
                                           self.SEMI_SIDE + ninja.RADIUS)

    def logical_collision(self):
        """Return the wall normal if the ninja can interact with a thwump's side.
        Kill the ninja if it touches the lethal region on the thwump's charging face.
        """
        ninja = self.sim.ninja
        if ninja.is_valid_target():
            depen = penetration_square_vs_point(self.xpos, self.ypos, ninja.xpos, ninja.ypos,
                                                self.SEMI_SIDE + ninja.RADIUS + 0.1)
            if depen:
                if self.is_horizontal:
                    dx = (self.SEMI_SIDE + 2) * self.direction
                    dy = self.SEMI_SIDE - 2
                    px1, py1 = self.xpos + dx, self.ypos - dy
                    px2, py2 = self.xpos + dx, self.ypos + dy
                else:
                    dx = self.SEMI_SIDE - 2
                    dy = (self.SEMI_SIDE + 2) * self.direction
                    px1, py1 = self.xpos - dx, self.ypos + dy
                    px2, py2 = self.xpos + dx, self.ypos + dy
                if overlap_circle_vs_segment(ninja.xpos, ninja.ypos, ninja.RADIUS + 2, px1, py1, px2, py2):
                    ninja.kill(0, 0, 0, 0, 0)
                return depen[0][0]


class EntityLaser(Entity):
    RADIUS = 5.9
    SPIN_SPEED = 0.010471975  # roughly 2pi/600
    SURFACE_FLAT_SPEED = 0.1
    SURFACE_CORNER_SPEED = 0.005524805665672641  # roughly 0.1/(5.9*pi)

    def __init__(self, type, sim, xcoord, ycoord, orientation, mode):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_thinkable = True
        # Find out what is the laser mode : spinner or surface. Surface mode if segment close enough.
        result, closest_point = get_single_closest_point(
            self.sim, self.xpos, self.ypos, 12)
        if result == -1:
            self.mode = 1
        else:
            if closest_point is None:
                self.mode = 1
            else:
                dist = math.sqrt(
                    (closest_point[0] - self.xpos)**2 + (closest_point[1] - self.ypos)**2)
                self.mode = 1 if dist < 7 else 0
        if self.mode == 0:  # Spinner mode
            self.xend, self.yend = self.xpos, self.ypos
            dx, dy = map_orientation_to_vector(orientation)
            self.angle = math.atan2(dy, dx)
            self.dir = -1 if mode == 1 else 1
        elif self.mode == 1:  # Surface mode
            self.xvec, self.yvec = 0, 0
            self.angle = 0
            self.dir = -1 if mode == 1 else 1
            self.sx, self.sy = 0, 0

    def think(self):
        # TODO
        if self.mode == 0:
            self.think_spinner()

    def think_spinner(self):
        # TODO
        angle_new = (self.angle + self.SPIN_SPEED*self.dir) % (2*math.pi)
        dx = math.cos(self.angle)
        dy = math.sin(self.angle)
        self.len = get_raycast_distance(self.sim, self.xpos, self.ypos, dx, dy)
        if self.len:
            self.xend = self.xpos + dx*self.len
            self.yend = self.ypos + dy*self.len
        else:
            self.xend = self.xpos
            self.yend = self.ypos
        ninja = self.sim.ninja
        if ninja.is_valid_target():
            if raycast_vs_player(self.sim, self.xpos, self.ypos, ninja.xpos, ninja.ypos, ninja.RADIUS):
                ninja_angle = math.atan2(
                    ninja.ypos-self.ypos, ninja.xpos-self.xpos)
                angle_diff = abs(self.angle - ninja_angle) % (2*math.pi)
                if angle_diff <= 0.0052359875:
                    ninja.kill(0, 0, 0, 0, 0)
                else:
                    if check_lineseg_vs_ninja(self.xpos, self.ypos, self.xend, self.yend, ninja):
                        ninja.kill(0, 0, 0, 0, 0)
        self.angle = angle_new

    def think_surface(self):
        segments = gather_segments_from_region(self.sim, self.xpos-12, self.ypos-12,
                                               self.xpos+12, self.ypos+12)
        if not segments:
            return
        while True:
            xspeed = -self.dir*self.yvec*self.SURFACE_FLAT_SPEED
            yspeed = self.dir*self.xvec*self.SURFACE_FLAT_SPEED
            xpos_new = self.xpos + xspeed
            ypos_new = self.ypos + yspeed
            shortest_distance = 9999999
            result = 0
            closest_point = (0, 0)
            for segment in segments:
                is_back_facing, a, b = segment.get_closest_point(
                    xpos_new, ypos_new)
                distance_sq = (xpos_new - a)**2 + (ypos_new - b)**2
                if distance_sq < shortest_distance:
                    shortest_distance = distance_sq
                    closest_point = (a, b)
                    result = -1 if is_back_facing else 1
            dx = xpos_new - closest_point[0]
            dy = ypos_new - closest_point[1]
            if ((self.xpos - self.sx)*dx + (self.ypos - self.sy)*dy) > 0.01 and segment.oriented:
                dist = math.sqrt(
                    (closest_point[0] - self.sx)**2 + (closest_point[1] - self.sy)**2)
                if dist >= 0.0000001:
                    pass
                else:
                    angle = math.atan2(self.yvec, self.xvec)
                    angle += self.dir*self.SURFACE_CORNER_SPEED
                    self.xvec = math.cos(angle)
                    self.yvec = math.sin(angle)

    def get_state(self):
        state = super().get_state()
        # # Normalize angle
        # state[5] = max(0.0, min(1.0, self.angle / (2 * math.pi)))
        # # Normalize direction (-1 or 1)
        # state[6] = max(0.0, min(1.0, (float(self.dir) + 1) / 2))
        return state


class EntityBoostPad(Entity):
    ENTITY_TYPE = 24
    RADIUS = 6
    MAX_COUNT_PER_LEVEL = 128

    def __init__(self, type, sim, xcoord, ycoord):
        super().__init__(type, sim, xcoord, ycoord)
        self.is_movable = True
        self.is_touching_ninja = False

    def move(self):
        """If the ninja starts touching the booster, add 2 to its velocity norm."""
        ninja = self.sim.ninja
        if not ninja.is_valid_target():
            self.is_touching_ninja = False
            return
        if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                    ninja.xpos, ninja.ypos, ninja.RADIUS):
            if not self.is_touching_ninja:
                vel_norm = math.sqrt(ninja.xspeed**2 + ninja.yspeed**2)
                if vel_norm > 0:
                    x_boost = 2 * ninja.xspeed/vel_norm
                    y_boost = 2 * ninja.yspeed/vel_norm
                    ninja.xspeed += x_boost
                    ninja.yspeed += y_boost
                self.is_touching_ninja = True
        else:
            self.is_touching_ninja = False


class EntityDeathBall(Entity):
    ENTITY_TYPE = 25
    RADIUS = 5  # radius for collisions against ninjas
    RADIUS2 = 8  # radius for collisions against other balls and tiles
    ACCELERATION = 0.04
    MAX_SPEED = 0.85
    DRAG_MAX_SPEED = 0.9
    DRAG_NO_TARGET = 0.95
    MAX_COUNT_PER_LEVEL = 64

    def __init__(self, type, sim, xcoord, ycoord):
        super().__init__(type, sim, xcoord, ycoord)
        self.log_positions = self.sim.sim_config.full_export
        self.is_thinkable = True
        self.is_logical_collidable = True
        self.xspeed, self.yspeed = 0, 0

    def think(self):
        """Make the ball move towards the closest ninja. Handle collision with tiles and bounces
        against other balls and ninjas."""
        ninja = self.sim.ninja
        if not ninja.is_valid_target():  # If no valid targets, decelerate ball to a stop
            self.xspeed *= self.DRAG_NO_TARGET
            self.yspeed *= self.DRAG_NO_TARGET
        else:  # Otherwise, apply acceleration towards closest ninja. Apply drag if speed exceeds 0.85.
            dx = ninja.xpos - self.xpos
            dy = ninja.ypos - self.ypos
            dist = math.sqrt(dx**2 + dy**2)
            if dist > 0:
                dx /= dist
                dy /= dist
            self.xspeed += dx * self.ACCELERATION
            self.yspeed += dy * self.ACCELERATION
            speed = math.sqrt(self.xspeed**2 + self.yspeed**2)
            if speed > self.MAX_SPEED:
                new_speed = (speed - self.MAX_SPEED)*self.DRAG_MAX_SPEED
                if new_speed <= 0.01:  # If speed exceed the cap by a tiny amount, remove the excedent
                    new_speed = 0
                new_speed += self.MAX_SPEED
                self.xspeed = self.xspeed / speed * new_speed
                self.yspeed = self.yspeed / speed * new_speed
        xpos_old = self.xpos
        ypos_old = self.ypos
        self.xpos += self.xspeed
        self.ypos += self.yspeed

        # Interpolation routine for high-speed wall collisions.
        time = sweep_circle_vs_tiles(
            self.sim, xpos_old, ypos_old, self.xspeed, self.yspeed, self.RADIUS2 * 0.5)
        self.xpos = xpos_old + time*self.xspeed
        self.ypos = ypos_old + time*self.yspeed

        # Depenetration routine for collision against tiles.
        xnormal, ynormal = 0, 0
        for _ in range(16):
            result, closest_point = get_single_closest_point(
                self.sim, self.xpos, self.ypos, self.RADIUS2)
            if result == 0:
                break
            a, b = closest_point
            dx = self.xpos - a
            dy = self.ypos - b
            dist = math.sqrt(dx**2 + dy**2)
            depen_len = self.RADIUS2 - dist*result
            if depen_len < 0.0000001:
                break
            if dist == 0:
                return
            xnorm = dx / dist
            ynorm = dy / dist
            self.xpos += xnorm * depen_len
            self.ypos += ynorm * depen_len
            xnormal += xnorm
            ynormal += ynorm

        # If there has been tile colision, project speed of deathball onto surface and add bounce if applicable.
        normal_len = math.sqrt(xnormal**2 + ynormal**2)
        if normal_len > 0:
            dx = xnormal / normal_len
            dy = ynormal / normal_len
            dot_product = self.xspeed*dx + self.yspeed*dy
            if dot_product < 0:  # Project velocity onto surface only if moving towards surface
                speed = math.sqrt(self.xspeed**2 + self.yspeed**2)
                bounce_strength = 1 if speed <= 1.35 else 2
                self.xspeed -= dx * dot_product * bounce_strength
                self.yspeed -= dy * dot_product * bounce_strength

        # Handle bounces with other deathballs
        db_count = self.sim.map_data[1200]
        if self.index + 1 < db_count:
            db_targets = self.sim.entity_dic[self.type][self.index+1:]
            for db_target in db_targets:
                dx = self.xpos - db_target.xpos
                dy = self.ypos - db_target.ypos
                dist = math.sqrt(dx**2 + dy**2)
                if dist < 16:
                    dx = dx / dist * 4
                    dy = dy / dist * 4
                    self.xspeed += dx
                    self.yspeed += dy
                    db_target.xspeed -= dx
                    db_target.yspeed -= dy
        self.grid_move()

    def logical_collision(self):
        """If the ninja touches the ball, kill it and make the ball bounce from it."""
        ninja = self.sim.ninja
        if ninja.is_valid_target():
            if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                        ninja.xpos, ninja.ypos, ninja.RADIUS):
                dx = self.xpos - ninja.xpos
                dy = self.ypos - ninja.ypos
                dist = math.sqrt(dx**2 + dy**2)
                self.xspeed += dx / dist * 10
                self.yspeed += dy / dist * 10
                ninja.kill(0, 0, 0, 0, 0)


class EntityMiniDrone(EntityDroneBase):
    ENTITY_TYPE = 26
    MAX_COUNT_PER_LEVEL = 512

    def __init__(self, type, sim, xcoord, ycoord, orientation, mode):
        super().__init__(type, sim, xcoord, ycoord, orientation, mode, 1.3)
        self.is_logical_collidable = True
        self.RADIUS = 4
        self.GRID_WIDTH = 12

    def logical_collision(self):
        """Kill the ninja if it touches the mini drone."""
        ninja = self.sim.ninja
        if ninja.is_valid_target():
            if overlap_circle_vs_circle(self.xpos, self.ypos, self.RADIUS,
                                        ninja.xpos, ninja.ypos, ninja.RADIUS):
                ninja.kill(0, 0, 0, 0, 0)


class EntityShoveThwump(Entity):
    ENTITY_TYPE = 28
    SEMI_SIDE = 12
    RADIUS = 8  # for the projectile inside
    MAX_COUNT_PER_LEVEL = 128

    def __init__(self, type, sim, xcoord, ycoord):
        super().__init__(type, sim, xcoord, ycoord)
        self.log_positions = self.sim.sim_config.full_export
        self.is_thinkable = True
        self.is_logical_collidable = True
        self.is_physical_collidable = True
        self.xorigin, self.yorigin = self.xpos, self.ypos
        self.xdir, self.ydir = 0, 0
        self.set_state(0)  # 0:immobile, 1:activated, 2:launching, 3:retreating
        self.activated = False

    def set_state(self, state):
        """Changes the state of the shwump. 0:immobile, 1:activated, 2:launching, 3:retreating
        Also logs it, combined with the direction information into a single integer."""
        self.state = state
        if self.sim.sim_config.full_export:
            dir = map_vector_to_orientation(self.xdir, self.ydir)
            self.log_collision(4 * state + dir // 2)

    def think(self):
        """Update the state of the shwump and move it if possible."""
        if self.state == 1:
            if self.activated:
                self.activated = False
                return
            self.set_state(2)
        if self.state == 3:
            origin_dist = abs(self.xpos - self.xorigin) + \
                abs(self.ypos - self.yorigin)
            if origin_dist >= 1:
                self.move_if_possible(self.xdir, self.ydir, 1)
            else:
                self.xpos = self.xorigin
                self.ypos = self.yorigin
                self.set_state(0)
        elif self.state == 2:
            self.move_if_possible(-self.xdir, -self.ydir, 4)

    def move_if_possible(self, xdir, ydir, speed):
        """Move the shwump depending of state and orientation.
        Not called in Simulator.tick like other entity move functions.
        """
        if self.ydir == 0:
            xpos_new = self.xpos + xdir * speed
            cell_x = math.floor(self.xpos / 12)
            cell_x_new = math.floor(xpos_new / 12)
            if cell_x != cell_x_new:
                cell_y1 = math.floor((self.ypos - 8) / 12)
                cell_y2 = math.floor((self.ypos + 8) / 12)
                if not is_empty_column(self.sim, cell_x, cell_y1, cell_y2, xdir):
                    self.set_state(3)
                    return
            self.xpos = xpos_new
        else:
            ypos_new = self.ypos + ydir * speed
            cell_y = math.floor(self.ypos / 12)
            cell_y_new = math.floor(ypos_new / 12)
            if cell_y != cell_y_new:
                cell_x1 = math.floor((self.xpos - 8) / 12)
                cell_x2 = math.floor((self.xpos + 8) / 12)
                if not is_empty_row(self.sim, cell_x1, cell_x2, cell_y, ydir):
                    self.set_state(3)
                    return
            self.ypos = ypos_new
        self.grid_move()

    def physical_collision(self):
        """Return the depenetration vector for the ninja if it collides with the shwump.
        Note that if the shwump is in activated state, only one of its sides is collidable.
        """
        ninja = self.sim.ninja
        if self.state <= 1:
            depen = penetration_square_vs_point(self.xpos, self.ypos, ninja.xpos, ninja.ypos,
                                                self.SEMI_SIDE + ninja.RADIUS)
            if depen:
                depen_x, depen_y = depen[0]
                if self.state == 0 or self.xdir * depen_x + self.ydir * depen_y >= 0.01:
                    return depen

    def logical_collision(self):
        """Return the wall normal if the ninja interacts with an active vertical side.
        Kill the ninja if it touches the lethal core of the shwump.
        """
        ninja = self.sim.ninja
        depen = penetration_square_vs_point(self.xpos, self.ypos, ninja.xpos, ninja.ypos,
                                            self.SEMI_SIDE + ninja.RADIUS + 0.1)
        if depen and self.state <= 1:
            depen_x, depen_y = depen[0]
            if self.state == 0:
                self.activated = True
                if depen[1][1] > 0.2:
                    self.xdir = depen_x
                    self.ydir = depen_y
                    self.set_state(1)
            elif self.state == 1:
                if self.xdir * depen_x + self.ydir * depen_y >= 0.01:
                    self.activated = True
                else:
                    return
            return depen_x
        if overlap_circle_vs_circle(ninja.xpos, ninja.ypos, ninja.RADIUS,
                                    self.xpos, self.ypos, self.RADIUS):
            ninja.kill(0, 0, 0, 0, 0)
