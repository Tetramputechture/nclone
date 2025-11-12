import math
import array
import struct
import numpy as np

from . import render_utils
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

        # Cache the bounding box for faster spatial checks
        self._bounds_cache = (
            min(self.x1, self.x2),
            min(self.y1, self.y2),
            max(self.x1, self.x2),
            max(self.y1, self.y2),
        )

    def get_closest_point(self, xpos, ypos):
        """Find the closest point on the segment from the given position.
        is_back_facing is false if the position is facing the segment's outter edge.
        """
        dx = xpos - self.x1
        dy = ypos - self.y1
        # seg_lensq is now pre-calculated and accessed via self.seg_lensq
        u = (dx * self.px + dy * self.py) / self.seg_lensq
        u = max(u, 0)
        u = min(u, 1)
        # If u is between 0 and 1, position is closest to the line segment.
        # If u is exactly 0 or 1, position is closest to one of the two edges.
        a = self.x1 + u * self.px
        b = self.y1 + u * self.py
        # Note: can't be backfacing if segment belongs to a door.
        is_back_facing = dy * self.px - dx * self.py < 0 and self.oriented
        return is_back_facing, a, b

    def get_bounds(self):
        """Return the bounding box of the linear segment as (min_x, min_y, max_x, max_y)."""
        return self._bounds_cache
    
    def intersects_bounds(self, min_x, min_y, max_x, max_y):
        """Fast AABB intersection test.
        
        Args:
            min_x, min_y, max_x, max_y: Query bounding box
            
        Returns:
            True if segment bounds overlap query bounds, False otherwise
        """
        seg_min_x, seg_min_y, seg_max_x, seg_max_y = self._bounds_cache
        return not (seg_max_x < min_x or seg_min_x > max_x or 
                    seg_max_y < min_y or seg_min_y > max_y)

    def intersect_with_ray(self, xpos, ypos, dx, dy, radius):
        """Return the time of intersection (as a fraction of a frame) for the collision
        between the segment and a circle moving along a given direction. Return 0 if the circle
        is already intersecting or 1 if it won't intersect within the frame.
        """
        time1 = get_time_of_intersection_circle_vs_circle(
            xpos, ypos, dx, dy, self.x1, self.y1, radius
        )
        time2 = get_time_of_intersection_circle_vs_circle(
            xpos, ypos, dx, dy, self.x2, self.y2, radius
        )
        time3 = get_time_of_intersection_circle_vs_lineseg(
            xpos, ypos, dx, dy, self.x1, self.y1, self.x2, self.y2, radius
        )
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
        self.p_hor = (self.xpos + self.radius * self.hor, self.ypos)
        self.p_ver = (self.xpos, self.ypos + self.radius * self.ver)
        self.active = True
        self.type = "circular"
        self.convex = convex

        # Cache the bounding box for faster spatial checks
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
            # Use math.sqrt directly to avoid circular import issues
            dist = math.sqrt(dist_sq)
            if dist == 0:  # Avoid division by zero if dist is zero
                # Handle this case: maybe point is exactly at the center.
                # For now, let's assume this means we use the edge points or a default.
                # This behavior might need more refinement based on game logic.
                if dx * self.hor > dy * self.ver:
                    a, b = self.p_hor
                else:
                    a, b = self.p_ver
                return is_back_facing, a, b

            a = self.xpos + self.radius * dx / dist
            b = self.ypos + self.radius * dy / dist
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
    
    def intersects_bounds(self, min_x, min_y, max_x, max_y):
        """Fast AABB intersection test.
        
        Args:
            min_x, min_y, max_x, max_y: Query bounding box
            
        Returns:
            True if segment bounds overlap query bounds, False otherwise
        """
        seg_min_x, seg_min_y, seg_max_x, seg_max_y = self._bounds_cache
        return not (seg_max_x < min_x or seg_min_x > max_x or 
                    seg_max_y < min_y or seg_min_y > max_y)

    def intersect_with_ray(self, xpos, ypos, dx, dy, radius):
        """Return the time of intersection (as a fraction of a frame) for the collision
        between the segment and a circle moving along a given direction. Return 0 if the circle
        is already intersecting or 1 if it won't intersect within the frame.
        """
        time1 = get_time_of_intersection_circle_vs_circle(
            xpos, ypos, dx, dy, self.p_hor[0], self.p_hor[1], radius
        )
        time2 = get_time_of_intersection_circle_vs_circle(
            xpos, ypos, dx, dy, self.p_ver[0], self.p_ver[1], radius
        )
        time3 = get_time_of_intersection_circle_vs_arc(
            xpos,
            ypos,
            dx,
            dy,
            self.xpos,
            self.ypos,
            self.hor,
            self.ver,
            self.radius,
            radius,
        )
        return min(time1, time2, time3)


class Entity:
    """Class that all entity types (gold, bounce blocks, thwumps, etc.) inherit from."""

    def __init__(self, entity_type, sim, xcoord, ycoord):
        """Inititate a member from map data"""
        self.type = entity_type
        # Initialize entity_counts for this simulator instance if not already present
        if not hasattr(sim, "entity_counts"):
            sim.entity_counts = [0] * 40
        self.index = sim.entity_counts[self.type]
        sim.entity_counts[self.type] += 1
        self.sim = sim
        self.xpos = xcoord * 6
        self.ypos = ycoord * 6
        self.poslog = array.array("h")
        self.active = True
        self.is_logical_collidable = False
        self.is_physical_collidable = False
        self.is_movable = False
        self.is_thinkable = False
        self.log_positions = False
        self.log_collisions = True
        self.cell = clamp_cell(math.floor(self.xpos / 24), math.floor(self.ypos / 24))
        self.last_exported_state = None
        self.last_exported_frame = None
        self.last_exported_coords = None
        self.exported_chunks = array.array("H")

    def get_state(self, minimal_state: bool = False):
        """Get the entity's state as a list of normalized float values between 0 and 1."""
        # Validate positions before normalization
        if np.isnan(self.xpos) or np.isinf(self.xpos):
            raise ValueError(
                f"Invalid xpos in Entity.get_state(): {self.xpos} "
                f"(type={self.type}, index={self.index})"
            )
        if np.isnan(self.ypos) or np.isinf(self.ypos):
            raise ValueError(
                f"Invalid ypos in Entity.get_state(): {self.ypos} "
                f"(type={self.type}, index={self.index})"
            )

        # Basic attributes that all entities have
        state = [
            float(self.active),  # Already 0 or 1
        ]

        if not minimal_state:
            # SRCWIDTH, clamped to [0,1]
            state.append(max(0.0, min(1.0, self.xpos / render_utils.SRCWIDTH)))
            # SRCHEIGHT, clamped to [0,1]
            state.append(max(0.0, min(1.0, self.ypos / render_utils.SRCHEIGHT)))
            # Add entity type
            # Normalize type by max type (28)
            state.append(max(0.0, min(1.0, float(self.type) / 28.0)))

        return state

    def grid_move(self):
        """As the entity is moving, if its center goes from one grid cell to another,
        remove it from the previous cell and insert it into the new cell.
        """
        cell_new = clamp_cell(math.floor(self.xpos / 24), math.floor(self.ypos / 24))
        if cell_new != self.cell:
            self.sim.grid_entity[self.cell].remove(self)
            self.cell = cell_new
            self.sim.grid_entity[self.cell].append(self)

    def log_collision(self, state=1):
        """Log an interaction with this entity"""
        if (
            self.log_collisions
            and self.sim.sim_config.log_data
            and self.sim.frame > 0
            and state != self.last_exported_state
        ):
            self.sim.collisionlog.append(
                struct.pack("<HBHB", self.sim.frame, self.type, self.index, state)
            )
            self.last_exported_state = state

    def log_position(self):
        """Log position of entity on current frame"""
        # Only export position if enabled and the entity has moved enough
        if not (self.active and self.sim.sim_config.log_data and self.log_positions):
            return
        last = self.last_exported_coords
        dist = abs(last[0] - self.xpos) + abs(last[1] - self.ypos) if last else 0
        if last and dist < self.sim.sim_config.tolerance:
            return

        # Determine if a new chunk needs to be started or the last one extended
        if (
            not self.last_exported_frame
            or self.sim.frame > self.last_exported_frame + 1
        ):
            self.exported_chunks.extend((self.sim.frame, 1))
        else:
            self.exported_chunks[-1] += 1

        # Update logs
        self.poslog.extend((pack_coord(self.xpos), pack_coord(self.ypos)))
        self.last_exported_frame = self.sim.frame
        self.last_exported_coords = (self.xpos, self.ypos)
