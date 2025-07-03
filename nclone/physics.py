import math
from itertools import product
from .constants import NINJA_RADIUS


def clamp(n, a, b):
    """Force a number n into a range (a, b)"""
    return a if n < a else b if n > b else n


def clamp_cell(xcell, ycell):
    """If necessary, adjust coordinates of cell so it is in bounds."""
    return (clamp(xcell, 0, 43), clamp(ycell, 0, 24))


def clamp_half_cell(xcell, ycell):
    """If necessary, adjust coordinates of half cell so it is in bounds."""
    return (clamp(xcell, 0, 87), clamp(ycell, 0, 49))


def pack_coord(coord):
    """Pack a coordinate into a signed short for exporting"""
    lim = (1 << 15) - 1
    return clamp(round(10 * coord), -lim, lim)


# Add caching for frequently used calculations
_cell_cache = {}
_sqrt_cache = {}


def get_cached_sqrt(n):
    """Cache sqrt calculations for common values"""
    if n not in _sqrt_cache:
        _sqrt_cache[n] = math.sqrt(n)
    return _sqrt_cache[n]


def gather_segments_from_region(sim, x1, y1, x2, y2):
    """Return a list containing all collidable segments from the cells in a
    rectangular region bounded by 2 points.
    """
    # Ensure sim has a quadtree initialized
    if not hasattr(sim, 'collision_quadtree') or sim.collision_quadtree is None:
        # Fallback or error, ideally quadtree should always be present
        print("Warning: sim.collision_quadtree not found in gather_segments_from_region")
        return []

    query_bounds_tuple = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    
    # Get segments directly from quadtree - filter by active state in a single pass
    # Avoid multiple iterations by combining the query and filter operations
    candidate_segments = sim.collision_quadtree.query(query_bounds_tuple)
    
    # Using list comprehension is faster than filter() for this case
    return [segment for segment in candidate_segments if segment.active]


def gather_entities_from_neighbourhood(sim, xpos, ypos):
    """Return a list that contains all active entities from the nine neighbour cells."""
    cx, cy = clamp_cell(math.floor(xpos/24), math.floor(ypos/24))

    # Cache key for cell neighborhood
    cache_key = (cx, cy)
    if cache_key in _cell_cache:
        cells = _cell_cache[cache_key]
    else:
        cells = list(product(range(max(cx - 1, 0), min(cx + 1, 43) + 1),
                             range(max(cy - 1, 0), min(cy + 1, 24) + 1)))
        _cell_cache[cache_key] = cells

    entity_list = []
    for cell in cells:
        entity_list.extend([entity for entity in sim.grid_entity[cell]
                            if entity.active])
    return entity_list


def sweep_circle_vs_tiles(sim, xpos_old, ypos_old, dx, dy, radius):
    """Fetch all segments from neighbourhood. Return shortest intersection time from interpolation."""
    xpos_new = xpos_old + dx
    ypos_new = ypos_old + dy
    width = radius + 1
    x1 = min(xpos_old, xpos_new) - width
    y1 = min(ypos_old, ypos_new) - width
    x2 = max(xpos_old, xpos_new) + width
    y2 = max(ypos_old, ypos_new) + width
    segments = gather_segments_from_region(sim, x1, y1, x2, y2)
    shortest_time = 1
    for segment in segments:
        time = segment.intersect_with_ray(xpos_old, ypos_old, dx, dy, radius)
        shortest_time = min(time, shortest_time)
    return shortest_time


def get_single_closest_point(sim, xpos, ypos, radius):
    """Find the closest point belonging to a collidable segment from the given position."""
    segments = gather_segments_from_region(
        sim, xpos-radius, ypos-radius, xpos+radius, ypos+radius)
    shortest_distance = float('inf')
    result = 0
    closest_point = None

    for segment in segments:
        is_back_facing, a, b = segment.get_closest_point(xpos, ypos)
        # Use cached sqrt for distance calculation
        distance_sq = (xpos - a)**2 + (ypos - b)**2

        # This is to prioritize correct side collisions when multiple close segments
        if not is_back_facing:
            distance_sq -= 0.1
        if distance_sq < shortest_distance:
            shortest_distance = distance_sq
            closest_point = (a, b)
            result = -1 if is_back_facing else 1
    return result, closest_point


def penetration_square_vs_point(s_xpos, s_ypos, p_xpos, p_ypos, semi_side):
    """If a point is inside an orthogonal square, return the orientation of the shortest vector
    to depenetate the point out of the square, and return the penetrations on both axis.
    The square is defined by its center and semi side length. In the case of depenetrating the
    ninja out of square entity (bounce block, thwump, shwump), we consider a square of with a
    semi side equal to the semi side of the entity plus the radius of the ninja.
    """
    dx = p_xpos - s_xpos
    dy = p_ypos - s_ypos
    penx = semi_side - abs(dx)
    peny = semi_side - abs(dy)
    if penx > 0 and peny > 0:
        if peny <= penx:
            depen_normal = (0, -1) if dy < 0 else (0, 1)
            depen_values = (peny, penx)
        else:
            depen_normal = (-1, 0) if dx < 0 else (1, 0)
            depen_values = (penx, peny)
        return depen_normal, depen_values


def overlap_circle_vs_circle(xpos1, ypos1, radius1, xpos2, ypos2, radius2):
    """Given two cirles definied by their center and radius, return true if they overlap."""
    dist = math.sqrt((xpos1 - xpos2)**2 + (ypos1 - ypos2)**2)
    return dist < radius1 + radius2


def is_empty_row(sim, xcoord1, xcoord2, ycoord, dir):
    """Return true if the cell has no solid horizontal edge in the specified direction."""
    xcoords = range(xcoord1, xcoord2+1)
    if dir == 1:
        return not any(sim.hor_grid_edge_dic[clamp_half_cell(xcoord, ycoord+1)] for xcoord in xcoords)
    if dir == -1:
        return not any(sim.hor_grid_edge_dic[clamp_half_cell(xcoord, ycoord)] for xcoord in xcoords)


def is_empty_column(sim, xcoord, ycoord1, ycoord2, dir):
    """Return true if the cell has no solid vertical edge in the specified direction."""
    ycoords = range(ycoord1, ycoord2+1)
    if dir == 1:
        return not any(sim.ver_grid_edge_dic[clamp_half_cell(xcoord+1, ycoord)] for ycoord in ycoords)
    if dir == -1:
        return not any(sim.ver_grid_edge_dic[clamp_half_cell(xcoord, ycoord)] for ycoord in ycoords)


def map_vector_to_orientation(xdir, ydir):
    """Returns an orientation value (0-7) from a vector. The vector can be
    arbitrary, as rounding to these 8 directions is performed."""
    angle = math.atan2(ydir, xdir)
    if angle < 0:
        angle += 2 * math.pi
    return round(8 * angle / (2 * math.pi)) % 8


def get_time_of_intersection_circle_vs_circle(xpos, ypos, vx, vy, a, b, radius):
    """Return time of intersection by interpolation by sweeping a circle onto an other circle, 
    given a combined radius.
    """
    dx = xpos - a
    dy = ypos - b
    dist_sq = dx**2 + dy**2
    vel_sq = vx**2 + vy**2
    dot_prod = dx * vx + dy * vy
    if dist_sq - radius**2 > 0:
        radicand = dot_prod**2 - vel_sq * (dist_sq - radius**2)
        if vel_sq > 0.0001 and dot_prod < 0 and radicand >= 0:
            return (-dot_prod - math.sqrt(radicand)) / vel_sq
        return 1
    return 0


def get_time_of_intersection_circle_vs_lineseg(xpos, ypos, dx, dy, a1, b1, a2, b2, radius):
    """Return time of intersection by interpolation by sweeping a circle onto a line segment."""
    wx = a2 - a1
    wy = b2 - b1
    seg_len = math.sqrt(wx**2 + wy**2)
    nx = wx / seg_len
    ny = wy / seg_len
    normal_proj = (xpos - a1) * ny - (ypos - b1) * nx
    hor_proj = (xpos - a1) * nx + (ypos - b1) * ny
    if abs(normal_proj) >= radius:
        dir = dx * ny - dy * nx
        if dir * normal_proj < 0:
            t = min((abs(normal_proj) - radius) / abs(dir), 1)
            hor_proj2 = hor_proj + t * (dx * nx + dy * ny)
            if 0 <= hor_proj2 <= seg_len:
                return t
    else:
        if 0 <= hor_proj <= seg_len:
            return 0
    return 1


def get_time_of_intersection_circle_vs_arc(xpos, ypos, vx, vy, a, b, hor, ver,
                                           radius_arc, radius_circle):
    """Return time of intersection by interpolation by sweeping a circle onto a circle arc.
    This algorithm assumes the radius of the circle is lesser than the radius of the arc.
    """
    dx = xpos - a
    dy = ypos - b
    dist_sq = dx**2 + dy**2
    vel_sq = vx**2 + vy**2
    dot_prod = dx * vx + dy * vy
    radius1 = radius_arc + radius_circle
    radius2 = radius_arc - radius_circle
    t = 1
    if dist_sq > radius1**2:
        radicand = dot_prod**2 - vel_sq * (dist_sq - radius1**2)
        if vel_sq > 0.0001 and dot_prod < 0 and radicand >= 0:
            t = (-dot_prod - math.sqrt(radicand)) / vel_sq
    elif dist_sq < radius2**2:
        radicand = dot_prod**2 - vel_sq * (dist_sq - radius2**2)
        if vel_sq > 0.0001:
            t = min((-dot_prod + math.sqrt(radicand)) / vel_sq, 1)
    else:
        t = 0
    if (dx + t*vx) * hor > 0 and (dy + t*vy) * ver > 0:
        return t
    return 1


def map_orientation_to_vector(orientation):
    """Return a normalized vector pointing in the direction of the orientation.
    Orientation is a value between 0 and 7 taken from map data.
    """
    diag = math.sqrt(2) / 2
    orientation_dic = {0: (1, 0), 1: (diag, diag), 2: (0, 1), 3: (-diag, diag),
                       4: (-1, 0), 5: (-diag, -diag), 6: (0, -1), 7: (diag, -diag)}
    return orientation_dic[orientation]


def get_raycast_distance(sim, xpos, ypos, dx, dy):
    """Return the length of a ray given its start point and direction. The ray stops when it hits a
    tile. Return None if the ray hits nothing after travelling for 2000 units. The algorithm works by
    finding the cells the ray traverses and testing it against the tile segments for each cell.
    """
    xcell = math.floor(xpos/24)
    ycell = math.floor(ypos/24)
    if dx > 0:
        step_x = 1
        delta_x = 24 / dx
        tmax_x = ((xcell + 1)*24 - xpos) / dx
    elif dx < 0:
        step_x = -1
        delta_x = -24 / dx
        tmax_x = (xcell*24 - xpos) / dx
    else:
        step_x = 0
        delta_x = 0
        tmax_x = 999999
    if dy > 0:
        step_y = 1
        delta_y = 24 / dy
        tmax_y = ((ycell + 1)*24 - ypos) / dy
    elif dy < 0:
        step_y = -1
        delta_y = -24 / dy
        tmax_y = (ycell*24 - ypos) / dy
    else:
        step_y = 0
        delta_y = 0
        tmax_y = 999999
    while True:
        result = intersect_ray_vs_cell_contents(
            sim, xcell, ycell, xpos, ypos, 2000*dx, 2000*dy)
        if result < 1:
            return 2000 * result
        if tmax_x < tmax_y:
            xcell += step_x
            if xcell < 0 or xcell >= 44:
                return
            tmax_x += delta_x
        else:
            ycell += step_y
            if ycell < 0 or ycell >= 25:
                return
            tmax_y += delta_y


def intersect_ray_vs_cell_contents(sim, xcell, ycell, xpos, ypos, dx, dy):
    """Given a cell and a ray, return the shortest time of intersection between the ray and one of
    the cell's tile segments. Return 1 if the ray hits nothing.
    """
    segments = sim.segment_dic[clamp_cell(xcell, ycell)]
    shortest_time = 1
    for segment in segments:
        time = segment.intersect_with_ray(xpos, ypos, dx, dy, 0)
        shortest_time = min(time, shortest_time)
    return shortest_time


def raycast_vs_player(sim, xstart, ystart, ninja_xpos, ninja_ypos):
    """Draw a segment that starts at a given position and goes towards the center of the ninja.
    Return true if the segment touches the ninja, meaning there were no tile segments in its path.
    """
    dx = ninja_xpos - xstart
    dy = ninja_ypos - ystart
    dist = math.sqrt(dx**2 + dy**2)
    if NINJA_RADIUS <= dist and dist > 0:
        dx /= dist
        dy /= dist
        length = get_raycast_distance(sim, xstart, ystart, dx, dy)
        return length > dist - NINJA_RADIUS
    return True


def check_lineseg_vs_ninja(x1, y1, x2, y2, ninja):
    # TODO
    dx = x2 - x1
    dy = y2 - y1
    len = math.sqrt(dx**2 + dy**2)
    if len == 0:
        return False
    # This part returns false if the segment does not interscet the ninja's circular hitbox, to speed things up.
    dx /= len
    dy /= len
    proj = (ninja.xpos - x1)*dx + (ninja.ypos - y1)*dy
    x = x1
    y = y1
    if proj > 0:
        x += dx*proj
        y += dy*proj
    if NINJA_RADIUS**2 <= (ninja.xpos - x)**2 + (ninja.ypos - y)**2:
        return False
    # Now test the segment against each of ninja's 11 segments. Return true if it intersects any.
    NINJA_SEGS = ((0, 12), (1, 12), (2, 8), (3, 9), (4, 10),
                  (5, 11), (6, 7), (8, 0), (9, 0), (10, 1), (11, 1))
    for seg in NINJA_SEGS:
        x3 = ninja.xpos + 24*ninja.bones[seg[0]][0]
        y3 = ninja.ypos + 24*ninja.bones[seg[0]][1]
        x4 = ninja.xpos + 24*ninja.bones[seg[1]][0]
        y4 = ninja.ypos + 24*ninja.bones[seg[1]][1]
        det1 = (x1 - x3)*(y2 - y3) - (y1 - y3)*(x2 - x3)
        det2 = (x1 - x4)*(y2 - y4) - (y1 - y4)*(x2 - x4)
        det3 = (x3 - x1)*(y4 - y1) - (y3 - y1)*(x4 - x1)
        det4 = (x3 - x2)*(y4 - y2) - (y3 - y2)*(x4 - x2)
        if det1*det2 < 0 and det3*det4 < 0:
            return True
    return False


def overlap_circle_vs_segment(xpos, ypos, radius, px1, py1, px2, py2):
    """Given a circle defined by its center and radius, and a segment defined by two points,
    return true if they overlap.
    """
    px = px2 - px1
    py = py2 - py1
    dx = xpos - px1
    dy = ypos - py1
    seg_lensq = px**2 + py**2
    u = (dx*px + dy*py)/seg_lensq
    u = max(u, 0)
    u = min(u, 1)
    a = px1 + u*px
    b = py1 + u*py
    return (xpos - a)**2 + (ypos - b)**2 < radius**2


# Clear cache periodically to prevent memory growth
def clear_caches():
    """Clear the quadtree and other caches to prevent memory growth.
    Preserves commonly used values while removing infrequently used ones."""
    
    # Selectively clear the cell cache - keep frequently used values
    # instead of clearing entirely
    if len(_cell_cache) > 500:  # Only clear if cache has grown large
        frequent_keys = set()
        for i in range(15, 35):
            for j in range(10, 20):
                # Keep most common central cells (near player's likely position)
                frequent_keys.add((i, j))
                
        _cell_cache_copy = {k: v for k, v in _cell_cache.items() if k in frequent_keys}
        _cell_cache.clear()
        _cell_cache.update(_cell_cache_copy)
    
    # Keep frequently used sqrt values but clear others if cache has grown too large
    if len(_sqrt_cache) > 100:
        common_values = {0, 1, 2, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 324, 400}
        _sqrt_cache_copy = {k: v for k, v in _sqrt_cache.items() if k in common_values}
        _sqrt_cache.clear()
        _sqrt_cache.update(_sqrt_cache_copy)
