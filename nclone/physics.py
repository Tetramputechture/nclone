import math
import os
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
    """Return all active collidable segments from cells overlapped by a rectangle.

    Iterates over cells covered by the query rectangle and returns all active
    segments registered in those cells via ``sim.segment_dic``. No additional
    bounding-box filtering is applied -- callers perform exact collision tests.

    **Order is load-bearing.** ``get_single_closest_point`` keeps the first
    segment of an exact tie (strict ``<``), so at a corner -- where two segments
    report the same closest point -- the gather order alone decides which normal
    the ninja is ejected along. The game's gather (``FUN_1050DDF0``,
    `docs/decomp/gs_sweep_DDF0.c`) runs **y outer, x inner**, and within a cell
    emits the static tile segments before the door segments. This loop is that
    order; ``itertools.product`` gave the transpose.
    """
    cx1, cy1 = clamp_cell(math.floor(x1 / 24), math.floor(y1 / 24))
    cx2, cy2 = clamp_cell(math.floor(x2 / 24), math.floor(y2 / 24))
    segment_dic = sim.segment_dic
    segment_list = []
    for ycell in range(cy1, cy2 + 1):
        for xcell in range(cx1, cx2 + 1):
            segment_list += [segment for segment in segment_dic[(xcell, ycell)]
                             if segment.active]
    return segment_list


def gather_entities_from_neighbourhood(sim, xpos, ypos):
    """Return a list that contains all active entities from the nine neighbour cells."""
    cx, cy = clamp_cell(math.floor(xpos / 24), math.floor(ypos / 24))

    # Cache key for cell neighborhood
    cache_key = (cx, cy)
    if cache_key in _cell_cache:
        cells = _cell_cache[cache_key]
    else:
        cells = list(
            product(
                range(max(cx - 1, 0), min(cx + 1, 43) + 1),
                range(max(cy - 1, 0), min(cy + 1, 24) + 1),
            )
        )
        _cell_cache[cache_key] = cells

    entity_list = []
    for cell in cells:
        entity_list.extend(
            [entity for entity in sim.grid_entity[cell] if entity.active]
        )
    return entity_list


def sweep_circle_vs_tiles(sim, xpos_old, ypos_old, dx, dy, radius):
    """Fetch all segments from neighbourhood. Return shortest intersection time from interpolation."""
    xpos_new = xpos_old + dx
    ypos_new = ypos_old + dy
    width = radius + 1

    # Use conditionals instead of min/max for 2-value comparisons (slightly faster)
    x1 = (xpos_old if xpos_old < xpos_new else xpos_new) - width
    y1 = (ypos_old if ypos_old < ypos_new else ypos_new) - width
    x2 = (xpos_old if xpos_old > xpos_new else xpos_new) + width
    y2 = (ypos_old if ypos_old > ypos_new else ypos_new) + width

    segments = gather_segments_from_region(sim, x1, y1, x2, y2)
    shortest_time = 1

    for segment in segments:
        time = segment.intersect_with_ray(xpos_old, ypos_old, dx, dy, radius)
        # Early exit: if we've found a collision at time 0, we can't find anything earlier
        if time == 0:
            return 0
        # Avoid min() call when time is not shorter
        if time < shortest_time:
            shortest_time = time

    return shortest_time


def get_single_closest_point(sim, xpos, ypos, radius, segments=None):
    """Find the closest point belonging to a collidable segment from the given position.
    Return result and position of the closest point. The result is 0 if no closest point
    found, 1 if belongs to outside edge, -1 if belongs from inside edge.
    """
    if segments is None:
        segments = gather_segments_from_region(
            sim, xpos - radius, ypos - radius, xpos + radius, ypos + radius
        )
    shortest_distance = 99999999.0  # DAT_10572540
    result = 0
    closest_point = None
    for segment in segments:
        is_back_facing, a, b = segment.get_closest_point(xpos, ypos)
        distance_sq = (xpos - a) ** 2 + (ypos - b) ** 2
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
    dist = math.sqrt((xpos1 - xpos2) ** 2 + (ypos1 - ypos2) ** 2)
    return dist < radius1 + radius2


def is_empty_row(sim, xcoord1, xcoord2, ycoord, dir):
    """Return true if the row of half cells have no solid horizontal edge in the specified direction.
    Checks both tile and door grid edge dictionaries (matching original)."""
    xcoords = range(xcoord1, xcoord2 + 1)
    if dir == 1:
        return not (
            any(sim.hor_grid_edge_dic[clamp_half_cell(xcoord, ycoord + 1)] for xcoord in xcoords) or
            any(sim.door_hor_grid_edge_dic[clamp_half_cell(xcoord, ycoord + 1)] for xcoord in xcoords)
        )
    if dir == -1:
        return not (
            any(sim.hor_grid_edge_dic[clamp_half_cell(xcoord, ycoord)] for xcoord in xcoords) or
            any(sim.door_hor_grid_edge_dic[clamp_half_cell(xcoord, ycoord)] for xcoord in xcoords)
        )


def is_empty_column(sim, xcoord, ycoord1, ycoord2, dir):
    """Return true if the column of half cells have no solid vertical edge in the specified direction.
    Checks both tile and door grid edge dictionaries (matching original)."""
    ycoords = range(ycoord1, ycoord2 + 1)
    if dir == 1:
        return not (
            any(sim.ver_grid_edge_dic[clamp_half_cell(xcoord + 1, ycoord)] for ycoord in ycoords) or
            any(sim.door_ver_grid_edge_dic[clamp_half_cell(xcoord + 1, ycoord)] for ycoord in ycoords)
        )
    if dir == -1:
        return not (
            any(sim.ver_grid_edge_dic[clamp_half_cell(xcoord, ycoord)] for ycoord in ycoords) or
            any(sim.door_ver_grid_edge_dic[clamp_half_cell(xcoord, ycoord)] for ycoord in ycoords)
        )


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


def get_time_of_intersection_circle_vs_lineseg(
    xpos, ypos, dx, dy, a1, b1, a2, b2, radius
):
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


def get_time_of_intersection_circle_vs_arc(
    xpos, ypos, vx, vy, a, b, hor, ver, radius_arc, radius_circle
):
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
    if (dx + t * vx) * hor > 0 and (dy + t * vy) * ver > 0:
        return t
    return 1


def map_orientation_to_vector(orientation):
    """Return a normalized vector pointing in the direction of the orientation.
    Orientation is a value between 0 and 7 taken from map data.
    """
    diag = math.sqrt(2) / 2
    orientation_dic = {
        0: (1, 0),
        1: (diag, diag),
        2: (0, 1),
        3: (-diag, diag),
        4: (-1, 0),
        5: (-diag, -diag),
        6: (0, -1),
        7: (diag, -diag),
    }
    return orientation_dic[orientation]


def get_raycast_distance(sim, xpos, ypos, dx, dy):
    """Return the length of a ray given its start point and direction. The ray stops when it hits a
    tile. Return None if the ray hits nothing after travelling for 2000 units. The algorithm works by
    finding the cells the ray traverses and testing it against the tile segments for each cell.
    """
    xcell = math.floor(xpos / 24)
    ycell = math.floor(ypos / 24)
    if dx > 0:
        step_x = 1
        delta_x = 24 / dx
        tmax_x = ((xcell + 1) * 24 - xpos) / dx
    elif dx < 0:
        step_x = -1
        delta_x = -24 / dx
        tmax_x = (xcell * 24 - xpos) / dx
    else:
        step_x = 0
        delta_x = 0
        tmax_x = 999999
    if dy > 0:
        step_y = 1
        delta_y = 24 / dy
        tmax_y = ((ycell + 1) * 24 - ypos) / dy
    elif dy < 0:
        step_y = -1
        delta_y = -24 / dy
        tmax_y = (ycell * 24 - ypos) / dy
    else:
        step_y = 0
        delta_y = 0
        tmax_y = 999999
    while True:
        result = intersect_ray_vs_cell_contents(
            sim, xcell, ycell, xpos, ypos, 2000 * dx, 2000 * dy
        )
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

    **Inactive segments do not block.** `FUN_1050E410` walks a cell in two
    passes: the static tile segments, which are always live, and then a linked
    list of the dynamic ones -- the door segments -- each gated on
    `(*seg & 8) != 0` before its hit is allowed to count. That bit is the same
    `active` flag `gather_segments_from_region` already filters on for the
    ninja's own collision, and this loop was the one place that ignored it, so an
    OPEN door still stopped every sightline in the game: shooter-drone
    acquisition, the turret, the rocket, the eye bat and the dual laser all
    funnel through here.
    """
    segments = sim.segment_dic[clamp_cell(xcell, ycell)]
    shortest_time = 1
    for segment in segments:
        if not segment.active:
            continue
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
        # `get_raycast_distance` returns None when the ray leaves the map without
        # hitting anything, which is a CLEAR sightline -- not a blocked one. The
        # original `length > dist - NINJA_RADIUS` raised TypeError on that path; it
        # had never been reached because no entity called this until the turret,
        # rocket and laser types landed.
        if length is None:
            return True
        return length > dist - NINJA_RADIUS
    return True


def find_visible_ninja(sim, xpos, ypos):
    """Return the nearest ninja with unobstructed line of sight, else None.

    Port of the shared query `FUN_1051E690` in `npp.dll`, which the Rocket (18),
    Turret (19) and EyeBat (27) all call to decide whether they have a target
    (`docs/ENTITY_INTEGRATION_SPEC.md` §10). The binary returns a ninja index or a
    negative value; returning the ninja object or None is the same contract.

    The game loops over the 8 co-op ninja slots and takes the NEAREST visible one
    (spec §6.5). This sim has a single ninja, so the loop degenerates -- but the
    signature is kept nearest-selecting rather than "the ninja" so multi-ninja
    support does not have to rewrite every caller.
    """
    ninja = sim.ninja
    if ninja is None or not ninja.is_valid_target():
        return None
    if raycast_vs_player(sim, xpos, ypos, ninja.xpos, ninja.ypos):
        return ninja
    return None


#: Radius the BEAM hit test uses. The game reads it from
#: `FUN_105136b0`, whose body is not in the decomp corpus; this
#: exists to sweep it. Defaults to NINJA_RADIUS, i.e. no change.
_BEAM_RADIUS = float(
    os.environ.get("NCLONE_AB_BEAM_RADIUS", NINJA_RADIUS)
)


def beam_hits_ninja(x1, y1, x2, y2, ninja, seg_len=None, use_limbs=None):
    """Does a laser/hitscan beam from (x1,y1) to (x2,y2) hit the ninja?

    Every hitscan hazard funnels through here -- turret (19), the two shooter
    drones (12, 13) and the dual laser (23) all call the game's one shared
    `check_lineseg_vs_ninja` -- so a change here moves all of them at once.

    **The real test is `check_lineseg_vs_ninja` below, and it is now the default.**
    It is a circle reject at NINJA_RADIUS followed by an exact intersection against
    the ninja's 11 skeleton segments, so a beam can thread *between* limbs and
    miss. The circle test alone is strictly MORE lethal than the game.

    Both engines now carry the skeleton (`nclone-native/src/anim.rs`) and both
    maintain it on exactly the maps that have a beam-capable entity, so this is
    frame-exact across engines. `NCLONE_AB_NO_LASER_LIMBS=1` restores the circle
    test in *both* -- it is the one env var that A/Bs this, deliberately, so the
    two can never be flipped independently.
    """
    # A zero-length beam hits nothing. Callers legitimately produce one -- every
    # entity here collapses (xend, yend) onto its own position when the raycast
    # finds no wall -- and `overlap_circle_vs_segment` divides by the segment
    # length, so without this guard those frames raise ZeroDivisionError.
    if x1 == x2 and y1 == y2:
        return False
    # `use_limbs=False` is an explicit per-entity opt-out: §73 shows the
    # chaingun's `FUN_1050c690` is a bare circle test with no skeleton, while the
    # global env var cannot distinguish entity types. None = global behaviour.
    if (
        use_limbs is False
        or os.environ.get("NCLONE_AB_NO_LASER_LIMBS") == "1"
        or not getattr(ninja, "bones", None)
    ):
        # Opted out, or no skeleton (a map with no beam never builds one): keep
        # the old circle test EXACTLY, because `nclone-native`'s fallback is a
        # byte-for-byte copy of it. The prefilter inside `check_lineseg_vs_ninja`
        # is the same test algebraically but divides by `seglen` rather than
        # `seg_lensq`, so it is not bit-identical.
        return overlap_circle_vs_segment(
            ninja.xpos, ninja.ypos, _BEAM_RADIUS, x1, y1, x2, y2
        )
    return check_lineseg_vs_ninja(x1, y1, x2, y2, ninja, seg_len)


#: The 11 limbs `FUN_1050B070` tests the beam against, as bone-index pairs
#: (round-7 §3.1). 0 and 1 are the two hubs, `2-8-0` and `3-9-0` the legs,
#: `4-10-1` and `5-11-1` the arms, 12 the head, and `6-7` one extra limb.
#: **Note this differs from `render_utils.LIMBS` in one entry** -- the renderer
#: draws `(1, 12)` where the collision skeleton has `(0, 1)`. Only ten of the
#: eleven are shared, so the two lists are deliberately kept separate.
NINJA_LIMB_PAIRS = (
    (12, 0), (0, 1), (2, 8), (8, 0), (3, 9), (9, 0),
    (4, 10), (10, 1), (5, 11), (11, 1), (6, 7),
)

#: Bones are stored normalised; the game places each at
#: `ninja.pos + 24.0 * bone` (`DAT_1056C360`), which is also how the renderer
#: draws them (`bone * 2 * radius`).
BONE_SCALE = 24.0


def check_lineseg_vs_ninja(x1, y1, x2, y2, ninja, seg_len=None):
    """The game's real beam-vs-ninja test: a circle prefilter, then 11 limbs.

    `FUN_1050B070` (`docs/decomp/full/physics_1050B070.c`). A beam that passes
    within the ninja's radius but *between* two limbs does **not** kill -- which
    is why the circle test alone is biased too lethal.

    The per-limb predicate is `FUN_1050B840`, proper intersection with **strict**
    `< 0` on both cross-product products, so a beam that merely grazes a limb
    endpoint is not a hit either.

    **`seg_len` is the raycast's returned distance, passed in.** The binary takes
    it as `param_5` and never recomputes it; `hypot(x2 - x1, y2 - y1)` equals it
    only up to rounding, because the endpoint was built as `A + dir * seg_len` in
    the first place. `seg_vs_circle` (the turret's test) takes it the same way and
    for the same reason. `None` recomputes, for callers with no raycast to hand.

    Both dot products below put the **y term first**, which is the binary's order
    and not this port's original one.
    """
    dx = x2 - x1
    dy = y2 - y1
    if seg_len is None:
        seg_len = math.sqrt(dx * dx + dy * dy)
    if seg_len == 0.0 or seg_len != seg_len:
        return False
    ux, uy = dx / seg_len, dy / seg_len
    # Projection is NOT clamped before the branch; the branch does the clamping.
    t = (ninja.ypos - y1) * uy + (ninja.xpos - x1) * ux
    if t <= 0.0:
        cx, cy = x1, y1
    elif t >= seg_len:
        cx, cy = x2, y2
    else:
        cx, cy = t * ux + x1, t * uy + y1
    if (ninja.ypos - cy) * (ninja.ypos - cy) + (ninja.xpos - cx) * (
        ninja.xpos - cx
    ) >= NINJA_RADIUS * NINJA_RADIUS:
        return False

    bones = getattr(ninja, "bones", None)
    if not bones:
        # No skeleton available: fall back to the prefilter alone, i.e. the old
        # circle test. Slightly more lethal than the game, never less.
        return True

    nx, ny = ninja.xpos, ninja.ypos
    for ia, ib in NINJA_LIMB_PAIRS:
        ax = nx + BONE_SCALE * bones[ia][0]
        ay = ny + BONE_SCALE * bones[ia][1]
        bx = nx + BONE_SCALE * bones[ib][0]
        by = ny + BONE_SCALE * bones[ib][1]
        d1 = dx * (ay - y1) - dy * (ax - x1)
        d2 = dx * (by - y1) - dy * (bx - x1)
        if d1 * d2 >= 0.0:
            continue
        lx, ly = bx - ax, by - ay
        d3 = lx * (y1 - ay) - ly * (x1 - ax)
        d4 = lx * (y2 - ay) - ly * (x2 - ax)
        if d3 * d4 < 0.0:
            return True
    return False

def seg_vs_circle(cx, cy, radius, ax, ay, bx, by, seg_len):
    """`FUN_1050C690`: the hitscan test the TURRET and the chaingun drone use.

    Distinct from `beam_hits_ninja` -- the callers split cleanly in the binary.
    `FUN_1050B070` (the 11-limb test) is called only by the laser family: dual
    laser (23, both modes) and the drone laser (13). `FUN_1050C690` is called by
    the turret (19, `1041A150`) and the chaingun drone (12, `103FE790`). Routing
    the turret through the limb test made it *less* lethal than the game.

    Project the centre onto the segment, clamp to the endpoints, compare the
    distance to the radius with a **strict** `<`. The radius is the ninja's own
    (`FUN_105136B0`) -- there is no separate shot radius.

    `seg_len` is passed in rather than recomputed because the game passes the
    raycast's returned distance. `|B - A|` equals it only up to rounding, and the
    two orderings below (the y term first in both dot products, and `A + d*t`
    with `d` pre-divided) are the binary's, not an algebraic simplification.
    """
    # `if (NAN(len) != (len == 0.0)) return false` -- reject a zero or NaN length
    # before the divide.
    if seg_len == 0.0 or seg_len != seg_len:
        return False
    dx = (bx - ax) / seg_len
    dy = (by - ay) / seg_len
    t = (cy - ay) * dy + (cx - ax) * dx
    if t > 0.0:
        if t < seg_len:
            px = dx * t + ax
            py = dy * t + ay
        else:
            px, py = bx, by
    else:
        px, py = ax, ay
    return (cy - py) * (cy - py) + (cx - px) * (cx - px) < radius * radius


def overlap_circle_vs_segment(xpos, ypos, radius, px1, py1, px2, py2):
    """Given a circle defined by its center and radius, and a segment defined by two points,
    return true if they overlap.
    """
    px = px2 - px1
    py = py2 - py1
    dx = xpos - px1
    dy = ypos - py1
    seg_lensq = px**2 + py**2
    u = (dx * px + dy * py) / seg_lensq
    u = max(u, 0)
    u = min(u, 1)
    a = px1 + u * px
    b = py1 + u * py
    return (xpos - a) ** 2 + (ypos - b) ** 2 < radius**2


# Clear cache periodically to prevent memory growth
def clear_caches():
    """Clear physics helper caches to prevent memory growth.
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
        common_values = {
            0,
            1,
            2,
            4,
            9,
            16,
            25,
            36,
            49,
            64,
            81,
            100,
            121,
            144,
            169,
            196,
            225,
            256,
            324,
            400,
        }
        _sqrt_cache_copy = {k: v for k, v in _sqrt_cache.items() if k in common_values}
        _sqrt_cache.clear()
        _sqrt_cache.update(_sqrt_cache_copy)
