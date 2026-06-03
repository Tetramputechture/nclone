"""Surface extraction from tile geometry for the surface graph PBRS.

Extracts walkable floor and wall surfaces from the level's tile collision
segments, merges adjacent segments into continuous surface runs, and samples
SurfacePoint positions at regular intervals for graph node placement.

The ninja's collision is a circle of NINJA_RADIUS pixels. SurfacePoint
positions store the ninja CENTER when standing on the surface (offset from
the contact point by NINJA_RADIUS along the surface outward normal).
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np

from ..constants.physics_constants import NINJA_RADIUS, TOGGLE_MINE_RADIUS_TOGGLED
from ..entities import GridSegmentCircular, GridSegmentLinear
from ..utils.tile_segment_factory import TileSegmentFactory

# ---------------------------------------------------------------------------
# Surface type constants
# ---------------------------------------------------------------------------
SURFACE_FLOOR = 0
SURFACE_WALL = 1
# Wall-slide node: co-located with a wall surface point but tagged explicitly
# as representing the ninja in a wall-slide state (falling with significant
# downward velocity).  Using a distinct class lets the enumerator attach
# slide-to-jump arc templates without polluting the stationary wall bins.
SURFACE_SLIDE = 2
# Launch-pad node: synthetic point placed at a launch pad entity.  These nodes
# are airborne-kick sources — they do not represent a walkable surface.
SURFACE_LAUNCH_PAD = 3


def is_wall_like(surface_type: int) -> bool:
    """Return ``True`` if the node behaves like a wall (wall or slide)."""
    return surface_type == SURFACE_WALL or surface_type == SURFACE_SLIDE


def is_ground_node(surface_type: int) -> bool:
    """Return ``True`` if the node lives on a static walkable surface."""
    return (
        surface_type == SURFACE_FLOOR
        or surface_type == SURFACE_WALL
        or surface_type == SURFACE_SLIDE
    )

# ---------------------------------------------------------------------------
# Classification thresholds
# ---------------------------------------------------------------------------
# In screen coordinates Y points down. Floor normals point up (ny < 0).
_FLOOR_NORMAL_Y_THRESHOLD = -0.3
# Wall normals are predominantly horizontal.
_WALL_NORMAL_X_THRESHOLD = 0.7
# Wall normals must not point significantly downward (ny < 0 means upward in
# screen coords). The underside of a slope tile has ny > 0 (facing down) and
# must be excluded — the ninja cannot wall-slide on a ceiling or overhang.
_WALL_NORMAL_Y_MAX = 0.05

# ---------------------------------------------------------------------------
# Sampling parameters (pixels)
# ---------------------------------------------------------------------------
SAMPLE_RESOLUTION = 6.0
EDGE_SAMPLE_RESOLUTION = 3.0
EDGE_PROXIMITY = 6.0

# ---------------------------------------------------------------------------
# Mine exclusion zone radii (pixels)
# ---------------------------------------------------------------------------
# Radius used to split surface segments at mine locations. Conservative (4 px
# buffer beyond kill radius) so the agent has room to initiate a jump before
# reaching the lethal zone.
MINE_SURFACE_EXCLUSION_RADIUS = NINJA_RADIUS + TOGGLE_MINE_RADIUS_TOGGLED + 1.0  # 15 px

# ---------------------------------------------------------------------------
# Merging / deduplication tolerances (pixels)
# ---------------------------------------------------------------------------
_ENDPOINT_TOLERANCE = 1.0
_MIN_SEGMENT_LENGTH = 2.0
_DEDUP_DISTANCE = 2.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class SurfacePoint:
    """A sampled point on a walkable surface for graph node placement.

    Attributes:
        x, y: Ninja CENTER position when standing on this surface (pixels).
        normal_x, normal_y: Outward surface normal (unit vector, away from solid).
        surface_type: SURFACE_FLOOR (0) or SURFACE_WALL (1).
        segment_id: ID of the merged surface run this point belongs to.
        is_edge: True if within EDGE_PROXIMITY of a platform / wall endpoint.
        tangent_x, tangent_y: Unit surface TANGENT, canonically oriented so the
            binned velocity is the signed along-surface speed (floor: +x = right;
            wall: +y = down).  For axis-aligned floors/walls this is exactly
            (1,0)/(0,1) so the legacy world-vx/vy model is reproduced; for
            slopes/curves it carries the surface angle.  Used only by the
            ``"tangential"`` momentum model (default ``(1,0)``).
    """

    x: float
    y: float
    normal_x: float
    normal_y: float
    surface_type: int
    segment_id: int
    is_edge: bool
    tangent_x: float = 1.0
    tangent_y: float = 0.0


@dataclass
class _ClassifiedSegment:
    """Internal representation of a classified collision segment."""

    x1: float
    y1: float
    x2: float
    y2: float
    normal_x: float
    normal_y: float
    surface_type: int
    length: float


# ---------------------------------------------------------------------------
# Segment classification
# ---------------------------------------------------------------------------
def _compute_linear_normal(seg: GridSegmentLinear) -> Optional[Tuple[float, float]]:
    """Compute the outward-facing unit normal of a linear segment.

    For oriented segments the outward normal is (-py, px) / length,
    perpendicular to the p1->p2 direction on the non-solid side.
    """
    length_sq = seg.px * seg.px + seg.py * seg.py
    if length_sq < 1e-12:
        return None
    length = math.sqrt(length_sq)
    return (-seg.py / length, seg.px / length)


def surface_tangent(
    normal_x: float, normal_y: float, surface_type: int,
) -> Tuple[float, float]:
    """Unit surface tangent, canonically oriented so the binned velocity is the
    signed along-surface speed.

    The raw tangent is ``(-ny, nx)`` (perpendicular to the outward normal).  We
    orient it so a FLOOR's tangent points right (+x, matching the legacy ``+vx``
    bin) and a WALL/SLIDE's tangent points down (+y, matching the legacy ``+vy``
    bin).  For an axis-aligned floor this is exactly ``(1, 0)`` and for a vertical
    wall exactly ``(0, 1)`` regardless of which side the wall is on — so the
    tangential model reproduces the world-vx/vy model on axis-aligned surfaces.
    For slopes/curves it carries the true surface angle.
    """
    tx, ty = -normal_y, normal_x
    if is_wall_like(surface_type):
        if ty < 0.0:                      # orient downward-positive (+vy)
            tx, ty = -tx, -ty
    else:
        if tx < 0.0:                      # orient rightward-positive (+vx)
            tx, ty = -tx, -ty
    return tx, ty


def _classify_linear_segment(seg) -> Optional[_ClassifiedSegment]:
    """Classify a linear segment as floor, wall, or neither (discarded)."""
    if not isinstance(seg, GridSegmentLinear):
        return None

    result = _compute_linear_normal(seg)
    if result is None:
        return None
    nx, ny = result
    length = math.sqrt(seg.px * seg.px + seg.py * seg.py)
    if length < _MIN_SEGMENT_LENGTH:
        return None

    # Floor: normal points significantly upward (negative Y in screen coords)
    if ny < _FLOOR_NORMAL_Y_THRESHOLD:
        return _ClassifiedSegment(
            seg.x1, seg.y1, seg.x2, seg.y2, nx, ny, SURFACE_FLOOR, length
        )

    # Wall: normal is predominantly horizontal AND does not point downward.
    # ny >= _WALL_NORMAL_Y_MAX would mean the surface faces downward (overhang /
    # slope underside) where the ninja cannot wall-slide, so those are discarded.
    if abs(nx) > _WALL_NORMAL_X_THRESHOLD and ny < _WALL_NORMAL_Y_MAX:
        return _ClassifiedSegment(
            seg.x1, seg.y1, seg.x2, seg.y2, nx, ny, SURFACE_WALL, length
        )

    return None


# ---------------------------------------------------------------------------
# Segment merging via union-find with spatial hashing
# ---------------------------------------------------------------------------
def _merge_segments(
    segments: List[_ClassifiedSegment],
) -> List[List[_ClassifiedSegment]]:
    """Merge connected segments of the same surface type into runs.

    Two segments are connected if they share an endpoint (within tolerance)
    and have compatible outward normals.  Uses union-find with a spatial hash
    on endpoints for O(N) average-case performance.
    """
    if not segments:
        return []

    n = len(segments)
    parent = list(range(n))
    uf_rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if uf_rank[ra] < uf_rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if uf_rank[ra] == uf_rank[rb]:
            uf_rank[ra] += 1

    # Spatial hash on endpoints
    cell_size = max(_ENDPOINT_TOLERANCE * 2.0, 1e-6)
    inv_cell = 1.0 / cell_size
    # key -> list of (segment_index, endpoint_x, endpoint_y)
    endpoint_cells: Dict[Tuple[int, int, int], List[Tuple[int, float, float]]] = {}

    for i, seg in enumerate(segments):
        for ex, ey in [(seg.x1, seg.y1), (seg.x2, seg.y2)]:
            gx = int(math.floor(ex * inv_cell))
            gy = int(math.floor(ey * inv_cell))
            key = (gx, gy, seg.surface_type)
            if key not in endpoint_cells:
                endpoint_cells[key] = []
            endpoint_cells[key].append((i, ex, ey))

    tol_sq = _ENDPOINT_TOLERANCE * _ENDPOINT_TOLERANCE

    for i, seg in enumerate(segments):
        for ex, ey in [(seg.x1, seg.y1), (seg.x2, seg.y2)]:
            gx = int(math.floor(ex * inv_cell))
            gy = int(math.floor(ey * inv_cell))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    key = (gx + dx, gy + dy, seg.surface_type)
                    entries = endpoint_cells.get(key)
                    if entries is None:
                        continue
                    for j, jx, jy in entries:
                        if j <= i:
                            continue
                        ddx = ex - jx
                        ddy = ey - jy
                        if ddx * ddx + ddy * ddy <= tol_sq:
                            sj = segments[j]
                            dot = (
                                seg.normal_x * sj.normal_x + seg.normal_y * sj.normal_y
                            )
                            if dot > 0.3:
                                union(i, j)

    groups: Dict[int, List[_ClassifiedSegment]] = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(segments[i])

    return list(groups.values())


# ---------------------------------------------------------------------------
# Open endpoint detection
# ---------------------------------------------------------------------------
def _find_open_endpoints(
    run_segments: List[_ClassifiedSegment],
) -> List[Tuple[float, float]]:
    """Find endpoints not shared between segments in a run (platform edges)."""
    tol = max(_ENDPOINT_TOLERANCE, 0.1)
    inv_cell = 1.0 / tol

    grid_counts: Dict[Tuple[int, int], int] = {}
    grid_to_point: Dict[Tuple[int, int], Tuple[float, float]] = {}

    for seg in run_segments:
        for ex, ey in [(seg.x1, seg.y1), (seg.x2, seg.y2)]:
            gx = round(ex * inv_cell)
            gy = round(ey * inv_cell)
            key = (gx, gy)
            grid_counts[key] = grid_counts.get(key, 0) + 1
            if key not in grid_to_point:
                grid_to_point[key] = (ex, ey)

    return [grid_to_point[k] for k, c in grid_counts.items() if c == 1]


# ---------------------------------------------------------------------------
# Point sampling along a segment
# ---------------------------------------------------------------------------
def _sample_segment(
    seg: _ClassifiedSegment,
    segment_id: int,
    open_endpoints: List[Tuple[float, float]],
) -> List[SurfacePoint]:
    """Sample SurfacePoint positions along a classified segment.

    Uses SAMPLE_RESOLUTION for general spacing and EDGE_SAMPLE_RESOLUTION
    for denser sampling near open (platform-edge) endpoints.
    """
    if seg.length < 1e-6:
        return []

    dx = seg.x2 - seg.x1
    dy = seg.y2 - seg.y1
    points: List[SurfacePoint] = []
    tan_x, tan_y = surface_tangent(seg.normal_x, seg.normal_y, seg.surface_type)

    # --- Regular sampling at SAMPLE_RESOLUTION ---
    n_samples = max(2, int(math.ceil(seg.length / SAMPLE_RESOLUTION)) + 1)

    for i in range(n_samples):
        t = i / max(1, n_samples - 1)
        cx = seg.x1 + t * dx
        cy = seg.y1 + t * dy

        is_edge = _near_open_endpoint(cx, cy, open_endpoints)
        ninja_x = cx + seg.normal_x * NINJA_RADIUS
        ninja_y = cy + seg.normal_y * NINJA_RADIUS

        points.append(
            SurfacePoint(
                x=ninja_x,
                y=ninja_y,
                normal_x=seg.normal_x,
                normal_y=seg.normal_y,
                surface_type=seg.surface_type,
                segment_id=segment_id,
                is_edge=is_edge,
                tangent_x=tan_x,
                tangent_y=tan_y,
            )
        )

    # --- Denser sampling near open endpoints ---
    for epx, epy in open_endpoints:
        for t_end in (0.0, 1.0):
            sx = seg.x1 + t_end * dx
            sy = seg.y1 + t_end * dy
            if (sx - epx) ** 2 + (sy - epy) ** 2 > _ENDPOINT_TOLERANCE**2:
                continue
            n_dense = int(math.ceil(EDGE_PROXIMITY / EDGE_SAMPLE_RESOLUTION))
            for j in range(1, n_dense + 1):
                t_offset = EDGE_SAMPLE_RESOLUTION * j / seg.length
                t = t_offset if t_end == 0.0 else 1.0 - t_offset
                if not (0.0 < t < 1.0):
                    continue
                cx = seg.x1 + t * dx
                cy = seg.y1 + t * dy
                ninja_x = cx + seg.normal_x * NINJA_RADIUS
                ninja_y = cy + seg.normal_y * NINJA_RADIUS
                points.append(
                    SurfacePoint(
                        x=ninja_x,
                        y=ninja_y,
                        normal_x=seg.normal_x,
                        normal_y=seg.normal_y,
                        surface_type=seg.surface_type,
                        segment_id=segment_id,
                        is_edge=True,
                        tangent_x=tan_x,
                        tangent_y=tan_y,
                    )
                )

    return points


def _near_open_endpoint(
    cx: float, cy: float, open_endpoints: List[Tuple[float, float]]
) -> bool:
    """Check if a contact point is near any open endpoint."""
    prox_sq = EDGE_PROXIMITY * EDGE_PROXIMITY
    for epx, epy in open_endpoints:
        if (cx - epx) ** 2 + (cy - epy) ** 2 < prox_sq:
            return True
    return False


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def _deduplicate_points(points: List[SurfacePoint]) -> List[SurfacePoint]:
    """Remove duplicate / very close surface points, prioritizing edge flags."""
    if not points:
        return points

    dedup_sq = _DEDUP_DISTANCE * _DEDUP_DISTANCE
    cell_size = max(_DEDUP_DISTANCE * 2.0, 1e-6)
    inv_cell = 1.0 / cell_size

    kept: List[SurfacePoint] = []
    # (gx, gy) -> list of indices into *kept*
    occupied: Dict[Tuple[int, int], List[int]] = {}

    # Process edge-flagged points first so they take priority
    sorted_pts = sorted(points, key=lambda p: (not p.is_edge, p.x, p.y))

    for pt in sorted_pts:
        gx = int(math.floor(pt.x * inv_cell))
        gy = int(math.floor(pt.y * inv_cell))

        too_close = False
        for ddx in (-1, 0, 1):
            if too_close:
                break
            for ddy in (-1, 0, 1):
                if too_close:
                    break
                key = (gx + ddx, gy + ddy)
                indices = occupied.get(key)
                if indices is None:
                    continue
                for ki in indices:
                    kp = kept[ki]
                    dx2 = pt.x - kp.x
                    dy2 = pt.y - kp.y
                    if dx2 * dx2 + dy2 * dy2 < dedup_sq:
                        if pt.is_edge and not kp.is_edge:
                            kept[ki] = pt
                        too_close = True
                        break

        if not too_close:
            idx = len(kept)
            kept.append(pt)
            key = (gx, gy)
            if key not in occupied:
                occupied[key] = []
            occupied[key].append(idx)

    return kept


# ---------------------------------------------------------------------------
# Mine exclusion zone splitting
# ---------------------------------------------------------------------------
def _find_segment_exclusion_params(
    seg: _ClassifiedSegment,
    mine_x: float,
    mine_y: float,
    excl_radius: float,
) -> Optional[Tuple[float, float]]:
    """Find where a segment's ninja-center trajectory enters a mine exclusion zone.

    Solves the line-circle intersection between the ninja center path along the
    segment and a circle of radius ``excl_radius`` centred on the mine.  The
    ninja center is offset from the tile contact line by NINJA_RADIUS along the
    surface normal.

    Args:
        seg: Classified tile segment.
        mine_x, mine_y: Mine centre coordinates.
        excl_radius: Exclusion zone radius.

    Returns:
        ``(t_enter, t_exit)`` parameter values in ``[0, 1]`` along the segment,
        or ``None`` if the circle does not intersect the segment.
    """
    # Ninja center at t=0 and t=1
    nx0 = seg.x1 + seg.normal_x * NINJA_RADIUS
    ny0 = seg.y1 + seg.normal_y * NINJA_RADIUS
    nx1 = seg.x2 + seg.normal_x * NINJA_RADIUS
    ny1 = seg.y2 + seg.normal_y * NINJA_RADIUS

    # Parametric direction
    dx = nx1 - nx0
    dy = ny1 - ny0

    # Quadratic: |P(t) - mine|² = excl_radius²
    #   a·t² + b·t + c = 0
    fx = nx0 - mine_x
    fy = ny0 - mine_y
    a = dx * dx + dy * dy
    if a < 1e-12:
        return None  # Degenerate (zero-length) segment

    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - excl_radius * excl_radius

    discriminant = b * b - 4.0 * a * c
    if discriminant <= 0.0:
        return None  # No real intersection

    sqrt_d = math.sqrt(discriminant)
    t_enter = max(0.0, (-b - sqrt_d) / (2.0 * a))
    t_exit = min(1.0, (-b + sqrt_d) / (2.0 * a))

    if t_enter >= t_exit:
        return None  # Intersection entirely outside segment range

    return (t_enter, t_exit)


def _split_one_segment(
    seg: _ClassifiedSegment,
    mine_positions: List[Tuple[float, float]],
    excl_radius: float,
) -> List[_ClassifiedSegment]:
    """Split a single segment at all mine exclusion zones.

    Collects all ``[t_enter, t_exit]`` intervals where the ninja center
    trajectory enters an exclusion zone, merges overlapping intervals, then
    returns the surviving pieces of the segment as new ``_ClassifiedSegment``
    objects.  Pieces shorter than ``_MIN_SEGMENT_LENGTH`` are dropped.

    Args:
        seg: The segment to split.
        mine_positions: List of ``(mine_x, mine_y)`` tuples.
        excl_radius: Exclusion zone radius.

    Returns:
        List of surviving segment pieces (may be empty if the entire segment
        is inside one or more exclusion zones).
    """
    # Collect and sort exclusion intervals along the segment's t-axis
    intervals: List[Tuple[float, float]] = []
    for mx, my in mine_positions:
        result = _find_segment_exclusion_params(seg, mx, my, excl_radius)
        if result is not None:
            intervals.append(result)

    if not intervals:
        return [seg]

    # Merge overlapping intervals
    intervals.sort(key=lambda iv: iv[0])
    merged: List[Tuple[float, float]] = [intervals[0]]
    for t_in, t_out in intervals[1:]:
        if t_in <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], t_out))
        else:
            merged.append((t_in, t_out))

    # Build surviving t-ranges: gaps between exclusion intervals
    surviving: List[Tuple[float, float]] = []
    prev_end = 0.0
    for t_in, t_out in merged:
        if t_in > prev_end:
            surviving.append((prev_end, t_in))
        prev_end = t_out
    if prev_end < 1.0:
        surviving.append((prev_end, 1.0))

    # Convert t-ranges back to _ClassifiedSegment objects
    full_dx = seg.x2 - seg.x1
    full_dy = seg.y2 - seg.y1
    pieces: List[_ClassifiedSegment] = []
    for t_start, t_end in surviving:
        px1 = seg.x1 + t_start * full_dx
        py1 = seg.y1 + t_start * full_dy
        px2 = seg.x1 + t_end * full_dx
        py2 = seg.y1 + t_end * full_dy
        length = math.sqrt((px2 - px1) ** 2 + (py2 - py1) ** 2)
        if length < _MIN_SEGMENT_LENGTH:
            continue
        pieces.append(
            _ClassifiedSegment(
                x1=px1,
                y1=py1,
                x2=px2,
                y2=py2,
                normal_x=seg.normal_x,
                normal_y=seg.normal_y,
                surface_type=seg.surface_type,
                length=length,
            )
        )
    return pieces


def _split_segments_at_mines(
    segments: List[_ClassifiedSegment],
    mine_positions: List[Tuple[float, float]],
    excl_radius: float,
) -> List[_ClassifiedSegment]:
    """Apply mine exclusion splitting to all segments.

    Args:
        segments: All classified segments.
        mine_positions: List of ``(mine_x, mine_y)`` tuples.
        excl_radius: Exclusion zone radius.

    Returns:
        Expanded list of segments with mine danger zones removed.
    """
    result: List[_ClassifiedSegment] = []
    for seg in segments:
        result.extend(_split_one_segment(seg, mine_positions, excl_radius))
    return result


# ---------------------------------------------------------------------------
# Mine boundary node precomputation
# ---------------------------------------------------------------------------
def compute_mine_boundary_nodes(
    surface_points: List[SurfacePoint],
    mine_positions_list: List[Tuple[float, float]],
    excl_radius: float = MINE_SURFACE_EXCLUSION_RADIUS,
) -> np.ndarray:
    """For each mine, find the closest is_edge=True surface nodes on each side.

    After segment splitting, the mine exclusion boundary produces open platform
    endpoints that are densely sampled with ``is_edge=True`` nodes.  This
    function locates the two closest such nodes on opposite sides of each mine
    (left/right for floor mines; top/bottom for wall mines).  At runtime these
    boundary node positions are used to pin the PBRS nearest-node lookup when
    the agent is inside an exclusion zone, preventing spurious rewards from
    snapping to the post-mine boundary node.

    Args:
        surface_points: All sampled surface points (ninja-center positions).
        mine_positions_list: List of (mine_x, mine_y) tuples — must cover
            only the mines for which boundary pinning should be active (e.g.
            always-deadly mines only).
        excl_radius: Mine surface exclusion radius (default
            ``MINE_SURFACE_EXCLUSION_RADIUS``).

    Returns:
        int32 array of shape ``(num_mines, 2)``.  For each mine:
          - ``[i, 0]``: index of the closest ``is_edge=True`` boundary node on
            the negative-axis side (x < mine_x for floor; y < mine_y for wall).
          - ``[i, 1]``: index on the positive-axis side.
          - ``-1`` if no candidate found on that side.
    """
    num_mines = len(mine_positions_list)
    if num_mines == 0 or not surface_points:
        return np.empty((0, 2), dtype=np.int32)

    # Boundary nodes are at ~excl_radius from the mine center (ninja-center
    # coordinates).  Dense edge samples extend EDGE_PROXIMITY inward from the
    # boundary endpoint, so the furthest reachable is_edge node is at
    # excl_radius + EDGE_PROXIMITY from the mine.  A search radius of
    # excl_radius + EDGE_PROXIMITY catches all candidates reliably.
    search_radius_sq = (excl_radius + EDGE_PROXIMITY) ** 2

    result = np.full((num_mines, 2), -1, dtype=np.int32)

    for mine_idx, (mx, my) in enumerate(mine_positions_list):
        # Collect all is_edge=True nodes within the search radius, recording
        # their squared distance and surface type for axis selection.
        nearby: List[Tuple[float, int, int]] = []  # (dist_sq, node_idx, surface_type)
        for i, sp in enumerate(surface_points):
            if not sp.is_edge:
                continue
            dx = sp.x - mx
            dy = sp.y - my
            d2 = dx * dx + dy * dy
            if d2 <= search_radius_sq:
                nearby.append((d2, i, sp.surface_type))

        if not nearby:
            continue

        # Choose split axis: use y (top/bottom) when wall nodes dominate,
        # otherwise use x (left/right) for floor mines.
        n_wall = sum(1 for _, _, st in nearby if st == SURFACE_WALL)
        use_y_split = n_wall > len(nearby) / 2

        neg_best_d2 = float("inf")
        pos_best_d2 = float("inf")

        for d2, node_idx, _ in nearby:
            sp = surface_points[node_idx]
            coord = sp.y if use_y_split else sp.x
            mine_coord = my if use_y_split else mx
            if coord < mine_coord:
                if d2 < neg_best_d2:
                    neg_best_d2 = d2
                    result[mine_idx, 0] = node_idx
            else:
                if d2 < pos_best_d2:
                    pos_best_d2 = d2
                    result[mine_idx, 1] = node_idx

    return result


def _sample_circular(seg: GridSegmentCircular, segment_id: int) -> List[SurfacePoint]:
    """Sample SurfacePoint positions along a circular (quarter-arc) tile segment.

    Uses the REAL arc geometry (centre, quadrant, convex) the collision system
    uses.  The arc is parameterised by ``alpha in [0, pi/2]`` from ``p_hor`` to
    ``p_ver``; the contact point is ``centre + radius*(hor*cos, ver*sin)``.  The
    OUTWARD normal (toward the walkable, non-back-facing side per
    ``GridSegmentCircular.get_closest_point``) is the radial direction away from
    the centre for a CONVEX corner and toward the centre for a CONCAVE pipe.
    Each sample is typed FLOOR/WALL by the same normal thresholds as a linear
    segment (so a quarter-pipe is wall near its side and floor near its base,
    and down-facing arc portions are skipped), and the tangent carries the local
    surface angle for the tangential momentum model.
    """
    r = seg.radius
    arc_len = r * (math.pi / 2.0)
    n = max(2, int(math.ceil(arc_len / SAMPLE_RESOLUTION)) + 1)
    pts: List[SurfacePoint] = []
    for i in range(n):
        alpha = (math.pi / 2.0) * (i / (n - 1))
        rx = seg.hor * math.cos(alpha)   # unit radial dir centre -> contact
        ry = seg.ver * math.sin(alpha)
        contact_x = seg.xpos + r * rx
        contact_y = seg.ypos + r * ry
        nx, ny = (rx, ry) if seg.convex else (-rx, -ry)   # outward normal
        if ny < _FLOOR_NORMAL_Y_THRESHOLD:
            stype = SURFACE_FLOOR
        elif abs(nx) > _WALL_NORMAL_X_THRESHOLD and ny < _WALL_NORMAL_Y_MAX:
            stype = SURFACE_WALL
        else:
            continue  # arc portion faces down / overhang: not walkable
        tan_x, tan_y = surface_tangent(nx, ny, stype)
        pts.append(
            SurfacePoint(
                x=contact_x + nx * NINJA_RADIUS,
                y=contact_y + ny * NINJA_RADIUS,
                normal_x=nx,
                normal_y=ny,
                surface_type=stype,
                segment_id=segment_id,
                is_edge=(i == 0 or i == n - 1),
                tangent_x=tan_x,
                tangent_y=tan_y,
            )
        )
    return pts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_surfaces(
    tile_dict: Dict[Tuple[int, int], int],
    mine_positions: Optional[List[Tuple[float, float]]] = None,
    momentum_model: str = "legacy",
    keep_mine_geometry: bool = False,
) -> List[SurfacePoint]:
    """Extract walkable surface points from level tile data.

    Pipeline:
      1. Build collision segments from tile geometry via TileSegmentFactory.
      2. Classify each linear segment as floor, wall, or neither (discarded).
      2b. Split segments at mine exclusion zones (virtual gap approach) so that
          no SurfacePoints are placed inside mine danger areas.  The split
          endpoints become open platform edges and receive 3 px dense sampling.
      3. Merge connected same-type segments into continuous surface runs.
      4. Sample SurfacePoint positions at 6 px intervals (3 px near edges).
      5. Deduplicate nearby points.

    Args:
        tile_dict: ``{(col, row): tile_type}`` from ``simulator.get_tile_dict()``.
        mine_positions: Optional list of ``(mine_x, mine_y)`` tuples.  When
            supplied, surface segments are split at each mine's exclusion zone
            (radius ``MINE_SURFACE_EXCLUSION_RADIUS``) so that no graph nodes
            are placed in the traversability-blocked region around the mine.

    Returns:
        List of SurfacePoint instances with ninja-center positions on walkable
        surfaces.  Typical count: 2 000 -- 4 000 per level.
    """
    # 1. Build collision segments
    segment_dic = TileSegmentFactory.create_segment_dictionary(tile_dict)

    # 2. Classify linear segments
    classified: List[_ClassifiedSegment] = []
    for _cell_key, cell_segments in segment_dic.items():
        for seg in cell_segments:
            cls_seg = _classify_linear_segment(seg)
            if cls_seg is not None:
                classified.append(cls_seg)

    # 2b. Split segments at mine exclusion zones (virtual gap approach).
    # Each mine creates a gap in the surface — the split endpoints become
    # open platform edges and automatically receive 3 px dense sampling in
    # step 4, giving the agent fine-resolution nodes for jump initiation.
    # ``keep_mine_geometry`` skips this DELETION step: the segment cut here is
    # measured on the raw tile-edge segment (where a surface-mounted mine sits at
    # perpendicular distance 0) with a flat ``MINE_SURFACE_EXCLUSION_RADIUS``,
    # which is wider than a 1-tile feature's half-width and so erases launchable
    # trick corners (e.g. a CDJ off a 1×1 block topped by a deadly mine — the
    # corner node is 2D-survivable by ~1.6 px but the edge-frame cut removes it).
    # When enabled the geometry is preserved and the PRECISE 2D node mask
    # (``find_mine_blocked_nodes``, ninja-center coords) is applied downstream by
    # the caller, so normal routing still avoids lethal nodes while trick
    # enumeration can launch from the surviving corner.
    if mine_positions and not keep_mine_geometry:
        classified = _split_segments_at_mines(
            classified, mine_positions, MINE_SURFACE_EXCLUSION_RADIUS
        )

    # 3. Merge into surface runs
    runs = _merge_segments(classified)

    # 4. Sample points along each run
    surface_points: List[SurfacePoint] = []
    for run_id, run_segments in enumerate(runs):
        open_endpoints = _find_open_endpoints(run_segments)
        for seg in run_segments:
            seg_points = _sample_segment(seg, run_id, open_endpoints)
            surface_points.extend(seg_points)

    # 4b. Curved tiles (tangential model only).  The legacy model drops every
    # circular segment (the isinstance(GridSegmentLinear) classifier); here we
    # sample arc nodes from the REAL GridSegmentCircular geometry so routes can
    # anchor on quarter-pipes / convex corners for velocity gain.  Mine masking
    # for these nodes is left to find_mine_blocked_nodes (per plan).
    if momentum_model == "tangential":
        next_id = len(runs)
        for _cell_key, cell_segments in segment_dic.items():
            for seg in cell_segments:
                if isinstance(seg, GridSegmentCircular):
                    surface_points.extend(_sample_circular(seg, next_id))
                    next_id += 1

    # 5. Deduplicate
    surface_points = _deduplicate_points(surface_points)

    return surface_points
