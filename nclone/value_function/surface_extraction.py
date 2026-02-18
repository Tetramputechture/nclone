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

from ..constants.physics_constants import NINJA_RADIUS
from ..entities import GridSegmentLinear
from ..utils.tile_segment_factory import TileSegmentFactory

# ---------------------------------------------------------------------------
# Surface type constants
# ---------------------------------------------------------------------------
SURFACE_FLOOR = 0
SURFACE_WALL = 1

# ---------------------------------------------------------------------------
# Classification thresholds
# ---------------------------------------------------------------------------
# In screen coordinates Y points down. Floor normals point up (ny < 0).
_FLOOR_NORMAL_Y_THRESHOLD = -0.3
# Wall normals are predominantly horizontal.
_WALL_NORMAL_X_THRESHOLD = 0.7

# ---------------------------------------------------------------------------
# Sampling parameters (pixels)
# ---------------------------------------------------------------------------
SAMPLE_RESOLUTION = 6.0
EDGE_SAMPLE_RESOLUTION = 3.0
EDGE_PROXIMITY = 12.0

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
    """
    x: float
    y: float
    normal_x: float
    normal_y: float
    surface_type: int
    segment_id: int
    is_edge: bool


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

    # Wall: normal is predominantly horizontal
    if abs(nx) > _WALL_NORMAL_X_THRESHOLD:
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
                            dot = (seg.normal_x * sj.normal_x
                                   + seg.normal_y * sj.normal_y)
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

    # --- Regular sampling at SAMPLE_RESOLUTION ---
    n_samples = max(2, int(math.ceil(seg.length / SAMPLE_RESOLUTION)) + 1)

    for i in range(n_samples):
        t = i / max(1, n_samples - 1)
        cx = seg.x1 + t * dx
        cy = seg.y1 + t * dy

        is_edge = _near_open_endpoint(cx, cy, open_endpoints)
        ninja_x = cx + seg.normal_x * NINJA_RADIUS
        ninja_y = cy + seg.normal_y * NINJA_RADIUS

        points.append(SurfacePoint(
            x=ninja_x, y=ninja_y,
            normal_x=seg.normal_x, normal_y=seg.normal_y,
            surface_type=seg.surface_type,
            segment_id=segment_id,
            is_edge=is_edge,
        ))

    # --- Denser sampling near open endpoints ---
    for epx, epy in open_endpoints:
        for t_end in (0.0, 1.0):
            sx = seg.x1 + t_end * dx
            sy = seg.y1 + t_end * dy
            if (sx - epx) ** 2 + (sy - epy) ** 2 > _ENDPOINT_TOLERANCE ** 2:
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
                points.append(SurfacePoint(
                    x=ninja_x, y=ninja_y,
                    normal_x=seg.normal_x, normal_y=seg.normal_y,
                    surface_type=seg.surface_type,
                    segment_id=segment_id,
                    is_edge=True,
                ))

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
# Public API
# ---------------------------------------------------------------------------
def extract_surfaces(
    tile_dict: Dict[Tuple[int, int], int],
) -> List[SurfacePoint]:
    """Extract walkable surface points from level tile data.

    Pipeline:
      1. Build collision segments from tile geometry via TileSegmentFactory.
      2. Classify each linear segment as floor, wall, or neither (discarded).
      3. Merge connected same-type segments into continuous surface runs.
      4. Sample SurfacePoint positions at 6 px intervals (3 px near edges).
      5. Deduplicate nearby points.

    Args:
        tile_dict: ``{(col, row): tile_type}`` from ``simulator.get_tile_dict()``.

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

    # 3. Merge into surface runs
    runs = _merge_segments(classified)

    # 4. Sample points along each run
    surface_points: List[SurfacePoint] = []
    for run_id, run_segments in enumerate(runs):
        open_endpoints = _find_open_endpoints(run_segments)
        for seg in run_segments:
            seg_points = _sample_segment(seg, run_id, open_endpoints)
            surface_points.extend(seg_points)

    # 5. Deduplicate
    surface_points = _deduplicate_points(surface_points)

    return surface_points
