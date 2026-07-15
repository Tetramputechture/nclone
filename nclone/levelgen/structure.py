"""Structure descriptor — turn "sparse vs rich" from a vibe into a *measured* axis.

The #1 generation problem (``docs/SYNTHETIC_LEVELGEN_NEXT_STEPS.md`` §2) is that the
transformer emits *sparse open rooms with floating entities* instead of
*structurally-rich obstacle courses*. To fix that we first have to **measure** it: a
level is "rich" if it has internal platforms / ledges / corridors the ninja must
navigate, and "sparse" if it is a single bordered room.

This module reads the 42×23 tile grid (and, optionally, the free-space route) and
returns a :class:`StructureDescriptor` of cheap, tile-only structural features plus a
single ``structure_score`` in ``[0, 1]``. It is used by:

* the ACCEL mutation engine (``mutations.py``) — assert mutated children gain structural
  complexity (``docs`` §6 "does mutation actually produce *structure*?");
* the QD coverage archive (``scripts/levelgen/qd_archive.py``) — a discretised structure
  bin is one of the behaviour descriptors;
* generation monitoring — a ``sample/structure`` scalar that the W0 run was missing.

All features are derived purely from the tile grid (no sim load) so this is ~µs/level
and safe to call inside tight generate-and-gate loops.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

from .tokenizer import TILE_END, TILE_START

GRID_W = 42
GRID_H = 23

# Tile-type families (mirror ``labeling.py``).
_CURVE_TILES = frozenset({10, 11, 12, 13, 14, 15, 16, 17})
_SLOPE_TILES = frozenset(range(6, 10)) | frozenset(range(18, 34))


def tile_grid(map_data: Sequence[int]) -> List[List[int]]:
    """``GRID_H`` rows × ``GRID_W`` cols of tile ids from a full ``map_data``."""
    flat = list(map_data[TILE_START:TILE_END])
    return [flat[r * GRID_W : (r + 1) * GRID_W] for r in range(GRID_H)]


def grid_from_tiles(tiles: Sequence[int]) -> List[List[int]]:
    """``GRID_H`` rows × ``GRID_W`` cols from a bare 966-tile list."""
    flat = list(tiles)
    return [flat[r * GRID_W : (r + 1) * GRID_W] for r in range(GRID_H)]


@dataclass
class StructureDescriptor:
    """Measured structural richness of a level's tile geometry."""

    tile_fill: float           # non-empty fraction of the whole grid
    interior_fill: float       # non-empty fraction of the strict interior
    standable_surfaces: int    # interior solid tiles with empty directly above
    n_platform_components: int  # solid components touching no outer border (floating)
    n_air_components: int      # distinct connected air regions
    largest_air_frac: float    # largest air region / total air (1.0 ⇒ one open room)
    air_vertical_extent: int   # tile rows spanned by air
    air_horizontal_extent: int  # tile cols spanned by air
    slope_tiles: int
    curve_tiles: int
    climbable_walls: int       # tall interior walls with an adjacent empty face
    structure_score: float     # single richness scalar in [0, 1]
    is_open_room: bool         # heuristic "sparse single room" flag

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# --- richness-score normalisation constants (calibrated to corpus, not magic) ---
# Saturate each sub-feature at a value typical of a *rich* level so the score spreads
# the open-room↔obstacle-course axis rather than the long rich tail.
_STANDABLE_SAT = 40.0      # ~rich levels have many ledge tiles
_PLATFORM_SAT = 6.0        # several distinct floating structures
_WALL_SAT = 4.0
_GEOM_SAT = 30.0           # slope+curve tiles
_AIR_COMP_SAT = 4.0


def _occupied(t: int) -> bool:
    return t != 0


def _air_components(grid: List[List[int]]):
    """Return ``(n_components, largest_size, total_air, (min_r,max_r,min_c,max_c))``.

    4-connected flood over empty (``0``) tiles.
    """
    seen = [[False] * GRID_W for _ in range(GRID_H)]
    sizes = []
    total = 0
    minr, maxr, minc, maxc = GRID_H, -1, GRID_W, -1
    for sy in range(GRID_H):
        for sx in range(GRID_W):
            if grid[sy][sx] != 0 or seen[sy][sx]:
                continue
            stack = [(sx, sy)]
            seen[sy][sx] = True
            size = 0
            while stack:
                cx, cy = stack.pop()
                size += 1
                total += 1
                minr, maxr = min(minr, cy), max(maxr, cy)
                minc, maxc = min(minc, cx), max(maxc, cx)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < GRID_W and 0 <= ny < GRID_H and grid[ny][nx] == 0 and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            sizes.append(size)
    if not sizes:
        return 0, 0, 0, (0, 0, 0, 0)
    return len(sizes), max(sizes), total, (minr, maxr, minc, maxc)


def _platform_components(grid: List[List[int]]) -> int:
    """Count solid components that touch NO outer border tile (floating structures)."""
    seen = [[False] * GRID_W for _ in range(GRID_H)]
    n = 0
    for sy in range(GRID_H):
        for sx in range(GRID_W):
            if not _occupied(grid[sy][sx]) or seen[sy][sx]:
                continue
            stack = [(sx, sy)]
            seen[sy][sx] = True
            touches_border = False
            while stack:
                cx, cy = stack.pop()
                if cx in (0, GRID_W - 1) or cy in (0, GRID_H - 1):
                    touches_border = True
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < GRID_W and 0 <= ny < GRID_H and _occupied(grid[ny][nx]) and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            if not touches_border:
                n += 1
    return n


def _standable_surfaces(grid: List[List[int]]) -> int:
    """Interior solid tiles with an empty tile directly above (a place to stand)."""
    cnt = 0
    for gy in range(1, GRID_H - 1):
        for gx in range(1, GRID_W - 1):
            if _occupied(grid[gy][gx]) and grid[gy - 1][gx] == 0:
                cnt += 1
    return cnt


def _climbable_walls(grid: List[List[int]], run_min: int = 5, face_min: int = 3) -> int:
    """Count distinct interior columns with a tall solid run + adjacent empty face."""
    n = 0
    for gx in range(1, GRID_W - 1):
        run = face = 0
        counted = False
        for gy in range(1, GRID_H - 1):
            if _occupied(grid[gy][gx]):
                run += 1
                if grid[gy][gx - 1] == 0 or grid[gy][gx + 1] == 0:
                    face += 1
                if run >= run_min and face >= face_min and not counted:
                    n += 1
                    counted = True
            else:
                run = face = 0
                counted = False
    return n


def describe_tiles(tiles: Sequence[int]) -> StructureDescriptor:
    """Structure descriptor from a bare 966-tile grid."""
    return describe_grid(grid_from_tiles(tiles))


def describe_map_data(map_data: Sequence[int]) -> StructureDescriptor:
    """Structure descriptor from a full ``map_data``."""
    return describe_grid(tile_grid(map_data))


def describe_grid(grid: List[List[int]]) -> StructureDescriptor:
    flat = [t for row in grid for t in row]
    n_tiles = len(flat) or 1
    tile_fill = sum(1 for t in flat if t != 0) / n_tiles

    interior = [grid[gy][gx] for gy in range(1, GRID_H - 1) for gx in range(1, GRID_W - 1)]
    interior_fill = (sum(1 for t in interior if t != 0) / len(interior)) if interior else 0.0

    standable = _standable_surfaces(grid)
    n_plat = _platform_components(grid)
    n_air, largest_air, total_air, (minr, maxr, minc, maxc) = _air_components(grid)
    largest_air_frac = (largest_air / total_air) if total_air else 0.0
    v_extent = (maxr - minr + 1) if total_air else 0
    h_extent = (maxc - minc + 1) if total_air else 0

    slope_tiles = sum(1 for t in flat if t in _SLOPE_TILES)
    curve_tiles = sum(1 for t in flat if t in _CURVE_TILES)
    walls = _climbable_walls(grid)

    # Richness score: mean of normalised sub-features. Each ∈ [0,1]; an open bordered
    # room scores ~0 (no standable interior, no platforms, single air comp, no geom).
    parts = [
        min(standable / _STANDABLE_SAT, 1.0),
        min(n_plat / _PLATFORM_SAT, 1.0),
        min(walls / _WALL_SAT, 1.0),
        min((slope_tiles + curve_tiles) / _GEOM_SAT, 1.0),
        min((n_air - 1) / _AIR_COMP_SAT, 1.0),  # compartmentalisation
        1.0 - largest_air_frac,                 # corridors vs one open blob
    ]
    structure_score = sum(parts) / len(parts)

    # "Open room": almost-empty interior, one dominant air blob, no internal structure.
    is_open_room = (
        interior_fill < 0.06
        and n_plat == 0
        and standable <= 2
        and largest_air_frac > 0.85
    )

    return StructureDescriptor(
        tile_fill=round(tile_fill, 4),
        interior_fill=round(interior_fill, 4),
        standable_surfaces=standable,
        n_platform_components=n_plat,
        n_air_components=n_air,
        largest_air_frac=round(largest_air_frac, 4),
        air_vertical_extent=v_extent,
        air_horizontal_extent=h_extent,
        slope_tiles=slope_tiles,
        curve_tiles=curve_tiles,
        climbable_walls=walls,
        structure_score=round(structure_score, 4),
        is_open_room=is_open_room,
    )


def structure_bin(score: float, n_bins: int = 4) -> int:
    """Discretise ``structure_score`` ∈ [0,1] into ``0..n_bins-1`` (QD descriptor)."""
    if score <= 0:
        return 0
    return min(n_bins - 1, int(score * n_bins))


__all__ = [
    "StructureDescriptor",
    "describe_tiles",
    "describe_map_data",
    "describe_grid",
    "tile_grid",
    "grid_from_tiles",
    "structure_bin",
    "GRID_W",
    "GRID_H",
]
