"""Heuristic labels for ingested levels: subset validity, mechanic tags, difficulty.

These labels are *conditioning signal* for the generator, not ground truth. The
mechanic tagger is a cheap tile/entity-composition heuristic aligned with the
compose-registry domains (jump / wall_jump / mine / slope / curve / composite /
precision / flat). Difficulty uses the winning-demo frame length as a free proxy
(see the plan); exact lattice ``d*`` is reserved for calibration + the Phase-2 gate.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..constants.entity_types import MINIMAL_TRAINING_SUBSET, PHANTOM_ENTITY_TYPES
from .tokenizer import ENT_START, ENT_STRIDE, TILE_END, TILE_START, MECHANIC_TAGS

# tile-type families (see nclone tile semantics)
_CURVE_TILES = frozenset({10, 11, 12, 13, 14, 15, 16, 17})
_SLOPE_TILES = frozenset(range(6, 10)) | frozenset(range(18, 34))  # 6-9, 18-33
_SOLID_TILES = frozenset({1, 2, 3, 4, 5})  # full + half tiles count as wall-able
_MINE_TYPES = frozenset({1, 21})
GRID_W = 42
GRID_H = 23


def _tile_grid(map_data: Sequence[int]) -> List[List[int]]:
    flat = list(map_data[TILE_START:TILE_END])
    return [flat[r * GRID_W : (r + 1) * GRID_W] for r in range(GRID_H)]


def parse_entities(map_data: Sequence[int]) -> List[Tuple[int, int, int, int, int]]:
    ents = []
    i = ENT_START
    n = len(map_data)
    while i + ENT_STRIDE <= n:
        t = map_data[i]
        if t == 0:
            break
        ents.append(tuple(map_data[i : i + ENT_STRIDE]))
        i += ENT_STRIDE
    return ents


def entity_counts(map_data: Sequence[int]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for e in parse_entities(map_data):
        counts[e[0]] = counts.get(e[0], 0) + 1
    return counts


def subset_ok(
    map_data: Sequence[int], allowed: Optional[Iterable[int]] = None
) -> bool:
    """Entities are within ``allowed``, and there is exactly 1 exit door + 1 switch.

    ``allowed`` defaults to :data:`MINIMAL_TRAINING_SUBSET` -- the *contract* a
    procedural generator declares about its own output, not a statement about
    what the simulator can run (which is much wider; see
    :data:`NATIVE_IMPLEMENTED_TYPES`). Callers ingesting **human** levels should
    pass the wider profile explicitly; callers gating **generated** levels should
    keep the default, since a narrow contract is the point there.
    """
    allowed_set = MINIMAL_TRAINING_SUBSET if allowed is None else frozenset(allowed)
    counts = entity_counts(map_data)
    # A locked/trap door writes a second record carrying its switch coordinates,
    # tagged with the phantom type 7/9. No entity is ever constructed for those,
    # so they are not a capability question -- but counting them as ordinary
    # entities rejects every door level, which is what kept the packable corpus
    # at ~7% even after the allowed-set was widened.
    if set(counts) - allowed_set - PHANTOM_ENTITY_TYPES:
        return False
    return counts.get(3, 0) == 1 and counts.get(4, 0) == 1


def _max_vertical_solid_run(grid: List[List[int]]) -> int:
    """Longest run of solid tiles in any *interior* column (border frame excluded).

    NOTE: on dense human levels this saturates at the interior height, so it is a
    poor wall-jump signal -- use :func:`_climbable_wall` instead. Kept for callers.
    """
    best = 0
    for gx in range(2, GRID_W - 2):
        run = 0
        for gy in range(2, GRID_H - 2):
            if grid[gy][gx] in _SOLID_TILES:
                run += 1
                best = max(best, run)
            else:
                run = 0
    return best


# Salience thresholds, calibrated to the subset corpus distribution (slope p75~56,
# curve p90~24, mines p75~44). Goal: tags fire on the MINORITY of levels where the
# mechanic dominates, so the conditioning signal actually partitions the data
# (the old presence-based tagger labelled ~98% jump / ~96% composite -> no signal).
_SLOPE_MIN = 40      # slope-heavy levels (~top 30%)
_CURVE_MIN = 8       # curves are rare; ~top 20%
_PRECISION_MIN = 44  # dense-mine precision levels (~top 25%)
_WALLJUMP_RUN = 8    # tall wall ...
_WALLJUMP_FACE = 5   # ... with an adjacent empty face (climbable)
_PLATFORM_MIN = 6    # floating platforms => real jump structure
_FLAT_DENSITY = 0.25


def _climbable_wall(grid: List[List[int]]) -> bool:
    """A tall interior solid column with an adjacent empty face (wall-jumpable)."""
    for gx in range(2, GRID_W - 2):
        run = face = 0
        for gy in range(2, GRID_H - 2):
            if grid[gy][gx] in _SOLID_TILES:
                run += 1
                if grid[gy][gx - 1] == 0 or grid[gy][gx + 1] == 0:
                    face += 1
                if run >= _WALLJUMP_RUN and face >= _WALLJUMP_FACE:
                    return True
            else:
                run = face = 0
    return False


def _floating_platforms(grid: List[List[int]]) -> int:
    """Count solid cells with empty directly above AND below (jump-between platforms)."""
    cnt = 0
    for gy in range(3, GRID_H - 3):
        for gx in range(2, GRID_W - 2):
            if grid[gy][gx] in _SOLID_TILES and grid[gy - 1][gx] == 0 and grid[gy + 1][gx] == 0:
                cnt += 1
    return cnt


def mechanic_tags(map_data: Sequence[int]) -> List[str]:
    """Salience-based composition tags (each fires when its mechanic dominates).

    Always returns >=1 tag. Designed so the tag distribution partitions the corpus
    (vs the old presence-based tagger that labelled nearly everything composite).
    """
    grid = _tile_grid(map_data)
    counts = entity_counts(map_data)
    n_mines = sum(counts.get(t, 0) for t in _MINE_TYPES)
    flat_tiles = [t for row in grid for t in row]
    slope_count = sum(1 for t in flat_tiles if t in _SLOPE_TILES)
    curve_count = sum(1 for t in flat_tiles if t in _CURVE_TILES)
    density = sum(1 for t in flat_tiles if t != 0) / max(len(flat_tiles), 1)

    tags: List[str] = []
    if n_mines > 0:
        tags.append("mine")
    if slope_count >= _SLOPE_MIN:
        tags.append("slope")
    if curve_count >= _CURVE_MIN:
        tags.append("curve")
    if _climbable_wall(grid):
        tags.append("wall_jump")
    if _floating_platforms(grid) >= _PLATFORM_MIN:
        tags.append("jump")
    if n_mines >= _PRECISION_MIN:
        tags.append("precision")

    # composite only when several GEOMETRY domains are jointly salient
    geo = {t for t in tags if t in ("jump", "wall_jump", "slope", "curve")}
    if len(geo) >= 3:
        tags.append("composite")
    if not tags or (not geo and density < _FLAT_DENSITY):
        tags.append("flat")
    return [t for t in MECHANIC_TAGS if t in set(tags)]


def difficulty_edges(demo_lens: Sequence[int], n_bands: int = 5) -> List[float]:
    """Return the ``n_bands-1`` internal quantile edges over positive demo lengths."""
    vals = sorted(v for v in demo_lens if v and v > 0)
    if not vals:
        return []
    edges = []
    for k in range(1, n_bands):
        q = k / n_bands
        idx = min(len(vals) - 1, int(q * len(vals)))
        edges.append(float(vals[idx]))
    return edges


def assign_band(demo_len: int, edges: Sequence[float]) -> str:
    """Map a demo length to ``unknown`` (no demo) or ``d0``..``d{n}``."""
    if not demo_len or demo_len <= 0 or not edges:
        return "unknown"
    b = 0
    for e in edges:
        if demo_len > e:
            b += 1
        else:
            break
    return f"d{b}"


__all__ = [
    "subset_ok",
    "mechanic_tags",
    "entity_counts",
    "parse_entities",
    "difficulty_edges",
    "assign_band",
]
