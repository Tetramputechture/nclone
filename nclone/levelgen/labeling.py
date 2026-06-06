"""Heuristic labels for ingested levels: subset validity, mechanic tags, difficulty.

These labels are *conditioning signal* for the generator, not ground truth. The
mechanic tagger is a cheap tile/entity-composition heuristic aligned with the
compose-registry domains (jump / wall_jump / mine / slope / curve / composite /
precision / flat). Difficulty uses the winning-demo frame length as a free proxy
(see the plan); exact lattice ``d*`` is reserved for calibration + the Phase-2 gate.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

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


def subset_ok(map_data: Sequence[int]) -> bool:
    """Entities subset of {mine,gold,exit_door,exit_switch,toggled_mine}, 1 door+1 switch."""
    allowed = {1, 2, 3, 4, 21}
    counts = entity_counts(map_data)
    if set(counts) - allowed:
        return False
    return counts.get(3, 0) == 1 and counts.get(4, 0) == 1


def _max_vertical_solid_run(grid: List[List[int]]) -> int:
    """Longest run of solid tiles in any *interior* column (border frame excluded)."""
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


def mechanic_tags(map_data: Sequence[int]) -> List[str]:
    """Coarse composition-based tags. Always returns >=1 tag."""
    grid = _tile_grid(map_data)
    counts = entity_counts(map_data)
    n_mines = sum(counts.get(t, 0) for t in _MINE_TYPES)

    tile_vals = {t for row in grid for t in row}
    tags: List[str] = []
    if n_mines > 0:
        tags.append("mine")
    if tile_vals & _SLOPE_TILES:
        tags.append("slope")
    if tile_vals & _CURVE_TILES:
        tags.append("curve")
    if _max_vertical_solid_run(grid) >= 4:
        tags.append("wall_jump")
    # a non-trivial interior floor with empty space implies jumping
    interior_solid = sum(
        1 for gy in range(2, GRID_H - 2) for gx in range(2, GRID_W - 2)
        if grid[gy][gx] in _SOLID_TILES
    )
    interior_cells = (GRID_H - 4) * (GRID_W - 4)
    if 0 < interior_solid < interior_cells:
        tags.append("jump")
    if n_mines >= 8:
        tags.append("precision")

    # composite if it spans several distinct mechanic domains
    domains = {t for t in tags if t in ("jump", "wall_jump", "mine", "slope", "curve")}
    if len(domains) >= 3:
        tags.append("composite")
    if not tags:
        tags.append("flat")
    # de-dup, preserve registry order
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
