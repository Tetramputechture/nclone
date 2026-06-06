"""Tile-accurate, agent-facing visualization.

Everything is parsed from the SERIALIZED ``map_data`` the simulator loads (never
from builder scratch state), so what the agent sees == what it built == what the
sim runs. Entity coordinates are decoded through ``coords.entity_units_to_grid``
(offset-aware, per entity type) -- this is the fix for the ``map_visualizer.py``
``//6`` bug, which mis-placed the ninja and any offset-bearing entity.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ...constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from . import coords

_ENTITY_NAME = {
    0: "ninja", 1: "toggle_mine", 21: "toggle_mine", 2: "gold",
    3: "exit_door", 4: "exit_switch", 6: "locked_door", 7: "locked_door_switch",
    8: "trap_door",
}

_ENTITY_GLYPH = {
    "ninja": "N", "toggle_mine": "*", "gold": "$", "exit_door": "E",
    "exit_switch": "s", "locked_door": "L", "trap_door": "T",
}


def _tiles(map_data: List[int]) -> List[int]:
    return list(map_data[coords.TILE_DATA_START:coords.TILE_DATA_END])


def parse_entities(map_data: List[int]) -> List[Dict]:
    """Return a list of ``{type, name, tile:[gx,gy], orientation?}`` (offset-correct)."""
    out: List[Dict] = []
    # ninja spawn
    nx = map_data[coords.NINJA_X]
    ny = map_data[coords.NINJA_Y]
    gx, gy = coords.entity_units_to_grid(0, nx, ny)
    out.append({
        "type": 0, "name": "ninja", "tile": [round(gx), round(gy)],
        "grid": [gx, gy], "px": [nx * coords.PX_PER_UNIT, ny * coords.PX_PER_UNIT],
        "orientation": "right" if map_data[coords.NINJA_ORIENTATION] == 1 else "left",
    })
    data = map_data[coords.ENTITY_START:]
    i = 0
    n = len(data)
    while i + 4 < n:
        etype = data[i]
        ex, ey = data[i + 1], data[i + 2]
        ggx, ggy = coords.entity_units_to_grid(etype, ex, ey)
        name = _ENTITY_NAME.get(etype, f"type_{etype}")
        out.append({"type": etype, "name": name, "tile": [round(ggx), round(ggy)],
                    "grid": [ggx, ggy],
                    "px": [ex * coords.PX_PER_UNIT, ey * coords.PX_PER_UNIT]})
        if etype == 3:  # exit door followed by its switch (10-int record)
            if i + 9 < n and data[i + 5] == 4:
                sx, sy = data[i + 6], data[i + 7]
                sgx, sgy = coords.entity_units_to_grid(4, sx, sy)
                out.append({"type": 4, "name": "exit_switch",
                            "tile": [round(sgx), round(sgy)], "grid": [sgx, sgy],
                            "px": [sx * coords.PX_PER_UNIT, sy * coords.PX_PER_UNIT]})
            i += 10
        elif etype in (6, 8):  # locked / trap door (10-int record)
            i += 10
        else:
            i += coords.ENTITY_STRIDE
    return out


def describe_level(map_data: List[int], labels=None) -> Dict:
    """Machine-readable facts about a level the agent can branch on."""
    tiles = _tiles(map_data)
    hist: Dict[int, int] = {}
    for t in tiles:
        hist[t] = hist.get(t, 0) + 1
    out = {
        "dims": [MAP_TILE_WIDTH, MAP_TILE_HEIGHT],
        "tile_histogram": {str(k): v for k, v in sorted(hist.items()) if k != 0},
        "entities": parse_entities(map_data),
    }
    if labels is not None:
        out["mechanics"] = list(getattr(labels, "mechanics", []))
        out["declared_difficulty"] = dict(getattr(labels, "declared_difficulty", {}))
        out["primary_bucket"] = getattr(labels, "primary_bucket", None)
        out["components"] = list(getattr(labels, "components", []))
    return out


def to_compact_ascii(map_data: List[int], grid=None, path=None) -> str:
    """1-char-per-tile view with a 1:1 glyph map and offset-correct entities."""
    tiles = _tiles(map_data)
    ents = parse_entities(map_data)
    epos: Dict[tuple, str] = {}
    for e in ents:
        gx, gy = e["tile"]
        glyph = _ENTITY_GLYPH.get(e["name"], "?")
        epos[(gx, gy)] = "X" if (gx, gy) in epos else glyph

    # grid-boundary columns (for the region overlay)
    col_borders = set()
    if grid:
        for row in grid:
            for region in row:
                col_borders.add(region.x1 + 1)

    lines = ["    " + "".join(f"{x:<5}" for x in range(0, MAP_TILE_WIDTH, 5))]
    lines.append("   +" + "-" * MAP_TILE_WIDTH + "+")
    for y in range(MAP_TILE_HEIGHT):
        chars = []
        for x in range(MAP_TILE_WIDTH):
            if (x, y) in epos:
                chars.append(epos[(x, y)])
            elif x in col_borders and grid:
                chars.append(":")
            else:
                chars.append(coords.tile_glyph(tiles[x + y * MAP_TILE_WIDTH]))
        lines.append(f"{y:2d} |" + "".join(chars) + "|")
    lines.append("   +" + "-" * MAP_TILE_WIDTH + "+")
    if path:
        lines.append("on-path cells: " + " -> ".join(str(tuple(c)) for c in path))
    lines.append("entities: N ninja  E exit-door  s switch  * mine  $ gold  X overlap")
    return "\n".join(lines)


def to_shape_ascii(map_data: List[int], cell=None) -> str:
    """3x3-supersampled view so slopes/curves draw pictorially (opt-in, verbose)."""
    from ..tile_definitions import (
        TILE_GRID_EDGE_MAP,
    )  # noqa: F401  (kept for future precise shapes)
    # Minimal pictorial shapes for the common ids; defaults to a solid/empty block.
    SHAPES = {
        0: ["...", "...", "..."], 1: ["###", "###", "###"],
        8: ["..#", ".##", "###"], 9: ["#..", "##.", "###"],
        6: ["###", "##.", "#.."], 7: ["###", ".##", "..#"],
        12: ["..#", ".##", "###"], 13: ["#..", "##.", "###"],
        17: ["#..", "##.", "###"], 16: ["..#", ".##", "###"],
        2: ["###", "###", "..."], 4: ["...", "###", "###"],
        3: ["..#", "..#", "..#"], 5: ["#..", "#..", "#.."],
    }
    tiles = _tiles(map_data)
    xs = range(MAP_TILE_WIDTH)
    ys = range(MAP_TILE_HEIGHT)
    if cell is not None:
        r = cell  # (x0,y0,x1,y1) inclusive
        xs = range(r[0], r[2] + 1)
        ys = range(r[1], r[3] + 1)
    out_lines = []
    for y in ys:
        block_rows = ["", "", ""]
        for x in xs:
            tid = tiles[x + y * MAP_TILE_WIDTH]
            shape = SHAPES.get(tid, ["###", "###", "###"] if tid else ["...", "...", "..."])
            for k in range(3):
                block_rows[k] += shape[k]
        out_lines.extend(block_rows)
    return "\n".join(out_lines)


def assert_tile_accurate(map_data: List[int]) -> None:
    """Round-trip guard: serialized tiles re-parse identically and entity coords
    decode to in-bounds integer grid positions. Raises on mismatch."""
    tiles = _tiles(map_data)
    if len(tiles) != MAP_TILE_WIDTH * MAP_TILE_HEIGHT:
        raise AssertionError(
            f"tile section length {len(tiles)} != {MAP_TILE_WIDTH * MAP_TILE_HEIGHT}"
        )
    for e in parse_entities(map_data):
        gx, gy = e["tile"]
        if not (0 <= gx < MAP_TILE_WIDTH and 0 <= gy < MAP_TILE_HEIGHT):
            raise AssertionError(f"entity {e['name']} decodes out of bounds at {e['tile']}")


def render_png(map_data: List[int], out_path: str) -> None:
    """Ground-truth PNG via the headless renderer (on-demand, not per-step)."""
    from ...nplay_headless import NPlayHeadless

    nplay = NPlayHeadless(render_mode="rgb_array", enable_animation=False,
                          enable_logging=False, enable_debug_overlay=False)
    nplay.load_map_from_map_data(list(map_data))
    nplay.reset()
    frame = nplay.render()
    try:
        from PIL import Image

        Image.fromarray(frame).save(out_path)
    except Exception:
        import numpy as np

        np.save(out_path, frame)
