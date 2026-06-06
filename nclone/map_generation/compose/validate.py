"""Buildability / playability check for a from-scratch map.

``check_level(map_data)`` returns a structured report an agent can branch on:
hard ``errors`` (no/duplicate objective, player or switch buried in solid),
soft ``warnings`` (mine embedded in solid, mine on top of spawn), entity
``counts``, and whether the level is ``solvable`` (the exit switch AND door are
reachable from the ninja spawn via the grid traversability graph). ``ok`` is True
iff there are no hard errors and the level is solvable.
"""

from __future__ import annotations

from typing import Dict, List

from ...constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from .visualize import _tiles, parse_entities

FULL_SOLID = 1


def _tile_at(tiles: List[int], gx: int, gy: int) -> int:
    if 0 <= gx < MAP_TILE_WIDTH and 0 <= gy < MAP_TILE_HEIGHT:
        return tiles[gx + gy * MAP_TILE_WIDTH]
    return -1


def check_level(map_data: List[int]) -> Dict:
    tiles = _tiles(map_data)
    ents = parse_entities(map_data)
    by = {}
    for e in ents:
        by.setdefault(e["name"], []).append(e)

    counts = {
        "player": len(by.get("ninja", [])),
        "exit_door": len(by.get("exit_door", [])),
        "exit_switch": len(by.get("exit_switch", [])),
        "mine": len(by.get("toggle_mine", [])),
        "gold": len(by.get("gold", [])),
    }
    errors: List[str] = []
    warnings: List[str] = []

    if counts["exit_door"] != 1:
        errors.append(f"need exactly 1 exit door, found {counts['exit_door']}")
    if counts["exit_switch"] != 1:
        errors.append(f"need exactly 1 exit switch, found {counts['exit_switch']}")
    if counts["player"] != 1:
        errors.append(f"need exactly 1 player spawn, found {counts['player']}")

    # buried-in-solid checks
    def buried(e):
        gx, gy = e["tile"]
        return _tile_at(tiles, gx, gy) == FULL_SOLID

    for ninja in by.get("ninja", []):
        gx, gy = ninja["tile"]
        if gx <= 0 or gy <= 0 or gx >= MAP_TILE_WIDTH - 1 or gy >= MAP_TILE_HEIGHT - 1:
            warnings.append(f"player spawn {ninja['tile']} is on/over the border")
        if buried(ninja):
            errors.append(f"player spawn {ninja['tile']} is inside a solid tile")
    for sw in by.get("exit_switch", []):
        if buried(sw):
            errors.append(f"exit switch {sw['tile']} is inside a solid tile (unreachable)")
    for dr in by.get("exit_door", []):
        if buried(dr):
            errors.append(f"exit door {dr['tile']} is inside a solid tile (unreachable)")

    # soft: mines embedded in solid (pointless) or sitting on the spawn
    spawn = by.get("ninja", [{}])[0].get("tile") if by.get("ninja") else None
    for mine in by.get("toggle_mine", []):
        if buried(mine):
            warnings.append(f"mine {mine['tile']} is inside a solid tile (no effect)")
        if spawn and abs(mine["tile"][0] - spawn[0]) <= 1 and abs(mine["tile"][1] - spawn[1]) <= 1:
            warnings.append(f"mine {mine['tile']} is adjacent to the spawn (instant death risk)")

    # sub-tile overlap checks (entities live on a 6px lattice, finer than tiles):
    # stacked mines waste a placement; a mine on an objective/spawn can corrupt it.
    def _px(e):
        return e.get("px", [e["tile"][0] * 24, e["tile"][1] * 24])

    mines = by.get("toggle_mine", [])
    for a in range(len(mines)):
        for b in range(a + 1, len(mines)):
            pa, pb = _px(mines[a]), _px(mines[b])
            if abs(pa[0] - pb[0]) < 6 and abs(pa[1] - pb[1]) < 6:
                warnings.append(
                    f"mines at {mines[a]['grid']} and {mines[b]['grid']} are stacked "
                    f"(<6px apart) -- they render as one")
    objectives = (by.get("ninja", []) + by.get("exit_switch", [])
                  + by.get("exit_door", []))
    for mine in mines:
        mp = _px(mine)
        for obj in objectives:
            op = _px(obj)
            if abs(mp[0] - op[0]) < 12 and abs(mp[1] - op[1]) < 12:
                warnings.append(
                    f"mine {mine['grid']} overlaps {obj['name']} {obj.get('grid', obj['tile'])} "
                    f"(<12px) -- likely unintended")

    # solvability (only meaningful once the objective pair is well-formed)
    solvable = None
    if not any("exit" in e for e in errors) and counts["exit_door"] == 1 and counts["exit_switch"] == 1:
        try:
            from ..solvability import is_map_solvable

            solvable = bool(is_map_solvable(map_data))
            if not solvable:
                errors.append("not solvable: exit switch/door not reachable from spawn")
        except Exception as exc:  # pragma: no cover
            warnings.append(f"solvability check skipped: {exc}")

    ok = not errors and solvable is True
    return {
        "ok": ok,
        "solvable": solvable,
        "errors": errors,
        "warnings": warnings,
        "counts": counts,
        "entities": ents,
        "positions": [
            {"name": e["name"], "tile": e["tile"],
             "grid": e.get("grid", e["tile"]), "px": e.get("px")}
            for e in ents
        ],
    }
