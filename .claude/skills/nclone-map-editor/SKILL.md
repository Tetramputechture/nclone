---
name: nclone-map-editor
description: Build an nclone level FROM SCRATCH with direct single + batch tile/entity tools, then test that it is actually playable. Use when asked to make a specific or pictorial level ("a level that looks like a smiley face", "draw an arrow / the letter A / a castle"), hand-place tiles/walls/mines/the player/the exit, build a level from an ASCII picture, or verify a hand-built map is solvable. Complements the nclone-map-author skill (structured templates) and nclone-map-render skill (images).
---

# Build an nclone level from scratch (MapEditor)

`MapEditor` (in `nclone/map_generation/compose/editor.py`) is the direct authoring
surface: place individual or batched tiles/entities, draw shapes or an ASCII
picture, then `check()` that the level is buildable AND solvable. Use the project
venv: `source .venv/bin/activate`.

Grid: integer tiles, x in [0,41], y in [0,22], (0,0) top-left, y DOWN. Out-of-bounds
writes are skipped. Player/exit placement is idempotent (re-calling overwrites).

## The loop: draw -> check -> render -> fix

```python
from nclone.map_generation.compose import MapEditor

ed = MapEditor(fill="empty")          # or fill="solid" to carve out of rock
ed.border()                            # solid frame
# ... draw your level ...
rep = ed.check()                       # structured buildability + solvability report
print(rep["ok"], rep["solvable"], rep["errors"], rep["warnings"])
ed.render("/tmp/level.png")            # schematic image (see nclone-map-render)
print(ed.to_ascii())                   # terminal view
```

`check()` returns `{ok, solvable, errors[], warnings[], counts{}, entities[],
positions[]}`. A level is `ok` iff there are **no hard errors AND it is solvable**.
Iterate until `ok` is True. Hard errors include: not exactly one exit door / exit
switch / player; player or switch/door buried in a solid tile; not solvable (the
exit switch+door must be reachable from the spawn). Warnings include mines embedded
in solid, sitting on the spawn, **stacked on each other (<6px apart), or overlapping
an objective** — read these to self-correct *before* you spend a render. Each entry
in `positions[]` gives `{name, tile, grid (sub-tile float), px (world pixels)}`, so
you can verify exact placement in text.

## Entity placement: tiles are 24px, but entities live on a finer 6px lattice

This is the #1 source of "my mines look misaligned/sparse." A tile is 24px;
entities are stored in integer map-units where **1 unit = 6px = a quarter tile**
(the finest byte-safe step — *not* 1px). `add_mine`/`add_gold` accept floats and
snap to that lattice. By default they bias to the **tile centre**
(`anchor="center"`), because a bare integer grid coord places the entity on the
tile *corner* (the seam between four tiles), which reads up-and-left of intent:

- `ed.add_mine(21, 9)` → centre of tile (21,9), world px (516,228). **Use this.**
- `ed.add_mine(21, 9, anchor="corner")` → tile corner px (504,216) — the old trap.
- `ed.add_mine(21.5, 9.25, anchor="corner")` → absolute sub-tile point (fine nudges).

**Spatial anchors** (import from the same package) save you the arithmetic:
`MIDDLE`/`CENTER`, `TOP_MIDDLE`, `BOTTOM_MIDDLE`, `LEFT_MIDDLE`, `RIGHT_MIDDLE`,
the four corners, plus `above/below/left_of/right_of(point, n=1)`. So
`ed.add_mine(*MIDDLE)` means "a mine in the middle of the map" literally.

**Fill a *shape* with mines** instead of hand-listing dots — a "nose", a filled
letter, a hazard blob:
- `ed.fill_mines(x0,y0,x1,y1, spacing_px=12, mask=lambda gx,gy: ...)` — tiles a
  region on the sub-tile lattice; `mask` restricts to any shape; solid tiles are
  skipped (`on_empty_only=True`). `spacing_px` 12 = every half-tile, 6 = every
  quarter-tile (denser).
- `ed.mine_disk(cx, cy, r, spacing_px=12)` — a filled circular cluster of radius
  `r` tiles.

## Tools

**Single:** `set_tile(x,y,tile=1)`, `erase(x,y)`, `place_player(x,y,facing="right")`,
`place_exit(switch=(x,y), door=(x,y))`, `add_mine(x,y,anchor="center")`,
`add_gold(x,y,anchor="center")` (entities snap to the 6px lattice — see below).

**Batch tiles:** `fill_rect / clear_rect / rect_outline(x0,y0,x1,y1,tile)`,
`border(tile)`, `hline(y,x0,x1) / vline(x,y0,y1)`, `line(x0,y0,x1,y1)` (Bresenham),
`disk(cx,cy,r)` (filled circle), `ring(cx,cy,r,thickness=)` (circle outline),
`arc(cx,cy,r,a0_deg,a1_deg)` (0°=east, angles go clockwise since y is down),
`set_tiles([(x,y),...])`.

**Batch entities:** `add_mines([(x,y),...])`, `fill_mines(x0,y0,x1,y1,spacing_px=,
mask=)` and `mine_disk(cx,cy,r,spacing_px=)` (shape-fill, see below),
`scatter_mines(x0,y0,x1,y1,n, on_empty_only=True)` (seeded, deterministic).

Tile ids: 0 empty, 1 solid; slopes/curves available (8/9 diag, 12/13 hump,
16/17 bowl, 18-33 mild/steep) — but for pictures you usually only need 0 and 1.
All shape methods return `self`, so calls chain.

## Pictorial levels via ASCII art (the "looks like X" entry point)

Write a picture and a legend mapping each char to a tile id / name / entity token
(`player`, `mine`, `gold`, `exit_door`, `exit_switch`). The door+switch are paired
automatically when both appear. Unlisted chars are left untouched.

```python
ed = MapEditor(fill="empty")
art = '''
#########
#P..*..E#
#..###..#
#..#S#..#
#########
'''
ed.draw_ascii(art, {"#": "solid", ".": "empty", "*": "mine",
                    "P": "player", "E": "exit_door", "S": "exit_switch"},
              origin=(2, 2))
assert ed.check()["ok"]
```

## Worked example: "a level that looks like a smiley face"

Draw the face as decoration, THEN add a reachable floor + spawn/switch/door so it
is actually playable (a pure picture has no path and fails `check()`):

```python
from nclone.map_generation.compose import MapEditor
ed = MapEditor(fill="empty")
ed.border()
cx, cy = 21, 9
ed.ring(cx, cy, 7, tile=1, thickness=1.5)        # head
ed.disk(cx-4, cy-2, 1); ed.disk(cx+4, cy-2, 1)   # eyes (wide-set: room for a nose)
ed.arc(cx, cy, 4, 25, 155, tile=1)               # smiling mouth (lower arc)
# nose = a dense, CENTERED mine cluster — let fill_mines tile a shape mask,
# don't hand-list dots (they land on tile corners and read sparse/misaligned):
top, bot = cy-1.5, cy+1.5
ed.fill_mines(cx-1.5, top, cx+1.5, bot, spacing_px=8,
              mask=lambda gx, gy: abs(gx-cx) <= 0.15 + (gy-top)/(bot-top)*1.05)
ed.hline(20, 1, 40, tile=1)                       # playable floor
ed.place_player(3, 19, facing="right")
ed.place_exit(switch=(33, 19), door=(38, 19))
rep = ed.check()                                  # -> ok=True, solvable=True
ed.render("/tmp/smiley.png", backend="ground_truth", title="smiley face")
```

## Save / load a map

```python
ed.save("levels/smiley.map")                       # raw N++ bytes -> sim/env can load it directly
ed.save("levels/smiley.json", metadata={"author": "claude"})   # human-readable, keeps metadata
ed.save("levels/smiley.pkl", metadata={"split": "train"}, secondary=["hazard:mine.field"])  # dataset-shaped

from nclone.map_generation.compose import load_map_data, load_map
md  = load_map_data("levels/smiley.map")           # back to a map_data list
m   = load_map("levels/smiley.map")                # back to a Map
```

Format is inferred from the extension (`.map`/`.bin`/none = **binary**, `.json`,
`.pkl`) or forced with `fmt=`. **Binary is what the simulator loads**
(`NPlayHeadless.load_map(path)`), so save `.map` for anything you want to play/train
on; use `.json` to inspect/diff; use `.pkl` (dataset record shape: `map_data` +
`secondary` + `metadata`) to feed the benchmark/loader path. Binary refuses values
outside 0-255 (use json if you somehow have non-byte data). `save_map(map_data, path,
...)` / `load_map_data(path)` are the standalone functions if you don't have an editor.

## Notes

- `check()`'s solvability uses the grid traversability graph (jump/fall aware,
  same predicate as the physics/planner) — it is the real "is this beatable?"
  test, not a plain flood-fill. Decorative geometry that seals off an upper region
  is fine as long as the spawn->switch->door path is reachable.
- Entities only allowed for Layer-1 maps are the player, toggle mines (type 1),
  and the exit door+switch — keep to those for training compatibility.
- Output: `ed.map_data()` (sim-ready), `ed.to_map()` (a `Map`), `ed.to_ascii()`,
  `ed.describe()` (offset-correct facts), `ed.render(path, backend="schematic"|"ground_truth")`.
- Reference: `nclone/map_generation/compose/{editor,validate}.py`; tests in
  `tests/test_compose_editor.py`.
