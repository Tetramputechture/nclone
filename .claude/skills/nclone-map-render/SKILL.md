---
name: nclone-map-render
description: Export an nclone map to a verification image showing tiles, the player, mines, and the exit door + exit switch. Use when asked to render/visualize/preview/screenshot a map or level, export a map to a PNG/image, or visually verify a generated map. Works from a composable template name, a natural-language phrase, a dataset .pkl, or a raw map_data list. Two backends — a labelled matplotlib schematic (default) and the pixel-faithful headless game frame.
---

# Render an nclone map to a verification image

Produces a PNG that makes the four things you usually need to verify unambiguous:
**tiles** (solid terrain shaded with true slope/curve shapes), the **player**, the
**mines**, the **exit door**, and the **exit switch** — each a distinct, legended
marker. Run with the project venv: `source .venv/bin/activate`.

## CLI (preferred — one command)

```bash
# from a composable template
python -m nclone.map_generation.compose.render_image --template mixed_traverse_2x3 --seed 0 --out /tmp/map.png

# from a natural-language phrase
python -m nclone.map_generation.compose.render_image --phrase "steep hill then two gaps and mines" --seed 2 --out /tmp/map.png

# from a dataset pickle (dict with map_data, or a raw list)
python -m nclone.map_generation.compose.render_image --pickle path/to/level.pkl --out /tmp/map.png

# from a JSON map_data file, or a binary .map file (raw N++ bytes the sim loads)
python -m nclone.map_generation.compose.render_image --json path/to/map_data.json --out /tmp/map.png
python -m nclone.map_generation.compose.render_image --map  path/to/level.map     --out /tmp/map.png
```

Flags: `--seed N`, `--backend schematic|ground_truth` (default `schematic`),
`--out PATH`, `--save PATH` (also persist the resolved map as `.map`/`.json`/`.pkl`),
`--no-facts` (skip the printed summary). Unless `--no-facts`, it also
prints `describe_level` (dims, tile histogram, **offset-correct** entity grid
coords) and `is_map_solvable`. After writing the PNG, **open/Read the image to
confirm** the player, mines, switch and door are where you intended.

## Python API

```python
from nclone.map_generation.compose.render_image import render_schematic, render_ground_truth

render_schematic(map_data, "/tmp/map.png", title="my level")   # labelled schematic (default)
render_ground_truth(map_data, "/tmp/map.png")                  # exact game frame (GL)
```

## Which backend?

- **schematic** (default): fast, dependency-light matplotlib. Tile solidity is
  sampled from the AUTHORITATIVE collision predicate
  (`graph_builder._check_subnode_validity_simple`) — the same geometry the physics
  sim uses — so slopes/curves/half-tiles draw with their true shape and there is no
  renderer y-flip ambiguity. Entities get a labelled legend (player + facing arrow,
  mine = red ✕, exit door = green ■, exit switch = purple ★, gold = yellow ●).
  Best for "did I build the right thing?".
- **ground_truth**: the exact 600×1056 game frame via the headless renderer
  (`NPlayHeadless`, `render_mode="rgb_array"`). Pixel-faithful to what the env shows.
  Works headlessly here despite harmless `libEGL`/`MESA` warnings.

## Terminal-only / structured alternatives (no image)

```python
from nclone.map_generation.compose import to_compact_ascii, describe_level
print(to_compact_ascii(map_data))     # 1-char-per-tile, 1:1 glyphs, correct entity coords
describe_level(map_data)              # machine-readable facts to branch on
```

`Template.preview(seed)` returns the compact ASCII + `describe_level` JSON in one call.

## Saving the map

```python
from nclone.map_generation.compose import save_map, load_map_data
save_map(map_data, "level.map")    # binary -> directly loadable by NPlayHeadless.load_map / the env
save_map(map_data, "level.json", metadata={...})   # inspectable
save_map(map_data, "level.pkl",  metadata={...}, secondary=[...])  # dataset record shape
md = load_map_data("level.map")
```
Or from the CLI add `--save level.map` to any render command. Format is inferred
from the extension (`.map`/`.bin`/none=binary, `.json`, `.pkl`).

## Notes / gotchas

- Entity coordinates are decoded **offset-aware per type** (ninja/door/switch carry
  a +6-unit store offset, mines carry 0) via `compose.coords` — this is the fix for
  the old `map_visualizer.py` `// 6` bug, so markers land on the correct tile.
- Off-path decoration mines may appear on sealed upper shelves; they are intentional
  unreachable distractors, not a solvability problem (`is_map_solvable` confirms).
- Reference: `nclone/map_generation/compose/render_image.py`,
  `nclone/map_generation/compose/visualize.py`.
