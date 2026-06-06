---
name: nclone-map-author
description: Author nclone maps and new map TYPES agentically with the composable 2D system in nclone/map_generation/compose/. Use when asked to create/design/build a map or level, invent a new map type / mechanic course / template / component, generate a map from a natural-language description, or add a generated map type to the training curriculum/benchmark. Covers building from existing templates, the NL phrase builder, and writing new @register_component / @register_template definitions.
---

# Authoring nclone maps (composable 2D system)

Maps are authored with `nclone/map_generation/compose/` — a 2D region-grid system
of **components** (labelled level fragments) composed by **templates** over a grid,
emitting a `Map` whose `.map_data()` the simulator loads. It reuses the proven 1D
segment engine (`composition.py`) and the verified curve/slope tile helpers.

Always run with the project venv: `source .venv/bin/activate`.
After building anything, **verify it with the `nclone-map-render` skill** (image)
and `is_map_solvable` (below).

## 1. Build from an existing template or a phrase (fastest)

```python
from nclone.map_generation import compose

m, labels = compose.compose("mixed_traverse_2x3", seed=0)   # -> (Map, LevelLabels)
md = m.map_data()
print(labels.primary_bucket, labels.mechanics, labels.declared_difficulty)
```

Natural language (a light keyword parser; the *agent* supplies the real NL):

```python
from nclone.map_generation.compose import nl
m, labels, warnings = nl.build_from_phrase("flat run, two gaps, a steep hill, then mines", seed=1)
```

Phrases map keywords -> components (flat/gap/jump/step/slope/steep/45/hump/bowl/
tunnel/mine). Capped at `nl.MAX_PATH_FEATURES` path features (2D cells get narrow);
for long linear courses use the `COMPOSITE` generator instead.

## 2. Inspect the building blocks (the agentic surface)

```python
from nclone.map_generation import compose
compose.list_components(tag="terrain:slope.steep")     # names with a tag
compose.describe("single_gap")                          # {tags, params, ports, difficulty, min_size, rollup_bucket}
compose.list_templates()
from nclone.map_generation.compose import registry
registry.list_templates_matching(allowed_tags={...}, ceilings={"move":2}, required_tag="move:curve.ride")
```

## 3. Author a NEW component (~30-50 LOC)

A component = one `@register_component` decorator + one `stamp(ctx)` method. `ctx`
is a **region-clamped** `ComponentContext`: every write is clamped to the cell's
column band, so a component cannot draw into a neighbour. Put new components in
`nclone/map_generation/compose/components/<file>.py` and import that file from
`components/__init__.py`.

```python
from ..component import Component, ComponentContext, IntRange, Choice, SOLID, EMPTY
from ..registry import register_component

@register_component(
    name="mine_ledge",
    tags={"hazard:mine.field", "move:jump.gap", "role:traverse"},  # MUST exist in labels.TAGS
    params={"mine_count": IntRange(1, 2)},
    difficulty={"hazard": 3, "move": 2},
    min_w=8,                       # fits() uses min_w/min_h
)
class MineLedge(Component):
    def stamp(self, ctx: ComponentContext):
        ctx.floor()                                  # solid floor at baseline + support below
        for gx in ctx.spread_x(int(ctx.p.mine_count), margin=2):
            ctx.add_mine(gx, ctx.floor_row - 1)      # mine on the walking surface
        ctx.set_edge("W", ctx.floor_row)             # realized edge floor rows (for carving)
        ctx.set_edge("E", ctx.floor_row)
        return ctx.result()
```

`ComponentContext` essentials (all x-clamped to the cell):
`set_tile/get_tile/clear_column`, `floor(row=None)`, `fill_support(gx, top)`,
`add_mine/add_gold`, `place_spawn`, `place_switch_door`, `spread_x(n, margin)`,
geometry helpers `slope_up/slope_down(gx, gy, height, kind="mild|steep|45")`,
`hump(gx, top_width)`, `bowl(gx, bottom_width)`, `fill_cell_with_segment(name)`
(reuses 1D `SEGMENT_BUILDERS`: flat/gap/slope_up/slope_down/bowl/hump/staircase),
`tag(*tags)`, `set_edge(edge, row)`, `result()`. `ctx.floor_row` = shared baseline
the ninja walks on; `ctx.base_row` = bottom support row; `ctx.region` = the cell.

**Robustness rule:** make path components SYMMETRIC — return the surface to
`ctx.floor_row` at both the W and E edges (a slope = a hill, a curve returns to
baseline, a tunnel only adds a roof). Then seams between cells connect trivially.

## 4. Author a NEW template

A template = a recipe: grid shape, per-cell tag filters, a path policy. Put it in
`compose/templates/<file>.py` and import from `templates/__init__.py`.

```python
from ..template import Template, tag_any, tag_all
from ..registry import register_template

@register_template(
    name="hazard_course_2x3",
    grid=(2, 3), grid_jitter=False,
    cells={
        "spawn": tag_any("terrain:flat"),                 # spawn cell must be flat footing
        "goal":  tag_any("terrain:flat"),                 # goal cell hosts switch+door
        "*":     tag_any("hazard:mine.field", "terrain:gap", "terrain:slope.steep"),
        "offpath": tag_any("role:filler", "role:decoration"),
    },
    path="monotone_walk",          # default: bottom row, left -> right
    decorate_offpath=True,
)
class HazardCourse(Template):
    pass
```

The engine auto-places the single ninja spawn (spawn cell) and the single exit
switch+door pair (goal cell) on guaranteed-flat footing, carves on-path corridors,
decorates off-path cells, and aggregates labels. To force specific components per
cell (ad-hoc authoring) set `explicit_components = {(row, col): "component_name"}`.

## 5. Geometry vocabulary

- Floor slopes: `connectors.slope_up/slope_down(kind=)` — `mild` (18-25), `steep`
  (26-33), `45` (8/9). Floor curves: `curve_utils.create_convex_hump` ([12][13]),
  `create_concave_bowl` ([17][16]) — `top_width`/`bottom_width` = radius.
- Ceiling/wall connectors (enclosed tunnels/shafts, solid-above): ceiling diagonals
  6/7, ceiling convex 10/11, ceiling concave 14/15 — emit via
  `connectors.ceiling_slope/ceiling_hump/ceiling_dip` (support fills UPWARD; never
  set these tiles by hand). The `tunnel` component demonstrates them.
- Orientation is **verified** against `graph_builder._check_subnode_validity_simple`
  (floor tiles solid-below, ceiling tiles solid-above) — see `tests/test_compose_geometry.py`.

## 6. Tags (controlled vocabulary — `compose/labels.py`)

Every tag is `domain:name[.qualifier]` and MUST be registered in `labels.TAGS`
(adding a component with an unknown tag raises). Domains: `move:` (skill),
`terrain:` (geometry), `hazard:`, `entity:`, `puzzle:`, `role:` (bookkeeping,
excluded from `mechanics`). Each tag declares a `rollup` bucket (∈ `BUCKET_CATALOG`)
and a difficulty 0-6. To add a NEW tag, append a `TagDef` to `_DEFS` in `labels.py`;
`labels.validate_labels()` must return `[]`. A level's primary bucket = the
highest-difficulty on-path tag's rollup (ties -> harder bucket).

## 7. Verify (do this every time)

```python
from nclone.map_generation.compose import describe_level, assert_tile_accurate
from nclone.map_generation.solvability import is_map_solvable

md = m.map_data()
assert_tile_accurate(md)                 # round-trip / coordinate guard
assert is_map_solvable(md)               # switch+door reachable from spawn (grid flood-fill)
print(describe_level(md, labels))        # dims, tile_histogram, OFFSET-CORRECT entities, labels
```

Invariants any catalog-bound map MUST satisfy (mirror existing tests):
**reproducible** (same seed -> identical `map_data`), **entity purity** (only types
{1, 21, 3, 4} = mines + exit door/switch), **exactly one** exit door + one switch.
Then render an image with the `nclone-map-render` skill.

## 8. Wire into the curriculum / benchmark (optional)

Every registered template is auto-exposed as a generator `template_<name>` with a
`default` preset (see `generator_factory._register_compose_templates`). To put it in
the benchmark, add `("template_<name>", "default", weight)` to a
`BUCKET_CATALOG[bucket][tier]` list in `generator_configs.py`. NOTE this changes the
existing dataset's per-index generator distribution, so only do it deliberately.
The dataset script (`scripts/generate_benchmark_dataset.py`) auto-enriches records
with `secondary` + `metadata.{mechanics,components,declared_difficulty}` for these.

## 9. Determinism contract (FROZEN — don't break)

A single seeded `random.Random` (`Map.rng`) is threaded in this order:
partition grid -> per-cell component assignment -> per-component param sampling +
stamp -> corridor carving -> spawn/goal placement -> off-path decoration. Never
instantiate a fresh `random` inside a component; always use `ctx.rng` / `ctx.p`.

## Related skills
- **nclone-map-editor** — build a level FROM SCRATCH with direct single/batch
  tile+entity tools and ASCII-art (e.g. "a level that looks like a smiley face"),
  with a `check()` buildability/solvability loop. Use it for one-off / pictorial /
  hand-placed levels; use THIS skill for reusable components & curriculum templates.
- **nclone-map-render** — export any map to a verification image.

## Reference
- Package: `nclone/map_generation/compose/` (`__init__.py` is the public API).
- Design doc: `docs/COMPOSABLE_MAP_GENERATION.md`.
- Tests (copy their patterns): `tests/test_compose_{core,geometry,authoring}.py`.
