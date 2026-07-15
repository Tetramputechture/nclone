"""Vertical / full-field composition templates (audit P0-3d).

* ``vertical_course_1x3`` -- a single-row grid of FULL-HEIGHT cells so the
  vertical-descent components (``drop_shaft`` / ``wall_chain``) actually fit
  (their ``min_h`` excludes the 2x3 grids' half-height cells).
* ``serpentine_full_field`` -- a boustrophedon (snake) route over THREE grid
  rows: traverse the bottom row left->right, wall-jump a 2-wide shaft up at
  the right wall, traverse the middle row right->left, climb again at the left
  wall, traverse the top row to the switch in the far corner -- then retrace
  the whole snake (dropping back down the shafts) to the exit door at spawn.
  Every prior template used only the bottom grid row (``template.py``'s
  default ``path_cells``); this is the first full-field route.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..canvas import Canvas
from ..component import ComponentContext, SOLID, EMPTY
from ..template import Template, LevelLabels, tag_any
from ..registry import register_template
from .. import labels as labels_mod
from ....constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT


@register_template(
    name="vertical_course_1x3",
    grid=(1, 3),
    cells={
        "spawn": tag_any("terrain:flat"),
        "goal": tag_any("terrain:flat"),
        "*": tag_any("move:fall.drop", "move:walljump.chain"),
    },
)
class VerticalCourse(Template):
    """Full-height cells chaining a sheer drop or a wall-jump slot climb."""

    pass


@register_template(name="serpentine_full_field", grid=(3, 3))
class SerpentineFullField(Template):
    """Boustrophedon over 3 rows: switch at the far top corner, exit at spawn.

    The three row corridors are sealed from each other by full-width floors;
    the only inter-row connections are 2-tile-wide wall-jump shafts (the
    validated wall-jumpable width) at alternating ends, so the route is forced
    through every row -- out AND back. Middle cells of each row draw from
    flat/mine-field components for hazard variety; the turn cells stay flat so
    no mine can sit in a shaft doorway.
    """

    # Interior columns of the two turn shafts (flush against the side walls).
    RIGHT_SHAFT = (MAP_TILE_WIDTH - 3, MAP_TILE_WIDTH - 2)  # (39, 40)
    LEFT_SHAFT = (1, 2)

    def path_cells(self, grid) -> List[Tuple[int, int]]:
        """Snake: bottom row L->R, middle row R->L, top row L->R."""
        cols = len(grid[0])
        cells: List[Tuple[int, int]] = []
        rows = len(grid)
        for i, r in enumerate(range(rows - 1, -1, -1)):
            order = range(cols) if i % 2 == 0 else range(cols - 1, -1, -1)
            cells.extend((r, c) for c in order)
        return cells

    def build(self, seed=None):
        canvas = Canvas(seed=seed, rows=self.grid[0], cols=self.grid[1])
        canvas.border()
        rng = canvas.rng

        on_path = self.path_cells(canvas.grid)
        canvas._path_cells = on_path
        rows = self.grid[0]

        # Per-row walking floor = the bottom tile row of each region band (the
        # row-(r-1) floor doubles as row r's ceiling). Unlike the base template
        # there is no shared baseline: each row band is its own corridor.
        floor_of = {r: canvas.region(r, 0).y1 for r in range(rows)}

        # 1. Solid everywhere, then carve each row's corridor interior empty
        #    (keeping every band's floor line intact -- the seal between rows).
        m = canvas.map
        for x in range(1, MAP_TILE_WIDTH - 1):
            for y in range(1, MAP_TILE_HEIGHT - 1):
                m.set_tile(x, y, SOLID)
        for r in range(rows):
            region = canvas.region(r, 0)
            for x in range(1, MAP_TILE_WIDTH - 1):
                for y in range(region.y, floor_of[r]):
                    m.set_tile(x, y, EMPTY)

        # 2. Stamp components cell-by-cell along the snake. Turn cells (the
        #    shaft ends), the spawn cell, and the goal cell stay FLAT; middle
        #    cells draw flat / mine-field variety. Contexts are built manually
        #    so each row band uses ITS OWN floor as both floor_row and
        #    base_row (the shared canvas base_row would fill lower rows solid).
        spawn_cell = on_path[0]
        goal_cell = on_path[-1]
        turn_cells = set()
        for a, b in zip(on_path, on_path[1:]):
            if a[0] != b[0]:  # row change: both cells flank a shaft
                turn_cells.update((a, b))
        flat_only = tag_any("terrain:flat")
        variety = tag_any("terrain:flat", "hazard:mine.field")
        for (r, c) in on_path:
            region = canvas.region(r, c)
            filt = (
                flat_only
                if (r, c) in turn_cells or (r, c) in (spawn_cell, goal_cell)
                else variety
            )
            comp = self._pick(region, filt, rng)
            params = comp.sample_params(rng)
            ctx = ComponentContext(
                canvas, region, params, rng, floor_of[r], floor_of[r]
            )
            res = comp.stamp(ctx)
            res.tags = set(res.tags) | set(comp.tags)
            for k, v in (comp.difficulty or {}).items():
                res.difficulty[k] = max(res.difficulty.get(k, 0), v)
            canvas._stamps[(r, c)] = res

        # 3. Carve the turn shafts: a 2-wide slot through the separating floor,
        #    flush against the map wall, with a short interior wall enclosing
        #    the slot on its open side and a 2-tall doorway at the lower row's
        #    ground so the ninja can walk in. 2-wide + walls on both sides is
        #    the validated wall-jumpable geometry (loop_corridor cw=2).
        for lower in range(rows - 1, 0, -1):
            upper = lower - 1
            on_right = (rows - lower) % 2 == 1  # bottom row turns right first
            shaft = self.RIGHT_SHAFT if on_right else self.LEFT_SHAFT
            wall_x = shaft[0] - 1 if on_right else shaft[1] + 1
            up_floor = floor_of[upper]  # the floor line the shaft punches through
            low_floor = floor_of[lower]
            low_top = canvas.region(lower, 0).y
            for gx in shaft:
                m.set_tile(gx, up_floor, EMPTY)
            # Enclosing wall inside the lower row, leaving a 2-tall doorway.
            for gy in range(low_top, low_floor - 2):
                m.set_tile(wall_x, gy, SOLID)
            for gy in (low_floor - 2, low_floor - 1):
                m.set_tile(wall_x, gy, EMPTY)

        # 4. Objectives: spawn bottom-left, exit door right beside it, switch on
        #    the top row's far corner -- the full snake out AND back.
        spawn_r = spawn_cell[0]
        goal_r = goal_cell[0]
        spawn_x = self.LEFT_SHAFT[1] + 2  # clear of the left shaft's wall
        door_x = spawn_x + 2
        switch_x = self.RIGHT_SHAFT[0] - 2  # clear of the right shaft columns
        spawn_y = floor_of[spawn_r] - 1
        switch_y = floor_of[goal_r] - 1
        for gx in (spawn_x, door_x):
            m.set_tile(gx, floor_of[spawn_r], SOLID)
            m.set_tile(gx, spawn_y, EMPTY)
        m.set_tile(switch_x, floor_of[goal_r], SOLID)
        m.set_tile(switch_x, switch_y, EMPTY)
        m.set_ninja_spawn(spawn_x, spawn_y, orientation=1)
        m.add_entity(3, door_x, spawn_y, 0, 0, switch_x, switch_y)

        labels = self._serpentine_labels(canvas, on_path)
        return canvas.to_map(), labels

    def _serpentine_labels(self, canvas: Canvas, on_path) -> LevelLabels:
        on_path_tags: set = {
            "puzzle:exit_switch",
            "terrain:shaft",
            "move:walljump.chain",
            "move:fall.drop",
            "move:turn",
        }
        components_meta: List[Dict] = []
        for (r, c) in on_path:
            res = canvas._stamps.get((r, c))
            if res is None:
                continue
            on_path_tags |= set(res.tags)
            components_meta.append({"name": "", "cell": [r, c], "on_path": True})
        mechanics = labels_mod.mechanics_of(on_path_tags)
        return LevelLabels(
            mechanics=mechanics,
            components=components_meta,
            primary_bucket=labels_mod.rollup_bucket(on_path_tags),
            declared_difficulty=labels_mod.difficulty_vector(on_path_tags),
            secondary=mechanics,
        )
