"""Template: a composition recipe over the region grid + label aggregation.

A template declares a grid shape, per-cell tag filters (which components may fill
which cells), a traversal-path policy, and whether off-path cells are decorated.
``build(seed)`` produces a ``(Map, LevelLabels)`` pair deterministically. The
frozen RNG-consumption order (see ``compose/__init__``) is:
  partition -> per-cell component assignment -> per-component param sampling +
  stamp -> corridor carving -> spawn/goal placement -> off-path decoration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .canvas import Canvas
from .component import Component, SOLID
from . import registry as registry_mod
from . import labels as labels_mod

CellFilter = Callable[[Component], bool]


def tag_any(*tags: str) -> CellFilter:
    want = set(tags)
    return lambda c: bool(c.tags & want)


def tag_all(*tags: str) -> CellFilter:
    want = set(tags)
    return lambda c: want <= set(c.tags)


def any_component() -> CellFilter:
    return lambda c: True


@dataclass
class LevelLabels:
    mechanics: List[str] = field(default_factory=list)
    components: List[Dict] = field(default_factory=list)
    primary_bucket: str = "FLAT_RUN"
    declared_difficulty: Dict[str, int] = field(default_factory=dict)
    secondary: List[str] = field(default_factory=list)


class Template:
    name: str = "template"
    grid: Tuple[int, int] = (2, 3)
    grid_jitter: bool = False
    cells: Dict[str, CellFilter] = {}
    path: str = "monotone_walk"
    decorate_offpath: bool = True
    # optional explicit per-cell component names {(row, col): "name"} -- honored
    # before the tag-filter pick (used by the NL / ad-hoc authoring path).
    explicit_components: Dict = None

    # --- path policy ----------------------------------------------------------
    def path_cells(self, grid) -> List[Tuple[int, int]]:
        """Default monotone walk along the bottom row, left -> right."""
        rows = len(grid)
        cols = len(grid[0])
        r = rows - 1
        return [(r, c) for c in range(cols)]

    # --- component selection --------------------------------------------------
    def _filter_for(self, role: str) -> CellFilter:
        if role in self.cells:
            return self.cells[role]
        if "*" in self.cells:
            return self.cells["*"]
        return any_component()

    def _pick(self, region, filt: CellFilter, rng) -> Component:
        candidates = sorted(
            (name for name, comp in registry_mod.COMPONENT_REGISTRY.items()
             if filt(comp) and comp.fits(region)),
        )
        if not candidates:
            # guaranteed-fitting fallback
            if "filler_solid" in registry_mod.COMPONENT_REGISTRY:
                return registry_mod.COMPONENT_REGISTRY["filler_solid"]
            # last resort: any fitting component
            candidates = sorted(
                name for name, comp in registry_mod.COMPONENT_REGISTRY.items()
                if comp.fits(region)
            )
        name = rng.choice(candidates)
        return registry_mod.COMPONENT_REGISTRY[name]

    # --- build ----------------------------------------------------------------
    def build(self, seed: Optional[int] = None) -> Tuple["object", LevelLabels]:
        canvas = Canvas(seed=seed, rows=self.grid[0], cols=self.grid[1],
                        jitter=self.grid_jitter)
        canvas.border()
        rng = canvas.rng

        on_path = self.path_cells(canvas.grid)
        canvas._path_cells = on_path
        spawn_cell = on_path[0]
        goal_cell = on_path[-1]
        path_set = set(on_path)

        assignments: Dict[Tuple[int, int], Component] = {}
        # deterministic assignment order: row-major over the whole grid
        for r, row in enumerate(canvas.grid):
            for c, region in enumerate(row):
                if (r, c) == spawn_cell:
                    role = "spawn"
                elif (r, c) == goal_cell:
                    role = "goal"
                elif (r, c) in path_set:
                    role = "*"
                else:
                    role = "offpath"
                if role == "offpath":
                    if not self.decorate_offpath:
                        continue
                    filt = self.cells.get("offpath", tag_any("role:filler", "role:decoration"))
                else:
                    filt = self._filter_for(role)
                if self.explicit_components and (r, c) in self.explicit_components:
                    name = self.explicit_components[(r, c)]
                    comp = registry_mod.COMPONENT_REGISTRY.get(name) or self._pick(
                        region, filt, rng
                    )
                else:
                    comp = self._pick(region, filt, rng)
                assignments[(r, c)] = comp

        # stamp on-path cells first (so seam carving reads their realized edges)
        for cell in on_path:
            comp = assignments.get(cell)
            if comp is not None:
                canvas.stamp_component(canvas.region(*cell), comp)
        for (r, c), comp in assignments.items():
            if (r, c) in path_set:
                continue
            canvas.stamp_component(canvas.region(r, c), comp)

        # carve guaranteed connections along the path
        for a, b in zip(on_path, on_path[1:]):
            canvas.carve_corridor(canvas.region(*a), canvas.region(*b))

        # place spawn + the single switch/door pair on guaranteed-flat footing
        self._place_objectives(canvas, spawn_cell, goal_cell)

        labels = self._aggregate_labels(canvas, assignments, path_set)
        return canvas.to_map(), labels

    def _place_objectives(self, canvas: Canvas, spawn_cell, goal_cell) -> None:
        fr = canvas.path_floor_row
        m = canvas.map
        s_region = canvas.region(*spawn_cell)
        g_region = canvas.region(*goal_cell)
        spawn_x = s_region.x + 1
        door_x = g_region.x1 - 1
        switch_x = g_region.x1 - 3
        # guarantee flat footing under each objective column
        for gx in (spawn_x, switch_x, door_x):
            m.set_tile(gx, fr, SOLID)
            for gy in range(fr + 1, canvas.base_row + 1):
                if m.get_tile(gx, gy) == 0:
                    m.set_tile(gx, gy, SOLID)
            for gy in range(max(1, fr - 3), fr):
                m.set_tile(gx, gy, 0)
        m.set_ninja_spawn(spawn_x, fr - 1, orientation=1)
        m.add_entity(3, door_x, fr - 1, 0, 0, switch_x, fr - 1)

    def _aggregate_labels(self, canvas: Canvas, assignments, path_set) -> LevelLabels:
        on_path_tags: set = set()
        components_meta: List[Dict] = []
        for (r, c), comp in sorted(assignments.items()):
            res = canvas._stamps.get((r, c))
            on = (r, c) in path_set
            tags = (res.tags if res else set(comp.tags))
            if on:
                on_path_tags |= set(tags)
            components_meta.append({"name": comp.name, "cell": [r, c], "on_path": on})
        on_path_tags.add("puzzle:exit_switch")
        mechanics = labels_mod.mechanics_of(on_path_tags)
        return LevelLabels(
            mechanics=mechanics,
            components=components_meta,
            primary_bucket=labels_mod.rollup_bucket(on_path_tags),
            declared_difficulty=labels_mod.difficulty_vector(on_path_tags),
            secondary=mechanics,
        )

    def preview(self, seed: Optional[int] = None, detail: str = "compact") -> str:
        from .visualize import to_compact_ascii, describe_level
        import json

        m, labels = self.build(seed)
        md = m.map_data()
        ascii_view = to_compact_ascii(md)
        facts = describe_level(md, labels)
        return ascii_view + "\n\n" + json.dumps(facts, indent=2)
