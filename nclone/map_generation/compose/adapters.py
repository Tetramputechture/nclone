"""Adapters bridging the composable system to the legacy generator interface.

``TemplateGenerator`` exposes a registered template as a ``Map``-shaped generator
(``.generate(seed)``), so a composable template can flow through
``GeneratorFactory.create_from_preset`` -> ``BUCKET_CATALOG`` ->
``generate_benchmark_dataset.py`` -> the env-side ``GeneratorRegistry`` with zero
changes to those systems. It also carries the aggregated fine-grained labels as
duck-typed attributes (``mechanics`` / ``secondary_tags`` / ``declared_difficulty``
/ ``components_meta``) the dataset script reads opportunistically.

``LegacyGeneratorComponent`` wraps an existing ``GENERATOR_REGISTRY`` generator as
a whole-canvas component, so a template can drop a real maze next to a new mine
component (proving coexistence).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from ..map import Map
from .component import Component, ComponentContext, StampResult
from .registry import TEMPLATE_REGISTRY


class TemplateGenerator(Map):
    """A ``Map`` subclass that builds itself from a registered template."""

    TEMPLATE_NAME: Optional[str] = None

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self._labels = None

    def generate(self, seed: Optional[int] = None) -> "TemplateGenerator":
        cls = TEMPLATE_REGISTRY[self.TEMPLATE_NAME]
        built_map, labels = cls().build(seed)
        # copy the realized state onto ourselves so .map_data() works as usual
        self.tile_data = built_map.tile_data
        self.entity_data = built_map.entity_data
        self.entity_counts = built_map.entity_counts
        self.ninja_spawn_x = built_map.ninja_spawn_x
        self.ninja_spawn_y = built_map.ninja_spawn_y
        self.ninja_orientation = built_map.ninja_orientation
        self._labels = labels
        return self

    # --- duck-typed label surface read by the dataset script ------------------
    @property
    def mechanics(self) -> List[str]:
        return list(self._labels.mechanics) if self._labels else []

    @property
    def secondary_tags(self) -> List[str]:
        return list(self._labels.secondary) if self._labels else []

    @property
    def declared_difficulty(self) -> Dict[str, int]:
        return dict(self._labels.declared_difficulty) if self._labels else {}

    @property
    def components_meta(self) -> List[Dict]:
        return list(self._labels.components) if self._labels else []

    @property
    def primary_bucket(self) -> Optional[str]:
        return self._labels.primary_bucket if self._labels else None


def make_template_generator(template_name: str) -> Type[TemplateGenerator]:
    """Build a ``TemplateGenerator`` subclass bound to ``template_name``.

    Tolerates the factory's ``setattr(generator, FIELD.upper(), value)`` config
    patching: arbitrary uppercase attributes simply land on the instance and are
    ignored (templates sample their params internally from the seed).
    """
    return type(
        f"TemplateGenerator_{template_name}",
        (TemplateGenerator,),
        {"TEMPLATE_NAME": template_name},
    )


class LegacyGeneratorComponent(Component):
    """Wrap a legacy ``GENERATOR_REGISTRY`` generator as a whole-canvas component."""

    def __init__(self, generator_name: str, tags, difficulty=None, preset: Optional[str] = None):
        self.generator_name = generator_name
        self.preset = preset
        self.name = f"legacy:{generator_name}"
        self.tags = set(tags)
        self.difficulty = difficulty or {}
        self.param_spec = {}
        self.ports = {}
        # only fits the whole canvas
        self.min_w = 40
        self.min_h = 21

    def stamp(self, ctx: ComponentContext) -> StampResult:
        from ..generator_factory import GeneratorFactory, GENERATOR_REGISTRY

        seed = ctx.rng.randint(0, 2**31 - 1)
        if self.preset is not None:
            gen = GeneratorFactory.create_from_preset(self.generator_name, self.preset, seed)
        else:
            gen = GENERATOR_REGISTRY[self.generator_name](seed=seed)
        gen.generate(seed=seed)
        # copy the legacy generator's tiles wholesale into the canvas map
        ctx.map.tile_data = list(gen.tile_data)
        ctx.map.entity_data = list(gen.entity_data)
        ctx.map.entity_counts = dict(gen.entity_counts)
        ctx.map.ninja_spawn_x = gen.ninja_spawn_x
        ctx.map.ninja_spawn_y = gen.ninja_spawn_y
        ctx.map.ninja_orientation = gen.ninja_orientation
        res = ctx.result()
        res.placed_spawn = True
        res.placed_goal = True
        return res
