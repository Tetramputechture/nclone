"""Composable 2D region-grid map generation.

An agent (or human) authors **components** (labelled level fragments) and
**templates** (composition recipes over a region grid); the framework stamps
them into a :class:`~nclone.map_generation.map.Map`, carves guaranteed on-path
corridors, and aggregates fine-grained mechanic tags into a coarse bucket so the
legacy benchmark pipeline keeps working unchanged.

Determinism contract (FROZEN -- preserves bit-for-bit reproducibility):
    partition grid -> per-cell component assignment -> per-component param
    sampling + stamp -> corridor carving -> spawn/goal placement -> off-path
    decoration. A single seeded ``random.Random`` (``Map.rng``) is threaded in
    exactly this order.

Agentic loop (the "natural-language" surface):
    list_components(tag=...) / describe(name)  -> enumerate building blocks
    assemble or pick a template                -> compose(template, seed)
    Template.preview(seed) / describe_level    -> tile-accurate ASCII + facts
    adjust -> repeat
"""

from __future__ import annotations

from typing import Optional, Union

# populate the registries (decorators run on import)
from . import components as _components  # noqa: F401
from . import templates as _templates  # noqa: F401

from .registry import (  # noqa: F401
    COMPONENT_REGISTRY,
    TEMPLATE_REGISTRY,
    register_component,
    register_template,
    list_components,
    list_templates,
    describe,
    describe_template,
)
from .template import Template, LevelLabels, tag_any, tag_all
from .visualize import describe_level, to_compact_ascii, to_shape_ascii, assert_tile_accurate
from .editor import (
    MapEditor,
    MIDDLE, CENTER, TOP_MIDDLE, BOTTOM_MIDDLE, LEFT_MIDDLE, RIGHT_MIDDLE,
    TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT,
    above, below, left_of, right_of,
)
from .validate import check_level
from .mapio import save_map, load_map_data, load_map
from . import labels


def compose(template: Union[str, Template], seed: Optional[int] = None):
    """Build a ``(Map, LevelLabels)`` from a template name or instance."""
    if isinstance(template, str):
        cls = TEMPLATE_REGISTRY[template]
        template = cls()
    return template.build(seed)
