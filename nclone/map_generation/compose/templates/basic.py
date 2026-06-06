"""Phase-1 templates: a flat intro and the doc's mixed-traverse worked example."""

from __future__ import annotations

from ..template import Template, tag_any
from ..registry import register_template


@register_template(
    name="flat_intro_2x3",
    grid=(2, 3),
    grid_jitter=False,
    cells={
        "spawn": tag_any("terrain:flat"),
        "goal": tag_any("terrain:flat"),
        "*": tag_any("terrain:flat"),
        "offpath": tag_any("role:filler"),
    },
    path="monotone_walk",
    decorate_offpath=True,
)
class FlatIntro(Template):
    pass


@register_template(
    name="mixed_traverse_2x3",
    grid=(2, 3),
    grid_jitter=False,
    cells={
        "spawn": tag_any("terrain:flat"),
        "goal": tag_any("terrain:flat"),
        "*": tag_any("terrain:gap", "hazard:mine.field", "terrain:flat"),
        "offpath": tag_any("role:filler"),
    },
    path="monotone_walk",
    decorate_offpath=True,
)
class MixedTraverse(Template):
    pass
