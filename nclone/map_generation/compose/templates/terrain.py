"""Phase-3 templates exercising slopes, curves, and ceiling/wall connectors."""

from __future__ import annotations

from ..template import Template, tag_any
from ..registry import register_template


@register_template(
    name="slope_course_2x3",
    grid=(2, 3),
    cells={
        "spawn": tag_any("terrain:flat"),
        "goal": tag_any("terrain:flat"),
        "*": tag_any("terrain:slope.mild", "terrain:slope.steep", "terrain:slope.45"),
        "offpath": tag_any("role:filler", "role:decoration"),
    },
)
class SlopeCourse(Template):
    pass


@register_template(
    name="curve_course_2x3",
    grid=(2, 3),
    cells={
        "spawn": tag_any("terrain:flat"),
        "goal": tag_any("terrain:flat"),
        "*": tag_any("terrain:curve.convex", "terrain:curve.concave"),
        "offpath": tag_any("role:filler", "role:decoration"),
    },
)
class CurveCourse(Template):
    pass


@register_template(
    name="tunnel_curves_2x3",
    grid=(2, 3),
    cells={
        "spawn": tag_any("terrain:flat"),
        "goal": tag_any("terrain:flat"),
        "*": tag_any("terrain:tunnel"),
        "offpath": tag_any("role:filler"),
    },
)
class TunnelCurves(Template):
    pass


@register_template(
    name="grand_mix_2x3",
    grid=(2, 3),
    cells={
        "spawn": tag_any("terrain:flat"),
        "goal": tag_any("terrain:flat"),
        "*": tag_any(
            "terrain:gap", "terrain:step", "hazard:mine.field",
            "terrain:slope.mild", "terrain:slope.steep",
            "terrain:curve.convex", "terrain:curve.concave",
        ),
        "offpath": tag_any("role:filler", "role:decoration"),
    },
)
class GrandMix(Template):
    pass
