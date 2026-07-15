"""Movement-expansion composition templates.

Chain the movement components into multi-mechanic courses (the "compose" half of
the master-in-isolation-then-compose curriculum): a long-jump course, a mine
gauntlet (floor + ceiling mines + gaps), and a full movement mix. Each
auto-registers as a ``template_<name>`` generator via the compose factory hook,
so it drops straight into the generator registry / curriculum.
"""

from __future__ import annotations

from ..template import Template, tag_any
from ..registry import register_template


@register_template(
    name="long_jump_course_2x3",
    grid=(2, 3),
    cells={
        "spawn": tag_any("terrain:flat"),
        "goal": tag_any("terrain:flat"),
        "*": tag_any("terrain:gap", "terrain:step"),
        "offpath": tag_any("role:filler", "role:decoration"),
    },
)
class LongJumpCourse(Template):
    """Wide gaps + steps: build speed and clear running long jumps."""

    pass


@register_template(
    name="mine_gauntlet_2x3",
    grid=(2, 3),
    cells={
        "spawn": tag_any("terrain:flat"),
        "goal": tag_any("terrain:flat"),
        "*": tag_any(
            "hazard:mine.field", "hazard:mine.ceiling", "terrain:gap",
        ),
        "offpath": tag_any("role:filler", "role:decoration"),
    },
)
class MineGauntlet(Template):
    """Floor mines, ceiling mines, and gaps: thread a mined corridor."""

    pass


@register_template(
    name="movement_mix_2x3",
    grid=(2, 3),
    cells={
        "spawn": tag_any("terrain:flat"),
        "goal": tag_any("terrain:flat"),
        "*": tag_any(
            "terrain:gap", "terrain:step",
            "terrain:slope.mild", "terrain:slope.steep", "terrain:slope.45",
            "terrain:curve.convex", "terrain:curve.concave",
            "hazard:mine.field", "hazard:mine.ceiling",
        ),
        "offpath": tag_any("role:filler", "role:decoration"),
    },
)
class MovementMix(Template):
    """The full movement vocabulary chained into one course (COMPOSITE)."""

    pass
