"""A light natural-language phrase parser over the composable system.

This is a convenience layer on top of the real agentic surface (``list_components``
/ ``describe`` / ``Template.preview`` / ``describe_level``). The LLM agent supplies
the actual natural language; this maps common phrases like
``"flat run, two gaps, a steep hill, then mines"`` to a declarative build-spec and
composes an ad-hoc template left-to-right.

It is intentionally small and keyword-based. For anything richer the agent should
assemble a :class:`Template` directly. Unfittable / overflow features degrade
gracefully and are reported in the returned ``warnings`` list rather than raising.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .template import Template, tag_any
from .canvas import PIT_DEPTH  # noqa: F401  (documents the floor model)

# phrase keyword -> component name (longest keyword wins)
KEYWORD_TO_COMPONENT = [
    ("steep hill", "slope_steep"),
    ("steep slope", "slope_steep"),
    ("45", "slope_45"),
    ("hill", "slope_mild"),
    ("slope", "slope_mild"),
    ("ramp", "slope_mild"),
    ("hump", "convex_hump"),
    ("bump", "convex_hump"),
    ("bowl", "concave_bowl"),
    ("dip", "concave_bowl"),
    ("tunnel", "tunnel"),
    ("step", "step_up"),
    ("ledge", "step_up"),
    ("gap", "single_gap"),
    ("pit", "single_gap"),
    ("jump", "single_gap"),
    ("mine", "mine_field"),
    ("flat", "flat_floor"),
    ("run", "flat_floor"),
    ("walk", "flat_floor"),
]

_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "couple": 2, "few": 3, "several": 3,
}

MAX_PATH_FEATURES = 4  # keeps 2D cells wide enough for the bigger components


@dataclass
class BuildSpec:
    path_components: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _count_before(text: str, idx: int) -> int:
    """Pull a leading count word/number from the clause ending at ``idx``."""
    head = text[:idx].split()
    if head:
        last = head[-1]
        if last.isdigit():
            return max(1, int(last))
        if last in _NUMBER_WORDS:
            return _NUMBER_WORDS[last]
    return 1


def parse(text: str) -> BuildSpec:
    """Turn a phrase into a :class:`BuildSpec` of ordered path components."""
    spec = BuildSpec()
    lowered = text.lower()
    # split into clauses on commas / "then" / "and"
    clauses = re.split(r",|\bthen\b|\band\b", lowered)
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        for kw, comp in KEYWORD_TO_COMPONENT:
            pos = clause.find(kw)
            if pos != -1:
                n = _count_before(clause, pos)
                spec.path_components.extend([comp] * n)
                break
    if not spec.path_components:
        spec.warnings.append("no recognizable features; defaulting to a flat run")
        spec.path_components = ["flat_floor"]
    if len(spec.path_components) > MAX_PATH_FEATURES:
        dropped = spec.path_components[MAX_PATH_FEATURES:]
        spec.path_components = spec.path_components[:MAX_PATH_FEATURES]
        spec.warnings.append(
            f"capped to {MAX_PATH_FEATURES} path features for the 2D grid; "
            f"dropped {dropped} (use a Template or the COMPOSITE generator for longer courses)"
        )
    return spec


class AdHocTemplate(Template):
    """A single-row template whose path cells are explicitly named components."""

    def __init__(self, path_components: List[str]):
        # spawn flat + features + goal flat, one cell each
        names = ["flat_floor"] + list(path_components) + ["flat_floor"]
        ncols = len(names)
        self.grid = (1, ncols)
        self.grid_jitter = False
        self.decorate_offpath = False
        self.cells = {
            "spawn": tag_any("terrain:flat"),
            "goal": tag_any("terrain:flat"),
            "*": tag_any("terrain:flat"),
        }
        self.explicit_components = {(0, c): names[c] for c in range(ncols)}
        self.name = "adhoc"


def build_from_spec(spec: BuildSpec, seed: Optional[int] = None):
    """Compose a ``(Map, LevelLabels)`` from a :class:`BuildSpec`."""
    tmpl = AdHocTemplate(spec.path_components)
    return tmpl.build(seed)


def build_from_phrase(text: str, seed: Optional[int] = None) -> Tuple[object, object, List[str]]:
    """Parse ``text`` and build a level. Returns ``(Map, LevelLabels, warnings)``."""
    spec = parse(text)
    m, labels = build_from_spec(spec, seed)
    return m, labels, spec.warnings
