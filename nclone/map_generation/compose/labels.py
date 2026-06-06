"""Fine-grained, mechanic-based tag vocabulary for composable maps.

A component declares fine tags like ``move:jump.gap`` or ``terrain:slope.steep``;
the framework DERIVES the coarse ``BUCKET_CATALOG`` bucket (the ``rollup``) so the
legacy primary-bucket pipeline keeps working while curriculum gains a finer lever.

Each tag is ``domain:name[.qualifier]`` and is registered here once (a controlled
vocabulary prevents tag drift). ``role:*`` tags are composition bookkeeping and
are excluded from a level's ``mechanics`` label.

See ``docs/COMPOSABLE_MAP_GENERATION.md`` §4 for the source taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

DOMAINS = ("move", "terrain", "hazard", "entity", "puzzle", "role")


@dataclass(frozen=True)
class TagDef:
    tag: str
    domain: str
    rollup: str  # -> a BUCKET_CATALOG key (validated by validate_labels)
    difficulty: int  # 0-6, declared (not A*-verified)
    layer1_in_scope: bool = True
    requires: Tuple[str, ...] = field(default_factory=tuple)


def _t(tag, domain, rollup, diff, l1=True, requires=()):
    return TagDef(tag, domain, rollup, diff, l1, tuple(requires))


# --- the frozen vocabulary ----------------------------------------------------
# rollup must be a key in BUCKET_CATALOG (FLAT_RUN, BASIC_JUMP, WALL_JUMP,
# MINE_NAVIGATION, SLOPES, CHAMBER_MAZE, CURVES, MULTI_JUMP, COMPOSITE, PRECISION).
_DEFS: List[TagDef] = [
    # move: locomotion & aerial skills
    _t("move:walk", "move", "FLAT_RUN", 0),
    _t("move:run", "move", "FLAT_RUN", 1),
    _t("move:turn", "move", "FLAT_RUN", 1),
    _t("move:jump.hop", "move", "BASIC_JUMP", 1),
    _t("move:jump.gap", "move", "BASIC_JUMP", 2),
    _t("move:jump.stepup", "move", "BASIC_JUMP", 2),
    _t("move:jump.obstacle", "move", "BASIC_JUMP", 2),
    _t("move:fall.drop", "move", "BASIC_JUMP", 1),
    _t("move:fall.control", "move", "BASIC_JUMP", 2),
    _t("move:air.steer", "move", "BASIC_JUMP", 2),
    _t("move:jump.sequence", "move", "MULTI_JUMP", 3),
    _t("move:walljump.single", "move", "WALL_JUMP", 3),
    _t("move:walljump.chain", "move", "WALL_JUMP", 4),
    _t("move:walljump.ledge", "move", "WALL_JUMP", 4),
    _t("move:slope.ascend", "move", "SLOPES", 1),
    _t("move:slope.descend", "move", "SLOPES", 1),
    _t("move:slope.launch", "move", "SLOPES", 3),
    _t("move:curve.ride", "move", "CURVES", 2),
    _t("move:precision.land", "move", "PRECISION", 4),
    # terrain: static geometry shapes (roll up to a movement-ish bucket)
    _t("terrain:flat", "terrain", "FLAT_RUN", 0),
    _t("terrain:gap", "terrain", "BASIC_JUMP", 1),
    _t("terrain:step", "terrain", "BASIC_JUMP", 1),
    _t("terrain:platform", "terrain", "BASIC_JUMP", 1),
    _t("terrain:shaft", "terrain", "WALL_JUMP", 3),
    _t("terrain:corridor", "terrain", "FLAT_RUN", 1),
    _t("terrain:tunnel", "terrain", "FLAT_RUN", 2),
    _t("terrain:chamber", "terrain", "CHAMBER_MAZE", 2),
    _t("terrain:maze", "terrain", "CHAMBER_MAZE", 3),
    _t("terrain:slope.mild", "terrain", "SLOPES", 1),
    _t("terrain:slope.steep", "terrain", "SLOPES", 2),
    _t("terrain:slope.45", "terrain", "SLOPES", 2),
    _t("terrain:curve.convex", "terrain", "CURVES", 2),
    _t("terrain:curve.concave", "terrain", "CURVES", 2),
    # hazard: lethal things to avoid
    _t("hazard:mine.static", "hazard", "MINE_NAVIGATION", 2),
    _t("hazard:mine.field", "hazard", "MINE_NAVIGATION", 3),
    _t("hazard:mine.floor", "hazard", "MINE_NAVIGATION", 2),
    _t("hazard:mine.ceiling", "hazard", "MINE_NAVIGATION", 3),
    _t("hazard:fall.lethal", "hazard", "BASIC_JUMP", 2),
    # entity: interactive (mostly out-of-scope at Layer 1)
    _t("entity:gold", "entity", "FLAT_RUN", 1),
    # puzzle: objective gating
    _t("puzzle:exit_switch", "puzzle", "FLAT_RUN", 0),
    # role: composition bookkeeping (NOT mechanics)
    _t("role:spawn", "role", "FLAT_RUN", 0),
    _t("role:goal", "role", "FLAT_RUN", 0),
    _t("role:traverse", "role", "FLAT_RUN", 0),
    _t("role:filler", "role", "FLAT_RUN", 0),
    _t("role:decoration", "role", "FLAT_RUN", 0),
]

TAGS: Dict[str, TagDef] = {d.tag: d for d in _DEFS}

# Canonical bucket order for tie-breaking the primary rollup (matches the
# Phase-1 progression: simplest movement first, hardest last).
_BUCKET_ORDER = [
    "FLAT_RUN", "BASIC_JUMP", "SLOPES", "CURVES", "MULTI_JUMP",
    "WALL_JUMP", "MINE_NAVIGATION", "PRECISION", "COMPOSITE", "CHAMBER_MAZE",
]


def validate_tag(tag: str) -> None:
    if tag not in TAGS:
        raise KeyError(f"unknown tag {tag!r} (not in the controlled vocabulary)")


def _non_role(tags: Iterable[str]) -> List[TagDef]:
    out = []
    for t in tags:
        validate_tag(t)
        d = TAGS[t]
        if d.domain != "role":
            out.append(d)
    return out


def mechanics_of(tags: Iterable[str]) -> List[str]:
    """Sorted fine-tag list with ``role:*`` removed (the level ``mechanics`` label)."""
    return sorted(d.tag for d in _non_role(tags))


def rollup_bucket(tags: Iterable[str]) -> str:
    """Primary ``BUCKET_CATALOG`` bucket = rollup of the highest-difficulty tag.

    Ties broken by canonical bucket order. Empty / all-role input -> FLAT_RUN.
    """
    defs = _non_role(tags)
    if not defs:
        return "FLAT_RUN"
    # highest declared difficulty wins; ties go to the HARDER bucket (later in
    # the canonical progression order).
    best = max(
        defs,
        key=lambda d: (d.difficulty, _bucket_rank(d.rollup)),
    )
    return best.rollup


def _bucket_rank(bucket: str) -> int:
    return _BUCKET_ORDER.index(bucket) if bucket in _BUCKET_ORDER else len(_BUCKET_ORDER)


def difficulty_vector(tags: Iterable[str]) -> Dict[str, int]:
    """Per-domain max declared difficulty over non-role tags (e.g. {move:2, hazard:3})."""
    vec: Dict[str, int] = {}
    for d in _non_role(tags):
        vec[d.domain] = max(vec.get(d.domain, 0), d.difficulty)
    return vec


def validate_labels() -> List[str]:
    """Assert every rollup target is a real bucket and every ``requires`` resolves.

    Returns a list of human-readable problems (empty == valid). Imports
    ``BUCKET_CATALOG`` lazily to avoid an import cycle.
    """
    from ..generator_configs import BUCKET_CATALOG

    problems: List[str] = []
    bucket_keys = set(BUCKET_CATALOG.keys())
    for d in TAGS.values():
        if d.rollup not in bucket_keys:
            problems.append(f"tag {d.tag!r} rolls up to unknown bucket {d.rollup!r}")
        for req in d.requires:
            if req not in TAGS:
                problems.append(f"tag {d.tag!r} requires unknown tag {req!r}")
    return problems
