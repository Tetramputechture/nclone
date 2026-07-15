"""Tests for the movement-mechanic expansion.

Covers the additive pieces layered on the existing system:

* the new compose components (``long_gap`` wide running-jump gaps, ``ceiling_mine``
  jump-height-capping ceiling mines) — registration, tags, and that they emit
  only baseline-legal entities / valid tiles;
* the new composition templates (``long_jump_course_2x3``, ``mine_gauntlet_2x3``,
  ``movement_mix_2x3``) — reproducible, subset-pure, and solvable;
* the curriculum wiring — those templates are referenced by ``BUCKET_CATALOG``
  and both the catalog and the compose label vocabulary validate clean.
"""

from __future__ import annotations

import pytest

from nclone.map_generation import compose
from nclone.map_generation.compose import describe_level
from nclone.map_generation.compose import registry as creg
from nclone.map_generation.compose.visualize import _tiles
from nclone.map_generation.solvability import is_map_solvable
from nclone.levelgen.labeling import subset_ok

NEW_COMPONENTS = ("long_gap", "ceiling_mine")
NEW_TEMPLATES = ("long_jump_course_2x3", "mine_gauntlet_2x3", "movement_mix_2x3")
ALLOWED_ENTITY_TYPES = {0, 1, 21, 3, 4}  # empty, mine, deadly mine, exit door, switch
SEEDS = (0, 1, 7, 42)


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", NEW_COMPONENTS)
def test_new_components_registered(name):
    assert name in creg.COMPONENT_REGISTRY


def test_ceiling_mine_tag_and_places_overhead_mine():
    comp = creg.COMPONENT_REGISTRY["ceiling_mine"]
    assert "hazard:mine.ceiling" in comp.tags
    # build a tiny canvas cell and stamp it; the mine must sit ABOVE the floor.
    from nclone.map_generation.compose.canvas import Canvas

    cv = Canvas(seed=3, rows=1, cols=1)
    cv.border()
    res = cv.stamp_component(cv.region(0, 0), comp)
    mines = [e for e in res.entities if e[0] == 1]
    assert mines, "ceiling_mine placed no mine"
    assert all(my < cv.path_floor_row for (_t, _mx, my) in mines), \
        "ceiling mines must hang above the walking floor"


def test_long_gap_is_wider_than_single_gap():
    """long_gap targets the running long-jump band (single_gap is a 2-3 hop)."""
    long_gap = creg.COMPONENT_REGISTRY["long_gap"]
    single = creg.COMPONENT_REGISTRY["single_gap"]
    assert long_gap.param_spec["gap_width"].lo > single.param_spec["gap_width"].hi


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("template", NEW_TEMPLATES)
def test_new_templates_registered(template):
    assert template in creg.TEMPLATE_REGISTRY


@pytest.mark.parametrize("template", NEW_TEMPLATES)
@pytest.mark.parametrize("seed", SEEDS)
def test_new_templates_reproducible_and_subset_pure(template, seed):
    m, _ = compose.compose(template, seed=seed)
    md = m.map_data()
    # byte-reproducible for a fixed seed
    again, _ = compose.compose(template, seed=seed)
    assert md == again.map_data()
    # baseline-legal entities, exactly one exit door + one switch
    assert subset_ok(md)
    ents = describe_level(md)["entities"]
    assert {e["type"] for e in ents} <= ALLOWED_ENTITY_TYPES
    assert sum(1 for e in ents if e["type"] == 3) == 1
    assert sum(1 for e in ents if e["type"] == 4) == 1


@pytest.mark.parametrize("template", NEW_TEMPLATES)
@pytest.mark.parametrize("seed", SEEDS)
def test_new_templates_solvable(template, seed):
    m, _ = compose.compose(template, seed=seed)
    assert is_map_solvable(m.map_data())


def test_mine_gauntlet_can_emit_ceiling_mines():
    """Across seeds the mine gauntlet exercises the new ceiling-mine component."""
    seen_ceiling = False
    for s in range(40):
        _m, lab = compose.compose("mine_gauntlet_2x3", seed=s)
        if "hazard:mine.ceiling" in set(getattr(lab, "mechanics", [])):
            seen_ceiling = True
            break
    assert seen_ceiling, "mine_gauntlet never selected the ceiling_mine component"


# --------------------------------------------------------------------------- #
# Curriculum wiring
# --------------------------------------------------------------------------- #
def test_catalog_and_labels_validate_clean():
    # import the factory first so compose templates are registered as generators
    from nclone.map_generation import generator_factory  # noqa: F401
    from nclone.map_generation.generator_configs import validate_bucket_catalog
    from nclone.map_generation.compose.labels import validate_labels

    assert validate_bucket_catalog() == []
    assert validate_labels() == []


def test_new_templates_referenced_by_buckets():
    from nclone.map_generation import generator_factory  # noqa: F401
    from nclone.map_generation.generator_configs import BUCKET_CATALOG

    referenced = {
        gen_type
        for tiers in BUCKET_CATALOG.values()
        for entries in tiers.values()
        for (gen_type, _preset, _w) in entries
    }
    for template in NEW_TEMPLATES:
        assert f"template_{template}" in referenced, template
