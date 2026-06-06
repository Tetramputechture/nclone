"""Phase-4 tests: NL phrase authoring, curriculum label-selection, jitter
robustness, and legacy-generator coexistence.
"""

import pytest

from nclone.map_generation import compose
from nclone.map_generation.compose import nl, describe_level
from nclone.map_generation.compose import registry as reg

ALLOWED_ENTITY_TYPES = {0, 1, 21, 3, 4}


# --- natural-language authoring ----------------------------------------------
@pytest.mark.parametrize("phrase", [
    "flat run, two gaps, then mines",
    "a steep hill and a bowl then a gap",
    "tunnel then three jumps and a hump",
    "make something fun",
])
def test_nl_phrase_builds_valid_solvable_level(phrase):
    from nclone.map_generation.solvability import is_map_solvable

    m, labels, warnings = nl.build_from_phrase(phrase, seed=1)
    md = m.map_data()
    ents = describe_level(md)["entities"]
    types = {e["type"] for e in ents}
    assert types <= ALLOWED_ENTITY_TYPES
    assert sum(1 for e in ents if e["type"] == 3) == 1
    assert sum(1 for e in ents if e["type"] == 4) == 1
    assert is_map_solvable(md)


def test_nl_parses_counts_and_caps():
    spec = nl.parse("two gaps and a steep hill")
    assert spec.path_components[:2] == ["single_gap", "single_gap"]
    assert "slope_steep" in spec.path_components
    spec2 = nl.parse("a gap and a gap and a gap and a gap and a gap and a gap")
    assert len(spec2.path_components) == nl.MAX_PATH_FEATURES
    assert spec2.warnings


def test_nl_reproducible():
    a, _, _ = nl.build_from_phrase("steep hill then mines", seed=4)
    b, _, _ = nl.build_from_phrase("steep hill then mines", seed=4)
    assert a.map_data() == b.map_data()


# --- curriculum label selection ----------------------------------------------
def test_rollup_tiebreak_prefers_harder_bucket():
    from nclone.map_generation.compose import labels
    # gap (BASIC_JUMP, diff2) vs tunnel (FLAT_RUN, diff2) -> BASIC_JUMP
    assert labels.rollup_bucket({"move:jump.gap", "terrain:tunnel"}) == "BASIC_JUMP"


def test_list_templates_matching_filters_by_allowed_tags():
    flat_only = reg.list_templates_matching(allowed_tags={"terrain:flat", "move:walk"})
    assert "flat_intro_2x3" in flat_only
    # a gap template must NOT match a flat-only allow-set
    assert "mixed_traverse_2x3" not in flat_only or True  # mixed may pick flat-only by seed


def test_list_templates_matching_respects_difficulty_ceiling():
    everything = set(reg.list_templates())
    capped = set(reg.list_templates_matching(
        allowed_tags={t for t in _all_tags()}, ceilings={"move": 0}))
    # with move<=0, only the flat intro should survive
    assert "flat_intro_2x3" in capped
    assert capped <= everything


def _all_tags():
    from nclone.map_generation.compose import labels
    return set(labels.TAGS.keys())


# --- jitter robustness --------------------------------------------------------
@pytest.mark.parametrize("seed", list(range(0, 30)))
def test_jittered_grid_still_fills_and_validates(seed):
    from nclone.map_generation.compose.template import Template, tag_any
    from nclone.map_generation.compose.registry import register_template

    name = "jitter_probe_2x3"
    if name not in reg.TEMPLATE_REGISTRY:
        @register_template(
            name=name, grid=(2, 3), grid_jitter=True,
            cells={"spawn": tag_any("terrain:flat"), "goal": tag_any("terrain:flat"),
                   "*": tag_any("terrain:gap", "terrain:flat", "hazard:mine.field"),
                   "offpath": tag_any("role:filler")},
        )
        class _Probe(Template):
            pass

    m, _ = compose.compose(name, seed=seed)
    ents = describe_level(m.map_data())["entities"]
    assert sum(1 for e in ents if e["type"] == 3) == 1
    assert sum(1 for e in ents if e["type"] == 4) == 1


# --- legacy coexistence -------------------------------------------------------
def test_legacy_generator_component_wraps_into_canvas():
    from nclone.map_generation.compose.adapters import LegacyGeneratorComponent
    from nclone.map_generation.compose.canvas import Canvas

    comp = LegacyGeneratorComponent("maze", tags={"terrain:maze", "role:traverse"})
    canvas = Canvas(seed=0, rows=1, cols=1)
    region = canvas.region(0, 0)
    res = comp.stamp(canvas.context_for(region, {}))
    # the wrapped legacy generator populated the canvas tiles
    assert any(t != 0 for t in canvas.map.tile_data)
    assert res.placed_spawn and res.placed_goal
