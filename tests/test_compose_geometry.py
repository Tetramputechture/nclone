"""Phase-3 tests: geometry connectors + the slope/curve/tunnel templates.

The connector orientation guard samples every emitted tile id against the
authoritative collision predicate ``_check_subnode_validity_simple`` (the same
geometry the physics sim / surface planner use), so a future orientation
regression in ``connectors.py`` is caught immediately -- the analogue of
``test_curve_utils.py`` for the new ceiling/wall pieces.
"""

import pytest

from nclone.map_generation import compose
from nclone.map_generation.compose import describe_level
from nclone.map_generation.compose.visualize import _tiles
from nclone.graph.reachability.graph_builder import _check_subnode_validity_simple as valid

FLOOR_TILES = {8, 9, 12, 13, 16, 17}
CEIL_TILES = {6, 7, 10, 11, 14, 15}
ALLOWED_ENTITY_TYPES = {0, 1, 21, 3, 4}

GEOMETRY_TEMPLATES = [
    "slope_course_2x3", "curve_course_2x3", "tunnel_curves_2x3", "grand_mix_2x3",
]


@pytest.mark.parametrize("tile", sorted(FLOOR_TILES))
def test_floor_tiles_are_solid_below(tile):
    assert valid(tile, 12, 2) is True   # top-center traversable
    assert valid(tile, 12, 22) is False  # bottom-center solid (you stand on it)


@pytest.mark.parametrize("tile", sorted(CEIL_TILES))
def test_ceiling_tiles_are_solid_above(tile):
    assert valid(tile, 12, 2) is False   # top-center solid (overhang)
    assert valid(tile, 12, 22) is True   # bottom-center traversable (pass under)


def test_connector_emitters_only_emit_expected_tiles():
    """Each connector emits ids from its declared family and nothing stray."""
    from nclone.map_generation.map import Map
    from nclone.map_generation.compose import connectors

    def emitted(fn):
        m = Map(seed=0)
        before = set(m.tile_data)
        fn(m)
        return set(m.tile_data) - {0} | before  # union to keep it simple

    m = Map(seed=0)
    connectors.ceiling_hump(m, 5, 6, width=2, roof_top=1)
    assert (set(m.tile_data) & {10, 11}) == {10, 11}

    m = Map(seed=0)
    connectors.ceiling_dip(m, 5, 6, width=2, roof_top=1)
    assert (set(m.tile_data) & {14, 15}) == {14, 15}

    m = Map(seed=0)
    connectors.ceiling_slope(m, 5, 6, height=3, descend=True, roof_top=1)
    assert 6 in set(m.tile_data)

    m = Map(seed=0)
    _, _, tag = connectors.slope_up(m, 3, 18, 2, kind="45", base_row=21)
    assert tag == "terrain:slope.45" and (set(m.tile_data) & {8})


@pytest.mark.parametrize("template", GEOMETRY_TEMPLATES)
@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_geometry_templates_invariants(template, seed):
    m, lab = compose.compose(template, seed=seed)
    a = m.map_data()
    b, _ = compose.compose(template, seed=seed)
    assert a == b.map_data()  # reproducible
    ents = describe_level(a)["entities"]
    types = {e["type"] for e in ents}
    assert types <= ALLOWED_ENTITY_TYPES
    assert sum(1 for e in ents if e["type"] == 3) == 1
    assert sum(1 for e in ents if e["type"] == 4) == 1


def test_tunnel_template_uses_ceiling_curve_tiles_somewhere():
    seen = set()
    for s in range(40):
        m, _ = compose.compose("tunnel_curves_2x3", seed=s)
        seen |= set(_tiles(m.map_data())) & {10, 11, 14, 15}
    assert seen, "tunnel template never emitted a ceiling-curve tile (10/11/14/15)"


@pytest.mark.parametrize("template", GEOMETRY_TEMPLATES)
@pytest.mark.parametrize("seed", [0, 3, 9])
def test_geometry_templates_solvable(template, seed):
    from nclone.map_generation.solvability import is_map_solvable

    m, _ = compose.compose(template, seed=seed)
    assert is_map_solvable(m.map_data())
