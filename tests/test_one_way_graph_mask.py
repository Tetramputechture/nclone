"""Directional graph-edge masking for one-way platforms (entity type 11).

A one-way platform severs graph traversals that cross it from the solid (+normal)
side while leaving the permissive-side traversal open. These tests pin the
geometry directly on ``EntityMask`` and confirm the gym entity extractor surfaces
one-ways into the level-data the mask consumes.
"""

from __future__ import annotations

from nclone.graph.reachability.entity_mask import EntityMask
from nclone.constants.entity_types import EntityType


def _mask_with_one_way(orientation: int) -> EntityMask:
    # world = inner + NODE_WORLD_COORD_OFFSET (24); inner center -> (100, 100).
    level = {
        "entities": [
            {"type": EntityType.ONE_WAY, "x": 124, "y": 124, "orientation": orientation}
        ],
        "switch_states": {},
    }
    return EntityMask(level)


def test_one_way_floor_blocks_downward_only():
    """Orientation 6 (normal up): blocks dropping DOWN through it from above;
    passing UP through it from below stays open."""
    em = _mask_with_one_way(6)
    assert len(em.one_ways) == 1
    adj = {
        (100, 90): [((100, 110), 1.0)],   # down through, from above -> blocked
        (100, 110): [((100, 90), 1.0)],   # up through, from below -> open
        (200, 90): [((200, 110), 1.0)],   # different column, no crossing -> open
        (90, 100): [((110, 100), 1.0)],   # tangential move (d·n == 0) -> open
    }
    blocked = em.get_blocked_edges(adj)
    assert blocked == {((100, 90), (100, 110))}


def test_one_way_wall_blocks_one_direction():
    """Orientation 0 (normal right): blocks moving LEFT across it; moving RIGHT
    across it stays open."""
    em = _mask_with_one_way(0)
    adj = {
        (110, 100): [((90, 100), 1.0)],   # leftward crossing -> blocked
        (90, 100): [((110, 100), 1.0)],   # rightward crossing -> open
    }
    blocked = em.get_blocked_edges(adj)
    assert blocked == {((110, 100), (90, 100))}


def test_no_one_way_no_blocked_edges():
    em = EntityMask({"entities": [], "switch_states": {}})
    assert em.get_blocked_edges({(0, 0): [((10, 0), 1.0)]}) == set()
    # No adjacency -> empty, even if one-ways exist.
    em2 = _mask_with_one_way(6)
    assert em2.get_blocked_edges(None) == set()


def test_extractor_surfaces_one_ways():
    """The gym entity extractor must emit one-way dicts (with orientation) so the
    EntityMask can see them. Verified on both sim backends."""
    import pytest

    from nclone.gym_environment.entity_extractor import EntityExtractor
    from nclone.map_generation.map import Map

    m = Map()
    for x in range(2, 41):
        m.set_tile(x, 18, 1)
    m.set_ninja_spawn(20, 16)
    m.add_entity(3, 38, 17, 0, 0, switch_x=36, switch_y=17)
    m.add_entity(11, 12, 10, 6, 0)  # one-way, orientation 6
    md = [int(v) for v in m.map_data()]

    for backend in ("python", "native"):
        if backend == "python":
            from nclone.nplay_headless import NPlayHeadless

            h = NPlayHeadless()
        else:
            from nclone.native_headless import NativePlayHeadless

            h = NativePlayHeadless()
        h.load_map_from_map_data(list(md))
        h.reset()
        ents = EntityExtractor(h).extract_graph_entities()
        ow = [e for e in ents if e["type"] == EntityType.ONE_WAY]
        assert len(ow) == 1, f"[{backend}] expected one one-way, got {len(ow)}"
        assert ow[0]["orientation"] == 6, f"[{backend}] orientation lost"
