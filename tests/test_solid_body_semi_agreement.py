"""The two solid-body half-extent tables must agree -- with each other, and with
the simulator that actually collides.

`_SOLID_BODY_SEMI` is DUPLICATED:

* `graph.reachability.entity_mask._SOLID_BODY_SEMI` -- blocks edges in the
  reachability graph;
* `value_function.free_space_distance._SOLID_BODY_SEMI` -- blocks cells in the
  global potential field.

If they drift, the lattice and the global field disagree about what a level even
contains, and the disagreement is silent: both still build, both still produce a
potential, and the reward just becomes wrong in a way no gate reads. The free
space source hashes its table into its cache key, so a drift there also
invalidates caches asymmetrically.

The third assertion is the one that matters most, and it is the one a
table-vs-table check would miss: both tables can agree with each other and still
be wrong about the sim. `physics_constants.BOUNCE_BLOCK_SEMI_SIDE` is 4.5 and is
read by NOTHING -- the Python collision uses `EntityBounceBlock.SEMI_SIDE` (9)
and `nclone-native/src/constants.rs` hardcodes 9.0 with a comment saying exactly
that. So 9 is the value the simulator and both tables agree on, and a future
"cleanup" that wires the tables to the 4.5 constant would halve every bounce
block's blocking footprint without failing anything else.
"""

from nclone.constants.entity_types import EntityType
from nclone.constants.physics_constants import (
    SHOVE_THWUMP_SEMI_SIDE,
    THWUMP_SEMI_SIDE,
)
from nclone.entity_classes.entity_bounce_block import EntityBounceBlock
from nclone.graph.reachability.entity_mask import EntityMask
from nclone.value_function.free_space_distance import _SOLID_BODY_SEMI as FS_SEMI

GRAPH_SEMI = EntityMask._SOLID_BODY_SEMI


def test_the_two_tables_cover_the_same_entity_types():
    assert set(GRAPH_SEMI) == set(FS_SEMI), (
        "reachability graph and global potential disagree about which entity "
        "types have a solid body"
    )


def test_the_two_tables_agree_on_every_half_extent():
    assert dict(GRAPH_SEMI) == dict(FS_SEMI), (
        f"half-extent drift: graph={dict(GRAPH_SEMI)} field={dict(FS_SEMI)}"
    )


def test_half_extents_match_the_simulator_that_actually_collides():
    """Both tables agreeing is not enough -- they must match the sim."""
    authoritative = {
        EntityType.BOUNCE_BLOCK: float(EntityBounceBlock.SEMI_SIDE),
        EntityType.THWUMP: float(THWUMP_SEMI_SIDE),
        EntityType.SHWUMP: float(SHOVE_THWUMP_SEMI_SIDE),
    }
    for entity_type, semi in authoritative.items():
        assert float(GRAPH_SEMI[entity_type]) == semi, (
            f"{entity_type!r}: reachability graph uses "
            f"{GRAPH_SEMI[entity_type]}, simulator collides at {semi}"
        )
        assert float(FS_SEMI[entity_type]) == semi, (
            f"{entity_type!r}: global potential uses {FS_SEMI[entity_type]}, "
            f"simulator collides at {semi}"
        )


def test_bounce_block_is_not_wired_to_the_dead_constant():
    """Pin the 4.5-vs-9 trap so a 'cleanup' cannot land silently."""
    from nclone.constants.physics_constants import BOUNCE_BLOCK_SEMI_SIDE

    assert float(EntityBounceBlock.SEMI_SIDE) == 9.0
    assert float(BOUNCE_BLOCK_SEMI_SIDE) == 4.5, (
        "if this changed, re-check which value the tables and the Rust "
        "constant now follow"
    )
    assert float(FS_SEMI[EntityType.BOUNCE_BLOCK]) == 9.0
    assert float(GRAPH_SEMI[EntityType.BOUNCE_BLOCK]) == 9.0
