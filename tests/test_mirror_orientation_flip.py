"""The horizontal mirror must reflect the 8-way orientation byte correctly.

The expectations here are DERIVED from the simulator's own
`physics.map_orientation_to_vector`, not written out by hand -- a hand-written
table is exactly what was wrong before. A horizontal mirror negates x and leaves
y alone, so the correct mapping is whatever that operation says it is.

History this pins: the old table was `{1: 3, 2: 6, 3: 1, 5: 7, 6: 2, 7: 5}` with
a `not in (0, 4)` guard at the call site. The four diagonals were right and all
four axis-aligned cases were wrong -- it left right(0)/left(4) untouched and
swapped down(2)/up(6), precisely backwards. It survived a long time because mines
carry no orientation byte, so a mines-only corpus never exercises it.
"""

import math

from nclone.map_augmentation.transforms import (
    _ENTITY_STRIDE,
    _ORIENT_FLIP,
    _ORIENT_TYPES,
    mirror_map_data,
)
from nclone.physics import map_orientation_to_vector


def _mirrored_orientation(o: int) -> int:
    """The orientation whose vector is (-x, y) of orientation `o`."""
    vx, vy = map_orientation_to_vector(o)
    for cand in range(8):
        cx, cy = map_orientation_to_vector(cand)
        if math.isclose(cx, -vx, abs_tol=1e-9) and math.isclose(cy, vy, abs_tol=1e-9):
            return cand
    raise AssertionError(f"no mirrored orientation for {o}")


def test_flip_matches_the_simulators_own_geometry():
    for o in range(8):
        assert _ORIENT_FLIP[o] == _mirrored_orientation(o), (
            f"orientation {o} = {map_orientation_to_vector(o)} mirrors to "
            f"{_ORIENT_FLIP[o]}, but the sim's geometry says "
            f"{_mirrored_orientation(o)}"
        )


def test_right_and_left_swap():
    """The exact case the old `not in (0, 4)` guard skipped."""
    assert map_orientation_to_vector(0) == (1, 0)
    assert map_orientation_to_vector(4) == (-1, 0)
    assert _ORIENT_FLIP[0] == 4
    assert _ORIENT_FLIP[4] == 0


def test_down_and_up_are_unchanged():
    """The exact case the old table wrongly swapped."""
    assert map_orientation_to_vector(2) == (0, 1)
    assert map_orientation_to_vector(6) == (0, -1)
    assert _ORIENT_FLIP[2] == 2
    assert _ORIENT_FLIP[6] == 6


def test_flip_is_an_involution():
    """Mirroring twice is the identity, so the map must be its own inverse."""
    for o in range(8):
        assert _ORIENT_FLIP[_ORIENT_FLIP[o]] == o


def test_mirroring_map_data_rewrites_the_orientation_byte():
    """End-to-end through `mirror_map_data`, including the 0/4 case."""
    orient_type = sorted(_ORIENT_TYPES)[0]
    # A map_data buffer long enough to carry one entity record in the tail.
    md = [0] * (1250 + _ENTITY_STRIDE)
    base = 1250
    for orientation in range(8):
        md[base + 0] = orient_type
        md[base + 1] = 40  # x in map-data units
        md[base + 2] = 40
        md[base + 3] = orientation
        md[base + 4] = 0
        out = mirror_map_data(md)
        assert out[base + 3] == _mirrored_orientation(orientation), (
            f"mirror_map_data left orientation {orientation} as "
            f"{out[base + 3]}"
        )
