"""Physics-preserving map augmentation transforms operating on raw ``map_data``.

The fundamental augmentation primitive is :func:`mirror_map_data`, a clean,
involutive left-right reflection of a level expressed as a raw ``map_data``
integer list (the same format produced by :meth:`Map.map_data` and read from
official ``.map`` binaries). Operating directly on ``map_data`` -- rather than
on :class:`~nclone.map_generation.map.Map` objects -- lets the same primitive
augment official binary maps and procedurally generated dataset levels
uniformly, and keeps the transform a *pure function* that is byte-stable under
double application.

Why only a horizontal mirror? N++ physics has gravity along +y, so a vertical
flip (or a rotation) would produce a level the ninja cannot play under the same
physics. A horizontal mirror is the one non-trivial isometry that preserves
solvability, so it is the only geometric augmentation exposed here. It cleanly
doubles a dataset.

Correctness notes (all empirically verified against the official corpus):

* The reflection axis is ``(MAP_TILE_WIDTH + 2) * GRID_SIZE_FACTOR`` units.
  Tile data is stored as the ``MAP_TILE_WIDTH``-wide *interior* grid, but entity
  coordinates live in the *absolute* frame that includes the 1-tile border on
  each side, so the shared reflection is ``u -> 176 - u`` for both tiles and
  entities (44 * 4 = 176 with the default constants).
* Every entity record is a fixed 5-byte ``[type, x, y, orientation, mode]``
  slot. Exit doors / locked doors store their switch as a *separate* record (an
  exit switch, or the door's phantom switch slot whose ``x, y`` are the switch
  coordinates). Mirroring every record independently therefore mirrors the
  switch correctly too -- no special-casing, and the door<->switch pairing
  (which the loader resolves positionally) is preserved because record order and
  counts are untouched.
* ``mirror_map_data(mirror_map_data(md)) == md`` byte-for-byte.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterator, List, Tuple

from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from ..map_generation.constants import GRID_SIZE_FACTOR

# Reflection axis in map-data units. Entities live in the absolute coordinate
# frame (interior grid + 1-tile border each side), so the axis is the full
# 44-tile width, not the stored 42-tile interior width.
MIRROR_AXIS_UNITS: int = int((MAP_TILE_WIDTH + 2) * GRID_SIZE_FACTOR)

# map_data layout landmarks (see Map.map_data / MapLoader.load_map_entities).
_TILE_START = 184
_TILE_END = 1150  # exclusive; MAP_TILE_WIDTH * MAP_TILE_HEIGHT tiles
_NINJA_X = 1231
_NINJA_Y = 1232
_NINJA_ORIENTATION = 1233
_ENTITY_START = 1235
_ENTITY_STRIDE = 5

# Tile-type left<->right reflection. A tile whose geometry is asymmetric about
# the vertical axis (slopes, quarter-moons, half tiles) maps to its mirror
# variant; symmetric tiles (and full/empty) map to themselves.
_TILE_MIRROR: Dict[int, int] = {}
for _a, _b in (
    (3, 5),  # half tiles (vertical orientation pair)
    (6, 7),  # 45-degree slopes
    (8, 9),
    (10, 11),  # quarter moons (convex)
    (12, 13),
    (14, 15),  # quarter moons (concave)
    (16, 17),
    (18, 19),  # short mild slopes
    (20, 21),
    (22, 23),  # raised mild slopes
    (24, 25),
    (26, 27),  # short steep slopes
    (28, 29),
    (30, 31),  # raised steep slopes
    (32, 33),
):
    _TILE_MIRROR[_a] = _b
    _TILE_MIRROR[_b] = _a

# Entity types whose ``orientation`` byte encodes a compass direction that must
# be reflected. The 8-way orientation reflects via this map; the purely vertical
# orientations (0 = right-of-axis-symmetric "up", 4 = "down") are unchanged.
_ORIENT_TYPES = frozenset({5, 6, 8, 10, 11, 14, 15, 20, 23, 26})
_ORIENT_FLIP: Dict[int, int] = {1: 3, 2: 6, 3: 1, 5: 7, 6: 2, 7: 5}

# Drone entity types whose ``mode`` byte encodes a chirality (clockwise vs
# counter-clockwise wall-follow / wander) that must be swapped under reflection.
_DRONE_TYPES = frozenset({14, 15, 26})
_DRONE_MODE_SWAP: Dict[int, int] = {0: 1, 1: 0, 2: 3, 3: 2}


#: Entity types whose reflection is KNOWN to be wrong, with the defect.
#:
#: 1. ``_ORIENT_FLIP`` above is provably wrong for the four AXIS-ALIGNED
#:    orientations. Checked against the simulator's own
#:    ``map_orientation_to_vector``: a horizontal mirror must map 0 <-> 4
#:    (right / left) and leave 2 and 6 (down / up) alone. The table does the
#:    opposite -- it leaves 0 and 4 untouched (the ``not in (0, 4)`` guard) and
#:    swaps 2 <-> 6. The four diagonals are correct. Its comment shows the
#:    misconception: it calls 0 "up" and 4 "down", but the sim says 0 is (+1, 0).
#: 2. Drone TRAJECTORIES do not reflect even once the orientation is corrected.
#:    Measured 2026-08-03: spawn positions mirror exactly but the paths diverge
#:    from frame 1, up to 312 px over 200 frames on `official/R/173 variational
#:    mechanics`. Fixing the orientation shrinks it (312 -> 240 px, and 252 -> 36
#:    px on `R/034 oh, they'll swarm ya`) but does not zero it, so at least one
#:    further defect remains and is not yet located.
#:
#: Consequence for LEVEL mirroring (Tier 0): the mirrored level is still a valid
#: level, but it is not the reflection of the original -- a wrongly-oriented
#: launcher or patrol can make it unsolvable, and an unsolvable level in the pool
#: reads as a policy failure and stalls the phase gate on merit it never lacked.
#: Consequence for SYNTHESISED observation mirroring (the aux / equivariance
#: tiers): M(obs) places threats where the mirrored level does not have them.
#:
#: This set is used by :func:`is_mirror_faithful` to *exclude* affected maps
#: rather than to mirror them wrongly. It went unnoticed for a long time because
#: mines carry no orientation byte, so a mines-only corpus cannot exercise it.
_MIRROR_UNFAITHFUL_TYPES = frozenset(
    _ORIENT_TYPES
    | _DRONE_TYPES
    # Every phase-1 pure hazard that MOVES or AIMS. Kept explicit rather than
    # imported from `constants.entity_types.PURE_HAZARD_TYPES` so this module
    # stays a leaf with no dependency on the training-scope profiles.
    | {
        12,  # drone chaingun
        13,  # drone laser
        16,  # floor guard
        18,  # rocket
        19,  # turret
        22,  # evil ninja -- replays the ninja's own tape, so it inherits every
        #      divergence in the ninja's path rather than causing one
        25,  # death ball
        27,  # eye bat
        28,  # shove thwump
    }
)


def unfaithful_mirror_types(map_data: List[int]) -> Tuple[int, ...]:
    """Entity types present in ``map_data`` whose reflection is known wrong.

    Empty tuple means the level mirrors exactly. See
    :data:`_MIRROR_UNFAITHFUL_TYPES` for what "wrong" means per type.
    """
    found = set()
    i = _ENTITY_START
    n = len(map_data)
    while i + 4 < n:
        etype = map_data[i]
        if etype in _MIRROR_UNFAITHFUL_TYPES:
            found.add(int(etype))
        i += _ENTITY_STRIDE
    return tuple(sorted(found))


def is_mirror_faithful(map_data: List[int]) -> bool:
    """Whether :func:`mirror_map_data` produces the TRUE reflection of this level.

    A mines-and-gold level mirrors exactly and is safe to augment. A level
    carrying any orientation-bearing or moving entity does not -- see
    :data:`_MIRROR_UNFAITHFUL_TYPES`.
    """
    return not unfaithful_mirror_types(map_data)


def mirror_map_data(map_data: List[int]) -> List[int]:
    """Return a horizontally mirrored copy of ``map_data``.

    Pure and involutive: ``mirror_map_data(mirror_map_data(md)) == md``.

    Args:
        map_data: Raw level data in simulator format (header, tiles, entity
            records). Not modified.

    Returns:
        A new ``map_data`` list reflected left-to-right.
    """
    md = list(map_data)
    out = list(md)
    axis = MIRROR_AXIS_UNITS
    w, h = MAP_TILE_WIDTH, MAP_TILE_HEIGHT

    # Tiles: reflect column index within the interior grid and swap tile variant.
    tiles = md[_TILE_START:_TILE_END]
    mirrored_tiles = [0] * len(tiles)
    for y in range(h):
        row = y * w
        for x in range(w):
            t = tiles[x + row]
            mirrored_tiles[(w - 1 - x) + row] = _TILE_MIRROR.get(t, t)
    out[_TILE_START:_TILE_END] = mirrored_tiles

    # Ninja spawn: reflect x, flip facing orientation (+-1).
    out[_NINJA_X] = axis - md[_NINJA_X]
    if md[_NINJA_ORIENTATION] in (-1, 1):
        out[_NINJA_ORIENTATION] = -md[_NINJA_ORIENTATION]

    # Entities: uniform per-record reflection. Switch records (exit switch /
    # door phantom slots) are ordinary records and reflect correctly here.
    i = _ENTITY_START
    n = len(md)
    while i + 4 < n:
        etype = md[i]
        out[i + 1] = axis - md[i + 1]
        if etype in _ORIENT_TYPES and md[i + 3] not in (0, 4):
            out[i + 3] = _ORIENT_FLIP.get(md[i + 3], md[i + 3])
        if etype in _DRONE_TYPES:
            out[i + 4] = _DRONE_MODE_SWAP.get(md[i + 4], md[i + 4])
        i += _ENTITY_STRIDE

    return out


def identity_map_data(map_data: List[int]) -> List[int]:
    """Return an unmodified copy of ``map_data`` (the no-op augmentation)."""
    return list(map_data)


def canonical_map_data(map_data: List[int]) -> Tuple[List[int], bool]:
    """Pick one of ``{map_data, mirror(map_data)}`` deterministically.

    Returns ``(canonical, was_mirrored)`` where ``canonical`` is the
    lexicographically smaller of the level and its reflection, and
    ``was_mirrored`` says whether the caller's level is the reflected one.

    This is the lever that makes *tie-breaking* computations reflection-exact.
    A Dijkstra heap ordered by ``(cost, x, y)``, or a route planner choosing
    between equal-length legs, cannot be made symmetric by any pure function of
    position -- but it does not have to be. Run it once on the canonical
    orientation and reflect the RESULT, and a level and its mirror produce
    exactly mirrored fields, bit for bit, tie-breaks included.

    Both members of the pair map to the same canonical form, so a cache keyed on
    it is shared between a level and its mirror -- which also halves the build
    cost under Tier 0 level mirroring.

    A left-right symmetric level is its own mirror; it reports
    ``was_mirrored=False`` and needs no reflection.
    """
    md = list(map_data)
    mirrored = mirror_map_data(md)
    if md <= mirrored:
        return md, False
    return mirrored, True


# Registry of named, physics-preserving augmentations. ``identity`` is included
# so callers can request the original alongside its variants from one code path.
AUGMENTATIONS: Dict[str, Callable[[List[int]], List[int]]] = {
    "identity": identity_map_data,
    "mirror_h": mirror_map_data,
}

# Augmentations that genuinely transform the map (exclude the no-op).
NONTRIVIAL_AUGMENTATIONS: Tuple[str, ...] = ("mirror_h",)


def augment_map_data(map_data: List[int], kind: str) -> List[int]:
    """Apply a named augmentation from :data:`AUGMENTATIONS`.

    Args:
        map_data: Raw level data; not modified.
        kind: Augmentation name (e.g. ``"mirror_h"`` or ``"identity"``).

    Returns:
        The transformed ``map_data``.

    Raises:
        KeyError: If ``kind`` is not a registered augmentation.
    """
    try:
        fn = AUGMENTATIONS[kind]
    except KeyError as exc:
        raise KeyError(
            f"Unknown augmentation {kind!r}; available: {sorted(AUGMENTATIONS)}"
        ) from exc
    return fn(map_data)


def iter_augmented(
    map_data: List[int],
    kinds: Tuple[str, ...] = NONTRIVIAL_AUGMENTATIONS,
    *,
    dedup: bool = True,
) -> Iterator[Tuple[str, List[int]]]:
    """Yield ``(kind, augmented_map_data)`` for each requested augmentation.

    Args:
        map_data: Source level data; not modified.
        kinds: Augmentation names to apply, in order.
        dedup: When True, skip an augmentation whose output is byte-identical to
            the source (e.g. mirroring a left-right symmetric level), so that
            symmetric levels do not produce duplicate dataset entries.

    Yields:
        ``(kind, map_data)`` pairs for each non-skipped augmentation.
    """
    base = list(map_data)
    for kind in kinds:
        variant = augment_map_data(base, kind)
        if dedup and variant == base:
            continue
        yield kind, variant
