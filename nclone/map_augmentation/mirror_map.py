"""Horizontal map mirroring on :class:`~nclone.map_generation.map.Map` objects.

This is a thin :class:`Map`-level wrapper around the raw-``map_data`` primitive
:func:`nclone.map_augmentation.transforms.mirror_map_data`, which is the
authoritative, verified implementation (pure, involutive, and solvability- and
entity-count-preserving). Earlier versions reconstructed the mirrored map via
``Map.add_entity`` and drifted entity positions by two tiles while duplicating
the exit switch on every call; delegating to the primitive avoids both bugs.
"""

from ..map_generation.map import Map
from .transforms import mirror_map_data


def mirror_map_horizontally(original_map: Map) -> Map:
    """Create a horizontally mirrored copy of ``original_map``.

    Tiles, the ninja spawn, and all entities are reflected left-to-right; slope
    and quarter-moon tiles swap to their mirror variants, directional entity
    orientations flip, and drone chirality (CW<->CCW) is swapped so gameplay is
    preserved under reflection.

    Args:
        original_map: The :class:`Map` to mirror. Not modified.

    Returns:
        A new mirrored :class:`Map`.
    """
    return Map.from_map_data(mirror_map_data(original_map.map_data()))
