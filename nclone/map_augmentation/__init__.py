"""Physics-preserving map augmentation.

Augmentations operate on raw ``map_data`` (see :mod:`.transforms`) so they apply
uniformly to official binary maps and procedurally generated dataset levels. The
only geometric augmentation is a horizontal mirror -- the one isometry that
preserves N++ gravity physics and therefore solvability.
"""

from .transforms import (
    AUGMENTATIONS,
    canonical_map_data,
    MIRROR_AXIS_UNITS,
    NONTRIVIAL_AUGMENTATIONS,
    augment_map_data,
    identity_map_data,
    iter_augmented,
    mirror_map_data,
)
from .mirror_map import mirror_map_horizontally

__all__ = [
    "AUGMENTATIONS",
    "canonical_map_data",
    "NONTRIVIAL_AUGMENTATIONS",
    "MIRROR_AXIS_UNITS",
    "augment_map_data",
    "identity_map_data",
    "iter_augmented",
    "mirror_map_data",
    "mirror_map_horizontally",
]
