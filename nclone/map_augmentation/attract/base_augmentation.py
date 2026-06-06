"""Abstract base class for attract-file augmentations.

An augmentation takes the fully decoded representation of an npp_attract file
and returns a modified copy that, when replayed through the N++ physics
simulator, still produces a successful episode (ninja reaches the exit).

Decoded attract data dict keys:
    tiles       : list[int]  -- 966 tile IDs (42x23, index = x + y*42)
    entities    : list[dict] -- entity records with keys type/x/y/orientation/mode
    ninja_spawn : tuple[int,int] -- (x, y) in map-data units
    header      : bytes      -- raw 184-byte npp_attract header (preserved verbatim)
    demo_data   : list[int]  -- input sequence, one byte per frame (values 0-7)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


# Type alias for the decoded attract-data dictionary used throughout this package.
AttractData = Dict[str, Any]


class BaseAugmentation(ABC):
    """Abstract base class for all attract-file augmentations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in output filenames, e.g. 'hmirror'."""

    @abstractmethod
    def augment(self, data: AttractData) -> AttractData:
        """Return a new AttractData that is an augmented copy of *data*.

        The input dict must not be mutated.  All values that are not modified
        by the augmentation should be copied (shallow-copy is fine for
        unchanged scalars/bytes; lists must be new objects).

        Args:
            data: Decoded attract-file data as returned by
                  ``NppAttractDecoder.decode_npp_attract_file()``.

        Returns:
            A new AttractData representing the augmented demonstration.

        Raises:
            ValueError: If the input data is structurally invalid for this
                        augmentation (e.g. empty tile list).
        """
