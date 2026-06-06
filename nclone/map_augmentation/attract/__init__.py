"""Attract-file augmentation package.

Public API
----------
BaseAugmentation          -- abstract base for all augmentations
HorizontalMirror          -- horizontal mirror (guaranteed valid, 2x multiplier)
UnreachableTileRandomizer -- randomize tiles in areas unreachable from spawn
ReachableAreaTranslator   -- translate the reachable region within the grid
AttractAugmentor          -- decode → augment → encode → save orchestrator
verify_sample             -- batch simulator verification of augmented files
"""

from .base_augmentation import AttractData, BaseAugmentation
from .horizontal_mirror import HorizontalMirror
from .unreachable_tile_randomizer import UnreachableTileRandomizer
from .reachable_area_translator import ReachableAreaTranslator
from .attract_augmentor import AttractAugmentor
from .verify_augmentation import verify_sample, verify_replay_file, VerificationReport

__all__ = [
    "AttractData",
    "BaseAugmentation",
    "HorizontalMirror",
    "UnreachableTileRandomizer",
    "ReachableAreaTranslator",
    "AttractAugmentor",
    "verify_sample",
    "verify_replay_file",
    "VerificationReport",
]
