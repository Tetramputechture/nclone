"""Attract-file augmentation package.

Public API
----------
BaseAugmentation    -- abstract base for all augmentations
HorizontalMirror    -- horizontal mirror (guaranteed valid, 2x multiplier)
AttractAugmentor    -- decode → augment → encode → save orchestrator
verify_sample       -- batch simulator verification of augmented files
"""

from .base_augmentation import AttractData, BaseAugmentation
from .horizontal_mirror import HorizontalMirror
from .attract_augmentor import AttractAugmentor
from .verify_augmentation import verify_sample, verify_replay_file, VerificationReport

__all__ = [
    "AttractData",
    "BaseAugmentation",
    "HorizontalMirror",
    "AttractAugmentor",
    "verify_sample",
    "verify_replay_file",
    "VerificationReport",
]
