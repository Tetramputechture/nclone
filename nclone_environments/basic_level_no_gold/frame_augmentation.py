"""Frame augmentation utilities for preprocessing observation frames.

This module implements various image augmentation techniques using the albumentations library.
The augmentations are applied randomly to input frames to increase training diversity.
"""

import numpy as np
import albumentations as A
from typing import Optional
from nclone_environments.basic_level_no_gold.constants import PLAYER_FRAME_WIDTH, PLAYER_FRAME_HEIGHT


def get_augmentation_pipeline(p: float = 0.5) -> A.Compose:
    """Creates an augmentation pipeline with all supported transformations.

    Args:
        p: Probability of applying each augmentation.

    Returns:
        Composed albumentations pipeline
    """
    return A.Compose([
        # Crop - Randomly crop and resize back
        A.RandomResizedCrop(
            height=PLAYER_FRAME_HEIGHT,
            width=PLAYER_FRAME_WIDTH,
            scale=(0.5, 1.0),
            p=p
        ),

        # Translate - Shift image
        A.ShiftScaleRotate(
            shift_limit=0.15,
            scale_limit=0,
            rotate_limit=0,
            p=p
        ),

        # Window - Apply random window mask
        A.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2),
            contrast_limit=(-0.2, 0.2),
            p=p
        ),

        # Cutout - Apply random rectangular masks
        A.CoarseDropout(
            max_holes=4,
            max_height=24,
            max_width=24,
            min_height=12,
            min_width=12,
            fill_value=0,
            p=p
        ),

        # Flip - Horizontal flip
        A.HorizontalFlip(p=p),

        # Rotate - Random rotation
        A.Rotate(
            limit=20,
            p=p
        ),

        # Random Gaussian Noise
        A.GaussNoise(
            var_limit=(5.0, 30.0),
            p=p
        ),
    ])


def apply_augmentation(
    frame: np.ndarray,
    seed: Optional[int] = None
) -> np.ndarray:
    """Applies random augmentations to the input frame.

    Args:
        frame: Input frame of shape (H, W, C)
        seed: Optional random seed for reproducibility

    Returns:
        Augmented frame with same shape as input
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure frame is in uint8 format for albumentations
    frame = frame.astype(np.uint8)

    # Get augmentation pipeline
    transform = get_augmentation_pipeline()

    # Apply augmentations
    augmented = transform(image=frame)['image']

    return augmented
