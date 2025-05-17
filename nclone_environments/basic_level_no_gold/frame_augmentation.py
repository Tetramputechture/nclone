"""Frame augmentation utilities for preprocessing observation frames.

This module implements various image augmentation techniques using the albumentations library.
The augmentations are applied randomly to input frames to increase training diversity.
"""

import numpy as np
import albumentations as A
from typing import Optional, List, Dict, Any, Tuple


def get_augmentation_pipeline(p: float = 0.5) -> A.ReplayCompose:
    """Creates an augmentation pipeline with all supported transformations.

    Args:
        p: Probability of applying each augmentation.

    Returns:
        ReplayCompose pipeline that can record and replay exact transformations
    """
    return A.ReplayCompose([
        # Cutout - Apply random rectangular masks
        A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(10, 20),
            hole_width_range=(10, 20),
            p=p
        ),
    ])


def apply_augmentation(
    frame: np.ndarray,
    seed: Optional[int] = None,
    saved_params: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Applies random augmentations to the input frame.

    Args:
        frame: Input frame of shape (H, W, C)
        seed: Optional random seed for reproducibility
        saved_params: Optional parameters from a previous transform to replay

    Returns:
        Tuple of (augmented frame, saved parameters for replay)
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure frame is in uint8 format for albumentations
    frame = frame.astype(np.uint8)

    # Get augmentation pipeline
    transform = get_augmentation_pipeline()

    # Apply augmentations
    if saved_params is not None:
        # Replay exact same augmentation
        augmented = transform.replay(saved_params, image=frame)['image']
        return augmented, None
    else:
        # Generate new random augmentation and save parameters
        data = transform(image=frame)
        return data['image'], data['replay']


def apply_consistent_augmentation(
    frames: List[np.ndarray],
    seed: Optional[int] = None
) -> List[np.ndarray]:
    """Applies the same random augmentations to all frames in a stack.

    Args:
        frames: List of frames to augment, each of shape (H, W, C)
        seed: Optional random seed for reproducibility

    Returns:
        List of augmented frames with same shapes as inputs
    """
    if seed is not None:
        np.random.seed(seed)

    if not frames:
        return []

    # Apply augmentation to first frame and get parameters
    first_frame_aug, saved_params = apply_augmentation(frames[0])
    augmented_frames = [first_frame_aug]

    # Apply exact same augmentation to remaining frames
    for frame in frames[1:]:
        aug_frame, _ = apply_augmentation(frame, saved_params=saved_params)
        augmented_frames.append(aug_frame)

    return augmented_frames
