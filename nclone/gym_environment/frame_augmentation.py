"""Frame augmentation utilities for preprocessing observation frames.

This module implements various image augmentation techniques using the albumentations library.
The augmentations are applied randomly to input frames to increase training diversity.
"""

import numpy as np
import albumentations as A
from typing import Optional, List, Dict, Any, Tuple
import functools


@functools.lru_cache(maxsize=None)
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
    if not frames:
        return []

    if seed is not None:
        np.random.seed(seed)

    # Ensure frames are in uint8 format for albumentations
    # (assuming all frames have the same dtype, apply to first for check,
    # but albumentations will handle individual frame types if needed,
    # though it's better if they are consistent)
    first_frame_uint8 = frames[0].astype(np.uint8)

    # Get augmentation pipeline (it will be cached after the first call with a given 'p')
    # Assuming the 'p' value used in apply_augmentation is consistent,
    # or that get_augmentation_pipeline is called with a default 'p'
    # that matches the implicit 'p' in the original apply_augmentation.
    # For simplicity, we'll assume the default p=0.5 is used.
    # If 'p' needs to be configurable here, it should be passed to this function.
    transform = get_augmentation_pipeline() # Default p will be used

    # Apply augmentation to the first frame and get parameters
    data = transform(image=first_frame_uint8)
    augmented_frames = [data['image']]
    saved_params = data['replay']

    # Apply exact same augmentation to remaining frames
    for frame in frames[1:]:
        frame_uint8 = frame.astype(np.uint8)
        aug_frame = transform.replay(saved_params, image=frame_uint8)['image']
        augmented_frames.append(aug_frame)

    return augmented_frames
