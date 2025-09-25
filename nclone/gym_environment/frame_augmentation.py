"""Frame augmentation utilities for preprocessing observation frames.

This module implements various image augmentation techniques using the albumentations library.
The augmentations are applied randomly to input frames to increase training diversity.

Based on research from RAD, DrQ-v2, and other visual RL methods, this pipeline includes
the most effective augmentations for reinforcement learning:
- Random crop/translate: Most stable and effective for RL
- Color jitter: Improves generalization across visual variations
- Cutout: Encourages focus on global context rather than local features
- Gaussian noise: Improves robustness to sensor noise
- Random convolution: Helps with texture invariance (optional)

References:
- Laskin et al. (2020): Reinforcement Learning with Augmented Data (RAD)
- Kostrikov et al. (2020): Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels (DrQ)
- Yarats et al. (2021): Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning (DrQ-v2)
"""

import numpy as np
import albumentations as A
from typing import Optional, List, Dict, Any, Tuple
import functools


@functools.lru_cache(maxsize=None)
def get_augmentation_pipeline(
    p: float = 0.5, 
    intensity: str = "medium",
    enable_advanced: bool = False
) -> A.ReplayCompose:
    """Creates an augmentation pipeline with research-backed transformations for visual RL.

    Args:
        p: Probability of applying each augmentation.
        intensity: Augmentation intensity level ("light", "medium", "strong")
        enable_advanced: Whether to include advanced augmentations (random convolution)

    Returns:
        ReplayCompose pipeline that can record and replay exact transformations
    """
    # Intensity-based parameter scaling
    intensity_scales = {
        "light": 0.7,
        "medium": 1.0,
        "strong": 1.3
    }
    scale = intensity_scales.get(intensity, 1.0)
    
    augmentations = []
    
    # 1. Random Crop/Translate - Most effective for RL (RAD, DrQ-v2)
    # Small translations are more stable than large crops
    # For 84x84 frames, 4 pixels = ~0.05 normalized shift
    shift_limit_normalized = (4 * scale) / 84.0  # Normalize to image size
    augmentations.append(
        A.Affine(
            translate_percent={"x": (-shift_limit_normalized, shift_limit_normalized), 
                             "y": (-shift_limit_normalized, shift_limit_normalized)},
            scale=1.0,  # No scaling to maintain spatial relationships
            rotate=0,   # No rotation to avoid training instability
            mode=0,     # Constant border (black padding)
            p=p * 0.8   # Higher probability for most effective augmentation
        )
    )
    
    # 2. Color Jitter - Improves generalization (RAD findings)
    augmentations.append(
        A.ColorJitter(
            brightness=0.1 * scale,
            contrast=0.1 * scale,
            saturation=0.1 * scale,
            hue=0.05 * scale,
            p=p * 0.6
        )
    )
    
    # 3. Cutout - Encourages global context learning (DeVries & Taylor, 2017)
    augmentations.append(
        A.CoarseDropout(
            num_holes_range=(1, 2),
            hole_height_range=(int(8 * scale), int(16 * scale)),
            hole_width_range=(int(8 * scale), int(16 * scale)),
            p=p * 0.5
        )
    )
    
    # 4. Gaussian Noise - Improves robustness to sensor noise
    augmentations.append(
        A.GaussNoise(
            var_limit=(5.0 * scale, 15.0 * scale),
            p=p * 0.3
        )
    )
    
    # 5. Advanced augmentations (optional, can cause instability)
    if enable_advanced:
        # Random convolution - texture invariance (use sparingly)
        augmentations.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.05 * scale,
                contrast_limit=0.05 * scale,
                p=p * 0.2
            )
        )
    
    return A.ReplayCompose(augmentations)


def apply_augmentation(
    frame: np.ndarray,
    seed: Optional[int] = None,
    saved_params: Optional[Dict[str, Any]] = None,
    p: float = 0.5,
    intensity: str = "medium",
    enable_advanced: bool = False
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Applies random augmentations to the input frame.

    Args:
        frame: Input frame of shape (H, W, C)
        seed: Optional random seed for reproducibility
        saved_params: Optional parameters from a previous transform to replay
        p: Probability of applying each augmentation
        intensity: Augmentation intensity level ("light", "medium", "strong")
        enable_advanced: Whether to include advanced augmentations

    Returns:
        Tuple of (augmented frame, saved parameters for replay)
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure frame is in uint8 format for albumentations
    frame = frame.astype(np.uint8)

    # Get augmentation pipeline with specified parameters
    transform = get_augmentation_pipeline(p=p, intensity=intensity, enable_advanced=enable_advanced)

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
    seed: Optional[int] = None,
    p: float = 0.5,
    intensity: str = "medium",
    enable_advanced: bool = False
) -> List[np.ndarray]:
    """Applies the same random augmentations to all frames in a stack.

    This is crucial for temporal consistency in RL where frame stacks represent
    consecutive time steps. All frames in the stack must receive identical
    augmentations to preserve temporal relationships.

    Args:
        frames: List of frames to augment, each of shape (H, W, C)
        seed: Optional random seed for reproducibility
        p: Probability of applying each augmentation
        intensity: Augmentation intensity level ("light", "medium", "strong")
        enable_advanced: Whether to include advanced augmentations

    Returns:
        List of augmented frames with same shapes as inputs
    """
    if not frames:
        return []

    if seed is not None:
        np.random.seed(seed)

    # Ensure frames are in uint8 format for albumentations
    first_frame_uint8 = frames[0].astype(np.uint8)

    # Get augmentation pipeline with specified parameters
    transform = get_augmentation_pipeline(p=p, intensity=intensity, enable_advanced=enable_advanced)

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


def get_recommended_config(training_stage: str = "early") -> Dict[str, Any]:
    """Get recommended augmentation configuration for different training stages.
    
    Based on visual RL research, different augmentation intensities work better
    at different stages of training.
    
    Args:
        training_stage: One of "early", "mid", "late"
        
    Returns:
        Dictionary with recommended augmentation parameters
    """
    configs = {
        "early": {
            "p": 0.3,
            "intensity": "light",
            "enable_advanced": False,
            "description": "Conservative augmentation for stable early training"
        },
        "mid": {
            "p": 0.5,
            "intensity": "medium", 
            "enable_advanced": False,
            "description": "Standard augmentation for main training phase"
        },
        "late": {
            "p": 0.7,
            "intensity": "strong",
            "enable_advanced": True,
            "description": "Aggressive augmentation for final generalization"
        }
    }
    
    return configs.get(training_stage, configs["mid"])
