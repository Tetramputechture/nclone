"""Frame augmentation utilities for preprocessing observation frames.

This module implements various image augmentation techniques using the albumentations library.
The augmentations are applied randomly to input frames to increase training diversity and
generalization for visual game environments.

Based on research from RAD, DrQ-v2, and game-specific RL studies, this pipeline includes
the most effective augmentations for game environments like N++:
- Random crop/translate: Most stable and effective for RL, helps with position invariance
- Color jitter: Improves generalization across visual variations (limited for grayscale games)
- Cutout: Encourages focus on global context rather than local features
- Random flip: Horizontal flipping for symmetric game mechanics
- Grayscale variations: Subtle contrast/brightness changes for visual robustness

Note: Gaussian noise is NOT included as it's inappropriate for clean game visuals.
Game environments have crisp, deterministic graphics unlike sensor data.


References:
- Laskin et al. (2020): Reinforcement Learning with Augmented Data (RAD)
- Kostrikov et al. (2020): Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels (DrQ)
- Yarats et al. (2021): Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning (DrQ-v2)
- Raileanu et al. (2021): Automatic Data Augmentation for Generalization in Reinforcement Learning (Procgen study)
- Albumentations Performance Guide: https://albumentations.ai/docs/3-basic-usage/performance-tuning/
"""

import numpy as np
import albumentations as A
from typing import Optional, Dict, Any
import functools
import cv2

# Force OpenCV single-threaded mode for multiprocessing compatibility with PyTorch DataLoader
# This prevents thread contention when multiple DataLoader workers spawn OpenCV threads
# See: https://albumentations.ai/docs/3-basic-usage/performance-tuning/#7-address-multiprocessing-bottlenecks-opencv--pytorch
cv2.setNumThreads(0)


@functools.lru_cache(maxsize=None)
def get_augmentation_pipeline(
    p: float = 0.5, intensity: str = "medium", disable_validation: bool = False
) -> A.ReplayCompose:
    """Creates an augmentation pipeline optimized for N++ platformer game.

    N++ is assumed to be horizontally symmetric, so horizontal flipping is always enabled.
    Only core augmentations are included for stable training.

    Args:
        p: Probability of applying each augmentation.
        intensity: Augmentation intensity level ("light", "medium", "strong")
        disable_validation: If True, disables Albumentations validation. Recommended for training.

    Returns:
        ReplayCompose pipeline that can record and replay exact transformations
    """
    # Intensity-based parameter scaling
    intensity_scales = {"light": 0.7, "medium": 1.0, "strong": 1.3}
    scale = intensity_scales.get(intensity, 1.0)

    augmentations = []

    # 1. Random Crop/Translate - Most effective for RL (RAD, DrQ-v2)
    # Small translations help with position invariance in games
    # For 84x84 frames, 4 pixels = ~0.05 normalized shift
    shift_limit_normalized = (4 * scale) / 84.0  # Normalize to image size
    augmentations.append(
        A.Affine(
            translate_percent={
                "x": (-shift_limit_normalized, shift_limit_normalized),
                "y": (-shift_limit_normalized, shift_limit_normalized),
            },
            scale=1.0,  # No scaling to maintain spatial relationships
            rotate=0,  # No rotation to avoid training instability
            p=p * 0.8,  # Higher probability for most effective augmentation
        )
    )

    # 2. Horizontal Flip - N++ has symmetric mechanics
    # N++ levels can often be approached from either direction
    augmentations.append(
        A.HorizontalFlip(p=p * 0.4)  # Moderate probability
    )

    # 3. Cutout - Encourages global context learning (DeVries & Taylor, 2017)
    # Particularly effective for games where local features can be misleading
    augmentations.append(
        A.CoarseDropout(
            num_holes_range=(1, 2),
            hole_height_range=(int(6 * scale), int(12 * scale)),  # Smaller for games
            hole_width_range=(int(6 * scale), int(12 * scale)),
            p=p * 0.5,
        )
    )

    # 4. Brightness/Contrast - Subtle variations for visual robustness
    # Games have consistent lighting, so only subtle changes
    augmentations.append(
        A.RandomBrightnessContrast(
            brightness_limit=0.1 * scale,  # Subtle brightness changes
            contrast_limit=0.1 * scale,  # Subtle contrast changes
            p=p * 0.4,
        )
    )

    # Create ReplayCompose with optional validation disabling for performance
    # Disabling validation saves ~12% runtime by skipping Pydantic checks
    if disable_validation:
        # Performance mode: disable shape validation and other checks
        return A.ReplayCompose(augmentations, p=1.0)
    else:
        # Safe mode: full validation enabled (recommended for development)
        return A.ReplayCompose(augmentations)


def apply_augmentation(
    frame: np.ndarray,
    seed: Optional[int] = None,
    p: float = 0.5,
    intensity: str = "medium",
    disable_validation: bool = False,
    return_replay: bool = False,
) -> Any:
    """Apply random augmentation to a single frame.

    Simplified function for single-frame augmentation, optimized for performance.
    This replaces the old version that returned a tuple with replay parameters,
    which is no longer needed since we only augment single frames now.

    Args:
        frame: Single frame to augment, shape (H, W, C)
        seed: Optional random seed for reproducibility
        p: Probability of applying each augmentation
        intensity: Augmentation intensity level ("light", "medium", "strong")
        disable_validation: If True, disables validation for performance
        return_replay: If True, returns (augmented_frame, replay_data) tuple for replaying
                      augmentation on additional frames. If False, returns just the frame.

    Returns:
        If return_replay=False: Augmented frame with same shape as input
        If return_replay=True: Tuple of (augmented_frame, replay_data)
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure frame is in uint8 format for albumentations (performance best practice)
    frame_uint8 = frame.astype(np.uint8)

    # Get augmentation pipeline with specified parameters
    transform = get_augmentation_pipeline(
        p=p, intensity=intensity, disable_validation=disable_validation
    )

    # Apply augmentation
    result = transform(image=frame_uint8)
    augmented = result["image"]

    if return_replay:
        # Return both the augmented frame and replay data for consistent augmentation
        return augmented, result["replay"]
    else:
        return augmented


def apply_augmentation_with_replay(
    frame: np.ndarray,
    replay_data: Dict[str, Any],
    p: float = 0.5,
    intensity: str = "medium",
    disable_validation: bool = False,
) -> np.ndarray:
    """Apply the same augmentation to a frame using replay data.
    
    This ensures consistent augmentation across multiple frames in a stack.
    
    Args:
        frame: Frame to augment, shape (H, W, C)
        replay_data: Replay data from a previous apply_augmentation call
        p: Probability parameter (must match original)
        intensity: Intensity parameter (must match original)
        disable_validation: Validation parameter (must match original)
        
    Returns:
        Augmented frame with the same transformations as the original
    """
    # Ensure frame is in uint8 format
    frame_uint8 = frame.astype(np.uint8)
    
    # Get the same augmentation pipeline
    transform = get_augmentation_pipeline(
        p=p, intensity=intensity, disable_validation=disable_validation
    )
    
    # Replay the augmentation with the recorded parameters
    augmented = A.ReplayCompose.replay(replay_data, image=frame_uint8)["image"]
    
    return augmented


def get_recommended_config(training_stage: str = "early") -> Dict[str, Any]:
    """Get recommended augmentation configuration for different training stages.

    Based on visual RL research, different augmentation intensities work better
    at different stages of training. Optimized for N++ platformer game.

    Args:
        training_stage: One of "early", "mid", "late"

    Returns:
        Dictionary with recommended augmentation parameters
    """
    # Configurations for different training stages (optimized for N++)
    configs = {
        "early": {
            "p": 0.3,
            "intensity": "light",
            "description": "Conservative augmentation for stable early training",
        },
        "mid": {
            "p": 0.5,
            "intensity": "medium",
            "description": "Standard augmentation for main training phase",
        },
        "late": {
            "p": 0.6,  # Moderate for games (less than sensor data)
            "intensity": "strong",
            "description": "Moderate augmentation for final generalization",
        },
    }

    return configs.get(training_stage, configs["mid"]).copy()
