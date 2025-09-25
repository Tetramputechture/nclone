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
"""

import numpy as np
import albumentations as A
from typing import Optional, List, Dict, Any, Tuple
import functools


@functools.lru_cache(maxsize=None)
def get_augmentation_pipeline(
    p: float = 0.5, 
    intensity: str = "medium",
    enable_advanced: bool = False,
    game_symmetric: bool = True
) -> A.ReplayCompose:
    """Creates an augmentation pipeline optimized for visual game environments.

    Args:
        p: Probability of applying each augmentation.
        intensity: Augmentation intensity level ("light", "medium", "strong")
        enable_advanced: Whether to include advanced augmentations
        game_symmetric: Whether the game has horizontal symmetry (enables flipping)

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
    # Small translations help with position invariance in games
    # For 84x84 frames, 4 pixels = ~0.05 normalized shift
    shift_limit_normalized = (4 * scale) / 84.0  # Normalize to image size
    augmentations.append(
        A.Affine(
            translate_percent={"x": (-shift_limit_normalized, shift_limit_normalized), 
                             "y": (-shift_limit_normalized, shift_limit_normalized)},
            scale=1.0,  # No scaling to maintain spatial relationships
            rotate=0,   # No rotation to avoid training instability
            p=p * 0.8   # Higher probability for most effective augmentation
        )
    )
    
    # 2. Horizontal Flip - For games with symmetric mechanics (like N++)
    # N++ levels can often be approached from either direction
    if game_symmetric:
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
            p=p * 0.5
        )
    )
    
    # 4. Brightness/Contrast - Subtle variations for visual robustness
    # Games have consistent lighting, so only subtle changes
    augmentations.append(
        A.RandomBrightnessContrast(
            brightness_limit=0.1 * scale,  # Subtle brightness changes
            contrast_limit=0.1 * scale,    # Subtle contrast changes
            p=p * 0.4
        )
    )
    
    # 5. Advanced augmentations (optional, use with caution for games)
    if enable_advanced:
        # Color jitter for RGB games (limited effect on grayscale)
        augmentations.append(
            A.ColorJitter(
                brightness=0.05 * scale,  # Very subtle for games
                contrast=0.05 * scale,
                saturation=0.02 * scale,  # Minimal saturation change
                hue=0.01 * scale,         # Minimal hue change
                p=p * 0.2
            )
        )
        
        # Random grayscale conversion (for RGB inputs)
        augmentations.append(
            A.ToGray(p=p * 0.1)  # Very low probability
        )
    
    return A.ReplayCompose(augmentations)


def apply_augmentation(
    frame: np.ndarray,
    seed: Optional[int] = None,
    saved_params: Optional[Dict[str, Any]] = None,
    p: float = 0.5,
    intensity: str = "medium",
    enable_advanced: bool = False,
    game_symmetric: bool = True
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Applies random augmentations to the input frame for game environments.

    Args:
        frame: Input frame of shape (H, W, C)
        seed: Optional random seed for reproducibility
        saved_params: Optional parameters from a previous transform to replay
        p: Probability of applying each augmentation
        intensity: Augmentation intensity level ("light", "medium", "strong")
        enable_advanced: Whether to include advanced augmentations
        game_symmetric: Whether the game has horizontal symmetry (enables flipping)

    Returns:
        Tuple of (augmented frame, saved parameters for replay)
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure frame is in uint8 format for albumentations
    frame = frame.astype(np.uint8)

    # Get augmentation pipeline with specified parameters
    transform = get_augmentation_pipeline(
        p=p, 
        intensity=intensity, 
        enable_advanced=enable_advanced,
        game_symmetric=game_symmetric
    )

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
    enable_advanced: bool = False,
    game_symmetric: bool = True
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
        game_symmetric: Whether the game has horizontal symmetry (enables flipping)

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
    transform = get_augmentation_pipeline(
        p=p, 
        intensity=intensity, 
        enable_advanced=enable_advanced,
        game_symmetric=game_symmetric
    )

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


def get_recommended_config(training_stage: str = "early", game_type: str = "platformer") -> Dict[str, Any]:
    """Get recommended augmentation configuration for different training stages and game types.
    
    Based on visual RL research and game-specific considerations, different augmentation 
    intensities work better at different stages of training.
    
    Args:
        training_stage: One of "early", "mid", "late"
        game_type: Type of game ("platformer", "puzzle", "action")
        
    Returns:
        Dictionary with recommended augmentation parameters
    """
    # Base configurations for different stages
    base_configs = {
        "early": {
            "p": 0.3,
            "intensity": "light",
            "enable_advanced": False,
            "game_symmetric": True,
            "description": "Conservative augmentation for stable early training"
        },
        "mid": {
            "p": 0.5,
            "intensity": "medium", 
            "enable_advanced": False,
            "game_symmetric": True,
            "description": "Standard augmentation for main training phase"
        },
        "late": {
            "p": 0.6,  # Slightly lower than sensor data
            "intensity": "strong",
            "enable_advanced": True,
            "game_symmetric": True,
            "description": "Moderate augmentation for final generalization (games need less than sensor data)"
        }
    }
    
    config = base_configs.get(training_stage, base_configs["mid"]).copy()
    
    # Game-specific adjustments
    if game_type == "platformer":
        # Platformers like N++ benefit from horizontal flipping
        config["game_symmetric"] = True
    elif game_type == "puzzle":
        # Puzzle games may not benefit from flipping
        config["game_symmetric"] = False
        config["p"] *= 0.8  # Reduce overall augmentation
    elif game_type == "action":
        # Action games need careful augmentation to preserve reaction timing
        config["p"] *= 0.9
        config["game_symmetric"] = False
    
    return config
