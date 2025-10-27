"""
Curriculum and category configuration for map generation.

This module provides a centralized configuration system for curriculum learning
and map generation categories, making it easy to extend and modify difficulty levels.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CategoryConfig:
    """Configuration for a single difficulty category."""
    
    name: str  # Category identifier (e.g., "simple", "medium")
    display_name: str  # Human-readable name
    default_weight: float  # Default sampling weight for training
    description: str  # Brief description of difficulty level


# Centralized category definitions - single source of truth
CATEGORIES = [
    CategoryConfig(
        name="simplest",
        display_name="Simplest",
        default_weight=10.0,
        description="Minimal levels: direct path from ninja to switch to exit"
    ),
    CategoryConfig(
        name="simpler",
        display_name="Simpler",
        default_weight=10.0,
        description="Simple levels with slight variations in layout"
    ),
    CategoryConfig(
        name="simple",
        display_name="Simple",
        default_weight=30.0,
        description="Basic platforming with jumps and small mazes"
    ),
    CategoryConfig(
        name="medium",
        display_name="Medium",
        default_weight=30.0,
        description="Medium-sized mazes and multi-chamber levels"
    ),
    CategoryConfig(
        name="complex",
        display_name="Complex",
        default_weight=20.0,
        description="Large mazes, islands, and complex navigation"
    ),
    CategoryConfig(
        name="mine_heavy",
        display_name="Mine-Heavy",
        default_weight=10.0,
        description="Significant mine obstacles requiring careful movement"
    ),
    CategoryConfig(
        name="exploration",
        display_name="Exploration",
        default_weight=10.0,
        description="Large areas requiring extensive exploration"
    ),
]

# Quick lookup by name
CATEGORY_MAP = {cat.name: cat for cat in CATEGORIES}

# Ordered list of category names (used for iteration)
CATEGORY_NAMES = [cat.name for cat in CATEGORIES]


def get_category(name: str) -> Optional[CategoryConfig]:
    """Get category config by name.
    
    Args:
        name: Category name
        
    Returns:
        CategoryConfig or None if not found
    """
    return CATEGORY_MAP.get(name)


def get_default_weights() -> List[float]:
    """Get default weights for all categories in order.
    
    Returns:
        List of default weights matching CATEGORIES order
    """
    return [cat.default_weight for cat in CATEGORIES]


def get_category_weights_dict() -> dict:
    """Get default weights as a dictionary.
    
    Returns:
        Dictionary mapping category names to default weights
    """
    return {cat.name: cat.default_weight for cat in CATEGORIES}


def validate_category(name: str) -> bool:
    """Check if a category name is valid.
    
    Args:
        name: Category name to validate
        
    Returns:
        True if valid, False otherwise
    """
    return name in CATEGORY_MAP


def validate_weights(weights: dict) -> tuple[bool, Optional[str]]:
    """Validate a weights dictionary.
    
    Args:
        weights: Dictionary mapping category names to weights
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check all categories present
    for cat in CATEGORIES:
        if cat.name not in weights:
            return False, f"Missing weight for category '{cat.name}'"
    
    # Check all weights non-negative
    for name, weight in weights.items():
        if weight < 0:
            return False, f"Weight for '{name}' must be non-negative, got {weight}"
        if name not in CATEGORY_MAP:
            return False, f"Unknown category '{name}'"
    
    return True, None
