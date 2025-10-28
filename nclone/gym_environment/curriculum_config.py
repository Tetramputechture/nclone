"""
Curriculum and category configuration for map generation.

This module provides a centralized configuration system for curriculum learning
and map generation categories. Category definitions are imported from
generator_configs.py to maintain a single source of truth.
"""

from dataclasses import dataclass
from typing import List, Optional
from ..map_generation.generator_configs import CATEGORIES as GENERATOR_CATEGORIES


@dataclass
class CategoryConfig:
    """Configuration for a single difficulty category."""

    name: str
    display_name: str
    default_weight: float
    description: str


# Build curriculum categories from generator categories
# This ensures category names stay synchronized with generator_configs.py
def _build_curriculum_categories():
    """Build curriculum categories from generator categories."""
    # Default weights for curriculum learning (can be overridden)
    default_weights = {
        "simplest": 10.0,
        "simpler": 10.0,
        "simple": 30.0,
        "medium": 30.0,
        "complex": 20.0,
        "mine_heavy": 10.0,
        "exploration": 10.0,
    }

    categories = []
    for name, gen_config in GENERATOR_CATEGORIES.items():
        categories.append(
            CategoryConfig(
                name=name,
                display_name=name.replace("_", " ").title(),
                default_weight=default_weights.get(name, 10.0),
                description=gen_config.description,
            )
        )
    return categories


CATEGORIES = _build_curriculum_categories()

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
