"""
Mine context analysis for level-wide mine statistics.

This module provides functions to compute global mine context features that help
the agent understand the overall mine layout and danger level of a level. These
features are added to the reachability summary to provide mine-aware navigation.
"""

from typing import Optional, Dict, Any

from ..constants.entity_types import EntityType
from .common import GraphData


def compute_mine_context(
    level_data: Any,
    graph_data: Optional[GraphData] = None,
) -> Dict[str, float]:
    """
    Compute global mine statistics for level understanding.

    Features:
    - total_mines_norm: Total number of mines in level [0, 1] (normalized by 256 max)
    - deadly_mine_ratio: Ratio of deadly (toggled) mines to total mines [0, 1]

    Args:
        level_data: Level data with entities list
        graph_data: Optional GraphData for additional analysis

    Returns:
        Dict with normalized mine context features
    """
    # Initialize counters
    total_mines = 0
    deadly_mines = 0

    # Extract mine data from level entities
    if hasattr(level_data, "entities") and level_data.entities:
        for entity in level_data.entities:
            entity_type = entity.get("type", 0)
            if entity_type in [EntityType.TOGGLE_MINE, EntityType.TOGGLE_MINE_TOGGLED]:
                total_mines += 1
                # Check if mine is deadly (state 0 = toggled/deadly)
                mine_state = entity.get("state", 0.0)
                if mine_state == 0.0:
                    deadly_mines += 1

    # Normalize total_mines by max (256)
    total_mines_norm = min(total_mines / 256.0, 1.0)

    # Compute deadly ratio (avoid division by zero)
    deadly_mine_ratio = deadly_mines / max(total_mines, 1) if total_mines > 0 else 0.0

    return {
        "total_mines_norm": total_mines_norm,
        "deadly_mine_ratio": deadly_mine_ratio,
    }
