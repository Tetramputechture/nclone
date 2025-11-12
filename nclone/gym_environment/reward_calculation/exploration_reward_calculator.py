"""Completion-focused exploration reward calculator.

Rewards exploration across the level to help discover switches and exits.
Uses multi-scale spatial exploration following count-based methods from
Bellemare et al. (2016): "Unifying Count-Based Exploration and Intrinsic Motivation".

Resets when switch is activated to encourage re-exploration for exit discovery.

The N++ level is a grid of 42x23 cells (24 pixels each = 1056x600 pixels total).
We track exploration at multiple spatial scales:
- Individual cells (24x24 pixels) - fine-grained exploration
- 4x4 cell areas (96x96 pixels) - room-sized regions
- 8x8 cell areas (192x192 pixels) - section-sized regions
- 16x16 cell areas (384x384 pixels) - major level regions

This multi-scale approach encourages both thorough local exploration and
broad coverage of the entire level.
"""

import numpy as np
from .reward_constants import (
    EXPLORATION_GRID_WIDTH,
    EXPLORATION_GRID_HEIGHT,
    EXPLORATION_CELL_SIZE,
    EXPLORATION_CELL_REWARD,
    EXPLORATION_AREA_4X4_REWARD,
    EXPLORATION_AREA_8X8_REWARD,
    EXPLORATION_AREA_16X16_REWARD,
    EXPLORATION_DECAY_ENABLED,
    EXPLORATION_DECAY_START_STEP,
    EXPLORATION_DECAY_END_STEP,
    EXPLORATION_MIN_SCALE,
)


class ExplorationRewardCalculator:
    """Handles calculation of completion-focused exploration rewards.

    Implements multi-scale count-based exploration rewards that encourage
    the agent to discover all areas of the level, including switches and exits.

    The exploration state resets when the switch is activated, encouraging
    the agent to re-explore to find the newly-accessible exit door.

    All constants defined in reward_constants.py.
    """

    def __init__(self):
        """Initialize exploration tracking matrices."""
        # Initialize visited tracking for each scale
        self.visited_cells = np.zeros(
            (EXPLORATION_GRID_HEIGHT, EXPLORATION_GRID_WIDTH), dtype=bool
        )

        # For 4x4 areas
        self.area_4x4_height = EXPLORATION_GRID_HEIGHT // 4 + (
            1 if EXPLORATION_GRID_HEIGHT % 4 else 0
        )
        self.area_4x4_width = EXPLORATION_GRID_WIDTH // 4 + (
            1 if EXPLORATION_GRID_WIDTH % 4 else 0
        )
        self.visited_4x4 = np.zeros(
            (self.area_4x4_height, self.area_4x4_width), dtype=bool
        )

        # For 8x8 areas
        self.area_8x8_height = EXPLORATION_GRID_HEIGHT // 8 + (
            1 if EXPLORATION_GRID_HEIGHT % 8 else 0
        )
        self.area_8x8_width = EXPLORATION_GRID_WIDTH // 8 + (
            1 if EXPLORATION_GRID_WIDTH % 8 else 0
        )
        self.visited_8x8 = np.zeros(
            (self.area_8x8_height, self.area_8x8_width), dtype=bool
        )

        # For 16x16 areas
        self.area_16x16_height = EXPLORATION_GRID_HEIGHT // 16 + (
            1 if EXPLORATION_GRID_HEIGHT % 16 else 0
        )
        self.area_16x16_width = EXPLORATION_GRID_WIDTH // 16 + (
            1 if EXPLORATION_GRID_WIDTH % 16 else 0
        )
        self.visited_16x16 = np.zeros(
            (self.area_16x16_height, self.area_16x16_width), dtype=bool
        )

        # Exploration decay tracking
        self.step_count = 0

    def _get_cell_coords(self, x: float, y: float) -> tuple[int, int]:
        """Convert pixel coordinates to cell grid coordinates."""
        cell_x = int(x / EXPLORATION_CELL_SIZE)
        cell_y = int(y / EXPLORATION_CELL_SIZE)
        # Clamp to valid grid coordinates
        cell_x = max(0, min(cell_x, EXPLORATION_GRID_WIDTH - 1))
        cell_y = max(0, min(cell_y, EXPLORATION_GRID_HEIGHT - 1))
        return cell_x, cell_y

    def calculate_exploration_reward(self, player_x: float, player_y: float) -> float:
        """Calculate exploration reward based on newly visited areas at multiple scales."""
        self.step_count += 1
        
        reward = 0.0
        cell_x, cell_y = self._get_cell_coords(player_x, player_y)

        # Check cell-level exploration
        if not self.visited_cells[cell_y, cell_x]:
            reward += EXPLORATION_CELL_REWARD
            self.visited_cells[cell_y, cell_x] = True

        # Check 4x4 area exploration
        area_4x4_x = cell_x // 4
        area_4x4_y = cell_y // 4
        if not self.visited_4x4[area_4x4_y, area_4x4_x]:
            reward += EXPLORATION_AREA_4X4_REWARD
            self.visited_4x4[area_4x4_y, area_4x4_x] = True

        # Check 8x8 area exploration
        area_8x8_x = cell_x // 8
        area_8x8_y = cell_y // 8
        if not self.visited_8x8[area_8x8_y, area_8x8_x]:
            reward += EXPLORATION_AREA_8X8_REWARD
            self.visited_8x8[area_8x8_y, area_8x8_x] = True

        # Check 16x16 area exploration
        area_16x16_x = cell_x // 16
        area_16x16_y = cell_y // 16
        if not self.visited_16x16[area_16x16_y, area_16x16_x]:
            reward += EXPLORATION_AREA_16X16_REWARD
            self.visited_16x16[area_16x16_y, area_16x16_x] = True

        # Apply decay if enabled
        if EXPLORATION_DECAY_ENABLED:
            decay_scale = self._calculate_decay_scale()
            reward *= decay_scale

        return reward

    def _calculate_decay_scale(self) -> float:
        """Calculate exploration reward decay scale based on episode progress."""
        if self.step_count < EXPLORATION_DECAY_START_STEP:
            return 1.0  # No decay early in episode
        elif self.step_count >= EXPLORATION_DECAY_END_STEP:
            return EXPLORATION_MIN_SCALE  # Full decay
        else:
            # Linear interpolation
            progress = (self.step_count - EXPLORATION_DECAY_START_STEP) / (
                EXPLORATION_DECAY_END_STEP - EXPLORATION_DECAY_START_STEP
            )
            return 1.0 - progress * (1.0 - EXPLORATION_MIN_SCALE)

    def reset(self):
        """Reset exploration tracking for new episode."""
        self.visited_cells.fill(False)
        self.visited_4x4.fill(False)
        self.visited_8x8.fill(False)
        self.visited_16x16.fill(False)
        self.step_count = 0
