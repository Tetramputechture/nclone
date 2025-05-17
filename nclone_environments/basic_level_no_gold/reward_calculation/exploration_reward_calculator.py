"""Reward calculator for exploration.

Rewards exploration across the level, and penalizes excessive backtracking.

Our level is a grid of 42x23 24 pixel cells.
We use a history of the ninja position to reward exploration at multiple scales:
- Individual cells (24x24)
- 4x4 cell areas (96x96)
- 8x8 cell areas (192x192)
- 16x16 cell areas (384x384)
"""

import numpy as np

class ExplorationRewardCalculator:
    """Handles calculation of exploration-based rewards."""

    # Grid dimensions in cells (24x24 pixels each)
    GRID_WIDTH = 44
    GRID_HEIGHT = 25
    CELL_SIZE = 24.0

    # Scale rewards to keep total step reward <= 0.1
    CELL_REWARD = 0.001  # New cell
    AREA_4x4_REWARD = 0.001  # New 4x4 area
    AREA_8x8_REWARD = 0.001  # New 8x8 area
    AREA_16x16_REWARD = 0.001  # New 16x16 area

    def __init__(self):
        """Initialize exploration tracking matrices."""
        # Initialize visited tracking for each scale
        self.visited_cells = np.zeros(
            (self.GRID_HEIGHT, self.GRID_WIDTH), dtype=bool)

        # For 4x4 areas
        self.area_4x4_height = self.GRID_HEIGHT // 4 + \
            (1 if self.GRID_HEIGHT % 4 else 0)
        self.area_4x4_width = self.GRID_WIDTH // 4 + \
            (1 if self.GRID_WIDTH % 4 else 0)
        self.visited_4x4 = np.zeros(
            (self.area_4x4_height, self.area_4x4_width), dtype=bool)

        # For 8x8 areas
        self.area_8x8_height = self.GRID_HEIGHT // 8 + \
            (1 if self.GRID_HEIGHT % 8 else 0)
        self.area_8x8_width = self.GRID_WIDTH // 8 + \
            (1 if self.GRID_WIDTH % 8 else 0)
        self.visited_8x8 = np.zeros(
            (self.area_8x8_height, self.area_8x8_width), dtype=bool)

        # For 16x16 areas
        self.area_16x16_height = self.GRID_HEIGHT // 16 + \
            (1 if self.GRID_HEIGHT % 16 else 0)
        self.area_16x16_width = self.GRID_WIDTH // 16 + \
            (1 if self.GRID_WIDTH % 16 else 0)
        self.visited_16x16 = np.zeros(
            (self.area_16x16_height, self.area_16x16_width), dtype=bool)

    def _get_cell_coords(self, x: float, y: float) -> tuple[int, int]:
        """Convert pixel coordinates to cell grid coordinates."""
        cell_x = int(x / self.CELL_SIZE)
        cell_y = int(y / self.CELL_SIZE)
        # Clamp to valid grid coordinates
        cell_x = max(0, min(cell_x, self.GRID_WIDTH - 1))
        cell_y = max(0, min(cell_y, self.GRID_HEIGHT - 1))
        return cell_x, cell_y

    def calculate_exploration_reward(self, player_x: float, player_y: float) -> float:
        """Calculate exploration reward based on newly visited areas at multiple scales."""
        reward = 0.0
        cell_x, cell_y = self._get_cell_coords(player_x, player_y)

        # Check cell-level exploration
        if not self.visited_cells[cell_y, cell_x]:
            reward += self.CELL_REWARD
            self.visited_cells[cell_y, cell_x] = True

        # Check 4x4 area exploration
        area_4x4_x = cell_x // 4
        area_4x4_y = cell_y // 4
        if not self.visited_4x4[area_4x4_y, area_4x4_x]:
            reward += self.AREA_4x4_REWARD
            self.visited_4x4[area_4x4_y, area_4x4_x] = True

        # Check 8x8 area exploration
        area_8x8_x = cell_x // 8
        area_8x8_y = cell_y // 8
        if not self.visited_8x8[area_8x8_y, area_8x8_x]:
            reward += self.AREA_8x8_REWARD
            self.visited_8x8[area_8x8_y, area_8x8_x] = True

        # Check 16x16 area exploration
        area_16x16_x = cell_x // 16
        area_16x16_y = cell_y // 16
        if not self.visited_16x16[area_16x16_y, area_16x16_x]:
            reward += self.AREA_16x16_REWARD
            self.visited_16x16[area_16x16_y, area_16x16_x] = True

        return reward

    def reset(self):
        """Reset exploration tracking for new episode."""
        self.visited_cells.fill(False)
        self.visited_4x4.fill(False)
        self.visited_8x8.fill(False)
        self.visited_16x16.fill(False)
