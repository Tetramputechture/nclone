"""State discretization for value function precomputation.

This module implements conversion between continuous state space
(position, velocity, jump_duration) and discrete grid indices for
value function storage.

Grid parameters follow the proposal:
- Position: 12px resolution (half tile) -> 88x50 grid
- Velocity: 0.5 px/frame resolution -> 14x25 grid
- Jump frames: 5-frame buckets -> 10 buckets
- Total: 88 * 50 * 14 * 25 * 10 = 15.4M states

Storage: 15.4M * 4 bytes (float32) = 62MB per level
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class GridConfig:
    """Configuration for state space discretization grid.

    Attributes:
        x_min, x_max: Position bounds in pixels (level dimensions)
        y_min, y_max: Position bounds in pixels
        vx_min, vx_max: Velocity bounds in pixels/frame
        vy_min, vy_max: Velocity bounds in pixels/frame
        jump_min, jump_max: Jump duration bounds in frames
        dx, dy: Position grid resolution in pixels
        dvx, dvy: Velocity grid resolution in pixels/frame
        djump: Jump duration grid resolution in frames
    """

    # Position bounds (N++ level dimensions)
    x_min: float = 0.0
    x_max: float = 1056.0
    y_min: float = 0.0
    y_max: float = 600.0

    # Velocity bounds (slightly beyond physics limits for robustness)
    vx_min: float = -3.5
    vx_max: float = 3.5
    vy_min: float = -6.5
    vy_max: float = 6.5

    # Jump duration bounds
    jump_min: int = 0
    jump_max: int = 45  # MAX_JUMP_DURATION constant

    # Grid resolutions
    dx: float = 12.0  # Half tile resolution
    dy: float = 12.0
    dvx: float = 0.5  # Sub-pixel velocity resolution
    dvy: float = 0.5
    djump: int = 5  # 5-frame buckets for jump duration

    def __post_init__(self):
        """Compute grid dimensions from bounds and resolutions."""
        self.nx = int((self.x_max - self.x_min) / self.dx) + 1
        self.ny = int((self.y_max - self.y_min) / self.dy) + 1
        self.nvx = int((self.vx_max - self.vx_min) / self.dvx) + 1
        self.nvy = int((self.vy_max - self.vy_min) / self.dvy) + 1
        self.njump = int((self.jump_max - self.jump_min) / self.djump) + 1

        # Total states
        self.total_states = self.nx * self.ny * self.nvx * self.nvy * self.njump

        # Memory estimate (float32)
        self.memory_mb = (self.total_states * 4) / (1024 * 1024)

    def get_shape(self) -> Tuple[int, int, int, int, int]:
        """Get grid shape as (nx, ny, nvx, nvy, njump)."""
        return (self.nx, self.ny, self.nvx, self.nvy, self.njump)

    def get_info_str(self) -> str:
        """Get human-readable grid information."""
        return (
            f"Grid dimensions: {self.nx}x{self.ny}x{self.nvx}x{self.nvy}x{self.njump}\n"
            f"Total states: {self.total_states:,}\n"
            f"Memory: {self.memory_mb:.1f} MB (float32)"
        )


class StateDiscretizer:
    """Convert between continuous states and discrete grid indices.

    This class handles conversion between continuous state tuples
    (x, y, vx, vy, jump_duration) and discrete grid indices (ix, iy, ivx, ivy, ijump).

    Supports both nearest-neighbor discretization (for indexing) and
    continuous-to-fractional conversion (for interpolation).

    Attributes:
        config: GridConfig specifying grid bounds and resolutions
    """

    def __init__(self, config: Optional[GridConfig] = None):
        """Initialize state discretizer.

        Args:
            config: Grid configuration (uses defaults if None)
        """
        self.config = config or GridConfig()

    def state_to_index(
        self, state: Tuple[float, float, float, float, int]
    ) -> Tuple[int, int, int, int, int]:
        """Convert continuous state to discrete grid indices.

        Uses nearest-neighbor discretization with clamping to valid range.

        Args:
            state: (x, y, vx, vy, jump_duration) in continuous space

        Returns:
            (ix, iy, ivx, ivy, ijump) discrete indices
        """
        x, y, vx, vy, jump = state

        # Position indices
        ix = int(
            np.clip((x - self.config.x_min) / self.config.dx, 0, self.config.nx - 1)
        )
        iy = int(
            np.clip((y - self.config.y_min) / self.config.dy, 0, self.config.ny - 1)
        )

        # Velocity indices
        ivx = int(
            np.clip((vx - self.config.vx_min) / self.config.dvx, 0, self.config.nvx - 1)
        )
        ivy = int(
            np.clip((vy - self.config.vy_min) / self.config.dvy, 0, self.config.nvy - 1)
        )

        # Jump duration index
        ijump = int(np.clip(jump / self.config.djump, 0, self.config.njump - 1))

        return (ix, iy, ivx, ivy, ijump)

    def index_to_state(
        self, index: Tuple[int, int, int, int, int]
    ) -> Tuple[float, float, float, float, int]:
        """Convert discrete grid indices to continuous state.

        Returns state at grid cell centers.

        Args:
            index: (ix, iy, ivx, ivy, ijump) discrete indices

        Returns:
            (x, y, vx, vy, jump_duration) continuous state
        """
        ix, iy, ivx, ivy, ijump = index

        # Position at cell center
        x = self.config.x_min + (ix + 0.5) * self.config.dx
        y = self.config.y_min + (iy + 0.5) * self.config.dy

        # Velocity at cell center
        vx = self.config.vx_min + (ivx + 0.5) * self.config.dvx
        vy = self.config.vy_min + (ivy + 0.5) * self.config.dvy

        # Jump duration at cell center (integer)
        jump = ijump * self.config.djump

        return (x, y, vx, vy, jump)

    def state_to_fractional(
        self, state: Tuple[float, float, float, float, int]
    ) -> Tuple[float, float, float, float, float]:
        """Convert continuous state to fractional grid coordinates.

        Used for interpolation - returns floating-point indices that
        can fall between grid points.

        Args:
            state: (x, y, vx, vy, jump_duration) continuous state

        Returns:
            (fx, fy, fvx, fvy, fjump) fractional indices
        """
        x, y, vx, vy, jump = state

        # Fractional position indices
        fx = (x - self.config.x_min) / self.config.dx
        fy = (y - self.config.y_min) / self.config.dy

        # Fractional velocity indices
        fvx = (vx - self.config.vx_min) / self.config.dvx
        fvy = (vy - self.config.vy_min) / self.config.dvy

        # Fractional jump index
        fjump = jump / self.config.djump

        # Clamp to valid range
        fx = np.clip(fx, 0, self.config.nx - 1.001)
        fy = np.clip(fy, 0, self.config.ny - 1.001)
        fvx = np.clip(fvx, 0, self.config.nvx - 1.001)
        fvy = np.clip(fvy, 0, self.config.nvy - 1.001)
        fjump = np.clip(fjump, 0, self.config.njump - 1.001)

        return (fx, fy, fvx, fvy, fjump)

    def is_in_bounds(self, state: Tuple[float, float, float, float, int]) -> bool:
        """Check if state is within grid bounds.

        Args:
            state: (x, y, vx, vy, jump_duration)

        Returns:
            True if state is in bounds, False otherwise
        """
        x, y, vx, vy, jump = state

        return (
            self.config.x_min <= x <= self.config.x_max
            and self.config.y_min <= y <= self.config.y_max
            and self.config.vx_min <= vx <= self.config.vx_max
            and self.config.vy_min <= vy <= self.config.vy_max
            and self.config.jump_min <= jump <= self.config.jump_max
        )

    def get_all_states(self):
        """Generator yielding all discrete states in grid.

        Yields tuples of (ix, iy, ivx, ivy, ijump) for every grid point.
        Total: nx * ny * nvx * nvy * njump states.

        Yields:
            Tuple of (ix, iy, ivx, ivy, ijump) for each grid point
        """
        for ix in range(self.config.nx):
            for iy in range(self.config.ny):
                for ivx in range(self.config.nvx):
                    for ivy in range(self.config.nvy):
                        for ijump in range(self.config.njump):
                            yield (ix, iy, ivx, ivy, ijump)

    def index_to_linear(self, index: Tuple[int, int, int, int, int]) -> int:
        """Convert 5D index to linear (flat) index.

        Uses row-major ordering: jump varies fastest, then vy, vx, y, x.

        Args:
            index: (ix, iy, ivx, ivy, ijump)

        Returns:
            Linear index in range [0, total_states)
        """
        ix, iy, ivx, ivy, ijump = index
        return (
            ((ix * self.config.ny + iy) * self.config.nvx + ivx) * self.config.nvy + ivy
        ) * self.config.njump + ijump

    def linear_to_index(self, linear_idx: int) -> Tuple[int, int, int, int, int]:
        """Convert linear (flat) index to 5D index.

        Inverse of index_to_linear().

        Args:
            linear_idx: Linear index in range [0, total_states)

        Returns:
            (ix, iy, ivx, ivy, ijump)
        """
        ijump = linear_idx % self.config.njump
        linear_idx //= self.config.njump

        ivy = linear_idx % self.config.nvy
        linear_idx //= self.config.nvy

        ivx = linear_idx % self.config.nvx
        linear_idx //= self.config.nvx

        iy = linear_idx % self.config.ny
        ix = linear_idx // self.config.ny

        return (ix, iy, ivx, ivy, ijump)
