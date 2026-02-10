"""5D linear interpolation for value function lookups.

This module implements efficient 5D linear interpolation for querying
the value function V*(s) at arbitrary continuous states. Since physics
produces continuous states that may fall between grid points, accurate
interpolation is critical for smooth PBRS gradients.

Interpolation methods:
- Nearest neighbor: Fast, simple, introduces discretization artifacts
- Linear (trilinear): Smooth, accurate, ~32x slower than nearest
- Hybrid: Linear for position, nearest for velocity/jump (compromise)
"""

import numpy as np
from typing import Tuple
from scipy import ndimage

from .state_discretization import StateDiscretizer


class ValueInterpolator:
    """Interpolate value function at continuous states.
    
    Provides multiple interpolation strategies with different
    speed/accuracy tradeoffs.
    
    Attributes:
        discretizer: State discretizer for grid operations
        method: Interpolation method ('nearest', 'linear', 'hybrid')
    """
    
    def __init__(self, discretizer: StateDiscretizer, method: str = 'linear'):
        """Initialize value interpolator.
        
        Args:
            discretizer: State discretizer
            method: Interpolation method in {'nearest', 'linear', 'hybrid'}
        """
        self.discretizer = discretizer
        self.method = method
        
        if method not in ('nearest', 'linear', 'hybrid'):
            raise ValueError(f"Unknown interpolation method: {method}")
    
    def interpolate(self, V: np.ndarray, 
                   state: Tuple[float, float, float, float, int]) -> float:
        """Interpolate value at continuous state.
        
        Args:
            V: Value function array shape (nx, ny, nvx, nvy, njump)
            state: Continuous state (x, y, vx, vy, jump_duration)
        
        Returns:
            Interpolated value V(state)
        """
        # Check bounds
        if not self.discretizer.is_in_bounds(state):
            return np.inf
        
        if self.method == 'nearest':
            return self._interpolate_nearest(V, state)
        elif self.method == 'linear':
            return self._interpolate_linear(V, state)
        else:  # hybrid
            return self._interpolate_hybrid(V, state)
    
    def _interpolate_nearest(self, V: np.ndarray,
                            state: Tuple[float, float, float, float, int]) -> float:
        """Nearest neighbor interpolation (fastest).
        
        Simply rounds to nearest grid point and looks up value.
        """
        # Get fractional indices
        fx, fy, fvx, fvy, fjump = self.discretizer.state_to_fractional(state)
        
        # Round to nearest
        ix = int(np.round(np.clip(fx, 0, self.discretizer.config.nx - 1)))
        iy = int(np.round(np.clip(fy, 0, self.discretizer.config.ny - 1)))
        ivx = int(np.round(np.clip(fvx, 0, self.discretizer.config.nvx - 1)))
        ivy = int(np.round(np.clip(fvy, 0, self.discretizer.config.nvy - 1)))
        ijump = int(np.round(np.clip(fjump, 0, self.discretizer.config.njump - 1)))
        
        return float(V[ix, iy, ivx, ivy, ijump])
    
    def _interpolate_linear(self, V: np.ndarray,
                           state: Tuple[float, float, float, float, int]) -> float:
        """5D linear interpolation (most accurate).
        
        Uses scipy's map_coordinates for efficient N-dimensional
        linear interpolation.
        """
        # Get fractional indices
        fx, fy, fvx, fvy, fjump = self.discretizer.state_to_fractional(state)
        
        # Use scipy's map_coordinates (order=1 for linear interpolation)
        coords = np.array([[fx], [fy], [fvx], [fvy], [fjump]])
        
        value = ndimage.map_coordinates(
            V,
            coords,
            order=1,  # Linear interpolation
            mode='nearest',  # Use nearest value for out-of-bounds
            cval=np.inf,  # Fill value for truly out-of-bounds
        )[0]
        
        return float(value)
    
    def _interpolate_hybrid(self, V: np.ndarray,
                           state: Tuple[float, float, float, float, int]) -> float:
        """Hybrid interpolation: linear for position, nearest for velocity/jump.
        
        Compromise between speed and accuracy. Position interpolation
        provides smooth spatial gradients, while velocity/jump use
        nearest neighbor for speed.
        """
        # Get fractional indices
        fx, fy, fvx, fvy, fjump = self.discretizer.state_to_fractional(state)
        
        # Round velocity and jump to nearest
        ivx = int(np.round(np.clip(fvx, 0, self.discretizer.config.nvx - 1)))
        ivy = int(np.round(np.clip(fvy, 0, self.discretizer.config.nvy - 1)))
        ijump = int(np.round(np.clip(fjump, 0, self.discretizer.config.njump - 1)))
        
        # Extract 2D slice for this velocity/jump bucket
        V_slice = V[:, :, ivx, ivy, ijump]
        
        # Linear interpolation on position only
        coords = np.array([[fx], [fy]])
        value = ndimage.map_coordinates(
            V_slice,
            coords,
            order=1,
            mode='nearest',
            cval=np.inf,
        )[0]
        
        return float(value)


def multilinear_interpolate_4d_batch(
    V_4d: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    fvx: np.ndarray,
    fvy: np.ndarray,
) -> np.ndarray:
    """Vectorized 4D multilinear interpolation.

    Interpolates V_4d at arbitrary continuous points specified by
    fractional grid coordinates. Each query point evaluates 2^4 = 16
    corners of the enclosing 4D hypercube and blends their values
    using the fractional offsets as weights.

    This function is designed for batch evaluation where millions of
    points are interpolated per call. All operations are vectorized
    NumPy with no Python loops over data points.

    Args:
        V_4d: Value array of shape (nx, ny, nvx, nvy)
        fx: Fractional x-coordinates, any broadcastable shape S
        fy: Fractional y-coordinates, shape S
        fvx: Fractional vx-coordinates, shape S
        fvy: Fractional vy-coordinates, shape S

    Returns:
        Interpolated values, shape S
    """
    nx, ny, nvx, nvy = V_4d.shape

    # Floor indices (clamped so floor+1 is always a valid index)
    ix0 = np.clip(np.floor(fx).astype(np.int32), 0, nx - 2)
    iy0 = np.clip(np.floor(fy).astype(np.int32), 0, ny - 2)
    ivx0 = np.clip(np.floor(fvx).astype(np.int32), 0, nvx - 2)
    ivy0 = np.clip(np.floor(fvy).astype(np.int32), 0, nvy - 2)

    # Fractional weights for upper (ceil) index
    wx = np.clip(fx - ix0, 0.0, 1.0).astype(np.float32)
    wy = np.clip(fy - iy0, 0.0, 1.0).astype(np.float32)
    wvx = np.clip(fvx - ivx0, 0.0, 1.0).astype(np.float32)
    wvy = np.clip(fvy - ivy0, 0.0, 1.0).astype(np.float32)

    # Sum 16 corners of the 4D hypercube with interpolation weights.
    # Fully unrolled for better performance (no Python loop overhead).
    
    # Precompute complementary weights
    wx1 = 1.0 - wx
    wy1 = 1.0 - wy
    wvx1 = 1.0 - wvx
    wvy1 = 1.0 - wvy
    
    # Index offsets for adjacent cells
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    ivx1 = ivx0 + 1
    ivy1 = ivy0 + 1
    
    # Compute all 16 corners in one vectorized expression
    result = (
        # Corner (0,0,0,0)
        wx1 * wy1 * wvx1 * wvy1 * V_4d[ix0, iy0, ivx0, ivy0] +
        # Corner (0,0,0,1)
        wx1 * wy1 * wvx1 * wvy * V_4d[ix0, iy0, ivx0, ivy1] +
        # Corner (0,0,1,0)
        wx1 * wy1 * wvx * wvy1 * V_4d[ix0, iy0, ivx1, ivy0] +
        # Corner (0,0,1,1)
        wx1 * wy1 * wvx * wvy * V_4d[ix0, iy0, ivx1, ivy1] +
        # Corner (0,1,0,0)
        wx1 * wy * wvx1 * wvy1 * V_4d[ix0, iy1, ivx0, ivy0] +
        # Corner (0,1,0,1)
        wx1 * wy * wvx1 * wvy * V_4d[ix0, iy1, ivx0, ivy1] +
        # Corner (0,1,1,0)
        wx1 * wy * wvx * wvy1 * V_4d[ix0, iy1, ivx1, ivy0] +
        # Corner (0,1,1,1)
        wx1 * wy * wvx * wvy * V_4d[ix0, iy1, ivx1, ivy1] +
        # Corner (1,0,0,0)
        wx * wy1 * wvx1 * wvy1 * V_4d[ix1, iy0, ivx0, ivy0] +
        # Corner (1,0,0,1)
        wx * wy1 * wvx1 * wvy * V_4d[ix1, iy0, ivx0, ivy1] +
        # Corner (1,0,1,0)
        wx * wy1 * wvx * wvy1 * V_4d[ix1, iy0, ivx1, ivy0] +
        # Corner (1,0,1,1)
        wx * wy1 * wvx * wvy * V_4d[ix1, iy0, ivx1, ivy1] +
        # Corner (1,1,0,0)
        wx * wy * wvx1 * wvy1 * V_4d[ix1, iy1, ivx0, ivy0] +
        # Corner (1,1,0,1)
        wx * wy * wvx1 * wvy * V_4d[ix1, iy1, ivx0, ivy1] +
        # Corner (1,1,1,0)
        wx * wy * wvx * wvy1 * V_4d[ix1, iy1, ivx1, ivy0] +
        # Corner (1,1,1,1)
        wx * wy * wvx * wvy * V_4d[ix1, iy1, ivx1, ivy1]
    )

    return result


def interpolate_value_fast(V: np.ndarray,
                          state: Tuple[float, float, float, float, int],
                          discretizer: StateDiscretizer) -> float:
    """Fast nearest-neighbor interpolation (standalone function).
    
    This function provides a lightweight alternative to ValueInterpolator
    for performance-critical code where nearest-neighbor is acceptable.
    
    Args:
        V: Value function array
        state: Continuous state
        discretizer: State discretizer
    
    Returns:
        Value at nearest grid point
    """
    # Check bounds
    if not discretizer.is_in_bounds(state):
        return np.inf
    
    # Get nearest index
    idx = discretizer.state_to_index(state)
    return float(V[idx])
