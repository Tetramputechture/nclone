"""Value iteration algorithm for computing optimal value functions.

This module implements backward dynamic programming (value iteration)
to compute V*(s) - the optimal cost-to-go from every state to the goal.

The algorithm:
1. Initialize V[s] = 0 for goal states, inf elsewhere
2. Repeat until convergence:
   For each state s:
     V[s] = min over actions a of [cost(s,a) + V[s']]
   where s' = physics_step(s, a)

Convergence typically occurs in 100-500 iterations with tolerance < 0.01.
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any
import logging

try:
    from tqdm import tqdm
except ImportError:
    # Simple fallback if tqdm not available
    def tqdm(iterable, disable=False, desc=""):
        """Simple progress indicator fallback."""
        if not disable:
            print(f"{desc}...")
        return iterable

from .state_discretization import StateDiscretizer, GridConfig
from .physics_step_wrapper import PhysicsStepWrapper

logger = logging.getLogger(__name__)


# Action space: 6 actions (h_input, jump_input)
ACTIONS = [
    (-1, 0),  # Left
    (-1, 1),  # Left + Jump
    (0, 0),   # No input
    (0, 1),   # Jump
    (1, 0),   # Right
    (1, 1),   # Right + Jump
]


class ValueIterator:
    """Compute optimal value function via backward dynamic programming.
    
    This class implements value iteration to compute V*(s) - the minimum
    cost (time) to reach the goal from any state. Uses the existing physics
    simulation through PhysicsStepWrapper for exact dynamics.
    
    Attributes:
        physics: PhysicsStepWrapper for querying state transitions
        discretizer: StateDiscretizer for grid operations
        goal_pos: Goal position (x, y) in pixels
        goal_radius: Goal radius for success detection
    """
    
    def __init__(self, 
                 physics: PhysicsStepWrapper,
                 discretizer: StateDiscretizer,
                 goal_pos: Tuple[float, float],
                 goal_radius: float = 20.0):
        """Initialize value iterator.
        
        Args:
            physics: Physics step wrapper for state transitions
            discretizer: State discretizer for grid operations
            goal_pos: Goal position (x, y) in pixels
            goal_radius: Radius for goal detection (default 20px)
        """
        self.physics = physics
        self.discretizer = discretizer
        self.goal_pos = goal_pos
        self.goal_radius = goal_radius
        
        # Get grid shape
        self.shape = discretizer.config.get_shape()
        
        logger.info(f"ValueIterator initialized:")
        logger.info(f"  Grid: {self.shape}")
        logger.info(f"  Total states: {np.prod(self.shape):,}")
        logger.info(f"  Goal: {goal_pos}, radius: {goal_radius}")
    
    def compute_value_function(self, 
                               max_iterations: int = 500,
                               tolerance: float = 0.01,
                               cost_fn: Optional[Callable] = None,
                               verbose: bool = True,
                               position_validity_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute optimal value function V*(s) via value iteration.
        
        Args:
            max_iterations: Maximum number of iterations (default 500)
            tolerance: Convergence tolerance (default 0.01)
            cost_fn: Optional step cost function (s, a) -> cost
                    Default: constant cost of frames_per_step (minimum time in frames)
            verbose: Show progress bar and logging
            position_validity_mask: Optional boolean mask (nx, ny) indicating valid positions.
                                   If provided, invalid positions are marked as V=inf.
        
        Returns:
            V: Value function array of shape (nx, ny, nvx, nvy, njump)
               V[s] = minimum cost to reach goal from state s
        """
        if cost_fn is None:
            cost_fn = lambda s, a: float(self.physics.frames_per_step)  # Minimum time in frames
        
        # Initialize value function
        V = np.full(self.shape, np.inf, dtype=np.float32)
        
        # Set goal states to zero
        logger.info("Initializing goal states...")
        goal_count = self._initialize_goal_states(V)
        logger.info(f"  {goal_count} goal states initialized")
        
        # Mark invalid positions (inside geometry, overlapping mines)
        if position_validity_mask is not None:
            logger.info("Marking invalid positions...")
            invalid_count = self._mark_invalid_positions(V, position_validity_mask)
            logger.info(f"  {invalid_count} invalid positions marked")
        
        
        # Value iteration
        logger.info(f"Starting value iteration (max_iter={max_iterations}, tol={tolerance})...")
        
        for iteration in tqdm(range(max_iterations), disable=not verbose, desc="Value iteration"):
            V_old = V.copy()
            max_change = 0.0
            updates_this_iter = 0  # Track actual updates
            
            # BUG FIX: Must iterate over ALL states, not just reachable ones!
            # Old buggy code only iterated over V < inf, but initially all non-goal states are inf,
            # so no propagation could happen.
            num_goal = np.sum(V_old == 0.0)
            
            if iteration == 0:
                logger.info(f"  Iteration {iteration}: {num_goal:,} goal states")
            
            # Update ALL states (not just reachable), but skip invalid positions
            for idx in np.ndindex(self.shape):
                # Skip goal states (already 0)
                if V_old[idx] == 0.0:
                    continue
                
                # Skip invalid positions (if mask provided)
                if position_validity_mask is not None:
                    ix, iy = idx[0], idx[1]
                    if not position_validity_mask[ix, iy]:
                        continue
                
                # Get continuous state
                state = self.discretizer.index_to_state(idx)
                
                # Bellman update: V(s) = min_a [c(s,a) + V(s')]
                best_cost = np.inf
                for action in ACTIONS:
                    try:
                        next_state, is_death = self.physics.step(state, action)
                        
                        if is_death:
                            # Dead state has infinite cost
                            continue
                        
                        # Get value at next state (with interpolation for off-grid)
                        V_next = self._interpolate_value(V_old, next_state)
                        
                        # Total cost
                        step_cost = cost_fn(state, action)
                        total_cost = step_cost + V_next
                        
                        best_cost = min(best_cost, total_cost)
                    except Exception as e:
                        # Physics error - treat as infinite cost
                        logger.debug(f"Physics error at {idx}: {e}")
                        continue
                
                # Update value (only if improved)
                if best_cost < V_old[idx]:
                    V[idx] = best_cost
                    max_change = max(max_change, abs(V[idx] - V_old[idx]))
                    updates_this_iter += 1
            
            # Check convergence
            if max_change < tolerance:
                logger.info(f"Converged after {iteration + 1} iterations (max_change={max_change:.6f}, updates={updates_this_iter})")
                break
            
            if verbose and (iteration + 1) % 10 == 0:
                reachable = np.sum(V < np.inf)
                logger.info(f"  Iter {iteration + 1}: max_change={max_change:.6f}, reachable={reachable:,}, updates={updates_this_iter}")
        
        if max_change >= tolerance:
            logger.warning(f"Did not converge after {max_iterations} iterations (max_change={max_change:.6f})")
        
        # Summary statistics
        reachable = np.sum(V < np.inf)
        goal_states = np.sum(V == 0)
        logger.info(f"Value function computed:")
        logger.info(f"  Reachable states: {reachable:,} / {np.prod(self.shape):,}")
        logger.info(f"  Goal states: {goal_states:,}")
        logger.info(f"  Value range: [{np.min(V[V < np.inf]):.1f}, {np.max(V[V < np.inf]):.1f}]")
        
        return V
    
    def _initialize_goal_states(self, V: np.ndarray) -> int:
        """Mark goal states with V=0.
        
        A state is a goal if position is within goal_radius of goal_pos.
        Goal detection is velocity-independent (any velocity at goal counts).
        
        Args:
            V: Value function array (modified in-place)
        
        Returns:
            Number of goal states initialized
        """
        count = 0
        gx, gy = self.goal_pos
        goal_cells = []  # Track which position cells are goals
        
        # Iterate over position grid
        for ix in range(self.discretizer.config.nx):
            for iy in range(self.discretizer.config.ny):
                # Get continuous position
                state = self.discretizer.index_to_state((ix, iy, 0, 0, 0))
                x, y = state[0], state[1]
                
                # Check if within goal radius
                dist = np.sqrt((x - gx)**2 + (y - gy)**2)
                if dist <= self.goal_radius:
                    goal_cells.append({"ix": ix, "iy": iy, "x": x, "y": y, "dist": dist})
                    # Mark all velocity/jump states at this position as goal
                    for ivx in range(self.discretizer.config.nvx):
                        for ivy in range(self.discretizer.config.nvy):
                            for ijump in range(self.discretizer.config.njump):
                                V[ix, iy, ivx, ivy, ijump] = 0.0
                                count += 1
        
        return count
    
    def _mark_invalid_positions(self, V: np.ndarray, 
                                position_validity_mask: np.ndarray) -> int:
        """Mark positions where ninja cannot physically exist with V=inf.
        
        Invalid positions include:
        - Positions where ninja center falls inside solid tile geometry
        - Positions where ninja overlaps toggled toggle mines
        - Positions outside level bounds
        
        Args:
            V: Value function array (modified in-place)
            position_validity_mask: Boolean mask (nx, ny), True = valid position
        
        Returns:
            Number of invalid positions marked (across all velocity/jump states)
        """
        count = 0
        
        # Mark all velocity/jump states at invalid positions as inf
        for ix in range(self.discretizer.config.nx):
            for iy in range(self.discretizer.config.ny):
                if not position_validity_mask[ix, iy]:
                    # This position is invalid - mark all velocity/jump combinations
                    for ivx in range(self.discretizer.config.nvx):
                        for ivy in range(self.discretizer.config.nvy):
                            for ijump in range(self.discretizer.config.njump):
                                V[ix, iy, ivx, ivy, ijump] = np.inf
                                count += 1
        
        return count
    
    def _interpolate_value(self, V: np.ndarray, state: Tuple[float, float, float, float, int]) -> float:
        """Interpolate value function at continuous state.
        
        Uses 5D linear interpolation for smooth value estimates at
        off-grid states. Critical for accurate Bellman updates since
        physics produces continuous states.
        
        Args:
            V: Value function array
            state: Continuous state (x, y, vx, vy, jump_duration)
        
        Returns:
            Interpolated value V(state)
        """
        # Check if in bounds
        if not self.discretizer.is_in_bounds(state):
            return np.inf
        
        # Get fractional indices
        fx, fy, fvx, fvy, fjump = self.discretizer.state_to_fractional(state)
        
        # Use scipy's map_coordinates for 5D interpolation
        # For now, use nearest-neighbor (simple and fast)
        ix = int(np.round(fx))
        iy = int(np.round(fy))
        ivx = int(np.round(fvx))
        ivy = int(np.round(fvy))
        ijump = int(np.round(fjump))
        
        # Clamp to valid indices
        ix = np.clip(ix, 0, self.discretizer.config.nx - 1)
        iy = np.clip(iy, 0, self.discretizer.config.ny - 1)
        ivx = np.clip(ivx, 0, self.discretizer.config.nvx - 1)
        ivy = np.clip(ivy, 0, self.discretizer.config.nvy - 1)
        ijump = np.clip(ijump, 0, self.discretizer.config.njump - 1)
        
        return V[ix, iy, ivx, ivy, ijump]


def save_value_function(V: np.ndarray, 
                       discretizer: StateDiscretizer,
                       goal_pos: Tuple[float, float],
                       filepath: str,
                       position_validity_mask: Optional[np.ndarray] = None):
    """Save computed value function to file.
    
    Args:
        V: Value function array
        discretizer: State discretizer (for grid config)
        goal_pos: Goal position
        filepath: Output file path (.npz format)
        position_validity_mask: Optional validity mask to store
    """
    save_dict = {
        'V': V,
        'grid_config': {
            'x_min': discretizer.config.x_min,
            'x_max': discretizer.config.x_max,
            'y_min': discretizer.config.y_min,
            'y_max': discretizer.config.y_max,
            'vx_min': discretizer.config.vx_min,
            'vx_max': discretizer.config.vx_max,
            'vy_min': discretizer.config.vy_min,
            'vy_max': discretizer.config.vy_max,
            'jump_min': discretizer.config.jump_min,
            'jump_max': discretizer.config.jump_max,
            'dx': discretizer.config.dx,
            'dy': discretizer.config.dy,
            'dvx': discretizer.config.dvx,
            'dvy': discretizer.config.dvy,
            'djump': discretizer.config.djump,
        },
        'goal_pos': goal_pos,
    }
    
    if position_validity_mask is not None:
        save_dict['position_validity_mask'] = position_validity_mask
    
    np.savez_compressed(filepath, **save_dict)
    logger.info(f"Value function saved to {filepath}")


def load_value_function(filepath: str) -> Tuple[np.ndarray, StateDiscretizer, Tuple[float, float]]:
    """Load value function from file.
    
    Args:
        filepath: Input file path (.npz format)
    
    Returns:
        Tuple of (V, discretizer, goal_pos)
    """
    data = np.load(filepath, allow_pickle=True)
    V = data['V']
    grid_cfg = data['grid_config'].item()
    goal_pos = tuple(data['goal_pos'])
    
    # Reconstruct grid config
    config = GridConfig(**grid_cfg)
    discretizer = StateDiscretizer(config)
    
    logger.info(f"Value function loaded from {filepath}")
    logger.info(f"  Shape: {V.shape}")
    logger.info(f"  Goal: {goal_pos}")
    
    return V, discretizer, goal_pos
