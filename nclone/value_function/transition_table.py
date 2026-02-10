"""Transition table precomputation for accelerated value iteration.

This module precomputes all state transitions (s, a) -> s' for the
entire state space, enabling vectorized Bellman updates.

Precomputing 15.4M states × 6 actions = 92.4M transitions is expensive
(~25 hours CPU) but enables massive speedup for value iteration:
- CPU value iteration: 2-3 hours -> 10-30 minutes
- GPU value iteration: 10-30 minutes -> 1-5 minutes

The transition table is parallelizable across CPU cores.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
import multiprocessing as mp
from functools import partial

try:
    from tqdm import tqdm
except ImportError:
    # Simple fallback if tqdm not available
    def tqdm(iterable, disable=False, desc="", total=None):
        """Simple progress indicator fallback."""
        if not disable:
            print(f"{desc}...")
        return iterable

from .state_discretization import StateDiscretizer, GridConfig
from .physics_step_wrapper import PhysicsStepWrapper
from .value_iteration import ACTIONS

logger = logging.getLogger(__name__)


class TransitionTable:
    """Precomputed transition table for state-action pairs.
    
    Stores next-state indices for all (state, action) pairs, enabling
    O(1) lookup during value iteration instead of O(physics_step).
    
    Storage format:
        transitions: shape (nx, ny, nvx, nvy, njump, num_actions, 6)
        Last dimension: [ix', iy', ivx', ivy', ijump', valid]
        - (ix', iy', ivx', ivy', ijump'): Next state indices
        - valid: 1 if transition is valid, 0 if leads to death
    
    Memory: 15.4M × 6 actions × 6 int16 = ~555MB per level
    """
    
    def __init__(self, discretizer: StateDiscretizer):
        """Initialize transition table.
        
        Args:
            discretizer: State discretizer for grid operations
        """
        self.discretizer = discretizer
        self.shape = discretizer.config.get_shape()
        self.num_actions = len(ACTIONS)
        
        # Transition table: (nx, ny, nvx, nvy, njump, num_actions, 6)
        self.transitions = None
    
    def precompute(self, 
                   physics: PhysicsStepWrapper,
                   num_workers: int = 4,
                   batch_size: int = 1000,
                   verbose: bool = True,
                   position_validity_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Precompute all state transitions using multiprocessing.
        
        OPTIMIZATION: When position_validity_mask is provided, states at
        invalid positions are skipped entirely (early-exit), avoiding expensive
        physics computation for states that cannot be occupied.
        
        Args:
            physics: Physics step wrapper (frames_per_step from this is used)
            num_workers: Number of parallel workers (default 4)
            batch_size: States per batch for progress tracking
            verbose: Show progress bar
            position_validity_mask: Optional boolean mask (nx, ny) indicating valid positions.
                                   Transitions FROM invalid positions are skipped
                                   (early-exit optimization) and marked invalid.
        
        Returns:
            Transition table array
        """
        frames_per_step = physics.frames_per_step
        logger.info(f"Precomputing transition table with {num_workers} workers...")
        logger.info(f"  Total states: {np.prod(self.shape):,}")
        logger.info(f"  Total transitions: {np.prod(self.shape) * self.num_actions:,}")
        logger.info(f"  Frames per step: {frames_per_step}")
        
        # Allocate transition table
        table_shape = self.shape + (self.num_actions, 6)
        self.transitions = np.zeros(table_shape, dtype=np.int16)
        
        # Get all state indices
        all_indices = list(self.discretizer.get_all_states())
        total_states = len(all_indices)
        
        # Early-exit optimization: filter out states at invalid positions
        if position_validity_mask is not None:
            valid_indices = []
            for idx in all_indices:
                ix, iy = idx[0], idx[1]
                if position_validity_mask[ix, iy]:
                    valid_indices.append(idx)
            
            # Count unique invalid positions (not states)
            invalid_pos_count = np.sum(~position_validity_mask)
            # States skipped = invalid positions × velocity × jump combinations
            states_per_position = self.shape[2] * self.shape[3] * self.shape[4]
            skipped_states = int(invalid_pos_count * states_per_position)
            
            logger.info(f"  Early-exit optimization: skipping {invalid_pos_count:,} "
                       f"invalid positions ({skipped_states:,} states)")
            all_indices = valid_indices
        
        compute_states = len(all_indices)
        logger.info(f"  States to compute: {compute_states:,} / {total_states:,} "
                   f"({100.0 * compute_states / total_states:.1f}%)")
        
        # Extract grid config as dict for passing to workers
        config = self.discretizer.config
        grid_config_dict = {
            'x_min': config.x_min, 'x_max': config.x_max,
            'y_min': config.y_min, 'y_max': config.y_max,
            'vx_min': config.vx_min, 'vx_max': config.vx_max,
            'vy_min': config.vy_min, 'vy_max': config.vy_max,
            'jump_min': config.jump_min, 'jump_max': config.jump_max,
            'dx': config.dx, 'dy': config.dy,
            'dvx': config.dvx, 'dvy': config.dvy,
            'djump': config.djump,
        }
        
        if num_workers <= 1:
            # Single-threaded (for debugging)
            logger.info("Using single-threaded computation")
            for idx in tqdm(all_indices, disable=not verbose, desc="Computing transitions"):
                self._compute_state_transitions(idx, physics)
        else:
            # Multi-threaded
            logger.info(f"Using {num_workers} workers")
            
            # Create batches
            batches = [all_indices[i:i+batch_size] 
                      for i in range(0, compute_states, batch_size)]
            
            # Process batches with progress bar
            with mp.Pool(num_workers) as pool:
                worker_fn = partial(self._compute_batch_transitions, 
                                   physics_sim=physics.sim,
                                   grid_config_dict=grid_config_dict,
                                   frames_per_step=frames_per_step)
                
                for batch_results in tqdm(
                    pool.imap(worker_fn, batches),
                    total=len(batches),
                    disable=not verbose,
                    desc="Transition batches"
                ):
                    # Merge batch results into main table
                    for idx, transitions in batch_results:
                        self.transitions[idx] = transitions
        
        logger.info("Transition table computation complete")
        
        # Log statistics about valid vs invalid transitions
        valid_count = np.sum(self.transitions[..., 5] == 1)
        invalid_count = np.sum(self.transitions[..., 5] == 0)
        total_trans = np.prod(self.transitions.shape[:-1])
        logger.info(f"  Valid transitions: {valid_count:,} / {total_trans:,} ({100.0 * valid_count / total_trans:.2f}%)")
        
        return self.transitions
    
    def _compute_state_transitions(self, idx: Tuple[int, int, int, int, int],
                                   physics: PhysicsStepWrapper):
        """Compute transitions for a single state (all actions).
        
        Args:
            idx: State index (ix, iy, ivx, ivy, ijump)
            physics: Physics step wrapper
        """
        # Get continuous state
        state = self.discretizer.index_to_state(idx)
        
        # Compute transition for each action
        for ia, action in enumerate(ACTIONS):
            try:
                next_state, is_death = physics.step(state, action)
                
                if is_death:
                    # Mark as invalid (leads to death)
                    self.transitions[idx][ia] = [-1, -1, -1, -1, -1, 0]
                else:
                    # Get discrete index of next state
                    next_idx = self.discretizer.state_to_index(next_state)
                    self.transitions[idx][ia] = [*next_idx, 1]
            except Exception as e:
                # Physics error - mark as invalid
                logger.debug(f"Physics error at {idx}, action {ia}: {e}")
                self.transitions[idx][ia] = [-1, -1, -1, -1, -1, 0]
    
    @staticmethod
    def _compute_batch_transitions(batch_indices, physics_sim, grid_config_dict=None, frames_per_step=5):
        """Compute transitions for a batch of states (worker function).
        
        This function is executed in parallel workers. It creates its
        own PhysicsStepWrapper instance to avoid sharing state.
        
        Args:
            batch_indices: List of state indices
            physics_sim: Simulator instance (copied per worker)
            grid_config_dict: Optional dict of grid config parameters
            frames_per_step: Number of physics frames per transition step
        
        Returns:
            List of (idx, transitions) tuples
        """
        # Create worker-local physics wrapper with multi-frame stepping
        wrapper = PhysicsStepWrapper(physics_sim, frames_per_step=frames_per_step)
        
        # Use provided grid config or defaults
        if grid_config_dict:
            config = GridConfig(**grid_config_dict)
            discretizer = StateDiscretizer(config)
        else:
            discretizer = StateDiscretizer()
        
        results = []
        for idx in batch_indices:
            state = discretizer.index_to_state(idx)
            transitions = np.zeros((len(ACTIONS), 6), dtype=np.int16)
            
            for ia, action in enumerate(ACTIONS):
                try:
                    next_state, is_death = wrapper.step(state, action)
                    
                    if is_death:
                        transitions[ia] = [-1, -1, -1, -1, -1, 0]
                    else:
                        next_idx = discretizer.state_to_index(next_state)
                        transitions[ia] = [*next_idx, 1]
                except Exception:
                    transitions[ia] = [-1, -1, -1, -1, -1, 0]
            
            results.append((idx, transitions))
        
        return results
    
    def save(self, filepath: str):
        """Save transition table to file.
        
        Args:
            filepath: Output file path (.npz format)
        """
        if self.transitions is None:
            raise ValueError("Transition table not computed yet")
        
        np.savez_compressed(
            filepath,
            transitions=self.transitions,
            shape=self.shape,
        )
        
        size_mb = self.transitions.nbytes / (1024 * 1024)
        logger.info(f"Transition table saved to {filepath} ({size_mb:.1f} MB)")
    
    def load(self, filepath: str):
        """Load transition table from file.
        
        Args:
            filepath: Input file path (.npz format)
        """
        data = np.load(filepath)
        self.transitions = data['transitions']
        expected_shape = tuple(data['shape'])
        
        if self.transitions.shape[:5] != expected_shape:
            raise ValueError(
                f"Shape mismatch: expected {expected_shape}, "
                f"got {self.transitions.shape[:5]}"
            )
        
        size_mb = self.transitions.nbytes / (1024 * 1024)
        logger.info(f"Transition table loaded from {filepath} ({size_mb:.1f} MB)")
    
    def get_next_state_indices(self, idx: Tuple[int, int, int, int, int], 
                               action_idx: int) -> Optional[Tuple[int, int, int, int, int]]:
        """Get next state indices for a given state-action pair.
        
        Args:
            idx: State index (ix, iy, ivx, ivy, ijump)
            action_idx: Action index [0, 5]
        
        Returns:
            Next state indices, or None if transition leads to death
        """
        if self.transitions is None:
            raise ValueError("Transition table not loaded")
        
        trans = self.transitions[idx][action_idx]
        valid = trans[5]
        
        if valid == 0:
            return None  # Death state
        
        return tuple(trans[:5])


def value_iteration_with_table(V_init: np.ndarray,
                               transition_table: TransitionTable,
                               max_iterations: int = 500,
                               tolerance: float = 0.01,
                               verbose: bool = True,
                               position_validity_mask: Optional[np.ndarray] = None,
                               frames_per_step: int = 5) -> np.ndarray:
    """Accelerated value iteration using precomputed transitions.
    
    This version uses VECTORIZED numpy operations for massive speedup.
    Instead of Python loops over 18M states, it uses advanced indexing
    to compute all Bellman updates in parallel.
    
    Args:
        V_init: Initial value function (with goals=0, deaths=inf)
        transition_table: Precomputed transition table
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        verbose: Show progress
        position_validity_mask: Optional boolean mask (nx, ny) indicating valid positions.
                               Invalid positions are never updated to finite values.
        frames_per_step: Physics frames per transition (for correct cost calculation)
    
    Returns:
        Converged value function V*
    """
    V = V_init.copy()
    shape = V.shape
    num_actions = transition_table.num_actions
    trans = transition_table.transitions  # (nx, ny, nvx, nvy, njump, num_actions, 6)
    
    logger.info(f"Starting VECTORIZED value iteration...")
    logger.info(f"  Grid shape: {shape}")
    logger.info(f"  Total states: {np.prod(shape):,}")
    
    # Extract transition indices and validity masks for all actions
    # Shape: (nx, ny, nvx, nvy, njump, num_actions)
    valid_mask = trans[..., 5] == 1
    
    # Separate index arrays for advanced indexing
    # Shape: (nx, ny, nvx, nvy, njump, num_actions)
    next_ix = trans[..., 0].astype(np.int32)
    next_iy = trans[..., 1].astype(np.int32)
    next_ivx = trans[..., 2].astype(np.int32)
    next_ivy = trans[..., 3].astype(np.int32)
    next_ijump = trans[..., 4].astype(np.int32)
    
    # Clamp invalid indices to 0 (they'll be masked out anyway)
    next_ix = np.where(valid_mask, next_ix, 0)
    next_iy = np.where(valid_mask, next_iy, 0)
    next_ivx = np.where(valid_mask, next_ivx, 0)
    next_ivy = np.where(valid_mask, next_ivy, 0)
    next_ijump = np.where(valid_mask, next_ijump, 0)
    
    # Identify goal states (don't update these)
    goal_mask = V == 0.0
    
    # Expand position validity mask to 5D if provided
    invalid_position_mask_5d = None
    if position_validity_mask is not None:
        # Expand (nx, ny) to (nx, ny, nvx, nvy, njump) by broadcasting
        invalid_position_mask_5d = ~position_validity_mask[:, :, None, None, None]
    
    for iteration in tqdm(range(max_iterations), disable=not verbose, desc="Value iteration"):
        V_old = V.copy()
        
        # VECTORIZED Bellman update:
        # For each state and action, look up V(next_state) using advanced indexing
        # V_next shape: (nx, ny, nvx, nvy, njump, num_actions)
        V_next = V_old[next_ix, next_iy, next_ivx, next_ivy, next_ijump]
        
        # Add step cost (frames_per_step) and mask invalid transitions with inf
        Q_values = np.where(valid_mask, float(frames_per_step) + V_next, np.inf)
        
        # Take min over actions: best_V shape (nx, ny, nvx, nvy, njump)
        best_V = np.min(Q_values, axis=-1)
        
        # Don't update goal states or invalid positions
        if invalid_position_mask_5d is not None:
            V = np.where(goal_mask | invalid_position_mask_5d, V_old, best_V)
        else:
            V = np.where(goal_mask, 0.0, best_V)
        
        # Compute convergence metric
        changed = V != V_old
        if np.any(changed):
            max_change = np.max(np.abs(V[changed] - V_old[changed]))
        else:
            max_change = 0.0
        
        updates_this_iter = np.sum(changed & ~goal_mask)
        reachable = np.sum(V < np.inf)
        
        # Convergence check
        if max_change < tolerance:
            logger.info(f"Converged after {iteration + 1} iterations (max_change={max_change:.6f}, updates={updates_this_iter:,})")
            break
        
        if verbose and (iteration + 1) % 10 == 0:
            logger.info(f"  Iter {iteration + 1}: max_change={max_change:.4f}, reachable={reachable:,}, updates={updates_this_iter:,}")
    
    return V
