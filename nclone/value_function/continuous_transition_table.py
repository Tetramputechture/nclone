"""Continuous transition table and Semi-Lagrangian value iteration.

This module implements the Semi-Lagrangian approach to value iteration:
instead of snapping next-states to grid indices (int16), it stores exact
continuous-space next-states as fractional grid coordinates (float32).
During value iteration, 4D multilinear interpolation evaluates V at these
continuous points, producing meaningful within-cell gradients even when
the agent moves less than one grid cell per step.

This eliminates the flat-gradient problem where standard discrete VI
produces zero PBRS gradient within cells because transitions map back
to the same cell index.

Key insight: even a 1.5px displacement within a 12px cell is captured
precisely as a fractional position. Multilinear interpolation weights
the surrounding grid values differently depending on the exact sub-grid
position, so the Bellman backup produces distinct values for neighboring
grid points even when their transitions land in the same cell.

Reference: Falcone & Ferretti (SIAM, 2014) - Semi-Lagrangian schemes
for Hamilton-Jacobi-Bellman equations.
"""

import numpy as np
from typing import Tuple, Optional
import logging
import multiprocessing as mp
from functools import partial

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, disable=False, desc="", total=None):
        """Simple progress indicator fallback."""
        if not disable:
            print(f"{desc}...")
        return iterable

from .state_discretization import StateDiscretizer, GridConfig
from .physics_step_wrapper import PhysicsStepWrapper
from .value_iteration import ACTIONS

logger = logging.getLogger(__name__)


class ContinuousTransitionTable:
    """Precomputed transition table storing continuous next-state coordinates.

    Unlike TransitionTable which snaps next-states to grid indices (int16),
    this class stores exact fractional grid coordinates (float32). This
    preserves sub-grid position information critical for Semi-Lagrangian
    value iteration with multilinear interpolation.

    Storage format:
        transitions: shape (nx, ny, nvx, nvy, njump, num_actions, 6) float32
        Last dimension: [fx', fy', fvx', fvy', ijump', valid]
        - (fx', fy', fvx', fvy'): Fractional grid coordinates of next state
        - ijump': Discrete jump state index (integer stored as float)
        - valid: 1.0 if transition is valid, 0.0 if leads to death

    Memory: ~2.6GB uncompressed per level (15.4M states x 6 actions x 6 float32)
    """

    def __init__(self, discretizer: StateDiscretizer):
        """Initialize continuous transition table.

        Args:
            discretizer: State discretizer for grid operations
        """
        self.discretizer = discretizer
        self.shape = discretizer.config.get_shape()
        self.num_actions = len(ACTIONS)
        self.transitions = None

    def precompute(self, physics: PhysicsStepWrapper, num_workers: int = 4,
                   batch_size: int = 1000, verbose: bool = True,
                   position_validity_mask: Optional[np.ndarray] = None
                   ) -> np.ndarray:
        """Precompute all continuous state transitions using multiprocessing.

        For each (state, action) pair, simulates physics and stores the
        resulting next-state as fractional grid coordinates instead of
        snapping to the nearest grid index.

        OPTIMIZATION: When position_validity_mask is provided, states at
        invalid positions are skipped entirely (early-exit), avoiding expensive
        physics computation for states that cannot be occupied.

        Args:
            physics: Physics step wrapper
            num_workers: Number of parallel workers
            batch_size: States per batch for progress tracking
            verbose: Show progress bar
            position_validity_mask: Optional boolean mask (nx, ny) for valid
                positions. Transitions FROM invalid positions are skipped
                (early-exit optimization) and marked invalid.

        Returns:
            Continuous transition table array (float32)
        """
        frames_per_step = physics.frames_per_step
        logger.info("Precomputing CONTINUOUS transition table...")
        logger.info(f"  Total states: {np.prod(self.shape):,}")
        logger.info(f"  Total transitions: {np.prod(self.shape) * self.num_actions:,}")
        logger.info(f"  Frames per step: {frames_per_step}")
        logger.info("  Storage type: float32 (continuous fractional coordinates)")

        table_shape = self.shape + (self.num_actions, 6)
        self.transitions = np.zeros(table_shape, dtype=np.float32)

        # Filter out invalid positions early to skip physics computation
        all_indices = list(self.discretizer.get_all_states())
        total_states = len(all_indices)
        
        # Early-exit optimization: filter out states at invalid positions
        skipped_positions = 0
        if position_validity_mask is not None:
            valid_indices = []
            for idx in all_indices:
                ix, iy = idx[0], idx[1]
                if position_validity_mask[ix, iy]:
                    valid_indices.append(idx)
                else:
                    skipped_positions += 1
            
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
            logger.info("Using single-threaded computation")
            for idx in tqdm(all_indices, disable=not verbose,
                            desc="Computing continuous transitions"):
                self._compute_state_transitions(idx, physics)
        else:
            logger.info(f"Using {num_workers} workers")
            batches = [all_indices[i:i + batch_size]
                       for i in range(0, compute_states, batch_size)]

            worker_fn = partial(
                self._compute_batch_transitions,
                physics_sim=physics.sim,
                grid_config_dict=grid_config_dict,
                frames_per_step=frames_per_step,
            )

            with mp.Pool(num_workers) as pool:
                for batch_results in tqdm(
                    pool.imap(worker_fn, batches),
                    total=len(batches),
                    disable=not verbose,
                    desc="Continuous transition batches",
                ):
                    for idx, transitions in batch_results:
                        self.transitions[idx] = transitions

        logger.info("Continuous transition table computation complete")
        valid_count = int(np.sum(self.transitions[..., 5] > 0.5))
        total_trans = int(np.prod(self.transitions.shape[:-1]))
        logger.info(f"  Valid transitions: {valid_count:,} / {total_trans:,} "
                     f"({100.0 * valid_count / total_trans:.2f}%)")

        return self.transitions

    def _compute_state_transitions(self, idx: Tuple[int, int, int, int, int],
                                   physics: PhysicsStepWrapper):
        """Compute continuous transitions for a single state (all actions).

        Args:
            idx: State index (ix, iy, ivx, ivy, ijump)
            physics: Physics step wrapper
        """
        state = self.discretizer.index_to_state(idx)
        njump = self.discretizer.config.njump

        for ia, action in enumerate(ACTIONS):
            try:
                next_state, is_death = physics.step(state, action)
                if is_death:
                    self.transitions[idx][ia] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                else:
                    frac = self.discretizer.state_to_fractional(next_state)
                    fx, fy, fvx, fvy, fjump = frac
                    ijump = int(round(min(max(fjump, 0), njump - 1)))
                    self.transitions[idx][ia] = [
                        fx, fy, fvx, fvy, float(ijump), 1.0
                    ]
            except Exception as e:
                logger.debug(f"Physics error at {idx}, action {ia}: {e}")
                self.transitions[idx][ia] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    @staticmethod
    def _compute_batch_transitions(batch_indices, physics_sim,
                                   grid_config_dict=None, frames_per_step=5):
        """Compute continuous transitions for a batch of states (worker).

        This function is executed in parallel workers. It creates its
        own PhysicsStepWrapper instance to avoid sharing state.

        Args:
            batch_indices: List of state indices
            physics_sim: Simulator instance (copied per worker)
            grid_config_dict: Optional dict of grid config parameters
            frames_per_step: Number of physics frames per transition step

        Returns:
            List of (idx, transitions) tuples with float32 transitions
        """
        wrapper = PhysicsStepWrapper(physics_sim, frames_per_step=frames_per_step)

        if grid_config_dict:
            config = GridConfig(**grid_config_dict)
            discretizer = StateDiscretizer(config)
        else:
            discretizer = StateDiscretizer()

        njump = discretizer.config.njump
        results = []
        for idx in batch_indices:
            state = discretizer.index_to_state(idx)
            transitions = np.zeros((len(ACTIONS), 6), dtype=np.float32)

            for ia, action in enumerate(ACTIONS):
                try:
                    next_state, is_death = wrapper.step(state, action)
                    if is_death:
                        transitions[ia] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    else:
                        frac = discretizer.state_to_fractional(next_state)
                        fx, fy, fvx, fvy, fjump = frac
                        ijump = int(round(min(max(fjump, 0), njump - 1)))
                        transitions[ia] = [
                            fx, fy, fvx, fvy, float(ijump), 1.0
                        ]
                except Exception:
                    transitions[ia] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            results.append((idx, transitions))

        return results

    def save(self, filepath: str):
        """Save continuous transition table to file.

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
        logger.info(f"Continuous transition table saved to {filepath} "
                     f"({size_mb:.1f} MB uncompressed)")

    def load(self, filepath: str):
        """Load continuous transition table from file.

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

        # Verify float32 (continuous format, not int16 discrete format)
        if self.transitions.dtype != np.float32:
            raise ValueError(
                f"Expected float32 continuous transitions, got "
                f"{self.transitions.dtype}. This may be a discrete "
                f"transition table instead."
            )

        size_mb = self.transitions.nbytes / (1024 * 1024)
        logger.info(f"Continuous transition table loaded from {filepath} "
                     f"({size_mb:.1f} MB)")


def sl_value_iteration_with_table(
    V_init: np.ndarray,
    continuous_table: ContinuousTransitionTable,
    max_iterations: int = 500,
    tolerance: float = 0.01,
    verbose: bool = True,
    position_validity_mask: Optional[np.ndarray] = None,
    frames_per_step: int = 5,
) -> np.ndarray:
    # #region agent log
    import json
    _debug_log_path = "/home/tetra/projects/nclone/.cursor/debug.log"
    def _debug_log(location, message, data, hypothesis_id):
        with open(_debug_log_path, "a") as f:
            f.write(json.dumps({"location": location, "message": message, "data": data, "hypothesisId": hypothesis_id, "timestamp": __import__("time").time()}) + "\n")
    # #endregion
    """Semi-Lagrangian value iteration using continuous precomputed transitions.

    Unlike standard value iteration which looks up V at snapped grid indices,
    this method uses 4D multilinear interpolation to evaluate V at the exact
    continuous next-state. This produces meaningful within-cell gradients
    even when transitions stay within the same grid cell.

    The interpolation operates over (x, y, vx, vy) with the jump dimension
    handled discretely (no interpolation needed since jump_frames transitions
    deterministically). Each interpolation evaluates 2^4 = 16 grid corners.

    Args:
        V_init: Initial value function (goals=0, elsewhere=inf)
        continuous_table: Precomputed continuous transition table
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        verbose: Show progress
        position_validity_mask: Optional boolean mask (nx, ny) for valid positions
        frames_per_step: Physics frames per transition step

    Returns:
        Converged value function V*
    """
    V = V_init.copy()
    shape = V.shape
    nx, ny, nvx, nvy, njump = shape
    trans = continuous_table.transitions
    num_actions = continuous_table.num_actions

    logger.info("Starting Semi-Lagrangian value iteration...")
    logger.info(f"  Grid shape: {shape}")
    logger.info(f"  Total states: {np.prod(shape):,}")
    logger.info("  Interpolation: 4D multilinear (16 corners per lookup)")

    # #region agent log
    # Hypothesis B: Check how many transitions are valid
    valid_trans_count = int(np.sum(trans[..., 5] > 0.5))
    total_trans_count = int(np.prod(trans.shape[:-1]))
    _debug_log("sl_vi:init", "Transition validity stats", {
        "valid_transitions": valid_trans_count,
        "total_transitions": total_trans_count,
        "valid_pct": round(100.0 * valid_trans_count / total_trans_count, 2)
    }, "B")
    
    # Hypothesis E: Check goal state count and positions
    goal_count = int(np.sum(V_init == 0.0))
    goal_positions = []
    for ix in range(nx):
        for iy in range(ny):
            if V_init[ix, iy, 0, 0, 0] == 0.0:
                goal_positions.append((ix, iy))
    _debug_log("sl_vi:init", "Goal state statistics", {
        "goal_states_total": goal_count,
        "goal_position_cells": len(goal_positions),
        "goal_positions": goal_positions[:10]
    }, "E")
    # #endregion

    # --- Precompute interpolation indices and weights (one-time cost) ---
    # Extract floor indices, fractional weights, and jump indices from
    # the continuous transition table. These are constant across iterations.
    # Shape of each array: (nx, ny, nvx, nvy, njump, num_actions)

    EPS = 1e-3  # Keep fractional coords strictly within interpolation bounds

    valid_mask = trans[..., 5] > 0.5
    
    # #region agent log
    # Hypothesis C & D: Analyze where transitions land
    # Sample transitions from goal-adjacent positions
    trans_analysis = {"self_loops": 0, "to_goal": 0, "to_valid_nongoal": 0, "to_invalid": 0, "deaths": 0, "sampled": 0}
    goal_set = set(goal_positions)
    for gix, giy in goal_positions[:2]:
        for dix in range(-5, 6):
            for diy in range(-5, 6):
                src_ix, src_iy = gix + dix, giy + diy
                if 0 <= src_ix < nx and 0 <= src_iy < ny:
                    for ivx in [7]:  # Center velocity
                        for ivy in [13]:
                            for ijump in [0]:
                                for ia in range(num_actions):
                                    trans_analysis["sampled"] += 1
                                    if trans[src_ix, src_iy, ivx, ivy, ijump, ia, 5] < 0.5:
                                        trans_analysis["deaths"] += 1
                                    else:
                                        dst_fx = trans[src_ix, src_iy, ivx, ivy, ijump, ia, 0]
                                        dst_fy = trans[src_ix, src_iy, ivx, ivy, ijump, ia, 1]
                                        dst_ix = int(np.clip(np.floor(dst_fx), 0, nx-1))
                                        dst_iy = int(np.clip(np.floor(dst_fy), 0, ny-1))
                                        
                                        if dst_ix == src_ix and dst_iy == src_iy:
                                            trans_analysis["self_loops"] += 1
                                        elif (dst_ix, dst_iy) in goal_set:
                                            trans_analysis["to_goal"] += 1
                                        elif position_validity_mask is not None and not position_validity_mask[dst_ix, dst_iy]:
                                            trans_analysis["to_invalid"] += 1
                                        else:
                                            trans_analysis["to_valid_nongoal"] += 1
    _debug_log("sl_vi:trans_analysis", "Transition destination analysis near goal", trans_analysis, "C,D")
    # #endregion

    # Clamp fractional coordinates; set invalid transitions to safe defaults
    clamped_fx = np.clip(trans[..., 0], 0, nx - 1 - EPS)
    clamped_fy = np.clip(trans[..., 1], 0, ny - 1 - EPS)
    clamped_fvx = np.clip(trans[..., 2], 0, nvx - 1 - EPS)
    clamped_fvy = np.clip(trans[..., 3], 0, nvy - 1 - EPS)

    fx = np.where(valid_mask, clamped_fx, 0.0).astype(np.float32)
    fy = np.where(valid_mask, clamped_fy, 0.0).astype(np.float32)
    fvx = np.where(valid_mask, clamped_fvx, 0.0).astype(np.float32)
    fvy = np.where(valid_mask, clamped_fvy, 0.0).astype(np.float32)
    del clamped_fx, clamped_fy, clamped_fvx, clamped_fvy

    # Discrete jump index for next state
    ijump_next = np.where(
        valid_mask,
        np.clip(trans[..., 4].astype(np.int32), 0, njump - 1),
        0,
    )

    # Floor indices for interpolation (clamped so floor+1 is always valid)
    ix0 = np.clip(np.floor(fx).astype(np.int32), 0, nx - 2)
    iy0 = np.clip(np.floor(fy).astype(np.int32), 0, ny - 2)
    ivx0 = np.clip(np.floor(fvx).astype(np.int32), 0, nvx - 2)
    ivy0 = np.clip(np.floor(fvy).astype(np.int32), 0, nvy - 2)

    # Interpolation weights (fractional part = weight for upper/ceil index)
    wx = np.clip(fx - ix0, 0.0, 1.0).astype(np.float32)
    wy = np.clip(fy - iy0, 0.0, 1.0).astype(np.float32)
    wvx = np.clip(fvx - ivx0, 0.0, 1.0).astype(np.float32)
    wvy = np.clip(fvy - ivy0, 0.0, 1.0).astype(np.float32)
    del fx, fy, fvx, fvy  # Free fractional coord arrays

    logger.info("  Precomputed interpolation indices and weights")
    logger.info(f"  Valid transitions: {int(np.sum(valid_mask)):,}")

    # Goal and position validity masks
    goal_mask = V == 0.0
    invalid_5d = None
    if position_validity_mask is not None:
        invalid_5d = ~position_validity_mask[:, :, None, None, None]

    # FIX for Hypothesis A: Use reasonable max value instead of inf for interpolation
    # When V has inf values, multilinear interpolation breaks (inf * weight = inf).
    # We use a reasonable maximum (enough for any level traversal) so interpolation
    # produces meaningful gradients without contaminating values with 1e9.
    # Level is 1056px, at ~3px/frame = ~350 frames. With 5 frames/step = 70 steps max.
    # Use 10000 frames as conservative max (allows for complex paths).
    INF_REPLACEMENT = 10000.0  # Reasonable max frames - prevents contamination
    
    # --- Value iteration loop ---
    for iteration in tqdm(range(max_iterations), disable=not verbose,
                          desc="SL-VI"):
        V_old = V.copy()
        best_V = np.full(shape, np.inf, dtype=np.float32)
        
        # FIX: Create interpolation-safe version of V_old with inf replaced
        V_old_safe = np.where(V_old == np.inf, INF_REPLACEMENT, V_old).astype(np.float32)
        
        # #region agent log
        # Hypothesis A: After first few iterations, sample interpolation diagnostics
        debug_interp_samples = []
        debug_sample_count = 0
        debug_max_samples = 20
        # #endregion

        # Process each action and accumulate the best Q-value
        for ia in range(num_actions):
            # Extract this action's precomputed data (views, no copy)
            a_ix0 = ix0[..., ia]
            a_iy0 = iy0[..., ia]
            a_ivx0 = ivx0[..., ia]
            a_ivy0 = ivy0[..., ia]
            a_wx = wx[..., ia]
            a_wy = wy[..., ia]
            a_wvx = wvx[..., ia]
            a_wvy = wvy[..., ia]
            a_ijump = ijump_next[..., ia]
            a_valid = valid_mask[..., ia]

            # 4D multilinear interpolation with discrete jump indexing.
            # Sum 2^4=16 corners of the (x,y,vx,vy) hypercube, each
            # looked up in V_old_safe at the target jump state.
            # Fully unrolled for better performance (no Python loop overhead).
            # NOTE: Using V_old_safe (inf replaced with large finite) to allow propagation
            
            # Precompute complementary weights
            a_wx1 = 1.0 - a_wx
            a_wy1 = 1.0 - a_wy
            a_wvx1 = 1.0 - a_wvx
            a_wvy1 = 1.0 - a_wvy
            
            # Index offsets for adjacent cells
            ix1 = a_ix0 + 1
            iy1 = a_iy0 + 1
            ivx1 = a_ivx0 + 1
            ivy1 = a_ivy0 + 1
            
            # Compute all 16 corners in one vectorized expression
            V_interp = (
                # Corner (0,0,0,0)
                a_wx1 * a_wy1 * a_wvx1 * a_wvy1 * V_old_safe[a_ix0, a_iy0, a_ivx0, a_ivy0, a_ijump] +
                # Corner (0,0,0,1)
                a_wx1 * a_wy1 * a_wvx1 * a_wvy * V_old_safe[a_ix0, a_iy0, a_ivx0, ivy1, a_ijump] +
                # Corner (0,0,1,0)
                a_wx1 * a_wy1 * a_wvx * a_wvy1 * V_old_safe[a_ix0, a_iy0, ivx1, a_ivy0, a_ijump] +
                # Corner (0,0,1,1)
                a_wx1 * a_wy1 * a_wvx * a_wvy * V_old_safe[a_ix0, a_iy0, ivx1, ivy1, a_ijump] +
                # Corner (0,1,0,0)
                a_wx1 * a_wy * a_wvx1 * a_wvy1 * V_old_safe[a_ix0, iy1, a_ivx0, a_ivy0, a_ijump] +
                # Corner (0,1,0,1)
                a_wx1 * a_wy * a_wvx1 * a_wvy * V_old_safe[a_ix0, iy1, a_ivx0, ivy1, a_ijump] +
                # Corner (0,1,1,0)
                a_wx1 * a_wy * a_wvx * a_wvy1 * V_old_safe[a_ix0, iy1, ivx1, a_ivy0, a_ijump] +
                # Corner (0,1,1,1)
                a_wx1 * a_wy * a_wvx * a_wvy * V_old_safe[a_ix0, iy1, ivx1, ivy1, a_ijump] +
                # Corner (1,0,0,0)
                a_wx * a_wy1 * a_wvx1 * a_wvy1 * V_old_safe[ix1, a_iy0, a_ivx0, a_ivy0, a_ijump] +
                # Corner (1,0,0,1)
                a_wx * a_wy1 * a_wvx1 * a_wvy * V_old_safe[ix1, a_iy0, a_ivx0, ivy1, a_ijump] +
                # Corner (1,0,1,0)
                a_wx * a_wy1 * a_wvx * a_wvy1 * V_old_safe[ix1, a_iy0, ivx1, a_ivy0, a_ijump] +
                # Corner (1,0,1,1)
                a_wx * a_wy1 * a_wvx * a_wvy * V_old_safe[ix1, a_iy0, ivx1, ivy1, a_ijump] +
                # Corner (1,1,0,0)
                a_wx * a_wy * a_wvx1 * a_wvy1 * V_old_safe[ix1, iy1, a_ivx0, a_ivy0, a_ijump] +
                # Corner (1,1,0,1)
                a_wx * a_wy * a_wvx1 * a_wvy * V_old_safe[ix1, iy1, a_ivx0, ivy1, a_ijump] +
                # Corner (1,1,1,0)
                a_wx * a_wy * a_wvx * a_wvy1 * V_old_safe[ix1, iy1, ivx1, a_ivy0, a_ijump] +
                # Corner (1,1,1,1)
                a_wx * a_wy * a_wvx * a_wvy * V_old_safe[ix1, iy1, ivx1, ivy1, a_ijump]
            )

            # Q(s, a) = step_cost + V_interp(s'); mask invalid with inf
            Q_a = np.where(
                a_valid, float(frames_per_step) + V_interp, np.inf
            )
            best_V = np.minimum(best_V, Q_a)
            
            # #region agent log
            # Hypothesis A: Sample cases near goal where V_interp might be inf due to corner infs
            if iteration == 0 and ia == 0 and debug_sample_count < debug_max_samples:
                # Find states near goal with valid transitions but inf V_interp
                for gix, giy in goal_positions[:3]:  # Sample near first few goal positions
                    for dix in range(-3, 4):
                        for diy in range(-3, 4):
                            if debug_sample_count >= debug_max_samples:
                                break
                            test_ix = gix + dix
                            test_iy = giy + diy
                            if 0 <= test_ix < nx and 0 <= test_iy < ny:
                                if V_old[test_ix, test_iy, 7, 13, 0] == np.inf:  # Center velocity, no jump
                                    if a_valid[test_ix, test_iy, 7, 13, 0]:
                                        # Get the interpolation corner values
                                        t_ix0 = int(ix0[test_ix, test_iy, 7, 13, 0, ia])
                                        t_iy0 = int(iy0[test_ix, test_iy, 7, 13, 0, ia])
                                        t_ivx0 = int(ivx0[test_ix, test_iy, 7, 13, 0, ia])
                                        t_ivy0 = int(ivy0[test_ix, test_iy, 7, 13, 0, ia])
                                        t_ijump = int(ijump_next[test_ix, test_iy, 7, 13, 0, ia])
                                        
                                        # Count how many corners are finite vs inf
                                        corner_vals = []
                                        for dx in [0, 1]:
                                            for dy in [0, 1]:
                                                for dvx in [0, 1]:
                                                    for dvy in [0, 1]:
                                                        cval = float(V_old[t_ix0+dx, t_iy0+dy, t_ivx0+dvx, t_ivy0+dvy, t_ijump])
                                                        corner_vals.append(cval)
                                        
                                        finite_corners = sum(1 for v in corner_vals if v < np.inf)
                                        interp_val = float(V_interp[test_ix, test_iy, 7, 13, 0])
                                        
                                        debug_interp_samples.append({
                                            "state_ix": test_ix, "state_iy": test_iy,
                                            "dist_to_goal": abs(dix) + abs(diy),
                                            "target_ix0": t_ix0, "target_iy0": t_iy0,
                                            "finite_corners": finite_corners,
                                            "total_corners": 16,
                                            "v_interp": interp_val if interp_val < 1e10 else "inf",
                                            "corner_sample": [v if v < 1e10 else "inf" for v in corner_vals[:4]]
                                        })
                                        debug_sample_count += 1
            # #endregion

        # Bellman update: V(s) = min_a Q(s, a)
        # CRITICAL FIX: Only update if new value is actually an IMPROVEMENT
        # Without this, self-loops cause V to increase (5 + V_old > V_old)
        # and INF_REPLACEMENT values pollute the entire state space
        improved = best_V < V_old
        
        # Preserve goal states (V=0) and invalid positions (V=inf), only update if improved
        if invalid_5d is not None:
            V = np.where(goal_mask | invalid_5d | ~improved, V_old, best_V)
        else:
            V = np.where(goal_mask | ~improved, V_old, best_V)
        
        # #region agent log
        if iteration == 0 and debug_interp_samples:
            _debug_log("sl_vi:iter0", "Interpolation corner diagnostics", {
                "samples": debug_interp_samples[:10]
            }, "A")
        # #endregion

        # --- Convergence check ---
        changed = V != V_old
        if np.any(changed):
            max_change = float(np.max(np.abs(V[changed] - V_old[changed])))
        else:
            max_change = 0.0

        updates_this_iter = int(np.sum(changed & ~goal_mask))
        reachable = int(np.sum(V < np.inf))

        if max_change < tolerance:
            logger.info(
                f"Converged after {iteration + 1} iterations "
                f"(max_change={max_change:.6f}, "
                f"updates={updates_this_iter:,})"
            )
            break

        if verbose and (iteration + 1) % 10 == 0:
            logger.info(
                f"  Iter {iteration + 1}: max_change={max_change:.4f}, "
                f"reachable={reachable:,}, updates={updates_this_iter:,}"
            )
        
        # #region agent log
        if iteration < 5:
            _debug_log("sl_vi:iter", "Iteration stats", {
                "iteration": iteration,
                "reachable_states": reachable,
                "updates": updates_this_iter,
                "max_change": max_change if max_change < 1e10 else "inf"
            }, "A")
        # #endregion

    if max_change >= tolerance:
        logger.warning(
            f"Did not converge after {max_iterations} iterations "
            f"(max_change={max_change:.6f})"
        )

    # Summary statistics
    reachable = int(np.sum(V < np.inf))
    goal_states = int(np.sum(V == 0))
    finite_V = V[V < np.inf]
    logger.info("Semi-Lagrangian value function computed:")
    logger.info(f"  Reachable states: {reachable:,} / {np.prod(shape):,}")
    logger.info(f"  Goal states: {goal_states:,}")
    if len(finite_V) > 0:
        logger.info(
            f"  Value range: [{np.min(finite_V):.1f}, {np.max(finite_V):.1f}]"
        )

    return V
