"""GPU-accelerated Semi-Lagrangian value iteration using PyTorch.

This module provides a CUDA-accelerated implementation of the Semi-Lagrangian
value iteration algorithm. The key optimizations are:

1. All arrays moved to GPU memory
2. 16-corner interpolation computed incrementally (memory-efficient)
3. All 6 actions processed in parallel within each corner iteration
4. Efficient tensor operations for value lookups

Expected speedup: 10-30x compared to NumPy CPU implementation.
On RTX 3080ti (12GB): ~10-30 minutes CPU -> ~2-5 minutes GPU.

Memory usage: ~4-6GB VRAM for default grid (fits in 12GB cards).
"""

import numpy as np
from typing import Optional, TYPE_CHECKING
import logging
import time

if TYPE_CHECKING:
    from .continuous_transition_table import ContinuousTransitionTable

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Try to import tqdm for progress
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, disable=False, desc="", total=None):
        """Simple progress indicator fallback."""
        if not disable:
            print(f"{desc}...")
        return iterable


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available.
    
    Returns:
        True if PyTorch with CUDA is available, False otherwise.
    """
    if not TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()


def get_gpu_info() -> dict:
    """Get information about available GPU.
    
    Returns:
        Dictionary with GPU info, or empty dict if no GPU available.
    """
    if not is_gpu_available():
        return {}
    
    device = torch.cuda.current_device()
    return {
        'name': torch.cuda.get_device_name(device),
        'memory_total_gb': torch.cuda.get_device_properties(device).total_memory / (1024**3),
        'memory_allocated_gb': torch.cuda.memory_allocated(device) / (1024**3),
        'compute_capability': f"{torch.cuda.get_device_properties(device).major}.{torch.cuda.get_device_properties(device).minor}",
    }


def sl_value_iteration_gpu(
    V_init: np.ndarray,
    continuous_table: 'ContinuousTransitionTable',
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
    """GPU-accelerated Semi-Lagrangian value iteration (memory-efficient).
    
    This is a drop-in replacement for sl_value_iteration_with_table() that
    uses PyTorch CUDA tensors for speedup. The algorithm is identical,
    but operations run on GPU.
    
    Memory-efficient design:
    - Computes 16 interpolation corners incrementally (not all at once)
    - Peak VRAM usage: ~4-6GB for default grid (fits 12GB cards)
    - Trades some compute for memory efficiency
    
    Args:
        V_init: Initial value function (goals=0, elsewhere=inf)
        continuous_table: Precomputed continuous transition table
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        verbose: Show progress
        position_validity_mask: Optional boolean mask (nx, ny) for valid positions
        frames_per_step: Physics frames per transition step
    
    Returns:
        Converged value function V* as numpy array
    
    Raises:
        RuntimeError: If CUDA is not available
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. Install with: pip install torch"
        )
    
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. GPU acceleration requires an NVIDIA GPU "
            "with CUDA support."
        )
    
    device = torch.device('cuda')
    start_time = time.time()
    
    # Get grid dimensions
    shape = V_init.shape
    nx, ny, nvx, nvy, njump = shape
    total_states = np.prod(shape)
    num_actions = continuous_table.num_actions
    trans = continuous_table.transitions
    
    # Log GPU info
    gpu_info = get_gpu_info()
    logger.info("Starting GPU-accelerated Semi-Lagrangian value iteration...")
    logger.info(f"  GPU: {gpu_info.get('name', 'Unknown')}")
    logger.info(f"  GPU Memory: {gpu_info.get('memory_total_gb', 0):.1f} GB")
    logger.info(f"  Grid shape: {shape}")
    logger.info(f"  Total states: {total_states:,}")
    logger.info("  Mode: Memory-efficient (incremental corner computation)")
    
    # #region agent log
    # Hypothesis B: Check how many transitions are valid
    valid_trans_count = int(np.sum(trans[..., 5] > 0.5))
    total_trans_count = int(np.prod(trans.shape[:-1]))
    _debug_log("gpu_sl_vi:init", "Transition validity stats", {
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
    _debug_log("gpu_sl_vi:init", "Goal state statistics", {
        "goal_states_total": goal_count,
        "goal_position_cells": len(goal_positions),
        "goal_positions": goal_positions[:10]
    }, "E")
    # #endregion
    
    # --- Transfer data to GPU ---
    logger.info("  Transferring data to GPU...")
    transfer_start = time.time()
    
    # Value function
    V = torch.from_numpy(V_init.copy()).to(device)
    
    # Extract transition data
    EPS = 1e-3
    valid_mask_np = trans[..., 5] > 0.5
    
    # Clamp fractional coordinates
    fx_np = np.clip(trans[..., 0], 0, nx - 1 - EPS).astype(np.float32)
    fy_np = np.clip(trans[..., 1], 0, ny - 1 - EPS).astype(np.float32)
    fvx_np = np.clip(trans[..., 2], 0, nvx - 1 - EPS).astype(np.float32)
    fvy_np = np.clip(trans[..., 3], 0, nvy - 1 - EPS).astype(np.float32)
    
    # Set invalid to 0
    fx_np = np.where(valid_mask_np, fx_np, 0.0)
    fy_np = np.where(valid_mask_np, fy_np, 0.0)
    fvx_np = np.where(valid_mask_np, fvx_np, 0.0)
    fvy_np = np.where(valid_mask_np, fvy_np, 0.0)
    
    # Floor indices (use int32 to save memory - grid dimensions are small)
    ix0_np = np.clip(np.floor(fx_np).astype(np.int32), 0, nx - 2)
    iy0_np = np.clip(np.floor(fy_np).astype(np.int32), 0, ny - 2)
    ivx0_np = np.clip(np.floor(fvx_np).astype(np.int32), 0, nvx - 2)
    ivy0_np = np.clip(np.floor(fvy_np).astype(np.int32), 0, nvy - 2)
    
    # Jump index
    ijump_np = np.where(
        valid_mask_np,
        np.clip(trans[..., 4].astype(np.int32), 0, njump - 1),
        0
    )
    
    # Interpolation weights
    wx_np = np.clip(fx_np - ix0_np, 0.0, 1.0).astype(np.float32)
    wy_np = np.clip(fy_np - iy0_np, 0.0, 1.0).astype(np.float32)
    wvx_np = np.clip(fvx_np - ivx0_np, 0.0, 1.0).astype(np.float32)
    wvy_np = np.clip(fvy_np - ivy0_np, 0.0, 1.0).astype(np.float32)
    
    # Free numpy fractional arrays
    del fx_np, fy_np, fvx_np, fvy_np
    
    # Keep numpy arrays on CPU, transfer per-action to save GPU memory
    # This approach processes one action at a time, using ~2GB instead of ~12GB
    
    # Goal mask (where V == 0)
    goal_mask = V == 0.0
    
    # Position validity mask
    invalid_5d = None
    if position_validity_mask is not None:
        invalid_5d = torch.from_numpy(
            ~position_validity_mask[:, :, None, None, None]
        ).to(device)
    
    transfer_time = time.time() - transfer_start
    logger.info(f"  Data transfer complete ({transfer_time:.2f}s)")
    logger.info(f"  Valid transitions: {int(np.sum(valid_mask_np)):,}")
    
    # #region agent log
    # Hypothesis C & D: Analyze where transitions land near goal
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
    _debug_log("gpu_sl_vi:trans_analysis", "Transition destination analysis near goal (zero vel)", trans_analysis, "C,D")
    
    # #region agent log
    # Hypothesis F: Check transitions at HIGH velocity (should actually move across cells)
    trans_high_vel = {"self_loops": 0, "pos_changes": 0, "deaths": 0, "sampled": 0}
    for gix, giy in goal_positions[:2]:
        for dix in range(-5, 6):
            src_ix, src_iy = gix + dix, giy
            if 0 <= src_ix < nx and 0 <= src_iy < ny:
                for ivx in [12, 13, 14]:  # High rightward velocity (towards goal if left of it)
                    for ivy in [13]:  # ~zero vertical
                        for ijump in [0]:
                            for ia in range(num_actions):
                                trans_high_vel["sampled"] += 1
                                if trans[src_ix, src_iy, ivx, ivy, ijump, ia, 5] < 0.5:
                                    trans_high_vel["deaths"] += 1
                                else:
                                    dst_fx = trans[src_ix, src_iy, ivx, ivy, ijump, ia, 0]
                                    dst_ix = int(np.clip(np.floor(dst_fx), 0, nx-1))
                                    if dst_ix != src_ix:
                                        trans_high_vel["pos_changes"] += 1
                                    else:
                                        trans_high_vel["self_loops"] += 1
    _debug_log("gpu_sl_vi:trans_high_vel", "Transition analysis at HIGH velocity", trans_high_vel, "F")
    # #endregion
    
    # Hypothesis A: Check interpolation corners at positions adjacent to goal
    interp_corner_analysis = []
    for gix, giy in goal_positions[:2]:
        for dix in range(-3, 4):
            for diy in range(-3, 4):
                src_ix, src_iy = gix + dix, giy + diy
                if 0 <= src_ix < nx and 0 <= src_iy < ny:
                    if V_init[src_ix, src_iy, 7, 13, 0] == np.inf:  # Not a goal state
                        for ia in range(1):  # Just first action
                            if valid_mask_np[src_ix, src_iy, 7, 13, 0, ia]:
                                t_ix0 = ix0_np[src_ix, src_iy, 7, 13, 0, ia]
                                t_iy0 = iy0_np[src_ix, src_iy, 7, 13, 0, ia]
                                t_ivx0 = ivx0_np[src_ix, src_iy, 7, 13, 0, ia]
                                t_ivy0 = ivy0_np[src_ix, src_iy, 7, 13, 0, ia]
                                t_ijump = ijump_np[src_ix, src_iy, 7, 13, 0, ia]
                                
                                # Count finite corners
                                finite_corners = 0
                                for dx in [0, 1]:
                                    for dy in [0, 1]:
                                        for dvx in [0, 1]:
                                            for dvy in [0, 1]:
                                                if V_init[t_ix0+dx, t_iy0+dy, t_ivx0+dvx, t_ivy0+dvy, t_ijump] < np.inf:
                                                    finite_corners += 1
                                
                                interp_corner_analysis.append({
                                    "src_ix": src_ix, "src_iy": src_iy, "dist_to_goal": abs(dix) + abs(diy),
                                    "tgt_ix0": int(t_ix0), "tgt_iy0": int(t_iy0),
                                    "finite_corners": finite_corners
                                })
    _debug_log("gpu_sl_vi:interp_analysis", "Interpolation corner analysis near goal", {
        "samples": interp_corner_analysis[:20]
    }, "A")
    # #endregion
    
    # Report GPU memory usage
    mem_allocated = torch.cuda.memory_allocated(device) / (1024**3)
    logger.info(f"  GPU memory used: {mem_allocated:.2f} GB")
    
    # --- Value iteration loop ---
    iteration_start = time.time()
    max_change = float('inf')
    
    # Preallocate best_V tensor
    best_V = torch.full(shape, float('inf'), dtype=torch.float32, device=device)
    
    # FIX for Hypothesis A: Use reasonable max value instead of inf for interpolation
    # When V has inf values, multilinear interpolation breaks (inf * weight = inf).
    # We use a reasonable maximum (enough for any level traversal) so interpolation
    # produces meaningful gradients without contaminating values with 1e9.
    # Level is 1056px, at ~3px/frame = ~350 frames. With 5 frames/step = 70 steps max.
    # Use 10000 frames as conservative max (allows for complex paths).
    INF_REPLACEMENT = 10000.0  # Reasonable max frames - prevents contamination
    
    for iteration in tqdm(range(max_iterations), disable=not verbose, desc="GPU SL-VI"):
        V_old = V.clone()
        best_V.fill_(float('inf'))
        
        # FIX: Create interpolation-safe version of V_old with inf replaced
        # This allows multilinear interpolation to produce finite values even when
        # some corners are unreached. The large value (1e9) naturally penalizes
        # paths through unreached states while allowing propagation.
        V_old_safe = torch.where(
            V_old == float('inf'),
            torch.tensor(INF_REPLACEMENT, device=device, dtype=torch.float32),
            V_old
        )
        
        # Process each action separately to minimize GPU memory
        # This trades some overhead for fitting in 12GB VRAM
        for ia in range(num_actions):
            # Extract this action's data (single action = 6x less memory)
            a_ix0 = torch.from_numpy(ix0_np[..., ia].astype(np.int64)).to(device)
            a_iy0 = torch.from_numpy(iy0_np[..., ia].astype(np.int64)).to(device)
            a_ivx0 = torch.from_numpy(ivx0_np[..., ia].astype(np.int64)).to(device)
            a_ivy0 = torch.from_numpy(ivy0_np[..., ia].astype(np.int64)).to(device)
            a_ijump = torch.from_numpy(ijump_np[..., ia].astype(np.int64)).to(device)
            
            a_wx = torch.from_numpy(wx_np[..., ia]).to(device)
            a_wy = torch.from_numpy(wy_np[..., ia]).to(device)
            a_wvx = torch.from_numpy(wvx_np[..., ia]).to(device)
            a_wvy = torch.from_numpy(wvy_np[..., ia]).to(device)
            
            a_valid = torch.from_numpy(valid_mask_np[..., ia]).to(device)
            
            # Precompute 1-weights
            a_wx1 = 1.0 - a_wx
            a_wy1 = 1.0 - a_wy
            a_wvx1 = 1.0 - a_wvx
            a_wvy1 = 1.0 - a_wvy
            
            # 4D multilinear interpolation - fully vectorized (no Python loop)
            # Precompute all 16 weight products and gather all corners at once
            # This eliminates Python loop overhead and allows better GPU parallelism
            
            # Index offsets for adjacent cells
            ix1 = a_ix0 + 1
            iy1 = a_iy0 + 1
            ivx1 = a_ivx0 + 1
            ivy1 = a_ivy0 + 1
            
            # Compute 4D multilinear interpolation in one vectorized expression
            # Each term is weight * V_old_safe[corner], summed over all 16 corners
            # Binary encoding: (dx, dy, dvx, dvy) where 0=low, 1=high
            # NOTE: Using V_old_safe (inf replaced with large finite) to allow propagation
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
            
            # Q(s, a) = step_cost + V_interp; mask invalid with inf
            Q_a = torch.where(
                a_valid,
                frames_per_step + V_interp,
                torch.tensor(float('inf'), device=device)
            )
            
            # Update best_V
            best_V = torch.minimum(best_V, Q_a)
            
            # Free per-action tensors
            del a_ix0, a_iy0, a_ivx0, a_ivy0, a_ijump
            del ix1, iy1, ivx1, ivy1
            del a_wx, a_wy, a_wvx, a_wvy, a_wx1, a_wy1, a_wvx1, a_wvy1
            del a_valid, V_interp, Q_a
        
        # Bellman update: V(s) = min_a Q(s, a) (already computed in best_V)
        # CRITICAL FIX: Only update if new value is actually an IMPROVEMENT
        # Without this, self-loops cause V to increase (5 + V_old > V_old)
        # and INF_REPLACEMENT values pollute the entire state space
        improved = best_V < V_old
        
        # Preserve goal states and invalid positions, only update if improved
        if invalid_5d is not None:
            V = torch.where(goal_mask | invalid_5d | ~improved, V_old, best_V)
        else:
            V = torch.where(goal_mask | ~improved, V_old, best_V)
        
        # Convergence check
        changed = V != V_old
        if changed.any():
            diff = torch.abs(V[changed] - V_old[changed])
            max_change = float(diff.max())
        else:
            max_change = 0.0
        
        if max_change < tolerance:
            logger.info(
                f"Converged after {iteration + 1} iterations "
                f"(max_change={max_change:.6f})"
            )
            break
        
        if verbose and (iteration + 1) % 10 == 0:
            reachable = int((V < float('inf')).sum())
            updates = int((changed & ~goal_mask).sum())
            logger.info(
                f"  Iter {iteration + 1}: max_change={max_change:.4f}, "
                f"reachable={reachable:,}, updates={updates:,}"
            )
        
        # #region agent log
        if iteration < 10 or iteration % 20 == 0:
            reachable_now = int((V < float('inf')).sum())
            updates_now = int((changed & ~goal_mask).sum())
            # Value distribution analysis
            V_np_sample = V.cpu().numpy()
            finite_vals = V_np_sample[V_np_sample < float('inf')]
            val_stats = {}
            if len(finite_vals) > 0:
                val_stats = {
                    "min": float(np.min(finite_vals)),
                    "max": float(np.max(finite_vals)),
                    "median": float(np.median(finite_vals)),
                    "pct_under_100": float(np.sum(finite_vals < 100) / len(finite_vals) * 100),
                    "pct_under_1000": float(np.sum(finite_vals < 1000) / len(finite_vals) * 100),
                }
            _debug_log("gpu_sl_vi:iter", "Iteration stats", {
                "iteration": iteration,
                "reachable_states": reachable_now,
                "updates": updates_now,
                "max_change": max_change if max_change < 1e10 else "inf",
                "value_stats": val_stats
            }, "A,F")
        # #endregion
    
    if max_change >= tolerance:
        logger.warning(
            f"Did not converge after {max_iterations} iterations "
            f"(max_change={max_change:.6f})"
        )
    
    iteration_time = time.time() - iteration_start
    total_time = time.time() - start_time
    
    # Transfer result back to CPU
    V_np = V.cpu().numpy()
    
    # Summary statistics
    reachable = int(np.sum(V_np < np.inf))
    goal_states = int(np.sum(V_np == 0))
    finite_V = V_np[V_np < np.inf]
    
    logger.info("GPU Semi-Lagrangian value function computed:")
    logger.info(f"  Reachable states: {reachable:,} / {total_states:,}")
    logger.info(f"  Goal states: {goal_states:,}")
    if len(finite_V) > 0:
        logger.info(f"  Value range: [{np.min(finite_V):.1f}, {np.max(finite_V):.1f}]")
    logger.info(f"  Iteration time: {iteration_time:.2f}s")
    logger.info(f"  Total time: {total_time:.2f}s")
    
    # Clean up GPU memory
    del V, V_old, best_V, goal_mask
    if invalid_5d is not None:
        del invalid_5d
    torch.cuda.empty_cache()
    
    return V_np


def benchmark_gpu_vs_cpu(
    V_init: np.ndarray,
    continuous_table: 'ContinuousTransitionTable',
    max_iterations: int = 50,
    tolerance: float = 0.01,
    position_validity_mask: Optional[np.ndarray] = None,
    frames_per_step: int = 5,
) -> dict:
    """Benchmark GPU vs CPU Semi-Lagrangian value iteration.
    
    Runs both implementations for a limited number of iterations and
    compares timing and results.
    
    Args:
        V_init: Initial value function
        continuous_table: Precomputed transition table
        max_iterations: Iterations to run (default 50 for quick benchmark)
        tolerance: Convergence tolerance
        position_validity_mask: Optional validity mask
        frames_per_step: Physics frames per step
    
    Returns:
        Dictionary with benchmark results including times and speedup.
    """
    from .continuous_transition_table import sl_value_iteration_with_table
    
    results = {
        'cpu_available': True,
        'gpu_available': is_gpu_available(),
    }
    
    # CPU benchmark
    logger.info("Running CPU benchmark...")
    cpu_start = time.time()
    V_cpu = sl_value_iteration_with_table(
        V_init=V_init.copy(),
        continuous_table=continuous_table,
        max_iterations=max_iterations,
        tolerance=tolerance,
        verbose=False,
        position_validity_mask=position_validity_mask,
        frames_per_step=frames_per_step,
    )
    cpu_time = time.time() - cpu_start
    results['cpu_time_seconds'] = cpu_time
    results['cpu_iterations'] = max_iterations
    
    # GPU benchmark (if available)
    if results['gpu_available']:
        logger.info("Running GPU benchmark...")
        gpu_start = time.time()
        V_gpu = sl_value_iteration_gpu(
            V_init=V_init.copy(),
            continuous_table=continuous_table,
            max_iterations=max_iterations,
            tolerance=tolerance,
            verbose=False,
            position_validity_mask=position_validity_mask,
            frames_per_step=frames_per_step,
        )
        gpu_time = time.time() - gpu_start
        results['gpu_time_seconds'] = gpu_time
        results['gpu_iterations'] = max_iterations
        results['speedup'] = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        
        # Check numerical agreement
        finite_mask = np.isfinite(V_cpu) & np.isfinite(V_gpu)
        if finite_mask.any():
            max_diff = np.max(np.abs(V_cpu[finite_mask] - V_gpu[finite_mask]))
            results['max_difference'] = float(max_diff)
            results['results_match'] = max_diff < 1e-3
        else:
            results['max_difference'] = 0.0
            results['results_match'] = True
        
        # GPU info
        results['gpu_info'] = get_gpu_info()
    else:
        results['speedup'] = 1.0
        results['gpu_time_seconds'] = None
    
    # Summary
    logger.info("=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"CPU time:  {cpu_time:.2f}s ({max_iterations} iterations)")
    if results['gpu_available']:
        logger.info(f"GPU time:  {gpu_time:.2f}s ({max_iterations} iterations)")
        logger.info(f"Speedup:   {results['speedup']:.1f}x")
        logger.info(f"Max diff:  {results.get('max_difference', 0):.6f}")
        logger.info(f"Results match: {results.get('results_match', False)}")
    else:
        logger.info("GPU: Not available")
    logger.info("=" * 60)
    
    return results
