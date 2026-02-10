#!/usr/bin/env python3
"""Benchmark script for GPU vs CPU Semi-Lagrangian value iteration.

This script provides a quick way to benchmark GPU acceleration on your hardware.
It creates a synthetic test case and compares GPU vs CPU performance.

Usage:
    # Quick benchmark (50 iterations)
    python -m nclone.value_function.benchmark_gpu

    # Full benchmark (200 iterations) 
    python -m nclone.value_function.benchmark_gpu --iterations 200

    # With existing transition table
    python -m nclone.value_function.benchmark_gpu --load-transitions path/to/table.npz
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from nclone.value_function import (
    StateDiscretizer,
    GridConfig,
    ContinuousTransitionTable,
    is_gpu_available,
    get_gpu_info,
    sl_value_iteration_with_table,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_transition_table(discretizer: StateDiscretizer) -> ContinuousTransitionTable:
    """Create a synthetic continuous transition table for benchmarking.
    
    This creates a simple transition table where each state transitions
    to a nearby state with some velocity-dependent offset. It's not
    physically accurate but allows testing the value iteration performance
    without requiring an actual level.
    
    Args:
        discretizer: State discretizer
    
    Returns:
        ContinuousTransitionTable with synthetic transitions
    """
    logger.info("Creating synthetic transition table for benchmarking...")
    
    shape = discretizer.config.get_shape()
    nx, ny, nvx, nvy, njump = shape
    num_actions = 6
    
    # Allocate table
    table = ContinuousTransitionTable(discretizer)
    table.transitions = np.zeros(
        shape + (num_actions, 6), dtype=np.float32
    )
    
    # Fill with synthetic transitions
    # For benchmarking, we just need valid transitions that exercise
    # the interpolation code path
    
    logger.info(f"  Grid shape: {shape}")
    logger.info(f"  Total transitions: {np.prod(shape) * num_actions:,}")
    
    # Create coordinate arrays
    ix = np.arange(nx)[:, None, None, None, None, None]
    iy = np.arange(ny)[None, :, None, None, None, None]
    ivx = np.arange(nvx)[None, None, :, None, None, None]
    ivy = np.arange(nvy)[None, None, None, :, None, None]
    ijump = np.arange(njump)[None, None, None, None, :, None]
    
    # Broadcast to full shape
    ix = np.broadcast_to(ix, shape + (num_actions,))
    iy = np.broadcast_to(iy, shape + (num_actions,))
    ivx = np.broadcast_to(ivx, shape + (num_actions,))
    ivy = np.broadcast_to(ivy, shape + (num_actions,))
    ijump = np.broadcast_to(ijump, shape + (num_actions,))
    
    # Action-dependent offsets (simplified)
    # Actions: left, left+jump, none, jump, right, right+jump
    action_dx = np.array([-0.5, -0.3, 0.0, 0.0, 0.5, 0.3])
    action_dy = np.array([0.0, -0.2, 0.0, -0.3, 0.0, -0.2])
    
    # Compute next fractional positions (with some randomness for realism)
    np.random.seed(42)
    noise = np.random.uniform(-0.1, 0.1, shape + (num_actions,)).astype(np.float32)
    
    # Next x position (fractional grid coordinate)
    fx_next = ix + action_dx + noise
    fx_next = np.clip(fx_next, 0, nx - 1.001).astype(np.float32)
    
    # Next y position
    fy_next = iy + action_dy + noise * 0.5
    fy_next = np.clip(fy_next, 0, ny - 1.001).astype(np.float32)
    
    # Velocity stays roughly the same with small drift
    fvx_next = np.clip(ivx + noise * 0.2, 0, nvx - 1.001).astype(np.float32)
    fvy_next = np.clip(ivy + noise * 0.2, 0, nvy - 1.001).astype(np.float32)
    
    # Jump index
    fjump_next = np.clip(ijump.astype(np.float32), 0, njump - 1)
    
    # Valid mask (mark edges as invalid for more realistic test)
    valid = np.ones(shape + (num_actions,), dtype=np.float32)
    valid[0, :, :, :, :, :] = 0  # Left edge
    valid[-1, :, :, :, :, :] = 0  # Right edge
    valid[:, 0, :, :, :, :] = 0  # Top edge
    valid[:, -1, :, :, :, :] = 0  # Bottom edge
    
    # Pack into transition table
    table.transitions[..., 0] = fx_next
    table.transitions[..., 1] = fy_next
    table.transitions[..., 2] = fvx_next
    table.transitions[..., 3] = fvy_next
    table.transitions[..., 4] = fjump_next
    table.transitions[..., 5] = valid
    
    valid_count = int(np.sum(valid > 0.5))
    total = int(np.prod(valid.shape))
    logger.info(f"  Valid transitions: {valid_count:,} / {total:,}")
    logger.info("  Synthetic table created")
    
    return table


def run_benchmark(
    iterations: int = 50,
    load_transitions: str = None,
    coarse_grid: bool = False,
) -> dict:
    """Run GPU vs CPU benchmark.
    
    Args:
        iterations: Number of value iteration iterations to run
        load_transitions: Optional path to existing transition table
        coarse_grid: If True, use coarser grid for faster test
    
    Returns:
        Dictionary with benchmark results
    """
    print("=" * 70)
    print("GPU vs CPU Semi-Lagrangian Value Iteration Benchmark")
    print("=" * 70)
    
    # Check GPU availability
    gpu_available = is_gpu_available()
    if gpu_available:
        gpu_info = get_gpu_info()
        print(f"\nGPU detected: {gpu_info.get('name', 'Unknown')}")
        print(f"  Memory: {gpu_info.get('memory_total_gb', 0):.1f} GB")
        print(f"  Compute capability: {gpu_info.get('compute_capability', 'Unknown')}")
    else:
        print("\n⚠️  No GPU detected. Only CPU benchmark will run.")
    
    # Create grid config
    if coarse_grid:
        print("\nUsing coarse grid for quick test...")
        config = GridConfig(dx=24.0, dy=24.0, dvx=1.0, dvy=1.0, djump=10)
    else:
        print("\nUsing default grid resolution...")
        config = GridConfig()
    
    discretizer = StateDiscretizer(config)
    shape = discretizer.config.get_shape()
    total_states = np.prod(shape)
    
    print(f"\nGrid configuration:")
    print(f"  Shape: {shape}")
    print(f"  Total states: {total_states:,}")
    print(f"  Iterations: {iterations}")
    
    # Load or create transition table
    if load_transitions and os.path.exists(load_transitions):
        print(f"\nLoading transition table from: {load_transitions}")
        table = ContinuousTransitionTable(discretizer)
        table.load(load_transitions)
    else:
        print("\nCreating synthetic transition table...")
        table = create_synthetic_transition_table(discretizer)
    
    # Create initial value function with a goal region
    print("\nInitializing value function...")
    V_init = np.full(shape, np.inf, dtype=np.float32)
    
    # Set goal region (center of level, all velocities)
    goal_ix = shape[0] // 2
    goal_iy = shape[1] // 2
    V_init[goal_ix-2:goal_ix+2, goal_iy-2:goal_iy+2, :, :, :] = 0.0
    goal_count = int(np.sum(V_init == 0))
    print(f"  Goal states: {goal_count:,}")
    
    results = {
        'grid_shape': shape,
        'total_states': total_states,
        'iterations': iterations,
        'gpu_available': gpu_available,
    }
    
    # CPU benchmark
    print("\n" + "-" * 70)
    print("Running CPU benchmark...")
    print("-" * 70)
    
    cpu_start = time.time()
    V_cpu = sl_value_iteration_with_table(
        V_init=V_init.copy(),
        continuous_table=table,
        max_iterations=iterations,
        tolerance=0.001,  # Very low tolerance to ensure all iterations run
        verbose=True,
        frames_per_step=5,
    )
    cpu_time = time.time() - cpu_start
    results['cpu_time'] = cpu_time
    results['cpu_iterations_per_second'] = iterations / cpu_time
    
    print(f"\nCPU Results:")
    print(f"  Total time: {cpu_time:.2f}s")
    print(f"  Per iteration: {cpu_time/iterations*1000:.1f}ms")
    print(f"  Reachable states: {int(np.sum(V_cpu < np.inf)):,}")
    
    # GPU benchmark
    if gpu_available:
        print("\n" + "-" * 70)
        print("Running GPU benchmark...")
        print("-" * 70)
        
        # Import GPU function (only if available)
        from nclone.value_function import sl_value_iteration_gpu
        
        # Warm-up run (first GPU call has overhead)
        print("  Warm-up run...")
        _ = sl_value_iteration_gpu(
            V_init=V_init.copy(),
            continuous_table=table,
            max_iterations=5,
            tolerance=0.001,
            verbose=False,
            frames_per_step=5,
        )
        
        # Actual benchmark
        print("  Timed run...")
        gpu_start = time.time()
        V_gpu = sl_value_iteration_gpu(
            V_init=V_init.copy(),
            continuous_table=table,
            max_iterations=iterations,
            tolerance=0.001,
            verbose=True,
            frames_per_step=5,
        )
        gpu_time = time.time() - gpu_start
        results['gpu_time'] = gpu_time
        results['gpu_iterations_per_second'] = iterations / gpu_time
        results['speedup'] = cpu_time / gpu_time
        
        print(f"\nGPU Results:")
        print(f"  Total time: {gpu_time:.2f}s")
        print(f"  Per iteration: {gpu_time/iterations*1000:.1f}ms")
        print(f"  Reachable states: {int(np.sum(V_gpu < np.inf)):,}")
        
        # Check numerical agreement
        finite_mask = np.isfinite(V_cpu) & np.isfinite(V_gpu)
        if finite_mask.any():
            max_diff = np.max(np.abs(V_cpu[finite_mask] - V_gpu[finite_mask]))
            results['max_difference'] = float(max_diff)
            results['results_match'] = max_diff < 0.1
        
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"\nGrid: {shape} ({total_states:,} states)")
    print(f"Iterations: {iterations}")
    print(f"\nCPU time:  {cpu_time:.2f}s ({cpu_time/iterations*1000:.1f}ms/iter)")
    
    if gpu_available:
        print(f"GPU time:  {gpu_time:.2f}s ({gpu_time/iterations*1000:.1f}ms/iter)")
        print(f"\n🚀 SPEEDUP: {results['speedup']:.1f}x")
        if 'max_difference' in results:
            print(f"Max numerical difference: {results['max_difference']:.6f}")
            print(f"Results match: {'✓ Yes' if results['results_match'] else '✗ No'}")
        
        # Project to full run
        full_iters = 500
        est_cpu = cpu_time / iterations * full_iters
        est_gpu = gpu_time / iterations * full_iters
        print(f"\nEstimated time for {full_iters} iterations:")
        print(f"  CPU: {est_cpu/60:.1f} minutes")
        print(f"  GPU: {est_gpu/60:.1f} minutes")
    else:
        print("\nGPU: Not available")
    
    print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GPU vs CPU Semi-Lagrangian value iteration"
    )
    parser.add_argument(
        "--iterations", type=int, default=50,
        help="Number of iterations to run (default: 50)"
    )
    parser.add_argument(
        "--load-transitions", type=str, default=None,
        help="Path to existing transition table (.npz)"
    )
    parser.add_argument(
        "--coarse-grid", action="store_true",
        help="Use coarser grid for faster benchmark"
    )
    
    args = parser.parse_args()
    
    run_benchmark(
        iterations=args.iterations,
        load_transitions=args.load_transitions,
        coarse_grid=args.coarse_grid,
    )


if __name__ == "__main__":
    main()
