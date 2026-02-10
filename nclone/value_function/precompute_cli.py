#!/usr/bin/env python3
"""Command-line tool for precomputing value functions.

This script precomputes optimal value functions V*(s) for specific levels,
enabling exact physics-aware PBRS. The precomputation uses backward dynamic
programming (value iteration) with the actual N++ physics simulation.

Usage:
    # Single level (exit goal only)
    python -m nclone.value_function.precompute_cli --map-path levels/006.map --goal exit

    # Two-phase level (switch + exit)
    python -m nclone.value_function.precompute_cli --map-path levels/006.map --goal switch
    python -m nclone.value_function.precompute_cli --map-path levels/006.map --goal exit

    # With custom grid resolution
    python -m nclone.value_function.precompute_cli --map-path levels/006.map --goal exit \\
        --dx 8 --dy 8 --dvx 0.25 --dvy 0.25

    # With transition table precomputation (much faster iteration)
    python -m nclone.value_function.precompute_cli --map-path levels/006.map --goal exit \\
        --precompute-transitions --num-workers 8

Output: value_functions/{level_name}_{goal}.npz
"""

import argparse
import logging
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from nclone.nplay_headless import NPlayHeadless
from nclone.value_function import (
    PhysicsStepWrapper,
    StateDiscretizer,
    GridConfig,
    ValueIterator,
    TransitionTable,
    ContinuousTransitionTable,
    save_value_function,
    value_iteration_with_table,
    sl_value_iteration_with_table,
    # GPU acceleration
    is_gpu_available,
    get_gpu_info,
    sl_value_iteration_gpu,
)
from nclone.value_function.position_validity import compute_position_validity_mask
from nclone.value_function.value_iteration import ACTIONS

logger = logging.getLogger(__name__)


def _probe_transitions(physics, discretizer, position_validity_mask, goal_pos, frames_per_step):
    """Probe transitions at representative positions to diagnose connectivity.
    
    Tests a sample of positions across the level with various velocities to 
    understand death rates, self-loop rates, and inter-position transition rates.
    """
    config = discretizer.config
    goal_ix = int((goal_pos[0] - config.x_min) / config.dx)
    goal_iy = int((goal_pos[1] - config.y_min) / config.dy)
    
    # Collect valid positions sorted by distance from goal
    valid_positions = []
    for ix in range(config.nx):
        for iy in range(config.ny):
            if position_validity_mask[ix, iy]:
                dist = ((ix - goal_ix)**2 + (iy - goal_iy)**2)**0.5
                valid_positions.append((ix, iy, dist))
    valid_positions.sort(key=lambda x: x[2])
    
    n = len(valid_positions)
    samples = []
    if n > 0:
        samples.append(("goal_area", valid_positions[0]))
    if n > 1:
        samples.append(("near_25pct", valid_positions[n // 4]))
    if n > 2:
        samples.append(("mid_50pct", valid_positions[n // 2]))
    if n > 3:
        samples.append(("far_75pct", valid_positions[3 * n // 4]))
    if n > 4:
        samples.append(("furthest", valid_positions[-1]))
    
    test_velocities = [
        ("zero", 0.0, 0.0, 0),
        ("walk_right", 2.0, 0.0, 0),
        ("walk_left", -2.0, 0.0, 0),
        ("max_right", 3.0, 0.0, 0),
        ("falling_slow", 0.0, 2.0, 0),
        ("falling_fast", 0.0, 5.0, 0),
        ("jumping", 0.0, -3.0, 10),
        ("diag_right", 2.0, -2.0, 5),
    ]
    
    print(f"  Probing {len(samples)} positions x {len(test_velocities)} velocities x {len(ACTIONS)} actions...")
    print(f"  frames_per_step = {frames_per_step}")
    
    all_probe_results = []
    for label, (ix, iy, dist) in samples:
        pos_results = {"label": label, "ix": ix, "iy": iy, "dist": round(dist, 1),
                       "px": round(ix * config.dx + config.x_min, 1),
                       "py": round(iy * config.dy + config.y_min, 1),
                       "velocities": []}
        
        for vel_label, vx, vy, jump_dur in test_velocities:
            state = discretizer.index_to_state((ix, iy,
                int(np.clip((vx - config.vx_min) / config.dvx, 0, config.nvx - 1)),
                int(np.clip((vy - config.vy_min) / config.dvy, 0, config.nvy - 1)),
                int(np.clip(jump_dur / config.djump, 0, config.njump - 1))))
            
            deaths = 0
            self_loops = 0
            pos_changes = 0
            valid_to_invalid = 0
            valid_total = 0
            
            for action in ACTIONS:
                try:
                    next_state, is_death = physics.step(state, action)
                    if is_death:
                        deaths += 1
                    else:
                        valid_total += 1
                        next_idx = discretizer.state_to_index(next_state)
                        if next_idx[0] == ix and next_idx[1] == iy:
                            self_loops += 1
                        elif not position_validity_mask[next_idx[0], next_idx[1]]:
                            valid_to_invalid += 1
                        else:
                            pos_changes += 1
                except Exception:
                    deaths += 1
            
            pos_results["velocities"].append({
                "vel_label": vel_label, "vx": vx, "vy": vy, "jump": jump_dur,
                "deaths": deaths, "self_loops": self_loops,
                "pos_changes": pos_changes, "valid_to_invalid": valid_to_invalid,
                "valid": valid_total, "total_actions": len(ACTIONS),
            })
        
        all_probe_results.append(pos_results)
        
        total_deaths = sum(v["deaths"] for v in pos_results["velocities"])
        total_actions = sum(v["total_actions"] for v in pos_results["velocities"])
        total_pos_changes = sum(v["pos_changes"] for v in pos_results["velocities"])
        total_self_loops = sum(v["self_loops"] for v in pos_results["velocities"])
        total_to_invalid = sum(v["valid_to_invalid"] for v in pos_results["velocities"])
        print(f"    {label:>12} ({pos_results['px']:>6.0f}, {pos_results['py']:>6.0f}) dist={dist:>6.1f}: "
              f"deaths={total_deaths}/{total_actions} "
              f"self={total_self_loops} "
              f"pos_change={total_pos_changes} "
              f"to_invalid={total_to_invalid}")
    
    print("  Probe complete.")


def print_banner(text: str, char: str = "=", width: int = 70):
    """Print a banner with centered text."""
    print(f"\n{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}\n")


def print_section(text: str, char: str = "-", width: int = 70):
    """Print a section header."""
    print(f"\n{text}")
    print(char * width)


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_level(map_path: str) -> NPlayHeadless:
    """Load level from map file path.
    
    Args:
        map_path: Path to map file (e.g., "levels/006.map", "bc_replays_tmp/train_simple_100011.map")
    
    Returns:
        Loaded headless instance
    """
    print(f"📂 Loading level: {map_path}")
    
    # Create headless instance
    headless = NPlayHeadless(
        render_mode="grayscale_array",
        enable_animation=False,
        enable_logging=False,
        enable_rendering=False,  # No rendering needed
    )
    
    # Load map file
    try:
        headless.load_map(map_path)
        headless.reset()  # Reset to ensure entities are properly initialized
    except Exception as e:
        print(f"❌ Failed to load map: {e}")
        raise
    
    print(f"✓ Level loaded successfully\n")
    
    return headless


def get_goal_position(headless: NPlayHeadless, goal_type: str) -> Tuple[float, float]:
    """Extract goal position from level.
    
    Args:
        headless: Loaded headless instance
        goal_type: "switch" or "exit"
    
    Returns:
        Goal position (x, y) in pixels
    """
    print(f"🎯 Extracting goal position ({goal_type})...")
    
    if goal_type == "switch":
        pos = headless.exit_switch_position()
        if pos is None:
            print(f"❌ Level has no switch")
            raise ValueError(f"Level has no switch")
        print(f"✓ Switch position: ({pos[0]:.1f}, {pos[1]:.1f})\n")
        return pos
    elif goal_type == "exit":
        pos = headless.exit_door_position()
        if pos is None:
            print(f"❌ Level has no exit door")
            raise ValueError(f"Level has no exit door")
        print(f"✓ Exit position: ({pos[0]:.1f}, {pos[1]:.1f})\n")
        return pos
    else:
        raise ValueError(f"Unknown goal type: {goal_type}")


def precompute_value_function(
    map_path: str,
    goal_type: str,
    output_dir: str = "value_functions",
    grid_config: Optional[GridConfig] = None,
    max_iterations: int = 500,
    tolerance: float = 0.01,
    precompute_transitions: bool = False,
    load_transitions: Optional[str] = None,
    num_workers: int = 4,
    frames_per_step: int = 5,
    verbose: bool = False,
    method: str = "discrete",
    use_gpu: bool = False,
):
    """Precompute optimal value function for a level.
    
    Args:
        map_path: Path to map file
        goal_type: "switch" or "exit"
        output_dir: Output directory for value functions
        grid_config: Custom grid configuration (None = use defaults)
        max_iterations: Maximum value iteration iterations
        tolerance: Convergence tolerance
        precompute_transitions: If True, precompute transition table first
        load_transitions: Path to existing transition table to load
        num_workers: Number of workers for transition precomputation
        frames_per_step: Physics frames per transition (default 5)
        verbose: Enable verbose logging
        method: Value iteration method - "discrete" (standard grid-snapped VI)
                or "sl" (Semi-Lagrangian VI with 4D multilinear interpolation)
        use_gpu: If True, use GPU-accelerated value iteration (requires CUDA)
    """
    start_time = time.time()
    
    # Print header
    level_name = os.path.splitext(os.path.basename(map_path))[0]
    method_label = "Semi-Lagrangian" if method == "sl" else "Discrete"
    print_banner(f"Value Function Precomputation ({method_label})", char="=")
    method_suffix = "_sl" if method == "sl" else ""
    print(f"Level:  {level_name}")
    print(f"Goal:   {goal_type}")
    print(f"Method: {method_label}")
    print(f"Output: value_functions/{level_name}_{goal_type}{method_suffix}.npz")
    
    # Load level
    print_section("Step 1: Loading Level")
    headless = load_level(map_path)
    goal_pos = get_goal_position(headless, goal_type)
    
    # Create physics wrapper
    print_section("Step 2: Initializing Physics Wrapper")
    print(f"🔧 Creating physics step wrapper (frames_per_step={frames_per_step})...")
    physics = PhysicsStepWrapper(headless.sim, frames_per_step=frames_per_step)
    print("✓ Physics wrapper ready\n")
    
    # Create discretizer
    print_section("Step 3: Configuring State Grid")
    if grid_config is None:
        grid_config = GridConfig()
    discretizer = StateDiscretizer(grid_config)
    
    print(discretizer.config.get_info_str())
    print()
    
    # Create value iterator
    print("🔧 Creating value iterator...")
    iterator = ValueIterator(
        physics=physics,
        discretizer=discretizer,
        goal_pos=goal_pos,
        goal_radius=20.0,
    )
    print("✓ Value iterator ready\n")
    
    # Compute position validity mask
    print_section("Step 3b: Computing Position Validity Mask")
    print("🔍 Checking which positions are valid (not inside geometry/mines)...")
    position_validity_mask = compute_position_validity_mask(headless.sim, discretizer)
    num_valid_positions = np.sum(position_validity_mask)
    num_invalid_positions = np.sum(~position_validity_mask)
    total_positions = position_validity_mask.size
    valid_pct = 100.0 * num_valid_positions / total_positions
    
    print(f"✓ Validity mask computed")
    print(f"  Valid positions:   {num_valid_positions:,} / {total_positions:,} ({valid_pct:.2f}%)")
    print(f"  Invalid positions: {num_invalid_positions:,} ({100.0 - valid_pct:.2f}%)")
    print()
    
    # Transition probe: test representative positions to understand transition quality
    print_section("Step 3c: Transition Probe (Diagnostic)")
    _probe_transitions(physics, discretizer, position_validity_mask, goal_pos, frames_per_step)
    print()
    
    # Compute value function
    if method == "sl":
        # --- Semi-Lagrangian VI: requires continuous transition table ---
        if not (load_transitions or precompute_transitions):
            print("❌ Semi-Lagrangian VI requires precomputed transitions.")
            print("   Use --precompute-transitions or --load-transitions.")
            raise ValueError(
                "Semi-Lagrangian VI requires precomputed continuous transitions. "
                "Use --precompute-transitions or --load-transitions."
            )

        continuous_table = ContinuousTransitionTable(discretizer)
        trans_suffix = "_continuous_transitions.npz"

        if load_transitions and os.path.exists(load_transitions):
            print_section("Step 4a: Loading Continuous Transition Table")
            print(f"📂 Loading from: {load_transitions}")
            trans_start = time.time()
            continuous_table.load(load_transitions)
            trans_time = time.time() - trans_start
            trans_size = os.path.getsize(load_transitions) / (1024 * 1024)
            print(f"✓ Continuous transition table loaded ({trans_size:.1f} MB)")
            print(f"⏱️  Load time: {format_time(trans_time)}\n")
        elif precompute_transitions:
            print_section("Step 4a: Precomputing Continuous Transition Table")
            print(f"Workers: {num_workers}")
            print(f"Storage: float32 fractional grid coordinates")
            print(f"This will compute {np.prod(discretizer.config.get_shape()) * 6:,} transitions...")
            print()

            trans_start = time.time()
            continuous_table.precompute(
                physics=physics,
                num_workers=num_workers,
                verbose=True,
                position_validity_mask=position_validity_mask,
            )
            trans_time = time.time() - trans_start

            # Save continuous transition table for reuse
            trans_path = os.path.join(
                output_dir, f"{level_name}{trans_suffix}"
            )
            os.makedirs(output_dir, exist_ok=True)
            print(f"\n💾 Saving continuous transition table to: {trans_path}")
            continuous_table.save(trans_path)
            trans_size = os.path.getsize(trans_path) / (1024 * 1024)
            print(f"✓ Continuous transition table saved ({trans_size:.1f} MB)")
            print(f"⏱️  Precomputation time: {format_time(trans_time)}\n")
        else:
            print(f"❌ Transition table not found: {load_transitions}")
            raise FileNotFoundError(
                f"Transition table not found: {load_transitions}"
            )

        print_section("Step 4b: Semi-Lagrangian Value Iteration")
        print(f"Max iterations: {max_iterations}")
        print(f"Tolerance: {tolerance}")
        print("Interpolation: 4D multilinear (16 corners per lookup)")
        
        # Check GPU availability
        if use_gpu:
            if is_gpu_available():
                gpu_info = get_gpu_info()
                print("🚀 GPU acceleration: ENABLED")
                print(f"   Device: {gpu_info.get('name', 'Unknown')}")
                print(f"   Memory: {gpu_info.get('memory_total_gb', 0):.1f} GB")
            else:
                print("⚠️  GPU requested but not available, falling back to CPU")
                use_gpu = False
        else:
            print("   GPU acceleration: disabled (use --use-gpu to enable)")
        print()

        iter_start = time.time()

        # Initialize value function with goals
        V_init = np.full(
            discretizer.config.get_shape(), np.inf, dtype=np.float32
        )
        iterator._initialize_goal_states(V_init)

        # Run Semi-Lagrangian value iteration (GPU or CPU)
        if use_gpu and is_gpu_available():
            V = sl_value_iteration_gpu(
                V_init=V_init,
                continuous_table=continuous_table,
                max_iterations=max_iterations,
                tolerance=tolerance,
                verbose=True,
                position_validity_mask=position_validity_mask,
                frames_per_step=frames_per_step,
            )
        else:
            V = sl_value_iteration_with_table(
                V_init=V_init,
                continuous_table=continuous_table,
                max_iterations=max_iterations,
                tolerance=tolerance,
                verbose=True,
                position_validity_mask=position_validity_mask,
                frames_per_step=frames_per_step,
            )
        iter_time = time.time() - iter_start
        print(f"\n⏱️  SL-VI time: {format_time(iter_time)}")

    elif load_transitions or precompute_transitions:
        # --- Discrete VI with precomputed transition table ---
        transition_table = TransitionTable(discretizer)
        
        if load_transitions and os.path.exists(load_transitions):
            # Load existing transition table
            print_section("Step 4a: Loading Existing Transition Table")
            print(f"📂 Loading from: {load_transitions}")
            trans_start = time.time()
            transition_table.load(load_transitions)
            trans_time = time.time() - trans_start
            trans_size = os.path.getsize(load_transitions) / (1024*1024)
            print(f"✓ Transition table loaded ({trans_size:.1f} MB)")
            print(f"⏱️  Load time: {format_time(trans_time)}\n")
        elif precompute_transitions:
            # Precompute new transition table
            print_section("Step 4a: Precomputing Transition Table")
            print(f"Workers: {num_workers}")
            print(f"This will compute {np.prod(discretizer.config.get_shape()) * 6:,} transitions...")
            print()
            
            trans_start = time.time()
            transition_table.precompute(
                physics=physics,
                num_workers=num_workers,
                verbose=True,
                position_validity_mask=position_validity_mask,
            )
            trans_time = time.time() - trans_start
            
            # Save transition table for reuse
            trans_path = os.path.join(output_dir, f"{level_name}_transitions.npz")
            os.makedirs(output_dir, exist_ok=True)
            print(f"\n💾 Saving transition table to: {trans_path}")
            transition_table.save(trans_path)
            trans_size = os.path.getsize(trans_path) / (1024*1024)
            print(f"✓ Transition table saved ({trans_size:.1f} MB)")
            print(f"⏱️  Precomputation time: {format_time(trans_time)}\n")
        else:
            print(f"❌ Transition table not found: {load_transitions}")
            raise FileNotFoundError(f"Transition table not found: {load_transitions}")
        
        print_section("Step 4b: Value Iteration (Accelerated)")
        print(f"Max iterations: {max_iterations}")
        print(f"Tolerance: {tolerance}")
        print()
        
        iter_start = time.time()
        
        # Initialize value function with goals/deaths
        V_init = np.full(discretizer.config.get_shape(), np.inf, dtype=np.float32)
        iterator._initialize_goal_states(V_init)
        
        # Run accelerated value iteration
        V = value_iteration_with_table(
            V_init=V_init,
            transition_table=transition_table,
            max_iterations=max_iterations,
            tolerance=tolerance,
            verbose=True,
            position_validity_mask=position_validity_mask,
            frames_per_step=frames_per_step,
        )
        iter_time = time.time() - iter_start
        print(f"\n⏱️  Value iteration time: {format_time(iter_time)}")
    else:
        # --- Direct physics VI (no precomputed transitions) ---
        print_section("Step 4: Value Iteration (Direct Physics)")
        print(f"Max iterations: {max_iterations}")
        print(f"Tolerance: {tolerance}")
        print(f"⚠️  Using direct physics (slower, but no transition table required)")
        print()
        
        iter_start = time.time()
        V = iterator.compute_value_function(
            max_iterations=max_iterations,
            tolerance=tolerance,
            verbose=True,
            position_validity_mask=position_validity_mask,
        )
        iter_time = time.time() - iter_start
        print(f"\n⏱️  Value iteration time: {format_time(iter_time)}")
    
    # Compute statistics
    print_section("Step 5: Analyzing Value Function")
    num_reachable = np.sum(np.isfinite(V))
    total_states = np.prod(V.shape)
    reachability_pct = 100.0 * num_reachable / total_states
    
    finite_values = V[np.isfinite(V)]
    min_val = np.min(finite_values) if len(finite_values) > 0 else 0.0
    max_val = np.max(finite_values) if len(finite_values) > 0 else 0.0
    mean_val = np.mean(finite_values) if len(finite_values) > 0 else 0.0
    
    print(f"Total states:        {total_states:,}")
    print(f"Reachable states:    {num_reachable:,} ({reachability_pct:.2f}%)")
    print(f"Min cost-to-go:      {min_val:.2f} frames")
    print(f"Max cost-to-go:      {max_val:.2f} frames")
    print(f"Mean cost-to-go:     {mean_val:.2f} frames")
    print()
    
    # Save value function
    print_section("Step 6: Saving Value Function")
    method_suffix = "_sl" if method == "sl" else ""
    output_path = os.path.join(output_dir, f"{level_name}_{goal_type}{method_suffix}.npz")
    os.makedirs(output_dir, exist_ok=True)
    print(f"💾 Saving to: {output_path}")
    save_value_function(V, discretizer, goal_pos, output_path, position_validity_mask)
    file_size = os.path.getsize(output_path) / (1024*1024)
    print(f"✓ Value function saved ({file_size:.1f} MB)\n")
    
    # Final summary
    total_time = time.time() - start_time
    print_banner("✓ Precomputation Complete", char="=")
    print(f"Output file:   {output_path}")
    print(f"File size:     {file_size:.1f} MB")
    print(f"Total time:    {format_time(total_time)}")
    print(f"Reachability:  {reachability_pct:.1f}%")
    print()
    
    return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Precompute optimal value functions for PBRS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else None
    )
    
    parser.add_argument(
        "--map-path",
        required=True,
        help="Path to map file (e.g., 'levels/006.map', 'bc_replays_tmp/train_simple_100011.map')"
    )
    parser.add_argument(
        "--goal",
        required=True,
        choices=["switch", "exit"],
        help="Goal type (switch or exit)"
    )
    parser.add_argument(
        "--output-dir",
        default="value_functions",
        help="Output directory (default: value_functions/)"
    )
    parser.add_argument(
        "--method",
        choices=["discrete", "sl"],
        default="discrete",
        help="Value iteration method: 'discrete' (standard grid-snapped VI) "
             "or 'sl' (Semi-Lagrangian with 4D multilinear interpolation). "
             "SL-VI requires --precompute-transitions or --load-transitions. "
             "(default: discrete)"
    )
    
    # Grid resolution options
    parser.add_argument(
        "--dx", type=float, default=12.0,
        help="Position grid resolution X (pixels, default: 12)"
    )
    parser.add_argument(
        "--dy", type=float, default=12.0,
        help="Position grid resolution Y (pixels, default: 12)"
    )
    parser.add_argument(
        "--dvx", type=float, default=0.5,
        help="Velocity grid resolution X (px/frame, default: 0.5)"
    )
    parser.add_argument(
        "--dvy", type=float, default=0.5,
        help="Velocity grid resolution Y (px/frame, default: 0.5)"
    )
    parser.add_argument(
        "--djump", type=int, default=5,
        help="Jump duration grid resolution (frames, default: 5)"
    )
    
    # Value iteration options
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=500,
        help="Maximum value iteration iterations (default: 500)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Convergence tolerance (default: 0.01)"
    )
    
    # Transition table options
    parser.add_argument(
        "--precompute-transitions",
        action="store_true",
        help="Precompute transition table for faster iteration"
    )
    parser.add_argument(
        "--load-transitions",
        type=str,
        default=None,
        help="Path to existing transition table (.npz) to load instead of recomputing"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for transition precomputation (default: 4)"
    )
    parser.add_argument(
        "--frames-per-step",
        type=int,
        default=5,
        help="Physics frames per transition step (default: 5). Higher values allow "
             "ninja to travel further per transition, needed for coarse grids."
    )
    
    # Logging
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # GPU acceleration
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU-accelerated value iteration (requires PyTorch with CUDA). "
             "Only applies to SL-VI method (--method sl). Provides 10-50x speedup."
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Create custom grid config if specified
    grid_config = GridConfig(
        dx=args.dx,
        dy=args.dy,
        dvx=args.dvx,
        dvy=args.dvy,
        djump=args.djump,
    )
    
    # Run precomputation
    try:
        precompute_value_function(
            map_path=args.map_path,
            goal_type=args.goal,
            output_dir=args.output_dir,
            grid_config=grid_config,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            precompute_transitions=args.precompute_transitions,
            load_transitions=args.load_transitions,
            num_workers=args.num_workers,
            frames_per_step=args.frames_per_step,
            verbose=args.verbose,
            method=args.method,
            use_gpu=args.use_gpu,
        )
    except Exception as e:
        logger.error(f"Precomputation failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
