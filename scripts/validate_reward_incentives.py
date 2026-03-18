#!/usr/bin/env python3
"""Validate precomputed PBRS and reward structure for FiGAR-PPO mine navigation.

Diagnoses why the agent may prefer timing out at the left wall over making
forward progress toward the goal when mines are in the path. Runs five
diagnostic checks on reward constants, discounted returns, and PBRS potential.

Usage::

    python scripts/validate_reward_incentives.py \\
        [--data value_functions/basics_surface_pbrs.npz] \\
        [--map-path test-single-level/basics] \\
        [--output validation_output/]

If --data is omitted, checks 1–3 and 5 run (no PBRS landscape sampling).
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Constants matching FigarPPO and reward pipeline
GAMMA = 0.99
TRUNCATION_FRAMES = 10_000


def _load_reward_constants() -> Dict[str, float]:
    """Load current reward constants from reward_config and reward_constants."""
    from nclone.gym_environment.reward_calculation.reward_config import RewardConfig
    from nclone.gym_environment.reward_calculation.reward_constants import (
        R_DEATH,
        R_DEATH_MINE,
        R_EXIT,
        R_SWITCH,
        TIME_PENALTY_PER_STEP,
        TRUNCATION_PENALTY,
    )

    config = RewardConfig()
    return {
        "R_DEATH": R_DEATH,
        "R_DEATH_MINE": R_DEATH_MINE,
        "R_SWITCH": R_SWITCH,
        "R_EXIT": R_EXIT,
        "TIME_PENALTY_PER_STEP": TIME_PENALTY_PER_STEP,
        "TRUNCATION_PENALTY": TRUNCATION_PENALTY,
        "time_penalty_per_step_actual": config.time_penalty_per_step,
    }


def check_1_discounted_return_table(
    gamma: float,
    time_penalty: float,
    r_death: float,
    r_death_mine: float,
    r_switch: float,
    r_exit: float,
    truncation_frames: int,
    truncation_penalty: float = 0.0,
) -> Tuple[Dict[str, float], bool]:
    """Check 1: Compute discounted returns for key scenarios.

    Returns (results_dict, fail_flag).
    FAIL if V_timeout > V_mine_early (agent rational to avoid mines).
    """
    total_mine_death = r_death + r_death_mine  # -2.5 by default

    # Discounted sum of constant per-step penalty over T steps:
    # sum_{t=0}^{T-1} gamma^t * c = c * (1 - gamma^T) / (1 - gamma)
    def discounted_time_penalty(T: int, c: float) -> float:
        if gamma >= 1.0:
            return c * T
        return c * (1.0 - gamma ** T) / (1.0 - gamma)

    T = truncation_frames
    v_timeout = discounted_time_penalty(T, time_penalty)
    # Add truncation penalty at last step (discounted by gamma^T)
    if truncation_penalty != 0.0:
        v_timeout += truncation_penalty * (gamma ** T)

    t_mine_100 = 100
    t_mine_500 = 500
    v_mine_early = discounted_time_penalty(t_mine_100, time_penalty) + total_mine_death * (
        gamma ** t_mine_100
    )
    v_mine_500 = discounted_time_penalty(t_mine_500, time_penalty) + total_mine_death * (
        gamma ** t_mine_500
    )

    # Success: switch at t=250, exit at t=500 (simplified)
    t_switch = 250
    t_exit = 500
    v_success = (
        discounted_time_penalty(t_exit, time_penalty)
        + r_switch * (gamma ** t_switch)
        + r_exit * (gamma ** t_exit)
    )

    results = {
        "V_timeout": v_timeout,
        "V_mine_early": v_mine_early,
        "V_mine_500": v_mine_500,
        "V_success": v_success,
    }

    fail = v_timeout > v_mine_early
    return results, fail


def check_2_time_penalty_discrepancy(constants: Dict[str, float]) -> Tuple[bool, str]:
    """Check 2: Verify time_penalty_per_step matches TIME_PENALTY_PER_STEP."""
    expected = constants["TIME_PENALTY_PER_STEP"]
    actual = constants["time_penalty_per_step_actual"]
    if abs(expected - actual) < 1e-9:
        return False, "PASS: time_penalty_per_step matches TIME_PENALTY_PER_STEP"
    return True, (
        f"FAIL: RewardConfig.time_penalty_per_step={actual} "
        f"!= TIME_PENALTY_PER_STEP={expected}"
    )


def check_3_truncation_terminal_signal() -> Tuple[bool, str]:
    """Check 3: Verify explicit terminal penalty exists for truncation."""
    from nclone.gym_environment.reward_calculation.reward_constants import (
        TRUNCATION_PENALTY,
    )

    if TRUNCATION_PENALTY != 0.0:
        return False, f"PASS: TRUNCATION_PENALTY = {TRUNCATION_PENALTY} (non-zero)"
    return True, (
        "FAIL: No explicit truncation terminal penalty. "
        "TRUNCATION_PENALTY must be non-zero to discourage timeout-at-wall behavior."
    )


def check_4_pbrs_potential_landscape(
    data_path: str,
    map_path: Optional[str],
    gamma: float,
    alpha: float,
) -> Tuple[Dict[str, float], Optional[bool], str]:
    """Check 4: Sample phi(s) at key positions and estimate path PBRS.

    Returns (phi_samples, fail_flag, message).
    fail_flag is None if PBRS data could not be loaded.
    """
    try:
        from nclone.value_function.two_phase_pbrs import TwoPhaseValuePBRS
    except ImportError:
        return {}, None, "SKIP: Could not import TwoPhaseValuePBRS"

    if not os.path.exists(data_path):
        return {}, None, f"SKIP: PBRS data not found: {data_path}"

    try:
        pbrs = TwoPhaseValuePBRS(data_path)
    except Exception as e:
        return {}, None, f"SKIP: Failed to load PBRS: {e}"

    def phi_at(x: float, y: float, switch_activated: bool = False) -> float:
        state = {
            "player_x": x,
            "player_y": y,
            "player_xspeed": 0.0,
            "player_yspeed": 0.0,
            "switch_activated": switch_activated,
            "contact_state": 0,
        }
        return pbrs.get_potential(state)

    # Load sample positions from npz surface points if available
    data = np.load(data_path, allow_pickle=True)
    sx = data["surface_points_x"].astype(np.float64)
    sy = data["surface_points_y"].astype(np.float64)
    stypes = data["surface_types"] if "surface_types" in data else np.zeros(len(sx))

    from nclone.value_function.surface_extraction import SURFACE_FLOOR

    floor_mask = stypes == SURFACE_FLOOR
    if floor_mask.any():
        fx = sx[floor_mask]
        fy = sy[floor_mask]
        left_x, left_y = float(np.min(fx)), float(fy[np.argmin(fx)])
        right_x, right_y = float(np.max(fx)), float(fy[np.argmax(fx)])
        mid_x, mid_y = float(np.mean(fx)), float(np.mean(fy))
    else:
        left_x, left_y = 50.0, 300.0
        right_x, right_y = 1000.0, 300.0
        mid_x, mid_y = 500.0, 300.0

    # Optional: override with map goals
    if map_path:
        try:
            from nclone.value_function.physics_step_wrapper import PhysicsStepWrapper

            pw = PhysicsStepWrapper(map_path=map_path)
            switch_pos = pw.get_goal_position("switch")
            exit_pos = pw.get_goal_position("exit")
            if switch_pos:
                right_x, right_y = switch_pos[0], switch_pos[1]
            if exit_pos:
                pass  # Could add exit sampling
        except Exception:
            pass

    phi_left = phi_at(left_x, left_y, False)
    phi_mid = phi_at(mid_x, mid_y, False)
    phi_right = phi_at(right_x, right_y, False)

    # Nominal path: left -> mid -> right (switch)
    # F_t = gamma * phi(s') - phi(s) per step
    phi_seq = [phi_left, phi_mid, phi_right]
    cumulative_pbrs = 0.0
    for i in range(len(phi_seq) - 1):
        F_t = gamma * phi_seq[i + 1] - phi_seq[i]
        cumulative_pbrs += alpha * F_t

    samples = {
        "phi_left_wall": phi_left,
        "phi_mid": phi_mid,
        "phi_switch": phi_right,
        "cumulative_path_pbrs": cumulative_pbrs,
    }

    # Compare to discounted mine-death risk at t=100: -2.5 * 0.99^100 ≈ -0.92
    mine_risk_discounted = 2.5 * (gamma ** 100)
    fail = cumulative_pbrs < mine_risk_discounted
    msg = (
        f"Cumulative path PBRS (left->switch) = {cumulative_pbrs:.4f}. "
        f"Discounted mine-death risk at t=100 = {mine_risk_discounted:.4f}. "
    )
    if fail:
        msg += "FAIL: Path PBRS weaker than mine risk — agent underincentivized to progress."
    else:
        msg += "PASS: Path PBRS exceeds mine risk."

    return samples, fail, msg


def check_5_reward_signal_sufficient_condition(
    gamma: float,
    time_penalty_actual: float,
    r_death_mine: float,
    r_death: float,
    t_mine: int = 100,
    T: Optional[int] = None,
) -> Tuple[float, str]:
    """Check 5: Maximum time_penalty (least negative) so V_timeout < V_mine_early.

    For agent to prefer attempting progress over timing out at the wall:
    V_mine_early > V_timeout. Both are negative; we need timeout to be worse.
    Solve: time_penalty < -total_mine * gamma^t_mine * (1-gamma) / (gamma^t_mine - gamma^T)
    """
    if T is None:
        T = TRUNCATION_FRAMES
    total_mine = abs(r_death + r_death_mine)
    gamma_t = gamma ** t_mine
    gamma_T = gamma ** T

    # time_penalty_max = most positive (least punitive) value that still makes
    # timeout worse than mine death
    denom = gamma_t - gamma_T
    if denom <= 0:
        time_penalty_max = -0.001
    else:
        time_penalty_max = -total_mine * gamma_t * (1.0 - gamma) / denom

    # We need time_penalty <= time_penalty_max (e.g. -0.025 or more negative)
    pass_flag = time_penalty_actual <= time_penalty_max

    return time_penalty_max, (
        f"To make timeout worse than mine death at t={t_mine}: "
        f"time_penalty_per_step <= {time_penalty_max:.6f}. "
        f"Current: {time_penalty_actual}. "
        f"{'PASS' if pass_flag else 'FAIL'}: "
        f"timeout {'is' if pass_flag else 'is NOT'} "
        "worse than early mine death."
    )


def run_all_checks(
    data_path: Optional[str],
    map_path: Optional[str],
    output_dir: str,
    gamma: float = GAMMA,
    truncation_frames: int = TRUNCATION_FRAMES,
) -> int:
    """Run all five checks and write report. Returns 1 if any FAIL, 0 otherwise."""
    constants = _load_reward_constants()
    time_penalty = constants["time_penalty_per_step_actual"]
    r_death = constants["R_DEATH"]
    r_death_mine = constants["R_DEATH_MINE"]
    r_switch = constants["R_SWITCH"]
    r_exit = constants["R_EXIT"]
    truncation_penalty = constants.get("TRUNCATION_PENALTY", 0.0)

    lines: List[str] = []
    fails = 0

    lines.append("=" * 70)
    lines.append("PBRS Reward Incentives Validation Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Constants:")
    lines.append(f"  gamma = {gamma}")
    lines.append(f"  truncation_frames = {truncation_frames}")
    lines.append(f"  TIME_PENALTY_PER_STEP = {constants['TIME_PENALTY_PER_STEP']}")
    lines.append(f"  time_penalty_per_step (actual) = {time_penalty}")
    lines.append(f"  TRUNCATION_PENALTY = {truncation_penalty}")
    lines.append(f"  R_DEATH = {r_death}, R_DEATH_MINE = {r_death_mine}")
    lines.append(f"  R_SWITCH = {r_switch}, R_EXIT = {r_exit}")
    lines.append("")

    # Check 1
    results, fail1 = check_1_discounted_return_table(
        gamma=gamma,
        time_penalty=time_penalty,
        r_death=r_death,
        r_death_mine=r_death_mine,
        r_switch=r_switch,
        r_exit=r_exit,
        truncation_frames=truncation_frames,
        truncation_penalty=truncation_penalty,
    )
    lines.append("--- Check 1: Discounted Return Comparison ---")
    lines.append("Scenario                          | Discounted Return")
    lines.append("-" * 55)
    for k, v in results.items():
        lines.append(f"{k:34} | {v:.4f}")
    if fail1:
        lines.append("FAIL: V_timeout > V_mine_early — agent rational to avoid mines.")
        fails += 1
    else:
        lines.append("PASS")
    lines.append("")

    # Check 2
    fail2, msg2 = check_2_time_penalty_discrepancy(constants)
    lines.append("--- Check 2: time_penalty_per_step Discrepancy ---")
    lines.append(msg2)
    if fail2:
        fails += 1
    lines.append("")

    # Check 3
    fail3, msg3 = check_3_truncation_terminal_signal()
    lines.append("--- Check 3: Truncation Terminal Signal ---")
    lines.append(msg3)
    if fail3:
        fails += 1
    lines.append("")

    # Check 4
    if data_path:
        samples, fail4, msg4 = check_4_pbrs_potential_landscape(
            data_path, map_path, gamma=gamma, alpha=2.0
        )
        lines.append("--- Check 4: PBRS Potential Landscape ---")
        for k, v in samples.items():
            lines.append(f"  {k}: {v:.4f}")
        lines.append(msg4)
        if fail4 is True:
            fails += 1
    else:
        lines.append("--- Check 4: PBRS Potential Landscape ---")
        lines.append("SKIP: No --data provided")
    lines.append("")

    # Check 5
    time_penalty_min, msg5 = check_5_reward_signal_sufficient_condition(
        gamma=gamma,
        time_penalty_actual=time_penalty,
        r_death_mine=r_death_mine,
        r_death=r_death,
    )
    lines.append("--- Check 5: Reward Signal Sufficient Condition ---")
    lines.append(f"  Minimum time_penalty for timeout worse than mine death: {time_penalty_min:.6f}")
    lines.append(msg5)
    if "FAIL" in msg5:
        fails += 1
    lines.append("")

    lines.append("=" * 70)
    lines.append(f"Total FAILs: {fails}")
    lines.append("")
    lines.append("Recommendations:")
    if fail1:
        lines.append("  - Check 1: Increase time_penalty magnitude or TRUNCATION_PENALTY")
    lines.append("  - Use RewardConfig._time_penalty_per_step_override for experimentation")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "reward_incentives_validation_report.txt")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nReport written to {out_path}")

    return 1 if fails > 0 else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate PBRS reward incentives for FiGAR-PPO mine navigation"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to surface_pbrs.npz (optional; enables Check 4)",
    )
    parser.add_argument(
        "--map-path",
        type=str,
        default=None,
        help="Path to level map for switch/exit positions (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_output",
        help="Output directory for report",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=GAMMA,
        help="Discount factor (default 0.99)",
    )
    parser.add_argument(
        "--truncation-frames",
        type=int,
        default=TRUNCATION_FRAMES,
        help="Episode truncation limit in frames (default 10000)",
    )
    args = parser.parse_args()

    return run_all_checks(
        data_path=args.data,
        map_path=args.map_path,
        output_dir=args.output,
        gamma=args.gamma,
        truncation_frames=args.truncation_frames,
    )


if __name__ == "__main__":
    sys.exit(main())
