#!/usr/bin/env python3
"""Run real NppEnvironment steps on `basics` and validate cumulative reward behavior.

Uses the same reward path as FiGAR + precomputed PBRS (primitive rewards per frame).

Checks (see nclone/test_maps/basics_description.md):
  1) Moving left from spawn by ~50px yields positive cumulative step reward.
  2) After that, moving right ~50px then left ~50px again yields a *less* positive
     total cumulative reward than check (1), for comparable net displacement toward
     the Phase-0 goal (left along lower corridor).

Usage::

    python scripts/validate_pbrs_gameplay_scenarios.py \\
        --map-path nclone/test_maps/basics \\
        --value-function-dir value_functions

Environment variables:
    NCLONE_REPO  Optional override for repo root (default: parent of scripts/).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Tuple

# Repo root: .../nclone
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Actions (BaseNppEnvironment._actions_to_execute)
ACTION_NOOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2


@dataclass
class SegmentResult:
    cumulative_reward: float
    steps: int
    start_x: float
    end_x: float
    completed: bool


def _ninja_x(env) -> float:
    return float(env.nplay_headless.ninja_position()[0])


def _run_until_x_delta(
    env,
    action: int,
    start_x: float,
    delta_fn: Callable[[float], bool],
    max_steps: int = 4000,
) -> Tuple[SegmentResult, bool]:
    """Step `action` until delta_fn(current_x) is True or limit / terminal."""
    total_r = 0.0
    steps = 0
    x0 = _ninja_x(env)
    completed = False
    while steps < max_steps:
        _, r, terminated, truncated, _ = env.step(action)
        total_r += float(r)
        steps += 1
        x = _ninja_x(env)
        if terminated or truncated:
            return (
                SegmentResult(
                    cumulative_reward=total_r,
                    steps=steps,
                    start_x=x0,
                    end_x=x,
                    completed=False,
                ),
                False,
            )
        if delta_fn(x):
            completed = True
            break
    end_x = _ninja_x(env)
    return (
        SegmentResult(
            cumulative_reward=total_r,
            steps=steps,
            start_x=start_x,
            end_x=end_x,
            completed=completed,
        ),
        completed,
    )


def validate_basics_hairpin(
    map_path: str,
    value_function_dir: str,
    px_target: float,
    tol: float,
    seed: int,
) -> int:
    from nclone.gym_environment.config import EnvironmentConfig, PBRSConfig
    from nclone.gym_environment.npp_environment import NppEnvironment

    map_path = os.path.abspath(map_path)
    value_function_dir = os.path.abspath(value_function_dir)

    if not os.path.isfile(map_path):
        print(f"ERROR: map file not found: {map_path}")
        return 2
    vf_file = os.path.join(
        value_function_dir, f"{os.path.basename(map_path)}_surface_pbrs.npz"
    )
    if not os.path.isfile(vf_file):
        print(f"ERROR: value function not found: {vf_file}")
        return 2

    cfg = EnvironmentConfig(
        seed=seed,
        custom_map_path=map_path,
        frame_skip=1,
        enable_visual_observations=False,
        pbrs=PBRSConfig(
            pbrs_gamma=0.99,
            use_precomputed_pbrs=True,
            value_function_dir=value_function_dir,
        ),
    )
    env = NppEnvironment(cfg)
    env.reset()

    if getattr(env, "_current_precomputed_pbrs", None) is None:
        print(
            "ERROR: precomputed PBRS did not load — check map stem matches "
            f"{os.path.basename(vf_file)}"
        )
        return 2

    # --- Scenario A: move left >= px_target from spawn ---
    spawn_x = _ninja_x(env)

    def left_goal(x: float) -> bool:
        return spawn_x - x >= px_target - tol

    seg_a, ok_a = _run_until_x_delta(env, ACTION_LEFT, spawn_x, left_goal)
    if not ok_a:
        print("FAIL: scenario A did not complete (terminated early or step limit).")
        print(f"  segment: {seg_a}")
        return 1

    reward_a = seg_a.cumulative_reward
    print("=== Scenario A: move LEFT from spawn (~50px) ===")
    print(
        f"  spawn_x={spawn_x:.2f} end_x={seg_a.end_x:.2f} "
        f"delta_x={spawn_x - seg_a.end_x:.2f}px steps={seg_a.steps}"
    )
    print(f"  cumulative reward (env step sum) = {reward_a:.6f}")
    if reward_a <= 0:
        print("FAIL: expected strictly positive cumulative reward for progress left.")
        return 1
    print("PASS: positive reward for leftward progress.\n")

    # --- Scenario B: reset, left 50, right 50, left 50 ---
    env.reset()
    spawn_b = _ninja_x(env)

    seg_b1, ok_b1 = _run_until_x_delta(
        env, ACTION_LEFT, spawn_b, lambda x: spawn_b - x >= px_target - tol
    )
    if not ok_b1:
        print("FAIL: scenario B phase 1 (left 50px).")
        return 1
    x_after_left = _ninja_x(env)

    seg_b2, ok_b2 = _run_until_x_delta(
        env,
        ACTION_RIGHT,
        x_after_left,
        lambda x: x - x_after_left >= px_target - tol,
    )
    if not ok_b2:
        print("FAIL: scenario B phase 2 (right 50px).")
        return 1
    x_after_right = _ninja_x(env)

    seg_b3, ok_b3 = _run_until_x_delta(
        env,
        ACTION_LEFT,
        x_after_right,
        lambda x: x_after_right - x >= px_target - tol,
    )
    if not ok_b3:
        print("FAIL: scenario B phase 3 (left 50px again).")
        return 1

    reward_b = seg_b1.cumulative_reward + seg_b2.cumulative_reward + seg_b3.cumulative_reward
    end_b = _ninja_x(env)

    print("=== Scenario B: LEFT 50px, RIGHT 50px, LEFT 50px ===")
    print(
        f"  spawn_x={spawn_b:.2f} end_x={end_b:.2f} "
        f"net from spawn={spawn_b - end_b:.2f}px (left positive)"
    )
    print(f"  steps: phase1={seg_b1.steps} phase2={seg_b2.steps} phase3={seg_b3.steps}")
    print(f"  cumulative reward (sum of 3 segments) = {reward_b:.6f}")
    print(f"  (scenario A only, for reference)     = {reward_a:.6f}")

    if reward_b >= reward_a:
        print(
            "FAIL: expected scenario B total < scenario A "
            "(backtracking + extra time should reduce net return)."
        )
        return 1
    print("PASS: backtrack scenario is less positive than one-way left progress.\n")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate PBRS cumulative rewards on basics hairpin (real env)."
    )
    parser.add_argument(
        "--map-path",
        default="nclone/test_maps/basics",
        help="Path to level file (default: nclone/test_maps/basics)",
    )
    parser.add_argument(
        "--value-function-dir",
        default="value_functions",
        help="Directory with {map_stem}_surface_pbrs.npz",
    )
    parser.add_argument(
        "--px",
        type=float,
        default=50.0,
        help="Target displacement in pixels per leg (default: 50)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=3.0,
        help="Tolerance on px target (physics discretization)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = os.environ.get("NCLONE_REPO", _REPO_ROOT)
    map_path = (
        args.map_path
        if os.path.isabs(args.map_path)
        else os.path.join(root, args.map_path)
    )
    vf_dir = (
        args.value_function_dir
        if os.path.isabs(args.value_function_dir)
        else os.path.join(root, args.value_function_dir)
    )

    return validate_basics_hairpin(
        map_path=map_path,
        value_function_dir=vf_dir,
        px_target=args.px,
        tol=args.tol,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
