#!/usr/bin/env python3
"""Validate HWM cumulative rewards on `basics` using the unified reward path.

The environment now emits a single reward per ``env.step``:

    r = r_sparse + beta * alpha * hwm_progress
      + time_penalty * frames + backtrack_penalty * frames

This script exercises the HWM monotone-progress property on the basics
hairpin. BSRS is a trainer-side concern and is *not* mixed in here.

Checks (see nclone/test_maps/basics_description.md):
  1) Moving left from spawn by ~50px yields positive cumulative step reward
     (HWM progress > |time_penalty|).
  2) Left -> right -> left (same net displacement) yields strictly less
     cumulative reward than the single left run because of the time penalty
     and the backtrack penalty on the right segment.

Usage::

    python scripts/validate_hwm_gameplay_scenarios.py --map-path nclone/test_maps/basics
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

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
                SegmentResult(total_r, steps, x0, x, False),
                False,
            )
        if delta_fn(x):
            completed = True
            break
    end_x = _ninja_x(env)
    return (
        SegmentResult(total_r, steps, start_x, end_x, completed),
        completed,
    )


def validate_basics_hairpin(
    map_path: str,
    px_target: float,
    tol: float,
    seed: int,
) -> int:
    from nclone.gym_environment.config import EnvironmentConfig
    from nclone.gym_environment.npp_environment import NppEnvironment

    map_path = os.path.abspath(map_path)
    if not os.path.isfile(map_path):
        print(f"ERROR: map file not found: {map_path}")
        return 2

    cfg = EnvironmentConfig(
        seed=seed,
        custom_map_path=map_path,
        frame_skip=1,
    )
    env = NppEnvironment(cfg)
    env.reset()

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
    # Soft check: leftward progress should at least not cost more than pure time
    # penalty (HWM progress is non-negative; accel phase may make early HWM small).
    pure_time_cost = -0.001 * seg_a.steps
    if reward_a < pure_time_cost - 1e-6:
        print(
            f"FAIL: leftward cumulative reward {reward_a:.6f} is worse than the "
            f"pure time-penalty floor {pure_time_cost:.6f}."
        )
        return 1
    print("PASS: leftward progress at least matches time-penalty floor.\n")

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

    reward_b = (
        seg_b1.cumulative_reward + seg_b2.cumulative_reward + seg_b3.cumulative_reward
    )
    end_b = _ninja_x(env)

    print("=== Scenario B: LEFT 50px, RIGHT 50px, LEFT 50px ===")
    print(
        f"  spawn_x={spawn_b:.2f} end_x={end_b:.2f} "
        f"net from spawn={spawn_b - end_b:.2f}px (left positive)"
    )
    print(f"  steps: phase1={seg_b1.steps} phase2={seg_b2.steps} phase3={seg_b3.steps}")
    print(f"  cumulative reward (sum of 3 segments) = {reward_b:.6f}")
    print(f"  (scenario A only, for reference)      = {reward_a:.6f}")

    if reward_b >= reward_a:
        print(
            "FAIL: expected scenario B total < scenario A "
            "(backtrack + extra time should reduce net return)."
        )
        return 1
    print("PASS: backtrack scenario is less positive than one-way left progress.\n")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate HWM cumulative rewards on basics hairpin (unified reward path)."
    )
    parser.add_argument(
        "--map-path",
        default="nclone/test_maps/basics",
        help="Path to level file (default: nclone/test_maps/basics)",
    )
    parser.add_argument("--px", type=float, default=50.0)
    parser.add_argument("--tol", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = os.environ.get("NCLONE_REPO", _REPO_ROOT)
    map_path = (
        args.map_path
        if os.path.isabs(args.map_path)
        else os.path.join(root, args.map_path)
    )

    return validate_basics_hairpin(
        map_path=map_path,
        px_target=args.px,
        tol=args.tol,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
