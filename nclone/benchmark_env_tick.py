"""
Benchmark end-to-end environment tick: Native Rust vs Python wrapper.

Compares full step performance (physics + observation + reward) between:
- Single env: NppEnvironment + NPlayHeadless vs NativePlayHeadless
- VecEnv: DummyVecEnv-style sequential vs NativeGymVecEnv (Rust batch physics)
- Physics-only: raw Python Simulator vs Rust Simulator (reference)

Usage:
    python -m nclone.benchmark_env_tick --map nclone/test_maps/simple-walk --steps 1000
    python -m nclone.benchmark_env_tick --modes single vec --num-envs 4 --output results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Preload nclone_native from build dir when available
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_NCLONE_ROOT = os.path.dirname(_SCRIPT_DIR)
_BUILD_SO = os.path.join(_NCLONE_ROOT, "nclone-native", "target", "debug", "libnclone_native.so")
if os.path.isfile(_BUILD_SO) and "nclone_native" not in sys.modules:
    import importlib.util
    spec = importlib.util.spec_from_file_location("nclone_native", _BUILD_SO)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nclone_native"] = mod
    spec.loader.exec_module(mod)

try:
    import nclone_native
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False


def load_map_data(map_path: str) -> Tuple[List[int], str]:
    """Load raw binary map data. Returns (map_data, resolved_path)."""
    if not os.path.isabs(map_path):
        # Try relative to nclone root
        candidate = os.path.join(_NCLONE_ROOT, map_path)
        if os.path.isfile(candidate):
            map_path = candidate
    if not os.path.isfile(map_path):
        raise FileNotFoundError(f"Map not found: {map_path}")
    with open(map_path, "rb") as f:
        return [int(b) for b in f.read()], os.path.abspath(map_path)


def _make_base_config(
    map_path: str,
    frame_skip: int,
    use_native: bool,
    enable_profiling: bool = False,
):
    """Create EnvironmentConfig for benchmarking."""
    from nclone.gym_environment.config import (
        EnvironmentConfig,
        RenderConfig,
        GraphConfig,
        ReachabilityConfig,
    )
    return EnvironmentConfig(
        custom_map_path=map_path,
        enable_logging=False,
        render=RenderConfig(
            render_mode="grayscale_array",
            enable_animation=False,
            enable_debug_overlay=False,
        ),
        graph=GraphConfig(debug=False),
        reachability=ReachabilityConfig(debug=False),
        frame_skip=frame_skip,
        use_native_physics=use_native,
        enable_profiling=enable_profiling,
    )


def _run_single_env_benchmark(
    map_path: str,
    frame_skip: int,
    steps: int,
    warmup: int,
    seed: int,
    use_native: bool,
) -> Dict[str, float]:
    """Benchmark single env full step (Python or Native physics)."""
    from nclone.gym_environment.environment_factory import create_visual_testing_env

    config = _make_base_config(map_path, frame_skip, use_native)
    env = create_visual_testing_env(config=config)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)

    # Warmup
    for _ in range(warmup):
        action = env.action_space.sample()
        if hasattr(action, "__iter__") and not isinstance(action, (int, np.integer)):
            action = rng.integers(0, 6)
        env.step(action)

    # Timed run
    times: List[float] = []
    for _ in range(steps):
        action = env.action_space.sample()
        if hasattr(action, "__iter__") and not isinstance(action, (int, np.integer)):
            action = rng.integers(0, 6)
        t0 = time.perf_counter()
        env.step(action)
        times.append((time.perf_counter() - t0) * 1000)  # ms

    env.close()

    arr = np.array(times)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "steps_per_sec": float(1000.0 / np.mean(arr)),
    }


def _run_vecenv_python_benchmark(
    map_path: str,
    frame_skip: int,
    num_envs: int,
    steps: int,
    warmup: int,
    seed: int,
) -> Dict[str, float]:
    """Benchmark Python VecEnv (sequential NppEnvironment instances, like DummyVecEnv)."""
    from nclone.gym_environment.environment_factory import create_visual_testing_env

    config = _make_base_config(map_path, frame_skip, use_native=False)
    envs = [create_visual_testing_env(config=config) for _ in range(num_envs)]
    rng = np.random.default_rng(seed)
    for env in envs:
        env.reset(seed=seed)

    # Warmup
    for _ in range(warmup):
        for i, env in enumerate(envs):
            action = rng.integers(0, 6)
            env.step(action)

    # Timed run: step all envs (sequential, like DummyVecEnv)
    times: List[float] = []
    for _ in range(steps):
        t0 = time.perf_counter()
        for i, env in enumerate(envs):
            action = rng.integers(0, 6)
            env.step(action)
        times.append((time.perf_counter() - t0) * 1000)

    for env in envs:
        env.close()

    arr = np.array(times)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "steps_per_sec": float(1000.0 / np.mean(arr)),
        "env_steps_per_sec": float(num_envs * 1000.0 / np.mean(arr)),
    }


def _run_vecenv_native_benchmark(
    map_data: List[int],
    map_path: str,
    frame_skip: int,
    num_envs: int,
    steps: int,
    warmup: int,
    seed: int,
) -> Dict[str, float]:
    """Benchmark NativeGymVecEnv (Rust batch physics + Python obs/reward)."""
    from nclone.gym_environment.environment_factory import create_native_vecenv

    env = create_native_vecenv(
        map_data=map_data,
        num_envs=num_envs,
        config=_make_base_config(map_path, frame_skip, use_native=True),
        frame_skip=frame_skip,
        map_path=map_path,
    )
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)

    # Warmup
    for _ in range(warmup):
        actions = np.array([rng.integers(0, 6) for _ in range(num_envs)])
        env.step(actions)

    # Timed run
    times: List[float] = []
    for _ in range(steps):
        actions = np.array([rng.integers(0, 6) for _ in range(num_envs)])
        t0 = time.perf_counter()
        env.step(actions)
        times.append((time.perf_counter() - t0) * 1000)

    env.close()

    arr = np.array(times)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "steps_per_sec": float(1000.0 / np.mean(arr)),
        "env_steps_per_sec": float(num_envs * 1000.0 / np.mean(arr)),
    }


def _run_physics_only_benchmark(
    map_data: List[int],
    steps: int,
    warmup: int,
    seed: int,
    use_native: bool,
) -> Dict[str, float]:
    """Benchmark raw physics tick only (no obs/reward)."""
    rng = np.random.default_rng(seed)
    ACTIONS = [
        (0, 0), (-1, 0), (1, 0), (0, 1), (-1, 1), (1, 1),
    ]

    if use_native:
        sim = nclone_native.Simulator.load(map_data)
    else:
        from nclone.sim_config import SimConfig
        from nclone.nsim import Simulator as PySimulator
        sc = SimConfig(basic_sim=True, enable_anim=False, log_data=False, debug=False)
        sim = PySimulator(sc)
        sim.load(map_data)

    # Warmup
    for _ in range(warmup):
        a = ACTIONS[rng.integers(0, 6)]
        sim.tick(a[0], a[1])
        if use_native:
            _ = sim.get_state()

    # Timed run: tick + get_state for native (matches vecenv); Python has no get_state
    times: List[float] = []
    for _ in range(steps):
        a = ACTIONS[rng.integers(0, 6)]
        t0 = time.perf_counter()
        sim.tick(a[0], a[1])
        if use_native:
            _ = sim.get_state()
        else:
            _ = (sim.ninja.xpos, sim.ninja.ypos)  # minimal state read for parity
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    steps_per_sec = 1000.0 / np.mean(arr)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "steps_per_sec": float(steps_per_sec),
    }


def run_benchmarks(
    map_path: str,
    steps: int = 1000,
    num_envs: int = 4,
    frame_skip: int = 4,
    warmup: int = 50,
    seed: int = 42,
    modes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run all selected benchmark modes and return results."""
    if modes is None:
        modes = ["single", "vec", "physics"]

    map_data, resolved_path = load_map_data(map_path)
    results: Dict[str, Any] = {
        "config": {
            "map": resolved_path,
            "steps": steps,
            "num_envs": num_envs,
            "frame_skip": frame_skip,
            "warmup": warmup,
            "seed": seed,
        },
        "single_env": {},
        "vec_env": {},
        "physics_only": {},
    }

    # Single env
    if "single" in modes:
        print("Running single env (Python)...")
        results["single_env"]["python"] = _run_single_env_benchmark(
            resolved_path, frame_skip, steps, warmup, seed, use_native=False
        )
        if NATIVE_AVAILABLE:
            print("Running single env (Native)...")
            results["single_env"]["native"] = _run_single_env_benchmark(
                resolved_path, frame_skip, steps, warmup, seed, use_native=True
            )
        else:
            results["single_env"]["native"] = None
            print("  (nclone_native not available, skipping native single env)")

    # VecEnv
    if "vec" in modes:
        print("Running VecEnv (Python SyncVectorEnv)...")
        results["vec_env"]["python"] = _run_vecenv_python_benchmark(
            resolved_path, frame_skip, num_envs, steps, warmup, seed
        )
        if NATIVE_AVAILABLE:
            print("Running VecEnv (Native NativeGymVecEnv)...")
            results["vec_env"]["native"] = _run_vecenv_native_benchmark(
                map_data, resolved_path, frame_skip, num_envs, steps, warmup, seed
            )
        else:
            results["vec_env"]["native"] = None
            print("  (nclone_native not available, skipping native VecEnv)")

    # Physics-only
    if "physics" in modes and NATIVE_AVAILABLE:
        print("Running physics-only (Python)...")
        results["physics_only"]["python"] = _run_physics_only_benchmark(
            map_data, steps, warmup, seed, use_native=False
        )
        print("Running physics-only (Native)...")
        results["physics_only"]["native"] = _run_physics_only_benchmark(
            map_data, steps, warmup, seed, use_native=True
        )

    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print formatted benchmark results to stdout."""
    cfg = results["config"]
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Map: {cfg['map']}  |  Steps: {cfg['steps']}  |  Frame skip: {cfg['frame_skip']}")
    print("=" * 60)

    def fmt_row(name: str, data: Optional[Dict], unit: str = "ms") -> str:
        if data is None:
            return f"  {name:20}  N/A (native not available)"
        mean = data.get("mean_ms", data.get("mean_ms", 0))
        std = data.get("std_ms", 0)
        sps = data.get("steps_per_sec", 1000 / mean if mean else 0)
        return f"  {name:20}  {mean:7.2f} ± {std:5.2f} {unit}  |  {sps:,.0f} steps/s"

    # Single env
    if results["single_env"]:
        print("\nSingle env (full step):")
        print(fmt_row("Python", results["single_env"].get("python")))
        print(fmt_row("Native", results["single_env"].get("native")))

    # VecEnv
    if results["vec_env"]:
        print("\nVecEnv (full step):")
        py = results["vec_env"].get("python")
        nat = results["vec_env"].get("native")
        if py:
            print(fmt_row("Python", py))
            if "env_steps_per_sec" in py:
                print(f"  {'':20}  env steps/s: {py['env_steps_per_sec']:,.0f}")
        if nat:
            print(fmt_row("Native", nat))
            if "env_steps_per_sec" in nat:
                print(f"  {'':20}  env steps/s: {nat['env_steps_per_sec']:,.0f}")

    # Physics-only
    if results["physics_only"]:
        print("\nPhysics-only (tick + get_state):")
        print(fmt_row("Python", results["physics_only"].get("python")))
        print(fmt_row("Native", results["physics_only"].get("native")))

    print("\n" + "=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Native vs Python environment tick performance"
    )
    parser.add_argument(
        "--map",
        type=str,
        default=os.path.join(_NCLONE_ROOT, "nclone", "test_maps", "simple-walk"),
        help="Map path (default: nclone/test_maps/simple-walk)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Steps per run (default: 1000)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="VecEnv parallelism (default: 4)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Physics frames per step (default: 4)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Warmup steps before timing (default: 50)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["single", "vec", "physics"],
        choices=["single", "vec", "physics"],
        help="Modes to run (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    results = run_benchmarks(
        map_path=args.map,
        steps=args.steps,
        num_envs=args.num_envs,
        frame_skip=args.frame_skip,
        warmup=args.warmup,
        seed=args.seed,
        modes=args.modes,
    )
    print_results(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
