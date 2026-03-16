"""Validate reward sequences and structural integrity of an IQL HDF5 dataset.

Loads train and/or val HDF5 files and verifies:
  1. Reward structure per episode:
       - All non-terminal frames have reward = -1.0
       - Exactly one +1.0 reward for the switch collection (done=False)
         (0 allowed as an edge case when ninja spawns on the switch)
       - Exactly one +1.0 reward at the terminal frame (done=True, win)
  2. No death episodes: no (done=True, reward=-1.0) transition
  3. Tile grid shape and dtype sanity
  4. NaN/Inf checks on all float arrays
  5. Episode length distribution stats
  6. Level count and transition count against acceptance thresholds

Usage
-----
python -m nclone.offline_rl.dataset.validate_dataset \\
    --train iql_train_dataset.h5 \\
    --val   iql_val_dataset.h5   \\
    --sample 100 \\
    --seed 0
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (matching epic acceptance criteria)
# ---------------------------------------------------------------------------

MIN_TRAIN_EPISODES = 50_000
MIN_TRAIN_LEVELS = 500
MIN_VAL_EPISODES = 5_000
MIN_VAL_LEVELS = 50

MIN_EP_FRAMES = 2      # at minimum: switch frame + win frame
MAX_EP_FRAMES = 18_000  # 5 minutes at 60 fps

SWITCH_REWARD = 1.0
WIN_REWARD = 1.0
STEP_REWARD = -1.0
REWARD_TOLERANCE = 1e-4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EpisodeCheckResult:
    ep_idx: int
    length: int
    n_switch_events: int     # frames with reward=+1.0 and done=False
    n_win_events: int        # frames with reward=+1.0 and done=True
    n_death_events: int      # frames with done=True and reward=-1.0
    n_bad_step_rewards: int  # non-terminal frames with reward != -1.0 or +1.0
    errors: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0


@dataclass
class ValidationReport:
    path: str
    n_transitions: int
    n_episodes: int
    n_levels: int
    n_sampled: int
    passed: int
    failed: int
    errors_by_type: dict
    ep_length_min: int
    ep_length_median: float
    ep_length_p95: float
    ep_length_max: int
    nan_inf_arrays: List[str]
    structural_errors: List[str]
    threshold_errors: List[str]

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"Validation report: {self.path}",
            f"{'='*60}",
            f"  Transitions:     {self.n_transitions:,}",
            f"  Episodes:        {self.n_episodes:,}",
            f"  Levels:          {self.n_levels:,}",
            f"  Sample checked:  {self.n_sampled}  ({self.passed} pass / {self.failed} fail)",
            f"  Episode lengths: min={self.ep_length_min}  "
            f"median={self.ep_length_median:.0f}  "
            f"p95={self.ep_length_p95:.0f}  "
            f"max={self.ep_length_max}",
        ]
        if self.nan_inf_arrays:
            lines.append(f"  NaN/Inf in arrays: {', '.join(self.nan_inf_arrays)}")
        if self.errors_by_type:
            lines.append("  Episode error breakdown:")
            for k, v in sorted(self.errors_by_type.items(), key=lambda x: -x[1]):
                lines.append(f"    {k}: {v}")
        if self.structural_errors:
            lines.append("  Structural errors:")
            for e in self.structural_errors:
                lines.append(f"    {e}")
        if self.threshold_errors:
            lines.append("  Threshold failures:")
            for e in self.threshold_errors:
                lines.append(f"    {e}")

        overall = (
            "PASS"
            if (
                self.failed == 0
                and not self.nan_inf_arrays
                and not self.structural_errors
                and not self.threshold_errors
            )
            else "FAIL"
        )
        lines.append(f"\n  Overall: {overall}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------


def _find_episode_boundaries(dones: np.ndarray) -> List[Tuple[int, int]]:
    """Return (start, end_inclusive) indices for each episode from the dones array."""
    done_indices = np.where(dones)[0]
    boundaries = []
    start = 0
    for end in done_indices:
        boundaries.append((int(start), int(end)))
        start = int(end) + 1
    return boundaries


def _check_episode(
    ep_idx: int,
    rewards: np.ndarray,
    dones: np.ndarray,
) -> EpisodeCheckResult:
    """Check reward structure for a single episode slice."""
    length = len(rewards)

    n_switch = 0
    n_win = 0
    n_death = 0
    n_bad_step = 0
    errors: List[str] = []

    for i in range(length):
        r = float(rewards[i])
        d = bool(dones[i])

        if d:
            # Terminal frame
            if abs(r - WIN_REWARD) < REWARD_TOLERANCE:
                n_win += 1
            elif abs(r - STEP_REWARD) < REWARD_TOLERANCE:
                n_death += 1
            else:
                errors.append(
                    f"frame {i}: terminal frame has unexpected reward={r:.3f}"
                )
        else:
            # Non-terminal frame: should be -1.0 (step) or +1.0 (switch)
            if abs(r - SWITCH_REWARD) < REWARD_TOLERANCE:
                n_switch += 1
            elif abs(r - STEP_REWARD) < REWARD_TOLERANCE:
                pass  # normal step
            else:
                n_bad_step += 1
                errors.append(
                    f"frame {i}: non-terminal frame has unexpected reward={r:.3f}"
                )

    # Validate switch count: 0 (edge case: spawn-on-switch) or 1
    if n_switch > 1:
        errors.append(f"switch events={n_switch} (expected 0 or 1)")
    # Win count must be exactly 1
    if n_win != 1:
        errors.append(f"win events={n_win} (expected exactly 1)")
    # Death events must be 0
    if n_death > 0:
        errors.append(f"death events={n_death} (expected 0, extractor should drop losses)")
    # Episode length sanity
    if length < MIN_EP_FRAMES:
        errors.append(f"length={length} < MIN_EP_FRAMES={MIN_EP_FRAMES}")
    if length > MAX_EP_FRAMES:
        errors.append(f"length={length} > MAX_EP_FRAMES={MAX_EP_FRAMES}")

    return EpisodeCheckResult(
        ep_idx=ep_idx,
        length=length,
        n_switch_events=n_switch,
        n_win_events=n_win,
        n_death_events=n_death,
        n_bad_step_rewards=n_bad_step,
        errors=errors,
    )


def _check_nan_inf(h5_file: h5py.File) -> List[str]:
    """Return list of dataset names that contain NaN or Inf values."""
    bad: List[str] = []
    float_dtype_prefixes = ("float",)
    for name in h5_file.keys():
        ds = h5_file[name]
        if not hasattr(ds, "dtype"):
            continue
        if not any(str(ds.dtype).startswith(p) for p in float_dtype_prefixes):
            continue
        try:
            arr = ds[:]
            if not np.isfinite(arr).all():
                bad.append(name)
        except Exception:
            pass
    return bad


def validate_hdf5(
    path: str,
    sample_size: int = 100,
    seed: int = 0,
    min_episodes: int = 0,
    min_levels: int = 0,
) -> ValidationReport:
    """Validate an IQL HDF5 dataset file.

    Args:
        path: Path to the HDF5 file.
        sample_size: Number of episodes to sample for reward-structure checks.
        seed: RNG seed for reproducible sampling.
        min_episodes: Minimum required episode count (0 = no check).
        min_levels: Minimum required level count (0 = no check).

    Returns:
        ValidationReport with full details.
    """
    structural_errors: List[str] = []
    threshold_errors: List[str] = []

    with h5py.File(path, "r") as f:
        # Structural checks
        required_keys = [
            "states", "next_states", "actions", "rewards", "dones",
            "level_ids", "mine_features", "next_mine_features",
            "tile_grids", "switch_overlays", "exit_overlays",
        ]
        for key in required_keys:
            if key not in f:
                structural_errors.append(f"Missing dataset: {key}")

        if structural_errors:
            # Cannot proceed without required datasets
            return ValidationReport(
                path=path,
                n_transitions=0, n_episodes=0, n_levels=0, n_sampled=0,
                passed=0, failed=0, errors_by_type={},
                ep_length_min=0, ep_length_median=0.0, ep_length_p95=0.0,
                ep_length_max=0, nan_inf_arrays=[], structural_errors=structural_errors,
                threshold_errors=[],
            )

        nan_inf_arrays = _check_nan_inf(f)

        dones = f["dones"][:]
        rewards = f["rewards"][:]
        n_transitions = int(len(dones))
        n_levels = int(f["tile_grids"].shape[0])

        # Tile grid shape check
        tile_shape = f["tile_grids"].shape
        if len(tile_shape) != 3 or tile_shape[1] != 42 or tile_shape[2] != 23:
            structural_errors.append(
                f"tile_grids shape={tile_shape} (expected (L, 42, 23))"
            )

        # Switch/exit overlay shape check
        for overlay_name in ("switch_overlays", "exit_overlays"):
            ov_shape = f[overlay_name].shape
            if len(ov_shape) != 3 or ov_shape[1] != 42 or ov_shape[2] != 23:
                structural_errors.append(
                    f"{overlay_name} shape={ov_shape} (expected (L, 42, 23))"
                )

    # Build episode boundary list
    episode_boundaries = _find_episode_boundaries(dones)
    n_episodes = len(episode_boundaries)

    if n_episodes == 0:
        structural_errors.append("No episodes found (no done=True in dones array)")
        return ValidationReport(
            path=path,
            n_transitions=n_transitions, n_episodes=0, n_levels=n_levels,
            n_sampled=0, passed=0, failed=0, errors_by_type={},
            ep_length_min=0, ep_length_median=0.0, ep_length_p95=0.0,
            ep_length_max=0, nan_inf_arrays=nan_inf_arrays,
            structural_errors=structural_errors, threshold_errors=[],
        )

    # Episode length distribution
    ep_lengths = np.array([end - start + 1 for start, end in episode_boundaries])
    ep_length_min = int(ep_lengths.min())
    ep_length_median = float(np.median(ep_lengths))
    ep_length_p95 = float(np.percentile(ep_lengths, 95))
    ep_length_max = int(ep_lengths.max())

    # Sample episodes for reward-structure checks
    rng = random.Random(seed)
    n_sample = min(sample_size, n_episodes)
    sampled_indices = rng.sample(range(n_episodes), k=n_sample)

    episode_results: List[EpisodeCheckResult] = []
    for ep_idx in sampled_indices:
        start, end = episode_boundaries[ep_idx]
        ep_rewards = rewards[start : end + 1]
        ep_dones = dones[start : end + 1]
        result = _check_episode(ep_idx, ep_rewards, ep_dones)
        episode_results.append(result)

    passed = sum(1 for r in episode_results if r.passed)
    failed = n_sample - passed

    # Collate error types
    errors_by_type: dict = {}
    for r in episode_results:
        for e in r.errors:
            # Group by message prefix (first word)
            key = e.split(":")[0].strip()
            errors_by_type[key] = errors_by_type.get(key, 0) + 1

    # Threshold checks
    if min_episodes > 0 and n_episodes < min_episodes:
        threshold_errors.append(
            f"n_episodes={n_episodes:,} < required {min_episodes:,}"
        )
    if min_levels > 0 and n_levels < min_levels:
        threshold_errors.append(
            f"n_levels={n_levels:,} < required {min_levels:,}"
        )

    return ValidationReport(
        path=path,
        n_transitions=n_transitions,
        n_episodes=n_episodes,
        n_levels=n_levels,
        n_sampled=n_sample,
        passed=passed,
        failed=failed,
        errors_by_type=errors_by_type,
        ep_length_min=ep_length_min,
        ep_length_median=ep_length_median,
        ep_length_p95=ep_length_p95,
        ep_length_max=ep_length_max,
        nan_inf_arrays=nan_inf_arrays,
        structural_errors=structural_errors,
        threshold_errors=threshold_errors,
    )


# ---------------------------------------------------------------------------
# Level ID overlap check
# ---------------------------------------------------------------------------


def check_level_overlap(train_path: str, val_path: str) -> List[str]:
    """Return list of level ID strings present in both train and val HDF5 files."""
    def _get_level_ids(path: str) -> set:
        with h5py.File(path, "r") as f:
            if "level_id_map" not in f.attrs:
                return set()
            raw = f.attrs["level_id_map"]
            if isinstance(raw, bytes):
                raw = raw.decode()
            # Stored as str(dict) by StreamingDatasetBuilder, e.g. "{'foo': 0, 'bar': 1}"
            import ast
            try:
                mapping = ast.literal_eval(raw)
            except Exception:
                return set()
            return set(mapping.keys())

    train_ids = _get_level_ids(train_path)
    val_ids = _get_level_ids(val_path)
    overlap = sorted(train_ids & val_ids)
    return overlap


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="validate_dataset",
        description="Validate IQL HDF5 dataset reward sequences and structural integrity.",
    )
    parser.add_argument(
        "--train",
        type=str,
        default=None,
        help="Path to training HDF5 file.",
    )
    parser.add_argument(
        "--val",
        type=str,
        default=None,
        help="Path to validation HDF5 file.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100,
        metavar="N",
        help="Number of episodes to sample for reward-structure checks (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for episode sampling (default: 0).",
    )
    parser.add_argument(
        "--check-overlap",
        action="store_true",
        help="Check for level ID overlap between train and val datasets.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.train is None and args.val is None:
        parser.error("Provide at least one of --train or --val.")

    any_failure = False

    if args.train:
        if not Path(args.train).exists():
            print(f"ERROR: train file not found: {args.train}", file=sys.stderr)
            return 1
        report = validate_hdf5(
            args.train,
            sample_size=args.sample,
            seed=args.seed,
            min_episodes=MIN_TRAIN_EPISODES,
            min_levels=MIN_TRAIN_LEVELS,
        )
        print(report.summary())
        if report.failed > 0 or report.structural_errors or report.threshold_errors or report.nan_inf_arrays:
            any_failure = True

    if args.val:
        if not Path(args.val).exists():
            print(f"ERROR: val file not found: {args.val}", file=sys.stderr)
            return 1
        report_val = validate_hdf5(
            args.val,
            sample_size=args.sample,
            seed=args.seed,
            min_episodes=MIN_VAL_EPISODES,
            min_levels=MIN_VAL_LEVELS,
        )
        print(report_val.summary())
        if report_val.failed > 0 or report_val.structural_errors or report_val.threshold_errors or report_val.nan_inf_arrays:
            any_failure = True

    if args.check_overlap and args.train and args.val:
        overlap = check_level_overlap(args.train, args.val)
        if overlap:
            print(f"\nERROR: {len(overlap)} overlapping level IDs between train and val:")
            for lid in overlap[:20]:
                print(f"  {lid}")
            if len(overlap) > 20:
                print(f"  ... and {len(overlap) - 20} more")
            any_failure = True
        else:
            print("\nLevel ID overlap check: PASS (0 overlapping IDs)")

    return 1 if any_failure else 0


if __name__ == "__main__":
    sys.exit(main())
