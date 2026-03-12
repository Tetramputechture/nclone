"""Verification pipeline for augmented .replay files.

Loads each augmented file and replays it through the Rust nclone_native
simulator to confirm the episode ends in a win.  Any file where the replay
does not produce a win is flagged as invalid and removed (or just reported,
depending on the caller's preference).

This is the critical quality gate: it detects any coordinate-mirroring bugs
that would silently corrupt the training data.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class VerifyResult:
    """Outcome of verifying a single augmented replay file."""

    path: Path
    success: bool
    frames: int
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Single-file verification
# ---------------------------------------------------------------------------


def verify_replay_file(replay_path: Path) -> VerifyResult:
    """Replay *replay_path* through the simulator and check for a win.

    Args:
        replay_path: Path to a .replay file produced by AttractAugmentor.

    Returns:
        VerifyResult with success=True if the simulator reaches a win state.
    """
    try:
        import nclone_native
    except ImportError as exc:
        return VerifyResult(
            path=replay_path,
            success=False,
            frames=0,
            error=f"nclone_native not available: {exc}",
        )

    try:
        from ....replay.gameplay_recorder import CompactReplay
        from ....replay.replay_executor import decode_input_to_controls
        from ....constants.physics_constants import NINJA_STATE_CELEBRATING
    except ImportError as exc:
        return VerifyResult(
            path=replay_path,
            success=False,
            frames=0,
            error=f"Import error: {exc}",
        )

    try:
        data = replay_path.read_bytes()
        replay = CompactReplay.from_binary(data, episode_id=replay_path.stem)
    except Exception as exc:
        return VerifyResult(
            path=replay_path,
            success=False,
            frames=0,
            error=f"Failed to load replay: {exc}",
        )

    map_data = [int(b) for b in replay.map_data]
    try:
        sim = nclone_native.Simulator.load(map_data)
    except Exception as exc:
        return VerifyResult(
            path=replay_path,
            success=False,
            frames=0,
            error=f"Simulator.load failed: {exc}",
        )

    won = False
    frames = 0
    for frame_idx, input_byte in enumerate(replay.input_sequence):
        hor, jump = decode_input_to_controls(input_byte)
        sim.tick(hor, jump)
        frames = frame_idx + 1

        state = sim.get_state()
        if sim.has_won() or state.ninja.state == NINJA_STATE_CELEBRATING:
            won = True
            break
        if sim.has_died():
            break

    return VerifyResult(
        path=replay_path,
        success=won,
        frames=frames,
    )


# ---------------------------------------------------------------------------
# Batch verification
# ---------------------------------------------------------------------------


def verify_sample(
    replay_files: Sequence[Path],
    *,
    sample_size: int,
    seed: int = 0,
    remove_failures: bool = False,
) -> "VerificationReport":
    """Randomly sample *sample_size* files from *replay_files* and verify each.

    Args:
        replay_files:    Full list of candidate .replay paths.
        sample_size:     How many files to verify (0 or negative = all).
        seed:            RNG seed for reproducible sampling.
        remove_failures: If True, delete files that fail verification.

    Returns:
        VerificationReport summarising the outcome.
    """
    candidates = list(replay_files)
    if sample_size > 0 and sample_size < len(candidates):
        rng = random.Random(seed)
        candidates = rng.sample(candidates, sample_size)

    results: List[VerifyResult] = []
    for i, path in enumerate(candidates):
        result = verify_replay_file(path)
        results.append(result)
        status = "OK" if result.success else "FAIL"
        logger.info(
            "[%d/%d] %s  %s  (%d frames%s)",
            i + 1,
            len(candidates),
            status,
            path.name,
            result.frames,
            f" | {result.error}" if result.error else "",
        )
        if not result.success and remove_failures:
            try:
                path.unlink()
                logger.warning("Removed failing file: %s", path)
            except OSError as exc:
                logger.error("Could not remove %s: %s", path, exc)

    return VerificationReport(results)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class VerificationReport:
    """Aggregate statistics from a verification run."""

    results: List[VerifyResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def failures(self) -> List[VerifyResult]:
        return [r for r in self.results if not r.success]

    def summary(self) -> str:
        lines = [
            f"Verification: {self.passed}/{self.total} passed "
            f"({self.pass_rate:.1%})",
        ]
        for r in self.failures():
            lines.append(
                f"  FAIL  {r.path.name}"
                + (f"  [{r.error}]" if r.error else "")
            )
        return "\n".join(lines)
