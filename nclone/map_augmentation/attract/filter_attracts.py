"""Filter npp_attract files by entity criteria and symlink qualifying files.

Reads only the entity section (~500 bytes) from each binary attract file,
applies entity-type filtering, and symlinks qualifying files to an output
directory. Uses multiprocessing for I/O parallelism across 250K+ files.

Usage:
    python -m nclone.offline_rl.dataset.filter_attracts \
        --source /path/to/attracts \
        --output /path/to/attracts_filtered \
        --workers 16
"""

import argparse
import functools
import logging
import os
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ...constants.entity_types import (
    ENTITY_PROFILES,
    PHANTOM_ENTITY_TYPES,
    EntityType,
)

logger = logging.getLogger(__name__)

# Sanity caps from the plan's 4k-sample measurement: the subset skews entity-heavy
# (median 37 mines, max 1801) so caps drop troll/art levels and bound sequence
# length. These are a data-vs-quality knob -- tune against the expressive-range plot.
DEFAULT_MINE_CAP = 64
DEFAULT_GOLD_CAP = 40
_MINE_TYPES = frozenset({EntityType.TOGGLE_MINE, EntityType.TOGGLE_MINE_TOGGLED})

# Which entities a level may contain and still be ingestible. This used to be a
# hardcoded five-type set written before `nclone-native` gained entity parity;
# because it was a bare literal rather than a named profile, it silently kept the
# corpus at ~6.5% reach for the whole duration of the entity RE effort. It is now
# selected by name from the authoritative profiles, and the default follows the
# simulator.
DEFAULT_ENTITY_PROFILE = "native"

#: Back-compat alias for importers that expect a module-level set. Prefer passing
#: an explicit ``profile=`` so the choice is recorded rather than assumed.
ALLOWED_ENTITY_TYPES = ENTITY_PROFILES[DEFAULT_ENTITY_PROFILE]

ENTITY_SECTION_OFFSET = 1235
# Read enough of the entity section that any level within the sanity caps is fully
# visible (cap-sum <= ~106 entities -> ~530 bytes); the generous window means
# truncation only trips for genuine troll levels, which the caps reject anyway.
ENTITY_READ_SIZE = 4096


def _parse_entity_types_from_bytes(chunk: bytes) -> List[int]:
    """Parse entity type codes from raw attract-file entity bytes.

    Entities are 5 bytes each (type, x, y, orientation, mode).
    Parsing stops when entity type 0 is encountered or bytes run out.
    """
    entity_types: List[int] = []
    i = 0
    while i + 4 < len(chunk):
        etype = chunk[i]
        if etype == 0:
            break
        entity_types.append(etype)
        i += 5
    return entity_types


def check_file(
    path: str,
    mine_cap: Optional[int] = None,
    gold_cap: Optional[int] = None,
    allowed: Optional[frozenset] = None,
) -> Tuple[str, bool, str]:
    """Check whether a single attract file meets the entity criteria.

    Criteria: entities within ``allowed`` (default: everything the native sim
    implements), exactly one exit door + one exit switch, and (optionally) at
    most ``mine_cap`` mines and ``gold_cap`` gold.

    Returns:
        (filename, passed, rejection_reason)
    """
    allowed_set = ALLOWED_ENTITY_TYPES if allowed is None else allowed
    name = os.path.basename(path)
    try:
        with open(path, "rb") as f:
            f.seek(ENTITY_SECTION_OFFSET)
            chunk = f.read(ENTITY_READ_SIZE)

        if len(chunk) < 5:
            return (name, False, "too_short")

        entity_types = _parse_entity_types_from_bytes(chunk)
        # If the entity section filled the read buffer we may have truncated the
        # tail; mine/gold caps below still hold because they reject on lower bounds.
        truncated = len(entity_types) * 5 >= len(chunk) - 4

        exit_doors = entity_types.count(EntityType.EXIT_DOOR)
        exit_switches = entity_types.count(EntityType.EXIT_SWITCH)
        # A locked/trap door writes a SECOND 5-byte record holding its switch
        # coordinates, tagged with the phantom type 7/9. No entity is ever built
        # for those, so they are not a capability question -- but leaving them in
        # rejects every door level as "disallowed type 7".
        disallowed = set(entity_types) - allowed_set - PHANTOM_ENTITY_TYPES

        if disallowed:
            return (name, False, f"disallowed_types={sorted(disallowed)}")
        if exit_doors != 1:
            return (name, False, f"exit_doors={exit_doors}")
        if exit_switches != 1:
            return (name, False, f"exit_switches={exit_switches}")

        if mine_cap is not None:
            n_mines = sum(entity_types.count(t) for t in _MINE_TYPES)
            if n_mines > mine_cap:
                return (name, False, f"mines>{mine_cap}")
        if gold_cap is not None:
            n_gold = entity_types.count(EntityType.GOLD)
            if n_gold > gold_cap:
                return (name, False, f"gold>{gold_cap}")
        # A truncated read can only hide *more* entities, so a level that already
        # passed the caps within the read window but filled the buffer is ambiguous;
        # reject conservatively to keep the clean set clean.
        if truncated and (mine_cap is not None or gold_cap is not None):
            return (name, False, "entity_section_truncated")

        return (name, True, "")

    except Exception as e:
        return (name, False, f"error: {e}")


def filter_attract_files(
    source_dir: str,
    output_dir: str,
    workers: int = 8,
    dry_run: bool = False,
    mine_cap: Optional[int] = None,
    gold_cap: Optional[int] = None,
    profile: str = DEFAULT_ENTITY_PROFILE,
) -> Dict[str, int]:
    """Filter attract files and symlink qualifying ones to output_dir.

    Args:
        source_dir: Directory containing raw attract files.
        output_dir: Directory for symlinks to qualifying files.
        workers: Number of multiprocessing workers.
        dry_run: If True, only report statistics without creating symlinks.
        mine_cap: Reject levels with more than this many (toggle + toggled) mines.
        gold_cap: Reject levels with more than this many gold pieces.
        profile: Name in :data:`ENTITY_PROFILES` -- ``"native"`` (everything the
            simulator implements) or ``"minimal"`` (the legacy five-type set).

    Returns:
        Dictionary with filtering statistics.
    """
    if profile not in ENTITY_PROFILES:
        raise ValueError(
            f"unknown entity profile {profile!r}; "
            f"choose from {sorted(ENTITY_PROFILES)}"
        )
    allowed = ENTITY_PROFILES[profile]
    source_path = Path(source_dir).resolve()
    output_path = Path(output_dir).resolve()

    if not source_path.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    all_files = sorted(
        [
            str(f)
            for f in source_path.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]
    )
    total = len(all_files)
    logger.info("Found %d files in %s", total, source_dir)

    if total == 0:
        return {"total": 0, "passed": 0, "rejected": 0}

    t0 = time.time()

    # Parallel entity-byte checking
    logger.info(
        "Checking entity criteria with %d workers "
        "(profile=%s [%d types] mine_cap=%s gold_cap=%s)...",
        workers,
        profile,
        len(allowed),
        mine_cap,
        gold_cap,
    )
    check = functools.partial(
        check_file, mine_cap=mine_cap, gold_cap=gold_cap, allowed=allowed
    )
    with Pool(processes=workers) as pool:
        results = pool.map(check, all_files, chunksize=256)

    elapsed_check = time.time() - t0
    logger.info(
        "Entity check complete in %.1fs (%.0f files/sec)",
        elapsed_check,
        total / elapsed_check,
    )

    # Tally results
    passed_files: List[str] = []
    rejection_reasons: Counter = Counter()

    for filepath, (name, passed, reason) in zip(all_files, results):
        if passed:
            passed_files.append(filepath)
        else:
            rejection_reasons[reason] += 1

    n_passed = len(passed_files)
    n_rejected = total - n_passed

    logger.info("=== Filter Results ===")
    logger.info("  Entity profile: %s (%d types)", profile, len(allowed))
    logger.info("  Total files:    %d", total)
    logger.info("  Passed:         %d (%.1f%%)", n_passed, 100 * n_passed / total)
    logger.info("  Rejected:       %d (%.1f%%)", n_rejected, 100 * n_rejected / total)

    if rejection_reasons:
        logger.info("  Rejection breakdown:")
        for reason, count in rejection_reasons.most_common(20):
            logger.info("    %-40s %d", reason, count)

    if dry_run:
        logger.info("Dry run -- no symlinks created.")
        return {
            "total": total,
            "passed": n_passed,
            "rejected": n_rejected,
            "profile": profile,
        }

    # Create output directory and symlinks
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Creating %d symlinks in %s ...", n_passed, output_dir)

    t1 = time.time()
    created = 0
    for filepath in passed_files:
        src = Path(filepath)
        dst = output_path / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        created += 1

    elapsed_link = time.time() - t1
    logger.info("Created %d symlinks in %.1fs", created, elapsed_link)

    return {
        "total": total,
        "passed": n_passed,
        "rejected": n_rejected,
        "profile": profile,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter npp_attract files by entity criteria"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source directory containing npp_attract files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for symlinks to qualifying files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report statistics, don't create symlinks",
    )
    parser.add_argument(
        "--mine-cap",
        type=int,
        default=None,
        help=f"Reject levels with more than N mines (suggested {DEFAULT_MINE_CAP})",
    )
    parser.add_argument(
        "--gold-cap",
        type=int,
        default=None,
        help=f"Reject levels with more than N gold (suggested {DEFAULT_GOLD_CAP})",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(ENTITY_PROFILES),
        default=DEFAULT_ENTITY_PROFILE,
        help="Entity capability profile: 'native' = every type the simulator "
        "implements (default); 'minimal' = the legacy 5-type subset, for "
        "reproducing datasets built before entity parity landed.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    stats = filter_attract_files(
        source_dir=args.source,
        output_dir=args.output,
        workers=args.workers,
        dry_run=args.dry_run,
        mine_cap=args.mine_cap,
        gold_cap=args.gold_cap,
        profile=args.profile,
    )

    print(
        f"\nDone: {stats['passed']}/{stats['total']} files passed filtering "
        f"(profile={stats['profile']})."
    )


if __name__ == "__main__":
    main()
