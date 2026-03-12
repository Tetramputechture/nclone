"""CLI for batch augmentation of filtered attract / replay files.

Usage
-----
python -m nclone.offline_rl.dataset.augmentation.augment_cli \\
    --source  ../filtered_attracts \\
    --output  ../augmented_attracts \\
    --augmentations horizontal_mirror \\
    --verify-sample 200 \\
    --workers 8

The script:
  1. Collects all demo files (npp_attract or .replay) from --source.
  2. Applies the requested augmentations in parallel using --workers processes.
  3. Writes one .replay file per (source file, augmentation) pair to --output.
  4. Optionally verifies a random sample via the Rust simulator (--verify-sample).
  5. Writes a manifest JSON to --output/augmentation_manifest.json.

Data-leakage note
-----------------
Augmented files receive a new filename (e.g. ``S-A-00-00_hmirror.replay``)
which gives them a distinct level_id in TransitionExtractor (path.stem).
If you need to guarantee that an original level and all its augments end up in
the same train/val split, filter ``build_dataset_cli.py`` by the original stem
(the part before ``_hmirror``).
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Registered augmentation names.
_AUGMENTATION_REGISTRY = {
    "horizontal_mirror": "HorizontalMirror",
    "tile_rand": "UnreachableTileRandomizer",
    "translate": "ReachableAreaTranslator",
}


# ---------------------------------------------------------------------------
# Worker helpers (must be module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------


def _worker_init() -> None:
    """Per-process initialisation: configure logging in worker."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )


def _augment_one(args: Tuple[str, str, list, bool]) -> Tuple[str, int, str]:
    """Augment a single file; returns (source_stem, n_written, error_or_empty)."""
    src_path, output_dir, aug_names, overwrite = args
    try:
        from .attract_augmentor import AttractAugmentor
        from .horizontal_mirror import HorizontalMirror
        from .unreachable_tile_randomizer import UnreachableTileRandomizer
        from .reachable_area_translator import ReachableAreaTranslator

        name_to_aug = {
            "horizontal_mirror": HorizontalMirror(),
            "tile_rand": UnreachableTileRandomizer(),
            "translate": ReachableAreaTranslator(),
        }
        augmentations = [name_to_aug[n] for n in aug_names]

        augmentor = AttractAugmentor(augmentations)
        written = augmentor.augment_file(src_path, output_dir, overwrite=overwrite)
        stem = Path(src_path).stem
        return stem, len(written), ""
    except Exception as exc:
        return Path(src_path).stem, 0, str(exc)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _collect_source_files(source_dir: Path) -> List[Path]:
    """Collect npp_attract (no extension) and .replay files."""
    files = []
    for f in sorted(source_dir.iterdir()):
        if not f.is_file() or f.name.startswith("."):
            continue
        if f.suffix == "" or f.suffix == ".replay":
            files.append(f)
    return files


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="augment_cli",
        description="Batch augmentation of filtered attract/replay demo files.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Directory containing filtered attract files (or .replay files).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for augmented .replay files.",
    )
    parser.add_argument(
        "--augmentations",
        nargs="+",
        default=["horizontal_mirror"],
        choices=list(_AUGMENTATION_REGISTRY),
        help="Augmentation(s) to apply (default: horizontal_mirror).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit the number of source files processed (useful for testing).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate output files even if they already exist.",
    )
    parser.add_argument(
        "--verify-sample",
        type=int,
        default=0,
        metavar="N",
        help=(
            "After augmentation, replay N randomly sampled output files through "
            "the Rust simulator to confirm they still succeed (0 = skip). "
            "Requires nclone_native."
        ),
    )
    parser.add_argument(
        "--verify-remove-failures",
        action="store_true",
        help="Remove augmented files that fail simulation verification.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for --verify-sample sampling (default: 0).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    if not source_dir.is_dir():
        logger.error("Source directory does not exist: %s", source_dir)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect source files.
    src_files = _collect_source_files(source_dir)
    if not src_files:
        logger.error("No demo files found in %s", source_dir)
        return 1

    if args.max_files:
        src_files = src_files[: args.max_files]

    logger.info(
        "Found %d source files in %s", len(src_files), source_dir
    )
    logger.info(
        "Augmentations: %s → %s", ", ".join(args.augmentations), output_dir
    )

    # Build work items.
    work_items = [
        (str(f), str(output_dir), args.augmentations, args.overwrite)
        for f in src_files
    ]

    # Run augmentation.
    t0 = time.perf_counter()
    total_written = 0
    total_errors = 0
    manifest: dict = {"source": str(source_dir), "augmentations": args.augmentations, "files": {}}

    workers = max(1, min(args.workers, len(work_items)))
    if workers == 1:
        results = [_augment_one(item) for item in work_items]
    else:
        with multiprocessing.Pool(
            processes=workers, initializer=_worker_init
        ) as pool:
            results = list(pool.imap_unordered(_augment_one, work_items, chunksize=16))

    for stem, n_written, error in results:
        manifest["files"][stem] = {"written": n_written, "error": error}
        total_written += n_written
        if error:
            total_errors += 1
            logger.warning("Error for %s: %s", stem, error)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Augmentation complete: %d files written, %d errors, %.1fs elapsed",
        total_written,
        total_errors,
        elapsed,
    )

    # Save manifest.
    manifest_path = output_dir / "augmentation_manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Manifest written to %s", manifest_path)

    # Verification step.
    if args.verify_sample > 0:
        from .verify_augmentation import verify_sample

        replay_files = sorted(output_dir.glob("*.replay"))
        if not replay_files:
            logger.warning("No .replay files found for verification.")
        else:
            logger.info(
                "Verifying sample of %d from %d augmented files …",
                min(args.verify_sample, len(replay_files)),
                len(replay_files),
            )
            report = verify_sample(
                replay_files,
                sample_size=args.verify_sample,
                seed=args.seed,
                remove_failures=args.verify_remove_failures,
            )
            print(report.summary())
            if report.failed > 0:
                logger.error(
                    "%d verification failures detected!", report.failed
                )
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
