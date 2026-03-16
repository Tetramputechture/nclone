"""Split filtered attract files into train/val sets by level ID, then merge
augmented outputs into combined directories ready for build_dataset_cli.

Two subcommands:

  split   -- Scan a directory of filtered attract files, split level IDs 90/10,
             and create train/ and val/ subdirectories via symlinks.

  merge   -- After running augment_cli on train/ and val/, combine the original
             symlinks and augmented .replay files into combined_train/ and
             combined_val/ so build_dataset_cli sees one flat directory.

Usage
-----
# Step 1 — split
python -m nclone.offline_rl.dataset.prepare_splits split \\
    --source /path/to/filtered_attracts \\
    --output /path/to/splits \\
    --val-fraction 0.10 \\
    --seed 42

# Step 2 — augment (external, run separately)
python -m nclone.offline_rl.dataset.augmentation.augment_cli \\
    --source /path/to/splits/train \\
    --output /path/to/splits/augmented_train \\
    --augmentations horizontal_mirror tile_rand translate \\
    --workers 8

python -m nclone.offline_rl.dataset.augmentation.augment_cli \\
    --source /path/to/splits/val \\
    --output /path/to/splits/augmented_val \\
    --augmentations horizontal_mirror tile_rand translate \\
    --workers 8

# Step 3 — merge
python -m nclone.offline_rl.dataset.prepare_splits merge \\
    --splits-dir /path/to/splits

Data-leakage guarantee
----------------------
The split is performed by base level ID (file stem) *before* augmentation.
All augmented variants of a base level inherit the same split assignment
because augmented files are written into the split-specific augmented_train/
or augmented_val/ directories.  There is no cross-split contamination as long
as augment_cli is run separately on each split directory (as shown above).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_demo_files(source_dir: Path) -> List[Path]:
    """Collect npp_attract (no extension) and .replay files."""
    files = []
    for f in sorted(source_dir.iterdir()):
        if not f.is_file() or f.name.startswith("."):
            continue
        if f.suffix == "" or f.suffix == ".replay":
            files.append(f)
    return files


def _symlink(src: Path, dst: Path, overwrite: bool = False) -> None:
    """Create a symlink dst -> src, resolving relative paths."""
    if dst.exists() or dst.is_symlink():
        if overwrite:
            dst.unlink()
        else:
            return
    # Use absolute path so symlinks remain valid regardless of cwd
    dst.symlink_to(src.resolve())


# ---------------------------------------------------------------------------
# split subcommand
# ---------------------------------------------------------------------------


def cmd_split(
    source_dir: Path,
    output_dir: Path,
    val_fraction: float = 0.10,
    seed: int = 42,
    overwrite: bool = False,
) -> Tuple[int, int, int]:
    """Split filtered attract files into train/val by level ID.

    Creates:
        output_dir/train/          -- symlinks to training files
        output_dir/val/            -- symlinks to validation files
        output_dir/train_level_ids.txt
        output_dir/val_level_ids.txt
        output_dir/split_manifest.json

    Returns:
        (n_total, n_train, n_val)
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    files = _collect_demo_files(source_dir)
    if not files:
        raise ValueError(f"No demo files found in {source_dir}")

    n_total = len(files)
    logger.info("Found %d demo files in %s", n_total, source_dir)

    # Deduplicate level IDs (file stem, since one file = one episode here)
    level_ids = sorted({f.stem for f in files})
    n_levels = len(level_ids)
    logger.info("Unique level IDs: %d", n_levels)

    # Deterministic random split
    rng = random.Random(seed)
    shuffled = list(level_ids)
    rng.shuffle(shuffled)

    n_val_levels = max(1, round(n_levels * val_fraction))
    val_ids = set(shuffled[:n_val_levels])
    train_ids = set(shuffled[n_val_levels:])

    logger.info(
        "Split: %d train levels, %d val levels (seed=%d)",
        len(train_ids), len(val_ids), seed,
    )

    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Create symlinks
    n_train = n_val = 0
    for f in files:
        if f.stem in train_ids:
            _symlink(f, train_dir / f.name, overwrite=overwrite)
            n_train += 1
        elif f.stem in val_ids:
            _symlink(f, val_dir / f.name, overwrite=overwrite)
            n_val += 1
        else:
            logger.warning("File %s not in any split (stem=%s)", f.name, f.stem)

    logger.info("Symlinks: %d train, %d val", n_train, n_val)

    # Write level ID lists
    train_ids_path = output_dir / "train_level_ids.txt"
    val_ids_path = output_dir / "val_level_ids.txt"

    with train_ids_path.open("w") as fh:
        fh.write("\n".join(sorted(train_ids)) + "\n")
    with val_ids_path.open("w") as fh:
        fh.write("\n".join(sorted(val_ids)) + "\n")

    # Write manifest
    manifest = {
        "source": str(source_dir),
        "output": str(output_dir),
        "seed": seed,
        "val_fraction": val_fraction,
        "n_total_files": n_total,
        "n_train_files": n_train,
        "n_val_files": n_val,
        "n_total_levels": n_levels,
        "n_train_levels": len(train_ids),
        "n_val_levels": len(val_ids),
    }
    manifest_path = output_dir / "split_manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)

    logger.info("Manifest written to %s", manifest_path)
    logger.info(
        "Train level IDs -> %s | Val level IDs -> %s",
        train_ids_path,
        val_ids_path,
    )

    return n_total, n_train, n_val


# ---------------------------------------------------------------------------
# merge subcommand
# ---------------------------------------------------------------------------


def cmd_merge(
    splits_dir: Path,
    overwrite: bool = False,
) -> Tuple[int, int]:
    """Merge original split files and augmented .replay files.

    Expects the following layout under splits_dir (produced by cmd_split
    followed by two augment_cli runs):

        splits_dir/train/           -- original symlinks
        splits_dir/val/             -- original symlinks
        splits_dir/augmented_train/ -- augmented .replay files
        splits_dir/augmented_val/   -- augmented .replay files

    Creates:
        splits_dir/combined_train/  -- symlinks to originals + augmented
        splits_dir/combined_val/    -- symlinks to originals + augmented

    Returns:
        (n_combined_train, n_combined_val)
    """
    train_src = splits_dir / "train"
    val_src = splits_dir / "val"
    aug_train = splits_dir / "augmented_train"
    aug_val = splits_dir / "augmented_val"

    combined_train = splits_dir / "combined_train"
    combined_val = splits_dir / "combined_val"

    combined_train.mkdir(parents=True, exist_ok=True)
    combined_val.mkdir(parents=True, exist_ok=True)

    def _populate(sources: List[Path], dest: Path) -> int:
        count = 0
        for src_dir in sources:
            if not src_dir.is_dir():
                logger.warning("Directory not found, skipping: %s", src_dir)
                continue
            for f in sorted(src_dir.iterdir()):
                if not f.is_file() or f.name.startswith("."):
                    continue
                if f.suffix in ("", ".replay"):
                    _symlink(f, dest / f.name, overwrite=overwrite)
                    count += 1
        return count

    n_train = _populate([train_src, aug_train], combined_train)
    n_val = _populate([val_src, aug_val], combined_val)

    logger.info(
        "combined_train/: %d files | combined_val/: %d files",
        n_train, n_val,
    )
    return n_train, n_val


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_split_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "split",
        help="Split filtered attract files into train/val symlink directories.",
    )
    p.add_argument(
        "--source",
        required=True,
        help="Directory containing filtered attract files (npp_attract or .replay).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Root output directory; train/ and val/ will be created inside it.",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
        metavar="F",
        help="Fraction of level IDs held out for validation (default: 0.10).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing symlinks.",
    )


def _build_merge_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "merge",
        help="Merge original split files with augmented .replay files into combined_*/ dirs.",
    )
    p.add_argument(
        "--splits-dir",
        required=True,
        help=(
            "Root splits directory containing train/, val/, "
            "augmented_train/, and augmented_val/."
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing symlinks in combined_*/ directories.",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prepare_splits",
        description="Prepare train/val splits and combined directories for HDF5 building.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _build_split_parser(sub)
    _build_merge_parser(sub)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    t0 = time.perf_counter()

    if args.command == "split":
        n_total, n_train, n_val = cmd_split(
            source_dir=Path(args.source),
            output_dir=Path(args.output),
            val_fraction=args.val_fraction,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        elapsed = time.perf_counter() - t0
        print(
            f"\nSplit complete in {elapsed:.1f}s: "
            f"{n_total} files → {n_train} train, {n_val} val"
        )

    elif args.command == "merge":
        n_train, n_val = cmd_merge(
            splits_dir=Path(args.splits_dir),
            overwrite=args.overwrite,
        )
        elapsed = time.perf_counter() - t0
        print(
            f"\nMerge complete in {elapsed:.1f}s: "
            f"combined_train/ has {n_train} files, combined_val/ has {n_val} files"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
