"""CLI tool to build an HDF5 dataset from demo files.

Supports npp_attract files (no extension) and .replay files from
test_environment recording. Both formats can coexist in one folder.

Supports parallel extraction via multiprocessing and streaming HDF5 writes
for large-scale datasets (100K+ files, 50M+ transitions) that would OOM
with the in-memory approach.

Usage:
    python -m nclone.offline_rl.dataset.build_dataset_cli \
        --demo-dir /path/to/demos \
        --output dataset.h5 \
        [--workers 8] [--max-files 1000]
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

from .transition_extractor import TransitionExtractor, EpisodeTransitions
from .replay_dataset import StreamingDatasetBuilder

logger = logging.getLogger(__name__)


def _extract_single_file(args: Tuple[str, int]) -> Optional[EpisodeTransitions]:
    """Worker function for parallel extraction.

    Each worker creates its own TransitionExtractor (and thus its own
    NppAttractDecoder and Rust Simulator instance) to avoid shared state.

    Args:
        args: Tuple of (file_path, max_episode_frames).

    Returns:
        EpisodeTransitions or None if extraction failed.
    """
    file_path, max_frames = args
    try:
        extractor = TransitionExtractor(max_episode_frames=max_frames)
        ep = extractor.extract_from_file(file_path)
        if ep is not None and len(ep.states) > 0:
            return ep
        return None
    except Exception as e:
        logger.debug("Failed to extract %s: %s", file_path, e)
        return None


def _collect_demo_files(demo_dir: str, max_files: Optional[int] = None) -> List[str]:
    """Collect and sort demo file paths from a directory.

    Includes attract files (no extension) and .replay files from
    test_environment recording.
    """
    demo_path = Path(demo_dir)
    if not demo_path.is_dir():
        raise FileNotFoundError(f"Demo directory not found: {demo_dir}")

    def _is_demo_file(f: Path) -> bool:
        if not f.is_file() or f.name.startswith("."):
            return False
        return f.suffix == "" or f.suffix == ".replay"

    all_files = sorted([str(f) for f in demo_path.iterdir() if _is_demo_file(f)])
    if max_files is not None:
        all_files = all_files[:max_files]

    return all_files


def build_dataset_parallel(
    demo_dir: str,
    output_path: str,
    workers: int = 1,
    max_files: Optional[int] = None,
    max_episode_frames: int = 18000,
    compression: str = "gzip",
) -> str:
    """Build an HDF5 dataset with parallel extraction and streaming writes.

    Args:
        demo_dir: Directory containing npp_attract demo files (no extension)
            and/or .replay files from test_environment recording.
        output_path: Output HDF5 file path.
        workers: Number of parallel extraction workers.
        max_files: Maximum files to process (None = all).
        max_episode_frames: Maximum frames per episode.
        compression: HDF5 compression algorithm.

    Returns:
        Path to the created HDF5 file.
    """
    all_files = _collect_demo_files(demo_dir, max_files)
    n_files = len(all_files)
    logger.info("Found %d files in %s", n_files, demo_dir)

    if n_files == 0:
        raise ValueError("No demo files found in directory")

    comp = None if compression == "none" else compression
    builder = StreamingDatasetBuilder(output_path, compression=comp)

    t0 = time.time()
    worker_args = [(f, max_episode_frames) for f in all_files]

    if workers <= 1:
        # Sequential extraction
        extractor = TransitionExtractor(max_episode_frames=max_episode_frames)
        for i, fpath in enumerate(all_files):
            if (i + 1) % 500 == 0 or (i + 1) == n_files:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(
                    "Processing %d/%d (%.0f files/s) | %d episodes, %d transitions",
                    i + 1, n_files, rate,
                    builder.n_episodes, builder.n_transitions,
                )
            ep = extractor.extract_from_file(fpath)
            if ep is not None and len(ep.states) > 0:
                builder.add_episode(ep)
    else:
        # Parallel extraction with multiprocessing
        from multiprocessing import Pool

        # Use imap_unordered for memory-bounded streaming:
        # only ~chunksize * workers episodes buffered at once
        chunksize = max(1, min(64, n_files // (workers * 4)))
        logger.info("Starting %d workers (chunksize=%d)...", workers, chunksize)

        completed = 0
        with Pool(processes=workers) as pool:
            for ep in pool.imap_unordered(
                _extract_single_file, worker_args, chunksize=chunksize
            ):
                completed += 1
                if ep is not None:
                    builder.add_episode(ep)

                if completed % 500 == 0 or completed == n_files:
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    logger.info(
                        "Completed %d/%d (%.0f files/s) | %d episodes, %d transitions",
                        completed, n_files, rate,
                        builder.n_episodes, builder.n_transitions,
                    )

    if builder.n_episodes == 0:
        logger.error("No episodes extracted. Check demo directory and file format.")
        builder._h5.close()
        builder._h5 = None
        Path(output_path).unlink(missing_ok=True)
        raise ValueError("No episodes extracted. Cannot create dataset.")
    result_path = builder.finalize()
    elapsed = time.time() - t0

    logger.info("=== Dataset Statistics ===")
    logger.info("  Episodes:          %d", builder.n_episodes)
    logger.info("  Unique levels:     %d", builder.n_levels)
    logger.info("  Total transitions: %d", builder.n_transitions)
    logger.info("  Workers:           %d", workers)
    logger.info("  Output:            %s", result_path)
    logger.info("  Time:              %.1fs (%.0f files/s)", elapsed, n_files / elapsed)

    return result_path


def main():
    parser = argparse.ArgumentParser(
        description="Build HDF5 dataset from npp_attract and .replay demo files"
    )
    parser.add_argument(
        "--demo-dir",
        type=str,
        required=True,
        help="Directory containing npp_attract files (no extension) and/or "
        ".replay files from test_environment recording",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="iql_dataset.h5",
        help="Output HDF5 file path (default: iql_dataset.h5)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of demo files to process (default: all)",
    )
    parser.add_argument(
        "--max-episode-frames",
        type=int,
        default=18000,
        help="Maximum frames per episode (default: 18000 = 5min at 60fps)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel extraction workers (default: 1)",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        choices=["gzip", "lzf", "none"],
        help="HDF5 compression (default: gzip)",
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

    logger.info("Building dataset from: %s", args.demo_dir)

    build_dataset_parallel(
        demo_dir=args.demo_dir,
        output_path=args.output,
        workers=args.workers,
        max_files=args.max_files,
        max_episode_frames=args.max_episode_frames,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
