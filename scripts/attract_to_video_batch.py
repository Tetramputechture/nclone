#!/usr/bin/env python3
"""Generate MP4 videos from a sample of npp_attract files.

Usage:
    python scripts/attract_to_video_batch.py --attracts-dir ../attracts --output-dir attract_videos --sample 10
    python scripts/attract_to_video_batch.py --attracts-dir ../attracts --output-dir attract_videos --sample 10 --fps 60
"""

import argparse
import random
import sys
from pathlib import Path

# Add nclone to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Batch generate videos from npp_attract files")
    parser.add_argument(
        "--attracts-dir",
        type=Path,
        default=project_root / ".." / "attracts",
        help="Directory containing npp_attract files (no extension)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "attract_videos",
        help="Output directory for MP4 files",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Number of files to sample (default: 10)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video framerate (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sampling (default: random)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Explicit list of filenames to process (overrides --sample)",
    )

    args = parser.parse_args()

    attracts_dir = args.attracts_dir.resolve()
    if not attracts_dir.exists():
        print(f"Error: attracts directory not found: {attracts_dir}")
        return 1

    all_files = [f.name for f in attracts_dir.iterdir() if f.is_file()]
    if not all_files:
        print(f"Error: no files found in {attracts_dir}")
        return 1

    if args.files:
        files_to_process = [f for f in args.files if (attracts_dir / f).exists()]
        missing = set(args.files) - set(files_to_process)
        if missing:
            print(f"Warning: files not found: {missing}")
    else:
        sample_size = min(args.sample, len(all_files))
        if args.seed is not None:
            random.seed(args.seed)
        files_to_process = random.sample(all_files, sample_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from nclone.replay.video_generator import VideoGenerator

    success = 0
    failed = 0

    for i, filename in enumerate(files_to_process):
        input_path = attracts_dir / filename
        output_path = args.output_dir / f"{filename}.mp4"
        print(f"[{i+1}/{len(files_to_process)}] {filename} -> {output_path.name}")

        # Fresh generator per file so each uses the correct map (env caches map at init)
        generator = VideoGenerator(fps=args.fps)
        if generator.generate_video_from_npp_attract(input_path, output_path):
            success += 1
            print(f"  ✓ {output_path.name}")
        else:
            failed += 1
            print(f"  ✗ Failed")

    print()
    print(f"Done: {success} succeeded, {failed} failed")
    print(f"Output: {args.output_dir}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
