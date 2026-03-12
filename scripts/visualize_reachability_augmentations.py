#!/usr/bin/env python3
"""Visually validate the graph-based reachability augmentations.

Renders individual and side-by-side comparison videos for:
  - ORIGINAL             : unmodified level
  - TILE_RAND            : tiles outside the reachable area randomized
  - TRANSLATE            : entire reachable region shifted to a new position

Each augmentation produces:
  {stem}_original.mp4
  {stem}_tile_rand.mp4        (if --augmentation tile_rand or all)
  {stem}_translate.mp4        (if --augmentation translate or all)
  {stem}_comparison_tile_rand.mp4    (original | tile_rand, side by side)
  {stem}_comparison_translate.mp4   (original | translated, side by side)

The --grid flag combines all three into a single 1×3 tiled video.

Supported source formats
------------------------
  npp_attract  : no-extension file from filtered_attracts/
  .replay      : CompactReplay binary file from datasets/

Usage
-----
  # Visualize both augmentations on a replay (default)
  python scripts/visualize_reachability_augmentations.py \\
      --source datasets/000basics/my_level.replay \\
      --output validation_videos/

  # Only tile randomization, with a fixed seed
  python scripts/visualize_reachability_augmentations.py \\
      --source datasets/000basics/my_level.replay \\
      --augmentation tile_rand --seed 42 \\
      --output validation_videos/

  # Only translation, 30 fps
  python scripts/visualize_reachability_augmentations.py \\
      --source datasets/000basics/my_level.replay \\
      --augmentation translate --fps 30 \\
      --output validation_videos/

  # All three in a single 1×3 grid video
  python scripts/visualize_reachability_augmentations.py \\
      --source datasets/000basics/my_level.replay \\
      --grid --output validation_videos/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from nclone.replay.npp_attract_decoder import NppAttractDecoder
from nclone.replay.gameplay_recorder import CompactReplay
from nclone.replay.replay_executor import decode_input_to_controls
from nclone.offline_rl.dataset.augmentation.attract_augmentor import (
    AttractAugmentor,
    _build_map_data,
)
from nclone.offline_rl.dataset.augmentation.unreachable_tile_randomizer import (
    UnreachableTileRandomizer,
)
from nclone.offline_rl.dataset.augmentation.reachable_area_translator import (
    ReachableAreaTranslator,
)


# ── simulation + rendering ────────────────────────────────────────────────────


def render_frames(
    map_bytes: bytes,
    input_sequence: list[int],
    *,
    label: str = "",
) -> list[np.ndarray]:
    """Replay *input_sequence* on *map_bytes*, return a list of RGB (H,W,3) frames.

    Stops at the first WIN or DEATH terminal state.
    """
    from nclone.nplay_headless import NPlayHeadless

    sim = NPlayHeadless(
        render_mode="rgb_array",
        enable_animation=True,
        enable_logging=False,
        enable_debug_overlay=False,
        enable_rendering=True,
    )
    sim.load_map_from_map_data([int(b) for b in map_bytes])

    frames: list[np.ndarray] = []
    for idx, input_byte in enumerate(input_sequence):
        hor, jump = decode_input_to_controls(input_byte)
        sim.tick(hor, jump)

        frame = sim.render()
        if frame is not None:
            if frame.ndim == 3 and frame.shape[2] == 1:
                frame = np.repeat(frame, 3, axis=2)
            elif frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=2)
            frames.append(frame.astype(np.uint8))

        won = sim.ninja_has_won()
        died = sim.sim.ninja.has_died()
        if won or died:
            outcome = "WIN" if won else "DEATH"
            print(f"  [{label}] {outcome} at frame {idx}  ({len(frames)} frames rendered)")
            break

    if not frames:
        print(f"  [{label}] WARNING: no frames rendered")
    return frames


# ── ffmpeg helpers ────────────────────────────────────────────────────────────


def _ffmpeg_writer(output: Path, width: int, height: int, fps: int) -> subprocess.Popen:
    """Open an ffmpeg process that reads raw RGB frames from stdin."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-vsync", "cfr",
        "-movflags", "+faststart",
        str(output),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _write_and_close(proc: subprocess.Popen, frames: list[np.ndarray]) -> bool:
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait()
        return proc.returncode == 0
    except BrokenPipeError:
        proc.wait()
        return False


def write_video(frames: list[np.ndarray], output: Path, fps: int) -> bool:
    """Encode *frames* to *output* MP4 via an ffmpeg stdin pipe."""
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    proc = _ffmpeg_writer(output, w, h, fps)
    ok = _write_and_close(proc, frames)
    if not ok:
        print(f"  ffmpeg failed writing {output.name}")
    return ok


def _pad_to_length(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    """Extend *frames* to length *n* by repeating the last frame."""
    if not frames or len(frames) >= n:
        return frames
    return frames + [frames[-1]] * (n - len(frames))


def write_side_by_side(
    left_frames: list[np.ndarray],
    right_frames: list[np.ndarray],
    output: Path,
    fps: int,
    *,
    left_label: str = "left",
    right_label: str = "right",
) -> bool:
    """Write a horizontally-stacked comparison video (left | right)."""
    if not left_frames or not right_frames:
        print(f"  Skipping {output.name}: missing frames for {left_label!r} or {right_label!r}")
        return False
    n = max(len(left_frames), len(right_frames))
    left_pad  = _pad_to_length(left_frames, n)
    right_pad = _pad_to_length(right_frames, n)
    combined = [np.concatenate([l, r], axis=1) for l, r in zip(left_pad, right_pad)]
    return write_video(combined, output, fps)


def write_grid(
    frame_lists: list[list[np.ndarray]],
    output: Path,
    fps: int,
) -> bool:
    """Write a 1×N horizontal grid combining all frame lists into one video."""
    frame_lists = [fl for fl in frame_lists if fl]
    if not frame_lists:
        return False
    n = max(len(fl) for fl in frame_lists)
    padded = [_pad_to_length(fl, n) for fl in frame_lists]
    combined = [
        np.concatenate([fl[i] for fl in padded], axis=1)
        for i in range(n)
    ]
    return write_video(combined, output, fps)


# ── source loading and augmentation application ───────────────────────────────


def load_source(source: Path) -> tuple[bytes, list[int]]:
    """Return (map_bytes, input_sequence) from an attract or .replay file."""
    if source.suffix == ".replay":
        replay = CompactReplay.from_binary(source.read_bytes(), episode_id=source.stem)
        if not replay.success:
            raise ValueError(f"{source} is not a successful replay")
        return bytes(replay.map_data), replay.input_sequence

    # npp_attract (no extension)
    decoder = NppAttractDecoder()
    decoded = decoder.decode_npp_attract_file(str(source))
    if not decoded.get("demo_data"):
        raise ValueError(f"No demo data found in {source}")
    return decoder.create_nclone_map(str(source)), decoded["demo_data"]


def apply_augmentation(
    source: Path,
    aug,
) -> tuple[bytes, list[int]]:
    """Decode *source*, apply *aug*, return (map_bytes, input_sequence)."""
    augmentor = AttractAugmentor([aug])
    decoded = augmentor._decode(source)
    if decoded is None:
        raise ValueError(f"Could not decode {source}")

    aug_data = aug.augment(decoded)
    map_bytes = _build_map_data(aug_data)
    if map_bytes is None:
        raise ValueError(f"_build_map_data returned None for {aug.name} augmentation")
    return map_bytes, aug_data["demo_data"]


# ── main ──────────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render videos comparing original and reachability-augmented replays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source", type=Path, required=True,
        help="Source attract (no ext) or .replay file.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("validation_videos"),
        help="Output directory (default: validation_videos/).",
    )
    parser.add_argument(
        "--augmentation",
        choices=["tile_rand", "translate", "all"],
        default="all",
        help=(
            "Which augmentation(s) to visualize. "
            "'all' (default) renders both tile_rand and translate."
        ),
    )
    parser.add_argument(
        "--fps", type=int, default=60,
        help="Video frame rate (default: 60).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed passed to the augmentations (default: random).",
    )
    parser.add_argument(
        "--no-side-by-side", action="store_true",
        help="Skip per-augmentation side-by-side comparison videos.",
    )
    parser.add_argument(
        "--grid", action="store_true",
        help=(
            "Also write a single 1×N grid video combining all variants "
            "(only meaningful with --augmentation all)."
        ),
    )
    args = parser.parse_args(argv)

    src = args.source.resolve()
    if not src.exists():
        print(f"Error: source not found: {src}")
        return 1

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    print(f"Source      : {src}")
    print(f"Output      : {out_dir}")
    print(f"Augmentation: {args.augmentation}")
    print(f"Seed        : {args.seed if args.seed is not None else 'random'}")
    print(f"FPS         : {args.fps}")
    print()

    # ── decide which augmentations to run ─────────────────────────────────────
    augs_to_run: list[tuple[str, object]] = []
    if args.augmentation in ("tile_rand", "all"):
        augs_to_run.append(("tile_rand", UnreachableTileRandomizer(seed=args.seed)))
    if args.augmentation in ("translate", "all"):
        augs_to_run.append(("translate", ReachableAreaTranslator(seed=args.seed)))

    # ── load original ──────────────────────────────────────────────────────────
    print("Loading original …")
    try:
        orig_map, orig_inputs = load_source(src)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    print(f"  {len(orig_inputs)} input frames")

    # ── apply augmentations ────────────────────────────────────────────────────
    aug_results: list[tuple[str, bytes, list[int]]] = []
    for aug_name, aug in augs_to_run:
        print(f"Applying {aug_name} …")
        try:
            aug_map, aug_inputs = apply_augmentation(src, aug)
            print(f"  {len(aug_inputs)} input frames")
            aug_results.append((aug_name, aug_map, aug_inputs))
        except ValueError as exc:
            print(f"  Warning: {exc} — skipping {aug_name}")

    if not aug_results:
        print("No augmentations produced output. Exiting.")
        return 1

    # ── render ─────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()

    print("\nRendering original …")
    orig_frames = render_frames(orig_map, orig_inputs, label="original")

    rendered: list[tuple[str, list[np.ndarray]]] = [("original", orig_frames)]

    for aug_name, aug_map, aug_inputs in aug_results:
        print(f"Rendering {aug_name} …")
        aug_frames = render_frames(aug_map, aug_inputs, label=aug_name)
        rendered.append((aug_name, aug_frames))

    # ── encode individual videos ───────────────────────────────────────────────
    print("\nEncoding individual videos …")
    for label, frames in rendered:
        out_path = out_dir / f"{stem}_{label}.mp4"
        if write_video(frames, out_path, args.fps):
            print(f"  Saved: {out_path.name}")

    # ── encode side-by-side comparisons ───────────────────────────────────────
    if not args.no_side_by_side:
        print("\nEncoding comparison videos …")
        for aug_name, aug_frames in rendered[1:]:
            out_path = out_dir / f"{stem}_comparison_{aug_name}.mp4"
            if write_side_by_side(
                orig_frames, aug_frames, out_path, args.fps,
                left_label="original", right_label=aug_name,
            ):
                print(f"  Saved: {out_path.name}")

    # ── encode grid video ──────────────────────────────────────────────────────
    if args.grid and len(rendered) > 1:
        print("\nEncoding grid video …")
        grid_path = out_dir / f"{stem}_grid.mp4"
        all_frame_lists = [frames for _, frames in rendered]
        if write_grid(all_frame_lists, grid_path, args.fps):
            print(f"  Saved: {grid_path.name}")
        else:
            print(f"  Failed to write grid video")

    elapsed = time.perf_counter() - t0

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
