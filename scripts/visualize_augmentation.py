#!/usr/bin/env python3
"""Visually validate the horizontal-mirror augmentation.

Renders two videos and one side-by-side comparison:
  - LEFT  : original level replayed with original inputs
  - RIGHT : horizontally-mirrored level replayed with mirrored inputs

Frames are piped directly to ffmpeg's stdin as raw RGB — no PNG disk I/O,
no variable frame-rate artefacts.  Both individual and comparison videos are
produced in a single pass.

Supported source formats
------------------------
  npp_attract  : no-extension file from filtered_attracts
  .replay      : CompactReplay binary file from datasets/

Usage
-----
  # From a .replay file (fastest to try)
  python scripts/visualize_augmentation.py \\
      --source datasets/000basics/20260308_145411_basics.replay \\
      --output validation_videos/

  # From an npp_attract file
  python scripts/visualize_augmentation.py \\
      --source ../filtered_attracts/S-A-00-00 \\
      --output validation_videos/

  # Custom FPS (default 60)
  python scripts/visualize_augmentation.py \\
      --source datasets/000basics/20260308_145411_basics.replay \\
      --output validation_videos/ \\
      --fps 30
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ── path wrangling ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


from nclone.replay.npp_attract_decoder import NppAttractDecoder
from nclone.replay.gameplay_recorder import CompactReplay
from nclone.replay.replay_executor import decode_input_to_controls
from nclone.offline_rl.dataset.augmentation import AttractAugmentor, HorizontalMirror
from nclone.offline_rl.dataset.augmentation.attract_augmentor import _build_map_data


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
            # Ensure (H, W, 3) uint8
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
        # Input: raw RGB video on stdin
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        # Output: H.264 with constant frame rate
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-vsync", "cfr",
        "-movflags", "+faststart",
        str(output),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _write_and_close(proc: subprocess.Popen, frames: list[np.ndarray]) -> bool:
    """Write all *frames* to *proc* stdin and wait for completion."""
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
    """Encode *frames* (RGB H×W×3) to *output* MP4 at *fps* via ffmpeg stdin pipe."""
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    proc = _ffmpeg_writer(output, w, h, fps)
    ok = _write_and_close(proc, frames)
    if not ok:
        print(f"  ffmpeg failed writing {output.name}")
    return ok


def write_side_by_side(
    left_frames: list[np.ndarray],
    right_frames: list[np.ndarray],
    output: Path,
    fps: int,
) -> bool:
    """Write a horizontally-stacked comparison video in a single ffmpeg pass.

    Pads the shorter sequence with its last frame so both sides play to the end.
    """
    if not left_frames or not right_frames:
        return False

    # Pad the shorter sequence by repeating its last frame.
    n = max(len(left_frames), len(right_frames))
    left_pad  = left_frames  + [left_frames[-1]]  * (n - len(left_frames))
    right_pad = right_frames + [right_frames[-1]] * (n - len(right_frames))

    combined = [
        np.concatenate([l, r], axis=1)   # (H, 2W, 3)
        for l, r in zip(left_pad, right_pad)
    ]
    return write_video(combined, output, fps)


# ── source loading ────────────────────────────────────────────────────────────


def load_source(source: Path) -> tuple[bytes, list[int]]:
    """Return (map_bytes, input_sequence) for an attract or .replay file."""
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


def make_mirrored(source: Path) -> tuple[bytes, list[int]]:
    """Decode *source*, mirror horizontally, return (map_bytes, inputs)."""
    augmentor = AttractAugmentor([HorizontalMirror()])
    decoded = augmentor._decode(source)
    if decoded is None:
        raise ValueError(f"Could not decode {source}")
    aug = HorizontalMirror().augment(decoded)
    map_bytes = _build_map_data(aug)
    if map_bytes is None:
        raise ValueError("Failed to build mirrored map_data")
    return map_bytes, aug["demo_data"]


# ── main ──────────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render side-by-side original + horizontal-mirror videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source", type=Path, required=True,
                        help="Source attract (no ext) or .replay file.")
    parser.add_argument("--output", type=Path, default=Path("validation_videos"),
                        help="Output directory (default: validation_videos/).")
    parser.add_argument("--fps", type=int, default=60,
                        help="Video frame rate (default: 60).")
    parser.add_argument("--no-side-by-side", action="store_true",
                        help="Skip the combined side-by-side video.")
    args = parser.parse_args(argv)

    src = args.source.resolve()
    if not src.exists():
        print(f"Error: source not found: {src}")
        return 1

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    print(f"Source : {src}")
    print(f"Output : {out_dir}")
    print(f"FPS    : {args.fps}")
    print()

    # ── decode ───────────────────────────────────────────────────────────────
    print("Loading original …")
    orig_map, orig_inputs = load_source(src)
    print(f"  {len(orig_inputs)} input frames")

    print("Creating mirror …")
    mirr_map, mirr_inputs = make_mirrored(src)
    print(f"  {len(mirr_inputs)} input frames")

    # ── render both (in memory, no disk I/O) ─────────────────────────────────
    t0 = time.perf_counter()

    print("\nRendering original …")
    orig_frames = render_frames(orig_map, orig_inputs, label="original")

    print("Rendering mirror …")
    mirr_frames = render_frames(mirr_map, mirr_inputs, label="mirror")

    # ── encode ───────────────────────────────────────────────────────────────
    print("\nEncoding videos …")

    orig_mp4 = out_dir / f"{stem}_original.mp4"
    if write_video(orig_frames, orig_mp4, args.fps):
        print(f"  Saved: {orig_mp4}")
    else:
        orig_mp4 = None

    mirr_mp4 = out_dir / f"{stem}_hmirror.mp4"
    if write_video(mirr_frames, mirr_mp4, args.fps):
        print(f"  Saved: {mirr_mp4}")
    else:
        mirr_mp4 = None

    if not args.no_side_by_side and orig_frames and mirr_frames:
        sbs_mp4 = out_dir / f"{stem}_comparison.mp4"
        if write_side_by_side(orig_frames, mirr_frames, sbs_mp4, args.fps):
            print(f"  Saved: {sbs_mp4}")

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s  —  output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
