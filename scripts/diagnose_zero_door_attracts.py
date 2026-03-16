#!/usr/bin/env python3
"""Generate videos for attract files reported as "zero exit doors".

Investigation found two reasons a file can appear to have zero exit doors:

  1. Window-size truncation (most common, ~20% of files): the old
     filter_attracts.py read only 500 bytes from offset 1235, which caps at
     100 entities.  Files whose exit door appears later in the entity list are
     incorrectly rejected.  Fixed by reading the full entity section using the
     map_data_length field from the 8-byte outer header.

  2. Co-op format (~1.4% of files): these levels store TWO NINJA (type 0)
     entries at offsets 1230 and 1235 before the real entity list starts at
     1240.  A parser that stops at the first type-0 entry finds zero entities.
     Fixed by skipping consecutive leading type-0 (ninja) entries.

  After applying both fixes, every attract file in the corpus has exactly one
  exit door.  This script generates videos of co-op-format files (category 2)
  so the operator can visually confirm they are valid replays.

Usage:
    python scripts/diagnose_zero_door_attracts.py \\
        --attracts-dir ../attracts \\
        --output-dir attract_zero_door_validation \\
        [--scan-limit N]   # files to scan; default 5000
        [--videos N]       # videos to produce; default 2
        [--fps N]          # default 30
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import List

# Must be set before pygame/SDL init
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np  # noqa: E402
import pygame  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from nclone.nsim import Simulator as NcloneSimulator  # noqa: E402
from nclone.sim_config import SimConfig  # noqa: E402
from nclone.nsim_renderer import NSimRenderer  # noqa: E402
from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402

ENTITY_SECTION_OFFSET = 1235
_MAP_DATA_OFFSET = 8
ENTITY_NAMES = {
    0: "NINJA", 1: "TOGGLE_MINE", 2: "GOLD", 3: "EXIT_DOOR",
    4: "EXIT_SWITCH", 5: "REGULAR_DOOR", 6: "LOCKED_DOOR",
    7: "LOCKED_DOOR_SWITCH", 8: "TRAP_DOOR", 10: "LAUNCH_PAD",
    11: "ONE_WAY", 17: "BOUNCE_BLOCK", 21: "TOGGLE_MINE_TOGGLED",
}


# ── Entity parsing ────────────────────────────────────────────────────────────

def read_full_entity_types(path: Path) -> List[int]:
    """Return all entity types in the file using the full entity section.

    Skips leading type-0 NINJA entries (co-op levels store two ninja spawns
    at offsets 1230 and 1235 before the real entity list begins at 1240).
    """
    data = path.read_bytes()
    if len(data) < 8:
        return []
    map_data_length = struct.unpack("<I", data[0:4])[0]
    end = _MAP_DATA_OFFSET + map_data_length
    chunk = data[ENTITY_SECTION_OFFSET:end]
    i = 0
    # Skip co-op NINJA entries at the head of the list
    while i + 4 < len(chunk) and chunk[i] == 0:
        i += 5
    etypes: List[int] = []
    while i + 4 < len(chunk):
        et = chunk[i]
        if et == 0:
            break
        etypes.append(et)
        i += 5
    return etypes


def is_coop_format(path: Path) -> bool:
    """Return True if the entity section starts with a type-0 (NINJA) entry.

    These files look like zero-door to a naive full-entity-section parser
    because the parser stops immediately at the leading type-0, missing all
    subsequent gold/exit-door/etc. entities that start at offset 1240.
    """
    data = path.read_bytes()
    if len(data) <= ENTITY_SECTION_OFFSET:
        return False
    return data[ENTITY_SECTION_OFFSET] == 0


def find_coop_format_files(attracts_dir: Path, scan_limit: int, want: int):
    files = sorted(attracts_dir.iterdir())[:scan_limit]
    found = []
    print(f"Scanning up to {len(files)} attract files ...")
    for i, f in enumerate(files):
        if not f.is_file():
            continue
        if i % 2000 == 0 and i > 0:
            print(f"  ... {i} scanned, {len(found)} co-op-format so far")
        if is_coop_format(f):
            found.append(f)
        if len(found) >= want:
            break
    return found


# ── Input decoding ────────────────────────────────────────────────────────────

def decode_input(inp: int):
    jump = 1 if (inp & 0x01) else 0
    right = 1 if (inp & 0x02) else 0
    left = 1 if (inp & 0x04) else 0
    hor = 0 if (left and right) else (-1 if left else (1 if right else 0))
    return hor, jump


# ── Video rendering ───────────────────────────────────────────────────────────

def render_attract_to_mp4(attract_path: Path, output_path: Path, fps: int) -> bool:
    """Render an attract file to MP4 using NSimRenderer.

    Passes the raw attract file bytes directly to sim.load() (the simulator's
    own entity loader handles the full binary format, including co-op levels).
    """
    decoder = NppAttractDecoder()
    decoded = decoder.decode_npp_attract_file(str(attract_path))
    demo_inputs = decoded["demo_data"]
    if not demo_inputs:
        print(f"    No demo inputs found in {attract_path.name}")
        return False

    raw_map = attract_path.read_bytes()

    sc = SimConfig(basic_sim=True, enable_anim=False)
    sim = NcloneSimulator(sc)
    sim.load(list(raw_map))

    pygame.init()
    renderer = NSimRenderer(sim, render_mode="rgb_array")

    frames_dir = Path(tempfile.mkdtemp(prefix="zdoor_frames_"))
    try:
        for frame_idx, inp in enumerate(demo_inputs):
            hor, jump = decode_input(inp)
            sim.tick(hor, jump)
            surface = renderer.draw(init=(frame_idx == 0))
            arr = pygame.surfarray.array3d(surface).transpose(1, 0, 2)

            # Overlay frame counter and entity summary
            img = Image.fromarray(arr)
            draw = ImageDraw.Draw(img)
            state = sim.ninja.state
            draw.rectangle([0, 0, 220, 18], fill=(0, 0, 0))
            draw.text(
                (4, 2),
                f"f={frame_idx+1:4d}  state={state}  "
                f"pos=({sim.ninja.xpos:.0f},{sim.ninja.ypos:.0f})",
                fill=(255, 255, 0),
            )
            arr = np.array(img)

            frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
            Image.fromarray(arr).save(frame_path)

            if sim.ninja.state in (6, 7, 8):
                break

        # Assemble with ffmpeg
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "20",
            str(output_path),
        ]
        result = subprocess.run(
            ffmpeg_cmd, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"    ffmpeg error: {result.stderr[-300:]}")
            return False
        return True

    finally:
        for p in frames_dir.iterdir():
            p.unlink()
        frames_dir.rmdir()
        pygame.quit()


# ── Description helper ────────────────────────────────────────────────────────

def describe_file(path: Path) -> dict:
    data = path.read_bytes()
    map_data_length = struct.unpack("<I", data[0:4])[0] if len(data) >= 4 else 0
    demo_data_length = struct.unpack("<I", data[4:8])[0] if len(data) >= 8 else 0
    etypes = read_full_entity_types(path)
    counts = dict(Counter(etypes))

    # Also count raw 5-byte NINJA entries (co-op detection)
    end = _MAP_DATA_OFFSET + map_data_length
    chunk = data[ENTITY_SECTION_OFFSET:end]
    leading_ninjas = 0
    i = 0
    while i + 4 < len(chunk) and chunk[i] == 0:
        leading_ninjas += 1
        i += 5

    return {
        "file_size": len(data),
        "map_data_length": map_data_length,
        "demo_data_length": demo_data_length,
        "leading_ninja_entries": leading_ninjas,
        "entity_counts": counts,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate videos for attract files with legitimately zero exit doors"
    )
    parser.add_argument(
        "--attracts-dir", type=Path,
        default=project_root / ".." / "attracts",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=project_root / "attract_zero_door_validation",
    )
    parser.add_argument("--scan-limit", type=int, default=5000)
    parser.add_argument("--videos", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    attracts_dir = args.attracts_dir.resolve()
    if not attracts_dir.exists():
        print(f"Error: attracts directory not found: {attracts_dir}")
        return 1

    candidates = find_coop_format_files(attracts_dir, args.scan_limit, args.videos)

    print(f"\nFound {len(candidates)} co-op-format file(s) "
          f"(zero-door to naive parser) in the first {args.scan_limit} scanned.")
    if not candidates:
        print("No co-op-format files found in this sample.")
        return 0

    print("\nFiles selected for visual inspection:")
    for f in candidates:
        info = describe_file(f)
        named = {ENTITY_NAMES.get(t, f"type{t}"): c for t, c in info["entity_counts"].items()}
        print(f"\n  {f.name}  (co-op: {info['leading_ninja_entries']} leading NINJA entries)")
        print(f"    file_size          = {info['file_size']} bytes")
        print(f"    map_data_length    = {info['map_data_length']} bytes")
        print(f"    entity_counts      = {named}")
        print(f"    EXIT_DOOR count    = {info['entity_counts'].get(3, 0)}")
        print("    naive parser sees  = 0 exit doors (stops at leading NINJA type-0)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    succeeded = []
    for f in candidates:
        out = args.output_dir / f"{f.name}.mp4"
        print(f"\nRendering {f.name} -> {out.name} ...")
        ok = render_attract_to_mp4(f, out, args.fps)
        if ok:
            print(f"  OK: {out}")
            succeeded.append((f.name, out))
        else:
            print("  FAILED")

    if succeeded:
        print("\n" + "=" * 70)
        print("HUMAN VALIDATION NEEDED")
        print("=" * 70)
        print("Please review the video(s):")
        for name, path in succeeded:
            print(f"  {path}")
        print()
        print("Questions to answer while watching:")
        print("  • Is there a visible exit door on the map?")
        print("  • Is this a co-op level (two ninjas)?")
        print("  • Does the ninja/replay end in a win?")
        print("  • Does the level look valid?")
        print()
        input("Press Enter once you have reviewed the video(s) to continue ...")
        print("Thank you — investigation can continue.")
    else:
        print("\nNo videos generated successfully.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
