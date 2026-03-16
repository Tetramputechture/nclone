#!/usr/bin/env python3
"""
Generate a side-by-side comparison video of an original attract replay vs its
horizontal mirror to visualise physics asymmetry.

Usage:
    python scripts/visualize_mirror_failure.py \
        --file 10142352 \
        --output /tmp/mirror_failure.mp4 \
        [--fps 30] [--slow 3] [--focus-frame 54] [--pre 30] [--post 80]

    --slow N   : slow-motion multiplier around the divergence frame
    --pre N    : frames to show before divergence
    --post N   : frames to show after divergence
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nclone.replay.npp_attract_decoder import NppAttractDecoder
from nclone.nsim import Simulator as NcloneSimulator
from nclone.sim_config import SimConfig
from nclone.nsim_renderer import NSimRenderer
from nclone.offline_rl.dataset.augmentation.horizontal_mirror import HorizontalMirror
from nclone.offline_rl.dataset.augmentation.attract_augmentor import _build_map_data

ATTRACTS_DIR = Path(__file__).resolve().parent.parent.parent / "attracts_filtered"
SIM_W, SIM_H = 1056, 600


def decode_inp(inp: int):
    jump = 1 if (inp & 0x01) else 0
    right = 1 if (inp & 0x02) else 0
    left = 1 if (inp & 0x04) else 0
    hor = 0 if (left and right) else (-1 if left else (1 if right else 0))
    return hor, jump


def find_divergence_frame(
    data: dict, mdata: dict, orig_map: bytes, mirror_map: bytes, max_frames: int = 2000
) -> Optional[int]:
    sc_o = SimConfig(basic_sim=True, enable_anim=False)
    sim_o = NcloneSimulator(sc_o)
    sim_o.load(list(orig_map))
    sc_m = SimConfig(basic_sim=True, enable_anim=False)
    sim_m = NcloneSimulator(sc_m)
    sim_m.load(list(mirror_map))

    for i, (io, im) in enumerate(
        zip(data["demo_data"][:max_frames], mdata["demo_data"][:max_frames])
    ):
        sim_o.tick(*decode_inp(io))
        sim_m.tick(*decode_inp(im))
        ox, oy = sim_o.ninja.xpos, sim_o.ninja.ypos
        mx, my = sim_m.ninja.xpos, sim_m.ninja.ypos
        if abs(mx - (1056 - ox)) > 0.005 or abs(my - oy) > 0.005:
            return i + 1  # 1-indexed
    return None


def run_sim_to_frame(map_bytes: bytes, inputs: list, frame: int, anim: bool = True):
    """Run simulation up to `frame` and return (sim, renderer)."""
    sc = SimConfig(basic_sim=True, enable_anim=anim)
    sim = NcloneSimulator(sc)
    sim.load(list(map_bytes))
    renderer = NSimRenderer(sim, render_mode="rgb_array")
    for i, inp in enumerate(inputs):
        if i >= frame:
            break
        sim.tick(*decode_inp(inp))
        renderer.draw(init=(i == 0))
    return sim, renderer


def render_frame(sim: NcloneSimulator, renderer: NSimRenderer, inp: int, init: bool) -> np.ndarray:
    """Tick the sim and return an RGB numpy array (H x W x 3)."""
    sim.tick(*decode_inp(inp))
    surface = renderer.draw(init=init)
    return pygame.surfarray.array3d(surface).transpose(1, 0, 2)  # W,H,3 → H,W,3


def make_label_bar(
    width: int, height: int,
    left_text: str, right_text: str,
    bg_color=(30, 30, 30), fg_color=(255, 255, 255),
) -> np.ndarray:
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    # Use default PIL font (no external font needed)
    draw.text((10, height // 2 - 8), left_text, fill=fg_color)
    draw.text((width // 2 + 10, height // 2 - 8), right_text, fill=fg_color)
    return np.array(img)


def make_info_bar(
    width: int, height: int,
    frame: int, diverged: bool,
    orig_pos: tuple, mirror_pos: tuple, err_y: float,
) -> np.ndarray:
    bg = (60, 0, 0) if diverged else (30, 30, 30)
    img = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    fg = (255, 255, 255)
    lhs = f"f={frame:4d}  orig ({orig_pos[0]:7.2f}, {orig_pos[1]:7.2f})"
    rhs = f"mirror ({mirror_pos[0]:7.2f}, {mirror_pos[1]:7.2f})"
    err_str = f"  Δy={err_y:+.3f}" if diverged else ""
    draw.text((10, height // 2 - 8), lhs + rhs + err_str, fill=fg)
    return np.array(img)


def composite_frame(
    left: np.ndarray, right: np.ndarray,
    frame_idx: int, diverged: bool,
    orig_pos: tuple, mirror_pos: tuple, err_y: float,
    panel_w: int, panel_h: int,
) -> np.ndarray:
    """Stack label bar + two panels + info bar into one image."""
    label_h = 36
    info_h = 36
    total_h = label_h + panel_h + info_h
    total_w = panel_w * 2

    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    # Label bar
    label = make_label_bar(
        total_w, label_h,
        "  ORIGINAL", "  MIRROR",
    )
    canvas[:label_h] = label

    # Both panels side-by-side
    canvas[label_h: label_h + panel_h, :panel_w] = left
    canvas[label_h: label_h + panel_h, panel_w:] = right

    # Draw a thin divider
    canvas[label_h: label_h + panel_h, panel_w - 1: panel_w + 1] = [255, 255, 0]

    # Info bar
    info = make_info_bar(total_w, info_h, frame_idx, diverged, orig_pos, mirror_pos, err_y)
    canvas[label_h + panel_h:] = info

    # Red border tint when diverged
    if diverged:
        canvas[label_h: label_h + 4, :] = [200, 0, 0]
        canvas[label_h + panel_h - 4: label_h + panel_h, :] = [200, 0, 0]
        canvas[label_h:, :4] = [200, 0, 0]
        canvas[label_h:, -4:] = [200, 0, 0]

    return canvas


def scale_frame(arr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    img = Image.fromarray(arr)
    img = img.resize((target_w, target_h), Image.LANCZOS)
    return np.array(img)


def save_as_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path)


def build_video(
    frames_dir: Path, output: Path, fps: float
) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg error:", result.stderr[-500:])
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="10142352", help="Attract filename")
    parser.add_argument("--attracts-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("/tmp/mirror_failure.mp4"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--slow", type=int, default=3,
                        help="Slow-motion repeat count around divergence")
    parser.add_argument("--focus-frame", type=int, default=None,
                        help="Override auto-detected divergence frame")
    parser.add_argument("--pre", type=int, default=40,
                        help="Normal-speed frames before entering slow-mo zone")
    parser.add_argument("--post", type=int, default=80,
                        help="Normal-speed frames after slow-mo zone ends")
    parser.add_argument("--slow-window", type=int, default=20,
                        help="Number of frames on each side to slow down")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Scale factor for each panel (0.5 = 528x300)")
    args = parser.parse_args()

    attracts_dir = args.attracts_dir or ATTRACTS_DIR
    fpath = attracts_dir / args.file
    if not fpath.exists():
        print(f"File not found: {fpath}")
        return 1

    pygame.init()

    dec = NppAttractDecoder()
    mirror_aug = HorizontalMirror()

    print(f"Loading {args.file}...")
    data = dec.decode_npp_attract_file(str(fpath))
    orig_map = dec.create_nclone_map(str(fpath))
    mdata = mirror_aug.augment(data)
    mirror_map = _build_map_data(mdata)

    if not data.get("demo_data"):
        print("No demo data found")
        return 1

    # Find divergence frame
    div_frame = args.focus_frame
    if div_frame is None:
        print("Finding divergence frame...")
        div_frame = find_divergence_frame(data, mdata, orig_map, mirror_map)
        if div_frame is None:
            print("No divergence found in first 2000 frames — will render full run")
            div_frame = len(data["demo_data"])

    print(f"Divergence at frame {div_frame}")

    # Determine which frames to render
    slow_start = max(0, div_frame - args.slow_window - 1)
    slow_end = min(len(data["demo_data"]) - 1, div_frame + args.slow_window)
    video_start = max(0, slow_start - args.pre)
    video_end = min(len(data["demo_data"]) - 1, slow_end + args.post)

    panel_w = int(SIM_W * args.scale)
    panel_h = int(SIM_H * args.scale)

    # Create sims with full animation enabled for rendering
    print("Initialising simulators...")
    sc_o = SimConfig(basic_sim=False, enable_anim=True)
    sim_o = NcloneSimulator(sc_o)
    sim_o.load(list(orig_map))
    rend_o = NSimRenderer(sim_o, render_mode="rgb_array")

    sc_m = SimConfig(basic_sim=False, enable_anim=True)
    sim_m = NcloneSimulator(sc_m)
    sim_m.load(list(mirror_map))
    rend_m = NSimRenderer(sim_m, render_mode="rgb_array")

    # Warm-up: render frame 0 to initialise tile/entity caches, then fast-forward
    print(f"Fast-forwarding to frame {video_start}...")
    rend_o.draw(init=True)
    rend_m.draw(init=True)
    for i in range(video_start):
        sim_o.tick(*decode_inp(data["demo_data"][i]))
        rend_o.draw(init=False)
        sim_m.tick(*decode_inp(mdata["demo_data"][i]))
        rend_m.draw(init=False)

    print(f"Rendering frames {video_start}–{video_end}...")
    output_frames: List[Path] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        png_idx = 0

        for frame_idx in range(video_start, video_end + 1):
            if frame_idx >= len(data["demo_data"]):
                break

            inp_o = data["demo_data"][frame_idx]
            inp_m = mdata["demo_data"][frame_idx]

            # Render original
            sim_o.tick(*decode_inp(inp_o))
            surf_o = rend_o.draw(init=False)
            arr_o = pygame.surfarray.array3d(surf_o).transpose(1, 0, 2)

            # Render mirror
            sim_m.tick(*decode_inp(inp_m))
            surf_m = rend_m.draw(init=False)
            arr_m = pygame.surfarray.array3d(surf_m).transpose(1, 0, 2)

            # Scale panels
            left = scale_frame(arr_o, panel_w, panel_h)
            right = scale_frame(arr_m, panel_w, panel_h)

            # Compute error
            ox, oy = sim_o.ninja.xpos, sim_o.ninja.ypos
            mx, my = sim_m.ninja.xpos, sim_m.ninja.ypos
            exp_mx = 1056 - ox
            err_y = my - oy
            diverged = (frame_idx >= div_frame) and (abs(err_y) > 0.001 or abs(mx - exp_mx) > 0.001)

            composite = composite_frame(
                left, right, frame_idx + 1, diverged,
                (ox, oy), (mx, my), err_y,
                panel_w, panel_h,
            )

            # Slow-motion: repeat frame
            repeat = args.slow if slow_start <= frame_idx <= slow_end else 1
            for _ in range(repeat):
                png_path = tmp_path / f"frame_{png_idx:06d}.png"
                save_as_png(composite, png_path)
                png_idx += 1

        print(f"Encoding {png_idx} PNG frames → {args.output} …")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ok = build_video(tmp_path, args.output, fps=args.fps)
        if ok:
            size_mb = args.output.stat().st_size / 1e6
            print(f"Done: {args.output}  ({size_mb:.1f} MB)")
        else:
            print("Video encoding failed")
            return 1

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
