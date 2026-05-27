#!/usr/bin/env python3
"""Render an mp4 of a physics-planned route completing a level.

Plans a route with the simulation-lookahead spike (weighted-A*, default
``h_weight=1.25`` — fast and **near-optimal/satisficing**, NOT provably optimal:
coarse velocity bins alias states and break Dijkstra optimality, while fine bins
are intractable — true frame-optimality needs the velocity-augmented surface
graph + backward Dijkstra). It then replays the action tape on the rendering
sim, writing an mp4. Native↔Python physics are bit-identical (validated by
``route_executor_check``), so the open-loop tape completes deterministically.

Scope: toggle mines + exit switch + exit door (pure movement). The optimal
route is mine-aware by construction (deadly-mine deaths are pruned in search).

Usage::

    python -m nclone.value_function.tools.route_video \
        --map nclone/test_maps/minefield --output docs/stage0/video_minefield.mp4
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless rendering (WSL/CI)

import math

import numpy as np
import imageio
from PIL import Image, ImageDraw

from ...nplay_headless import NPlayHeadless
from .physics_route_spike import PhysicsRouteSpike, ACT, _MINE_KILL_PX


def _to_hwc(frame) -> np.ndarray:
    """Return an (H,W,3) uint8 frame for imageio. The N++ level is landscape
    (1056w x 600h), so a correct (H,W,3) frame has rows < cols; transpose only
    if we got (W,H,3) (rows > cols), e.g. raw pygame surfarray."""
    a = np.asarray(frame)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.ndim == 3 and a.shape[2] in (3, 4) and a.shape[0] > a.shape[1]:
        a = np.transpose(a, (1, 0, 2))
    return a[..., :3].astype(np.uint8)


def _draw_overlay(frame_hwc, full_traj, upto, ninja_xy, speed, clearance,
                  frame_no, total_frames, phase, deadly):
    """Composite route trace + mine kill-zones + HUD onto a rendered frame.
    World pixels map 1:1 to frame pixels (PIL uses (x,y)=(col,row))."""
    img = Image.fromarray(frame_hwc).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    # deadly-mine kill zones (translucent; flash brighter when ninja is near)
    near = clearance < 16.0
    for (mx, my) in deadly:
        d.ellipse([mx - _MINE_KILL_PX, my - _MINE_KILL_PX,
                   mx + _MINE_KILL_PX, my + _MINE_KILL_PX],
                  fill=(255, 0, 0, 70 if near else 35),
                  outline=(255, 60, 60, 200), width=1)
    # full planned route (faint) + breadcrumb travelled (bright)
    if len(full_traj) > 1:
        d.line([(p[0], p[1]) for p in full_traj], fill=(0, 160, 255, 80), width=2)
    if upto > 1:
        d.line([(p[0], p[1]) for p in full_traj[:upto + 1]], fill=(0, 220, 255, 230), width=3)
    # ninja marker
    d.ellipse([ninja_xy[0] - 4, ninja_xy[1] - 4, ninja_xy[0] + 4, ninja_xy[1] + 4],
              outline=(255, 230, 0, 255), width=2)
    img = Image.alpha_composite(img, ov)
    d2 = ImageDraw.Draw(img)
    d2.rectangle([0, 0, 360, 38], fill=(0, 0, 0, 150))
    d2.text((6, 3), f"frame {frame_no}/{total_frames}   {phase}   |v|={speed:.2f} px/f",
            fill=(255, 255, 255, 255))
    clr = (255, 90, 90, 255) if clearance < 16.0 else (170, 255, 170, 255)
    d2.text((6, 19), f"min mine clearance: {clearance:.1f} px (kill {_MINE_KILL_PX:.0f})", fill=clr)
    return np.asarray(img.convert("RGB"))


def render_route_video(map_path: str, output: str, macro: int = 6,
                       vel_bin: float = 0.25, fps: int = 60, overlay: bool = True,
                       h_weight: float = 1.25, max_expansions: int = 200000) -> dict:
    sp = PhysicsRouteSpike(map_path, macro=macro, vel_bin=vel_bin,
                           h_weight=h_weight, max_expansions=max_expansions)
    kind = "frame-optimal (Dijkstra)" if h_weight == 0.0 else f"near-optimal (weighted-A*, w={h_weight})"
    print(f"[{os.path.basename(map_path)}] planning {kind} route (macro={macro})...")
    res = sp.plan()
    if not res["solved"]:
        print(f"  NO ROUTE found ({res['expansions']} expansions) — cannot render.")
        return res
    print(f"  route: {res['frames']} frames, {res['expansions']} expansions, "
          f"{res['wall_time_s']:.1f}s")

    deadly = sp.deadly0
    full_traj = res["trajectory"]  # native == python (bit-identical, validated)
    total = len(full_traj) - 1

    def _clear(x, y):
        return min((math.hypot(x - mx, y - my) for (mx, my) in deadly), default=float("inf"))

    def _cap(frame_no):
        raw = _to_hwc(nh.render())
        if not overlay:
            return raw
        x, y = nh.ninja_position()
        vx, vy = nh.ninja_velocity()
        phase = "-> EXIT" if nh.exit_switch_activated() else "-> switch"
        return _draw_overlay(raw, full_traj, frame_no, (x, y), math.hypot(vx, vy),
                             _clear(x, y), frame_no, total, phase, deadly)

    nh = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    nh.load_map(map_path)
    nh.reset()
    frames = [_cap(0)]
    won = died = False
    frame_no = 0
    for a in res["actions"]:
        h, j = ACT[a]
        for _ in range(macro):
            nh.tick(h, j)
            frame_no += 1
            frames.append(_cap(frame_no))
            if nh.ninja_has_won():
                won = True
                break
            if nh.ninja_has_died():
                died = True
                break
        if won or died:
            break

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    # hold the final frame ~1s so the completion reads clearly
    frames.extend([frames[-1]] * fps)
    imageio.mimsave(output, frames, fps=fps, macro_block_size=1)
    print(f"  replay: won={won} died={died} | {len(frames)} frames @ {fps}fps "
          f"(overlay={overlay}) -> {output}")
    return {"won": won, "died": died, "n_frames": len(frames),
            "route_frames": res["frames"], "output": output}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", required=True)
    p.add_argument("--output", required=True, help="output .mp4 path")
    p.add_argument("--macro", type=int, default=6)
    p.add_argument("--vel-bin", type=float, default=0.25)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--no-overlay", action="store_true", help="render plain frames (no HUD/route overlay)")
    p.add_argument("--h-weight", type=float, default=1.25,
                   help="weighted-A* weight; 0 = Dijkstra (slow, and unsound under coarse bins)")
    args = p.parse_args()
    render_route_video(args.map, args.output, macro=args.macro, vel_bin=args.vel_bin,
                       fps=args.fps, overlay=not args.no_overlay, h_weight=args.h_weight)
    return 0


if __name__ == "__main__":
    sys.exit(main())
