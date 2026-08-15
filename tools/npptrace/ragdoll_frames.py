#!/usr/bin/env python3
"""Render a death and save the ragdoll frames as PNGs.

    NCLONE_ATTRACTS=/tmp/official python tools/npptrace/ragdoll_frames.py 1016 \
        --out /tmp/ragdoll --count 8 --every 6

Proof that the renderer draws the corpse rather than the frozen live skeleton:
the live path is ninja-relative and the ninja stops moving on the death frame,
so if the drawn limbs move, they are coming from `ragdoll.bones_pos`.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

ATTRACTS = os.environ.get("NCLONE_ATTRACTS", "/home/tetra/projects/attracts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo")
    ap.add_argument("--dir", default=ATTRACTS)
    ap.add_argument("--out", default="/tmp/ragdoll")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--zoom", type=int, default=0,
                    help="crop this many level pixels around the corpse and "
                         "upscale; 0 saves the whole level")
    args = ap.parse_args()

    from PIL import Image
    from nclone.replay.npp_attract_decoder import NppAttractDecoder
    from nclone.replay.replay_executor import decode_input_to_controls
    from nclone.nplay_headless import NPlayHeadless

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.dir, args.demo)
    dec = NppAttractDecoder()
    demo = dec.decode_npp_attract_file(path).get("demo_data", []) or []
    mapb = list(dec.create_nclone_map(path))
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False,
                      enable_animation=True, enable_rendering=True)
    h.load_map_from_map_data(mapb)
    ninja = h.sim.ninja

    def save(tag):
        img = Image.fromarray(h.render())
        if args.zoom:
            w, h_px = img.size
            # The render is the level scaled to the surface; map level pixels
            # to image pixels with the same factor the renderer used.
            scale = w / 1056.0
            cx = ninja.ragdoll.xpos * scale
            cy = ninja.ragdoll.ypos * scale
            r = args.zoom * scale
            box = (max(0, int(cx - r)), max(0, int(cy - r)),
                   min(w, int(cx + r)), min(h_px, int(cy + r)))
            img = img.crop(box)
            if img.size[0]:
                img = img.resize((img.size[0] * 6, img.size[1] * 6), Image.NEAREST)
        img.save(os.path.join(args.out, "%s.png" % tag))

    for b in demo:
        hor, jmp = decode_input_to_controls(int(b))
        h.tick(hor, jmp)
        if ninja.state == 6:
            break
    if ninja.state != 6:
        print("%s never died (state=%d)" % (args.demo, ninja.state))
        return 1

    rag = ninja.ragdoll
    print("%s: cause=%r, %s, %d bones"
          % (args.demo, ninja.death_cause,
             "exploded" if rag.state == 1 else "intact", rag.num))
    save("death+000")
    print("  wrote death+000  bone0=(%.2f, %.2f)"
          % (rag.bones_pos[0][0], rag.bones_pos[0][1]))
    for i in range(1, args.count + 1):
        for _ in range(args.every):
            h.tick(0, 0)
        tag = "death+%03d" % (i * args.every)
        save(tag)
        print("  wrote %s  bone0=(%.2f, %.2f)"
              % (tag, rag.bones_pos[0][0], rag.bones_pos[0][1]))
    print("frames in %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
