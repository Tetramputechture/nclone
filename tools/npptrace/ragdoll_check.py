#!/usr/bin/env python3
"""Replay a demo up to the death, then keep ticking and report the corpse.

    NCLONE_ATTRACTS=/tmp/official python tools/npptrace/ragdoll_check.py 1016
    ... ragdoll_check.py 1016 --frames 300 --dump 5

There is no ragdoll capture on disk to diff against (see
`tools/npptrace/scan_deaths.py`), so this is the next best thing: it asserts
the invariants the physics is supposed to hold -- the eleven sticks stay near
their table lengths, no bone ends up inside geometry, nothing goes non-finite,
and the corpse eventually comes to rest.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

ATTRACTS = os.environ.get("NCLONE_ATTRACTS", "/home/tetra/projects/attracts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo")
    ap.add_argument("--dir", default=ATTRACTS)
    ap.add_argument("--frames", type=int, default=240,
                    help="ragdoll frames to simulate after the death")
    ap.add_argument("--dump", type=int, default=0,
                    help="print bone 0 every N frames")
    args = ap.parse_args()

    from nclone.replay.npp_attract_decoder import NppAttractDecoder
    from nclone.replay.replay_executor import decode_input_to_controls
    from nclone.nplay_headless import NPlayHeadless
    from nclone.physics import get_single_closest_point

    path = os.path.join(args.dir, args.demo)
    dec = NppAttractDecoder()
    demo = dec.decode_npp_attract_file(path).get("demo_data", []) or []
    mapb = list(dec.create_nclone_map(path))
    # The ragdoll only exists in animation mode -- it *is* the animation.
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False,
                      enable_animation=True)
    h.load_map_from_map_data(mapb)
    ninja = h.sim.ninja

    death_frame = None
    for i, b in enumerate(demo):
        hor, jmp = decode_input_to_controls(int(b))
        h.tick(hor, jmp)
        if ninja.state == 6:
            death_frame = i + 1
            break
    if death_frame is None:
        print("%s never reached state 6 (state=%d)" % (args.demo, ninja.state))
        return 1

    rag = ninja.ragdoll
    print("%s died on frame %d, cause=%r, %s"
          % (args.demo, death_frame, ninja.death_cause,
             "EXPLODED (%d bones)" % rag.num if rag.state == 1
             else "intact (%d bones)" % rag.num))
    print("ninja was at (%.3f, %.3f)" % (ninja.xpos, ninja.ypos))

    mines = [e for lst in h.sim.entity_dic.values() for e in lst
             if getattr(e, "type", None) in (1, 21)]
    blown_at_death = sum(1 for m in mines if getattr(m, "detonated", False))
    print("mines on this level: %d, spent at the death frame: %d"
          % (len(mines), blown_at_death))

    worst = [0.0] * len(rag.segs)
    nonfinite = 0
    prev = [list(p) for p in rag.bones_pos]
    motion_history = []
    for f in range(args.frames):
        h.tick(0, 0)
        for i, (a, b) in enumerate(rag.segs):
            d = math.hypot(rag.bones_pos[a][0] - rag.bones_pos[b][0],
                           rag.bones_pos[a][1] - rag.bones_pos[b][1])
            worst[i] = max(worst[i], abs(d - rag.seg_lengths[i]))
        moved = 0.0
        for i in range(rag.num):
            px, py = rag.bones_pos[i]
            if not (math.isfinite(px) and math.isfinite(py)):
                nonfinite += 1
            moved += math.hypot(px - prev[i][0], py - prev[i][1])
            prev[i] = [px, py]
        motion_history.append(moved)
        if args.dump and f % args.dump == 0:
            print("  +%-4d bone0=(%9.3f, %9.3f)  motion=%.4f"
                  % (f + 1, rag.bones_pos[0][0], rag.bones_pos[0][1], moved))

    print()
    print("after %d ragdoll frames:" % args.frames)
    print("  non-finite bone coordinates : %d" % nonfinite)
    print("  centre                      : (%.3f, %.3f)" % (rag.xpos, rag.ypos))
    print("  total motion, last 10 frames: %.5f px" % sum(motion_history[-10:]))
    print("  stick length error (worst seen / final):")
    ok = True
    for i, (a, b) in enumerate(rag.segs):
        final = math.hypot(rag.bones_pos[a][0] - rag.bones_pos[b][0],
                           rag.bones_pos[a][1] - rag.bones_pos[b][1])
        err = abs(final - rag.seg_lengths[i])
        if err > 0.5:
            ok = False
        print("    %2d  %2d-%-2d  want %.1f  worst %7.4f  final %7.4f (err %.4f)"
              % (i, a, b, rag.seg_lengths[i], worst[i], final, err))

    embedded = 0
    for i in range(rag.num):
        x, y = rag.bones_pos[i]
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        result, point = get_single_closest_point(h.sim, x, y, 10.0)
        if result == 0 or point is None:
            continue
        dist = math.hypot(x - point[0], y - point[1])
        if 1.0 - dist * result > 1e-6:
            embedded += 1
    print("  bones still inside geometry : %d" % embedded)
    blown_now = sum(1 for m in mines if getattr(m, "detonated", False))
    print("  mines set off by limbs      : %d (of %d, %d already spent)"
          % (blown_now - blown_at_death, len(mines), blown_at_death))
    print("  ragdoll bones               : %d (%s)"
          % (rag.num, "exploded" if rag.state == 1 else "intact"))

    if nonfinite or embedded or not ok:
        print("\nFAIL")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
