#!/usr/bin/env python3
"""Replay demos and report how each one ends, with the death cause.

The fastest signal there is: one line per demo, no capture needed.

    python tools/npptrace/death_cause.py 1000622 10138417 10134679
    NCLONE_AB_TURRET_LEGACY=1 python tools/npptrace/death_cause.py 10138417
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

ATTRACTS = os.environ.get("NCLONE_ATTRACTS", "/home/tetra/projects/attracts")


def main():
    from nclone.replay.npp_attract_decoder import NppAttractDecoder
    from nclone.replay.replay_executor import decode_input_to_controls
    from nclone.nplay_headless import NPlayHeadless

    for demo_id in sys.argv[1:]:
        p = os.path.join(ATTRACTS, demo_id)
        dec = NppAttractDecoder()
        demo = dec.decode_npp_attract_file(p).get("demo_data", []) or []
        mapb = list(dec.create_nclone_map(p))
        h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
        h.load_map_from_map_data(mapb)
        n = h.sim.ninja
        frame = 0
        for b in demo:
            hor, jmp = decode_input_to_controls(int(b))
            h.tick(hor, jmp)
            frame += 1
            if h.ninja_has_died():
                print("%-10s DIED  frame %-5d cause=%-16r pos=(%.3f, %.3f)"
                      % (demo_id, frame, getattr(n, "death_cause", None),
                         n.xpos, n.ypos))
                break
            if h.ninja_has_won():
                print("%-10s WON   frame %d" % (demo_id, frame))
                break
        else:
            print("%-10s TIMEOUT after %d frames" % (demo_id, frame))


if __name__ == "__main__":
    main()
