#!/usr/bin/env python3
"""Compare nclone's 13 ninja bones against the game's.

Needs a capture taken with the WIDE ninja slab (`NINJA_SLAB = 0xD40` in
`agent_entities.py`/`trace_entities.py`); bones live at `ninja + 0xC60`, 13 pairs
of doubles, stride `0x10`, which is past the default `0x200` slab.

    NCLONE_TRACES=/tmp/out9b python tools/npptrace/compare_bones.py 10138417 149
"""
import os
import struct
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from read_entities import Capture  # noqa: E402

ATTRACTS = os.environ.get("NCLONE_ATTRACTS", "/home/tetra/projects/attracts")
TRACES = os.environ.get(
    "NCLONE_TRACES", os.path.join(HERE, "..", "..", "docs", "game_traces")
)
BONES = 0xC60


def main():
    demo = sys.argv[1] if len(sys.argv) > 1 else "10138417"
    frame = int(sys.argv[2]) if len(sys.argv) > 2 else 149

    from nclone.replay.npp_attract_decoder import NppAttractDecoder
    from nclone.replay.replay_executor import decode_input_to_controls
    from nclone.nplay_headless import NPlayHeadless

    cap = Capture(os.path.join(TRACES, demo))
    slab = cap.layout["ninjaSlab"]
    if slab < BONES + 13 * 0x10:
        raise SystemExit(
            "capture slab is 0x%X, too small for bones at 0x%X -- recapture with "
            "NINJA_SLAB = 0xD40" % (slab, BONES))

    dec = NppAttractDecoder()
    p = os.path.join(ATTRACTS, demo)
    tape = dec.decode_npp_attract_file(p).get("demo_data", []) or []
    mapb = list(dec.create_nclone_map(p))
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h.load_map_from_map_data(mapb)
    n = h.sim.ninja

    first = None
    for f in range(1, min(frame, cap.n, len(tape)) + 1):
        hor, jmp = decode_input_to_controls(int(tape[f - 1]))
        h.tick(hor, jmp)
        if not n.bones:
            continue
        s = cap.frame(f)[4:4 + slab]
        for i in range(13):
            gx = struct.unpack_from("<d", s, BONES + i * 0x10)[0]
            gy = struct.unpack_from("<d", s, BONES + i * 0x10 + 8)[0]
            if (n.bones[i][0] != gx or n.bones[i][1] != gy) and first is None:
                first = (f, i, n.bones[i][0], n.bones[i][1], gx, gy)

    if first:
        f, i, mx, my, gx, gy = first
        print("first bone divergence: frame %d bone %d  nclone=(%.12f, %.12f) "
              "game=(%.12f, %.12f)  d=(%.3e, %.3e)"
              % (f, i, mx, my, gx, gy, mx - gx, my - gy))
    else:
        print("all 13 bones match every frame up to %d" % frame)

    s = cap.frame(frame)[4:4 + slab]
    print("\nframe %d:" % frame)
    for i in range(13):
        gx = struct.unpack_from("<d", s, BONES + i * 0x10)[0]
        gy = struct.unpack_from("<d", s, BONES + i * 0x10 + 8)[0]
        mx, my = n.bones[i] if n.bones else (float("nan"),) * 2
        print("  bone %-3d nclone=(%10.6f, %10.6f) game=(%10.6f, %10.6f)%s"
              % (i, mx, my, gx, gy, "" if (mx == gx and my == gy) else "   <--"))


if __name__ == "__main__":
    main()
