#!/usr/bin/env python3
"""Replay an attract demo in nclone and emit the same JSONL schema as
`docs/game_traces/`, so a trace can be diffed field-by-field against the game.

    python tools/npptrace/sim_trace.py 10131738 --engine python -o /tmp/p.jsonl

Frame convention matches the capture: record N is the state at the END of frame
N, records start at 1.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

ATTRACTS = os.environ.get("NCLONE_ATTRACTS", "/home/tetra/projects/attracts")


def hexbits(x: float) -> str:
    return "%016x" % struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("demo")
    ap.add_argument("--engine", choices=("python", "native"), default="python")
    ap.add_argument("--max-frames", type=int, default=100000)
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    path = args.demo if os.path.exists(args.demo) else os.path.join(ATTRACTS, args.demo)
    if not os.path.exists(path):
        print("no such attract: %s" % path, file=sys.stderr)
        return 2

    from nclone.replay.npp_attract_decoder import NppAttractDecoder
    from nclone.replay.replay_executor import decode_input_to_controls

    dec = NppAttractDecoder()
    res = dec.decode_npp_attract_file(path)
    demo = res.get("demo_data", []) or []
    mapb = list(dec.create_nclone_map(path))
    if not demo:
        print("no demo tape in %s" % path, file=sys.stderr)
        return 2

    if args.engine == "python":
        from nclone.nplay_headless import NPlayHeadless

        h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    else:
        from nclone.native_headless import NativePlayHeadless

        h = NativePlayHeadless()
    h.load_map_from_map_data(mapb)

    out = open(args.out, "w") if args.out else sys.stdout
    outcome, frame = "timeout", 0
    for b in demo[: args.max_frames]:
        hor, jmp = decode_input_to_controls(int(b))
        h.tick(hor, jmp)
        frame += 1
        n = _ninja(h)
        rec = {
            "frame": frame,
            "xpos": _f(n, "xpos"), "ypos": _f(n, "ypos"),
            "xspeed": _f(n, "xspeed"), "yspeed": _f(n, "yspeed"),
            "state": int(_g(n, "state", 0)),
            "floor_x": _f(n, "floor_normalized_x"),
            "floor_y": _f(n, "floor_normalized_y"),
        }
        rec["hex"] = {k: hexbits(rec[k]) for k in
                      ("xpos", "ypos", "xspeed", "yspeed", "floor_x", "floor_y")}
        out.write(json.dumps(rec) + "\n")
        if h.ninja_has_won():
            outcome = "win"
            break
        if h.ninja_has_died():
            outcome = "die"
            break
    if args.out:
        out.close()
    print("%s engine=%s outcome=%s frames=%d"
          % (os.path.basename(path), args.engine, outcome, frame), file=sys.stderr)
    return 0


def _ninja(h):
    n = getattr(getattr(h, "sim", None), "ninja", None)
    if n is not None:
        return n
    for attr in ("ninja", "_ninja"):
        if hasattr(h, attr):
            return getattr(h, attr)
    raise SystemExit("cannot reach the ninja on %r" % type(h))


def _g(n, name, default=None):
    if isinstance(n, dict):
        return n.get(name, default)
    return getattr(n, name, default)


def _f(n, name) -> float:
    return float(_g(n, name, 0.0) or 0.0)


if __name__ == "__main__":
    raise SystemExit(main())
