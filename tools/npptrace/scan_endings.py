#!/usr/bin/env python3
"""Replay a whole attract directory and report how each demo ends.

    NCLONE_ATTRACTS=/tmp/official python tools/npptrace/scan_endings.py --limit 200
    python tools/npptrace/scan_endings.py --limit 400 --deaths-only

Like `death_cause.py`, but it takes a directory rather than a list of ids and
it survives demos that fail to decode, so it can be pointed at the game's own
attract cache. Prints the ids that end in a death -- the ones worth capturing
if you want ragdoll ground truth.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

ATTRACTS = os.environ.get("NCLONE_ATTRACTS", "/home/tetra/projects/attracts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=ATTRACTS)
    ap.add_argument("--limit", type=int, default=0, help="0 = every file")
    ap.add_argument("--deaths-only", action="store_true")
    args = ap.parse_args()

    from nclone.replay.npp_attract_decoder import NppAttractDecoder
    from nclone.replay.replay_executor import decode_input_to_controls
    from nclone.nplay_headless import NPlayHeadless

    ids = sorted(os.listdir(args.dir))
    if args.limit:
        ids = ids[: args.limit]

    tally = {"DIED": 0, "WON": 0, "TIMEOUT": 0, "ERROR": 0}
    deaths = []
    for demo_id in ids:
        path = os.path.join(args.dir, demo_id)
        if not os.path.isfile(path):
            continue
        try:
            dec = NppAttractDecoder()
            demo = dec.decode_npp_attract_file(path).get("demo_data", []) or []
            mapb = list(dec.create_nclone_map(path))
            h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
            h.load_map_from_map_data(mapb)
        except Exception:  # noqa: BLE001
            tally["ERROR"] += 1
            if not args.deaths_only:
                print("%-10s ERROR %s" % (demo_id, traceback.format_exc(0).strip()))
            continue

        ninja = h.sim.ninja
        outcome = "TIMEOUT"
        frame = 0
        for b in demo:
            hor, jmp = decode_input_to_controls(int(b))
            h.tick(hor, jmp)
            frame += 1
            if h.ninja_has_died():
                outcome = "DIED"
                break
            if h.ninja_has_won():
                outcome = "WON"
                break
        tally[outcome] += 1
        if outcome == "DIED":
            deaths.append((demo_id, frame, getattr(ninja, "death_cause", None)))
            print("%-10s DIED  frame %-5d state=%d cause=%r pos=(%.3f, %.3f)"
                  % (demo_id, frame, ninja.state,
                     getattr(ninja, "death_cause", None), ninja.xpos, ninja.ypos))
        elif not args.deaths_only:
            print("%-10s %s  frame %d" % (demo_id, outcome, frame))

    print()
    print("scanned %d: %s" % (sum(tally.values()), tally))
    print("deaths: %s" % ([d[0] for d in deaths],))
    return 0


if __name__ == "__main__":
    sys.exit(main())
