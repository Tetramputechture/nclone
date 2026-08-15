"""Render a minimum set of attract demos covering every entity type, to MP4.

    python tools/npptrace/covervideo.py --gate /tmp/gate_100.jsonl --out /tmp/coverage_videos

Greedily picks the fewest winning demos whose maps between them contain every
entity type present in the corpus, then replays each on the native-backed sim and
writes it out with OpenCV (the repo's VideoGenerator shells out to ffmpeg, which
is not installed in WSL).

`--concat` also writes a single reel of all of them in cover order.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

# Types the ninja can never see: the switch halves of locked (7) and trap (9)
# doors, which the door entity owns.
PHANTOM = frozenset({7, 9})

NAMES = {
    1: "toggle mine", 2: "gold", 3: "exit door", 4: "exit switch",
    5: "regular door", 6: "locked door", 8: "trap door", 10: "launch pad",
    11: "one-way platform", 12: "chaingun drone", 13: "laser drone",
    14: "zap drone", 15: "chaser drone", 16: "floor guard", 17: "bounce block",
    18: "rocket launcher", 19: "turret", 20: "thwump", 21: "toggle mine (armed)",
    22: "evil ninja", 23: "dual laser", 24: "boost pad", 25: "death ball",
    26: "mini drone", 27: "eye bat", 28: "shove thwump", 29: "player rocket",
}


def cover(rows):
    """Fewest demos covering every type; ties broken toward shorter demos."""
    universe = set()
    for r in rows:
        universe |= set(r["types"]) - PHANTOM
    need, chosen = set(universe), []
    while need:
        best = max(rows, key=lambda r: (len(need & set(r["types"])), -r["frames"]))
        got = need & set(best["types"])
        chosen.append((best, sorted(got)))
        need -= got
    return universe, chosen


def render(demo, path_out, fps, label=None, writer=None):
    dec = NppAttractDecoder()
    path = "/home/tetra/projects/attracts/%s" % demo
    tape = dec.decode_npp_attract_file(path).get("demo_data") or []
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h.load_map_from_map_data(list(dec.create_nclone_map(path)))

    own = writer is None
    frames = 0
    for f in range(1, len(tape) + 1):
        hor, jmp = decode_input_to_controls(int(tape[f - 1]))
        h.tick(hor, jmp)
        rgb = h.render()
        if rgb is None:
            print("   render() returned None -- rendering disabled?")
            return 0, False
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        if label:
            for text, y, scale, colour, thick in (
                (label, 22, 0.55, (0, 0, 0), 3),
                (label, 22, 0.55, (255, 255, 255), 1),
            ):
                cv2.putText(bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, colour, thick, cv2.LINE_AA)
        if writer is None:
            hgt, wid = bgr.shape[:2]
            writer = cv2.VideoWriter(str(path_out), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (wid, hgt))
        writer.write(bgr)
        frames += 1
        if h.ninja_has_won() or h.ninja_has_died():
            break
    won = h.ninja_has_won()
    if own and writer is not None:
        writer.release()
    return frames, won


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="/tmp/gate_100.jsonl")
    ap.add_argument("--out", default="/tmp/coverage_videos")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--concat", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.gate)]
    rows = [r for r in rows if r["outcome"] == "win"]
    universe, chosen = cover(rows)
    os.makedirs(a.out, exist_ok=True)

    print("%d winning demos; %d entity types to cover: %s\n"
          % (len(rows), len(universe), sorted(universe)))
    print("cover = %d demos:" % len(chosen))
    for r, got in chosen:
        print("   %-10s %5d frames  first shows: %s"
              % (r["f"], r["frames"], ", ".join("%d %s" % (t, NAMES.get(t, "?")) for t in got)))
    print()

    reel = None
    total = 0
    for i, (r, got) in enumerate(chosen, 1):
        demo = r["f"]
        out = os.path.join(a.out, "%02d_%s.mp4" % (i, demo))
        label = "%s  |  new: %s" % (demo, ", ".join(NAMES.get(t, str(t)) for t in got))
        print("[%d/%d] %s" % (i, len(chosen), demo))
        n, won = render(demo, out, a.fps, label=label)
        total += n
        print("   %d frames, won=%s -> %s" % (n, won, out))
        if a.concat:
            if reel is None:
                cap = cv2.VideoCapture(out)
                ok, fr = cap.read()
                cap.release()
                hgt, wid = fr.shape[:2]
                reel = cv2.VideoWriter(os.path.join(a.out, "coverage_reel.mp4"),
                                       cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (wid, hgt))
            cap = cv2.VideoCapture(out)
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                reel.write(fr)
            cap.release()
    if reel is not None:
        reel.release()
        print("\nreel: %s" % os.path.join(a.out, "coverage_reel.mp4"))
    print("\n%d frames total = %.1f s at %d fps" % (total, total / a.fps, a.fps))


if __name__ == "__main__":
    main()
