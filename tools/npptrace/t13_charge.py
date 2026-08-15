"""Does the LASER drone's gun track the ninja during its charge?

Every earlier measurement of this pooled types 12 and 13, which §26 established
is invalid. This one is type 13 only, forward-scored from the game's own previous
angle and its own drone position.
"""
import collections
import glob
import json
import math
import os
import struct
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

LERP = 0.2
POOL = 11
hits = collections.defaultdict(collections.Counter)


def wrap(d):
    while d >= math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


ids = open("/tmp/capture_ids4.txt").read().split()
stems = sorted(glob.glob("/home/tetra/projects/nclone/tools/npptrace/out13/*.layout.json"))
# The captures are still capNN; pair each with a demo by replaying candidates is
# expensive, so instead drive the ninja from the capture's OWN ninja.jsonl.
for s in stems:
    stem = s.replace(".layout.json", "")
    nj = stem + ".ninja.jsonl"
    if not os.path.exists(nj):
        continue
    ninja = [json.loads(l) for l in open(nj) if l.strip()]
    cap = Capture(stem)
    p = next((q for q in cap.pools if q["i"] == POOL and q["count"]), None)
    if p is None:
        continue
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    prev = {}
    for f in range(1, min(cap.n, len(ninja)) + 1):
        rec = cap.frame(f)
        n = ninja[f - 1]
        for i in range(p["count"]):
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            gun = struct.unpack_from("<f", b, 0x50)[0]
            st = struct.unpack_from("<i", b, 0x7C)[0]
            gx = struct.unpack_from("<d", b, 0x10)[0]
            gy = struct.unpack_from("<d", b, 0x18)[0]
            key = (stem, i)
            if key in prev:
                pgun, pst = prev[key]
                bearing = math.atan2(n["ypos"] - gy, n["xpos"] - gx)
                pred = pgun + wrap(bearing - pgun) * LERP
                err = abs(wrap(pred - gun))
                held = abs(wrap(gun - pgun)) < 1e-6
                if held:
                    hits[pst]["holds still"] += 1
                elif err < 2e-2:
                    hits[pst]["tracks the ninja"] += 1
                else:
                    hits[pst]["moves, unexplained"] += 1
            prev[key] = (gun, st)

print("type 13 gun behaviour, by the state it was in (ninja from the capture):")
for st in sorted(hits):
    tot = sum(hits[st].values())
    print("  state %d (%d frames)" % (st, tot))
    for k, v in hits[st].most_common(4):
        print("      %-20s %6d  %5.1f%%" % (k, v, 100.0 * v / max(tot, 1)))
