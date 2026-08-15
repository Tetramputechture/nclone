"""What is the gun lerping toward? Small scope so it finishes.

`target = prev + (actual - prev) / 0.2` recovers the target exactly on every
frame the gun moves. Test it against every candidate angle the slab offers.
"""
import collections
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
CASES = [("out9", "10131718", 10), ("out9", "10135227", 10),
         ("out10", "10138659", 10), ("out10", "10139152", 10)]
ROOT = "/home/tetra/projects/nclone/tools/npptrace"
hits = collections.defaultdict(collections.Counter)
samples = []


def wrap(d):
    while d >= math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


for sub, demo, pool in CASES:
    cap = Capture(os.path.join(ROOT, sub, demo))
    p = next((q for q in cap.pools if q["i"] == pool and q["count"]), None)
    if p is None:
        continue
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    dec = NppAttractDecoder()
    path = "/home/tetra/projects/attracts/" + demo
    inputs = dec.decode_npp_attract_file(path).get("demo_data", []) or []
    h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    h.load_map_from_map_data(list(dec.create_nclone_map(path)))
    prev = {}
    for f in range(1, min(cap.n, len(inputs)) + 1):
        hor, jmp = decode_input_to_controls(int(inputs[f - 1]))
        h.tick(hor, jmp)
        if h.ninja_has_died():
            break
        ninja = h.sim.ninja
        rec = cap.frame(f)
        for i in range(p["count"]):
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            gun = struct.unpack_from("<f", b, 0x50)[0]
            st = struct.unpack_from("<i", b, 0x7C)[0]
            gx = struct.unpack_from("<d", b, 0x10)[0]
            gy = struct.unpack_from("<d", b, 0x18)[0]
            e0 = struct.unpack_from("<f", b, 0xE0)[0]
            key = (demo, i)
            if key in prev:
                pgun, _ = prev[key]
                step = wrap(gun - pgun)
                bearing = math.atan2(ninja.ypos - gy, ninja.xpos - gx)
                pred = pgun + wrap(bearing - pgun) * LERP
                err = abs(wrap(pred - gun))
                for tol, name in ((2e-4, "1e-4"), (1e-3, "1e-3"),
                                  (5e-3, "5e-3"), (2e-2, "2e-2")):
                    if err < tol:
                        hits[st]["forward model within " + name] += 1
                        break
                else:
                    hits[st]["forward model FAILS"] += 1
                    if len(samples) < 8:
                        samples.append((demo, f, st, round(err, 5)))
            prev[key] = (gun, st)

print("forward tracking model vs +0x50, %d demos:" % len(CASES))
for st in sorted(hits):
    tot = sum(hits[st].values())
    print("  state %d (%d frames)" % (st, tot))
    for k, v in hits[st].most_common(6):
        print("      %-16s %5d  %5.1f%%" % (k, v, 100.0 * v / max(tot, 1)))
print("\nunexplained samples (demo, frame, state, prev, TARGET, bearing, +0xE0):")
for s in samples:
    print("   %s" % (s,))
