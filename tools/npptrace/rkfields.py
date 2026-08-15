"""Read the game's own rocket-launcher fields out of every capture.

FUN_10412780 / FUN_104111A0 field map:
    +0x20 max speed (ctor)      +0x28 initial accel (ctor)
    +0x30 accel multiplier 1.1  +0x38 homing rate (ctor)
    +0x40 launch delay, frames (ctor)   +0x48 aim lead on ninja velocity (ctor)
    +0x50/+0x58 projectile pos  +0x60/+0x68 direction
    +0x70 speed                 +0x78 accel     +0x80 distance
    +0x88 charge                +0x90 state     +0x94 target index
"""
import collections
import glob
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

D = lambda b, o: struct.unpack_from("<d", b, o)[0]
I = lambda b, o: struct.unpack_from("<i", b, o)[0]

ctor = collections.Counter()
launch_at = collections.Counter()
accel0 = collections.Counter()
state_hist = collections.Counter()
seen = 0

stems = sorted(set(
    s.replace(".layout.json", "")
    for s in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")
    + glob.glob("/home/tetra/projects/nclone/docs/game_traces/*.layout.json")))

for stem in stems:
    try:
        cap = Capture(stem)
    except Exception:
        continue
    p = next((q for q in cap.pools if q["i"] == 16 and q["count"]), None)
    if p is None or p["stride"] < 0x98:
        continue
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    seen += 1
    prev = {}
    for f in range(1, cap.n + 1):
        rec = cap.frame(f)
        for i in range(p["count"]):
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            ctor[("+0x20 maxspeed", D(b, 0x20))] += 1
            ctor[("+0x28 accel0", D(b, 0x28))] += 1
            ctor[("+0x30 accelmul", D(b, 0x30))] += 1
            ctor[("+0x38 homing", D(b, 0x38))] += 1
            ctor[("+0x40 delay", D(b, 0x40))] += 1
            ctor[("+0x48 lead", D(b, 0x48))] += 1
            st, ch = I(b, 0x90), D(b, 0x88)
            state_hist[st] += 1
            k = (stem, i)
            if k in prev:
                pst, pch = prev[k]
                if pst == 1 and st == 2:
                    launch_at[pch] += 1
                    accel0[round(D(b, 0x78), 12)] += 1
            prev[k] = (st, ch)

print("captures with a type-18 pool: %d\n" % seen)
print("constructor fields (value -> frames observed):")
for (name, v), c in sorted(ctor.items()):
    print("   %-18s %-24.17g  x%d" % (name, v, c))
print("\nstate histogram: %s" % dict(sorted(state_hist.items())))
print("\ncharge on the frame BEFORE state 1 -> 2 (i.e. the launch threshold):")
for v, c in sorted(launch_at.items()):
    print("   charge=%-8.17g  x%d" % (v, c))
print("\n+0x78 accel on the first FLYING frame:")
for v, c in sorted(accel0.items()):
    print("   %-24.17g x%d" % (v, c))
