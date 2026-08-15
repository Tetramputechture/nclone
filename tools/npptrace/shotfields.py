"""Is the chaingun's shot a travelling projectile rather than a hitscan?

Everything about the discharge is verified (10/10 discharges, 20 shots at 4-frame
spacing 969/969, sweep step and sign) yet turning the model on produces 88 false
kills. If the game's shot TRAVELS, a hitscan test would over-kill exactly like
that -- the ninja gets time to move out of the line.

Dump every double in the chaingun slab through a discharge and look for a pair
that moves like a projectile: starts at the drone and walks away at a constant
step. The rocket already showed projectiles can live inline in the owner's slab.
"""
import glob
import json
import math
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

POOL = 10  # chaingun (12)
shown = 0

for s in sorted(glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json")):
    stem = s.replace(".layout.json", "")
    nj = stem + ".ninja.jsonl"
    if not os.path.exists(nj):
        continue
    try:
        cap = Capture(stem)
    except Exception:
        continue
    p = next((q for q in cap.pools if q["i"] == POOL and q["count"]), None)
    if p is None:
        continue
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    stride = p["stride"]
    # Find a discharge: state (+0x7C) == 2.
    prev_state = None
    for f in range(1, cap.n + 1):
        b = cap.frame(f)[base: base + stride]
        st = struct.unpack_from("<i", b, 0x7C)[0]
        if prev_state != 2 and st == 2 and shown < 2:
            shown += 1
            gx, gy = struct.unpack_from("<dd", b, 0x10)
            print("\n=== %s discharge starts frame %d, drone at (%.2f, %.2f)"
                  % (os.path.basename(stem), f, gx, gy))
            # Print every 8-byte double slot for the first 12 frames of it.
            offs = [o for o in range(0, stride - 7, 8)]
            print("  off   " + "  ".join("f%-9d" % (f + k) for k in range(0, 10, 2)))
            for o in offs:
                vals = []
                ok = True
                for k in range(0, 10, 2):
                    if f + k > cap.n:
                        ok = False
                        break
                    bb = cap.frame(f + k)[base: base + stride]
                    v = struct.unpack_from("<d", bb, o)[0]
                    if not math.isfinite(v):
                        ok = False
                        break
                    vals.append(v)
                if not ok or all(abs(v - vals[0]) < 1e-12 for v in vals):
                    continue  # constant or unreadable -- not a projectile
                print("  0x%02X  %s" % (o, "  ".join("%10.3f" % v for v in vals)))
        prev_state = st
        if shown >= 2:
            break
    if shown >= 2:
        break
print("\n(only slots that CHANGE during the discharge are listed)")
