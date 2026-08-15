"""Map captured slots to attract ids via the ninja spawn stored in the map data."""
import glob
import json
import os
import struct

ATTRACT = r"D:\attracts"
IDS = ["10131255", "10138317", "10138288", "10131738"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Known from nclone's attract_info.json for 10131738:
#   ninja_spawn_raw [159, 10]  ->  pixels [954, 60]   (raw * 6)
KNOWN = ("10131738", 159, 10)


def mapsec(i):
    b = open(os.path.join(ATTRACT, i), "rb").read()
    n = struct.unpack_from("<I", b, 0)[0]
    return b[8:8 + n]


# 1. locate the spawn offset using the known file
m = mapsec(KNOWN[0])
offs = [k for k in range(len(m) - 1) if m[k] == KNOWN[1] and m[k + 1] == KNOWN[2]]
print("candidate spawn offsets in %s: %s" % (KNOWN[0], [hex(o) for o in offs]))

# 2. the offset must yield an in-bounds spawn for every file
good = []
for o in offs:
    ok = True
    for i in IDS:
        mm = mapsec(i)
        if o + 1 >= len(mm):
            ok = False
            break
        x, y = mm[o], mm[o + 1]
        if not (0 < x < 44 * 4 and 0 < y < 25 * 4):
            ok = False
            break
    if ok:
        good.append(o)
print("offsets valid across all four: %s" % [hex(o) for o in good])

for o in good:
    print("  -- using offset %s --" % hex(o))
    for i in IDS:
        mm = mapsec(i)
        print("     %s spawn_raw=(%d, %d) pixels=(%d, %d)"
              % (i, mm[o], mm[o + 1], mm[o] * 6, mm[o + 1] * 6))

# 3. match against captured slots
slots = {}
for f in glob.glob(os.path.join(OUT, "slot*.jsonl")):
    L = [json.loads(x) for x in open(f)]
    slots[os.path.basename(f)] = (len(L), L[0]["xpos"], L[0]["ypos"], L)
print("\ncaptured slots:")
for k in sorted(slots, key=lambda p: int(p[4:-6])):
    n, x, y, _ = slots[k]
    print("   %-14s frames=%4d spawn=(%.1f, %.1f)" % (k, n, x, y))
