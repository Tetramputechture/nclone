"""Find the pool the chaingun's shot lives in, if it is captured at all.

§17 established the discharge exactly -- 20 shots, one every 4 frames, along a
frozen direction -- but could not find what carries the shot. Scan EVERY pool
for slabs that change on or just after a fire frame, rather than guessing which
type it is.
"""
import collections
import os
import struct
import sys

sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

TRACES = os.environ.get("NCLONE_TRACES",
                        "/home/tetra/projects/nclone/tools/npptrace/out10")
demo = sys.argv[1]
cap = Capture(os.path.join(TRACES, demo))
pools = {p["i"]: p for p in cap.pools if p["count"]}
print("%s: %d frames, pools %s"
      % (demo, cap.n, sorted((p + 2, pools[p]["count"]) for p in pools)))

# Find the frames a shooter drone launches a shot: state 2 with sub-counter 0.
fire_frames = []
for pool in (10, 11):
    p = pools.get(pool)
    if p is None:
        continue
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    for f in range(1, cap.n + 1):
        rec = cap.frame(f)
        for i in range(p["count"]):
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            if (struct.unpack_from("<i", b, 0x7C)[0] == 2
                    and struct.unpack_from("<i", b, 0x8C)[0] == 0):
                fire_frames.append(f)
fire_frames = sorted(set(fire_frames))
print("shots launched on %d frames; first few %s"
      % (len(fire_frames), fire_frames[:6]))
if not fire_frames:
    raise SystemExit("no shots in this capture")

# For every pool, how many bytes change from the frame before a shot to the
# frames just after it?
changed = collections.Counter()
for pool, p in sorted(pools.items()):
    base = 4 + cap.layout["ninjaSlab"] + p["offset"]
    size = p["count"] * p["stride"]
    for f in fire_frames[:40]:
        if f + 2 > cap.n or f < 2:
            continue
        a = cap.frame(f - 1)[base:base + size]
        for d in (0, 1, 2):
            b = cap.frame(f + d)[base:base + size]
            if a != b:
                changed[pool] += 1
                break

print("\n%-8s %-8s %-8s %s" % ("pool", "type", "count", "shots where it changed"))
for pool, p in sorted(pools.items()):
    print("%-8d %-8d %-8d %d / %d"
          % (pool, pool + 2, p["count"], changed[pool], min(len(fire_frames), 40)))
