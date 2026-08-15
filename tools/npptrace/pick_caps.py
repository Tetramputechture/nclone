"""Choose capture candidates: current losers, one per distinct map.

§3's warning is that two demos from the same map in one batch cannot be told
apart afterwards, so key on the map bytes themselves rather than trusting the
type list.
"""
import collections
import hashlib
import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402

WANT = {"laser": 12, "drone_shot": 8}
rows = [json.loads(l) for l in open("/tmp/gate8.jsonl")]
losers = [r for r in rows if r.get("outcome") == "die" and r.get("cause") in WANT]
print("losers available: %s"
      % collections.Counter(r["cause"] for r in losers).most_common())

dec = NppAttractDecoder()
seen = set()
picked = collections.defaultdict(list)
for r in sorted(losers, key=lambda r: r["frames"]):
    cause = r["cause"]
    if len(picked[cause]) >= WANT[cause]:
        continue
    path = "/home/tetra/projects/attracts/" + r["f"]
    try:
        mapb = bytes(bytearray(x & 0xFF for x in dec.create_nclone_map(path)))
    except Exception:
        continue
    key = hashlib.sha1(mapb).hexdigest()
    if key in seen:
        continue
    seen.add(key)
    picked[cause].append((r["f"], r["frames"]))

for cause, items in picked.items():
    print("\n%s (%d):" % (cause, len(items)))
    for demo, frames in items:
        print("   %s  dies at %d" % (demo, frames))

ids = [d for items in picked.values() for d, _ in items]
open("/tmp/capture_ids.txt", "w").write(" ".join(ids) + "\n")
print("\n%d ids written to /tmp/capture_ids.txt" % len(ids))
print(" ".join(ids))
