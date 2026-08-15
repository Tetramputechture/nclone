"""Does the discharge sweep's SIGN follow the map's mode byte?

The dual laser takes its spin sign from `mode` the same way (§15), so it is the
first candidate. Map load only -- no replay -- so this is fast.
"""
import collections
import glob
import math
import os
import struct
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

POOLS = (10, 11)
tab = collections.Counter()


def wrap(d):
    while d >= math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


for d in ["/home/tetra/projects/nclone/tools/npptrace/out9",
          "/home/tetra/projects/nclone/tools/npptrace/out9_batch1",
          "/home/tetra/projects/nclone/tools/npptrace/out10",
          "/home/tetra/projects/nclone/tools/npptrace/out11"]:
    for s in sorted(glob.glob(d + "/*.layout.json")):
        demo = os.path.basename(s).replace(".layout.json", "")
        if demo.startswith("cap"):
            continue
        cap = Capture(os.path.join(d, demo))
        pools = [q for q in cap.pools if q["i"] in POOLS and q["count"]]
        if not pools:
            continue
        dec = NppAttractDecoder()
        h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
        h.load_map_from_map_data(
            list(dec.create_nclone_map("/home/tetra/projects/attracts/" + demo)))
        ents = collections.defaultdict(list)
        for lst in h.sim.entity_dic.values():
            for e in lst:
                if getattr(e, "type", None) in (12, 13):
                    ents[e.type].append(e)
        for p in pools:
            drones = ents.get(p["i"] + 2, [])
            base = 4 + cap.layout["ninjaSlab"] + p["offset"]
            cur = collections.defaultdict(list)
            last = {}
            for f in range(1, cap.n + 1):
                rec = cap.frame(f)
                for i in range(min(p["count"], len(drones))):
                    b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
                    st = struct.unpack_from("<i", b, 0x7C)[0]
                    gun = struct.unpack_from("<f", b, 0x50)[0]
                    key = (demo, p["i"], i)
                    if st == 2:
                        if last.get(key) != gun:
                            cur[key].append(gun)
                            last[key] = gun
                    else:
                        seq = cur.pop(key, [])
                        last.pop(key, None)
                        if len(seq) >= 4:
                            steps = [wrap(seq[j + 1] - seq[j])
                                     for j in range(1, len(seq) - 1)]
                            mean = sum(steps) / len(steps)
                            sign = "+" if mean > 0 else "-"
                            e = drones[i]
                            mode = getattr(e, "mode", None)
                            orient = getattr(e, "orientation", None)
                            tab[("mode", mode, sign)] += 1
                            tab[("orient", orient, sign)] += 1
                            tab[("initdir", getattr(e, "initial_dir",
                                                    getattr(e, "dir", None)), sign)] += 1

print("sweep sign against each candidate:")
for field in ("mode", "orient", "initdir"):
    print("  %s:" % field)
    keys = sorted({k[1] for k in tab if k[0] == field}, key=str)
    for v in keys:
        plus = tab[(field, v, "+")]
        minus = tab[(field, v, "-")]
        tot = plus + minus
        if not tot:
            continue
        print("     value %-6s  +%-4d -%-4d   %.0f%% consistent"
              % (v, plus, minus, 100.0 * max(plus, minus) / tot))
