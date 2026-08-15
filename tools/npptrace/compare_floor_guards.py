"""Diff nclone's floor guards against a game capture, frame by frame.

    python /tmp/fg_cmp.py <demo-id> <capture-stem> [first] [last]

Floor guard is type 16 -> pool 14, stride 0x48. Fields per the ctor
(FUN_1040B730) and move (FUN_1040B900):
    +0x10/+0x18 pos   +0x20 speed   +0x28 offset   +0x30 dir
    +0x34 render-float   +0x38 step2   +0x40 just_activated
"""
import gzip, json, os, struct, sys, math
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

demo, stem = sys.argv[1], sys.argv[2]
f0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1
f1 = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
POOL, TYPE = 14, 16

L = json.load(open(stem + ".layout.json"))
rec, slab = L["recBytes"], L["ninjaSlab"]
data = gzip.open(stem + ".ent.bin.gz", "rb").read()
n = len(data) // rec
p = next((q for q in L["pools"] if q["i"] == POOL), None)
if p is None:
    raise SystemExit("capture has no pool %d; pools: %s" % (POOL, [q["i"] for q in L["pools"]]))
base = 4 + slab + p["offset"]
D = lambda r, o: struct.unpack_from("<d", r, o)[0]
I = lambda r, o: struct.unpack_from("<i", r, o)[0]

from nclone.replay.npp_attract_decoder import NppAttractDecoder
from nclone.replay.replay_executor import decode_input_to_controls
from nclone.nplay_headless import NPlayHeadless
dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/" + demo
inputs = dec.decode_npp_attract_file(path).get("demo_data", []) or []
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(list(dec.create_nclone_map(path)))
sim = h.sim
mine = [e for l in sim.entity_dic.values() for e in l if getattr(e, "type", None) == TYPE]

r1 = data[0:rec]
gpos = [(D(r1, base + i * p["stride"] + 0x10), D(r1, base + i * p["stride"] + 0x18))
        for i in range(p["count"])]
print("capture frames=%d  guards: game %d, nclone %d" % (n, p["count"], len(mine)))
print("  game frame-1 positions:", [("%.1f,%.1f" % g) for g in gpos])
print("  nclone spawn         :", [("%.1f,%.1f" % (e.xpos, e.ypos)) for e in mine])
order = [min(range(len(mine)), key=lambda j: math.hypot(mine[j].xpos - g[0], mine[j].ypos - g[1]))
         for g in gpos]
print("  ctor consts (game): speed=%.6f offset=%.3f  (nclone %.6f / %.3f)\n"
      % (D(r1, base + 0x20), D(r1, base + 0x28), mine[0].SPEED, mine[0].OFFSET))

first_bad = None
lim = min(n, len(inputs), f1)
for f in range(1, lim + 1):
    hor, jmp = decode_input_to_controls(int(inputs[f - 1]))
    h.tick(hor, jmp)
    rows, bad = [], False
    for gi in range(p["count"]):
        o = base + gi * p["stride"]
        r = data[(f - 1) * rec: f * rec]
        gx, gdir = D(r, o + 0x10), I(r, o + 0x30)
        e = mine[order[gi]]
        if abs(gx - e.xpos) > 1e-9 or gdir != e.dir:
            bad = True
        rows.append("g(%9.4f d=%2d) n(%9.4f d=%2d)" % (gx, gdir, e.xpos, e.dir))
    if bad and first_bad is None:
        first_bad = f
        print("FIRST DIVERGENCE at frame %d" % f)
    if (bad and f <= (first_bad or 0) + 6) or f0 <= f <= min(f0 + 25, f1):
        print("f%-5d ninja g=(%9.4f,%9.4f) n=(%9.4f,%9.4f) st=%d | %s" % (
            f, D(data[(f-1)*rec:f*rec], 4 + 0x20), D(data[(f-1)*rec:f*rec], 4 + 0x28),
            sim.ninja.xpos, sim.ninja.ypos, sim.ninja.state, "  ".join(rows)))
if first_bad is None:
    print("\nfloor guards BIT-EXACT for %d frames" % lim)
