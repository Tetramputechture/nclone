"""First divergence of a captured laser, at full precision.

`entscore.py 21 --list` says which laser in which demo leaves the game and on
what frame. This says *by how much*, which is the difference between a wrong
rule (diverges by pixels on frame 1) and an arithmetic-order bug (diverges by
one ULP and amplifies). Prints every laser's per-frame delta around the first
divergence.

    laserdiv.py <demo> [laser_index]
"""
import glob
import os
import struct
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone")
sys.path.insert(0, "/home/tetra/projects/nclone/tools/npptrace")
from read_entities import Capture  # noqa: E402

from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.nplay_headless import NPlayHeadless  # noqa: E402

demo = sys.argv[1]
only = int(sys.argv[2]) if len(sys.argv) > 2 else None

stem = (glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/%s.layout.json" % demo)
        + glob.glob("/home/tetra/projects/nclone/docs/game_traces/%s.layout.json" % demo))
cap = Capture(stem[0].replace(".layout.json", ""))
p = next(q for q in cap.pools if q["i"] == 21 and q["count"])
base = 4 + cap.layout["ninjaSlab"] + p["offset"]

dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % demo
md = list(dec.create_nclone_map(path))
tape = dec.decode_npp_attract_file(path).get("demo_data")

h = NPlayHeadless(render_mode="rgb_array", enable_rendering=False)
h.load_map_from_map_data(md)
ours = h.sim.entity_dic.get(23, [])

n = min(cap.n, len(tape))
first = {}
for f in range(1, n + 1):
    hor, jmp = decode_input_to_controls(int(tape[f - 1]))
    h.tick(hor, jmp)
    row = cap.frame(f)
    for i, e in enumerate(ours):
        if i >= p["count"] or (only is not None and i != only) or i in first:
            continue
        b = row[base + i * p["stride"]: base + (i + 1) * p["stride"]]
        gx, gy = struct.unpack_from("<dd", b, 0x10)
        if e.xpos != gx or e.ypos != gy:
            first[i] = (f, gx, gy, e.xpos, e.ypos,
                        struct.unpack_from("<d", b, 0xB0)[0],
                        struct.unpack_from("<dd", b, 0x68))
    if len(first) == (1 if only is not None else p["count"]):
        break

print("%s: %d lasers, %d frames" % (demo, p["count"], n))
for i in sorted(first):
    f, gx, gy, ox, oy, gbeam, gn = first[i]
    print("\nlaser %d first differs on frame %d" % (i, f))
    print("   game pos  %.17g , %.17g" % (gx, gy))
    print("   ours pos  %.17g , %.17g" % (ox, oy))
    print("   delta     %+.4e , %+.4e" % (ox - gx, oy - gy))
    print("   game normal (%.17g, %.17g)  beam %.17g" % (gn[0], gn[1], gbeam))
    print("   ours normal (%.17g, %.17g)  beam %.17g"
          % (ours[i].xnormal, ours[i].ynormal, ours[i].beam_angle))
    print("   ours anchor (%.17g, %.17g)  mode %s"
          % (ours[i].xanchor, ours[i].yanchor, ours[i].mode))
if not first:
    print("   all lasers bit-exact for the whole demo")
