"""The zap drones are bit-exact, yet we die to one. Ninja, or contact test?

The capture carries the game's own ninja track, so this is decidable: find the
first frame our ninja leaves the game's, and measure the separation at the kill.
A kill at almost exactly RADIUS+NINJA_RADIUS means the test is marginal; a kill
deep inside means our ninja is somewhere else.
"""
import glob
import json
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
from nclone.ninja import NINJA_RADIUS  # noqa: E402
from nclone.constants.physics_constants import DRONE_RADIUS  # noqa: E402

DEMO, POOL = sys.argv[1], int(sys.argv[2])
ETYPE = POOL + 2
stem = next(os.path.join(c, DEMO)
            for c in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*")
            if os.path.exists(os.path.join(c, DEMO) + ".layout.json"))
cap = Capture(stem)
p = next(q for q in cap.pools if q["i"] == POOL and q["count"])
base = 4 + cap.layout["ninjaSlab"] + p["offset"]
gninja = [json.loads(l) for l in open(stem + ".ninja.jsonl") if l.strip()]

dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % DEMO
tape = dec.decode_npp_attract_file(path).get("demo_data") or []
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(list(dec.create_nclone_map(path)))
sim = h.sim
ours = sim.entity_dic.get(ETYPE, [])
print("%s: NINJA_RADIUS=%s DRONE_RADIUS=%s sum=%s"
      % (DEMO, NINJA_RADIUS, DRONE_RADIUS, NINJA_RADIUS + DRONE_RADIUS))

njfirst = None
for f in range(1, min(cap.n, len(tape), len(gninja)) + 1):
    hor, jmp = decode_input_to_controls(int(tape[f - 1]))
    h.tick(hor, jmp)
    n, g = sim.ninja, gninja[f - 1]
    if njfirst is None and (abs(n.xpos - g["xpos"]) > 1e-9
                            or abs(n.ypos - g["ypos"]) > 1e-9):
        njfirst = f
        print("\nNINJA first leaves the game's track at frame %d" % f)
        print("   game (%.10f, %.10f)  ours (%.10f, %.10f)  d=(%.3e, %.3e)"
              % (g["xpos"], g["ypos"], n.xpos, n.ypos,
                 n.xpos - g["xpos"], n.ypos - g["ypos"]))
    if h.ninja_has_died():
        rec = cap.frame(f)
        print("\nWE DIE at frame %d of %d" % (f, min(cap.n, len(tape))))
        print("   our ninja  (%.4f, %.4f)" % (n.xpos, n.ypos))
        print("   game ninja (%.4f, %.4f)" % (g["xpos"], g["ypos"]))
        for i in range(p["count"]):
            b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
            gx, gy = struct.unpack_from("<dd", b, 0x10)
            d_ours = ((n.xpos - gx) ** 2 + (n.ypos - gy) ** 2) ** 0.5
            d_game = ((g["xpos"] - gx) ** 2 + (g["ypos"] - gy) ** 2) ** 0.5
            mark = "  <== KILLS US" if d_ours < NINJA_RADIUS + DRONE_RADIUS else ""
            print("   drone %d at (%.2f, %.2f)  dist ours=%.4f  game=%.4f%s"
                  % (i, gx, gy, d_ours, d_game, mark))
        break
    if h.ninja_has_won():
        print("\n(we win at frame %d)" % f)
        break
else:
    print("\n(tape ended, no death)")
if njfirst is None:
    print("\nninja never left the game's track")
