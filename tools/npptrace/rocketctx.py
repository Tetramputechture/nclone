"""The rocket projectile, ours vs the game's, around our death.

Usage: rocketctx.py <demo> [span]
Fields (see rocketproj.py): +0x50/+0x58 projectile pos, +0x60/+0x68 direction,
+0x70 speed, +0x80 distance travelled, +0x30 state.
"""
import glob
import math
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
SPAN = int(sys.argv[2]) if len(sys.argv) > 2 else 12
POOL, ETYPE = 16, 18

caps = {}
for s in glob.glob("/home/tetra/projects/nclone/tools/npptrace/out*/*.layout.json"):
    caps[os.path.basename(s).replace(".layout.json", "")] = s.replace(".layout.json", "")
for s in glob.glob("/home/tetra/projects/nclone/docs/game_traces/*.layout.json"):
    caps.setdefault(os.path.basename(s).replace(".layout.json", ""), s.replace(".layout.json", ""))

cap = Capture(caps[demo])
p = next(q for q in cap.pools if q["i"] == POOL and q["count"])
base = 4 + cap.layout["ninjaSlab"] + p["offset"]

dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % demo
tape = dec.decode_npp_attract_file(path).get("demo_data") or []
h = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
h.load_map_from_map_data(list(dec.create_nclone_map(path)))
nj = h.sim.ninja
ours = h.sim.entity_dic.get(ETYPE, [])
print("%s: %d launchers (game %d), stride 0x%x" % (demo, len(ours), p["count"], p["stride"]))
e0 = ours[0]
print("our launcher attrs: %s\n"
      % sorted(k for k in vars(e0) if not k.startswith("_")))

hist = []
for f in range(1, min(cap.n, len(tape)) + 1):
    hor, jmp = decode_input_to_controls(int(tape[f - 1]))
    h.tick(hor, jmp)
    rec = cap.frame(f)
    nx = struct.unpack_from("<d", rec, 4 + 0x20)[0]
    ny = struct.unpack_from("<d", rec, 4 + 0x28)[0]
    row = []
    for i in range(p["count"]):
        b = rec[base + i * p["stride"]: base + (i + 1) * p["stride"]]
        st = struct.unpack_from("<i", b, 0x30)[0]
        px, py = struct.unpack_from("<dd", b, 0x50)
        dxx, dyy = struct.unpack_from("<dd", b, 0x60)
        spd = struct.unpack_from("<d", b, 0x70)[0]
        dist = struct.unpack_from("<d", b, 0x80)[0]
        o = ours[i] if i < len(ours) else None
        row.append((st, px, py, dxx, dyy, spd, dist, o))
    hist.append((f, nx, ny, nj.xpos, nj.ypos, row))
    if h.ninja_has_died() or h.ninja_has_won():
        print("ended f=%d died=%s won=%s\n" % (f, h.ninja_has_died(), h.ninja_has_won()))
        break

for f, gnx, gny, onx, ony, row in hist[-SPAN:]:
    print("f=%-5d ninja game (%8.3f,%8.3f) ours (%8.3f,%8.3f) d=%.3f"
          % (f, gnx, gny, onx, ony, math.hypot(onx - gnx, ony - gny)))
    for i, (st, px, py, dxx, dyy, spd, dist, o) in enumerate(row):
        if st == 0 and px == 0 and py == 0:
            continue
        opx = getattr(o, "xpos_proj", None)
        if opx is None:
            for cand in ("proj_x", "projectile_x", "shot_x"):
                if hasattr(o, cand):
                    opx = getattr(o, cand)
                    break
        print("   L%d st=%-3d game proj (%8.3f,%8.3f) dir(%7.4f,%7.4f) spd=%-7.4g dist=%-9.4g"
              "  d_to_gninja=%.3f"
              % (i, st, px, py, dxx, dyy, spd, dist, math.hypot(gnx - px, gny - py)))
        if o is not None:
            print("        ours st=%-3s %s"
                  % (getattr(o, "state", None),
                     {k: round(v, 4) for k, v in vars(o).items()
                      if isinstance(v, float) and k not in ("xpos", "ypos",
                                                            "_initial_xpos", "_initial_ypos")}))
