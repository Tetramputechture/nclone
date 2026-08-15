"""Why does the native drone's shot reach a ninja the Python one cannot see?

On `10244473` the ninja is pinned at (406, 302), a type-12 shooter sits at
(444, 300), and a regular door sits at (432, 300) squarely between them. Native
kills on frame 58 with `drone_shot`; Python plays on to a win. Either the door
is closed in one engine and open in the other, or it is closed in both and only
one raycast stops on it.

    doorbeam.py <demo> <frame>
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone")

import nclone_native  # noqa: E402

from nclone.nsim import Simulator  # noqa: E402
from nclone.sim_config import SimConfig  # noqa: E402
from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402
from nclone.physics import get_raycast_distance  # noqa: E402

demo = sys.argv[1]
upto = int(sys.argv[2])

dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % demo
tape = dec.decode_npp_attract_file(path).get("demo_data") or []
md = list(dec.create_nclone_map(path))

py = Simulator(SimConfig())
py.load(md)
nat = nclone_native.Simulator.load(md)
for b in tape[:upto]:
    hor, jmp = decode_input_to_controls(int(b))
    py.tick(hor, jmp)
    nat.tick(hor, jmp)

print("=== frame %d ===" % upto)
n = py.ninja
print("python ninja (%.4f, %.4f) state=%s" % (n.xpos, n.ypos, n.state))
nn = nat.get_state_dynamic().ninja
print("native ninja (%.4f, %.4f) state=%s" % (nn.xpos, nn.ypos, int(nn.state)))

print("\nnative doors:")
for e in nat.get_state_dynamic().entities:
    if e.entity_type in (5, 6, 8):
        print("   type=%d switch=(%.0f,%.0f) door=(%.0f,%.0f) orient=%s closed=%s active=%s"
              % (e.entity_type, e.x, e.y, e.door_x, e.door_y,
                 e.door_orientation, e.door_closed, e.active))

print("\npython doors:")
for t in (5, 6, 8):
    for e in py.entity_dic.get(t, []):
        print("   type=%d switch=(%.0f,%.0f) door=(%.0f,%.0f) orient=%s closed=%s active=%s"
              % (t, e.xpos, e.ypos, getattr(e, "door_xpos", -1),
                 getattr(e, "door_ypos", -1), getattr(e, "orientation", None),
                 getattr(e, "closed", None), e.active))

print("\npython shooters and their beams:")
for t, ents in sorted(py.entity_dic.items()):
    for e in ents:
        if not hasattr(e, "aim_dx"):
            continue
        d = get_raycast_distance(py, e.xpos, e.ypos, e.aim_dx, e.aim_dy)
        print("   type=%d at (%.0f,%.0f) state=%s aim=(%+.2f,%+.2f) raycast=%s"
              % (t, e.xpos, e.ypos, getattr(e, "state", None),
                 e.aim_dx, e.aim_dy, d))
        if d:
            print("      beam ends at (%.3f, %.3f)"
                  % (e.xpos + e.aim_dx * d, e.ypos + e.aim_dy * d))

print("\npython segments active in the cells between ninja and shooter:")
for key, segs in sorted(py.segment_dic.items()):
    if not segs:
        continue
    for s in segs:
        x1 = getattr(s, "x1", None)
        if x1 is None:
            continue
        if 400 <= x1 <= 460 and 280 <= getattr(s, "y1", 0) <= 320:
            print("   cell %s (%.0f,%.0f)-(%.0f,%.0f) active=%s oriented=%s"
                  % (key, s.x1, s.y1, s.x2, s.y2, s.active,
                     getattr(s, "oriented", None)))
