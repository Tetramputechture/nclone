"""Step both engines on one demo and report the first frame they disagree.

111 demos in the failing corpus are won by the Python engine and lost by the
native one. Python is the side that has been scored bit-exact against the game
captures, so on those demos native is wrong -- and finding where it goes wrong
needs no capture at all, just the two ports side by side.

Reports the first frame where any ninja field differs, with enough state either
side of it to see what happened.

    lockstep.py <demo> [max_frames]
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

demo = sys.argv[1]
maxf = int(sys.argv[2]) if len(sys.argv) > 2 else 0

dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % demo
tape = dec.decode_npp_attract_file(path).get("demo_data") or []
if maxf:
    tape = tape[:maxf]
md = list(dec.create_nclone_map(path))

py = Simulator(SimConfig())
py.load(md)
nat = nclone_native.Simulator.load(md)

# Both ports happen to name these identically, so one tuple serves for both.
# `state` last, so the printouts below can special-case it as non-numeric.
# `floor_normalized_x/y` earn their place: they feed `anim_rate`, which drives
# the animation frame, which poses the skeleton that every hitscan kill test
# reads. Without them a ninja can be bit-identical in position and speed while
# its skeleton silently drifts -- which is exactly how the laser splits hid.
FIELDS = ("xpos", "ypos", "xspeed", "yspeed",
          "floor_normalized_x", "floor_normalized_y",
          "ceiling_normalized_x", "ceiling_normalized_y",
          "floor_count", "wall_count", "ceiling_count",
          "airborn", "walled", "jump_buffer", "floor_buffer", "wall_buffer",
          "launch_pad_buffer", "jump_duration", "state")


def grab(n):
    out = []
    for f in FIELDS:
        v = getattr(n, f)
        out.append(int(v) if isinstance(v, bool) else v)
    return tuple(out)


def snap_py(s):
    return grab(s.ninja)


def snap_nat(s):
    return grab(s.get_state_dynamic().ninja)


hist = []
first = None
for f, b in enumerate(tape, 1):
    hor, jmp = decode_input_to_controls(int(b))
    py.tick(hor, jmp)
    nat.tick(hor, jmp)
    a, c = snap_py(py), snap_nat(nat)
    hist.append((f, hor, jmp, a, c))
    if len(hist) > 4:
        hist.pop(0)
    if first is None and a != c:
        first = f
        break
    if a[-1] in (6, 8):
        break

if first is None:
    print("%s: engines agree for all %d frames (final python state %s)"
          % (demo, len(tape), snap_py(py)[-1]))
    sys.exit()

print("%s: first divergence on frame %d of %d\n" % (demo, first, len(tape)))
for f, hor, jmp, a, c in hist:
    tag = "  <-- DIVERGES" if f == first else ""
    print("frame %-5d input hor=%+d jmp=%d%s" % (f, hor, jmp, tag))
    print("   python x=%.17g y=%.17g vx=%.17g vy=%.17g st=%d"
          % (a[0], a[1], a[2], a[3], a[-1]))
    print("   native x=%.17g y=%.17g vx=%.17g vy=%.17g st=%d"
          % (c[0], c[1], c[2], c[3], c[-1]))
    if f == first:
        for i, name in enumerate(FIELDS):
            if a[i] != c[i]:
                print("   FIELD %-18s python=%-24.17g native=%-24.17g  delta=%+.3e"
                      % (name, a[i], c[i], c[i] - a[i]))
