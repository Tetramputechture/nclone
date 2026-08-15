"""First tick the two ports' ANIMATION state diverges.

The skeleton is read by nothing but the hitscan kill test, so it can drift for
dozens of frames with every other check still passing -- and then decide whether
a laser threads between two limbs or clips one. Native's animation state is not
in the ninja snapshot; `NCLONE_DEBUG_ANIM=1` prints it, and this lines that up
against Python's.

Native prints BEFORE the advance, so its line for tick N+1 carries tick N's
result: `artifacts/anim_nat.txt` must already have its first (construction) line
stripped, and native line k is compared against Python tick k.

    animdiff.py <demo> <native_trace.txt>
"""
import os
import re
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/home/tetra/projects/nclone")

from nclone.nsim import Simulator  # noqa: E402
from nclone.sim_config import SimConfig  # noqa: E402
from nclone.replay.npp_attract_decoder import NppAttractDecoder  # noqa: E402
from nclone.replay.replay_executor import decode_input_to_controls  # noqa: E402

demo, nat_path = sys.argv[1], sys.argv[2]

nat = []
for line in open(nat_path):
    m = re.search(
        r"state=(\S+) frame=(\S+) prev=(\S+) cycle=(\S+) rate=(\S+) "
        r"residual=(\S+) blend=(\S+) tilt=(\S+) facing=(\S+) "
        r"bone0=\((\S+),(\S+)\)", line)
    if m:
        nat.append((int(m.group(1)), int(m.group(2)), int(m.group(3)),
                    int(m.group(4)), float(m.group(5)), float(m.group(6)),
                    float(m.group(7)), float(m.group(8)), float(m.group(9)),
                    float(m.group(10)), float(m.group(11))))

dec = NppAttractDecoder()
path = "/home/tetra/projects/attracts/%s" % demo
tape = (dec.decode_npp_attract_file(path).get("demo_data") or [])[:len(nat)]
md = list(dec.create_nclone_map(path))

py = Simulator(SimConfig())
py.load(md)
n = py.ninja

# `tilt` and `facing` matter as much as the clip: the last thing update_graphics
# does is rotate every bone by `tilt` and mirror it by `facing`, so the two ports
# can agree on the clip, the blend and the cycle and still pose the skeleton
# differently. `bone0` is carried too, as the end-to-end check.
FIELDS = ("anim_state", "anim_frame", "anim_frame_prev", "run_cycle",
          "anim_rate", "frame_residual", "anim_blend", "tilt", "facing",
          "bone0x", "bone0y")

first = None
for i, b in enumerate(tape):
    hor, jmp = decode_input_to_controls(int(b))
    py.tick(hor, jmp)
    if i >= len(nat):
        break
    # `run_cycle` is created lazily, on the first tick the running state runs.
    row = tuple(
        (n.bones[0][0] if f == "bone0x" else
         n.bones[0][1] if f == "bone0y" else
         getattr(n, f, 0))
        for f in FIELDS)
    if row != nat[i]:
        first = (i + 1, row, nat[i])
        break

if first is None:
    print("%s: animation state agrees for all %d compared ticks" % (demo, len(nat)))
    sys.exit()

tick, prow, nrow = first
print("%s: animation first diverges on tick %d\n" % (demo, tick))
for j, f in enumerate(FIELDS):
    mark = "   <-- " if prow[j] != nrow[j] else ""
    if isinstance(prow[j], float):
        print("   %-16s python=%-26.17g native=%-26.17g%s" % (f, prow[j], nrow[j], mark))
    else:
        print("   %-16s python=%-26s native=%-26s%s" % (f, prow[j], nrow[j], mark))
