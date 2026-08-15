"""First frame the two engines' ENTITIES disagree, not just their ninja.

`lockstep.py` compares the ninja, which is the last thing to diverge: on
`10244473` the ninja is identical right up to the frame native kills it, because
what actually went wrong was a drone that turned the wrong way hundreds of
frames earlier. Entities move first, the ninja follows.

Matches entities by their index within each type, which is the order both ports
build them in.

    entlock.py <demo> [max_frames]
"""
import collections
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


# The two ports do not model the exit pair the same way -- Python keeps the door
# and its switch as separate type-3/4 entities where native keeps one -- so those
# types are skipped rather than reported as a count mismatch every frame.
# Type 18 (rocket) joins them for the same reason with different details: native
# snapshots the PROJECTILE (`e.projectile.xpos`, `active = state == FLYING`)
# where Python snapshots the launcher, so an idle rocket reads (0,0)/inactive on
# one side and its emplacement on the other -- every frame, on every map with a
# rocket, with nothing wrong.
SKIP_TYPES = {3, 4, 18}

# `state` is NOT comparable across the ports: native's entity snapshot reports a
# drone's FACING there (it flips 0 -> 2 at a patrol turnaround) while Python's
# `state` is the shooter's acquire/charge/fire machine. Comparing them reported
# a divergence on every drone that simply turned around. Position and `active`
# mean the same thing on both sides, so only those are compared -- and any real
# state-machine divergence still shows up, one frame later, as a position or
# activity difference.


def py_snap():
    out = collections.defaultdict(list)
    for t, ents in py.entity_dic.items():
        if t in SKIP_TYPES:
            continue
        for e in ents:
            out[t].append((round(e.xpos, 12), round(e.ypos, 12), bool(e.active)))
    return out


def nat_snap():
    out = collections.defaultdict(list)
    for e in nat.get_state_dynamic().entities:
        if e.entity_type in SKIP_TYPES:
            continue
        out[e.entity_type].append((round(e.x, 12), round(e.y, 12), bool(e.active)))
    return out


first = None
for f, b in enumerate(tape, 1):
    hor, jmp = decode_input_to_controls(int(b))
    py.tick(hor, jmp)
    nat.tick(hor, jmp)
    a, c = py_snap(), nat_snap()
    diffs = []
    for t in sorted(set(a) | set(c)):
        la, lc = a.get(t, []), c.get(t, [])
        if len(la) != len(lc):
            diffs.append((t, -1, len(la), len(lc)))
            continue
        for i, (va, vc) in enumerate(zip(la, lc)):
            if va != vc:
                diffs.append((t, i, va, vc))
    if diffs:
        first = (f, diffs)
        break
    if py.ninja.state in (6, 8):
        break

if first is None:
    print("%s: entities agree for all %d simulated frames" % (demo, len(tape)))
    sys.exit()

f, diffs = first
print("%s: entities first diverge on frame %d of %d\n" % (demo, f, len(tape)))
for t, i, va, vc in diffs[:20]:
    if i < 0:
        print("   type %-3d COUNT python=%s native=%s" % (t, va, vc))
        continue
    print("   type %-3d idx %-2d" % (t, i))
    print("      python pos=(%.6f, %.6f) active=%s" % (va[0], va[1], va[2]))
    print("      native pos=(%.6f, %.6f) active=%s" % (vc[0], vc[1], vc[2]))
    if va[0] != vc[0] or va[1] != vc[1]:
        print("      dpos=(%+.3e, %+.3e)" % (vc[0] - va[0], vc[1] - va[1]))
print("\n(%d differing entities in total)" % len(diffs))
