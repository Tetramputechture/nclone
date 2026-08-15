"""Which loss causes are PORT divergences rather than physics gaps?

The death ball turned out to be native-only: python won all 60 sampled demos and
native lost all 60, and python is the side scored bit-exact against the captures.
Ask the same question of every other cause, in-process so the python engine is
paid for once per demo rather than once per subprocess pool.
"""
import collections
import json
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

GATE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gate_full2.jsonl"
PER = int(sys.argv[2]) if len(sys.argv) > 2 else 12
MAXF = int(sys.argv[3]) if len(sys.argv) > 3 else 3000

inv = {}
if os.path.exists("/tmp/inv_full.jsonl"):
    inv = {json.loads(l)["f"]: json.loads(l) for l in open("/tmp/inv_full.jsonl")}
rows = [json.loads(l) for l in open(GATE)]
by_cause = collections.defaultdict(list)
for r in rows:
    if r["outcome"] != "win" and r.get("cause"):
        if not inv or inv.get(r["f"], {}).get("outcome") == "win":
            by_cause[r["cause"]].append(r["f"])

dec = NppAttractDecoder()


def run(md, tape, engine):
    if engine == "python":
        sim = Simulator(SimConfig())
        sim.load(md)
        for b in tape:
            hor, jmp = decode_input_to_controls(int(b))
            sim.tick(hor, jmp)
            if sim.ninja.state == 6:
                return "die"
            if sim.ninja.state == 8:
                return "win"
        return "timeout"
    sim = nclone_native.Simulator.load(md)
    for b in tape:
        hor, jmp = decode_input_to_controls(int(b))
        sim.tick(hor, jmp)
        n = sim.get_state_dynamic().ninja
        if int(n.state) == 6:
            return "die"
        if int(n.state) == 8:
            return "win"
    return "timeout"


print("%-14s %8s  %-22s %-22s" % ("cause", "sampled", "native", "python"))
for cause in sorted(by_cause, key=lambda c: -len(by_cause[c])):
    demos = by_cause[cause][:PER]
    nat = collections.Counter()
    pyc = collections.Counter()
    for demo in demos:
        path = "/home/tetra/projects/attracts/%s" % demo
        try:
            tape = (dec.decode_npp_attract_file(path).get("demo_data") or [])[:MAXF]
            md = list(dec.create_nclone_map(path))
        except Exception:
            continue
        nat[run(md, tape, "native")] += 1
        pyc[run(md, tape, "python")] += 1
    verdict = ""
    if pyc.get("win", 0) and not nat.get("win", 0):
        verdict = "   <== NATIVE-ONLY BUG"
    elif nat.get("win", 0) and not pyc.get("win", 0):
        verdict = "   <== PYTHON-ONLY BUG"
    print("%-14s %5d/%-4d  %-22s %-22s%s"
          % (cause, len(demos), len(by_cause[cause]),
             dict(nat.most_common()), dict(pyc.most_common()), verdict))
    sys.stdout.flush()
