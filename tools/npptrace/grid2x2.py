"""Measure the two shooter-drone candidates alone and together.

They are suspected of being a §6 pair -- line-of-sight acquisition makes the
drone fire far more often, and a tracking gun makes each shot far more accurate,
so either alone should measure worse than both or than neither.
"""
import collections
import itertools
import json
import os
import subprocess

ROOT = "/home/tetra/projects/nclone"
PY = ROOT + "/.venv/bin/python"
SUB = "/tmp/drones.txt"
FLAGS = ["NCLONE_AB_DRONE_LOS_ACQUIRE", "NCLONE_AB_DRONE_AIM_TRACK"]

print("%-28s %-6s %s" % ("configuration", "wins", "drone_shot"))
for combo in itertools.product([0, 1], repeat=len(FLAGS)):
    env = dict(os.environ)
    on = []
    for f, v in zip(FLAGS, combo):
        if v:
            env[f] = "1"
            on.append(f.replace("NCLONE_AB_DRONE_", "").lower())
    out = "/tmp/grid_%s.jsonl" % ("_".join(on) or "base")
    subprocess.run([PY, ROOT + "/scripts/attract_gate_report.py", "--files-from", SUB,
                    "--out", out, "--workers", "8", "--engine", "python"],
                   cwd=ROOT, capture_output=True, env=env)
    rows = [json.loads(line) for line in open(out)]
    wins = sum(1 for r in rows if r.get("outcome") == "win")
    causes = collections.Counter(r.get("cause") for r in rows)
    print("%-28s %-6d %d" % (" + ".join(on) or "(neither)", wins,
                             causes.get("drone_shot", 0)))
