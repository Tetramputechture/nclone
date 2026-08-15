"""Run the losers subset on both engines, with and without a flag, and diff
which demos each engine wins. The two engines are required to agree exactly,
so any per-demo disagreement is a port bug in its own right.
"""
import json
import os
import subprocess
import sys

ROOT = "/home/tetra/projects/nclone"
PY = ROOT + "/.venv/bin/python"
SUB = ROOT + "/tools/npptrace/subsets/losers.txt"
flag = sys.argv[1] if len(sys.argv) > 1 else ""

won = {}
for eng in ("python", "native"):
    out = "/tmp/abcmp_%s.jsonl" % eng
    env = dict(os.environ)
    if flag:
        env[flag] = "1"
    subprocess.run([PY, ROOT + "/scripts/attract_gate_report.py", "--files-from", SUB,
                    "--out", out, "--workers", "8", "--engine", eng],
                   cwd=ROOT, capture_output=True, env=env)
    rows = [json.loads(l) for l in open(out)]
    won[eng] = {r["f"] for r in rows if r.get("outcome") == "win"}
    print("%-7s flag=%-28s wins %d" % (eng, flag or "(none)", len(won[eng])))

only_py = won["python"] - won["native"]
only_nat = won["native"] - won["python"]
print("  python-only wins: %s" % (sorted(only_py) or "none"))
print("  native-only wins: %s" % (sorted(only_nat) or "none"))
