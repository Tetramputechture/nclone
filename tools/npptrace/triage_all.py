"""Split every captured demo the port still loses into entity bugs and drift.

A demo whose ninja leaves the game's trajectory long before it dies cannot be
recovered by entity work, however good; one that is bit-exact until the kill is
a clean entity bug. Doing this first is what §15 got right and what the mine
round nearly got wrong.
"""
import collections
import glob
import os
import re
import subprocess

ROOT = "/home/tetra/projects/nclone"
PY = ROOT + "/.venv/bin/python"
DIRS = [ROOT + "/tools/npptrace/out9", ROOT + "/tools/npptrace/out9_batch1", ROOT + "/tools/npptrace/out10", ROOT + "/tools/npptrace/out11"]

rows = []
for d in DIRS:
    for s in sorted(glob.glob(d + "/*.ninja.jsonl")):
        demo = os.path.basename(s).replace(".ninja.jsonl", "")
        if demo.startswith("cap"):
            continue
        p = subprocess.run([PY, ROOT + "/tools/npptrace/sim_trace.py", demo,
                            "--engine", "native", "-o", "/tmp/ta.jsonl"],
                           cwd=ROOT, capture_output=True, text=True)
        txt = p.stdout + p.stderr
        m = re.search(r"outcome=(\w+) frames=(\d+)", txt)
        outcome, died = (m.group(1), int(m.group(2))) if m else ("?", -1)
        if outcome == "win":
            continue
        dd = subprocess.run([PY, ROOT + "/tools/npptrace/diff_trace.py", s,
                             "/tmp/ta.jsonl"], cwd=ROOT, capture_output=True,
                            text=True)
        t = dd.stdout + dd.stderr
        fm = re.search(r"FIRST DIVERGENCE\s+frame (\d+)\s+fields: (.+)", t)
        div = int(fm.group(1)) if fm else None
        fields = fm.group(2).strip() if fm else ""
        c = subprocess.run([PY, ROOT + "/tools/npptrace/death_cause.py", demo],
                           cwd=ROOT, capture_output=True, text=True)
        cm = re.search(r"cause=('[^']*'|None)", c.stdout + c.stderr)
        cause = cm.group(1).strip("'") if cm else "?"
        lead = (died - div) if div is not None else 0
        kind = ("entity" if div is None or lead <= 20 else "drift")
        rows.append((demo, cause, died, div, lead, kind, fields))

print("%-10s %-13s %-7s %-7s %-7s %-7s %s"
      % ("demo", "cause", "dies", "diverge", "lead", "kind", "first divergent fields"))
for r in sorted(rows, key=lambda r: (r[5], r[1])):
    print("%-10s %-13s %-7s %-7s %-7s %-7s %s"
          % (r[0], r[1], r[2], r[3], r[4], r[5], r[6][:44]))
k = collections.Counter(r[5] for r in rows)
print("\n%d captured losers: %d clean entity bugs, %d physics drift"
      % (len(rows), k["entity"], k["drift"]))
print("drift by cause:  %s"
      % collections.Counter(r[1] for r in rows if r[5] == "drift").most_common())
print("entity by cause: %s"
      % collections.Counter(r[1] for r in rows if r[5] == "entity").most_common())
print("floor_x in the first divergent fields: %d of %d drift cases"
      % (sum(1 for r in rows if r[5] == "drift" and "floor" in r[6]),
         k["drift"]))
