"""Is the `mine` bucket a mine bug, or physics drift arriving at a mine?

For every captured demo the port loses to a mine, report where the ninja first
leaves the game's trajectory relative to where it dies. A divergence long before
the kill means no amount of mine work will recover it.
"""
import os
import re
import subprocess

ROOT = "/home/tetra/projects/nclone"
PY = ROOT + "/.venv/bin/python"
DIRS = [ROOT + "/tools/npptrace/out9", ROOT + "/tools/npptrace/out9_batch1"]
DEMOS = ["10131474", "10132599", "10133925", "10137755"]

print("%-10s %-10s %-12s %s" % ("demo", "dies", "diverges", "verdict"))
for demo in DEMOS:
    trace = next((d + "/" + demo + ".ninja.jsonl" for d in DIRS
                  if os.path.exists(d + "/" + demo + ".ninja.jsonl")), None)
    if trace is None:
        continue
    p = subprocess.run([PY, ROOT + "/tools/npptrace/sim_trace.py", demo,
                        "--engine", "native", "-o", "/tmp/mt.jsonl"],
                       cwd=ROOT, capture_output=True, text=True)
    m = re.search(r"frames=(\d+)", p.stdout + p.stderr)
    died = int(m.group(1)) if m else -1
    d = subprocess.run([PY, ROOT + "/tools/npptrace/diff_trace.py", trace,
                        "/tmp/mt.jsonl"], cwd=ROOT, capture_output=True, text=True)
    txt = d.stdout + d.stderr
    fm = re.search(r"FIRST DIVERGENCE\s+frame (\d+)", txt)
    div = int(fm.group(1)) if fm else None
    if div is None:
        verdict = "bit-exact to the kill -- a real mine bug"
    elif died - div > 20:
        verdict = "drifted %d frames before dying -- physics, not the mine" % (died - div)
    else:
        verdict = "diverges at the kill -- a real mine bug"
    print("%-10s %-10s %-12s %s" % (demo, died, div, verdict))
