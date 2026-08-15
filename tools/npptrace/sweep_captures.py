"""For every capture in out9: replay in nclone, report outcome + first divergence."""
import glob, json, os, re, subprocess

ROOT = "/home/tetra/projects/nclone"
PY = ROOT + "/.venv/bin/python"
OUT = ROOT + "/tools/npptrace/out9"

stems = sorted(glob.glob(OUT + "/*.ninja.jsonl"))
print("%-10s %-9s %-6s %s" % ("demo", "outcome", "frame", "first divergent fields"))
for s in stems:
    demo = os.path.basename(s).replace(".ninja.jsonl", "")
    if demo.startswith("cap"):
        continue                      # unresolved -> skip, needs match_traces
    p = subprocess.run([PY, "tools/npptrace/sim_trace.py", demo, "--engine", "native",
                        "-o", "/tmp/sw.jsonl"], cwd=ROOT, capture_output=True, text=True)
    m = re.search(r"outcome=(\w+) frames=(\d+)", p.stdout + p.stderr)
    outcome = m.group(1) if m else "?"
    d = subprocess.run([PY, "tools/npptrace/diff_trace.py", s, "/tmp/sw.jsonl"],
                       cwd=ROOT, capture_output=True, text=True)
    txt = d.stdout + d.stderr
    if "BIT-EXACT" in txt:
        print("%-10s %-9s %-6s BIT-EXACT" % (demo, outcome, "-"))
        continue
    fm = re.search(r"FIRST DIVERGENCE\s+frame (\d+)\s+fields: (.+)", txt)
    denorm = "floor_x" in (fm.group(2) if fm else "")
    print("%-10s %-9s %-6s %s%s" % (
        demo, outcome, fm.group(1) if fm else "?",
        fm.group(2).strip() if fm else "(no diff line)",
        "   <-- denormal-floor signature" if denorm else ""))
