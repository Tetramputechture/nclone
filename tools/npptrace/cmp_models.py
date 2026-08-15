"""Outcome of a few captured drone demos under the legacy and full models."""
import os
import re
import subprocess

ROOT = "/home/tetra/projects/nclone"
PY = ROOT + "/.venv/bin/python"
DEMOS = ["10131718", "10135227", "10138659", "10139152"]

for demo in DEMOS:
    row = [demo]
    for label, env_extra in (("legacy", {}), ("full", {"NCLONE_AB_DRONE_FULL_SHOT": "1"})):
        env = dict(os.environ)
        env.update(env_extra)
        p = subprocess.run([PY, ROOT + "/tools/npptrace/death_cause.py", demo],
                           cwd=ROOT, capture_output=True, text=True, env=env)
        out = p.stdout + p.stderr
        m = re.search(r"(DIED\s+frame\s+\d+\s+cause=\S+|WON\s+frame\s+\d+)", out)
        row.append("%s: %s" % (label, m.group(1) if m else "?"))
    print("  ".join(row))
