#!/bin/bash
# Run ninjadiv over every demo we have a capture for. The ninja is the one
# subsystem every loss passes through, so this is the number that bounds how
# much of the remaining failure is core physics.
cd /home/tetra/projects/nclone
ls tools/npptrace/out*/*.layout.json 2>/dev/null \
  | xargs -n1 basename \
  | sed 's/\.layout\.json$//' \
  | grep -E '^[0-9]+$' \
  | sort -u > /tmp/capdemos.txt
echo "captured demos: $(wc -l < /tmp/capdemos.txt)"
.venv/bin/python tools/npptrace/ninjadiv.py $(cat /tmp/capdemos.txt) 2>&1 \
  | grep -v 'pygame\|animation'
