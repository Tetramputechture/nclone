#!/bin/bash
# Rule each remaining mechanic in or out against ground truth, cheaply.
# pool = type - 2:  21=dual_laser(23)  17=turret(19)  14=floor_guard(16)
#                   24=mini_drone(26)  12=zap(14)     10=chaingun(12)
cd /home/tetra/projects/nclone
for pool in 21 17 14 24 12 10; do
  echo "===== pool $pool ====="
  timeout 400 .venv/bin/python /tmp/entscore.py "$pool" 2>&1 \
    | grep -vE 'pygame|animation' | tail -3
done
