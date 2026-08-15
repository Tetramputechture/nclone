#!/bin/bash
# The chaser losses look like FALSE acquisition: one chaser per map climbs out of
# its corridor toward our ninja while the game's stays patrolling on the floor.
# §36 swept 12/16/18/19/20/24 -- all >= 12. Try NARROWER.
cd /home/tetra/projects/nclone
export NCLONE_AB_CHASER_INTERRUPT=1
export NCLONE_AB_CHASER_EARLY_LATCH=1
for w in 9 10 11; do
  echo "--- corridor half-width $w ---"
  NCLONE_AB_CORRIDOR_W=$w timeout 280 .venv/bin/python /tmp/chaserscore.py 2>&1 \
    | grep -vE 'pygame|animation' | tail -2
done
