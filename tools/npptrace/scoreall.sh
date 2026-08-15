#!/bin/bash
# Score the entity pools behind the biggest loss buckets against the game
# captures.
#
# The cross-engine split is down to 3, so the remaining ~2,428 losses are shared
# misunderstandings -- both ports wrong the same way. Neither port can find those.
# The captures can: they are the real game's own per-frame entity state, already
# on disk, and `entscore` runs the PYTHON engine against them, so anything short
# of ~100% is a rule this project has wrong with the evidence sitting right there.
#
# Pool index is `entity_type - 2`. Scored here, in loss-bucket order:
#   pool 19 -> type 21 toggle mine   (mine, 344 losses)
#   pool 10 -> type 12 chaingun      (drone_shot, 305)
#   pool 11 -> type 13 laser drone   (drone_shot)
#   pool 17 -> type 19 turret        (turret, 210)
#   pool 16 -> type 18 rocket        (rocket, 140)
# Type 23 (pool 21) is already at 98.55% after the corner-rate fix; type 25, 14
# and 17 were scored at 100% earlier.
#
# Deliberately NOT the full 0..27 sweep: a single pool can take 10-20 minutes,
# so scoring everything costs hours and most pools have no losses to explain.
cd /home/tetra/projects/nclone
for pool in 19 10 11 17 16; do
  echo "=== pool $pool (type $((pool + 2))) ==="
  timeout 1800 .venv/bin/python tools/npptrace/entscore.py "$pool" 2>&1 \
    | grep -Ev 'pygame|Loading' \
    | grep -E 'pool|TOTAL|whole demo'
done
