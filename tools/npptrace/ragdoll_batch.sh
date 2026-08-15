#!/bin/sh
# Run ragdoll_check.py over a list of demos and print one block each.
#
#     NCLONE_ATTRACTS=/tmp/official sh tools/npptrace/ragdoll_batch.sh 1016 1017 ...
#
# With no arguments, uses a spread of the official attract demos that covers
# every death cause nclone can produce, exploding and intact.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
[ -x "$PY" ] || PY=python

DEMOS="$*"
[ -n "$DEMOS" ] || DEMOS="1016 1017 1018 1031 1038 1027 1012 1009 1024 1032 1029"

fails=0
for d in $DEMOS; do
    echo "===== $d ====="
    $PY tools/npptrace/ragdoll_check.py "$d" --frames 400 2>&1 |
        grep -E 'died|never reached|mines|ragdoll bones|non-finite|inside geometry|last 10|^OK|^FAIL|Error' ||
        fails=$((fails + 1))
done
echo
echo "blocks with no output: $fails"
