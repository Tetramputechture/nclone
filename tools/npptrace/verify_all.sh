#!/usr/bin/env bash
# Replay every captured demo in ONE engine and diff against the game ground truth,
# printing the first divergence per demo.
#
# This is the diagnostic view -- run it when you want to read a specific
# divergence. The REGRESSION GUARD is `scripts/verify_game_traces.py`, which runs
# both engines, also checks them against each other, and fails against the
# baseline in `docs/game_traces/expected.json`. Use that one in a commit gate:
#
#     python scripts/verify_game_traces.py
#
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
ENGINE="${1:-python}"
fail=0
for g in docs/game_traces/*.jsonl; do
  id=$(basename "$g" .jsonl)
  out=$(mktemp)
  $PY tools/npptrace/sim_trace.py "$id" --engine "$ENGINE" -o "$out" 2>&1 \
    | grep -E 'outcome=' || true
  res=$($PY tools/npptrace/diff_trace.py "$g" "$out" 2>&1)
  if echo "$res" | grep -q 'BIT-EXACT'; then
    echo "  $id  BIT-EXACT  ($(echo "$res" | grep -o 'overlap=[0-9]*'))"
  else
    fail=1
    echo "  $id  DIVERGES"
    echo "$res" | sed -n '3,12p' | sed 's/^/      /'
  fi
  rm -f "$out"
done
exit $fail
