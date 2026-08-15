#!/usr/bin/env bash
# A/B each reverse-engineered entity type in isolation against the attract corpus.
#
# Why this exists: none of the RE'd types has a physical collision
# (docs/ENTITY_INTEGRATION_SPEC.md §6.6), so an over-lethal one shows up ONLY as a
# drop in the demo win rate -- and because several hazards usually share a map, one
# bad entity drags down every type it co-occurs with. Enabling them all at once
# tells you something is wrong but not what. This runs, for each type T:
#
#   OFF: every RE'd type disabled            <- the reference
#   ON : only T enabled
#
# over just the demos whose map contains T. A correct hazard leaves ON == OFF,
# because these demos are expert runs that survived the real entity. ON < OFF means
# the implementation kills where the game did not.
#
# Usage: scripts/attract_bisect_entities.sh <attract-dir> <out-dir> [limit]
set -u

IN_DIR="${1:?usage: $0 <attract-dir> <out-dir> [limit]}"
OUT_DIR="${2:?usage: $0 <attract-dir> <out-dir> [limit]}"
LIMIT="${3:-4000}"
WORKERS="${WORKERS:-10}"
PY="${PY:-.venv/bin/python}"

ALL_TYPES="12,13,15,16,18,19,22,23,27"
mkdir -p "$OUT_DIR"

# Every RE'd type except $1.
others() {
  local keep="$1" out=""
  for t in ${ALL_TYPES//,/ }; do
    [ "$t" = "$keep" ] || out="${out}${out:+,}${t}"
  done
  echo "$out"
}

for T in ${ALL_TYPES//,/ }; do
  for PHASE in off on; do
    if [ "$PHASE" = off ]; then
      DISABLE="$ALL_TYPES"
    else
      DISABLE="$(others "$T")"
    fi
    echo "=== type $T [$PHASE] (disabled: ${DISABLE:-none}) ==="
    NCLONE_DISABLE_ENTITY_TYPES="$DISABLE" "$PY" scripts/attract_gate_report.py \
      --in-dir "$IN_DIR" --out "$OUT_DIR/t${T}_${PHASE}.jsonl" \
      --limit "$LIMIT" --workers "$WORKERS" --engine python --require-type "$T" \
      2>/dev/null | grep -E "^[0-9]+ demos:|^    (die|timeout|err)"
  done
done
