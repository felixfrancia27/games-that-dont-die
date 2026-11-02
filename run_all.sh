#!/usr/bin/env bash
# Reproduce every result table of the paper from data/matches.json.
# Usage: ./run_all.sh            (all experiments, in parallel; roughly 1 hour on a laptop)
#        ./run_all.sh lying      (one experiment: lying | null | joint | modelfree | sybil)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
PY="python3 -I"
run() {
  case "$1" in
    lying)     $PY src/srtd.py data/matches.json results/lying_host.json > results/lying_host.log ;;
    null)      $PY src/null_calib.py src/srtd.py data/matches.json > results/null_calibration.txt ;;
    joint)     $PY src/joint.py src/srtd.py data/matches.json > results/joint.log ;;
    modelfree) $PY src/modelfree.py > results/modelfree.log ;;
    sybil)     $PY src/sybil.py tokens > results/sybil_tokens.log && $PY src/sybil.py rank > results/sybil_rank.log ;;
    *) echo "unknown experiment: $1" >&2; exit 1 ;;
  esac
  echo "done: $1"
}
if [ $# -gt 0 ]; then run "$1"; else
  for e in lying null joint modelfree sybil; do run "$e" & done; wait
fi
