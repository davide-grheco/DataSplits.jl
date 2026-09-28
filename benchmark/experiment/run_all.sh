#!/usr/bin/env bash
set -uo pipefail

DATA=${1:-benchmark/experiment/data}
OUT=${2:-benchmark/experiment/results}
PY=${3:-.venv/bin/python}
LOG="$OUT/run_all.log"
HERE=$(dirname "$0")

mkdir -p "$DATA" "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1

failed=0
stage() {
  local name=$1; shift
  echo "=== $name  $(date -Is) ===" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then
    echo "--- $name OK" | tee -a "$LOG"
  else
    echo "--- $name FAILED (continuing)" | tee -a "$LOG"
    failed=$((failed + 1))
  fi
}

: >"$LOG"

stage generate      julia --project=benchmark "$HERE/generate.jl" "$DATA"
stage julia-main    julia --project=benchmark "$HERE/run_julia.jl" "$DATA" "$OUT/julia_main.json"
stage python-main   "$PY" "$HERE/run_python.py" "$DATA" "$OUT/python_main.json"
stage rss           julia --project=benchmark "$HERE/rss_sweep.jl" "$DATA" "$OUT/rss.json" "$PY"
stage analyse       julia --project=benchmark "$HERE/analyse.jl" "$OUT"

echo "=== done $(date -Is): $failed stage(s) failed ===" | tee -a "$LOG"
exit $failed
