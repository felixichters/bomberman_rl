#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
[ -x "$PY" ] || PY=python3
R=${1:-250}
mkdir -p runs/expert

collect () {
  local name=$1 scenario=$2; shift 2
  local out=runs/expert/${name}.npz
  if [ -f "$out" ]; then echo "  $name already recorded"; return; fi
  if [ $# -gt 0 ]; then
    $PY training/collect_expert.py --rounds "$R" --scenario "$scenario" --opponents "$@" --out "$out" >"runs/expert/${name}.log" 2>&1
  else
    $PY training/collect_expert.py --rounds "$R" --scenario "$scenario" --out "$out" >"runs/expert/${name}.log" 2>&1
  fi
  echo "  $name: $(tail -1 "runs/expert/${name}.log")"
}

collect crates-sparse crates-sparse &
collect crates-medium crates-medium &
collect classic       classic &
collect classic-vs    classic coin_collector_agent coin_collector_agent &
wait
echo "all expert traces recorded"
