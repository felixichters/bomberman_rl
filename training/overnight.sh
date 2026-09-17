#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
[ -x "$PY" ] || PY=python3
TAG=${1:-linear}
FROM=${2:-t2a}
INIT=${3:-}

echo "=== recording demonstrations ==="
./training/collect_all_experts.sh 250

echo "=== curriculum from $FROM ==="
if [ -n "$INIT" ]; then
  $PY training/run_pipeline.py --tag "$TAG" --from "$FROM" --init "$INIT" --eval-every 500 --eval-rounds 60
else
  $PY training/run_pipeline.py --tag "$TAG" --from "$FROM" --eval-every 500 --eval-rounds 60
fi
