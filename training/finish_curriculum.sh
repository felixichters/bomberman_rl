#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
[ -x "$PY" ] || PY=python3
TAG=${1:-final}
INIT=${2:?need a checkpoint to start from}
FROM=${3:-t2}

rm -f runs/STOP
echo "=== curriculum: $FROM -> t4, starting from $INIT"
$PY training/run_pipeline.py --tag "$TAG" --from "$FROM" --init "$INIT" \
    --eval-every 500 --eval-rounds 60 --eval-workers 3 || true

BEST=$(ls -t runs/${TAG}_t4/best.npz runs/${TAG}_t3/best.npz runs/${TAG}_t3a/best.npz \
              runs/${TAG}_t2/best.npz 2>/dev/null | head -1)
if [ -z "$BEST" ]; then echo "no checkpoint produced"; exit 1; fi
echo "=== best checkpoint: $BEST"

echo "=== self-play pool"
$PY training/build_pool.py runs/${TAG}_t3 runs/${TAG}_t4 --keep 8 2>/dev/null || \
  $PY training/build_pool.py runs/${TAG}_t2 --keep 6

if [ -x tools/refresh_report.sh ]; then
  echo "=== report"
  tools/refresh_report.sh "$BEST"
fi

echo "=== submission"
$PY tools/make_submission.py --model "$BEST"
