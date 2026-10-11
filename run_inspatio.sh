#!/usr/bin/env bash
# Run one (or a few) WBench cases through the InSpatio-World 1.5 driver
# (src/models/camera/inspatio_model.py), then optionally evaluate them.
# Requires: ../inspatio-world-v1.5/.venv (its setup_env.sh) + checkpoints (its download_models.sh).
# This script only drives WBench's own generate.py/main.py against the registered "inspatio" model.
#
#   ./run_inspatio.sh --cases data/cases/case_1.json
#   ./run_inspatio.sh --cases data/cases/case_1.json --evaluate
#   ./run_inspatio.sh --limit 3
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ~/wbench-venv is the known-good WBench env (see run_worldcrafter.sh for why it is used directly).
WBENCH_PYTHON="${WBENCH_PYTHON:-$HOME/wbench-venv/bin/python}"
if [ ! -x "$WBENCH_PYTHON" ]; then
    echo "WBENCH_PYTHON not found/executable: $WBENCH_PYTHON" >&2
    echo "Set WBENCH_PYTHON to a venv with WBench's dependencies installed." >&2
    exit 1
fi

EVALUATE=0
GEN_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--evaluate" ]; then
        EVALUATE=1
    else
        GEN_ARGS+=("$arg")
    fi
done

echo "=========================================="
echo "WBench: generating with inspatio"
echo "=========================================="
"$WBENCH_PYTHON" generate.py --model inspatio "${GEN_ARGS[@]}"

if [ "$EVALUATE" = "1" ]; then
    echo
    echo "=========================================="
    echo "WBench: evaluating inspatio"
    echo "=========================================="
    "$WBENCH_PYTHON" main.py --model inspatio
fi

echo
echo "Done. Videos: work_dirs/inspatio/videos/"
[ "$EVALUATE" = "1" ] && echo "Results: work_dirs/inspatio/evaluation/ + work_dirs/inspatio/report.json"
exit 0
