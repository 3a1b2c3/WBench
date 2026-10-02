#!/usr/bin/env bash
# Run one (or a few) WBench cases through the Matrix-Game-3.5 driver
# (src/models/camera/matrix_game35_model.py), then optionally evaluate them.
# Requires: Matrix-Game-3.5/.venv set up (its own setup_env.sh) + checkpoints
# downloaded (its own download_models.sh) -- this script only drives
# WBench's own generate.py/main.py against the registered "matrix_game35"
# model. See Matrix-Game-3.5/run_example_inference.sh to sanity-check that
# half of the setup independently first.
#
#   ./run_matrix_game35.sh --cases data/cases/case_1.json
#   ./run_matrix_game35.sh --cases data/cases/case_1.json --evaluate
#   ./run_matrix_game35.sh --limit 3
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

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
echo "WBench: generating with matrix_game35"
echo "=========================================="
uv run python generate.py --model matrix_game35 "${GEN_ARGS[@]}"

if [ "$EVALUATE" = "1" ]; then
    echo
    echo "=========================================="
    echo "WBench: evaluating matrix_game35"
    echo "=========================================="
    uv run python main.py --model matrix_game35
fi

echo
echo "Done. Videos: work_dirs/matrix_game35/videos/"
[ "$EVALUATE" = "1" ] && echo "Results: work_dirs/matrix_game35/evaluation/ + work_dirs/matrix_game35/report.json"
