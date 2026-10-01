#!/usr/bin/env bash
# Run one (or a few) WBench cases through the WorldCrafter driver
# (src/models/camera/worldcrafter_model.py), then optionally evaluate them.
# Requires: WorldCrafter/uvenv/.venv set up + weights downloaded (see
# WorldCrafter/run_example.sh for that half) -- this script only drives
# WBench's own generate.py/main.py against the registered "worldcrafter" model.
#
# Any flag generate.py/main.py accept can be passed through, e.g.:
#   ./run_worldcrafter.sh --cases data/cases/case_1.json
#   ./run_worldcrafter.sh --cases data/cases/case_1.json --evaluate
#   ./run_worldcrafter.sh --limit 3               # generate first 3 cases
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
echo "WBench: generating with worldcrafter"
echo "=========================================="
uv run python generate.py --model worldcrafter "${GEN_ARGS[@]}"

if [ "$EVALUATE" = "1" ]; then
    echo
    echo "=========================================="
    echo "WBench: evaluating worldcrafter"
    echo "=========================================="
    uv run python main.py --model worldcrafter
fi

echo
echo "Done. Videos: work_dirs/worldcrafter/videos/"
[ "$EVALUATE" = "1" ] && echo "Results: work_dirs/worldcrafter/evaluation/ + work_dirs/worldcrafter/report.json"
