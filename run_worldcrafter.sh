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

# WBench has no pyproject.toml/uv.lock of its own, so `uv run` has nothing
# to sync and just falls through to whatever's active on PATH -- different
# every shell, missing a different dependency each time (cv2, scipy,
# requests, all seen in practice). ~/wbench-venv is the known-good env;
# used directly here so this script doesn't depend on which shell ran it.
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
echo "WBench: generating with worldcrafter"
echo "=========================================="
"$WBENCH_PYTHON" generate.py --model worldcrafter "${GEN_ARGS[@]}"

if [ "$EVALUATE" = "1" ]; then
    echo
    echo "=========================================="
    echo "WBench: evaluating worldcrafter"
    echo "=========================================="
    "$WBENCH_PYTHON" main.py --model worldcrafter
fi

echo
echo "Done. Videos: work_dirs/worldcrafter/videos/"
[ "$EVALUATE" = "1" ] && echo "Results: work_dirs/worldcrafter/evaluation/ + work_dirs/worldcrafter/report.json"
