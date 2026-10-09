#!/usr/bin/env bash
# Re-run the causal_fidelity (physics) VLM metric for one or more already-
# generated models, then regenerate report.json.
#
# This only retries cases that don't already have a valid causal_fidelity
# score (run_phase_vlm's _has_valid_score check in main.py skips the rest),
# so it's safe to re-run after a transient VLM API failure (429/400) without
# re-scoring everything. Needs the vlm_evaluator.py rate-limit/reasoning-model
# fixes already in place.
#
#   ./redo_physics.sh matrix_game35
#   ./redo_physics.sh matrix_game35 worldcrafter
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

WBENCH_PYTHON="${WBENCH_PYTHON:-$HOME/wbench-venv/bin/python}"
if [ ! -x "$WBENCH_PYTHON" ]; then
    echo "WBENCH_PYTHON not found/executable: $WBENCH_PYTHON" >&2
    echo "Set WBENCH_PYTHON to a venv with WBench's dependencies installed." >&2
    exit 1
fi

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <model> [model...]" >&2
    exit 1
fi

for MODEL in "$@"; do
    echo "=========================================="
    echo "WBench: re-running causal_fidelity for $MODEL"
    echo "=========================================="
    "$WBENCH_PYTHON" main.py --model "$MODEL" --phase vlm --metrics causal_fidelity --vlm_workers 8

    echo
    echo "=========================================="
    echo "WBench: regenerating report for $MODEL"
    echo "=========================================="
    "$WBENCH_PYTHON" main.py --model "$MODEL" --phase report
    echo
done

echo "Done."
