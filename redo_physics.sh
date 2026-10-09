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
#   VLM_WORKERS=2 ./redo_physics.sh matrix_game35   # lower concurrency if 429s pile up
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

# Each case already makes its causal_fidelity VLM calls serially (nproc=1 in
# evaluate_case, see causal_fidelity.py), so vlm_workers == concurrent API
# requests. 8 was saturating the endpoint's rate limit (seen: backoff
# climbing to hit 7/8, 8/8 before giving up on a case) -- 3 keeps a few
# cases in flight without tripping it every round.
VLM_WORKERS="${VLM_WORKERS:-3}"

for MODEL in "$@"; do
    echo "=========================================="
    echo "WBench: re-running causal_fidelity for $MODEL"
    echo "=========================================="
    "$WBENCH_PYTHON" main.py --model "$MODEL" --phase vlm --metrics causal_fidelity --vlm_workers "$VLM_WORKERS"

    echo
    echo "=========================================="
    echo "WBench: regenerating report for $MODEL"
    echo "=========================================="
    "$WBENCH_PYTHON" main.py --model "$MODEL" --phase report
    echo
done

echo "Done."
