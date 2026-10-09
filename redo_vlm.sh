#!/usr/bin/env bash
# Re-run ALL VLM-judged metrics (scene_adherence, subject_adherence,
# causal_fidelity, event_edit_adherence, subject_action_adherence,
# perspective_switch_adherence) for one or more already-generated models,
# then regenerate report.json.
#
# Only retries cases that don't already have a valid score per metric
# (run_phase_vlm's _has_valid_score check in main.py skips the rest), so
# it's safe to re-run without re-scoring everything that already succeeded.
# Needs the vlm_evaluator.py fixes already in place: exponential 429
# backoff with a real cap, enable_thinking=False for the reasoning model,
# and fast-fail + body logging on 4xx instead of blindly retrying a
# rejected payload.
#
#   ./redo_vlm.sh matrix_game35
#   ./redo_vlm.sh matrix_game35 worldcrafter
#   VLM_WORKERS=2 ./redo_vlm.sh matrix_game35   # lower concurrency if 429s pile up
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

# Same concurrency note as redo_physics.sh: 8 workers saturated the
# endpoint's rate limit (backoff climbing to hit 7/8, 8/8 before giving up
# on a case). 3 keeps a few cases in flight without tripping it every round.
VLM_WORKERS="${VLM_WORKERS:-3}"

for MODEL in "$@"; do
    echo "=========================================="
    echo "WBench: re-running all VLM metrics for $MODEL"
    echo "=========================================="
    "$WBENCH_PYTHON" main.py --model "$MODEL" --phase vlm --vlm_workers "$VLM_WORKERS"

    echo
    echo "=========================================="
    echo "WBench: regenerating report for $MODEL"
    echo "=========================================="
    "$WBENCH_PYTHON" main.py --model "$MODEL" --phase report
    echo
done

echo "Done."
