#!/usr/bin/env bash
# Fill in metrics that were never run for a model: MegaSAM-dependent metrics
# (spatial_consistency, gated_spatial_consistency, navigation_trajectory),
# subject_consistency (needs only SAM2 masks, already present but was never
# included in a --phase gpu run), and visual_plausibility (separate local
# reward-model pass, tools/run_visual_plausibility.py). Then regenerate
# report.json.
#
# Safe to re-run: precompute/gpu/run_visual_plausibility.py all skip cases
# that already have output, unless told otherwise.
#
#   ./fill_gaps.sh matrix_game35
#   ./fill_gaps.sh matrix_game35 worldcrafter
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
    echo "WBench: MegaSAM precompute for $MODEL"
    echo "=========================================="
    "$WBENCH_PYTHON" main.py --model "$MODEL" --phase precompute --skip_sam2 --skip_da3

    echo
    echo "=========================================="
    echo "WBench: subject_consistency / spatial_consistency / navigation_trajectory for $MODEL"
    echo "=========================================="
    "$WBENCH_PYTHON" main.py --model "$MODEL" --phase gpu \
        --metrics subject_consistency,spatial_consistency,navigation_trajectory

    echo
    echo "=========================================="
    echo "WBench: visual_plausibility for $MODEL"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=0 "$WBENCH_PYTHON" tools/run_visual_plausibility.py --model "$MODEL"

    echo
    echo "=========================================="
    echo "WBench: regenerating report for $MODEL"
    echo "=========================================="
    "$WBENCH_PYTHON" main.py --model "$MODEL" --phase report
    echo
done

echo "Done."
