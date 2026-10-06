"""
WBench Batch Video Generation for Matrix-Game-3.5.

generate.py calls MatrixGame35Model.generate_with_poses() once per case,
which reloads Matrix-Game-3.5's full checkpoint stack (DiT + Wan2.2-TI2V-5B
+ umt5-xxl + DA3NESTED) every time -- see matrix_game35_model.py's module
docstring. This script instead collects every (not yet generated) case up
front and makes ONE call to generate_batch_with_poses(), which loads the
model once per person group via infer.py's --batch-manifest support.

Only wired up for --model matrix_game35 (the only model with a batch
method so far). Output layout matches generate.py exactly
(work_dirs/<model>/videos/case_<id>_combined.mp4), so `python main.py
--model matrix_game35` evaluation works unchanged afterward.

Usage mirrors generate.py:
    python generate_matrix_game35_batch.py --cases data/cases/case_1.json
    python generate_matrix_game35_batch.py --limit 10
    python generate_matrix_game35_batch.py --resume

UNTESTED end-to-end, same caveat as generate_batch_with_poses() and
infer.py's --batch-manifest path it calls into.
"""
import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import get_model
from src.models.camera.poses import FPS, case_to_poses
from src.utils.case_loader import load_cases_raw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="WBench batch video generation (matrix_game35 only)")
    parser.add_argument("--data_dir", default="data", help="Path to data/ directory")
    parser.add_argument("--output_dir", default=None,
                        help="Output dir (default: work_dirs/matrix_game35/videos)")
    parser.add_argument("--cases", nargs="*", help="Specific case JSON files to process")
    parser.add_argument("--limit", type=int, default=None, help="Max cases to process")
    parser.add_argument("--resume", action="store_true", help="Skip cases with existing videos")
    args = parser.parse_args()

    model = get_model("matrix_game35")
    if not hasattr(model, "generate_batch_with_poses"):
        sys.exit("ERROR: matrix_game35 model has no generate_batch_with_poses "
                 "-- this script only supports models with a batch method.")
    logger.info(f"Using model: {model}")

    output_dir = args.output_dir or os.path.join("work_dirs", "matrix_game35", "videos")
    os.makedirs(output_dir, exist_ok=True)

    if args.cases:
        cases = []
        for f in args.cases:
            with open(f) as fp:
                cases.append(json.load(fp))
    else:
        cases = load_cases_raw(args.data_dir)

    if args.limit:
        cases = cases[:args.limit]

    # Resolve image/poses up front for every case still needing generation
    # (same per-case work generate_multi_turn does, just collected before
    # the single batch call instead of interleaved with it).
    pending = []  # list of (case, output_path, conv)
    results = {"success": 0, "failed": 0, "skipped": 0}

    for case in cases:
        case_id = case["id"]
        out_path = os.path.join(output_dir, f"case_{case_id}_combined.mp4")
        if args.resume and os.path.exists(out_path):
            logger.info(f"case_{case_id}: SKIP (exists)")
            results["skipped"] += 1
            continue

        image = model._resolve_image(case, args.data_dir)
        if not image or not os.path.exists(image):
            logger.error(f"case_{case_id}: FAIL -- initial_image not found: {image}")
            results["failed"] += 1
            continue

        conv = case_to_poses(case, duration=model.duration)
        pending.append((case, out_path, conv))

    if not pending:
        logger.info("Nothing to generate (all cases skipped or failed up front).")
        return

    logger.info(f"Batch-generating {len(pending)} case(s) -> {output_dir}")
    t0 = time.time()

    items = [
        {
            "image": model._resolve_image(case, args.data_dir),
            "poses": conv["poses"],
            "video_length": conv["video_length"],
            "perspective": conv["perspective"],
        }
        for case, _out_path, conv in pending
    ]

    try:
        batch_frames = model.generate_batch_with_poses(items)
    except Exception as e:  # noqa: BLE001 - surface as a run-level failure
        logger.error(f"generate_batch_with_poses failed: {e}")
        results["failed"] += len(pending)
        batch_frames = [None] * len(pending)

    for (case, out_path, conv), frames in zip(pending, batch_frames):
        case_id = case["id"]
        if not frames:
            logger.error(f"case_{case_id}: FAIL -- model returned no frames")
            results["failed"] += 1
            continue
        model._write_video(frames, out_path, fps=FPS)
        if model.dump:
            model._dump(out_path, case_id, "poses",
                       {"perspective": conv["perspective"],
                        "video_length": conv["video_length"],
                        "poses": conv["poses"]})
        logger.info(f"case_{case_id}: OK -> {out_path}")
        results["success"] += 1

    elapsed = time.time() - t0
    logger.info(
        f"\nDone in {elapsed:.1f}s — "
        f"success={results['success']}, failed={results['failed']}, skipped={results['skipped']}"
    )


if __name__ == "__main__":
    main()
