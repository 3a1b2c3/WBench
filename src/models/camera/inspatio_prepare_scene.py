"""Build an InSpatio-World single-image scene from a WBench case's first frame and camera trajectory.

Runs inside InSpatio-World's own venv (spawned by ``inspatio_model.py`` with cwd and PYTHONPATH set to
the inspatio-world checkout), because it imports that repo's depth and scene code:

    python inspatio_prepare_scene.py --image first.jpg --target-c2w traj.npz --prompt "..." \
        --fps 24 --scene-dir <dir> --output-id <id> --translation-scale 1.0

Depth for the first frame comes from Depth Anything 3 (InSpatio's own DepthEstimator). The first camera
is the world origin, so every target pose is expressed relative to the first one and inverted to the
OpenCV world-to-camera matrices that scene.json's target_tcw.txt expects.
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.scene_depth_io import DEPTH_ENCODING
from pipeline.depth_estimation import DepthEstimator
from pipeline.scene_schema import padded_frame_count
from run_scene_inference import save_estimated_cameras

HEIGHT, WIDTH = 480, 832


def center_crop_resize(image: Image.Image) -> Image.Image:
    """Scale to cover 832x480, then center-crop (same rule as run_scene_inference's ffmpeg scaling)."""
    scale = max(WIDTH / image.width, HEIGHT / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    left = (resized.width - WIDTH) // 2
    top = (resized.height - HEIGHT) // 2
    return resized.crop((left, top, left + WIDTH, top + HEIGHT))


def target_world_to_camera(target_c2w: np.ndarray, translation_scale: float) -> np.ndarray:
    """Per-frame camera-to-world poses -> world-to-camera poses relative to the first frame."""
    first_inverse = np.linalg.inv(target_c2w[0])
    relative = np.einsum("ij,njk->nik", first_inverse, target_c2w)
    relative[:, :3, 3] *= translation_scale
    return np.linalg.inv(relative)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--target-c2w", required=True, type=Path, help=".npz with target_c2w (N,4,4), one pose per output frame")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--translation-scale", required=True, type=float)
    parser.add_argument("--depth-model-path", required=True, type=Path)
    args = parser.parse_args()

    target_c2w = np.load(args.target_c2w)["target_c2w"].astype(np.float64)
    valid_frames = len(target_c2w)
    frames = padded_frame_count(valid_frames)
    target_tcw = target_world_to_camera(target_c2w, args.translation_scale)
    target_tcw = np.concatenate([target_tcw, np.repeat(target_tcw[-1:], frames - valid_frames, axis=0)])

    scene = args.scene_dir
    (scene / "input").mkdir(parents=True, exist_ok=True)
    (scene / "depth").mkdir(parents=True, exist_ok=True)

    view = scene / "input" / "view_00.png"
    with Image.open(args.image) as source:
        center_crop_resize(source.convert("RGB")).save(view)

    with tempfile.TemporaryDirectory(prefix=".da3_", dir=scene) as temporary:
        raw = Path(temporary)
        DepthEstimator({"model_path": str(args.depth_model_path)}).run({"images": [str(view)]}, str(raw))
        with Image.open(raw / "depth" / "0000.png") as depth_image:
            depth = np.frombuffer(np.asarray(depth_image).tobytes(), dtype=np.float32).reshape(HEIGHT, WIDTH)
        np.save(scene / "depth" / "source_depths.npy", depth[None])
        save_estimated_cameras(raw, scene, 1)

    np.savetxt(scene / "input" / "target_tcw.txt", target_tcw.reshape(frames, 16), fmt="%.9g")
    (scene / "input" / "prompt.txt").write_text(args.prompt.strip() + "\n")
    (scene / "scene.json").write_text(json.dumps({
        "output_id": args.output_id,
        "kind": "image",
        "views": 1,
        "frames": frames,
        "valid_frames": valid_frames,
        "fps": args.fps,
        "resolution": [HEIGHT, WIDTH],
        "target_intrinsics": "estimated_source_view_00_fixed",
        "depth_encoding": DEPTH_ENCODING,
    }, indent=2) + "\n")
    (scene / ".prepared").write_text("ok\n")
    print(f"Prepared {scene}: {valid_frames} frames ({frames} padded), depth range "
          f"{float(depth.min()):.3f}-{float(depth.max()):.3f}", flush=True)


if __name__ == "__main__":
    main()
